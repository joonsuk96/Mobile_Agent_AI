"""Ad skipping must not turn valid video evidence into a stale-snapshot failure."""
from contextlib import redirect_stdout
import io
import unittest
from unittest.mock import Mock, patch

from mobile_jarvis_app.agent import PhoneSession, build_tools, run_interactive, emit_final_answer
from mobile_jarvis_app.planning import CompletionCheck, SelectedItemEvidence
from test_agent import FakeDevice, FakeCatalog, call
from test_app_launch_routing import ToolRecordingModel
from test_youtube_workflow import youtube_screen
from mobile_jarvis_app.workflows import YOUTUBE_PACKAGE

TITLE="대체 한국은 왜 이렇게 부자냐는 태국 여사친 아버지ㅋㅋㅋㅋ"
COMMAND="유튜브 앱 켜서 조튜브 검색하고 최근 영상 중 조회수 90만 이상 아무거나 클릭해줘 광고 보이면 스킵해줘"


class EvidenceProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.device=FakeDevice()
        self.session=PhoneSession(self.device,FakeCatalog())
        self.session.original_command=COMMAND
        tracker=self.session.goal_tracker
        tracker.original_command=COMMAND
        tracker.set_plan("조튜브 조건 영상 열기",["조회수90만이상"],["조튜브 채널 확인","조회수90만이상 확인","광고 처리 후 본편 재생"],["검색","채널 동영상","후보 선택","광고 처리"])
        self.device.screen=youtube_screen(["조튜브","동영상"]);self.session.observe()
        self.device.screen=youtube_screen([]);self.session.observe()
        tracker.remember(2,self.session.screen,image_seen=True)
        tracker.register_candidate(2,TITLE,"조회수 94만회","image")
        self.device.screen=youtube_screen([TITLE,"광고","광고 건너뛰기"]);self.session.observe()
        self.device.tap.side_effect=lambda *args:setattr(self.device,"screen",youtube_screen([TITLE,"일시중지"]))
        delay=patch("mobile_jarvis_app.agent.time.sleep");delay.start();self.addCleanup(delay.stop)
        self.tools={t.name:t for t in build_tools(self.session)}
        with redirect_stdout(io.StringIO()):self.tools["skip_visible_ad"].invoke({"snapshot_id":3})
        self.assertEqual(self.session.snapshot_id,4)

    def checks(self):
        return [{"criterion_id":"C1","snapshot_id":4,"source":"ui","evidence_text":"조튜브"},
                {"criterion_id":"C2","snapshot_id":4,"source":"ui","evidence_text":"조회수 94만회"},
                {"criterion_id":"C3","snapshot_id":4,"source":"ui","evidence_text":"광고를 건너뛰고 본편 재생"}]

    def test_current_title_rebases_stale_post_click_snapshot(self):
        result=self.session.goal_tracker.confirm_selected(3,TITLE,"ui")
        self.assertTrue(result["matches"])
        self.assertEqual(result["snapshot_id"],4)
        self.assertEqual(result["requested_snapshot_id"],3)

    def test_wrong_ui_sources_resolve_to_retained_channel_candidate_and_ad_action(self):
        tracker=self.session.goal_tracker
        missing=tracker.apply_completion([CompletionCheck(**x) for x in self.checks()],None,
                    SelectedItemEvidence(snapshot_id=3,visible_title=TITLE),{},self.session.action_history)
        self.assertEqual(missing,[])
        self.assertEqual(tracker.criteria["C1"]["evidence"]["snapshot_id"],1)
        self.assertEqual(tracker.criteria["C2"]["evidence"]["source"],"video_candidate")
        self.assertEqual(tracker.criteria["C3"]["evidence"]["source"],"action")

    def test_full_finalization_finishes_once_with_title_without_retry_or_navigation(self):
        model=ToolRecordingModel(responses=[call("finish_task",{
            "summary":"조건에 맞는 영상을 열고 광고를 건너뛰었어.",
            "completion_checks":self.checks(),"selected_item":{"snapshot_id":3,"visible_title":TITLE}},1)])
        with patch.object(self.session,"prepare_app_route"),patch("builtins.input",side_effect=AssertionError("must not ask")),redirect_stdout(io.StringIO()) as output:
            run_interactive(self.session,COMMAND,model=model)
        self.assertTrue(self.session.completed)
        self.assertEqual(model.position,1)
        self.assertEqual(self.session.completion_validation_failures,0)
        self.assertEqual(self.session.autonomous_replans,0)
        self.device.navigate.assert_not_called()
        self.assertEqual(self.device.tap.call_count,1)
        self.assertIn(TITLE,output.getvalue())
        self.assertEqual(output.getvalue().count("자비스 :"),1)
        self.assertNotIn("확정하지 못했어",output.getvalue())

    def test_other_current_video_is_not_approved_using_old_title(self):
        self.session.goal_tracker.remember(5,youtube_screen(["완전히 다른 영상","일시중지"]))
        with self.assertRaises(ValueError):self.session.goal_tracker.confirm_selected(3,TITLE,"ui")

    def test_stale_confirmation_cannot_keep_old_success_after_different_title(self):
        tracker=self.session.goal_tracker
        tracker.confirm_selected(3,TITLE,"ui")
        self.assertTrue(tracker.selected_item_matches)
        tracker.remember(5,youtube_screen(["완전히 다른 영상","일시중지"]))
        with self.assertRaises(ValueError):tracker.confirm_selected(3,TITLE,"ui")
        self.assertFalse(tracker.selected_item_matches)

    def test_failed_progress_submission_retains_independent_title_fact(self):
        tracker=self.session.goal_tracker
        with self.assertRaises(ValueError):
            tracker.apply_completion([CompletionCheck(criterion_id="C1",snapshot_id=4,source="app",evidence_text="fake.app")],None,
                                     SelectedItemEvidence(snapshot_id=3,visible_title=TITLE),{},self.session.action_history)
        self.assertTrue(tracker.selected_item_matches)
        self.assertTrue(all(item["status"]=="pending" for item in tracker.criteria.values()))
        self.assertEqual(tracker.candidate_selection["view_count"],940000)

    def test_unverified_outcome_still_reports_only_the_confirmed_found_video(self):
        self.session.completion_validation_failures=2
        self.session.last_completion_problem="현재 재생 상태 근거 없음"
        with redirect_stdout(io.StringIO()) as output:emit_final_answer(self.session)
        self.assertIn(TITLE,output.getvalue())
        self.assertIn("현재 상태는 확인하지 못했어",output.getvalue())
        self.assertFalse(self.session.completed)
