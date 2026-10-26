"""Channel-first discovery and numeric views without a false ordinal requirement."""
from contextlib import redirect_stdout
import io
import unittest
from unittest.mock import Mock, patch

from mobile_jarvis_app.agent import PhoneSession, build_tools, run_interactive
from mobile_jarvis_app.media import displayed_views, view_requirement, meets_views
from mobile_jarvis_app.planning import GoalTracker, SelectedItemEvidence, RankedListEvidence, RankedCandidate
from test_agent import FakeDevice, FakeCatalog, call
from test_app_launch_routing import ToolRecordingModel
from test_youtube_workflow import youtube_screen
from mobile_jarvis_app.workflows import YOUTUBE_PACKAGE

LOW = "부티크 떡상템 정리해 드립니다 [팡이요] - 10분 8초 - 채널로 이동 - 팡이요 - 조회수 16만회 - 5시간 전 - 동영상 재생"
HIGH = "드디어 아델 스펙업 [팡이요] - 22분 40초 - 채널로 이동 - 팡이요 - 조회수 50만회 - 6일 전 - 동영상 재생"
TITLE = "드디어 아델 스펙업 [팡이요]"
COMMAND = "유튜브 앱 틀어서 팡이요 검색해주고 올라온 최신 영상들중에조회수 50만이상인거 아무거나 선택해서 재생해줘 광고 보이면 스킵해줘"


class ViewNumberTests(unittest.TestCase):
    def test_displayed_units_and_inclusive_threshold(self):
        for text, value in [("조회수 16만회",160000),("조회수 50만회",500000),("조회수 1.2천회",1200),("500K views",500000),("조회수 1.2억회",120000000)]:
            with self.subTest(text=text):self.assertEqual(displayed_views(text)["value"],value)
        requirement=view_requirement(COMMAND)
        self.assertFalse(meets_views(160000,requirement))
        self.assertTrue(meets_views(500000,requirement))
        self.assertTrue(meets_views(600000,requirement))
        self.assertEqual(view_requirement("조회수50만초과") ["operator"], "초과")
        self.assertFalse(meets_views(500000,view_requirement("조회수50만초과")))

    def test_numeric_equivalent_evidence_is_allowed_but_wrong_number_is_not(self):
        tracker=GoalTracker(COMMAND)
        tracker.set_plan("조건 영상 재생",[],["조회수 확인"],["영상 확인"])
        tracker.remember(1,youtube_screen([HIGH]))
        tracker.record("C1","satisfied","조회수 500,000회",1,"ui",{})
        with self.assertRaises(ValueError):tracker.evidence(1,"ui","조회수 600,000회")

    def test_non_ordinal_selection_uses_view_candidate_not_ranked_list(self):
        tracker=GoalTracker(COMMAND)
        tracker.set_plan("조건 영상 재생",[],["조회수 확인"],["선택"])
        tracker.remember(1,youtube_screen([HIGH]))
        tracker.remember(2,youtube_screen([TITLE,"일시정지"]))
        result=tracker.confirm_selected(2,TITLE,"ui")
        self.assertTrue(result["matches"])
        self.assertEqual(result["view_count"],500000)
        self.assertIsNone(tracker.ranked_selection)

    def test_under_threshold_candidate_cannot_be_confirmed(self):
        tracker=GoalTracker(COMMAND)
        tracker.set_plan("조건 영상 재생",[],["조회수 확인"],["선택"])
        tracker.remember(1,youtube_screen([LOW]))
        tracker.remember(2,youtube_screen(["부티크 떡상템 정리해 드립니다 [팡이요]","일시정지"]))
        with self.assertRaises(ValueError):tracker.confirm_selected(2,"부티크 떡상템 정리해 드립니다 [팡이요]","ui")


class ChannelFirstVideoTests(unittest.TestCase):
    def setUp(self):
        self.device=FakeDevice()
        self.device.current_app=Mock(return_value={"package":YOUTUBE_PACKAGE,"activity":YOUTUBE_PACKAGE+"/.Main"})
        self.session=PhoneSession(self.device,FakeCatalog())
        self.session.original_command=COMMAND
        self.session.goal_tracker.original_command=COMMAND
        self.session.goal_tracker.set_plan("50만 이상 영상 재생",["팡이요","조회수50만이상"],["조회수 조건 확인","본편 재생"],["채널 보기","동영상 탭","조건 후보 선택","재생 확인"])
        self.device.screen=youtube_screen(["팡이요","채널 보기"])
        self.session.observe()
        self.session.youtube_channel_pending=True
        delay=patch("mobile_jarvis_app.agent.time.sleep");delay.start();self.addCleanup(delay.stop)
        self.tools={t.name:t for t in build_tools(self.session)}

    def test_channel_then_videos_then_valid_candidate_finishes_without_rank_error(self):
        def tap(x,y):
            labels=[item["label"] for item in self.device.screen["elements"]]
            if "채널 보기" in labels:self.device.screen=youtube_screen(["팡이요","홈","동영상"])
            elif "홈" in labels:self.device.screen=youtube_screen(["동영상",LOW,HIGH],True)
            else:self.device.screen=youtube_screen([TITLE,"일시정지"])
        self.device.tap.side_effect=tap
        model=ToolRecordingModel(responses=[
            call("open_youtube_channel",{"snapshot_id":1},1),
            call("open_youtube_videos_tab",{"snapshot_id":2},2),
            call("click_element",{"snapshot_id":3,"element_id":"1"},3),
            call("click_element",{"snapshot_id":3,"element_id":"2"},4),
            call("finish_task",{"summary":"조회수 50만인 팡이요 영상을 재생했어.",
                "completion_checks":[{"criterion_id":"C1","source":"ui","snapshot_id":3,"evidence_text":"조회수 500000회"},{"criterion_id":"C2","source":"ui","snapshot_id":4,"evidence_text":"일시정지"}],
                "selected_item":{"snapshot_id":4,"visible_title":TITLE}},5),
        ])
        with patch("builtins.input",side_effect=AssertionError("must not ask")),redirect_stdout(io.StringIO()) as output:
            run_interactive(self.session,COMMAND,model=model)
        self.assertEqual(set(model.available[0]),{"open_youtube_channel", "set_youtube_search_strategy"})
        self.assertEqual(model.available[1],["open_youtube_videos_tab"])
        self.assertNotIn("record_ranked_candidates",model.available[2])
        self.assertTrue(self.session.completed)
        self.assertEqual(self.device.tap.call_count,3) # Channel, tab, valid video; 16만 was rejected.
        self.device.scroll.assert_not_called();self.device.navigate.assert_not_called()
        self.assertIn("자비스 :",output.getvalue())
        self.assertIn(TITLE,output.getvalue())
        self.assertIn("조회수 50만인 팡이요 영상을 재생했어.",output.getvalue())
        self.assertEqual(self.session.completion_validation_failures,0)

    def test_general_topic_scope_does_not_force_arbitrary_channel(self):
        self.session.youtube_channel_pending=False
        self.session.youtube_search_scope="topic"
        self.session.action_history=[{"action":"텍스트 입력","strategy":"type","status":"executed"}]
        with redirect_stdout(io.StringIO()):
            self.session.perform("클릭: 검색",lambda: None)
        self.assertFalse(self.session.youtube_channel_pending)
        self.assertFalse(self.session.youtube_channel_due())

    def test_search_thumbnail_is_blocked_until_channel_opened(self):
        self.device.screen=youtube_screen([LOW,HIGH])
        self.session.observe()
        with self.assertRaises(ValueError):
            self.tools["click_element"].invoke({"snapshot_id":2,"element_id":"1"})
        self.device.tap.assert_not_called()

    def test_non_ordinal_payload_does_not_require_or_process_spurious_ranked_list(self):
        tracker=self.session.goal_tracker
        self.device.screen=youtube_screen([HIGH]);self.session.observe()
        self.device.screen=youtube_screen([TITLE,"일시정지"]);self.session.observe()
        fake_rank=RankedListEvidence(snapshot_id=2,order="newest",sort_active=True,sort_evidence="없는 정렬 문구",candidates=[RankedCandidate(position=1,title=TITLE,publication_text="6일 전")])
        tracker.apply_completion([],fake_rank,SelectedItemEvidence(snapshot_id=3,visible_title=TITLE),{},[])
        self.assertTrue(tracker.selected_item_matches)
        self.assertIsNone(tracker.ranked_selection)
        self.assertEqual(tracker.candidate_selection["view_count"],500000)
