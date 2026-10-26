"""Finish already-playing content using retained evidence, without reopening pages."""
from contextlib import redirect_stdout
import io
import unittest
from unittest.mock import patch

from mobile_jarvis_app.agent import PhoneSession, build_tools, run_interactive
from mobile_jarvis_app.planning import CompletionCheck, RankedListEvidence, SelectedItemEvidence
from test_agent import FakeCatalog, FakeDevice, call
from test_app_launch_routing import ToolRecordingModel
from test_goal_planning import view


class AtomicFinalizationTests(unittest.TestCase):
    def setUp(self):
        self.device = FakeDevice()
        self.session = PhoneSession(self.device, FakeCatalog())
        self.goal = "3번째 최신 영상 재생하고 광고 보이면 스킵해줘. 재생되면 바로 종료"
        self.session.original_command = self.goal
        tracker = self.session.goal_tracker
        tracker.original_command = self.goal
        tracker.set_plan("세 번째 최신 영상 재생", ["최신순 세 번째", "광고 스킵"],
                         ["대상 앱 전면", "검색 결과 확인", "시간순 세 번째 확인", "본편 재생", "현재 광고 처리"],
                         ["앱 실행", "검색", "정렬 확인", "영상 선택", "광고 처리"])
        for labels in [["앱 첫 화면"], ["유봉훈 검색 결과"],
                       ["최신순", "첫 영상", "1일 전", "둘째 영상", "2일 전", "셋째 영상", "3일 전"],
                       ["셋째 영상", "광고", "광고 건너뛰기"]]:
            self.device.screen = view(labels)
            self.session.observe()
        self.device.tap.side_effect = lambda *args: setattr(self.device, "screen", view(["셋째 영상", "일시정지", "00:02"]))
        delay = patch("mobile_jarvis_app.agent.time.sleep")
        delay.start()
        self.addCleanup(delay.stop)
        self.tools = {tool.name: tool for tool in build_tools(self.session)}
        with redirect_stdout(io.StringIO()):
            self.tools["skip_visible_ad"].invoke({"snapshot_id": 4})
        self.assertEqual(self.session.snapshot_id, 5)

    def batch(self):
        return {"summary": "세 번째 최신 영상을 재생했고 광고를 건너뛰었어", "goal_completed": True,
                "completion_checks": [
                    {"criterion_id": "C1", "source": "app", "snapshot_id": 1, "evidence_text": "org.example.app"},
                    {"criterion_id": "C2", "source": "ui", "snapshot_id": 2, "evidence_text": "유봉훈 검색 결과"},
                    {"criterion_id": "C3", "source": "ui", "snapshot_id": 3, "evidence_text": "최신순"},
                    {"criterion_id": "C4", "source": "ui", "snapshot_id": 5, "evidence_text": "일시정지"},
                    {"criterion_id": "C5", "source": "action", "evidence_text": "광고 건너뛰기"},
                ],
                "ranked_list": {"snapshot_id": 3, "order": "newest", "sort_active": True, "sort_evidence": "최신순",
                                "candidates": [{"position": 1, "title": "첫 영상", "publication_text": "1일 전"},
                                               {"position": 2, "title": "둘째 영상", "publication_text": "2일 전"},
                                               {"position": 3, "title": "셋째 영상", "publication_text": "3일 전"}]},
                "selected_item": {"snapshot_id": 5, "visible_title": "셋째 영상"}}

    def test_all_five_pending_criteria_finish_in_one_submission_from_saved_views(self):
        model = ToolRecordingModel(responses=[call("finish_task", self.batch(), 1)])
        with patch("builtins.input", side_effect=AssertionError("must not ask")), redirect_stdout(io.StringIO()):
            run_interactive(self.session, self.goal, model=model)
        self.assertTrue(self.session.completed)
        self.assertEqual(model.position, 1)
        self.device.navigate.assert_not_called()
        self.assertEqual(self.device.tap.call_count, 1)  # Only the already-executed ad skip.

    def test_rejected_finish_enters_verification_phase_and_batch_then_finishes(self):
        model = ToolRecordingModel(responses=[
            call("finish_task", {"summary": "재생했어", "completion_checks": []}, 1),
            call("finish_task", self.batch(), 2),
        ])
        with patch("builtins.input", side_effect=AssertionError("must not ask")), redirect_stdout(io.StringIO()):
            run_interactive(self.session, self.goal, model=model)
        self.assertNotIn("navigate", model.available[1])
        self.assertNotIn("click_element", model.available[1])
        self.assertEqual(model.available[1], ["finish_task"])
        self.assertTrue(self.session.completed)
        self.assertFalse(self.session.finalization_pending)
        self.device.navigate.assert_not_called()

    def test_direct_back_action_cannot_bypass_finalization_gate(self):
        with redirect_stdout(io.StringIO()):
            self.tools["finish_task"].invoke({"summary": "재생했어", "goal_completed": True, "completion_checks": []})
        self.assertTrue(self.session.finalization_pending)
        with self.assertRaises(ValueError):
            self.tools["navigate"].invoke({"direction": "back"})
        self.device.navigate.assert_not_called()

    def test_batch_rolls_back_if_one_piece_of_evidence_is_fabricated(self):
        batch = self.batch()
        batch["completion_checks"][1]["evidence_text"] = "관찰에 없는 다른 채널"
        with redirect_stdout(io.StringIO()):
            result = self.tools["finish_task"].invoke(batch)
        self.assertEqual(result["status"], "not_verified")
        self.assertTrue(all(item["status"] == "pending" for item in self.session.goal_tracker.criteria.values()))
        self.assertEqual(self.session.goal_tracker.ranked_selection["title"], "셋째 영상")
        self.assertTrue(self.session.goal_tracker.selected_item_matches)
        self.assertFalse(self.session.completed)

    def test_action_evidence_requires_executed_record_and_app_requires_matching_package(self):
        tracker = self.session.goal_tracker
        with self.assertRaises(ValueError):
            tracker.record("C5", "satisfied", "광고 건너뛰기", 0, "action", {}, [])
        with self.assertRaises(ValueError):
            tracker.record("C1", "satisfied", "fake.app", 1, "app", {})

    def test_explicit_repair_can_resume_actions_only_for_unmet_condition(self):
        self.session.finalization_pending = True
        result = self.tools["resume_task_actions"].invoke({"criterion_id": "C3", "reason": "정렬이 실제 요청과 다름", "why_observations_insufficient": "저장된 화면에 활성 정렬이 없어 다시 확인해야 함"})
        self.assertEqual(result["status"], "actions_resumed")
        self.assertFalse(self.session.finalization_pending)
        self.session.goal_tracker.record("C1", "satisfied", "org.example.app", 1, "app", {})
        self.session.finalization_pending = True
        with self.assertRaises(ValueError):
            self.tools["resume_task_actions"].invoke({"criterion_id": "C1", "reason": "돌아가기", "why_observations_insufficient": "다시 보기"})

    def test_same_missing_rank_error_stops_after_one_correction_without_replanning(self):
        bad = {"summary": "영상 재생 완료", "completion_checks": [],
               "selected_item": {"snapshot_id": 5, "visible_title": "셋째 영상"}}
        model = ToolRecordingModel(responses=[
            call("finish_task", bad, 1),
            call("finish_task", bad, 2),
            call("navigate", {"direction": "back"}, 3),
        ])
        with patch("builtins.input", side_effect=AssertionError("must not ask")), redirect_stdout(io.StringIO()) as output:
            run_interactive(self.session, self.goal, model=model)
        self.assertEqual(model.position, 3)  # one tool-free final answer; no phone action
        self.assertEqual(model.available[1], ["finish_task"])
        self.assertFalse(self.session.completed)
        self.assertIsNotNone(self.session.terminal_reason)
        self.assertEqual(self.session.autonomous_replans, 0)
        self.device.navigate.assert_not_called()
        self.assertEqual(self.device.tap.call_count, 1)
        self.assertNotIn("[재판단", output.getvalue())
        self.assertIn("검증 종료", output.getvalue())

    def test_missing_rank_can_be_corrected_once_using_retained_list(self):
        bad = {"summary": "영상 재생 완료", "completion_checks": [],
               "selected_item": {"snapshot_id": 5, "visible_title": "셋째 영상"}}
        model = ToolRecordingModel(responses=[call("finish_task", bad, 1), call("finish_task", self.batch(), 2)])
        with patch("builtins.input", side_effect=AssertionError("must not ask")), redirect_stdout(io.StringIO()):
            run_interactive(self.session, self.goal, model=model)
        self.assertTrue(self.session.completed)
        self.assertEqual(model.position, 2)
        self.assertEqual(self.session.autonomous_replans, 0)
        self.device.navigate.assert_not_called()

    def test_repeated_unregistered_criteria_stop_without_metadata_exception(self):
        bad = {"summary": "재생 완료", "completion_checks": []}
        model = ToolRecordingModel(responses=[call("finish_task", bad, i) for i in range(1, 4)])
        with patch("builtins.input", side_effect=AssertionError("must not ask")), redirect_stdout(io.StringIO()):
            run_interactive(self.session, self.goal, model=model)
        self.assertEqual(model.position, 3)  # includes terminal wording call
        self.assertFalse(self.session.completed)
        self.assertEqual(self.session.autonomous_replans, 0)
        self.assertIsNotNone(self.session.terminal_reason)

    def test_invalid_finish_schema_is_not_sent_to_general_action_recovery(self):
        first = call("finish_task", {"summary": "재생 완료"}, 1)
        second = call("finish_task", {"summary": "재생 완료"}, 2)
        first.tool_calls[0]["args"].pop("completion_checks")
        second.tool_calls[0]["args"].pop("completion_checks")
        model = ToolRecordingModel(responses=[first, second])
        with patch("builtins.input", side_effect=AssertionError("must not ask")), redirect_stdout(io.StringIO()):
            run_interactive(self.session, self.goal, model=model)
        self.assertEqual(model.position, 2)
        self.assertEqual(self.session.autonomous_replans, 0)
        self.assertIsNotNone(self.session.terminal_reason)

    def test_finish_schema_requires_explicit_completion_checks(self):
        self.assertIn("completion_checks", self.tools["finish_task"].args_schema.model_json_schema()["required"])
