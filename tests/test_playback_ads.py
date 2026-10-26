"""Playback verification survives observation; requested ad skipping has priority."""
from contextlib import redirect_stdout
import io
import unittest
from unittest.mock import patch

from mobile_jarvis_app.agent import PhoneSession, build_tools, run_interactive
from mobile_jarvis_app.planning import RankedCandidate
from test_agent import FakeCatalog, FakeDevice, call
from test_app_launch_routing import ToolRecordingModel
from test_goal_planning import view


class PlaybackAdTests(unittest.TestCase):
    def setUp(self):
        self.device = FakeDevice()
        self.session = PhoneSession(self.device, FakeCatalog())
        self.goal = "두 번째 최근 항목 재생해줘 광고 보이면 스킵해줘"
        self.session.original_command = self.goal
        tracker = self.session.goal_tracker
        tracker.original_command = self.goal
        tracker.set_plan("두 번째 최근 항목 재생과 현재 광고 처리", ["두 번째 최신"], ["본편 재생 확인"], ["정렬 확인", "선택", "광고 처리", "재생 확인"])
        self.device.screen = view(["최신순", "첫 항목", "1일 전", "둘째 항목", "2일 전"])
        self.session.observe()
        tracker.record_ranked(1, "최신순", [RankedCandidate(position=1, title="첫 항목", publication_text="1일 전"), RankedCandidate(position=2, title="둘째 항목", publication_text="2일 전")], "ui")
        self.device.screen = view(["둘째 항목", "일시정지", "00:01"])
        self.session.observe()
        tracker.confirm_selected(2, "둘째 항목", "ui")
        delay = patch("mobile_jarvis_app.agent.time.sleep")
        delay.start()
        self.addCleanup(delay.stop)

    def test_repeated_progress_observations_keep_verified_selection(self):
        self.device.screen = view(["둘째 항목", "일시정지", "00:02", "새 자막"])
        self.session.observe()
        tracker = self.session.goal_tracker
        self.assertTrue(tracker.selected_item_matches)
        tracker.record("C1", "satisfied", "일시정지", 3, "ui", {})
        self.assertEqual(tracker.missing(), [])
        tools = {tool.name: tool for tool in build_tools(self.session)}
        with redirect_stdout(io.StringIO()):
            result = tools["finish_task"].invoke({"completion_checks": [], "summary": "둘째 항목 재생 중", "goal_completed": True})
        self.assertEqual(result["status"], "completed")
        self.device.navigate.assert_not_called()

    def test_ad_skip_is_first_tool_and_preserves_correct_video(self):
        self.device.screen = view(["둘째 항목", "광고", "광고 건너뛰기"])
        self.session.observe()
        self.device.tap.side_effect = lambda *args: setattr(self.device, "screen", view(["둘째 항목", "일시정지", "00:03"]))
        model = ToolRecordingModel(responses=[
            call("skip_visible_ad", {"snapshot_id": 3}, 1),
            call("record_task_progress", {"criterion_id": "C1", "status": "satisfied", "evidence_text": "일시정지", "snapshot_id": 4}, 2),
            call("finish_task", {"summary": "광고를 건너뛰고 둘째 항목을 재생했어"}, 3),
        ])
        with patch("builtins.input", side_effect=AssertionError("must not ask")), redirect_stdout(io.StringIO()):
            run_interactive(self.session, self.goal, model=model)
        self.assertEqual(model.available[0], ["skip_visible_ad"])
        self.device.tap.assert_called_once()
        self.device.navigate.assert_not_called()
        self.assertTrue(self.session.goal_tracker.selected_item_matches)
        self.assertTrue(self.session.completed)
        self.assertEqual(model.position, 3)

    def test_real_navigation_invalidates_selected_video(self):
        tools = {tool.name: tool for tool in build_tools(self.session)}
        self.device.navigate.side_effect = lambda direction: setattr(self.device, "screen", view(["채널 목록"]))
        with redirect_stdout(io.StringIO()):
            tools["navigate"].invoke({"direction": "back"})
        self.assertFalse(self.session.goal_tracker.selected_item_matches)

    def test_ad_skip_labels_and_unrelated_skip_are_distinguished(self):
        for label in ["광고 건너뛰기", "건너뛰기", "Skip ad", "Skip Ads"]:
            with self.subTest(label=label):
                self.device.screen = view(["광고", label])
                self.session.observe()
                self.assertIsNotNone(self.session.visible_ad_skip())
        self.device.screen = view(["초기 설정", "건너뛰기"])
        self.session.observe()
        self.assertIsNone(self.session.visible_ad_skip())
        self.session.original_command = "항목 재생해줘"
        self.device.screen = view(["광고", "광고 건너뛰기"])
        self.session.observe()
        self.assertIsNone(self.session.visible_ad_skip())

    def test_visible_ad_blocks_completion_even_when_criteria_are_satisfied(self):
        self.device.screen = view(["둘째 항목", "광고", "광고 건너뛰기"])
        self.session.observe()
        self.session.goal_tracker.record("C1", "satisfied", "둘째 항목", 3, "ui", {})
        tools = {tool.name: tool for tool in build_tools(self.session)}
        with redirect_stdout(io.StringIO()):
            result = tools["finish_task"].invoke({"completion_checks": [], "summary": "재생했어", "goal_completed": True})
        self.assertEqual(result["status"], "not_verified")
        self.assertIn("skip_visible_ad", result["remaining_work"])
        self.assertFalse(self.session.completed)

    def test_missing_criteria_are_printed_and_do_not_order_unnecessary_back_navigation(self):
        tools = {tool.name: tool for tool in build_tools(self.session)}
        with redirect_stdout(io.StringIO()) as output:
            result = tools["finish_task"].invoke({"completion_checks": [], "summary": "재생했어", "goal_completed": True})
        self.assertEqual(result["status"], "not_verified")
        self.assertIn("C1: 본편 재생 확인", output.getvalue())
        self.assertIn("뒤로 나가지 마", result["note"])
