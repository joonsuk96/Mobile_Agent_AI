"""Reject unfinished input/navigation as a terminal failure and keep global budgets."""
from contextlib import redirect_stdout
import io
import unittest
from unittest.mock import Mock, patch

from mobile_jarvis_app.agent import PhoneSession, build_tools, run_interactive
from test_agent import FakeCatalog, FakeDevice, ScriptedModel, call, screen
from test_text_input import input_screen


class PrematureExitTests(unittest.TestCase):
    def setUp(self):
        self.device = FakeDevice()
        self.device.screen = input_screen()
        self.session = PhoneSession(self.device, FakeCatalog(), require_plan=False)
        self.session.actions = 7
        self.session.observe()
        self.tools = {t.name: t for t in build_tools(self.session)}
        delay = patch("mobile_jarvis_app.agent.time.sleep")
        delay.start()
        self.addCleanup(delay.stop)

    def test_keyboard_and_empty_destination_do_not_end_task_after_seven_actions(self):
        def enter(element, text, clear=True):
            self.device.screen = input_screen(text)
            return {"verified": True}
        self.device.set_text = Mock(side_effect=enter)
        self.device.tap.side_effect = lambda *args: setattr(
            self.device, "screen", screen("org.example.app", "좌석 선택 화면"))
        model = ScriptedModel(responses=[
            call("finish_incomplete", {"reason": "도착지 검색어가 비어 있고 키보드가 화면을 가려 다음 입력이 필요합니다"}, 1),
            call("type_text", {"snapshot_id": 1, "element_id": "0", "text": "세종시"}, 2),
            call("click_element", {"snapshot_id": 2, "element_id": "0"}, 3),
            call("finish_task", {"summary": "좌석 선택 화면까지 열었어", "evidence": "좌석 선택 표시 확인"}, 4),
        ])
        with patch("builtins.input", side_effect=AssertionError("must not ask")), redirect_stdout(io.StringIO()) as output:
            run_interactive(self.session, "2026년 11월 6일 수원터미널에서 세종시 11시40분 근처 버스 좌석 화면까지", model=model)
        self.device.set_text.assert_called_once()
        self.assertEqual(self.device.set_text.call_args.args[1], "세종시")
        self.device.tap.assert_called_once()
        self.assertTrue(self.session.completed)
        self.assertEqual(self.session.actions, 9)
        self.assertEqual(self.session.autonomous_replans, 0)
        self.assertIn("종료 보류", output.getvalue())
        self.assertIn("좌석 선택 화면까지 열었어", output.getvalue())

    def test_claimed_missing_information_or_budget_cannot_end_with_input_pending(self):
        for blocker in ["route_failed", "missing_information", "budget", "authentication", "connection"]:
            with self.subTest(blocker=blocker), redirect_stdout(io.StringIO()):
                result = self.tools["finish_incomplete"].invoke({"reason": "검색어가 비어 있어 입력 필요", "blocker": blocker})
                self.assertEqual(result["status"], "not_verified")
                self.assertIsNone(self.session.terminal_reason)

    def test_repeated_exit_requests_cannot_reset_global_model_budget(self):
        self.session.max_model_calls = 4
        model = ScriptedModel(responses=[
            call("finish_incomplete", {"reason": "아직 날짜 지정이 필요해"}, i)
            for i in range(1, 10)
        ])
        with patch("builtins.input", side_effect=AssertionError("must not ask")), redirect_stdout(io.StringIO()):
            run_interactive(self.session, "도착지 날짜 입력 계속", model=model)
        self.assertFalse(self.session.completed)
        self.assertEqual(model.position, 4)
        self.assertEqual(self.session.model_calls_total, 4)
        self.assertEqual(self.session.model_window_start, 0)
        self.assertEqual(self.session.actions, 7)

    def test_repeated_exit_requests_cannot_reset_global_tool_budget(self):
        self.session.max_tool_calls = 2
        model = ScriptedModel(responses=[
            call("finish_incomplete", {"reason": "다음 화면 필요"}, i)
            for i in range(1, 10)
        ])
        with patch("builtins.input", side_effect=AssertionError("must not ask")), redirect_stdout(io.StringIO()):
            run_interactive(self.session, "화면 조정 계속", model=model)
        self.assertEqual(self.session.tool_calls_total, 2)
        self.assertEqual(self.session.tool_window_start, 0)
        self.assertEqual(model.position, 2)

    def test_completion_requires_explicit_goal_completed_declaration(self):
        required = self.tools["finish_task"].args_schema.model_json_schema()["required"]
        self.assertIn("goal_completed", required)

    def test_observed_connection_error_can_be_reported(self):
        self.session.last_observation_error = "DeviceError"
        self.session.screen = None
        with redirect_stdout(io.StringIO()):
            result = self.tools["finish_incomplete"].invoke({"reason": "ADB 연결 끊김", "blocker": "connection"})
        self.assertEqual(result["status"], "incomplete")
