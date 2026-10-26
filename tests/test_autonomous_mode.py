"""Default execution must not request human guidance, even after failure."""
from contextlib import redirect_stdout
import io
import unittest
from unittest.mock import patch

from mobile_jarvis_app.agent import PhoneSession, TaskStopped, build_tools, run_interactive
from test_agent import FakeCatalog, FakeDevice, ScriptedModel, call


class AutonomousModeTests(unittest.TestCase):
    def setUp(self):
        self.device = FakeDevice()
        self.session = PhoneSession(self.device, FakeCatalog(), require_plan=False)
        delay = patch("mobile_jarvis_app.agent.time.sleep")
        delay.start()
        self.addCleanup(delay.stop)

    def test_question_tools_are_replaced_with_autonomous_tools(self):
        tools = {t.name for t in build_tools(self.session)}
        self.assertNotIn("ask_user", tools)
        self.assertNotIn("confirm_app_choice", tools)
        self.assertIn("replan", tools)
        self.assertIn("finish_incomplete", tools)

    def test_replan_then_success_does_not_ask_or_reset_action_budget(self):
        self.session.actions = 4
        model = ScriptedModel(responses=[
            call("replan", {"reason": "입력칸을 다시 확인"}, 1),
            call("finish_task", {"summary": "확인했어"}, 2),
        ])
        with patch("builtins.input", side_effect=AssertionError("must not ask")), redirect_stdout(io.StringIO()):
            run_interactive(self.session, "현재 화면 확인", model=model)
        self.assertTrue(self.session.completed)
        self.assertEqual(self.session.actions, 4)
        self.assertEqual(self.session.action_window_start, 0)
        self.assertEqual(self.session.autonomous_replans, 1)

    def test_incomplete_report_ends_without_waiting_for_input(self):
        from test_text_input import input_screen
        self.device.screen = input_screen(password=True)
        self.session.observe()
        model = ScriptedModel(responses=[
            call("finish_incomplete", {"reason": "로그인이 필요해서 완료하지 못했어", "blocker": "authentication", "blocking_evidence": "보호된 비밀번호 입력란 표시"}, 1),
        ])
        with patch("builtins.input", side_effect=AssertionError("must not ask")), redirect_stdout(io.StringIO()) as output:
            run_interactive(self.session, "현재 화면 확인", model=model)
        self.assertIn("로그인이 필요", output.getvalue())
        self.assertFalse(self.session.completed)

    def test_action_limit_does_not_renew_itself_or_ask(self):
        self.session.max_actions = 1
        self.session.actions = 1
        self.session.observe()
        model = ScriptedModel(responses=[call("navigate", {"direction": "back"}, 1)])
        with patch("builtins.input", side_effect=AssertionError("must not ask")), redirect_stdout(io.StringIO()):
            result = run_interactive(self.session, "현재 화면에서 뒤로 이동", model=model)
        self.assertEqual(result["status"], "incomplete")
        self.device.navigate.assert_not_called()
        self.assertEqual(self.session.action_window_start, 0)

    def test_repeated_model_errors_stop_after_bounded_recovery_without_input(self):
        from unittest.mock import Mock
        broken = Mock()
        broken.invoke.side_effect = RuntimeError("private-error-sentinel")
        broken.get_state.side_effect = RuntimeError("private-state-sentinel")
        with patch("mobile_jarvis_app.agent.build_agent", return_value=broken), \
             patch("builtins.input", side_effect=AssertionError("must not ask")), redirect_stdout(io.StringIO()) as output:
            result = run_interactive(self.session, "현재 화면 확인")
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(broken.invoke.call_count, self.session.max_recoveries + 1)
        self.assertNotIn("private-", output.getvalue())

    def test_replan_limit_is_finite(self):
        with redirect_stdout(io.StringIO()):
            for _ in range(self.session.max_replans):
                self.session.autonomous_replan("다른 방법")
            with self.assertRaises(TaskStopped):
                self.session.autonomous_replan("다시")

    def test_interactive_opt_in_keeps_question_tools(self):
        session = PhoneSession(self.device, FakeCatalog(), interactive=True, require_plan=False)
        tools = {t.name for t in build_tools(session)}
        self.assertIn("ask_user", tools)
        self.assertIn("confirm_app_choice", tools)
