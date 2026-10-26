"""Autonomous actions pause only for required user facts and resume in place."""
from contextlib import redirect_stdout
import io
import unittest
from unittest.mock import patch

from mobile_jarvis_app.agent import PhoneSession, TaskCancelled, build_tools, run_interactive
from mobile_jarvis_app.device import DeviceError
from test_agent import FakeCatalog, FakeDevice, ScriptedModel, call, screen


class MissingInformationTests(unittest.TestCase):
    def setUp(self):
        self.device = FakeDevice()
        self.session = PhoneSession(self.device, FakeCatalog(), require_plan=False)
        self.tools = {t.name: t for t in build_tools(self.session)}
        delay = patch("mobile_jarvis_app.agent.time.sleep")
        delay.start()
        self.addCleanup(delay.stop)

    def args(self):
        return {"field": "승객 종류와 인원", "question": "성인/아동 구분과 인원을 알려줘.",
                "why_required": "선택한 좌석의 요금을 계산하려면 승객 구분과 인원이 필요해."}

    def test_default_mode_offers_only_targeted_question_not_general_guidance(self):
        self.assertIn("request_missing_info", self.tools)
        self.assertNotIn("ask_user", self.tools)
        self.assertNotIn("confirm_app_choice", self.tools)

    def test_reply_resumes_goal_and_next_action_without_renewing_budgets(self):
        self.session.actions = 7
        self.device.tap.side_effect = lambda *args: setattr(self.device, "screen", screen(text="성인 1명 선택 완료"))
        goal = "11월 6일 수원에서 세종 11시40분 버스 5번 좌석 선택"
        model = ScriptedModel(responses=[
            call("request_missing_info", self.args(), 1),
            call("click_element", {"snapshot_id": 1, "element_id": "0"}, 2),
            call("finish_task", {"summary": "성인 1명으로 좌석 선택을 확인했어"}, 3),
        ])
        def reply(prompt):
            self.assertEqual(model.position, 1)
            self.assertEqual(self.session.actions, 7)
            self.device.screen = screen(text="성인 1명")
            return "성인 1명이야"
        with patch("builtins.input", side_effect=reply) as ask, redirect_stdout(io.StringIO()):
            run_interactive(self.session, goal, model=model)
        ask.assert_called_once()
        self.device.tap.assert_called_once()
        self.assertTrue(self.session.completed)
        self.assertEqual(self.session.original_command, goal)
        self.assertEqual(self.session.user_information["승객 종류와 인원"], "성인 1명이야")
        self.assertEqual(self.session.actions, 8)
        self.assertEqual(self.session.action_window_start, 0)
        self.assertEqual(self.session.model_calls_total, 3)
        self.assertEqual(self.session.model_window_start, 0)

    def test_repeated_field_uses_stored_reply_without_reasking(self):
        with patch("builtins.input", return_value="성인 1명") as ask, redirect_stdout(io.StringIO()):
            self.tools["request_missing_info"].invoke(self.args())
            repeated = self.tools["request_missing_info"].invoke(self.args())
        ask.assert_called_once()
        self.assertEqual(repeated["source"], "previous_user_answer")
        self.assertEqual(repeated["answer"], "성인 1명")

    def test_empty_answer_waits_locally_and_cancel_exits(self):
        with patch("builtins.input", side_effect=["", "/cancel"]) as ask, redirect_stdout(io.StringIO()):
            with self.assertRaises(TaskCancelled):
                self.tools["request_missing_info"].invoke(self.args())
        self.assertEqual(ask.call_count, 2)
        self.assertEqual(self.session.model_calls_total, 0)

    def test_navigation_question_and_credentials_are_rejected_without_input(self):
        for field, question in [("버튼 위치", "어디를 눌러야 해?"), ("비밀번호", "비밀번호를 알려줘")]:
            args = {**self.args(), "field": field, "question": question}
            with self.subTest(field=field), patch("builtins.input") as ask, self.assertRaises(ValueError):
                self.tools["request_missing_info"].invoke(args)
            ask.assert_not_called()

    def test_reply_is_kept_even_when_screen_cannot_be_read(self):
        self.device.read_screen = lambda: (_ for _ in ()).throw(DeviceError("연결 오류"))
        with patch("builtins.input", return_value="성인 1명"), redirect_stdout(io.StringIO()):
            result = self.tools["request_missing_info"].invoke(self.args())
        self.assertIn("screen_error", result)
        self.assertEqual(self.session.user_information["승객 종류와 인원"], "성인 1명")

    def test_missing_information_failure_redirects_to_targeted_question(self):
        result = self.tools["finish_incomplete"].invoke({"reason": "승객 구분을 모름", "blocker": "missing_information"})
        self.assertEqual(result["status"], "not_verified")
        self.assertIn("request_missing_info", result["remaining_work"])
        self.assertIsNone(self.session.terminal_reason)
