"""Exactly one Jarvis answer, with immediate return on verified completion."""
from contextlib import redirect_stdout
import io
import unittest
from unittest.mock import Mock, patch

from mobile_jarvis_app.agent import PhoneSession, emit_final_answer, run_interactive
from test_agent import FakeCatalog, FakeDevice, ScriptedModel, call, screen


class FinalAnswerTests(unittest.TestCase):
    def setUp(self):
        self.device = FakeDevice()
        self.session = PhoneSession(self.device, FakeCatalog())

    def test_verified_answer_prints_once_and_no_later_action_or_model_call_occurs(self):
        self.device.screen = screen(text="게시글 제목: 테스트 제목")
        original_read = self.device.read_screen
        self.device.read_screen = Mock(side_effect=original_read)
        model = ScriptedModel(responses=[
            call("set_task_plan", {"goal": "게시글 제목 확인", "constraints": [], "success_criteria": ["제목 확인"], "steps": ["화면에서 제목 읽기"]}, 1),
            call("observe_screen", {}, 2),
            call("record_task_progress", {"criterion_id": "C1", "status": "satisfied", "snapshot_id": 1, "evidence_text": "테스트 제목"}, 3),
            call("finish_task", {"summary": "요청한 게시글 제목은 테스트 제목이야."}, 4),
            call("navigate", {"direction": "home"}, 5),
        ])
        with patch("builtins.input", side_effect=AssertionError("must not ask")), redirect_stdout(io.StringIO()) as output:
            run_interactive(self.session, "현재 게시글 제목 알려줘", model=model)
        lines = [line for line in output.getvalue().splitlines() if line.startswith("자비스 : ")]
        self.assertEqual(lines, ["자비스 : 요청한 게시글 제목은 테스트 제목이야."])
        self.assertEqual(model.position, 4)
        self.assertEqual(self.device.read_screen.call_count, 2)
        self.device.navigate.assert_not_called()
        self.assertTrue(self.session.final_answer_printed)
        self.assertTrue(self.session.completed)

    def test_renderer_is_idempotent_and_removes_duplicate_model_prefix(self):
        self.session.completed = True
        with redirect_stdout(io.StringIO()) as output:
            emit_final_answer(self.session, {"summary": "자비스: 요청한 영상을 재생했어.\n광고도 건너뛰었어."})
            emit_final_answer(self.session, {"summary": "두 번째 답변"})
        self.assertEqual(output.getvalue().count("자비스 :"), 1)
        self.assertIn("자비스 : 요청한 영상을 재생했어. 광고도 건너뛰었어.", output.getvalue())
        self.assertNotIn("두 번째 답변", output.getvalue())

    def test_verification_failure_has_short_honest_sentence_not_raw_dictionary(self):
        self.session.completion_validation_failures = 2
        self.session.last_completion_problem = "정렬된 후보 목록이 없어"
        self.session.terminal_reason = "technical details"
        with redirect_stdout(io.StringIO()) as output:
            emit_final_answer(self.session, {"summary": "영상 재생에 성공했습니다"})
        text = output.getvalue()
        self.assertIn("자비스 : 요청한 항목의 순번과 정렬을 검증하지 못해", text)
        self.assertNotIn("성공했습니다", text)
        self.assertNotIn("technical details", text)
        self.assertNotIn("{", text)

    def test_action_limit_exits_with_single_jarvis_answer(self):
        self.session = PhoneSession(self.device, FakeCatalog(), max_actions=1, require_plan=False)
        self.session.actions = 1
        self.session.observe()
        model = ScriptedModel(responses=[call("navigate", {"direction": "back"}, 1)])
        with patch("builtins.input", side_effect=AssertionError("must not ask")), redirect_stdout(io.StringIO()) as output:
            result = run_interactive(self.session, "뒤로 이동", model=model)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(output.getvalue().count("자비스 :"), 1)
        self.device.navigate.assert_not_called()
