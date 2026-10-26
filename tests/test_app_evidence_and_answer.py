"""Observed app identity and bounded model-written terminal explanations."""
from contextlib import redirect_stdout
import io
import unittest
from unittest.mock import Mock, patch
from langchain_core.messages import AIMessage
from mobile_jarvis_app.agent import PhoneSession, build_tools, emit_final_answer
from mobile_jarvis_app.planning import GoalTracker
from test_agent import FakeCatalog, FakeDevice, screen


class AppEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tracker = GoalTracker()
        self.tracker.set_plan("영상 재생", [], ["YouTube 앱이 전면 실행됨"], ["앱 열기"])
        self.tracker.app_names["com.google.android.youtube"] = {"YouTube", "유튜브"}
        self.tracker.remember(1, screen("com.google.android.youtube", "검색"))

    def test_display_name_app_claim_resolves_to_observed_package(self):
        for source in ("app", "ui"):
            for text in ("YouTube 앱이 전면 실행됨", "유튜브 앱 실행", "YouTube"):
                with self.subTest(source=source, text=text):
                    evidence = self.tracker.resolve_record_evidence("C1", text, 1, source, [])
                    self.assertEqual(evidence["source"], "app")
                    self.assertEqual(evidence["text"], "com.google.android.youtube")

    def test_batch_completes_with_display_name_instead_of_literal_package(self):
        from mobile_jarvis_app.planning import CompletionCheck
        missing = self.tracker.apply_completion([
            CompletionCheck(criterion_id="C1", evidence_text="YouTube 앱이 전면 실행됨", snapshot_id=1, source="app")
        ], None, None, {}, [])
        self.assertEqual(missing, [])
        self.assertEqual(self.tracker.criteria["C1"]["evidence"]["text"], "com.google.android.youtube")

    def test_wrong_package_or_unregistered_name_is_rejected(self):
        for text in ("fake.app", "YouTube Music 앱 실행", "Chrome 앱 실행"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.tracker.resolve_record_evidence("C1", text, 1, "app", [])
        self.tracker.remember(2, screen("org.other.app", "검색"))
        with self.assertRaises(ValueError):
            self.tracker.resolve_record_evidence("C1", "YouTube 앱 실행", 2, "app", [])

    def test_successful_launch_records_actual_catalog_name_and_user_alias(self):
        session = PhoneSession(FakeDevice(), FakeCatalog(), require_plan=False)
        session.prepare_app_route("버스타고 앱 열어줘")
        tools = {tool.name: tool for tool in build_tools(session)}
        with patch("mobile_jarvis_app.agent.time.sleep"), redirect_stdout(io.StringIO()):
            tools["launch_app"].invoke({"package": "org.example.bus"})
        self.assertIn("버스타고", session.goal_tracker.app_names["org.example.bus"])


class ModelFinalAnswerTests(unittest.TestCase):
    def session(self):
        session = PhoneSession(FakeDevice(), FakeCatalog())
        session.original_command = "성시경 플레이리스트 재생해줘"
        session.completion_validation_failures = 2
        session.last_completion_problem = "재생 상태 미확인"
        session.last_completion_summary = "성시경 플레이리스트 재생했습니다"
        return session

    def test_unverified_result_is_explained_by_model_once_without_changing_status(self):
        session = self.session()
        session.final_response_model = Mock()
        session.final_response_model.invoke.return_value = AIMessage(content="성시경 검색은 진행했어. 현재 재생 상태는 확인하지 못했어.")
        with redirect_stdout(io.StringIO()) as output:
            emit_final_answer(session)
            emit_final_answer(session)
        session.final_response_model.invoke.assert_called_once()
        self.assertIn("자비스 : 성시경 검색은 진행했어.", output.getvalue())
        self.assertFalse(session.completed)
        self.assertEqual(session.model_calls_total, 1)

    def test_verified_summary_has_no_additional_model_call(self):
        session = self.session()
        session.completed = True
        session.final_response_model = Mock()
        with redirect_stdout(io.StringIO()) as output:
            emit_final_answer(session, {"summary": "성시경 플레이리스트를 재생했어."})
        session.final_response_model.invoke.assert_not_called()
        self.assertIn("자비스 : 성시경 플레이리스트를 재생했어.", output.getvalue())

    def test_model_error_or_exhausted_budget_uses_fallback(self):
        for exhausted in (False, True):
            session = self.session()
            session.final_response_model = Mock()
            session.final_response_model.invoke.side_effect = RuntimeError("offline")
            if exhausted:
                session.model_calls_total = session.max_model_calls
            with redirect_stdout(io.StringIO()) as output:
                emit_final_answer(session)
            self.assertIn("자비스 :", output.getvalue())
            self.assertFalse(session.completed)
            if exhausted:
                session.final_response_model.invoke.assert_not_called()
