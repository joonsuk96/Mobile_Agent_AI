"""Dynamic playback UI must not prevent a verified goal from completing."""
from contextlib import redirect_stdout
import io
import unittest
from unittest.mock import Mock, patch

from mobile_jarvis_app.agent import PhoneSession, build_tools, run_interactive
from test_agent import FakeCatalog, FakeDevice, ScriptedModel, call, screen


class DynamicCompletionTests(unittest.TestCase):
    def setUp(self):
        self.device = FakeDevice()
        self.session = PhoneSession(self.device, FakeCatalog(), require_plan=False)
        self.tools = {t.name: t for t in build_tools(self.session)}

    def test_playback_progress_change_finishes_once_without_feedback_or_extra_actions(self):
        for package in ["com.google.android.youtube", "com.android.chrome", "com.samsung.android.video"]:
            with self.subTest(package=package):
                self.session = PhoneSession(self.device, FakeCatalog(), require_plan=False)
                elapsed = [screen(package, "요청 플레이리스트 · 일시정지 · 00:01"),
                           screen(package, "요청 플레이리스트 · 일시정지 · 00:02 · 새 자막")]
                self.device.read_screen = Mock(side_effect=elapsed)
                model = ScriptedModel(responses=[
                    call("observe_screen", {}, 1),
                    call("finish_task", {"summary": "요청한 플레이리스트를 재생했어.",
                                         "evidence": "플레이리스트 제목과 일시정지 버튼 및 재생 진행 확인"}, 2),
                ])
                with patch("builtins.input", side_effect=AssertionError("must not ask")), redirect_stdout(io.StringIO()) as output:
                    run_interactive(self.session, "요청한 플레이리스트 재생해줘", model=model)
                self.assertTrue(self.session.completed)
                self.assertEqual(self.device.read_screen.call_count, 2)
                self.assertEqual(model.position, 2)
                self.assertEqual(self.session.autonomous_replans, 0)
                self.assertEqual(output.getvalue().count("요청한 플레이리스트를 재생했어"), 1)
                self.assertNotIn("화면이 바뀌", output.getvalue())
                self.device.tap.assert_not_called()

    def test_same_app_player_activity_transition_is_allowed(self):
        self.session.observe()
        updated = screen()
        updated["activity"] = updated["package"] + "/.Player"
        self.device.screen = updated
        result = self.tools["finish_task"].invoke({"completion_checks": [], "goal_completed": True, "summary": "재생 중"})
        self.assertEqual(result["status"], "completed")

    def test_foreground_app_change_requires_new_observation(self):
        self.session.observe()
        self.device.screen = screen("other.app", "다른 앱")
        result = self.tools["finish_task"].invoke({"completion_checks": [], "goal_completed": True, "summary": "끝"})
        self.assertEqual(result["status"], "not_verified")
        self.assertFalse(self.session.completed)
        self.assertEqual(self.session.screen["package"], "other.app")
        self.assertEqual(self.session.snapshot_id, 2)

    def test_unfinished_goal_is_not_completed_just_because_app_is_same(self):
        self.session.observe()
        result = self.tools["finish_task"].invoke({"completion_checks": [], "summary": "아직 재생 전",
                 "goal_completed": False, "remaining_work": "요청 플레이리스트 재생 시작 확인"})
        self.assertEqual(result["status"], "not_verified")
        self.assertFalse(self.session.completed)

    def test_completion_without_observation_is_still_rejected(self):
        result = self.tools["finish_task"].invoke({"completion_checks": [], "goal_completed": True, "summary": "재생했어"})
        self.assertEqual(result["status"], "not_verified")
        self.assertFalse(self.session.completed)
