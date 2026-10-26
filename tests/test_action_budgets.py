"""Execution budget regression tests; no phone or model calls."""
from contextlib import redirect_stdout
import io
import os
import unittest
from unittest.mock import patch

from mobile_jarvis_app.agent import PhoneSession, TaskStopped
from test_agent import FakeCatalog, FakeDevice, screen


class ActionBudgetTests(unittest.TestCase):
    def make_session(self, **kwargs):
        with patch.dict(os.environ, {}, clear=True):
            return PhoneSession(FakeDevice(), FakeCatalog(), **kwargs, require_plan=False)

    def test_defaults_allow_one_hundred_actions_and_matching_call_limits(self):
        session = self.make_session()
        self.assertEqual(session.max_actions, 100)
        self.assertEqual(session.max_model_calls, 200)
        self.assertEqual(session.max_tool_calls, 400)

    def test_thirty_changing_scrolls_succeed_and_thirty_first_is_blocked(self):
        session = self.make_session(max_actions=30)
        with patch("mobile_jarvis_app.agent.time.sleep"), redirect_stdout(io.StringIO()):
            session.observe()
            for page in range(30):
                def move(page=page):
                    session.device.screen = screen(text=f"터미널 페이지 {page}")
                result = session.perform("스크롤: down", move)
            self.assertEqual(result["screen"]["actions_remaining"], 0)
            self.assertEqual(result["screen"]["action_limit"], 30)
            with self.assertRaisesRegex(TaskStopped, "30회"):
                session.perform("스크롤: down", lambda: None)
        self.assertEqual(session.actions, 30)

    def test_command_line_override_expands_default_model_and_tool_allowance(self):
        session = self.make_session(max_actions=50)
        self.assertEqual(session.max_actions, 50)
        self.assertEqual(session.max_model_calls, 100)
        self.assertEqual(session.max_tool_calls, 200)

    def test_environment_limits_and_command_line_precedence(self):
        with patch.dict(os.environ, {
            "JARVIS_MAX_ACTIONS": "40", "JARVIS_MAX_MODEL_CALLS": "90",
            "JARVIS_MAX_TOOL_CALLS": "180",
        }, clear=True):
            session = PhoneSession(FakeDevice(), FakeCatalog(), max_actions=50, require_plan=False)
        self.assertEqual((session.max_actions, session.max_model_calls, session.max_tool_calls),
                         (50, 90, 180))

    def test_replan_and_recovery_defaults_and_environment_overrides(self):
        session = self.make_session()
        self.assertEqual((session.max_replans, session.max_recoveries), (10, 5))
        with patch.dict(os.environ, {"JARVIS_MAX_REPLANS": "15", "JARVIS_MAX_RECOVERIES": "8"}, clear=True):
            session = PhoneSession(FakeDevice(), FakeCatalog(), require_plan=False)
        self.assertEqual((session.max_replans, session.max_recoveries), (15, 8))

    def test_invalid_replan_or_recovery_limit_is_rejected(self):
        for name in ("JARVIS_MAX_REPLANS", "JARVIS_MAX_RECOVERIES"):
            with self.subTest(name=name), patch.dict(os.environ, {name: "0"}, clear=True):
                with self.assertRaisesRegex(ValueError, name):
                    PhoneSession(FakeDevice(), FakeCatalog(), require_plan=False)

    def test_invalid_environment_limit_has_clear_message(self):
        for value in ("0", "-1", "abc"):
            with self.subTest(value=value), patch.dict(os.environ, {
                "JARVIS_MAX_ACTIONS": value,
            }, clear=True):
                with self.assertRaisesRegex(ValueError, "JARVIS_MAX_ACTIONS"):
                    PhoneSession(FakeDevice(), FakeCatalog(), require_plan=False)


if __name__ == "__main__":
    unittest.main()
