"""App failure should enable real alternative actions instead of immediate exit."""
from contextlib import redirect_stdout
import io
import unittest
from unittest.mock import Mock, patch

from mobile_jarvis_app.agent import PhoneSession, build_tools, run_interactive
from mobile_jarvis_app.device import DeviceError
from test_agent import FakeDevice, ScriptedModel, call, screen
from test_app_launch_routing import Catalog, YOUTUBE, BROWSER


class AlternativeRouteTests(unittest.TestCase):
    def setUp(self):
        self.device = FakeDevice()
        self.device.search_web = Mock(side_effect=self.open_search)
        self.device.launcher_components = Mock(return_value={YOUTUBE: YOUTUBE+"/.Main", BROWSER: BROWSER+"/.Main"})
        self.session = PhoneSession(self.device, Catalog(), require_plan=False)
        delay = patch("mobile_jarvis_app.agent.time.sleep")
        delay.start()
        self.addCleanup(delay.stop)
    def open_search(self, query, package):
        self.device.screen = screen(package, "YouTube 웹사이트")
        return {"query": query, "browser_package": package}

    def test_premature_give_up_uses_browser_then_finishes_without_input(self):
        model = ScriptedModel(responses=[
            call("finish_incomplete", {"reason": "유튜브 앱을 찾지 못했어"}, 1),
            call("search_apps", {"query": "Samsung Internet"}, 2),
            call("search_web", {"query": "YouTube", "browser_package": BROWSER}, 3),
            call("finish_task", {"summary": "브라우저에서 유튜브 페이지를 열었어"}, 4),
        ])
        with patch("builtins.input", side_effect=AssertionError("must not ask")), redirect_stdout(io.StringIO()) as output:
            run_interactive(self.session, "유튜브 켜줘", model=model)
        self.device.search_web.assert_called_once_with("YouTube", BROWSER)
        self.assertTrue(self.session.completed)
        self.assertIsNone(self.session.terminal_reason)
        self.assertEqual(self.session.alternative_actions, 1)
        self.assertIn("대체 경로", output.getvalue())

    def test_real_launch_error_allows_browser_alternative(self):
        self.device.launch = Mock(side_effect=DeviceError("앱 실행 실패"))
        model = ScriptedModel(responses=[
            call("launch_app", {"package": YOUTUBE}, 1),
            call("try_alternative", {"reason": "직접 실행 실패. 브라우저에서 같은 서비스를 열기"}, 2),
            call("search_web", {"query": "YouTube", "browser_package": BROWSER}, 3),
            call("finish_task", {"summary": "유튜브 웹페이지를 열었어"}, 4),
        ])
        with patch("builtins.input", side_effect=AssertionError("must not ask")), redirect_stdout(io.StringIO()):
            run_interactive(self.session, "유튜브 켜줘", model=model)
        self.device.launch.assert_called_once()
        self.device.search_web.assert_called_once()
        self.assertTrue(self.session.completed)
        self.assertEqual(self.session.actions, 2)

    def test_fallback_does_not_renew_action_budget_or_claim_success(self):
        self.session.actions = 7
        self.session.max_actions = 8
        with redirect_stdout(io.StringIO()):
            self.session.prepare_app_route("유튜브 켜줘")
            self.session.enable_alternative("실행 경로 실패")
        self.assertEqual(self.session.actions, 7)
        self.assertEqual(self.session.action_window_start, 0)
        self.assertFalse(self.session.completed)
        self.assertFalse(self.session.app_route_pending)

    def test_cannot_give_up_without_trying_alternative_action(self):
        with redirect_stdout(io.StringIO()):
            self.session.prepare_app_route("유튜브 켜줘")
        tools = {tool.name: tool for tool in build_tools(self.session)}
        with redirect_stdout(io.StringIO()):
            first = tools["finish_incomplete"].invoke({"reason": "앱 못 찾음"})
            second = tools["finish_incomplete"].invoke({"reason": "앱 못 찾음"})
        self.assertEqual(first["status"], "not_verified")
        self.assertEqual(second["status"], "not_verified")
        self.assertIsNone(self.session.terminal_reason)

    def test_authentication_failure_is_not_bypassed(self):
        with redirect_stdout(io.StringIO()):
            self.session.prepare_app_route("유튜브 켜줘")
        tools = {tool.name: tool for tool in build_tools(self.session)}
        from test_text_input import input_screen
        self.device.screen = input_screen(password=True)
        self.session.observe()
        result = tools["finish_incomplete"].invoke({"reason": "인증 필요", "blocker": "authentication", "blocking_evidence": "보호된 비밀번호 입력란이 표시됨"})
        self.assertEqual(result["status"], "incomplete")
        self.assertIsNone(self.session.alternative_route)
