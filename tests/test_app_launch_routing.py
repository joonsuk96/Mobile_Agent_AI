"""App launch routing must not fall back to web browsing or icon hunting."""
from contextlib import redirect_stdout
import io
import unittest
from unittest.mock import Mock, patch

from pydantic import Field

from mobile_jarvis_app.agent import PhoneSession, build_tools, requested_app_name, run_interactive
from test_agent import FakeDevice, ScriptedModel, call, screen

YOUTUBE = "com.google.android.youtube"
BROWSER = "com.sec.android.app.sbrowser"


class Catalog:
    def __init__(self, exact=False):
        self.exact = exact
        self.searches = []
    def search(self, query, **kwargs):
        self.searches.append(query)
        youtube = {"name": "YouTube", "labels": ["YouTube"], "package": YOUTUBE}
        if self.exact:
            return {"apps": [{**youtube, "match_type": "exact"}]}
        return {"apps": [], "resolution_mode": "semantic_required", "semantic_catalog": {
            "apps": [youtube, {"name": "Samsung Internet", "labels": [], "package": BROWSER}]}}
    def get(self, package):
        if package not in {YOUTUBE, BROWSER}:
            raise ValueError("unknown app")
        return {"name": "YouTube" if package == YOUTUBE else "Samsung Internet", "package": package}


class ToolRecordingModel(ScriptedModel):
    available: list = Field(default_factory=list)
    def bind_tools(self, tools, **kwargs):
        self.available.append([item.name for item in tools])
        return super().bind_tools(tools, **kwargs)


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.device = FakeDevice()
        self.device.launcher_components = Mock(return_value={YOUTUBE: YOUTUBE+"/.Main", BROWSER: BROWSER+"/.Main"})
        self.device.search_web = Mock()
        self.device.capture_screen = Mock()
        def launch(package, component):
            self.device.launch_calls.append((package, component))
            self.device.screen = screen(package, "YouTube")
            return {"foreground_matches": True}
        self.device.launch = launch
        self.catalog = Catalog()
        self.session = PhoneSession(self.device, self.catalog, require_plan=False)
        delay = patch("mobile_jarvis_app.agent.time.sleep")
        delay.start()
        self.addCleanup(delay.stop)

    def test_korean_request_discovers_english_app_and_launches_without_browser(self):
        model = ToolRecordingModel(responses=[
            call("launch_app", {"package": YOUTUBE}, 1),
            call("finish_task", {"summary": "유튜브를 열었어"}, 2),
        ])
        with patch("builtins.input") as ask, redirect_stdout(io.StringIO()):
            run_interactive(self.session, "유튜브 켜줘", model=model)
        self.assertEqual(self.catalog.searches, ["유튜브"])
        self.assertEqual(self.device.launch_calls, [(YOUTUBE, YOUTUBE+"/.Main")])
        self.device.search_web.assert_not_called()
        self.device.capture_screen.assert_not_called()
        ask.assert_not_called()
        self.assertNotIn("search_web", model.available[0])
        self.assertNotIn("click_screen", model.available[0])
        self.assertIn("launch_app", model.available[0])
        self.assertIn("search_web", model.available[1])
        self.assertTrue(self.session.completed)

    def test_browser_and_screen_actions_blocked_before_direct_app_launch(self):
        with redirect_stdout(io.StringIO()):
            self.session.prepare_app_route("유튜브 켜줘")
        tools = {t.name: t for t in build_tools(self.session)}
        with self.assertRaises(ValueError):
            tools["search_web"].invoke({"query": "YouTube", "browser_package": BROWSER})
        with self.assertRaises(ValueError):
            tools["navigate"].invoke({"direction": "home"})
        result = tools["finish_task"].invoke({"completion_checks": [], "goal_completed": True, "summary": "끝"})
        self.assertEqual(result["status"], "not_verified")
        self.device.search_web.assert_not_called()
        self.device.navigate.assert_not_called()

    def test_exact_target_rejects_browser_even_if_installed(self):
        self.catalog.exact = True
        with redirect_stdout(io.StringIO()):
            self.session.prepare_app_route("YouTube 열어줘")
        tools = {t.name: t for t in build_tools(self.session)}
        with self.assertRaises(ValueError):
            tools["launch_app"].invoke({"package": BROWSER})
        self.assertEqual(self.device.launch_calls, [])

    def test_general_web_request_keeps_web_tools_available(self):
        model = ToolRecordingModel(responses=[
            call("observe_screen", {}, 1),
            call("finish_task", {"summary": "확인"}, 2),
        ])
        with redirect_stdout(io.StringIO()):
            run_interactive(self.session, "인터넷에서 메이플인벤 검색해줘", model=model)
        self.assertEqual(self.catalog.searches, [])
        self.assertIn("search_web", model.available[0])
        self.assertFalse(self.session.app_route_pending)

    def test_compound_youtube_request_routes_to_app_before_search(self):
        model = ToolRecordingModel(responses=[
            call("launch_app", {"package": YOUTUBE}, 1),
            call("finish_task", {"summary": "두 번째 최근 영상 조회수는 7.5만 회"}, 2),
        ])
        with patch("builtins.input") as ask, redirect_stdout(io.StringIO()):
            run_interactive(self.session, "유튜브 앱 켜서 황룡갑 검색해서 2번째로 최근 영상 확인해서 조회수 알려줘", model=model)
        self.assertEqual(self.catalog.searches, ["유튜브"])
        self.device.search_web.assert_not_called()
        self.assertNotIn("search_web", model.available[0])
        ask.assert_not_called()
        self.assertTrue(self.session.completed)

    def test_compound_app_prefix_extraction(self):
        for text, name in [
            ("유튜브 앱 켜서 황룡갑 검색해서 조회수 알려줘", "유튜브"),
            ("유튜브를 열어서 영상 검색해줘", "유튜브"),
            ("버스타고 실행해서 수원 출발 버스 찾아줘", "버스타고"),
            ("카카오톡 앱에서 친구 목록 보여줘", "카카오톡"),
            ("먼저 유튜브 앱을 켜서 황룡갑 검색해줘", "유튜브"),
            ("open YouTube app and search for a video", "YouTube"),
        ]:
            with self.subTest(text=text):
                self.assertEqual(requested_app_name(text), name)

    def test_explicit_app_opening_phrases(self):
        for text, name in [("유튜브 켜줘", "유튜브"), ("유튜브 앱을 켜줘", "유튜브"),
                           ("버스타고를 실행시켜줘", "버스타고"), ("카카오톡 좀 열어줘", "카카오톡"),
                           ("open YouTube app", "YouTube"), ("삼성 인터넷 실행해줘", "삼성 인터넷")]:
            with self.subTest(text=text):
                self.assertEqual(requested_app_name(text), name)
        for text in ["인터넷에서 유튜브 검색해줘", "메이플인벤 검색해서 열어줘", "이 링크 열어줘"]:
            with self.subTest(text=text):
                self.assertIsNone(requested_app_name(text))
