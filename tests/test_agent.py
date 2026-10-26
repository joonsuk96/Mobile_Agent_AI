"""Offline tests: no phone, API key, or paid model calls are needed."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from mobile_jarvis_app.agent import PhoneSession, TaskStopped, build_agent, build_tools
from mobile_jarvis_app.apps import AppCatalog
from mobile_jarvis_app.device import AndroidDevice, parse_screen

def screen(package="com.android.settings", text="연결", checkable=False):
    xml = f"""<hierarchy><node package="{package}" enabled="true"
      clickable="true" checkable="{str(checkable).lower()}" class="android.widget.TextView"
      text="{text}" bounds="[0,0][100,100]" resource-id="row"/></hierarchy>"""
    return parse_screen(xml.encode(), {"package": package, "activity": package+"/.Main"})

class FakeDevice:
    serial = "test-phone"
    def __init__(self):
        self.screen = screen()
        self.tap = Mock()
        self.navigate = Mock()
        self.scroll = Mock()
        self.launch_calls = []
    def read_screen(self):
        return self.screen
    def launcher_components(self):
        return {"org.example.bus": "org.example.bus/.Launcher"}
    def launch(self, package, component):
        self.launch_calls.append((package, component))
        self.screen = screen(package, "버스타고")
        return {"requested_package": package, "foreground_matches": True,
                "package": package, "activity": component}

class FakeCatalog:
    def search(self, query, **kwargs):
        return {"apps": [{"name": "버스타고", "package": "org.example.bus"}]}
    def get(self, package):
        if package != "org.example.bus":
            raise ValueError("설치되지 않은 앱")
        return {"name": "버스타고", "package": package}

class ScriptedModel(BaseChatModel):
    responses: list[AIMessage]
    position: int = 0
    parallel_options: list = []
    @property
    def _llm_type(self):
        return "offline-scripted"
    def bind_tools(self, tools, **kwargs):
        self.parallel_options.append(kwargs.get("parallel_tool_calls"))
        return self
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        message = self.responses[self.position]
        self.position += 1
        return ChatResult(generations=[ChatGeneration(message=message)])

def call(name, arguments, number):
    if name == "finish_task":
        arguments = {"goal_completed": True, "completion_checks": [], **arguments}
    return AIMessage(content="", tool_calls=[
        {"name": name, "args": arguments, "id": f"call_{number}", "type": "tool_call"}
    ])

class AgentTests(unittest.TestCase):
    def setUp(self):
        self.device = FakeDevice()
        self.session = PhoneSession(self.device, FakeCatalog(), interactive=True, require_plan=False)
        self.tools = {tool.name: tool for tool in build_tools(self.session)}
        self.delay = patch("mobile_jarvis_app.agent.time.sleep")
        self.delay.start()
        self.addCleanup(self.delay.stop)

    def test_real_langchain_agent_discovers_launches_and_finishes(self):
        model = ScriptedModel(responses=[
            call("search_apps", {"query": "버스타고"}, 1),
            call("launch_app", {"package": "org.example.bus"}, 2),
            call("finish_task", {"summary": "버스타고를 열었어."}, 3),
        ])
        with redirect_stdout(io.StringIO()):
            result = build_agent(self.session, model=model).invoke(
                {"messages": [{"role": "user", "content": "버스타고 열어줘"}]},
                config={"recursion_limit": 30})
        self.assertTrue(self.session.completed)
        self.assertEqual(self.device.launch_calls,
                         [("org.example.bus", "org.example.bus/.Launcher")])
        self.device.tap.assert_not_called()
        self.assertEqual(model.parallel_options, [False, False, False])
        self.assertIn("completed", result["messages"][-1].content)

    def test_unknown_package_cannot_launch(self):
        with self.assertRaises(ValueError):
            self.tools["launch_app"].invoke({"package": "org.invented.app"})
        self.assertEqual(self.device.launch_calls, [])

    def test_stale_snapshot_does_not_click(self):
        self.session.observe()
        with self.assertRaises(ValueError):
            self.tools["click_element"].invoke({"snapshot_id": 0, "element_id": "0"})
        self.device.tap.assert_not_called()

    def test_changed_screen_does_not_click_old_coordinates(self):
        observation = self.session.observe()
        self.device.screen = screen(text="다른 화면")
        result = self.tools["click_element"].invoke(
            {"snapshot_id": observation["snapshot_id"], "element_id": "0"})
        self.assertIn("error", result)
        self.device.tap.assert_not_called()

    def test_checkable_element_is_blocked(self):
        self.device.screen = screen(text="Wi-Fi", checkable=True)
        observation = self.session.observe()
        with self.assertRaises(ValueError):
            self.tools["click_element"].invoke(
                {"snapshot_id": observation["snapshot_id"], "element_id": "0"})
        self.device.tap.assert_not_called()

    def test_action_limit_blocks_eleventh_action(self):
        self.session.actions = self.session.max_actions
        with self.assertRaises(TaskStopped):
            self.session.perform("test", self.device.tap)
        self.device.tap.assert_not_called()

    def test_unchanged_screen_stops(self):
        self.session.observe()
        with redirect_stdout(io.StringIO()):
            self.session.perform("test", lambda: None)
            with self.assertRaises(TaskStopped):
                self.session.perform("test", lambda: None)

    def test_ask_user_refreshes_screen_after_guidance(self):
        self.session.observe()
        with patch("builtins.input", return_value="와이파이"), redirect_stdout(io.StringIO()):
            result = self.tools["ask_user"].invoke({"question": "어떤 설정?"})
        self.assertEqual(result["reply"], "와이파이")
        self.assertIsNotNone(self.session.screen)
        self.assertIn("screen", result)

    def test_cannot_finish_without_observation(self):
        result = self.tools["finish_task"].invoke({"completion_checks": [], "goal_completed": True, "summary": "완료"})
        self.assertEqual(result["status"], "not_verified")
        self.assertFalse(self.session.completed)

    def test_parent_click_target_and_system_ui_filter(self):
        xml = b'''<hierarchy>
          <node package="com.android.systemui" text="12:34" bounds="[0,0][1,1]"/>
          <node package="example.app" clickable="true" enabled="true" bounds="[0,20][100,100]">
            <node package="example.app" text="Wi-Fi" bounds="[10,30][30,50]"/>
          </node></hierarchy>'''
        parsed = parse_screen(xml, {"package": "example.app", "activity": "example.app/.Main"})
        self.assertEqual(len(parsed["elements"]), 1)
        self.assertEqual(parsed["elements"][0]["x"], 50)
        self.assertEqual(parsed["elements"][0]["y"], 60)

    def test_tool_error_returns_to_model_instead_of_crashing(self):
        model = ScriptedModel(responses=[
            call("launch_app", {"package": "org.invented.app"}, 1),
            AIMessage(content="설치된 앱을 확인할 수 없어요."),
        ])
        result = build_agent(self.session, model=model).invoke(
            {"messages": [{"role": "user", "content": "앱 열어줘"}]})
        self.assertIn("확인할 수", result["messages"][-1].content)
        self.assertEqual(self.device.launch_calls, [])

    def test_password_text_is_not_exposed_in_ui_observation(self):
        xml = b'<hierarchy><node password="true" text="secret-value" enabled="true" bounds="[0,0][100,100]"/></hierarchy>'
        parsed = parse_screen(xml, {"package": "example.app", "activity": "example.app/.Main"})
        self.assertNotIn("secret-value", str(parsed))
        self.assertTrue(parsed["elements"][0]["password"])

    def test_model_call_limit_ends_run_without_extra_calls(self):
        self.session.max_model_calls = 3
        model = ScriptedModel(responses=[
            call("observe_screen", {}, number) for number in range(25)
        ])
        with redirect_stdout(io.StringIO()):
            build_agent(self.session, model=model).invoke(
                {"messages": [{"role": "user", "content": "설정 확인"}]},
                config={"recursion_limit": 200})
        self.assertEqual(model.position, 3)
        self.device.tap.assert_not_called()

class CatalogTests(unittest.TestCase):
    def test_korean_apk_label_search_cache_update_and_no_hardcoded_package(self):
        device = Mock()
        device.serial = "serial"
        device.launcher_components.return_value = {"org.random.transport": "org.random.transport/.Main"}
        device.text.return_value = "device-build"
        device.apk_paths.return_value = ["/data/app/version1/base.apk"]
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory, \
             patch("mobile_jarvis_app.apps.read_apk_labels",
                   return_value=["버스타고", "BusTago"]) as labels, \
             redirect_stdout(io.StringIO()):
            catalog = AppCatalog(device, Path(directory))
            found = catalog.search("버스타고")
            self.assertEqual(found["apps"][0]["package"], "org.random.transport")
            labels.assert_called_once()
            again = AppCatalog(device, Path(directory))
            again.search("BusTago")
            self.assertEqual(labels.call_count, 1)
            device.apk_paths.return_value = ["/data/app/version2/base.apk"]
            AppCatalog(device, Path(directory)).search("버스타고", refresh=True)
            self.assertEqual(labels.call_count, 2)
            self.assertFalse(list(Path(directory).rglob("*.apk")))

    def test_package_discovery_parser(self):
        device = AndroidDevice(Path("adb"))
        with patch.object(device, "text", return_value=
                          "org.random.transport/.Main\ncom.android.settings/.Settings\n"):
            self.assertEqual(device.launcher_components()["org.random.transport"],
                             "org.random.transport/.Main")

    def test_launch_reports_foreground_mismatch(self):
        device = AndroidDevice(Path("adb"))
        with patch.object(device, "text", return_value="Status: ok"), \
             patch.object(device, "current_app", return_value=
                          {"package": "com.android.launcher", "activity": "com.android.launcher/.Main"}), \
             patch("mobile_jarvis_app.device.time.sleep"):
            result = device.launch("org.random.transport", "org.random.transport/.Main")
        self.assertFalse(result["foreground_matches"])


class AppAliasTests(unittest.TestCase):
    def make_catalog(self, directory, labels=None, package="com.ebcard.bustago"):
        alias_path = Path(directory) / "app_aliases.json"
        alias_path.write_text(json.dumps({
            "버스타고": ["bustago", "bus tago"],
            "내메모": ["Notebook"],
        }, ensure_ascii=False), encoding="utf-8")
        catalog = AppCatalog(Mock(), Path(directory) / "cache", alias_path=alias_path)
        catalog.ready = True
        catalog.apps = {
            package: {"package": package, "component": package + "/.Main",
                      "name": labels[0] if labels else package, "labels": labels or [], "paths": []}
        }
        return catalog, alias_path

    def test_korean_alias_finds_package_when_apk_label_is_empty(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            catalog, _ = self.make_catalog(directory)
            result = catalog.search("버스타고")
            self.assertEqual(result["apps"][0]["package"], "com.ebcard.bustago")
            self.assertEqual(result["apps"][0]["matched_query"], "bustago")
            self.assertEqual(result["apps"][0]["match_type"], "alias_exact")
            self.assertIn("버스타고", result["apps"][0]["aliases"])

    def test_english_query_finds_korean_only_metadata(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            catalog, _ = self.make_catalog(directory, ["버스타고"], "org.random.transport")
            result = catalog.search("Bus Tago")
            self.assertEqual(result["apps"][0]["package"], "org.random.transport")

    def test_custom_alias_works_without_hardcoded_package(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            catalog, _ = self.make_catalog(directory, ["Notebook"], "org.other.notes")
            self.assertEqual(catalog.search("내메모")["apps"][0]["package"], "org.other.notes")

    def test_aliases_do_not_create_uninstalled_app_candidates(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            catalog, _ = self.make_catalog(directory, ["Calculator"], "org.other.calc")
            self.assertEqual(catalog.search("버스타고")["apps"], [])

    def test_spacing_case_punctuation_and_decomposed_korean(self):
        import unicodedata
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            catalog, _ = self.make_catalog(directory)
            for query in ["버스 타고", "버스타고!", "BUS-TAGO", unicodedata.normalize("NFD", "버스타고")]:
                with self.subTest(query=query):
                    self.assertEqual(catalog.search(query)["total_matches"], 1)

    def test_alias_file_changes_apply_without_rebuilding_app_cache(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            catalog, alias_path = self.make_catalog(directory)
            alias_path.write_text('{"교통앱": ["bustago"]}', encoding="utf-8")
            with patch.object(catalog, "build") as build:
                result = catalog.search("교통앱")
            self.assertEqual(result["apps"][0]["package"], "com.ebcard.bustago")
            build.assert_called_once_with(refresh=False)

    def test_ambiguous_alias_keeps_both_candidates(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            catalog, _ = self.make_catalog(directory)
            catalog.apps["org.other.bus"] = {
                "name": "BusTago", "labels": ["BusTago"], "package": "org.other.bus",
                "component": "org.other.bus/.Main", "paths": []}
            self.assertEqual(catalog.search("버스타고")["total_matches"], 2)

    def test_malformed_aliases_return_clear_error(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            catalog, alias_path = self.make_catalog(directory)
            alias_path.write_text('{"버스타고": "bustago"}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "문자열 목록"):
                catalog.search("버스타고")

    def test_cached_empty_labels_are_not_pulled_again(self):
        device = Mock()
        device.serial = "test-cache-device"
        device.launcher_components.return_value = {"com.ebcard.bustago": "com.ebcard.bustago/.Main"}
        device.text.return_value = "same-build"
        device.apk_paths.return_value = ["/data/app/version1/base.apk"]
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory, \
             patch("mobile_jarvis_app.apps.read_apk_labels", return_value=[]) as labels, \
             redirect_stdout(io.StringIO()):
            first = AppCatalog(device, Path(directory))
            first.build()
            second = AppCatalog(device, Path(directory))
            second.build()
            self.assertEqual(labels.call_count, 1)
            self.assertEqual(device.pull.call_count, 1)


if __name__ == "__main__":
    unittest.main()
