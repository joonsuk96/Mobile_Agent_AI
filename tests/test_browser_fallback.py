"""Browser search and multimodal fallback regression tests, without live API calls."""
import base64
from contextlib import redirect_stdout
import io
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from langchain_core.messages import ToolMessage
from langchain_openai.chat_models.base import _ensure_valid_tool_message_content

from mobile_jarvis_app.agent import PhoneSession, build_agent, build_tools, run_interactive, show_agent_result
from mobile_jarvis_app.device import AndroidDevice, DeviceError, parse_screen
from test_agent import FakeCatalog, FakeDevice, call, screen
from test_interactive_recovery import RecordingModel

PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a9XkAAAAASUVORK5CYII=")
PACKAGE = "org.example.bus"


def empty_screen():
    return parse_screen(b"<hierarchy/>", {"package": PACKAGE, "activity": PACKAGE + "/.Main"})


class BrowserFallbackTests(unittest.TestCase):
    def setUp(self):
        self.device = FakeDevice()
        self.device.screen = empty_screen()
        self.device.current_app = Mock(return_value={"package": PACKAGE, "activity": PACKAGE + "/.Main"})
        self.device.capture_screen = Mock(return_value={
            "image_url": "data:image/png;base64," + base64.b64encode(PNG).decode(),
            "width": 1, "height": 1, "package": PACKAGE, "activity": PACKAGE + "/.Main"})
        self.device.search_web = Mock(return_value={"query": "메이플인벤"})
        self.session = PhoneSession(self.device, FakeCatalog(), require_plan=False)
        self.tools = {tool.name: tool for tool in build_tools(self.session)}
        delay = patch("mobile_jarvis_app.agent.time.sleep")
        delay.start()
        self.addCleanup(delay.stop)

    def test_search_empty_ui_image_click_and_finish_without_human_help(self):
        self.device.tap.side_effect = lambda *args: setattr(
            self.device, "screen", screen(PACKAGE, "10추 첫 게시글 제목"))
        model = RecordingModel(responses=[
            call("search_web", {"query": "메이플인벤", "browser_package": PACKAGE}, 1),
            call("observe_screen", {}, 2),
            call("click_screen", {"snapshot_id": 2, "x": 0, "y": 0, "target": "인벤 결과 링크"}, 3),
            call("finish_task", {"summary": "10추 첫 게시글 제목을 확인했어"}, 4),
        ])
        with patch("builtins.input") as ask, redirect_stdout(io.StringIO()):
            run_interactive(self.session, "인벤 10추 첫 글 제목 확인", model=model)
        ask.assert_not_called()
        self.device.search_web.assert_called_once_with("메이플인벤", PACKAGE)
        self.device.tap.assert_called_once_with(0, 0)
        self.assertTrue(self.session.completed)
        images = [m for m in model.histories[2] if isinstance(m, ToolMessage)
                  and isinstance(m.content, list)]
        self.assertEqual(len(images), 1)
        blocks = images[0].content
        self.assertEqual(blocks[1]["type"], "image_url")
        self.assertNotIn("base64", blocks[0]["text"])
        # Verify the installed SDK sends an image block rather than a JSON string.
        converted = _ensure_valid_tool_message_content(blocks)
        self.assertEqual(converted[1]["type"], "input_image")

    def test_readable_ui_does_not_capture_image_by_default(self):
        self.device.screen = screen(PACKAGE)
        self.tools["observe_screen"].invoke({})
        self.device.capture_screen.assert_not_called()

    def test_missing_or_stale_visual_observation_never_taps(self):
        with self.assertRaises(ValueError):
            self.tools["click_screen"].invoke({"snapshot_id": 1, "x": 0, "y": 0, "target": "링크"})
        self.tools["observe_screen"].invoke({})
        self.session.visual_snapshot["captured_at"] -= 61
        with self.assertRaises(ValueError):
            self.tools["click_screen"].invoke({"snapshot_id": 1, "x": 0, "y": 0, "target": "링크"})
        self.device.tap.assert_not_called()

    def test_foreground_change_rejects_visual_click(self):
        self.tools["observe_screen"].invoke({})
        self.device.current_app.return_value = {"package": "other.app", "activity": "other.app/.Main"}
        with self.assertRaises(ValueError):
            self.tools["click_screen"].invoke({"snapshot_id": 1, "x": 0, "y": 0, "target": "링크"})
        self.device.tap.assert_not_called()

    def test_capture_fallback_when_ui_dump_fails(self):
        self.device.read_screen = Mock(side_effect=DeviceError("UI 오류"))
        result = self.tools["observe_screen"].invoke({})
        self.assertEqual(result["elements"], [])
        self.assertIn("ui_error", result)
        self.assertIsNotNone(self.session.visual_snapshot)

    def test_uninstalled_browser_does_not_launch(self):
        with self.assertRaises(ValueError):
            self.tools["search_web"].invoke({"query": "검색", "browser_package": "fake.browser"})
        self.device.search_web.assert_not_called()

    def test_responses_text_is_printed_without_raw_block_metadata(self):
        from langchain_core.messages import AIMessage
        output = io.StringIO()
        with redirect_stdout(output):
            show_agent_result({"messages": [AIMessage(content=[{"type": "text", "text": "검색 중"}])]})
        self.assertIn("검색 중", output.getvalue())
        self.assertNotIn("'type'", output.getvalue())


class BrowserDeviceTests(unittest.TestCase):
    def test_unicode_query_is_url_encoded_and_browser_is_explicit(self):
        device = AndroidDevice(Path("adb"))
        with patch.object(device, "text", return_value="Status: ok") as run:
            result = device.search_web("메이플 인벤 & test", "com.sec.android.app.sbrowser")
        args = run.call_args.args
        self.assertIn("android.intent.action.VIEW", args)
        self.assertEqual(args[-2:], ("-p", "com.sec.android.app.sbrowser"))
        self.assertIn("%26", result["url"])
        self.assertNotIn("메이플", result["url"])

    def test_capture_png_dimensions_and_binary_transport(self):
        device = AndroidDevice(Path("adb"))
        app = {"package": PACKAGE, "activity": PACKAGE + "/.Main"}
        with patch.object(device, "current_app", return_value=app), patch.object(device, "_run", return_value=PNG) as run:
            result = device.capture_screen()
        run.assert_called_once_with(["exec-out", "screencap", "-p"])
        self.assertEqual((result["width"], result["height"]), (1, 1))
        self.assertTrue(result["image_url"].startswith("data:image/png;base64,"))
