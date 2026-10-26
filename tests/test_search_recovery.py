"""Search recovery tools and evidence of actual alternative actions."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from mobile_jarvis_app.agent import PhoneSession, build_tools, run_interactive
from mobile_jarvis_app.device import AndroidDevice, DeviceError
from test_agent import FakeDevice, FakeCatalog, ScriptedModel, call, screen
from test_text_input import input_screen, FakeField, FakeMatches


class SearchRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.device = FakeDevice()
        self.device.screen = input_screen()
        self.device.current_app = Mock(return_value={"package": "org.example.app", "activity": "org.example.app/.Main"})
        self.device.set_focused_text = Mock(side_effect=self.enter_query)
        self.device.swipe_region = Mock(side_effect=lambda *args: setattr(self.device, "screen", input_screen("다른 목록")))
        self.session = PhoneSession(self.device, FakeCatalog(), require_plan=False)
        self.tools = {t.name: t for t in build_tools(self.session)}
        delay = patch("mobile_jarvis_app.agent.time.sleep")
        delay.start()
        self.addCleanup(delay.stop)
    def enter_query(self, text, clear=True):
        self.device.screen = input_screen(text)
        return {"verified": True}

    def test_observation_only_cannot_give_up_as_route_failure(self):
        for _ in range(3):
            self.session.observe()
        result = self.tools["finish_incomplete"].invoke({"reason": "대상이 안 보여"})
        self.assertEqual(result["status"], "not_verified")
        self.assertIsNone(self.session.terminal_reason)
        self.assertIn("관찰만 반복", self.session.public_screen()["next_step_hint"])

    def test_missing_list_item_is_entered_in_focused_search_then_selected(self):
        self.device.tap.side_effect = lambda *args: setattr(self.device, "screen", screen("org.example.app", "테스트 종목 상세 정보"))
        model = ScriptedModel(responses=[
            call("observe_screen", {}, 1),
            call("type_focused_text", {"snapshot_id": 1, "text": "테스트 종목"}, 2),
            call("click_element", {"snapshot_id": 2, "element_id": "0"}, 3),
            call("finish_task", {"summary": "종목 상세를 확인했어"}, 4),
        ])
        with patch("builtins.input", side_effect=AssertionError("must not ask")), redirect_stdout(io.StringIO()):
            run_interactive(self.session, "테스트 종목 상세 확인", model=model)
        self.device.set_focused_text.assert_called_once_with("테스트 종목", clear=True)
        self.device.tap.assert_called_once()
        self.assertEqual(self.session.attempted_strategies, {"type", "click"})
        self.assertTrue(self.session.completed)

    def test_panel_swipe_and_stale_snapshot_guard(self):
        self.session.observe()
        with redirect_stdout(io.StringIO()):
            self.tools["swipe_region"].invoke({"snapshot_id": 1, "x1": 200, "y1": 300, "x2": 200, "y2": 100})
        self.device.swipe_region.assert_called_once_with(200, 300, 200, 100, 400)
        with self.assertRaises(ValueError):
            self.tools["type_focused_text"].invoke({"snapshot_id": 1, "text": "오래된 화면"})
        self.device.set_focused_text.assert_not_called()
        self.assertEqual(self.session.actions, 1)

    def test_failed_action_is_recorded_without_private_error_contents(self):
        self.session.observe()
        self.device.set_focused_text.side_effect = DeviceError("private-sentinel")
        with redirect_stdout(io.StringIO()), self.assertRaises(DeviceError):
            self.tools["type_focused_text"].invoke({"snapshot_id": 1, "text": "검색어"})
        self.assertEqual(self.session.action_history[-1]["status"], "error")
        self.assertNotIn("private-sentinel", str(self.session.action_history))
        self.assertNotIn("검색어", str(self.session.action_history))


class FocusAndSwipeDeviceTests(unittest.TestCase):
    def test_focused_editor_uses_existing_unicode_input(self):
        device = AndroidDevice(Path("adb"))
        field = FakeField()
        field.info.update({"className": "android.widget.EditText", "packageName": "org.example.app", "resourceName": "search"})
        ui = Mock(return_value=FakeMatches([field]))
        device._ui_device = ui
        with patch.object(device, "set_text", return_value={"verified": True}) as setter:
            result = device.set_focused_text("삼성전자")
        ui.assert_called_once_with(focused=True)
        self.assertEqual(setter.call_args.args[1], "삼성전자")
        self.assertTrue(setter.call_args.args[0]["editable"])
        self.assertTrue(result["verified"])

    def test_password_and_non_editor_are_rejected_before_input(self):
        for extra in [{"className": "android.widget.EditText", "password": True}, {"className": "android.widget.Button"}]:
            device = AndroidDevice(Path("adb"))
            field = FakeField()
            field.info.update(extra)
            device._ui_device = Mock(return_value=FakeMatches([field]))
            with patch.object(device, "set_text") as setter, self.assertRaises(DeviceError):
                device.set_focused_text("문자")
            setter.assert_not_called()

    def test_region_swipe_validates_bounds(self):
        device = AndroidDevice(Path("adb"))
        with patch.object(device, "text", return_value="Physical size: 720x1600") as adb:
            device.swipe_region(500, 1200, 500, 500)
            self.assertEqual(adb.call_args.args, ("shell", "input", "swipe", "500", "1200", "500", "500", "400"))
            adb.reset_mock()
            with self.assertRaises(DeviceError):
                device.swipe_region(900, 1200, 500, 500)
            self.assertEqual(adb.call_count, 1)
