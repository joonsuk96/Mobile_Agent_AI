"""Unicode input regression tests; no phone or paid API calls."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import unittest
from unittest.mock import Mock, patch
from xml.sax.saxutils import escape

from mobile_jarvis_app.agent import PhoneSession, build_agent, build_tools
from mobile_jarvis_app.device import AndroidDevice, DeviceError, parse_screen
from test_agent import FakeCatalog, FakeDevice, ScriptedModel, call


def input_screen(value="", password=False):
    xml = f"""<hierarchy>
      <node package="org.example.app" class="android.widget.LinearLayout"
            clickable="true" bounds="[0,0][400,400]">
        <node package="org.example.app" class="android.widget.EditText"
              resource-id="org.example.app:id/search" text="{escape(value, {'"': '&quot;'})}"
              enabled="true" focused="true" password="{str(password).lower()}"
              bounds="[10,20][300,80]"/>
      </node></hierarchy>"""
    return parse_screen(xml.encode(), {"package": "org.example.app", "activity": "org.example.app/.Main"})


class FakeField:
    def __init__(self, text=""):
        self.value = text
        self.info = {"bounds": {"left": 10, "top": 20, "right": 300, "bottom": 80},
                     "enabled": True, "password": False}
        self.set_values = []
    def get_text(self, **kwargs):
        return self.value
    def set_text(self, value, **kwargs):
        self.set_values.append(value)
        self.value = value
        return True


class FakeMatches:
    def __init__(self, fields):
        self.fields = fields
    @property
    def count(self):
        return len(self.fields)
    def __getitem__(self, index):
        return self.fields[index]


class FakeUi:
    def __init__(self, fields):
        self.fields = fields
        self.selectors = []
    def __call__(self, **kwargs):
        self.selectors.append(kwargs)
        return FakeMatches(self.fields)


class UnicodeInputTests(unittest.TestCase):
    def device_and_field(self, existing=""):
        field = FakeField(existing)
        device = AndroidDevice(Path("adb"))
        device._ui_device = FakeUi([field])
        element = input_screen(existing)["elements"][0]
        return device, field, element

    def test_blank_editable_is_visible_and_uses_own_bounds(self):
        element = input_screen()["elements"][0]
        self.assertTrue(element["editable"])
        self.assertTrue(element["focused"])
        self.assertEqual(element["bounds"], "[10,20][300,80]")
        self.assertEqual(element["x"], 155)

    def test_password_is_masked_and_rejected(self):
        parsed = input_screen("secret-value", password=True)
        self.assertNotIn("secret-value", str(parsed))
        device = AndroidDevice(Path("adb"))
        with self.assertRaisesRegex(DeviceError, "비밀번호"):
            device.set_text(parsed["elements"][0], "anything")

    def test_korean_english_symbols_emoji_and_multiline_are_preserved(self):
        values = ["수원", "Sejong Station", "수원 Suwon 123 !@#$%&'();", "한글😀\nsecond line"]
        for value in values:
            with self.subTest(value=value):
                device, field, element = self.device_and_field()
                with patch.object(device, "_run") as adb:
                    result = device.set_text(element, value)
                self.assertEqual(field.set_values, [value])
                self.assertTrue(result["verified"])
                adb.assert_not_called()

    def test_replace_append_and_clear(self):
        device, field, element = self.device_and_field("old")
        self.assertTrue(device.set_text(element, "수원")["verified"])
        self.assertEqual(field.value, "수원")
        self.assertTrue(device.set_text(element, " 터미널", clear=False)["verified"])
        self.assertEqual(field.value, "수원 터미널")
        self.assertTrue(device.set_text(element, "")["verified"])
        self.assertEqual(field.value, "")

    def test_same_resource_id_uses_matching_bounds_not_first_field(self):
        device, field, element = self.device_and_field()
        wrong = FakeField()
        wrong.info["bounds"] = {"left": 0, "top": 100, "right": 300, "bottom": 150}
        device._ui_device = FakeUi([wrong, field])
        device.set_text(element, "수원")
        self.assertEqual(wrong.set_values, [])
        self.assertEqual(field.value, "수원")

    def test_moved_field_does_not_receive_text(self):
        device, field, element = self.device_and_field()
        field.info["bounds"]["top"] = 100
        with self.assertRaisesRegex(DeviceError, "이동"):
            device.set_text(element, "수원")
        self.assertEqual(field.set_values, [])

    def test_readonly_element_and_invalid_control_characters_are_rejected(self):
        device, field, element = self.device_and_field()
        with self.assertRaises(DeviceError):
            device.set_text({**element, "editable": False}, "수원")
        with self.assertRaises(DeviceError):
            device.set_text(element, "bad\x00text")
        self.assertEqual(field.set_values, [])

    def test_failed_readback_is_reported_unverified_not_retyped(self):
        device, field, element = self.device_and_field()
        field.get_text = Mock(side_effect=RuntimeError("field disappeared"))
        result = device.set_text(element, "수원")
        self.assertIsNone(result["verified"])
        self.assertEqual(field.set_values, ["수원"])

    def test_screen_reads_use_the_same_rpc_backend(self):
        device = AndroidDevice(Path("adb"))
        backend = Mock()
        backend.dump_hierarchy.return_value = '<hierarchy/>'
        device._ui_device = backend
        current = {"package": "org.example.app", "activity": "org.example.app/.Main"}
        with patch.object(device, "current_app", return_value=current), \
             patch.object(device, "_run") as adb:
            result = device.read_screen()
        backend.dump_hierarchy.assert_called_once()
        adb.assert_not_called()
        self.assertEqual(result["package"], "org.example.app")


class TextInputToolTests(unittest.TestCase):
    def setUp(self):
        self.device = FakeDevice()
        self.device.screen = input_screen()
        def set_value(element, value, clear=True):
            old = self.device.screen["elements"][0]["text"]
            self.device.screen = input_screen(value if clear else old + value)
            return {"verified": True, "characters": len(value)}
        self.device.set_text = Mock(side_effect=set_value)
        self.session = PhoneSession(self.device, FakeCatalog(), require_plan=False)
        self.tools = {tool.name: tool for tool in build_tools(self.session)}
        delay = patch("mobile_jarvis_app.agent.time.sleep")
        delay.start()
        self.addCleanup(delay.stop)

    def test_tool_inputs_and_checks_new_screen_and_counts_one_action(self):
        observation = self.session.observe()
        with redirect_stdout(io.StringIO()):
            result = self.tools["type_text"].invoke({
                "snapshot_id": observation["snapshot_id"], "element_id": "0", "text": "수원"})
        self.assertTrue(result["result"]["verified"])
        self.assertEqual(result["screen"]["elements"][0]["text"], "수원")
        self.assertEqual(self.session.actions, 1)

    def test_stale_screen_is_not_typed_into(self):
        self.session.observe()
        self.device.screen = input_screen("다른 내용")
        result = self.tools["type_text"].invoke({
            "snapshot_id": 1, "element_id": "0", "text": "수원"})
        self.assertIn("error", result)
        self.device.set_text.assert_not_called()

    def test_old_snapshot_id_is_rejected(self):
        self.session.observe()
        with self.assertRaises(ValueError):
            self.tools["type_text"].invoke({
                "snapshot_id": 0, "element_id": "0", "text": "수원"})
        self.device.set_text.assert_not_called()

    def test_langchain_observe_type_verify_and_finish(self):
        model = ScriptedModel(responses=[
            call("observe_screen", {}, 1),
            call("type_text", {"snapshot_id": 1, "element_id": "0", "text": "수원 Suwon"}, 2),
            call("finish_task", {"summary": "요청한 글자를 입력했어."}, 3),
        ])
        with redirect_stdout(io.StringIO()):
            build_agent(self.session, model).invoke({
                "messages": [{"role": "user", "content": "검색창에 수원 Suwon 입력해줘"}]},
                config={"recursion_limit": 100})
        self.assertTrue(self.session.completed)
        self.assertEqual(self.device.screen["elements"][0]["text"], "수원 Suwon")
        self.assertEqual(self.device.set_text.call_count, 1)


if __name__ == "__main__":
    unittest.main()
