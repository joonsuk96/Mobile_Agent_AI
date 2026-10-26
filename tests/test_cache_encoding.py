"""Regression tests for cache reuse and Windows text decoding."""
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from mobile_jarvis_app.apps import AppCatalog, load_alias_groups
from mobile_jarvis_app.device import AndroidDevice
from mobile_jarvis_app.textio import decode_text, read_local_text


class CacheReuseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.device = Mock()
        self.device.serial = "wireless-address"
        self.device.cache_identity.return_value = "hardware-serial"
        self.device.launcher_components.return_value = {
            "org.example.app": "org.example.app/.Main"}
        self.device.text.side_effect = lambda *args: "user0" if args[1] == "am" else "build1"
        self.device.apk_paths.return_value = ["/data/app/old/base.apk"]

    def build(self, refresh=False):
        catalog = AppCatalog(self.device, self.root)
        with redirect_stdout(io.StringIO()):
            catalog.build(refresh)
        return catalog

    def test_transient_windows_cache_lock_retries_atomic_replace(self):
        original_replace = Path.replace
        attempts = []
        def replace(source, target):
            attempts.append((source, target))
            if len(attempts) <= 2:
                raise PermissionError("transient lock")
            return original_replace(source, target)
        with patch.object(Path, "replace", autospec=True, side_effect=replace), \
             patch("mobile_jarvis_app.apps.time.sleep") as sleep:
            result = self.build()
        self.assertGreaterEqual(len(attempts), 3)
        self.assertEqual(sleep.call_count, 2)
        self.assertTrue(result.ready)
        self.assertTrue(list(self.root.glob("apps-*.json")))

    def test_permanent_cache_permission_failure_is_not_hidden(self):
        with patch.object(Path, "replace", side_effect=PermissionError("permanent")) as replace, \
             patch("mobile_jarvis_app.apps.time.sleep"), self.assertRaises(PermissionError):
            self.build()
        self.assertEqual(replace.call_count, 3)

    def test_unicode_apk_failure_is_cached_with_original_path(self):
        error = UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")
        with patch("mobile_jarvis_app.apps.read_apk_labels", side_effect=error) as labels:
            first = self.build()
            self.assertEqual(first.apps["org.example.app"]["paths"],
                             ["/data/app/old/base.apk"])
            self.assertEqual(first.apps["org.example.app"]["label_error"], "UnicodeDecodeError")
            self.build()
            labels.assert_called_once()
            self.device.apk_paths.assert_called_once()
            self.device.pull.assert_called_once()

    def test_cached_entries_skip_apk_path_queries(self):
        with patch("mobile_jarvis_app.apps.read_apk_labels", return_value=["Example"]):
            self.build()
            self.device.apk_paths.reset_mock()
            self.device.pull.reset_mock()
            catalog = self.build()
        self.device.apk_paths.assert_not_called()
        self.device.pull.assert_not_called()
        self.assertEqual(catalog.apps["org.example.app"]["name"], "Example")

    def test_only_new_app_is_extracted_and_removed_apps_disappear(self):
        with patch("mobile_jarvis_app.apps.read_apk_labels", return_value=["Example"]) as labels:
            self.build()
            self.device.launcher_components.return_value = {"org.new.app": "org.new.app/.Main"}
            second = self.build()
            self.assertEqual(set(second.apps), {"org.new.app"})
            self.assertEqual(labels.call_count, 2)

    def test_refresh_explicitly_retries_failed_apk(self):
        error = UnicodeDecodeError("utf-8", b"\xff", 0, 1, "bad")
        with patch("mobile_jarvis_app.apps.read_apk_labels", side_effect=error):
            self.build()
        with patch("mobile_jarvis_app.apps.read_apk_labels", return_value=["Recovered"]) as labels:
            catalog = self.build(refresh=True)
            labels.assert_called_once()
        self.assertEqual(catalog.apps["org.example.app"]["name"], "Recovered")

    def test_wireless_address_change_keeps_cache(self):
        with patch("mobile_jarvis_app.apps.read_apk_labels", return_value=["Example"]) as labels:
            self.build()
            self.device.serial = "another-wireless-address"
            self.build()
            self.assertEqual(labels.call_count, 1)

    def test_legacy_failed_entry_is_reused_without_reextracting(self):
        legacy_key = hashlib.sha256(b"wireless-address|user0|build1").hexdigest()[:20]
        legacy = {"version": 1, "apps": {
            "org.example.app": {"name": "org.example.app", "package": "org.example.app",
                                "component": "org.example.app/.Main", "labels": [], "paths": []}}}
        (self.root / f"apps-{legacy_key}.json").write_text(
            json.dumps(legacy), encoding="utf-8")
        with patch("mobile_jarvis_app.apps.read_apk_labels") as labels:
            self.build()
        labels.assert_not_called()
        self.device.apk_paths.assert_not_called()
        self.device.pull.assert_not_called()

    def test_utf8_bom_cache_is_read_without_reindexing(self):
        key = hashlib.sha256(b"hardware-serial|user0").hexdigest()[:20]
        data = {"version": 2, "apps": {
            "org.example.app": {"name": "Example", "labels": ["Example"], "paths": []}}}
        (self.root / f"apps-{key}.json").write_text(json.dumps(data), encoding="utf-8-sig")
        self.build()
        self.device.apk_paths.assert_not_called()

    def test_os_build_change_keeps_names_until_explicit_refresh(self):
        with patch("mobile_jarvis_app.apps.read_apk_labels", return_value=["Example"]) as labels:
            self.build()
            self.device.text.side_effect = lambda *args: "user0" if args[1] == "am" else "build2"
            self.build()
        self.assertEqual(labels.call_count, 1)


class TextDecodingTests(unittest.TestCase):
    def test_utf8_bom_cp949_and_utf16_text(self):
        value = 'OPENAI_API_KEY=test-placeholder\n# 한글 설정'
        for encoding in ("utf-8", "utf-8-sig", "cp949", "utf-16"):
            with self.subTest(encoding=encoding):
                self.assertEqual(decode_text(value.encode(encoding)), value)

    def test_cp949_alias_file_is_supported(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            path = Path(directory) / "app_aliases.json"
            path.write_bytes('{"버스타고":["bustago"]}'.encode("cp949"))
            self.assertEqual(load_alias_groups(path), [["버스타고", "bustago"]])

    def test_invalid_local_encoding_returns_actionable_error(self):
        with self.assertRaisesRegex(ValueError, "UTF-8"):
            decode_text(b"\xffbad")

    def test_adb_host_output_can_replace_invalid_bytes(self):
        self.assertIsInstance(decode_text(b"\xffdevice", replace=True), str)

    def test_wireless_fallback_identity_strips_changing_mdns_suffix(self):
        for serial in ("adb-R5CX41L9PHZ-aaaa._adb-tls-connect._tcp",
                       "adb-R5CX41L9PHZ-bbbb._adb-tls-connect._tcp"):
            device = AndroidDevice(Path("adb"), serial)
            with patch.object(device, "text", return_value=""):
                self.assertEqual(device.cache_identity(), "R5CX41L9PHZ")


if __name__ == "__main__":
    unittest.main()
