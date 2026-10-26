"""Realistic YouTube/Music metadata and nested Android Activity shell quoting."""
import json
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from mobile_jarvis_app.apps import AppCatalog
from mobile_jarvis_app.device import AndroidDevice

YOUTUBE = "com.google.android.youtube"
MUSIC = "com.google.android.apps.youtube.music"
COMPONENT = YOUTUBE + "/.app.honeycomb.Shell$HomeActivity"


class YouTubeDiscoveryTests(unittest.TestCase):
    def test_korean_short_and_english_names_prefer_youtube_over_music(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            aliases = root / "app_aliases.json"
            aliases.write_text(json.dumps({"유튜브": ["YouTube", "유튭"]}, ensure_ascii=False), encoding="utf-8")
            catalog = AppCatalog(None, root / "cache", aliases)
            catalog.ready = True
            catalog.apps = {
                YOUTUBE: {"package": YOUTUBE, "component": COMPONENT, "name": "YouTube", "labels": ["YouTube"]},
                MUSIC: {"package": MUSIC, "component": MUSIC+"/.MusicActivity", "name": "YouTube Music", "labels": ["YouTube Music"]},
            }
            for query in ["유튜브", "유튭", "YouTube"]:
                with self.subTest(query=query):
                    result = catalog.search(query)
                    self.assertEqual(result["resolution_mode"], "local")
                    self.assertEqual(result["apps"][0]["package"], YOUTUBE)
                    strong = [app["package"] for app in result["apps"] if app["match_type"] in {"exact", "alias_exact", "user_confirmed"}]
                    self.assertEqual(strong, [YOUTUBE])
                    self.assertEqual(catalog.get(YOUTUBE)["component"], COMPONENT)
            result = catalog.search("YouTube Music")
            self.assertEqual(result["apps"][0]["package"], MUSIC)

    def test_nested_activity_is_passed_as_a_quoted_remote_shell_argument(self):
        device = AndroidDevice(Path("adb"), "phone")
        app = {"package": YOUTUBE, "activity": COMPONENT}
        result = subprocess.CompletedProcess([], 0, stdout=b"Status: ok", stderr=b"")
        with patch("mobile_jarvis_app.device.subprocess.run", return_value=result) as run, \
             patch.object(device, "current_app", return_value=app):
            launched = device.launch(YOUTUBE, COMPONENT)
        command = run.call_args.args[0]
        self.assertEqual(command[:4], ["adb", "-s", "phone", "shell"])
        self.assertEqual(len(command), 5)
        self.assertIn("'" + COMPONENT + "'", command[-1])
        self.assertEqual(shlex.split(command[-1])[-2:], ["-n", COMPONENT])
        self.assertTrue(launched["foreground_matches"])
        self.assertFalse(run.call_args.kwargs.get("shell", False))

    def test_launcher_discovery_preserves_dollar_in_activity_name(self):
        device = AndroidDevice(Path("adb"))
        with patch.object(device, "text", return_value=COMPONENT):
            apps = device.launcher_components()
        self.assertEqual(apps[YOUTUBE], COMPONENT)

    def test_non_shell_adb_commands_keep_original_argument_boundaries(self):
        device = AndroidDevice(Path("adb"))
        result = subprocess.CompletedProcess([], 0, stdout=b"PNG bytes", stderr=b"")
        with patch("mobile_jarvis_app.device.subprocess.run", return_value=result) as run:
            output = device._run(["exec-out", "screencap", "-p"])
        self.assertEqual(run.call_args.args[0], ["adb", "exec-out", "screencap", "-p"])
        self.assertEqual(output, b"PNG bytes")
