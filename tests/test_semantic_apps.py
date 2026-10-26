"""General cross-language discovery tests, without a live model or phone."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from mobile_jarvis_app.agent import PhoneSession, build_agent, build_tools
from mobile_jarvis_app.apps import AppCatalog
from test_agent import FakeDevice, ScriptedModel, call


class SemanticDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.alias_path = self.root / "app_aliases.json"
        self.alias_path.write_text("{}", encoding="utf-8")
        self.device = FakeDevice()
        self.catalog = AppCatalog(self.device, self.root / "cache", self.alias_path)
        self.catalog.ready = True
        self.catalog.apps = {
            "org.mozilla.firefox": {
                "package": "org.mozilla.firefox", "name": "Firefox", "labels": ["Firefox"],
                "component": "org.mozilla.firefox/.Main", "paths": []},
            "com.spotify.music": {
                "package": "com.spotify.music", "name": "Spotify", "labels": ["Spotify"],
                "component": "com.spotify.music/.Main", "paths": []},
        }
        self.session = PhoneSession(self.device, self.catalog, interactive=True, require_plan=False)
        self.tools = {t.name: t for t in build_tools(self.session)}
        delay = patch("mobile_jarvis_app.agent.time.sleep")
        delay.start()
        self.addCleanup(delay.stop)

    def test_unknown_korean_name_exposes_real_english_apps_for_semantic_reasoning(self):
        for query in ("파이어폭스", "스포티파이", "모질라 브라우저"):
            with self.subTest(query=query):
                result = self.catalog.search(query)
                self.assertEqual(result["resolution_mode"], "semantic_required")
                self.assertEqual(
                    {app["package"] for app in result["semantic_catalog"]["apps"]},
                    set(self.catalog.apps))
                self.assertIn("의미", result["semantic_instruction"])

    def test_partial_english_match_also_gets_semantic_context(self):
        result = self.catalog.search("Spot")
        self.assertEqual(result["resolution_mode"], "semantic_required")
        self.assertIsNotNone(result["semantic_catalog"])

    def test_exact_name_does_not_send_whole_catalog(self):
        result = self.catalog.search("Firefox")
        self.assertEqual(result["resolution_mode"], "local")
        self.assertIsNone(result["semantic_catalog"])

    def test_confirmed_match_persists_and_skips_semantic_fallback(self):
        with patch("builtins.input", return_value="y"), redirect_stdout(io.StringIO()):
            result = self.tools["confirm_app_choice"].invoke(
                {"query": "파이어폭스", "package": "org.mozilla.firefox"})
        self.assertTrue(result["saved"])
        self.assertTrue(result["confirmed"])
        again = AppCatalog(self.device, self.root / "cache", self.alias_path)
        again.ready, again.apps = True, self.catalog.apps
        found = again.search("파이어폭스")
        self.assertEqual(found["resolution_mode"], "local")
        self.assertEqual(found["apps"][0]["match_type"], "user_confirmed")
        self.assertEqual(found["apps"][0]["package"], "org.mozilla.firefox")

    def test_rejection_is_not_saved_and_blocks_launch(self):
        with patch("builtins.input", return_value="n"), redirect_stdout(io.StringIO()):
            result = self.tools["confirm_app_choice"].invoke(
                {"query": "음악앱", "package": "com.spotify.music"})
        self.assertFalse(result["saved"])
        self.assertFalse(self.catalog.confirmed_names_path.exists())
        with self.assertRaisesRegex(ValueError, "거부"):
            self.tools["launch_app"].invoke({"package": "com.spotify.music"})
        self.assertEqual(self.device.launch_calls, [])

    def test_unrecognized_reply_does_not_save_a_guess(self):
        with patch("builtins.input", return_value="잘 모르겠어"), redirect_stdout(io.StringIO()):
            result = self.tools["confirm_app_choice"].invoke(
                {"query": "음악앱", "package": "com.spotify.music"})
        self.assertFalse(result["confirmed"])
        self.assertFalse(self.catalog.confirmed_names_path.exists())

    def test_uninstalled_package_cannot_be_confirmed(self):
        with patch("builtins.input") as prompt:
            with self.assertRaises(ValueError):
                self.tools["confirm_app_choice"].invoke(
                    {"query": "아무앱", "package": "com.invented.app"})
        prompt.assert_not_called()

    def test_removing_app_invalidates_learned_name(self):
        self.catalog.remember_confirmed_name("파이어폭스", "org.mozilla.firefox")
        del self.catalog.apps["org.mozilla.firefox"]
        found = self.catalog.search("파이어폭스")
        self.assertEqual(found["resolution_mode"], "semantic_required")
        self.assertNotIn("org.mozilla.firefox",
                         [app["package"] for app in found["semantic_catalog"]["apps"]])

    def test_semantic_catalog_is_paginated_without_losing_candidates(self):
        for number in range(60):
            package = f"org.sample.app{number}"
            self.catalog.apps[package] = {
                "package": package, "name": f"Sample {number}", "labels": [],
                "component": package + "/.Main", "paths": []}
        first = self.catalog.search("모르는앱")
        cursor = first["semantic_catalog"]["next_offset"]
        second = self.catalog.search("모르는앱", catalog_offset=cursor)
        packages = [a["package"] for a in first["semantic_catalog"]["apps"]]
        packages += [a["package"] for a in second["semantic_catalog"]["apps"]]
        self.assertEqual(len(packages), 62)
        self.assertEqual(len(set(packages)), 62)

    def test_langchain_general_semantic_path_and_user_confirmation(self):
        self.device.launcher_components = lambda: {
            package: app["component"] for package, app in self.catalog.apps.items()}
        model = ScriptedModel(responses=[
            call("search_apps", {"query": "파이어폭스"}, 1),
            call("confirm_app_choice",
                 {"query": "파이어폭스", "package": "org.mozilla.firefox"}, 2),
            call("launch_app", {"package": "org.mozilla.firefox"}, 3),
            call("finish_task", {"summary": "파이어폭스를 열었어."}, 4),
        ])
        with patch("builtins.input", return_value="y"), redirect_stdout(io.StringIO()):
            result = build_agent(self.session, model).invoke(
                {"messages": [{"role": "user", "content": "파이어폭스 열어줘"}]},
                config={"recursion_limit": 200})
        self.assertTrue(self.session.completed)
        self.assertEqual(self.device.launch_calls[0][0], "org.mozilla.firefox")
        self.assertIn("completed", result["messages"][-1].content)
        self.assertEqual(self.catalog.search("파이어폭스")["resolution_mode"], "local")


if __name__ == "__main__":
    unittest.main()
