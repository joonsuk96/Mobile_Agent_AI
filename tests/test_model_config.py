"""Offline model migration checks; no network or real API key is used."""
import os
import unittest
from unittest.mock import patch

from langchain_core.messages import HumanMessage
from mobile_jarvis_app.agent import build_model


class ModelConfigurationTests(unittest.TestCase):
    def test_default_is_luna_with_responses_and_low_reasoning(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "offline-test-key"}, clear=True):
            model = build_model()
        self.assertEqual(model.model_name, "gpt-6-luna")
        self.assertTrue(model.use_responses_api)
        self.assertEqual(model.reasoning, {"effort": "low"})
        self.assertEqual(model.max_tokens, 4096)
        self.assertIsNone(model.temperature)

    def test_serialized_payload_uses_supported_parameters_and_tools(self):
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "offline-test-key", "JARVIS_MODEL": "gpt-6-luna",
        }, clear=True):
            model = build_model()
        payload = model._get_request_payload(
            [HumanMessage(content="설치 앱 검색")],
            tools=[{
                "type": "function",
                "function": {
                    "name": "search_apps", "description": "Search installed apps",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            }],
            parallel_tool_calls=False,
        )
        self.assertEqual(payload["model"], "gpt-6-luna")
        self.assertEqual(payload["reasoning"]["effort"], "low")
        self.assertEqual(payload["max_output_tokens"], 4096)
        self.assertNotIn("temperature", payload)
        self.assertNotIn("max_tokens", payload)
        self.assertFalse(payload["parallel_tool_calls"])
        self.assertEqual(payload["tools"][0]["name"], "search_apps")

    def test_blank_override_uses_default(self):
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "offline-test-key", "JARVIS_MODEL": "  ",
        }, clear=True):
            self.assertEqual(build_model().model_name, "gpt-6-luna")

    def test_explicit_legacy_override_remains_compatible(self):
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "offline-test-key", "JARVIS_MODEL": "gpt-4.1-mini",
        }, clear=True):
            model = build_model()
        self.assertEqual(model.model_name, "gpt-4.1-mini")
        self.assertEqual(model.temperature, 0)
        self.assertEqual(model.max_tokens, 1200)
        self.assertIsNone(model.reasoning)


if __name__ == "__main__":
    unittest.main()
