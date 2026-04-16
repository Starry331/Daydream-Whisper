from __future__ import annotations

import unittest

from dwhisper.postprocess import OpenAICompatPostProcessor, PostProcessOptions
from dwhisper.transcriber import TranscribeResult


class PostProcessTests(unittest.TestCase):
    def test_apply_updates_result_and_preserves_raw_text(self) -> None:
        processor = OpenAICompatPostProcessor(
            PostProcessOptions(
                enabled=True,
                model="qwen-local",
                base_url="http://127.0.0.1:11435/v1",
            ),
            requester=lambda endpoint, headers, payload, timeout: {
                "choices": [{"message": {"content": "Hello, world."}}]
            },
        )
        result = TranscribeResult(text="helo world", language="en")

        processor.apply(result)

        self.assertEqual(result.raw_text, "helo world")
        self.assertEqual(result.text, "Hello, world.")
        self.assertTrue(result.postprocess["applied"])
        self.assertEqual(result.postprocess["model"], "qwen-local")

    def test_summary_mode_uses_custom_endpoint_shape(self) -> None:
        captured: dict[str, object] = {}

        def fake_request(endpoint, headers, payload, timeout):
            captured["endpoint"] = endpoint
            captured["headers"] = headers
            captured["payload"] = payload
            captured["timeout"] = timeout
            return {"choices": [{"message": {"content": "Short summary"}}]}

        processor = OpenAICompatPostProcessor(
            PostProcessOptions(
                enabled=True,
                model="qwen-local",
                base_url="http://127.0.0.1:11435",
                mode="summary",
                timeout=12.0,
            ),
            requester=fake_request,
        )

        text = processor.process_text(transcript="Long transcript", language="en")

        self.assertEqual(text, "Short summary")
        self.assertEqual(captured["endpoint"], "http://127.0.0.1:11435/v1/chat/completions")
        self.assertEqual(captured["timeout"], 12.0)
        payload = captured["payload"]
        assert isinstance(payload, dict)
        self.assertEqual(payload["model"], "qwen-local")
        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertIn("summarize transcripts", payload["messages"][0]["content"])


if __name__ == "__main__":
    unittest.main()
