from __future__ import annotations

import unittest

from unittest import mock

from dwhisper.postprocess import (
    MLXLMPostProcessor,
    OpenAICompatPostProcessor,
    PostProcessOptions,
    build_postprocessor,
    default_max_tokens_for_mode,
    clear_mlx_model_cache,
)
from dwhisper.transcriber import TranscribeResult


class PostProcessTests(unittest.TestCase):
    def test_apply_updates_result_and_preserves_raw_text(self) -> None:
        processor = OpenAICompatPostProcessor(
            PostProcessOptions(
                enabled=True,
                model="local-mm-model",
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
        self.assertEqual(result.postprocess["model"], "local-mm-model")

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
                model="local-mm-model",
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
        self.assertEqual(payload["model"], "local-mm-model")
        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertIn("summarize transcripts", payload["messages"][0]["content"])


class BackendResolutionTests(unittest.TestCase):
    def test_postprocess_mode_defaults_have_distinct_token_budgets(self) -> None:
        self.assertEqual(default_max_tokens_for_mode("clean"), 256)
        self.assertEqual(default_max_tokens_for_mode("summary"), 768)
        self.assertEqual(default_max_tokens_for_mode("speaker-format"), 1024)
        self.assertEqual(default_max_tokens_for_mode("meeting-notes"), 2048)

    def test_auto_prefers_http_when_base_url_is_set(self) -> None:
        options = PostProcessOptions(enabled=True, model="m", base_url="http://x/v1")
        self.assertEqual(options.resolved_backend(), "http")
        self.assertTrue(options.is_configured())

    def test_auto_falls_back_to_mlx_without_base_url(self) -> None:
        options = PostProcessOptions(enabled=True, model="qwen3-mlx")
        self.assertEqual(options.resolved_backend(), "mlx")
        self.assertTrue(options.is_configured())

    def test_explicit_mlx_backend_does_not_require_base_url(self) -> None:
        options = PostProcessOptions(enabled=True, model="qwen3-mlx", backend="mlx")
        self.assertTrue(options.is_configured())

    def test_build_postprocessor_routes_to_requested_backend(self) -> None:
        captured: dict[str, PostProcessOptions] = {}

        def http_factory(options: PostProcessOptions):
            captured["http"] = options
            return "http-processor"

        def mlx_factory(options: PostProcessOptions):
            captured["mlx"] = options
            return "mlx-processor"

        http_options = PostProcessOptions(
            enabled=True, model="m", base_url="http://x/v1", backend="http"
        )
        mlx_options = PostProcessOptions(enabled=True, model="m", backend="mlx")

        self.assertEqual(
            build_postprocessor(http_options, http_factory=http_factory, mlx_factory=mlx_factory),
            "http-processor",
        )
        self.assertEqual(
            build_postprocessor(mlx_options, http_factory=http_factory, mlx_factory=mlx_factory),
            "mlx-processor",
        )

    def test_invalid_backend_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            PostProcessOptions(enabled=True, model="m", backend="gibberish")


class MLXBackendTests(unittest.TestCase):
    def test_process_text_uses_injected_generator(self) -> None:
        captured: dict[str, object] = {}

        def fake_generate(prompt: str, model_name: str, max_tokens: int, timeout: float) -> str:
            captured["prompt"] = prompt
            captured["model"] = model_name
            captured["max_tokens"] = max_tokens
            captured["timeout"] = timeout
            return "Cleaned up transcript."

        processor = MLXLMPostProcessor(
            PostProcessOptions(
                enabled=True,
                model="qwen3-mlx",
                backend="mlx",
                max_tokens=256,
                timeout=15.0,
            ),
            generator=fake_generate,
        )

        text = processor.process_text(transcript="helo world", language="en")

        self.assertEqual(text, "Cleaned up transcript.")
        self.assertEqual(captured["model"], "qwen3-mlx")
        self.assertEqual(captured["max_tokens"], 256)
        self.assertEqual(captured["timeout"], 15.0)
        prompt = captured["prompt"]
        assert isinstance(prompt, str)
        self.assertIn("helo world", prompt)
        self.assertIn("clean speech-to-text transcripts", prompt)

    def test_mlx_model_is_loaded_once_across_processors(self) -> None:
        """Realtime listen runs many back-to-back chunks. The MLX LM must
        stay resident in the module-level cache instead of reloading per chunk."""
        clear_mlx_model_cache()
        try:
            fake_mlx_load = mock.Mock(return_value=("model", "tokenizer"))
            with mock.patch.dict(
                "sys.modules",
                {"mlx_lm": mock.MagicMock(load=fake_mlx_load)},
            ):
                first = MLXLMPostProcessor(
                    PostProcessOptions(enabled=True, model="qwen3-mlx", backend="mlx")
                )
                second = MLXLMPostProcessor(
                    PostProcessOptions(enabled=True, model="qwen3-mlx", backend="mlx")
                )
                first._ensure_loaded()
                second._ensure_loaded()
        finally:
            clear_mlx_model_cache()

        self.assertEqual(fake_mlx_load.call_count, 1)

    def test_apply_records_mlx_backend_metadata(self) -> None:
        processor = MLXLMPostProcessor(
            PostProcessOptions(enabled=True, model="qwen3-mlx", backend="mlx"),
            generator=lambda prompt, model, max_tokens, timeout: "Hello, world.",
        )
        result = TranscribeResult(text="helo world", language="en")

        processor.apply(result)

        self.assertEqual(result.raw_text, "helo world")
        self.assertEqual(result.text, "Hello, world.")
        self.assertEqual(result.postprocess["backend"], "mlx")
        self.assertTrue(result.postprocess["applied"])
        self.assertNotIn("base_url", result.postprocess)


class StreamingTests(unittest.TestCase):
    def test_http_stream_text_falls_back_to_process_text_chunks(self) -> None:
        """When a test injects a sync requester we still exercise SSE semantics."""

        def fake_request(endpoint, headers, payload, timeout):
            return {"choices": [{"message": {"content": "Hello world, friend."}}]}

        processor = OpenAICompatPostProcessor(
            PostProcessOptions(
                enabled=True,
                model="local-mm-model",
                base_url="http://127.0.0.1:11435/v1",
            ),
            requester=fake_request,
        )

        chunks = list(processor.stream_text(transcript="helo", language="en"))

        self.assertGreater(len(chunks), 1)
        self.assertEqual("".join(chunks).strip(), "Hello world, friend.")

    def test_mlx_stream_text_uses_injected_stream_generator(self) -> None:
        captured: dict[str, object] = {}

        def fake_stream(prompt, model_name, max_tokens, timeout):
            captured["prompt"] = prompt
            captured["model"] = model_name
            yield "Hello"
            yield ", "
            yield "world."

        processor = MLXLMPostProcessor(
            PostProcessOptions(enabled=True, model="qwen3-mlx", backend="mlx"),
            stream_generator=fake_stream,
        )

        chunks = list(processor.stream_text(transcript="helo world", language="en"))

        self.assertEqual(chunks, ["Hello", ", ", "world."])
        self.assertEqual(captured["model"], "qwen3-mlx")
        assert isinstance(captured["prompt"], str)
        self.assertIn("helo world", captured["prompt"])

    def test_mlx_stream_text_raises_when_no_content(self) -> None:
        processor = MLXLMPostProcessor(
            PostProcessOptions(enabled=True, model="qwen3-mlx", backend="mlx"),
            stream_generator=lambda *_: iter(()),
        )

        with self.assertRaises(RuntimeError):
            list(processor.stream_text(transcript="helo", language="en"))


if __name__ == "__main__":
    unittest.main()
