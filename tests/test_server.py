from __future__ import annotations

import dataclasses
import tempfile
import threading
import unittest
from unittest import mock

from dwhisper.server import (
    SpeechAPIConfig,
    SpeechAPIState,
    _preload_state,
    build_postprocess_response,
    build_transcription_response,
    parse_postprocess_api_request,
    parse_speech_api_request,
)
from dwhisper.transcriber import TranscribeResult


def build_multipart(fields: dict[str, str], *, filename: str = "sample.wav", content: bytes = b"RIFFdata") -> tuple[str, bytes]:
    boundary = "----daydream-boundary"
    parts: list[bytes] = []
    for key, value in fields.items():
        parts.append(
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{key}"\r\n\r\n'
                f"{value}\r\n"
            ).encode("utf-8")
        )
    parts.append(
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            "Content-Type: audio/wav\r\n\r\n"
        ).encode("utf-8")
        + content
        + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode("utf-8"))
    body = b"".join(parts)
    return f"multipart/form-data; boundary={boundary}", body


class FakeTranscriber:
    last_task: str | None = None
    last_options = None
    warmup_calls = 0
    close_calls = 0

    def __init__(self, model: str, **kwargs) -> None:
        self.model = model

    def warmup(self) -> None:
        FakeTranscriber.warmup_calls += 1

    def close(self) -> None:
        FakeTranscriber.close_calls += 1

    def transcribe_file(self, path, *, options):
        FakeTranscriber.last_task = options.task
        FakeTranscriber.last_options = options
        return TranscribeResult(
            text=f"hello from {self.model}",
            language=options.language or "en",
            duration=1.25,
            segments=[{"start": 0.0, "end": 1.25, "text": f"hello from {self.model}"}],
            source=str(path),
        )


class ServerTests(unittest.TestCase):
    def test_parse_speech_api_request_from_multipart(self) -> None:
        content_type, body = build_multipart(
            {
                "model": "whisper:base",
                "language": "zh",
                "response_format": "verbose_json",
                "timestamp_granularities[]": "word",
            }
        )

        request = parse_speech_api_request(
            content_type=content_type,
            body=body,
            default_model="whisper",
        )

        self.assertEqual(request.model, "whisper:base")
        self.assertEqual(request.options.language, "zh")
        self.assertEqual(request.response_format, "verbose_json")
        self.assertTrue(request.options.word_timestamps)
        self.assertIsNotNone(request.uploaded_file)
        self.assertEqual(request.uploaded_file.filename, "sample.wav")

    def test_parse_speech_api_request_accepts_profile_and_correction_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_path = f"{tmpdir}/sample.wav"
            with open(sample_path, "wb") as handle:
                handle.write(b"RIFFdata")
            request = parse_speech_api_request(
                content_type="application/json",
                body=(
                    "{"
                    f"\"audio_path\": \"{sample_path}\","
                    "\"profile\": \"meeting-zh\","
                    "\"hotwords\": [\"Daydream\"],"
                    "\"vocabulary\": {\"helo\": \"hello\"},"
                    "\"correction\": {\"capitalize_sentences\": true},"
                    "\"postprocess\": true,"
                    "\"postprocess_model\": \"local-mm-model\","
                    "\"postprocess_base_url\": \"http://127.0.0.1:11435/v1\","
                    "\"beam_size\": 4"
                    "}"
                ).encode("utf-8"),
                default_model="whisper:base",
            )

        self.assertEqual(request.profile, "meeting-zh")
        self.assertEqual(request.options.hotwords, ["Daydream"])
        self.assertEqual(request.options.vocabulary, {"helo": "hello"})
        self.assertEqual(request.options.correction, {"capitalize_sentences": True})
        self.assertEqual(request.options.beam_size, 4)
        self.assertTrue(request.options.postprocess)
        self.assertEqual(request.options.postprocess_model, "local-mm-model")
        self.assertEqual(request.options.postprocess_base_url, "http://127.0.0.1:11435/v1")
        self.assertIn("profile", request.provided_options)

    def test_state_transcribe_uses_cached_transcriber_and_translation_task(self) -> None:
        state = SpeechAPIState(
            SpeechAPIConfig(host="127.0.0.1", port=11434, model="whisper:base"),
            transcriber_factory=FakeTranscriber,
            model_lister=lambda: [("whisper", "base", "mlx-community/whisper-base-mlx")],
        )
        content_type, body = build_multipart(
            {
                "model": "whisper:base",
                "language": "en",
                "response_format": "verbose_json",
            }
        )
        request = parse_speech_api_request(
            content_type=content_type,
            body=body,
            default_model="whisper:base",
            forced_task="translate",
        )

        with mock.patch("dwhisper.server.ensure_runtime_model", return_value="whisper:base") as ensure_runtime_model:
            result_one = state.transcribe(request)
            result_two = state.transcribe(request)

        self.assertEqual(result_one.text, "hello from whisper:base")
        self.assertEqual(result_two.text, "hello from whisper:base")
        self.assertEqual(FakeTranscriber.last_task, "translate")
        ensure_runtime_model.assert_called_once_with("whisper:base", auto_pull=False, register_alias=False)

    def test_available_models_payload_and_response_formats(self) -> None:
        state = SpeechAPIState(
            SpeechAPIConfig(host="127.0.0.1", port=11434, model="whisper:base"),
            model_lister=lambda: [("whisper", "base", "mlx-community/whisper-base-mlx")],
        )
        payload = state.available_models_payload()
        self.assertEqual(payload["data"][0]["id"], "whisper:base")
        self.assertEqual(state.status_payload()["default_model"], "whisper:base")

        result = TranscribeResult(
            text="hello",
            language="en",
            duration=1.0,
            segments=[{"start": 0.0, "end": 1.0, "text": "hello"}],
        )
        content_type, body = build_transcription_response(result, response_format="text")
        self.assertEqual(content_type, "text/plain; charset=utf-8")
        self.assertEqual(body.decode("utf-8"), "hello")

        content_type, body = build_transcription_response(result, response_format="verbose_json")
        self.assertEqual(content_type, "application/json")
        self.assertIn('"language": "en"', body.decode("utf-8"))
        content_type, body = build_postprocess_response(
            {"text": "cleaned", "raw_text": "raw"},
            response_format="text",
        )
        self.assertEqual(content_type, "text/plain; charset=utf-8")
        self.assertEqual(body.decode("utf-8"), "cleaned")
        self.assertIn("uptime_seconds", state.ready_payload())
        self.assertIn(b"dwhisper_completed_requests", state.metrics_text())

    def test_parse_postprocess_api_request_uses_defaults_and_overrides(self) -> None:
        request = parse_postprocess_api_request(
            content_type="application/json",
            body=(
                "{"
                "\"text\": \"helo world\","
                "\"mode\": \"summary\","
                "\"model\": \"glm-4.1v\""
                "}"
            ).encode("utf-8"),
            default_options={
                "enabled": True,
                "model": "local-mm-model",
                "base_url": "http://127.0.0.1:11435/v1",
                "api_key": "local-key",
                "mode": "clean",
                "timeout": 22.0,
            },
        )

        self.assertEqual(request.text, "helo world")
        self.assertEqual(request.response_format, "json")
        self.assertTrue(request.options.enabled)
        self.assertEqual(request.options.model, "glm-4.1v")
        self.assertEqual(request.options.base_url, "http://127.0.0.1:11435/v1")
        self.assertEqual(request.options.mode, "summary")

    def test_state_postprocess_text_uses_configured_defaults(self) -> None:
        state = SpeechAPIState(
            SpeechAPIConfig(
                host="127.0.0.1",
                port=11434,
                model="whisper:base",
                postprocess_defaults={
                    "postprocess": True,
                    "postprocess_model": "local-mm-model",
                    "postprocess_base_url": "http://127.0.0.1:11435/v1",
                    "postprocess_mode": "clean",
                },
            ),
            postprocessor_factory=lambda options: mock.Mock(
                process_text=mock.Mock(return_value="Hello world.")
            ),
        )
        request = parse_postprocess_api_request(
            content_type="application/json",
            body=b'{"text": "helo world"}',
            default_options=state._postprocess_defaults,
        )

        payload = state.postprocess_text(request)

        self.assertEqual(payload["text"], "Hello world.")
        self.assertEqual(payload["raw_text"], "helo world")
        self.assertTrue(payload["postprocess"]["enabled"])
        self.assertEqual(payload["postprocess"]["model"], "local-mm-model")

    def test_state_warmup_and_close_delegate_to_cached_transcriber(self) -> None:
        FakeTranscriber.warmup_calls = 0
        FakeTranscriber.close_calls = 0
        state = SpeechAPIState(
            SpeechAPIConfig(host="127.0.0.1", port=11434, model="whisper:base", preload=True),
            transcriber_factory=FakeTranscriber,
            model_lister=lambda: [("whisper", "base", "mlx-community/whisper-base-mlx")],
        )

        with mock.patch("dwhisper.server.ensure_runtime_model", return_value="whisper:base") as ensure_runtime_model:
            state.warmup()
            state.close()

        ensure_runtime_model.assert_called_once_with("whisper:base", auto_pull=False, register_alias=False)
        self.assertEqual(FakeTranscriber.warmup_calls, 1)
        self.assertEqual(FakeTranscriber.close_calls, 1)

    def test_state_applies_profile_and_default_corrections_path(self) -> None:
        FakeTranscriber.last_options = None
        state = SpeechAPIState(
            SpeechAPIConfig(host="127.0.0.1", port=11434, model="whisper:base"),
            transcriber_factory=FakeTranscriber,
            model_lister=lambda: [("whisper", "base", "mlx-community/whisper-base-mlx")],
        )
        state._profile_store = mock.Mock()
        profile = mock.Mock()
        profile.name = "meeting-zh"
        profile.model = "whisper:large-v3-turbo"
        profile.output_format = "text"
        profile.transcribe = {
            "language": "zh",
            "correction": {"capitalize_sentences": True},
        }
        state._profile_store.get.return_value = profile
        state._corrections_path = "/tmp/corrections.yaml"
        state._vocabulary_path = "/tmp/vocabulary.yaml"
        state._postprocess_defaults = {
            "postprocess": True,
            "postprocess_model": "local-mm-model",
            "postprocess_base_url": "http://127.0.0.1:11435/v1",
            "postprocess_mode": "clean",
            "postprocess_timeout": 18.0,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_path = f"{tmpdir}/sample.wav"
            with open(sample_path, "wb") as handle:
                handle.write(b"RIFFdata")
            request = parse_speech_api_request(
                content_type="application/json",
                body=f'{{"audio_path": "{sample_path}", "profile": "meeting-zh"}}'.encode("utf-8"),
                default_model="whisper:base",
            )

        with mock.patch("dwhisper.server.ensure_runtime_model", return_value="whisper:large-v3-turbo"):
            resolved_model, options, response_format = state._resolve_request(request)

        self.assertEqual(resolved_model, "whisper:large-v3-turbo")
        self.assertEqual(response_format, "text")
        self.assertEqual(options.language, "zh")
        self.assertEqual(options.correction, {"capitalize_sentences": True})
        self.assertEqual(options.corrections_path, "/tmp/corrections.yaml")
        self.assertEqual(options.vocabulary_path, "/tmp/vocabulary.yaml")
        self.assertTrue(options.postprocess)
        self.assertEqual(options.postprocess_model, "local-mm-model")
        self.assertEqual(options.postprocess_base_url, "http://127.0.0.1:11435/v1")
        self.assertEqual(options.postprocess_timeout, 18.0)


class PostprocessBackendRoutingTests(unittest.TestCase):
    def test_mlx_backend_is_accepted_without_base_url(self) -> None:
        captured: dict[str, object] = {}

        def factory(options):
            captured["options"] = options
            return mock.Mock(process_text=mock.Mock(return_value="Cleaned text."))

        state = SpeechAPIState(
            SpeechAPIConfig(
                host="127.0.0.1",
                port=11434,
                model="whisper:base",
                postprocess_defaults={
                    "postprocess": True,
                    "postprocess_model": "qwen3-mlx",
                    "postprocess_backend": "mlx",
                    "postprocess_mode": "clean",
                },
            ),
            postprocessor_factory=factory,
        )
        request = parse_postprocess_api_request(
            content_type="application/json",
            body=b'{"text": "helo world"}',
            default_options=state._postprocess_defaults,
        )

        payload = state.postprocess_text(request)

        self.assertEqual(payload["text"], "Cleaned text.")
        self.assertEqual(payload["postprocess"]["backend"], "mlx")
        self.assertTrue(payload["postprocess"]["applied"])
        self.assertNotIn("base_url", payload["postprocess"])
        self.assertEqual(captured["options"].resolved_backend(), "mlx")

    def test_request_can_override_backend_and_max_tokens(self) -> None:
        request = parse_postprocess_api_request(
            content_type="application/json",
            body=(
                "{"
                "\"text\": \"helo world\","
                "\"backend\": \"mlx\","
                "\"max_tokens\": 128"
                "}"
            ).encode("utf-8"),
            default_options={"enabled": True, "model": "qwen3-mlx", "backend": "http"},
        )

        self.assertEqual(request.options.backend, "mlx")
        self.assertEqual(request.options.max_tokens, 128)
        self.assertTrue(request.options.max_tokens_explicit)

    def test_route_forced_mode_uses_mode_specific_tokens_when_max_tokens_is_implicit(self) -> None:
        state = SpeechAPIState(
            SpeechAPIConfig(
                host="127.0.0.1",
                port=11434,
                model="whisper:base",
                postprocess_defaults={
                    "postprocess": True,
                    "postprocess_model": "qwen3-mlx",
                    "postprocess_backend": "mlx",
                    "postprocess_mode": "clean",
                },
            ),
        )
        request = parse_postprocess_api_request(
            content_type="application/json",
            body=b'{"text": "helo world"}',
            default_options=state._postprocess_defaults,
        )
        request.options = dataclasses.replace(
            request.options,
            mode="meeting-notes",
            max_tokens=None if not request.options.max_tokens_explicit else request.options.max_tokens,
            max_tokens_explicit=request.options.max_tokens_explicit,
        )

        resolved = state._resolve_postprocess_options(request)

        self.assertEqual(resolved.mode, "meeting-notes")
        self.assertEqual(resolved.max_tokens, 2048)

    def test_failing_backend_returns_raw_text_and_error_metadata(self) -> None:
        def factory(options):
            processor = mock.Mock()
            processor.process_text.side_effect = RuntimeError("boom")
            return processor

        state = SpeechAPIState(
            SpeechAPIConfig(
                host="127.0.0.1",
                port=11434,
                model="whisper:base",
                postprocess_defaults={
                    "postprocess": True,
                    "postprocess_model": "qwen3-mlx",
                    "postprocess_backend": "mlx",
                },
            ),
            postprocessor_factory=factory,
        )
        request = parse_postprocess_api_request(
            content_type="application/json",
            body=b'{"text": "helo world"}',
            default_options=state._postprocess_defaults,
        )

        payload = state.postprocess_text(request)

        self.assertEqual(payload["text"], "helo world")
        self.assertFalse(payload["postprocess"]["applied"])
        self.assertEqual(payload["postprocess"]["error"], "boom")

    def test_mode_route_map_covers_expected_paths(self) -> None:
        from dwhisper.server import POSTPROCESS_ROUTE_MODES

        self.assertEqual(POSTPROCESS_ROUTE_MODES["/v1/text/clean"], "clean")
        self.assertEqual(POSTPROCESS_ROUTE_MODES["/v1/text/summary"], "summary")
        self.assertEqual(POSTPROCESS_ROUTE_MODES["/v1/text/meeting-notes"], "meeting-notes")
        self.assertEqual(POSTPROCESS_ROUTE_MODES["/v1/text/speakers"], "speaker-format")
        self.assertIsNone(POSTPROCESS_ROUTE_MODES["/v1/text/postprocess"])

    def test_health_payload_reports_postprocess_status(self) -> None:
        state = SpeechAPIState(
            SpeechAPIConfig(
                host="127.0.0.1",
                port=11500,
                model="whisper:base",
                postprocess_defaults={
                    "postprocess": True,
                    "postprocess_model": "qwen3-mlx",
                    "postprocess_backend": "mlx",
                    "postprocess_mode": "clean",
                },
            ),
        )

        status = state.status_payload()["postprocess"]

        self.assertTrue(status["enabled"])
        self.assertTrue(status["configured"])
        self.assertEqual(status["backend"], "mlx")
        self.assertEqual(status["model"], "qwen3-mlx")
        self.assertIn("/v1/text/clean", status["routes"])

    def test_warmup_postprocess_loads_mlx_backend(self) -> None:
        calls: list[PostProcessOptions] = []

        def factory(options):
            captured = mock.Mock()
            captured._ensure_loaded = mock.Mock(return_value=("m", "t"))
            calls.append(options)
            factory.processors = getattr(factory, "processors", [])
            factory.processors.append(captured)
            return captured

        state = SpeechAPIState(
            SpeechAPIConfig(
                host="127.0.0.1",
                port=11500,
                model="whisper:base",
                postprocess_defaults={
                    "postprocess": True,
                    "postprocess_model": "qwen3-mlx",
                    "postprocess_backend": "mlx",
                    "postprocess_mode": "clean",
                },
            ),
            postprocessor_factory=factory,
        )

        result = state.warmup_postprocess()

        self.assertTrue(result["warmed"])
        self.assertEqual(calls[0].resolved_backend(), "mlx")
        factory.processors[0]._ensure_loaded.assert_called_once_with()

    def test_warmup_postprocess_is_safe_when_not_configured(self) -> None:
        state = SpeechAPIState(
            SpeechAPIConfig(host="127.0.0.1", port=11500, model="whisper:base"),
        )

        result = state.warmup_postprocess()

        self.assertFalse(result["warmed"])
        self.assertFalse(result["configured"])


class DefaultPortTests(unittest.TestCase):
    def test_default_port_is_not_11434(self) -> None:
        from dwhisper.config import DEFAULT_PORT, get_default_port

        self.assertNotEqual(DEFAULT_PORT, 11434)
        self.assertEqual(get_default_port(), DEFAULT_PORT)


class StreamPostprocessTests(unittest.TestCase):
    def _build_state(self, processor) -> SpeechAPIState:
        return SpeechAPIState(
            SpeechAPIConfig(
                host="127.0.0.1",
                port=11500,
                model="whisper:base",
                postprocess_defaults={
                    "postprocess": True,
                    "postprocess_model": "qwen3-mlx",
                    "postprocess_backend": "mlx",
                    "postprocess_mode": "clean",
                },
            ),
            postprocessor_factory=lambda options: processor,
        )

    def test_stream_emits_deltas_then_done_with_aggregate(self) -> None:
        processor = mock.Mock()
        processor.stream_text = mock.Mock(
            return_value=iter(["Hello", ", ", "world."])
        )

        state = self._build_state(processor)
        request = parse_postprocess_api_request(
            content_type="application/json",
            body=b'{"text": "helo world", "stream": true}',
            default_options=state._postprocess_defaults,
        )

        events = list(state.stream_postprocess(request))

        # opening frame + 3 deltas + final done frame
        self.assertEqual(len(events), 5)
        self.assertFalse(events[0]["done"])
        self.assertEqual(events[0]["delta"], "")
        self.assertEqual(events[0]["response"], "")
        self.assertEqual(events[0]["message"]["content"], "")
        self.assertEqual([e["delta"] for e in events[1:4]], ["Hello", ", ", "world."])
        self.assertEqual([e["response"] for e in events[1:4]], ["Hello", ", ", "world."])
        final = events[-1]
        self.assertTrue(final["done"])
        self.assertEqual(final["text"], "Hello, world.")
        self.assertEqual(final["response"], "Hello, world.")
        self.assertEqual(final["message"]["content"], "Hello, world.")
        self.assertEqual(final["raw_text"], "helo world")
        self.assertTrue(final["postprocess"]["applied"])
        self.assertNotIn("error", final["postprocess"])

    def test_stream_falls_back_when_processor_lacks_stream_text(self) -> None:
        processor = mock.Mock(spec=["process_text"])
        processor.process_text.return_value = "Cleaned output."

        state = self._build_state(processor)
        request = parse_postprocess_api_request(
            content_type="application/json",
            body=b'{"text": "helo world", "stream": true}',
            default_options=state._postprocess_defaults,
        )

        events = list(state.stream_postprocess(request))

        # opening + single aggregated delta + done
        self.assertEqual(len(events), 3)
        self.assertEqual(events[1]["delta"], "Cleaned output.")
        self.assertEqual(events[1]["response"], "Cleaned output.")
        final = events[-1]
        self.assertTrue(final["done"])
        self.assertEqual(final["text"], "Cleaned output.")

    def test_stream_surfaces_backend_errors_in_final_event(self) -> None:
        processor = mock.Mock()
        processor.stream_text = mock.Mock(side_effect=RuntimeError("boom"))

        state = self._build_state(processor)
        request = parse_postprocess_api_request(
            content_type="application/json",
            body=b'{"text": "helo world", "stream": true}',
            default_options=state._postprocess_defaults,
        )

        events = list(state.stream_postprocess(request))
        final = events[-1]

        self.assertTrue(final["done"])
        self.assertFalse(final["postprocess"]["applied"])
        self.assertEqual(final["postprocess"]["error"], "boom")
        self.assertEqual(final["text"], "helo world")

    def test_stream_request_flag_is_parsed_from_body(self) -> None:
        request = parse_postprocess_api_request(
            content_type="application/json",
            body=b'{"text": "hi", "stream": true}',
            default_options={"enabled": True, "model": "qwen3-mlx", "backend": "mlx"},
        )
        self.assertTrue(request.stream)

        non_stream = parse_postprocess_api_request(
            content_type="application/json",
            body=b'{"text": "hi"}',
            default_options={"enabled": True, "model": "qwen3-mlx", "backend": "mlx"},
        )
        self.assertFalse(non_stream.stream)


class PreloadStateTests(unittest.TestCase):
    def test_preload_runs_warmups_in_parallel(self) -> None:
        state = mock.Mock()
        gate = threading.Event()
        started = {"asr": threading.Event(), "post": threading.Event()}
        overlapped = {"value": False}
        overlap_lock = threading.Lock()

        def warmup(model: str) -> None:
            started["asr"].set()
            if started["post"].is_set():
                with overlap_lock:
                    overlapped["value"] = True
            gate.wait(timeout=1.0)

        def warmup_postprocess() -> dict[str, bool]:
            started["post"].set()
            if started["asr"].is_set():
                with overlap_lock:
                    overlapped["value"] = True
            gate.wait(timeout=1.0)
            return {"warmed": True}

        state.warmup.side_effect = warmup
        state.warmup_postprocess.side_effect = warmup_postprocess

        runner = threading.Thread(target=_preload_state, args=(state, "whisper:base"))
        runner.start()
        self.assertTrue(started["asr"].wait(timeout=1.0))
        self.assertTrue(started["post"].wait(timeout=1.0))
        gate.set()
        runner.join(timeout=1.0)

        self.assertFalse(runner.is_alive())
        self.assertTrue(overlapped["value"])


if __name__ == "__main__":
    unittest.main()
