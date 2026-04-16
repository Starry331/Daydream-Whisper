from __future__ import annotations

import tempfile
import unittest
from unittest import mock

from dwhisper.server import (
    SpeechAPIConfig,
    SpeechAPIState,
    build_transcription_response,
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
        self.assertIn("uptime_seconds", state.ready_payload())
        self.assertIn(b"dwhisper_completed_requests", state.metrics_text())

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


if __name__ == "__main__":
    unittest.main()
