from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from dwhisper.audio import write_wav_file
from dwhisper.transcriber import TranscribeOptions, WhisperTranscriber
from tests.helpers import write_fake_local_whisper_model


class TranscriberTests(unittest.TestCase):
    def test_transcribe_file_passes_model_path_and_options(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "sample.wav"
            write_wav_file(audio_path, np.zeros(1600, dtype=np.float32), sample_rate=16000)

            with mock.patch("dwhisper.transcriber.load_whisper_model", return_value=Path("/models/whisper")):
                transcriber = WhisperTranscriber("whisper:base", transcribe_impl=mock.Mock(return_value={
                    "text": "hello",
                    "language": "en",
                    "segments": [{"start": 0.0, "end": 1.0, "text": "hello"}],
                    "duration": 1.0,
                }))

                result = transcriber.transcribe_file(
                    audio_path,
                    options=TranscribeOptions(language="en", task="transcribe", word_timestamps=True),
                )

        self.assertEqual(result.text, "hello")
        self.assertEqual(result.language, "en")
        kwargs = transcriber.transcribe_impl.call_args.kwargs
        self.assertEqual(kwargs["path_or_hf_repo"], "/models/whisper")
        self.assertEqual(kwargs["language"], "en")
        self.assertTrue(kwargs["word_timestamps"])

    def test_transcribe_samples_uses_temporary_wave_file(self) -> None:
        calls: list[str] = []

        def fake_transcribe(audio_source: str, **kwargs):
            calls.append(audio_source)
            return {"text": "live", "segments": [], "duration": 0.5}

        with mock.patch("dwhisper.transcriber.load_whisper_model", return_value=Path("/models/whisper")):
            transcriber = WhisperTranscriber("whisper:base", transcribe_impl=fake_transcribe)
            result = transcriber.transcribe_samples(np.ones(800, dtype=np.float32), sample_rate=16000)

        self.assertEqual(result.text, "live")
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0].endswith(".wav"))

    def test_transcribe_file_applies_hotword_prompt_and_vocabulary_correction(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "sample.wav"
            write_wav_file(audio_path, np.zeros(1600, dtype=np.float32), sample_rate=16000)

            transcribe_impl = mock.Mock(
                return_value={
                    "text": "helo from MLX",
                    "language": "en",
                    "segments": [{"start": 0.0, "end": 1.0, "text": "helo from MLX"}],
                    "duration": 1.0,
                }
            )
            with mock.patch("dwhisper.transcriber.load_whisper_model", return_value=Path("/models/whisper")):
                transcriber = WhisperTranscriber("whisper:base", transcribe_impl=transcribe_impl)
                result = transcriber.transcribe_file(
                    audio_path,
                    options=TranscribeOptions(
                        hotwords=["MLX"],
                        vocabulary={"helo": "hello"},
                    ),
                )

        self.assertEqual(result.text, "hello from MLX")
        kwargs = transcribe_impl.call_args.kwargs
        self.assertIn("MLX", kwargs["initial_prompt"])

    def test_extra_mlx_kwargs_only_include_explicit_passthrough_fields(self) -> None:
        options = TranscribeOptions(no_speech_threshold=0.6)

        self.assertEqual(options.extra_mlx_kwargs(), {"no_speech_threshold": 0.6})

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "sample.wav"
            write_wav_file(audio_path, np.zeros(1600, dtype=np.float32), sample_rate=16000)

            transcribe_impl = mock.Mock(return_value={"text": "hello", "segments": [], "duration": 1.0})
            with mock.patch("dwhisper.transcriber.load_whisper_model", return_value=Path("/models/whisper")):
                transcriber = WhisperTranscriber("whisper:base", transcribe_impl=transcribe_impl)
                transcriber.transcribe_file(audio_path, options=options)

        kwargs = transcribe_impl.call_args.kwargs
        self.assertEqual(kwargs["no_speech_threshold"], 0.6)
        self.assertNotIn("compression_ratio_threshold", kwargs)

    def test_transcribe_file_loads_corrections_from_yaml_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "sample.wav"
            corrections_path = Path(tmpdir) / "corrections.yaml"
            vocabulary_path = Path(tmpdir) / "vocabulary.yaml"
            write_wav_file(audio_path, np.zeros(1600, dtype=np.float32), sample_rate=16000)
            corrections_path.write_text(
                "capitalize_sentences: true\n"
                "ensure_terminal_punctuation: true\n",
                encoding="utf-8",
            )
            vocabulary_path.write_text(
                "vocabulary:\n"
                "  helo: hello\n",
                encoding="utf-8",
            )

            transcribe_impl = mock.Mock(
                return_value={
                    "text": "helo world",
                    "language": "en",
                    "segments": [{"start": 0.0, "end": 1.0, "text": "helo world"}],
                    "duration": 1.0,
                }
            )
            with mock.patch("dwhisper.transcriber.load_whisper_model", return_value=Path("/models/whisper")):
                transcriber = WhisperTranscriber("whisper:base", transcribe_impl=transcribe_impl)
                result = transcriber.transcribe_file(
                    audio_path,
                    options=TranscribeOptions(
                        corrections_path=str(corrections_path),
                        vocabulary_path=str(vocabulary_path),
                    ),
                )

        self.assertEqual(result.text, "Hello world.")

    def test_transcribe_file_can_optionally_postprocess_with_local_text_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "sample.wav"
            write_wav_file(audio_path, np.zeros(1600, dtype=np.float32), sample_rate=16000)

            transcribe_impl = mock.Mock(
                return_value={
                    "text": "helo world",
                    "language": "en",
                    "segments": [{"start": 0.0, "end": 1.0, "text": "helo world"}],
                    "duration": 1.0,
                }
            )
            with mock.patch("dwhisper.transcriber.load_whisper_model", return_value=Path("/models/whisper")), \
                mock.patch(
                    "dwhisper.postprocess.OpenAICompatPostProcessor._request",
                    return_value={"choices": [{"message": {"content": "Hello, world."}}]},
                ) as postprocess_request:
                transcriber = WhisperTranscriber("whisper:base", transcribe_impl=transcribe_impl)
                result = transcriber.transcribe_file(
                    audio_path,
                    options=TranscribeOptions(
                        postprocess=True,
                        postprocess_model="qwen-local",
                        postprocess_base_url="http://127.0.0.1:11435/v1",
                    ),
                )

        self.assertEqual(result.raw_text, "helo world")
        self.assertEqual(result.text, "Hello, world.")
        self.assertTrue(result.postprocess["applied"])
        postprocess_request.assert_called_once()

    def test_fixture_model_bypasses_mlx_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = write_fake_local_whisper_model(Path(tmpdir) / "fixture-model", fixture=True)
            audio_path = Path(tmpdir) / "sample.wav"
            write_wav_file(audio_path, np.zeros(1600, dtype=np.float32), sample_rate=16000)
            (model_path / "daydream_fixture.json").write_text(
                '{"text":"fixture transcript","language":"en"}',
                encoding="utf-8",
            )

            with mock.patch("dwhisper.transcriber.load_whisper_model", return_value=model_path), \
                mock.patch("dwhisper.transcriber.subprocess.run") as subprocess_run:
                transcriber = WhisperTranscriber("whisper:base")
                result = transcriber.transcribe_file(audio_path)

        self.assertEqual(result.text, "fixture transcript")
        subprocess_run.assert_not_called()

    def test_worker_crash_is_reported_as_runtime_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "sample.wav"
            write_wav_file(audio_path, np.zeros(1600, dtype=np.float32), sample_rate=16000)

            process = subprocess.CompletedProcess(
                args=["python3"],
                returncode=-6,
                stdout="",
                stderr="libc++abi: terminating due to uncaught exception of type NSException",
            )
            with mock.patch("dwhisper.transcriber.load_whisper_model", return_value=Path("/models/whisper")), \
                mock.patch("dwhisper.transcriber.subprocess.run", return_value=process):
                transcriber = WhisperTranscriber("whisper:base", persistent_worker=False)
                with self.assertRaisesRegex(RuntimeError, "MLX Whisper crashed during runtime initialization"):
                    transcriber.transcribe_file(audio_path)

    def test_invalid_task_is_rejected_early(self) -> None:
        with self.assertRaisesRegex(ValueError, "task must be"):
            TranscribeOptions(task="summarize")

    def test_transcribe_file_uses_persistent_worker_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "sample.wav"
            write_wav_file(audio_path, np.zeros(1600, dtype=np.float32), sample_rate=16000)

            with mock.patch("dwhisper.transcriber.load_whisper_model", return_value=Path("/models/whisper")), \
                mock.patch.object(
                    WhisperTranscriber,
                    "_transcribe_with_persistent_worker",
                    return_value={
                        "text": "cached worker",
                        "language": "en",
                        "segments": [],
                        "duration": 0.1,
                    },
                ) as persistent_worker:
                transcriber = WhisperTranscriber("whisper:base")
                result = transcriber.transcribe_file(audio_path)

        self.assertEqual(result.text, "cached worker")
        persistent_worker.assert_called_once()

    def test_warmup_requests_persistent_worker(self) -> None:
        with mock.patch("dwhisper.transcriber.load_whisper_model", return_value=Path("/models/whisper")), \
            mock.patch.object(WhisperTranscriber, "_request_persistent_worker", return_value={"ready": True}) as worker_request:
            transcriber = WhisperTranscriber("whisper:base")
            transcriber.warmup()

        worker_request.assert_called_once_with(
            {
                "command": "warmup",
                "model_path": "/models/whisper",
            }
        )


if __name__ == "__main__":
    unittest.main()
