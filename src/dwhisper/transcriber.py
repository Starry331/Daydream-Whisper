from __future__ import annotations

import argparse
import contextlib
import copy
import dataclasses
import json
import os
import select
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from dwhisper.audio import load_audio_file, write_wav_file
from dwhisper.correction import CorrectionConfig, TranscriptCorrector, load_correction_config
from dwhisper.models import ensure_runtime_model, validate_runtime_model
from dwhisper.postprocess import (
    MLXLMPostProcessor,
    OpenAICompatPostProcessor,
    PostProcessOptions,
    build_postprocessor,
)
from dwhisper.utils import format_srt, format_vtt


def _load_mlx_whisper_transcribe() -> Callable[..., dict[str, Any]]:
    try:
        import mlx_whisper
    except ImportError as exc:  # pragma: no cover - covered by dependency guard behavior
        raise RuntimeError(
            "mlx-whisper is not installed. Install project dependencies before transcribing audio."
        ) from exc

    transcribe = getattr(mlx_whisper, "transcribe", None)
    if transcribe is None:
        raise RuntimeError("mlx-whisper.transcribe is unavailable in the installed package.")
    return transcribe


def _serialize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _serialize_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize_json(item) for item in value]
    if isinstance(value, tuple):
        return [_serialize_json(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass(slots=True)
class TranscribeOptions:
    profile: str | None = None
    language: str | None = None
    task: str = "transcribe"
    word_timestamps: bool = False
    temperature: float = 0.0
    initial_prompt: str | None = None
    verbose: bool = False
    compression_ratio_threshold: float | None = None
    logprob_threshold: float | None = None
    no_speech_threshold: float | None = None
    condition_on_previous_text: bool | None = None
    hallucination_silence_threshold: float | None = None
    clip_timestamps: str | list[float] | None = None
    prepend_punctuations: str | None = None
    append_punctuations: str | None = None
    suppress_tokens: list[int] | None = None
    best_of: int | None = None
    beam_size: int | None = None
    patience: float | None = None
    hotwords: list[str] = field(default_factory=list)
    vocabulary: dict[str, str] = field(default_factory=dict)
    correction: dict[str, Any] | None = None
    corrections_path: str | None = None
    vocabulary_path: str | None = None
    postprocess: bool = False
    postprocess_model: str | None = None
    postprocess_base_url: str | None = None
    postprocess_api_key: str | None = None
    postprocess_mode: str = "clean"
    postprocess_prompt: str | None = None
    postprocess_timeout: float = 30.0
    postprocess_backend: str = "auto"
    postprocess_max_tokens: int | None = None

    def __post_init__(self) -> None:
        self.task = str(self.task).strip().lower() or "transcribe"
        if self.task not in {"transcribe", "translate"}:
            raise ValueError("task must be 'transcribe' or 'translate'.")

    def merged_with_overrides(
        self,
        overrides: dict[str, Any] | None,
        *,
        protected_fields: set[str] | None = None,
    ) -> "TranscribeOptions":
        if not overrides:
            return self

        data = self.to_dict()
        protected = protected_fields or set()
        for field_name, value in overrides.items():
            if field_name in protected or value is None or field_name not in data:
                continue
            if isinstance(value, list):
                data[field_name] = copy.deepcopy(value)
            elif isinstance(value, dict):
                data[field_name] = copy.deepcopy(value)
            else:
                data[field_name] = value
        return TranscribeOptions(**data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "language": self.language,
            "task": self.task,
            "word_timestamps": self.word_timestamps,
            "temperature": self.temperature,
            "initial_prompt": self.initial_prompt,
            "verbose": self.verbose,
            "compression_ratio_threshold": self.compression_ratio_threshold,
            "logprob_threshold": self.logprob_threshold,
            "no_speech_threshold": self.no_speech_threshold,
            "condition_on_previous_text": self.condition_on_previous_text,
            "hallucination_silence_threshold": self.hallucination_silence_threshold,
            "clip_timestamps": (
                list(self.clip_timestamps)
                if isinstance(self.clip_timestamps, list)
                else self.clip_timestamps
            ),
            "prepend_punctuations": self.prepend_punctuations,
            "append_punctuations": self.append_punctuations,
            "suppress_tokens": (
                list(self.suppress_tokens) if self.suppress_tokens is not None else None
            ),
            "best_of": self.best_of,
            "beam_size": self.beam_size,
            "patience": self.patience,
            "hotwords": list(self.hotwords),
            "vocabulary": dict(self.vocabulary),
            "correction": copy.deepcopy(self.correction) if self.correction is not None else None,
            "corrections_path": self.corrections_path,
            "vocabulary_path": self.vocabulary_path,
            "postprocess": self.postprocess,
            "postprocess_model": self.postprocess_model,
            "postprocess_base_url": self.postprocess_base_url,
            "postprocess_api_key": self.postprocess_api_key,
            "postprocess_mode": self.postprocess_mode,
            "postprocess_prompt": self.postprocess_prompt,
            "postprocess_timeout": self.postprocess_timeout,
            "postprocess_backend": self.postprocess_backend,
            "postprocess_max_tokens": self.postprocess_max_tokens,
        }

    def extra_mlx_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        passthrough_fields = (
            "compression_ratio_threshold",
            "logprob_threshold",
            "no_speech_threshold",
            "condition_on_previous_text",
            "hallucination_silence_threshold",
            "clip_timestamps",
            "prepend_punctuations",
            "append_punctuations",
            "suppress_tokens",
            "best_of",
            "beam_size",
            "patience",
        )
        for field_name in passthrough_fields:
            value = getattr(self, field_name)
            if value is None:
                continue
            if field_name == "suppress_tokens":
                kwargs[field_name] = list(value)
                continue
            if field_name == "clip_timestamps" and isinstance(value, list):
                kwargs[field_name] = list(value)
                continue
            kwargs[field_name] = value
        return kwargs

    def build_corrector(self) -> TranscriptCorrector | None:
        corrections_path = (
            Path(self.corrections_path).expanduser() if self.corrections_path else None
        )
        vocabulary_path = (
            Path(self.vocabulary_path).expanduser() if self.vocabulary_path else None
        )
        has_file_config = False
        if corrections_path is not None:
            if not corrections_path.exists():
                raise FileNotFoundError(f"Corrections file not found: {corrections_path}")
            has_file_config = True
        if vocabulary_path is not None:
            if not vocabulary_path.exists():
                raise FileNotFoundError(f"Vocabulary file not found: {vocabulary_path}")
            has_file_config = True

        if self.correction is None and not self.hotwords and not self.vocabulary and not has_file_config:
            return None

        config = load_correction_config(
            corrections_path=corrections_path if has_file_config else None,
            vocabulary_path=vocabulary_path if has_file_config else None,
        )
        if self.correction is not None:
            config = config.merged_with(CorrectionConfig.from_dict(self.correction))
        if self.hotwords or self.vocabulary:
            options_config = CorrectionConfig.from_dict(
                {
                    "hotwords": list(self.hotwords),
                    "vocabulary": dict(self.vocabulary),
                }
            )
            config = config.merged_with(options_config)

        if config.is_noop():
            return None
        return TranscriptCorrector(config=config)

    def build_postprocessor(self) -> OpenAICompatPostProcessor | MLXLMPostProcessor | None:
        if not self.postprocess:
            return None
        return build_postprocessor(
            PostProcessOptions(
                enabled=self.postprocess,
                model=self.postprocess_model,
                base_url=self.postprocess_base_url,
                api_key=self.postprocess_api_key,
                mode=self.postprocess_mode,
                prompt=self.postprocess_prompt,
                timeout=self.postprocess_timeout,
                backend=self.postprocess_backend,
                max_tokens=self.postprocess_max_tokens,
            )
        )


@dataclass(slots=True)
class TranscribeResult:
    text: str
    segments: list[dict[str, Any]] = field(default_factory=list)
    language: str | None = None
    duration: float = 0.0
    processing_time: float = 0.0
    source: str | None = None
    raw_text: str | None = None
    postprocess: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "segments": self.segments,
            "language": self.language,
            "duration": self.duration,
            "processing_time": self.processing_time,
            "source": self.source,
            "raw_text": self.raw_text,
            "postprocess": self.postprocess,
        }

    def render(self, output_format: str = "text") -> str:
        if output_format == "json":
            return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
        if output_format == "srt":
            return format_srt(self.segments)
        if output_format == "vtt":
            return format_vtt(self.segments)
        return self.text.strip()


def _default_worker_timeout() -> float | None:
    raw_value = os.environ.get("DWHISPER_TRANSCRIBE_TIMEOUT") or os.environ.get("DAYDREAM_TRANSCRIBE_TIMEOUT")
    if raw_value is None:
        return None
    try:
        timeout = float(raw_value)
    except ValueError:
        return None
    return timeout if timeout > 0 else None


def _load_fixture_metadata(model_path: Path) -> dict[str, Any] | None:
    metadata_path = model_path / "daydream_fixture.json"
    if not metadata_path.exists():
        return None
    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _fixture_duration_from_audio_file(path: Path) -> float:
    try:
        audio, sample_rate = load_audio_file(path)
    except Exception:
        return 0.0
    if sample_rate <= 0:
        return 0.0
    return float(audio.shape[0]) / float(sample_rate)


def _fixture_payload(
    fixture_metadata: dict[str, Any],
    *,
    duration: float,
    source: str | None = None,
) -> dict[str, Any]:
    text = str(
        fixture_metadata.get("text")
        or fixture_metadata.get("response")
        or "Fixture transcription from Daydream Whisper."
    ).strip()
    language = fixture_metadata.get("language", "en")
    segment_end = duration if duration > 0 else 0.0
    return {
        "text": text,
        "language": language,
        "duration": duration,
        "segments": [
            {
                "id": 1,
                "start": 0.0,
                "end": segment_end,
                "text": text,
            }
        ],
        "source": source,
    }


def load_whisper_model(model: str) -> Path:
    resolved = ensure_runtime_model(model, auto_pull=False, register_alias=False)
    return validate_runtime_model(resolved, source_name=model)


@dataclass(slots=True)
class WhisperTranscriber:
    model: str
    transcribe_impl: Callable[..., dict[str, Any]] | None = None
    worker_timeout: float | None = None
    use_subprocess: bool = True
    persistent_worker: bool = True
    model_path: Path = field(init=False)
    fixture_metadata: dict[str, Any] | None = field(init=False, default=None)
    _persistent_process: subprocess.Popen[str] | None = field(init=False, default=None, repr=False)
    _persistent_lock: threading.Lock = field(init=False, default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        self.model_path = load_whisper_model(self.model)
        self.fixture_metadata = _load_fixture_metadata(self.model_path)
        if self.worker_timeout is None:
            self.worker_timeout = _default_worker_timeout()

    def _transcribe_direct(self, audio_source: str, options: TranscribeOptions) -> dict[str, Any]:
        transcribe_impl = self.transcribe_impl or _load_mlx_whisper_transcribe()
        payload = transcribe_impl(
            audio_source,
            path_or_hf_repo=str(self.model_path),
            language=options.language,
            task=options.task,
            word_timestamps=options.word_timestamps,
            temperature=options.temperature,
            initial_prompt=options.initial_prompt,
            verbose=options.verbose,
            **options.extra_mlx_kwargs(),
        )
        return _serialize_json(payload)

    def _prepare_options(
        self,
        options: TranscribeOptions,
    ) -> tuple[TranscribeOptions, TranscriptCorrector | None, OpenAICompatPostProcessor | None]:
        corrector = options.build_corrector()
        postprocessor = options.build_postprocessor()
        if corrector is None:
            return options, None, postprocessor

        biased_prompt = corrector.biased_initial_prompt(options.initial_prompt)
        if biased_prompt == options.initial_prompt:
            return options, corrector, postprocessor
        return dataclasses.replace(options, initial_prompt=biased_prompt), corrector, postprocessor

    def _worker_command(self, request_path: Path, response_path: Path) -> list[str]:
        return [
            sys.executable,
            "-m",
            "dwhisper.transcriber",
            "--worker",
            str(request_path),
            str(response_path),
        ]

    def _persistent_worker_command(self) -> list[str]:
        return [
            sys.executable,
            "-m",
            "dwhisper.transcriber",
            "--persistent-worker",
        ]

    def _worker_failure_message(self, process: subprocess.CompletedProcess[str]) -> str:
        stderr = (process.stderr or "").strip()
        stdout = (process.stdout or "").strip()
        detail = stderr or stdout or "unknown worker failure"
        if (
            "NSRangeException" in detail
            or "libc++abi" in detail
            or "MetalAllocator" in detail
            or process.returncode < 0
        ):
            return (
                "MLX Whisper crashed during runtime initialization. "
                "This usually indicates a broken Metal / MLX environment on the current machine."
            )
        return f"Whisper worker failed with exit code {process.returncode}: {detail}"

    def _persistent_worker_failure_message(self, returncode: int | None, detail: str | None = None) -> str:
        detail = (detail or "").strip()
        if returncode is None:
            return "Whisper worker stopped unexpectedly before producing a response."
        if (
            "NSRangeException" in detail
            or "libc++abi" in detail
            or "MetalAllocator" in detail
            or returncode < 0
        ):
            return (
                "MLX Whisper crashed during runtime initialization. "
                "This usually indicates a broken Metal / MLX environment on the current machine."
            )
        if detail:
            return f"Whisper worker failed with exit code {returncode}: {detail}"
        return f"Whisper worker failed with exit code {returncode}."

    def _transcribe_with_worker(self, audio_source: str, options: TranscribeOptions) -> dict[str, Any]:
        request = {
            "audio_source": audio_source,
            "model_path": str(self.model_path),
            "options": options.to_dict(),
        }
        with tempfile.TemporaryDirectory(prefix="daydream-transcribe-worker-") as tmpdir:
            request_path = Path(tmpdir) / "request.json"
            response_path = Path(tmpdir) / "response.json"
            request_path.write_text(json.dumps(request, ensure_ascii=False), encoding="utf-8")

            try:
                process = subprocess.run(
                    self._worker_command(request_path, response_path),
                    capture_output=True,
                    text=True,
                    timeout=self.worker_timeout,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    f"Whisper worker timed out after {self.worker_timeout:.1f}s."
                ) from exc

            if process.returncode != 0:
                raise RuntimeError(self._worker_failure_message(process))
            if not response_path.exists():
                raise RuntimeError("Whisper worker did not produce a response payload.")

            with response_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}

    def _start_persistent_worker_locked(self) -> subprocess.Popen[str]:
        process = self._persistent_process
        if process is not None and process.poll() is None:
            return process

        process = subprocess.Popen(
            self._persistent_worker_command(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._persistent_process = process
        return process

    def _stop_persistent_worker_locked(self) -> None:
        process = self._persistent_process
        self._persistent_process = None
        if process is None:
            return

        stdin = process.stdin
        stdout = process.stdout
        if process.poll() is None and stdin is not None:
            try:
                stdin.write(json.dumps({"command": "shutdown"}, ensure_ascii=False) + "\n")
                stdin.flush()
            except Exception:
                pass

        try:
            if stdin is not None:
                stdin.close()
        except Exception:
            pass

        try:
            if stdout is not None:
                stdout.close()
        except Exception:
            pass

        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                process.kill()
                try:
                    process.wait(timeout=0.2)
                except subprocess.TimeoutExpired:
                    pass

    def _request_persistent_worker(self, request: dict[str, Any]) -> dict[str, Any]:
        with self._persistent_lock:
            process = self._start_persistent_worker_locked()
            stdin = process.stdin
            stdout = process.stdout
            if stdin is None or stdout is None:
                self._stop_persistent_worker_locked()
                raise RuntimeError("Whisper worker pipes are unavailable.")

            payload = json.dumps(request, ensure_ascii=False)
            try:
                stdin.write(payload + "\n")
                stdin.flush()
            except BrokenPipeError as exc:
                returncode = process.poll()
                self._stop_persistent_worker_locked()
                raise RuntimeError(self._persistent_worker_failure_message(returncode)) from exc

            timeout = self.worker_timeout
            if timeout is not None:
                ready, _, _ = select.select([stdout], [], [], timeout)
                if not ready:
                    self._stop_persistent_worker_locked()
                    raise RuntimeError(f"Whisper worker timed out after {timeout:.1f}s.")

            line = stdout.readline()
            if not line:
                returncode = process.poll()
                self._stop_persistent_worker_locked()
                raise RuntimeError(self._persistent_worker_failure_message(returncode))

            try:
                response = json.loads(line)
            except json.JSONDecodeError as exc:
                returncode = process.poll()
                self._stop_persistent_worker_locked()
                raise RuntimeError(self._persistent_worker_failure_message(returncode, detail=line)) from exc

            if not isinstance(response, dict):
                raise RuntimeError("Whisper worker returned an invalid response payload.")
            if response.get("ok") is not True:
                raise RuntimeError(str(response.get("error") or "unknown worker failure"))

            payload = response.get("payload")
            return payload if isinstance(payload, dict) else {}

    def _transcribe_with_persistent_worker(self, audio_source: str, options: TranscribeOptions) -> dict[str, Any]:
        return self._request_persistent_worker(
            {
                "command": "transcribe",
                "audio_source": audio_source,
                "model_path": str(self.model_path),
                "options": options.to_dict(),
            }
        )

    def warmup(self) -> None:
        if self.fixture_metadata is not None or self.transcribe_impl is not None:
            return
        if not self.use_subprocess or not self.persistent_worker:
            return
        self._request_persistent_worker(
            {
                "command": "warmup",
                "model_path": str(self.model_path),
            }
        )

    def close(self) -> None:
        with self._persistent_lock:
            self._stop_persistent_worker_locked()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _transcribe(self, audio_source: str, options: TranscribeOptions) -> dict[str, Any]:
        if self.fixture_metadata is not None:
            return _fixture_payload(
                self.fixture_metadata,
                duration=_fixture_duration_from_audio_file(Path(audio_source)),
                source=audio_source,
            )
        if self.transcribe_impl is not None or not self.use_subprocess:
            return self._transcribe_direct(audio_source, options)
        if self.persistent_worker:
            return self._transcribe_with_persistent_worker(audio_source, options)
        return self._transcribe_with_worker(audio_source, options)

    def _coerce_result(
        self,
        payload: dict[str, Any],
        *,
        started_at: float,
        source: str | None = None,
    ) -> TranscribeResult:
        segments = payload.get("segments") if isinstance(payload.get("segments"), list) else []
        duration = float(payload.get("duration", 0.0) or 0.0)
        if duration <= 0.0 and segments:
            duration = max(float(segment.get("end", 0.0) or 0.0) for segment in segments)
        return TranscribeResult(
            text=str(payload.get("text", "")).strip(),
            segments=segments,
            language=payload.get("language"),
            duration=duration,
            processing_time=max(0.0, time.perf_counter() - started_at),
            source=source,
        )

    def transcribe_file(
        self,
        path: str | Path,
        *,
        options: TranscribeOptions | None = None,
    ) -> TranscribeResult:
        options = options or TranscribeOptions()
        options, corrector, postprocessor = self._prepare_options(options)
        audio_path = Path(path).expanduser()
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        started_at = time.perf_counter()
        payload = self._transcribe(str(audio_path), options)
        result = self._coerce_result(payload, started_at=started_at, source=str(audio_path))
        if corrector is not None:
            corrector.apply(result)
        if postprocessor is not None:
            _safe_apply_postprocessor(postprocessor, result)
        return result

    def transcribe_samples(
        self,
        audio: np.ndarray,
        *,
        sample_rate: int = 16000,
        options: TranscribeOptions | None = None,
    ) -> TranscribeResult:
        options = options or TranscribeOptions()
        options, corrector, postprocessor = self._prepare_options(options)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            temp_path = Path(handle.name)

        try:
            write_wav_file(temp_path, np.asarray(audio, dtype=np.float32), sample_rate=sample_rate)
            started_at = time.perf_counter()
            payload = self._transcribe(str(temp_path), options)
            result = self._coerce_result(payload, started_at=started_at, source="live-audio")
            if corrector is not None:
                corrector.apply(result)
            if postprocessor is not None:
                _safe_apply_postprocessor(postprocessor, result)
            return result
        finally:
            temp_path.unlink(missing_ok=True)


def transcribe_file(
    audio_path: str | Path,
    *,
    model: str,
    options: TranscribeOptions | None = None,
) -> TranscribeResult:
    return build_transcriber(model).transcribe_file(audio_path, options=options)


# ---------------------------------------------------------------------------
# mlx-audio backend (Qwen3-ASR, Parakeet, SenseVoice, …)
#
# Unlike mlx-whisper, the ``mlx-audio`` package exposes a simple synchronous
# ``generate_transcription`` call that runs in-process. No subprocess worker
# is needed — MLX metal crashes that plague the old Whisper runtime are not
# an issue here, so the implementation is intentionally minimal.
# ---------------------------------------------------------------------------


def _load_mlx_audio_api() -> tuple[Callable[..., Any], Callable[..., Any]]:
    """Return ``(load_model, generate_transcription)`` from mlx-audio.

    Raises a clear ``RuntimeError`` if the package isn't installed so users
    know exactly what to ``pip install``.
    """
    try:
        from mlx_audio.stt.generate import generate_transcription
        from mlx_audio.stt.utils import load_model
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise RuntimeError(
            "mlx-audio is not installed. Install it to use non-Whisper ASR "
            "models (Qwen3-ASR, Parakeet, SenseVoice, …): pip install mlx-audio"
        ) from exc
    return load_model, generate_transcription


@dataclass(slots=True)
class MlxAudioTranscriber:
    """Transcriber for mlx-audio models (Qwen3-ASR and friends)."""

    model: str
    model_path: Path = field(init=False)
    _model: Any = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        # ``load_whisper_model`` name is historical — under the hood it just
        # resolves + validates any ASR model path, backend-agnostic.
        self.model_path = load_whisper_model(self.model)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        load_model, _ = _load_mlx_audio_api()
        self._model = load_model(str(self.model_path))

    def _coerce_stt_output(
        self,
        stt: Any,
        *,
        started_at: float,
        source: str | None,
        fallback_language: str | None,
    ) -> TranscribeResult:
        raw_segments = getattr(stt, "segments", None) or []
        segments: list[dict[str, Any]] = []
        for seg in raw_segments:
            if not isinstance(seg, dict):
                continue
            segments.append(
                {
                    "text": str(seg.get("text", "")),
                    "start": float(seg.get("start", 0.0) or 0.0),
                    "end": float(seg.get("end", 0.0) or 0.0),
                }
            )
        duration = 0.0
        if segments:
            duration = max(float(s.get("end", 0.0) or 0.0) for s in segments)
        return TranscribeResult(
            text=str(getattr(stt, "text", "") or "").strip(),
            segments=segments,
            language=getattr(stt, "language", None) or fallback_language,
            duration=duration,
            processing_time=max(0.0, time.perf_counter() - started_at),
            source=source,
        )

    def transcribe_file(
        self,
        path: str | Path,
        *,
        options: TranscribeOptions | None = None,
    ) -> TranscribeResult:
        options = options or TranscribeOptions()
        # mlx-audio does not consume ``initial_prompt`` the way Whisper does,
        # so we don't bias the prompt here. The corrector is still applied to
        # the resulting text below.
        corrector = options.build_corrector()
        postprocessor = options.build_postprocessor()

        audio_path = Path(path).expanduser()
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        self._ensure_loaded()
        _, generate_transcription = _load_mlx_audio_api()

        started_at = time.perf_counter()
        stt = generate_transcription(
            model=self._model,
            audio=str(audio_path),
            format="txt",
            verbose=options.verbose,
        )

        result = self._coerce_stt_output(
            stt,
            started_at=started_at,
            source=str(audio_path),
            fallback_language=options.language,
        )
        if corrector is not None:
            corrector.apply(result)
        if postprocessor is not None:
            _safe_apply_postprocessor(postprocessor, result)
        return result

    def transcribe_samples(
        self,
        audio: np.ndarray,
        *,
        sample_rate: int = 16000,
        options: TranscribeOptions | None = None,
    ) -> TranscribeResult:
        options = options or TranscribeOptions()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            temp_path = Path(handle.name)
        try:
            write_wav_file(temp_path, np.asarray(audio, dtype=np.float32), sample_rate=sample_rate)
            result = self.transcribe_file(temp_path, options=options)
            result.source = "live-audio"
            return result
        finally:
            temp_path.unlink(missing_ok=True)

    def close(self) -> None:
        """Release the loaded model reference.

        mlx-audio runs entirely in-process (no worker subprocess to reap), so
        this is effectively a no-op besides dropping the model handle to let
        GC reclaim weights.
        """
        self._model = None


def build_transcriber(model: str) -> "WhisperTranscriber | MlxAudioTranscriber":
    """Dispatch to the right transcriber class based on the model's backend.

    Uses :func:`dwhisper.registry.detect_backend` — the mlx-whisper path is
    returned for every Whisper family (the historical behavior) and the
    mlx-audio path for Qwen3-ASR / Parakeet / SenseVoice / anything else
    flagged in the registry.
    """
    from dwhisper.registry import BACKEND_MLX_AUDIO, detect_backend

    if detect_backend(model) == BACKEND_MLX_AUDIO:
        return MlxAudioTranscriber(model)
    return WhisperTranscriber(model)


def _safe_apply_postprocessor(postprocessor: Any, result: TranscribeResult) -> None:
    """Apply a post-processor while tolerating backend failures.

    A broken local model or unreachable endpoint should not fail the whole
    transcription — we keep the raw transcript and record the error in the
    ``postprocess`` metadata so callers can see what happened.
    """
    try:
        postprocessor.apply(result)
        return
    except Exception as exc:  # pragma: no cover - exercised via targeted test
        options = getattr(postprocessor, "options", None)
        existing = dict(result.postprocess or {})
        existing.update(
            {
                "enabled": bool(getattr(options, "enabled", False)),
                "applied": False,
                "mode": getattr(options, "mode", None),
                "model": getattr(options, "model", None),
                "backend": options.resolved_backend() if options is not None else None,
                "error": str(exc),
            }
        )
        base_url = getattr(options, "base_url", None)
        if base_url is not None:
            existing["base_url"] = base_url
        result.postprocess = existing


def _run_worker(request_path: Path, response_path: Path) -> int:
    with request_path.open("r", encoding="utf-8") as handle:
        request = json.load(handle)

    payload = _run_worker_transcription(request)
    response_path.write_text(
        json.dumps(_serialize_json(payload), ensure_ascii=False),
        encoding="utf-8",
    )
    return 0


def _run_worker_transcription(request: dict[str, Any]) -> dict[str, Any]:
    options = TranscribeOptions(**dict(request.get("options") or {}))
    transcribe_impl = _load_mlx_whisper_transcribe()
    with contextlib.redirect_stdout(sys.stderr):
        payload = transcribe_impl(
            str(request["audio_source"]),
            path_or_hf_repo=str(request["model_path"]),
            language=options.language,
            task=options.task,
            word_timestamps=options.word_timestamps,
            temperature=options.temperature,
            initial_prompt=options.initial_prompt,
            verbose=options.verbose,
            **options.extra_mlx_kwargs(),
        )
    return _serialize_json(payload)


def _warmup_worker_model(model_path: str) -> None:
    import mlx.core as mx
    from mlx_whisper.load_models import load_model
    from mlx_whisper.transcribe import ModelHolder

    ModelHolder.model = load_model(model_path, dtype=mx.float16)
    ModelHolder.model_path = model_path


def _run_persistent_worker() -> int:
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("worker requests must be JSON objects.")

            command = str(request.get("command") or "transcribe").strip().lower()
            if command == "shutdown":
                sys.stdout.write(json.dumps({"ok": True, "payload": {"shutdown": True}}, ensure_ascii=False) + "\n")
                sys.stdout.flush()
                return 0
            if command == "warmup":
                _warmup_worker_model(str(request["model_path"]))
                sys.stdout.write(json.dumps({"ok": True, "payload": {"ready": True}}, ensure_ascii=False) + "\n")
                sys.stdout.flush()
                continue
            if command != "transcribe":
                raise ValueError(f"Unsupported worker command '{command}'.")

            payload = _run_worker_transcription(request)
            sys.stdout.write(json.dumps({"ok": True, "payload": payload}, ensure_ascii=False) + "\n")
            sys.stdout.flush()
        except Exception as exc:
            sys.stdout.write(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False) + "\n")
            sys.stdout.flush()
    return 0


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", nargs=2, metavar=("REQUEST_PATH", "RESPONSE_PATH"))
    parser.add_argument("--persistent-worker", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    if args.persistent_worker:
        return _run_persistent_worker()
    if args.worker is None:
        return 0
    request_arg, response_arg = args.worker
    return _run_worker(Path(request_arg), Path(response_arg))


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess
    raise SystemExit(main())
