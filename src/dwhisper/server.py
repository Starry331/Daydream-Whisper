from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from email.parser import BytesParser
from email.policy import default as email_default_policy
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

from rich.console import Console

from dwhisper.config import (
    get_default_corrections_path,
    get_default_postprocess_api_key,
    get_default_postprocess_base_url,
    get_default_postprocess_enabled,
    get_default_postprocess_mode,
    get_default_postprocess_model,
    get_default_postprocess_timeout,
    get_default_profiles_path,
    get_default_serve_allow_origin,
    get_default_vocabulary_path,
)
from dwhisper.models import ensure_runtime_model, validate_runtime_model
from dwhisper.profiles import ProfileStore, load_profile_store
from dwhisper.registry import list_available
from dwhisper.transcriber import TranscribeOptions, TranscribeResult, WhisperTranscriber


console = Console()


class SpeechAPIError(Exception):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message


@dataclass(slots=True)
class UploadedFile:
    filename: str
    content: bytes
    content_type: str | None = None


@dataclass(slots=True)
class SpeechAPIRequest:
    model: str
    options: TranscribeOptions
    response_format: str = "json"
    uploaded_file: UploadedFile | None = None
    audio_path: str | None = None
    profile: str | None = None
    provided_options: set[str] = field(default_factory=set)
    model_provided: bool = False
    response_format_provided: bool = False


@dataclass(slots=True)
class SpeechAPIConfig:
    host: str
    port: int
    model: str
    auto_pull: bool = False
    max_concurrency: int = 2
    request_timeout: float = 120.0
    max_request_bytes: int = 50 * 1024 * 1024
    preload: bool = False
    allow_origin: str = "*"


@dataclass(slots=True)
class SpeechAPIState:
    config: SpeechAPIConfig
    transcriber_factory: Callable[..., WhisperTranscriber] = WhisperTranscriber
    model_lister: Callable[[], list[tuple[str, str, str]]] = list_available
    _transcribers: dict[str, WhisperTranscriber] = field(default_factory=dict, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _semaphore: threading.BoundedSemaphore = field(init=False)
    _active_requests: int = field(default=0, init=False)
    _started_at: float = field(default_factory=time.time, init=False)
    _completed_requests: int = field(default=0, init=False)
    _failed_requests: int = field(default=0, init=False)
    _total_processing_seconds: float = field(default=0.0, init=False)
    _profile_store: ProfileStore = field(init=False)
    _corrections_path: str | None = field(init=False, default=None)
    _vocabulary_path: str | None = field(init=False, default=None)
    _postprocess_defaults: dict[str, Any] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self._semaphore = threading.BoundedSemaphore(max(1, self.config.max_concurrency))
        self._profile_store = load_profile_store(get_default_profiles_path())
        corrections_path = get_default_corrections_path()
        vocabulary_path = get_default_vocabulary_path()
        self._corrections_path = str(corrections_path) if corrections_path.exists() else None
        self._vocabulary_path = str(vocabulary_path) if vocabulary_path.exists() else None
        postprocess_defaults: dict[str, Any] = {}
        if get_default_postprocess_enabled():
            postprocess_defaults["postprocess"] = True
        postprocess_model = get_default_postprocess_model()
        postprocess_base_url = get_default_postprocess_base_url()
        if postprocess_model:
            postprocess_defaults["postprocess_model"] = postprocess_model
        if postprocess_base_url:
            postprocess_defaults["postprocess_base_url"] = postprocess_base_url
        postprocess_defaults["postprocess_api_key"] = get_default_postprocess_api_key()
        postprocess_defaults["postprocess_mode"] = get_default_postprocess_mode()
        postprocess_defaults["postprocess_timeout"] = get_default_postprocess_timeout()
        self._postprocess_defaults = postprocess_defaults

    def _normalize_model(self, requested_model: str | None) -> str:
        model = (requested_model or self.config.model).strip()
        if not model:
            raise SpeechAPIError(HTTPStatus.BAD_REQUEST, "A model must be specified for transcription requests.")
        return model

    def available_models_payload(self) -> dict[str, Any]:
        entries: list[dict[str, Any]] = []
        seen: set[str] = set()
        for family, variant, target in self.model_lister():
            short_name = family if variant == "default" else f"{family}:{variant}"
            if short_name in seen:
                continue
            seen.add(short_name)
            entries.append(
                {
                    "id": short_name,
                    "object": "model",
                    "owned_by": "dwhisper",
                    "root": target,
                }
            )
        return {"object": "list", "data": entries}

    def status_payload(self) -> dict[str, Any]:
        with self._lock:
            loaded_models = sorted(self._transcribers)
            active_requests = self._active_requests
            completed_requests = self._completed_requests
            failed_requests = self._failed_requests
            total_processing_seconds = self._total_processing_seconds
        return {
            "status": "ok",
            "default_model": self.config.model,
            "loaded_models": loaded_models,
            "active_requests": active_requests,
            "completed_requests": completed_requests,
            "failed_requests": failed_requests,
            "avg_processing_seconds": (
                total_processing_seconds / completed_requests if completed_requests else 0.0
            ),
            "max_concurrency": self.config.max_concurrency,
            "preload": self.config.preload,
            "uptime_seconds": max(0.0, time.time() - self._started_at),
        }

    def ready_payload(self) -> dict[str, Any]:
        payload = self.status_payload()
        payload["ready"] = True
        payload["default_model_loaded"] = self.config.model in payload["loaded_models"]
        return payload

    def metrics_payload(self) -> dict[str, float | int]:
        status = self.status_payload()
        return {
            "uptime_seconds": float(status["uptime_seconds"]),
            "active_requests": int(status["active_requests"]),
            "completed_requests": int(status["completed_requests"]),
            "failed_requests": int(status["failed_requests"]),
            "loaded_models": len(status["loaded_models"]),
            "avg_processing_seconds": float(status["avg_processing_seconds"]),
        }

    def metrics_text(self) -> bytes:
        metrics = self.metrics_payload()
        lines = [
            "# HELP dwhisper_uptime_seconds Process uptime in seconds.",
            "# TYPE dwhisper_uptime_seconds gauge",
            f"dwhisper_uptime_seconds {metrics['uptime_seconds']:.6f}",
            "# HELP dwhisper_active_requests Active in-flight requests.",
            "# TYPE dwhisper_active_requests gauge",
            f"dwhisper_active_requests {metrics['active_requests']}",
            "# HELP dwhisper_completed_requests Total completed transcription requests.",
            "# TYPE dwhisper_completed_requests counter",
            f"dwhisper_completed_requests {metrics['completed_requests']}",
            "# HELP dwhisper_failed_requests Total failed transcription requests.",
            "# TYPE dwhisper_failed_requests counter",
            f"dwhisper_failed_requests {metrics['failed_requests']}",
            "# HELP dwhisper_loaded_models Loaded worker count.",
            "# TYPE dwhisper_loaded_models gauge",
            f"dwhisper_loaded_models {metrics['loaded_models']}",
            "# HELP dwhisper_avg_processing_seconds Average processing time for completed requests.",
            "# TYPE dwhisper_avg_processing_seconds gauge",
            f"dwhisper_avg_processing_seconds {metrics['avg_processing_seconds']:.6f}",
        ]
        return ("\n".join(lines) + "\n").encode("utf-8")

    def _resolve_request(self, request: SpeechAPIRequest) -> tuple[str, TranscribeOptions, str]:
        response_format = request.response_format
        model = request.model
        options = request.options

        profile = self._profile_store.get(request.profile)
        if profile is not None:
            if not request.model_provided and profile.model:
                model = profile.model
            if not request.response_format_provided and profile.output_format:
                response_format = str(profile.output_format).strip().lower()
            options = options.merged_with_overrides(
                profile.transcribe,
                protected_fields=request.provided_options,
            )
            options = options.merged_with_overrides(
                {"profile": profile.name},
                protected_fields=request.provided_options,
            )

        default_paths: dict[str, Any] = {}
        if self._corrections_path is not None:
            default_paths["corrections_path"] = self._corrections_path
        if self._vocabulary_path is not None:
            default_paths["vocabulary_path"] = self._vocabulary_path
        if default_paths:
            options = options.merged_with_overrides(
                default_paths,
                protected_fields=request.provided_options,
            )
        if self._postprocess_defaults:
            options = options.merged_with_overrides(
                self._postprocess_defaults,
                protected_fields=request.provided_options,
            )

        if response_format not in {"json", "text", "srt", "vtt", "verbose_json"}:
            raise SpeechAPIError(
                HTTPStatus.BAD_REQUEST,
                f"Unsupported response_format '{response_format}'.",
            )
        return self._normalize_model(model), options, response_format

    def record_success(self, processing_seconds: float) -> None:
        with self._lock:
            self._completed_requests += 1
            self._total_processing_seconds += max(0.0, processing_seconds)

    def record_failure(self) -> None:
        with self._lock:
            self._failed_requests += 1

    def _get_transcriber(self, model: str) -> WhisperTranscriber:
        normalized_model = self._normalize_model(model)
        with self._lock:
            transcriber = self._transcribers.get(normalized_model)
            if transcriber is not None:
                return transcriber

        ensure_runtime_model(normalized_model, auto_pull=self.config.auto_pull, register_alias=False)
        try:
            transcriber = self.transcriber_factory(
                normalized_model,
                worker_timeout=self.config.request_timeout,
            )
        except TypeError:
            transcriber = self.transcriber_factory(normalized_model)
        with self._lock:
            self._transcribers.setdefault(normalized_model, transcriber)
            return self._transcribers[normalized_model]

    def warmup(self, model: str | None = None) -> None:
        transcriber = self._get_transcriber(model or self.config.model)
        transcriber.warmup()

    def close(self) -> None:
        with self._lock:
            transcribers = list(self._transcribers.values())
            self._transcribers.clear()
        for transcriber in transcribers:
            transcriber.close()

    @contextmanager
    def acquire_slot(self):
        acquired = self._semaphore.acquire(timeout=self.config.request_timeout)
        if not acquired:
            raise SpeechAPIError(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "The transcription server is busy. Try again in a moment.",
            )
        with self._lock:
            self._active_requests += 1
        try:
            yield
        finally:
            with self._lock:
                self._active_requests = max(0, self._active_requests - 1)
            self._semaphore.release()

    def transcribe(self, request: SpeechAPIRequest) -> TranscribeResult:
        model, options, _ = self._resolve_request(request)
        with self.acquire_slot():
            transcriber = self._get_transcriber(model)
            if request.audio_path:
                return transcriber.transcribe_file(request.audio_path, options=options)
            if request.uploaded_file is None:
                raise SpeechAPIError(HTTPStatus.BAD_REQUEST, "No audio file was provided.")

            suffix = Path(request.uploaded_file.filename or "upload.wav").suffix or ".wav"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
                temp_path = Path(handle.name)
                handle.write(request.uploaded_file.content)
            try:
                return transcriber.transcribe_file(temp_path, options=options)
            finally:
                temp_path.unlink(missing_ok=True)


def _to_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _to_float(value: str | None, default: float = 0.0) -> float:
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise SpeechAPIError(HTTPStatus.BAD_REQUEST, f"Invalid float value '{value}'.") from exc


def _to_int(value: str | None, default: int | None = None) -> int | None:
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise SpeechAPIError(HTTPStatus.BAD_REQUEST, f"Invalid integer value '{value}'.") from exc


def _form_value(form: dict[str, list[str]], key: str) -> str | None:
    values = form.get(key)
    if not values:
        return None
    return values[0]


def _normalize_form_payload(form: dict[str, list[str]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, values in form.items():
        normalized_key = key[:-2] if key.endswith("[]") else key
        if not values:
            continue
        existing = payload.get(normalized_key)
        if key.endswith("[]"):
            combined = list(existing) if isinstance(existing, list) else []
            combined.extend(values)
            payload[normalized_key] = combined
            continue
        payload[normalized_key] = values if len(values) > 1 else values[0]
    return payload


def _payload_scalar(payload: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        if key not in payload:
            continue
        value = payload[key]
        if isinstance(value, list):
            if not value:
                continue
            value = value[0]
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _payload_jsonish(payload: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key not in payload:
            continue
        value = payload[key]
        if isinstance(value, (dict, list, bool, int, float)):
            return value
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        try:
            return json.loads(text)
        except Exception:
            return text
    return None


def _payload_string_list(payload: dict[str, Any], *keys: str) -> list[str]:
    value = _payload_jsonish(payload, *keys)
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.replace("\n", ",").split(",") if item.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _payload_vocabulary(payload: dict[str, Any], *keys: str) -> dict[str, str]:
    value = _payload_jsonish(payload, *keys)
    if value is None:
        return {}
    if isinstance(value, dict):
        return {
            str(key).strip(): str(item).strip()
            for key, item in value.items()
            if str(key).strip() and item is not None and str(item).strip()
        }
    if isinstance(value, list):
        mapped: dict[str, str] = {}
        for entry in value:
            if not isinstance(entry, dict):
                continue
            src = entry.get("from") or entry.get("term")
            dst = entry.get("to") or entry.get("replacement")
            if src is None or dst is None:
                continue
            mapped[str(src).strip()] = str(dst).strip()
        return mapped
    if isinstance(value, str):
        mapped = {}
        for line in value.splitlines():
            if "=" not in line:
                continue
            src, dst = line.split("=", 1)
            src = src.strip()
            dst = dst.strip()
            if src and dst:
                mapped[src] = dst
        return mapped
    return {}


def _payload_int_list(payload: dict[str, Any], *keys: str) -> list[int] | None:
    value = _payload_jsonish(payload, *keys)
    if value is None:
        return None
    if isinstance(value, list):
        try:
            return [int(item) for item in value]
        except (TypeError, ValueError) as exc:
            raise SpeechAPIError(HTTPStatus.BAD_REQUEST, "Invalid integer list payload.") from exc
    if isinstance(value, str):
        try:
            return [int(item.strip()) for item in value.split(",") if item.strip()]
        except ValueError as exc:
            raise SpeechAPIError(HTTPStatus.BAD_REQUEST, "Invalid integer list payload.") from exc
    raise SpeechAPIError(HTTPStatus.BAD_REQUEST, "Invalid integer list payload.")


def _parse_multipart_form(content_type: str, body: bytes) -> tuple[dict[str, list[str]], UploadedFile | None]:
    raw_message = (
        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8")
        + body
    )
    message = BytesParser(policy=email_default_policy).parsebytes(raw_message)
    if not message.is_multipart():
        raise SpeechAPIError(HTTPStatus.BAD_REQUEST, "Expected multipart form data.")

    form: dict[str, list[str]] = {}
    uploaded_file: UploadedFile | None = None
    for part in message.iter_parts():
        field_name = part.get_param("name", header="content-disposition")
        if not field_name:
            continue
        filename = part.get_filename()
        payload = part.get_payload(decode=True) or b""
        if filename is not None:
            uploaded_file = UploadedFile(
                filename=filename,
                content=payload,
                content_type=part.get_content_type(),
            )
            continue

        charset = part.get_content_charset() or "utf-8"
        text = payload.decode(charset, errors="replace")
        form.setdefault(field_name, []).append(text)
    return form, uploaded_file


def _parse_json_payload(body: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(body.decode("utf-8"))
    except Exception as exc:
        raise SpeechAPIError(HTTPStatus.BAD_REQUEST, "Malformed JSON request body.") from exc
    if not isinstance(payload, dict):
        raise SpeechAPIError(HTTPStatus.BAD_REQUEST, "JSON body must be an object.")
    return payload


def parse_speech_api_request(
    *,
    content_type: str,
    body: bytes,
    default_model: str,
    forced_task: str | None = None,
) -> SpeechAPIRequest:
    uploaded_file: UploadedFile | None = None
    payload: dict[str, Any]

    if content_type.startswith("multipart/form-data"):
        form, uploaded_file = _parse_multipart_form(content_type, body)
        payload = _normalize_form_payload(form)
    elif content_type.startswith("application/json"):
        payload = _parse_json_payload(body)
    elif content_type.startswith("application/x-www-form-urlencoded"):
        decoded = parse_qs(body.decode("utf-8"), keep_blank_values=False)
        payload = _normalize_form_payload({key: list(values) for key, values in decoded.items()})
    else:
        raise SpeechAPIError(
            HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
            f"Unsupported content type '{content_type}'.",
        )

    model_value = _payload_scalar(payload, "model")
    model_provided = model_value is not None
    model = (model_value or default_model).strip()
    response_format_value = _payload_scalar(payload, "response_format")
    response_format_provided = response_format_value is not None
    task_value = forced_task or _payload_scalar(payload, "task") or "transcribe"
    task = task_value.strip().lower()
    response_format = (response_format_value or "json").strip().lower()
    if response_format not in {"json", "text", "srt", "vtt", "verbose_json"}:
        raise SpeechAPIError(
            HTTPStatus.BAD_REQUEST,
            f"Unsupported response_format '{response_format}'.",
        )

    timestamp_values = _payload_string_list(payload, "timestamp_granularities")
    provided_options: set[str] = set()
    if forced_task is not None or _payload_scalar(payload, "task") is not None:
        provided_options.add("task")

    option_kwargs: dict[str, Any] = {"task": task}
    scalar_option_parsers = {
        "language": lambda: _payload_scalar(payload, "language"),
        "temperature": lambda: _to_float(_payload_scalar(payload, "temperature"), 0.0),
        "initial_prompt": lambda: _payload_scalar(payload, "prompt", "initial_prompt"),
        "verbose": lambda: _to_bool(_payload_scalar(payload, "verbose"), False),
        "compression_ratio_threshold": lambda: _to_float(_payload_scalar(payload, "compression_ratio_threshold"), 0.0),
        "logprob_threshold": lambda: _to_float(_payload_scalar(payload, "logprob_threshold"), 0.0),
        "no_speech_threshold": lambda: _to_float(_payload_scalar(payload, "no_speech_threshold"), 0.0),
        "condition_on_previous_text": lambda: _to_bool(_payload_scalar(payload, "condition_on_previous_text"), False),
        "hallucination_silence_threshold": lambda: _to_float(_payload_scalar(payload, "hallucination_silence_threshold"), 0.0),
        "prepend_punctuations": lambda: _payload_scalar(payload, "prepend_punctuations"),
        "append_punctuations": lambda: _payload_scalar(payload, "append_punctuations"),
        "best_of": lambda: _to_int(_payload_scalar(payload, "best_of"), None),
        "beam_size": lambda: _to_int(_payload_scalar(payload, "beam_size"), None),
        "patience": lambda: _to_float(_payload_scalar(payload, "patience"), 0.0),
        "profile": lambda: _payload_scalar(payload, "profile"),
        "postprocess_model": lambda: _payload_scalar(payload, "postprocess_model"),
        "postprocess_base_url": lambda: _payload_scalar(payload, "postprocess_base_url"),
        "postprocess_api_key": lambda: _payload_scalar(payload, "postprocess_api_key"),
        "postprocess_mode": lambda: _payload_scalar(payload, "postprocess_mode"),
        "postprocess_prompt": lambda: _payload_scalar(payload, "postprocess_prompt"),
        "postprocess_timeout": lambda: _to_float(_payload_scalar(payload, "postprocess_timeout"), 30.0),
    }
    for field_name, parser in scalar_option_parsers.items():
        aliases = {
            "initial_prompt": ("prompt", "initial_prompt"),
            "profile": ("profile",),
        }.get(field_name, (field_name,))
        if any(_payload_scalar(payload, alias) is not None for alias in aliases):
            provided_options.add(field_name)
            option_kwargs[field_name] = parser()

    if _payload_scalar(payload, "postprocess") is not None:
        provided_options.add("postprocess")
        option_kwargs["postprocess"] = _to_bool(_payload_scalar(payload, "postprocess"), False)
    if _payload_scalar(payload, "word_timestamps") is not None or "word" in timestamp_values:
        provided_options.add("word_timestamps")
        option_kwargs["word_timestamps"] = _to_bool(_payload_scalar(payload, "word_timestamps"), False) or ("word" in timestamp_values)
    if "clip_timestamps" in payload:
        provided_options.add("clip_timestamps")
        option_kwargs["clip_timestamps"] = _payload_jsonish(payload, "clip_timestamps")
    if "suppress_tokens" in payload:
        provided_options.add("suppress_tokens")
        option_kwargs["suppress_tokens"] = _payload_int_list(payload, "suppress_tokens")
    if "hotwords" in payload:
        provided_options.add("hotwords")
        option_kwargs["hotwords"] = _payload_string_list(payload, "hotwords")
    if "vocabulary" in payload:
        provided_options.add("vocabulary")
        option_kwargs["vocabulary"] = _payload_vocabulary(payload, "vocabulary")
    if "correction" in payload:
        provided_options.add("correction")
        correction = _payload_jsonish(payload, "correction")
        if correction is not None and not isinstance(correction, dict):
            raise SpeechAPIError(HTTPStatus.BAD_REQUEST, "correction must be a JSON object.")
        option_kwargs["correction"] = correction

    options = TranscribeOptions(**option_kwargs)

    audio_path_value = payload.get("audio_path") or payload.get("file_path")
    audio_path = str(audio_path_value).strip() if audio_path_value is not None else None
    if uploaded_file is None and audio_path is None:
        raise SpeechAPIError(HTTPStatus.BAD_REQUEST, "Request must include an audio file or audio_path.")

    if audio_path is not None:
        candidate = Path(audio_path).expanduser()
        if not candidate.exists():
            raise SpeechAPIError(HTTPStatus.BAD_REQUEST, f"Audio path not found: {candidate}")
        audio_path = str(candidate)

    return SpeechAPIRequest(
        model=model,
        options=options,
        response_format=response_format,
        uploaded_file=uploaded_file,
        audio_path=audio_path,
        profile=options.profile,
        provided_options=provided_options,
        model_provided=model_provided,
        response_format_provided=response_format_provided,
    )


def build_transcription_response(
    result: TranscribeResult,
    *,
    response_format: str,
) -> tuple[str, bytes]:
    if response_format == "text":
        return "text/plain; charset=utf-8", result.text.strip().encode("utf-8")
    if response_format == "srt":
        return "application/x-subrip; charset=utf-8", result.render("srt").encode("utf-8")
    if response_format == "vtt":
        return "text/vtt; charset=utf-8", result.render("vtt").encode("utf-8")
    if response_format == "verbose_json":
        return "application/json", json.dumps(result.to_dict(), ensure_ascii=False).encode("utf-8")
    return "application/json", json.dumps({"text": result.text}, ensure_ascii=False).encode("utf-8")


def _json_error(status_code: int, message: str) -> bytes:
    return json.dumps(
        {
            "error": {
                "message": message,
                "type": "invalid_request_error" if status_code < 500 else "server_error",
            }
        },
        ensure_ascii=False,
    ).encode("utf-8")


class DaydreamSpeechHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, server_address, RequestHandlerClass, *, state: SpeechAPIState):
        self.state = state
        self.request_queue_size = max(16, state.config.max_concurrency * 4)
        super().__init__(server_address, RequestHandlerClass)


def make_handler(state: SpeechAPIState):
    class SpeechAPIHandler(BaseHTTPRequestHandler):
        server: DaydreamSpeechHTTPServer
        protocol_version = "HTTP/1.1"

        def _write_common_headers(self, *, request_id: str | None = None, model: str | None = None) -> None:
            self.send_header("X-Daydream-Model", model or state.config.model)
            self.send_header("X-Daydream-Active-Requests", str(state.status_payload()["active_requests"]))
            self.send_header("X-Daydream-Max-Concurrency", str(state.config.max_concurrency))
            self.send_header("Access-Control-Allow-Origin", state.config.allow_origin)
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Request-ID")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            if request_id:
                self.send_header("X-Request-ID", request_id)

        def _send(
            self,
            status_code: int,
            body: bytes,
            *,
            content_type: str = "application/json",
            request_id: str | None = None,
            model: str | None = None,
        ) -> None:
            self.send_response(status_code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self._write_common_headers(request_id=request_id, model=model)
            self.end_headers()
            self.wfile.write(body)

        def _send_json(
            self,
            status_code: int,
            payload: dict[str, Any],
            *,
            request_id: str | None = None,
            model: str | None = None,
        ) -> None:
            self._send(
                status_code,
                json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                request_id=request_id,
                model=model,
            )

        def _read_body(self) -> bytes:
            raw_length = self.headers.get("Content-Length")
            if raw_length is None:
                raise SpeechAPIError(HTTPStatus.LENGTH_REQUIRED, "Content-Length header is required.")
            try:
                content_length = int(raw_length)
            except ValueError as exc:
                raise SpeechAPIError(HTTPStatus.BAD_REQUEST, "Invalid Content-Length header.") from exc
            if content_length < 0:
                raise SpeechAPIError(HTTPStatus.BAD_REQUEST, "Invalid Content-Length header.")
            if content_length > state.config.max_request_bytes:
                raise SpeechAPIError(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                    f"Request body exceeds the configured limit of {state.config.max_request_bytes} bytes.",
                )
            return self.rfile.read(content_length)

        def log_message(self, format: str, *args) -> None:
            return None

        def do_OPTIONS(self) -> None:
            self.send_response(HTTPStatus.NO_CONTENT)
            self.send_header("Content-Length", "0")
            self._write_common_headers()
            self.end_headers()

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/health":
                self._send_json(HTTPStatus.OK, state.status_payload())
                return
            if parsed.path == "/ready":
                self._send_json(HTTPStatus.OK, state.ready_payload())
                return
            if parsed.path == "/metrics":
                self._send(
                    HTTPStatus.OK,
                    state.metrics_text(),
                    content_type="text/plain; version=0.0.4; charset=utf-8",
                )
                return
            if parsed.path == "/v1/models":
                self._send_json(HTTPStatus.OK, state.available_models_payload())
                return
            self._send(HTTPStatus.NOT_FOUND, _json_error(HTTPStatus.NOT_FOUND, "Route not found."))

        def do_POST(self) -> None:
            started_at = time.perf_counter()
            request_id = self.headers.get("X-Request-ID") or uuid.uuid4().hex
            parsed = urlparse(self.path)
            forced_task: str | None = None
            if parsed.path == "/v1/audio/transcriptions":
                forced_task = None
            elif parsed.path == "/v1/audio/translations":
                forced_task = "translate"
            else:
                self._send(HTTPStatus.NOT_FOUND, _json_error(HTTPStatus.NOT_FOUND, "Route not found."))
                return

            try:
                body = self._read_body()
                content_type = self.headers.get("Content-Type", "application/octet-stream")
                request = parse_speech_api_request(
                    content_type=content_type,
                    body=body,
                    default_model=state.config.model,
                    forced_task=forced_task,
                )
                resolved_model, _, resolved_response_format = state._resolve_request(request)
                result = state.transcribe(request)
                response_content_type, response_body = build_transcription_response(
                    result,
                    response_format=resolved_response_format,
                )
                processing_ms = int((time.perf_counter() - started_at) * 1000)
                state.record_success(processing_ms / 1000.0)
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", response_content_type)
                self.send_header("Content-Length", str(len(response_body)))
                self.send_header("X-Daydream-Processing-Ms", str(processing_ms))
                self._write_common_headers(request_id=request_id, model=resolved_model)
                self.end_headers()
                self.wfile.write(response_body)
            except SpeechAPIError as exc:
                state.record_failure()
                self._send(
                    exc.status_code,
                    _json_error(exc.status_code, exc.message),
                    request_id=request_id,
                )
            except Exception as exc:
                state.record_failure()
                self._send(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    _json_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc)),
                    request_id=request_id,
                )

    return SpeechAPIHandler


def create_server(state: SpeechAPIState) -> DaydreamSpeechHTTPServer:
    handler = make_handler(state)
    return DaydreamSpeechHTTPServer((state.config.host, state.config.port), handler, state=state)


def start_server(
    *,
    model: str,
    host: str,
    port: int,
    auto_pull: bool = False,
    max_concurrency: int = 2,
    request_timeout: float = 120.0,
    max_request_bytes: int = 50 * 1024 * 1024,
    preload: bool = False,
    allow_origin: str | None = None,
) -> None:
    if port <= 0 or port > 65535:
        raise ValueError("port must be between 1 and 65535.")
    if max_concurrency <= 0:
        raise ValueError("max_concurrency must be greater than 0.")
    if request_timeout <= 0:
        raise ValueError("request_timeout must be greater than 0.")
    if max_request_bytes <= 0:
        raise ValueError("max_request_bytes must be greater than 0.")

    state = SpeechAPIState(
        SpeechAPIConfig(
            host=host,
            port=port,
            model=model,
            auto_pull=auto_pull,
            max_concurrency=max_concurrency,
            request_timeout=request_timeout,
            max_request_bytes=max_request_bytes,
            preload=preload,
            allow_origin=(allow_origin or get_default_serve_allow_origin()).strip() or "*",
        )
    )
    resolved_model = ensure_runtime_model(model, auto_pull=auto_pull, register_alias=False)
    validate_runtime_model(resolved_model, source_name=model)
    if preload:
        state.warmup(model)

    server = create_server(state)
    base_url = f"http://{host}:{port}"
    console.print(f"[green]Serving Daydream Whisper API[/] on [bold]{base_url}[/]")
    console.print(f"[dim]Default model:[/] {model}")
    console.print(f"[dim]Max concurrency:[/] {max_concurrency}")
    console.print(f"[dim]Preload:[/] {'enabled' if preload else 'disabled'}")
    console.print(f"[dim]CORS allow-origin:[/] {state.config.allow_origin}")
    console.print(
        "[dim]Routes:[/] GET /health, GET /ready, GET /metrics, GET /v1/models, "
        "POST /v1/audio/transcriptions, POST /v1/audio/translations"
    )

    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        console.print()
    finally:
        server.server_close()
        state.close()
