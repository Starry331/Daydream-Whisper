from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

def _env_lookup(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value is not None:
            return value
    return None


DAYDREAM_HOME = Path(_env_lookup("DWHISPER_HOME", "DAYDREAM_HOME") or "~/.dwhisper").expanduser()
REGISTRY_FILE = DAYDREAM_HOME / "registry.yaml"
CONFIG_FILE = DAYDREAM_HOME / "config.yaml"
CORRECTIONS_FILE = DAYDREAM_HOME / "corrections.yaml"
VOCABULARY_FILE = DAYDREAM_HOME / "vocabulary.yaml"
PROFILES_FILE = DAYDREAM_HOME / "profiles.yaml"

HF_HOME = Path(_env_lookup("HF_HOME") or "~/.cache/huggingface").expanduser()
MODEL_CACHE_DIR = Path(
    _env_lookup("DWHISPER_CACHE_DIR", "DAYDREAM_CACHE_DIR", "HF_HUB_CACHE") or str(HF_HOME / "hub")
).expanduser()
LOCAL_MODELS_DIR = Path(
    _env_lookup("DWHISPER_LOCAL_MODELS_DIR", "DAYDREAM_LOCAL_MODELS_DIR") or str(DAYDREAM_HOME / "models")
).expanduser()

DEFAULT_MODEL = "whisper:large-v3-turbo"
DEFAULT_LANGUAGE: str | None = None
DEFAULT_TASK = "transcribe"
DEFAULT_OUTPUT_FORMAT = "text"
DEFAULT_PROFILE: str | None = None
DEFAULT_WORD_TIMESTAMPS = False
DEFAULT_AUDIO_DEVICE: str | None = None
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHUNK_DURATION = 3.0
DEFAULT_OVERLAP_DURATION = 0.5
DEFAULT_SILENCE_THRESHOLD = 1.0
DEFAULT_VAD_SENSITIVITY = 0.6
DEFAULT_PUSH_TO_TALK = False
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 11434
DEFAULT_SERVE_MAX_CONCURRENCY = 2
DEFAULT_SERVE_REQUEST_TIMEOUT = 120.0
DEFAULT_SERVE_MAX_REQUEST_BYTES = 50 * 1024 * 1024
DEFAULT_SERVE_PRELOAD = False
DEFAULT_SERVE_ALLOW_ORIGIN = "*"


def ensure_home() -> None:
    DAYDREAM_HOME.mkdir(parents=True, exist_ok=True)
    LOCAL_MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _load_config() -> dict[str, Any]:
    if not CONFIG_FILE.exists():
        return {}
    try:
        with CONFIG_FILE.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _write_config(config: dict[str, Any]) -> None:
    ensure_home()
    with CONFIG_FILE.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def _get_nested(config: dict[str, Any], *keys: str) -> Any:
    current: Any = config
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _coerce_str(value: Any, default: str | None) -> str | None:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _coerce_float(value: Any, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value: Any, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return default


def _env_or_default(default: Any, *names: str) -> Any:
    value = _env_lookup(*names)
    return default if value is None else value


def _config_value(default: Any, *keys: str, env_names: tuple[str, ...] = ()) -> Any:
    env_value = _env_lookup(*env_names)
    if env_value is not None:
        return env_value
    return _get_nested(_load_config(), *keys) if keys else _load_config()


def get_default_model() -> str:
    value = _env_or_default(_load_config().get("model"), "DWHISPER_MODEL", "DAYDREAM_MODEL")
    return _coerce_str(value, DEFAULT_MODEL) or DEFAULT_MODEL


def get_default_language() -> str | None:
    value = _config_value(
        DEFAULT_LANGUAGE,
        "transcribe",
        "language",
        env_names=("DWHISPER_LANGUAGE", "DAYDREAM_LANGUAGE"),
    )
    return _coerce_str(value, DEFAULT_LANGUAGE)


def get_default_task() -> str:
    value = _coerce_str(
        _config_value(
            DEFAULT_TASK,
            "transcribe",
            "task",
            env_names=("DWHISPER_TASK", "DAYDREAM_TASK"),
        ),
        DEFAULT_TASK,
    ) or DEFAULT_TASK
    return value if value in {"transcribe", "translate"} else DEFAULT_TASK


def get_default_output_format() -> str:
    value = _coerce_str(
        _config_value(
            DEFAULT_OUTPUT_FORMAT,
            "transcribe",
            "output_format",
            env_names=("DWHISPER_OUTPUT_FORMAT", "DAYDREAM_OUTPUT_FORMAT"),
        ),
        DEFAULT_OUTPUT_FORMAT,
    ) or DEFAULT_OUTPUT_FORMAT
    return value if value in {"text", "json", "srt", "vtt"} else DEFAULT_OUTPUT_FORMAT


def get_default_profile() -> str | None:
    value = _config_value(
        DEFAULT_PROFILE,
        "transcribe",
        "profile",
        env_names=("DWHISPER_PROFILE", "DAYDREAM_PROFILE"),
    )
    return _coerce_str(value, DEFAULT_PROFILE)


def get_default_word_timestamps() -> bool:
    value = _config_value(
        DEFAULT_WORD_TIMESTAMPS,
        "transcribe",
        "word_timestamps",
        env_names=("DWHISPER_WORD_TIMESTAMPS", "DAYDREAM_WORD_TIMESTAMPS"),
    )
    return _coerce_bool(value, DEFAULT_WORD_TIMESTAMPS)


def get_default_audio_device() -> str | None:
    value = _config_value(
        DEFAULT_AUDIO_DEVICE,
        "audio",
        "device",
        env_names=("DWHISPER_AUDIO_DEVICE", "DAYDREAM_AUDIO_DEVICE"),
    )
    return _coerce_str(value, DEFAULT_AUDIO_DEVICE)


def get_default_sample_rate() -> int:
    value = _config_value(
        DEFAULT_SAMPLE_RATE,
        "audio",
        "sample_rate",
        env_names=("DWHISPER_SAMPLE_RATE", "DAYDREAM_SAMPLE_RATE"),
    )
    return _coerce_int(value, DEFAULT_SAMPLE_RATE)


def get_default_chunk_duration() -> float:
    value = _config_value(
        DEFAULT_CHUNK_DURATION,
        "listen",
        "chunk_duration",
        env_names=("DWHISPER_CHUNK_DURATION", "DAYDREAM_CHUNK_DURATION"),
    )
    return _coerce_float(value, DEFAULT_CHUNK_DURATION)


def get_default_overlap_duration() -> float:
    value = _config_value(
        DEFAULT_OVERLAP_DURATION,
        "listen",
        "overlap_duration",
        env_names=("DWHISPER_OVERLAP_DURATION", "DAYDREAM_OVERLAP_DURATION"),
    )
    return _coerce_float(value, DEFAULT_OVERLAP_DURATION)


def get_default_silence_threshold() -> float:
    value = _config_value(
        DEFAULT_SILENCE_THRESHOLD,
        "listen",
        "silence_threshold",
        env_names=("DWHISPER_SILENCE_THRESHOLD", "DAYDREAM_SILENCE_THRESHOLD"),
    )
    return _coerce_float(value, DEFAULT_SILENCE_THRESHOLD)


def get_default_vad_sensitivity() -> float:
    value = _config_value(
        DEFAULT_VAD_SENSITIVITY,
        "listen",
        "vad_sensitivity",
        env_names=("DWHISPER_VAD_SENSITIVITY", "DAYDREAM_VAD_SENSITIVITY"),
    )
    return _coerce_float(value, DEFAULT_VAD_SENSITIVITY)


def get_default_push_to_talk() -> bool:
    value = _config_value(
        DEFAULT_PUSH_TO_TALK,
        "listen",
        "push_to_talk",
        env_names=("DWHISPER_PUSH_TO_TALK", "DAYDREAM_PUSH_TO_TALK"),
    )
    return _coerce_bool(value, DEFAULT_PUSH_TO_TALK)


def get_default_host() -> str:
    value = _config_value(
        DEFAULT_HOST,
        "serve",
        "host",
        env_names=("DWHISPER_HOST", "DAYDREAM_HOST"),
    )
    return _coerce_str(value, DEFAULT_HOST) or DEFAULT_HOST


def get_default_port() -> int:
    value = _config_value(
        DEFAULT_PORT,
        "serve",
        "port",
        env_names=("DWHISPER_PORT", "DAYDREAM_PORT"),
    )
    return _coerce_int(value, DEFAULT_PORT)


def get_default_serve_max_concurrency() -> int:
    value = _config_value(
        DEFAULT_SERVE_MAX_CONCURRENCY,
        "serve",
        "max_concurrency",
        env_names=("DWHISPER_SERVE_MAX_CONCURRENCY", "DAYDREAM_SERVE_MAX_CONCURRENCY"),
    )
    return max(1, _coerce_int(value, DEFAULT_SERVE_MAX_CONCURRENCY))


def get_default_serve_request_timeout() -> float:
    value = _config_value(
        DEFAULT_SERVE_REQUEST_TIMEOUT,
        "serve",
        "request_timeout",
        env_names=("DWHISPER_SERVE_REQUEST_TIMEOUT", "DAYDREAM_SERVE_REQUEST_TIMEOUT"),
    )
    return max(1.0, _coerce_float(value, DEFAULT_SERVE_REQUEST_TIMEOUT))


def get_default_serve_max_request_bytes() -> int:
    value = _config_value(
        DEFAULT_SERVE_MAX_REQUEST_BYTES,
        "serve",
        "max_request_bytes",
        env_names=("DWHISPER_SERVE_MAX_REQUEST_BYTES", "DAYDREAM_SERVE_MAX_REQUEST_BYTES"),
    )
    return max(1024, _coerce_int(value, DEFAULT_SERVE_MAX_REQUEST_BYTES))


def get_default_serve_preload() -> bool:
    value = _config_value(
        DEFAULT_SERVE_PRELOAD,
        "serve",
        "preload",
        env_names=("DWHISPER_SERVE_PRELOAD", "DAYDREAM_SERVE_PRELOAD"),
    )
    return _coerce_bool(value, DEFAULT_SERVE_PRELOAD)


def get_default_serve_allow_origin() -> str:
    value = _config_value(
        DEFAULT_SERVE_ALLOW_ORIGIN,
        "serve",
        "allow_origin",
        env_names=("DWHISPER_SERVE_ALLOW_ORIGIN", "DAYDREAM_SERVE_ALLOW_ORIGIN"),
    )
    return _coerce_str(value, DEFAULT_SERVE_ALLOW_ORIGIN) or DEFAULT_SERVE_ALLOW_ORIGIN


def get_default_corrections_path() -> Path:
    value = _config_value(
        str(CORRECTIONS_FILE),
        "correction",
        "corrections_file",
        env_names=("DWHISPER_CORRECTIONS_FILE", "DAYDREAM_CORRECTIONS_FILE"),
    )
    return Path(str(value or CORRECTIONS_FILE)).expanduser()


def get_default_vocabulary_path() -> Path:
    value = _config_value(
        str(VOCABULARY_FILE),
        "correction",
        "vocabulary_file",
        env_names=("DWHISPER_VOCABULARY_FILE", "DAYDREAM_VOCABULARY_FILE"),
    )
    return Path(str(value or VOCABULARY_FILE)).expanduser()


def get_default_profiles_path() -> Path:
    value = _config_value(
        str(PROFILES_FILE),
        "profiles",
        "file",
        env_names=("DWHISPER_PROFILES_FILE", "DAYDREAM_PROFILES_FILE"),
    )
    return Path(str(value or PROFILES_FILE)).expanduser()


def get_local_model_roots() -> list[Path]:
    config = _load_config()
    configured = _get_nested(config, "models", "local_roots")
    roots: list[Path] = [LOCAL_MODELS_DIR]

    env_roots = _env_lookup("DWHISPER_MODELS_DIRS", "DAYDREAM_MODELS_DIRS")
    if env_roots:
        roots.extend(
            Path(part).expanduser()
            for part in env_roots.split(os.pathsep)
            if part.strip()
        )

    if isinstance(configured, list):
        roots.extend(Path(str(item)).expanduser() for item in configured if str(item).strip())

    unique: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        resolved = root.expanduser()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique
