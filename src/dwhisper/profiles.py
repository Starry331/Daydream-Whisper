from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from dwhisper.config import get_default_profile, get_default_profiles_path


TRANSCRIBE_PROFILE_FIELDS = {
    "language",
    "task",
    "word_timestamps",
    "temperature",
    "initial_prompt",
    "verbose",
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
    "hotwords",
    "vocabulary",
    "correction",
    "corrections_path",
    "vocabulary_path",
    "postprocess",
    "postprocess_model",
    "postprocess_base_url",
    "postprocess_api_key",
    "postprocess_mode",
    "postprocess_prompt",
    "postprocess_timeout",
}

LISTEN_PROFILE_FIELDS = {
    "device",
    "sample_rate",
    "chunk_duration",
    "overlap_duration",
    "silence_threshold",
    "vad_sensitivity",
    "push_to_talk",
    "output_format",
}


def _safe_load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _normalize_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _copy_mapping_subset(payload: dict[str, Any], allowed_keys: set[str]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key in allowed_keys and value is not None
    }


@dataclass(slots=True)
class TranscribeProfile:
    name: str
    description: str | None = None
    model: str | None = None
    output_format: str | None = None
    transcribe: dict[str, Any] = field(default_factory=dict)
    listen: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, name: str, payload: dict[str, Any]) -> "TranscribeProfile":
        if not isinstance(payload, dict):
            raise ValueError(f"Profile '{name}' must be a mapping.")

        transcribe_payload = _copy_mapping_subset(payload, TRANSCRIBE_PROFILE_FIELDS)
        for nested_key in ("transcribe", "options"):
            nested = payload.get(nested_key)
            if isinstance(nested, dict):
                transcribe_payload.update(_copy_mapping_subset(nested, TRANSCRIBE_PROFILE_FIELDS))

        listen_payload = _copy_mapping_subset(payload.get("listen", {}) if isinstance(payload.get("listen"), dict) else {}, LISTEN_PROFILE_FIELDS)
        listen_payload.update(_copy_mapping_subset(payload, LISTEN_PROFILE_FIELDS))

        return cls(
            name=name,
            description=_normalize_string(payload.get("description")),
            model=_normalize_string(payload.get("model")),
            output_format=_normalize_string(payload.get("output_format")),
            transcribe=transcribe_payload,
            listen=listen_payload,
        )


@dataclass(slots=True)
class ProfileStore:
    profiles: dict[str, TranscribeProfile] = field(default_factory=dict)
    default_profile: str | None = None

    def get(self, name: str | None = None) -> TranscribeProfile | None:
        requested = _normalize_string(name) or _normalize_string(self.default_profile) or get_default_profile()
        if not requested:
            return None
        profile = self.profiles.get(requested)
        if profile is None:
            available = ", ".join(sorted(self.profiles)) or "none"
            raise ValueError(f"Profile '{requested}' not found. Available profiles: {available}.")
        return profile

    def list(self) -> list[TranscribeProfile]:
        return [self.profiles[name] for name in sorted(self.profiles)]


def load_profile_store(profiles_path: Path | None = None) -> ProfileStore:
    path = profiles_path or get_default_profiles_path()
    payload = _safe_load_yaml(path)
    if not payload:
        return ProfileStore()

    default_profile = _normalize_string(payload.get("default_profile") or payload.get("default"))
    raw_profiles = payload.get("profiles")
    if isinstance(raw_profiles, dict):
        profiles_payload = raw_profiles
    else:
        profiles_payload = {
            key: value
            for key, value in payload.items()
            if key not in {"default", "default_profile", "profiles"}
        }

    profiles: dict[str, TranscribeProfile] = {}
    for name, value in profiles_payload.items():
        if not isinstance(name, str):
            continue
        if not isinstance(value, dict):
            continue
        profile = TranscribeProfile.from_payload(name, value)
        profiles[profile.name] = profile

    return ProfileStore(profiles=profiles, default_profile=default_profile)


def load_profile(name: str | None = None, *, profiles_path: Path | None = None) -> TranscribeProfile | None:
    return load_profile_store(profiles_path).get(name)
