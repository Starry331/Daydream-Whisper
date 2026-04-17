from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from dwhisper.config import (
    get_configured_profiles_path,
    get_default_profile,
    get_default_profiles_dir,
    get_default_profiles_path,
)


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
    "postprocess_backend",
    "postprocess_max_tokens",
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


def _merge_store(
    base: "ProfileStore",
    incoming: "ProfileStore",
) -> "ProfileStore":
    profiles = dict(base.profiles)
    profiles.update(incoming.profiles)
    return ProfileStore(
        profiles=profiles,
        default_profile=incoming.default_profile or base.default_profile,
    )


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


def _load_profiles_mapping(payload: dict[str, Any]) -> ProfileStore:
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


def _load_profile_file(path: Path, *, assume_single_profile: bool = False) -> ProfileStore:
    payload = _safe_load_yaml(path)
    if not payload:
        return ProfileStore()

    if assume_single_profile and not isinstance(payload.get("profiles"), dict):
        name = _normalize_string(payload.get("name")) or path.stem
        if not name:
            return ProfileStore()
        profile = TranscribeProfile.from_payload(name, payload)
        default_profile = name if payload.get("default") is True or payload.get("is_default") is True else None
        return ProfileStore(profiles={profile.name: profile}, default_profile=default_profile)

    return _load_profiles_mapping(payload)


def _load_profile_directory(path: Path) -> ProfileStore:
    if not path.exists() or not path.is_dir():
        return ProfileStore()

    store = ProfileStore()
    for candidate in sorted(path.iterdir()):
        if not candidate.is_file() or candidate.suffix.lower() not in {".yaml", ".yml"}:
            continue
        store = _merge_store(store, _load_profile_file(candidate, assume_single_profile=True))
    return store


def load_profile_store(profiles_path: Path | None = None) -> ProfileStore:
    if profiles_path is not None:
        path = profiles_path.expanduser()
        if path.is_dir():
            return _load_profile_directory(path)
        return _load_profile_file(path)

    configured_path = get_configured_profiles_path()
    if configured_path is not None:
        if configured_path.is_dir():
            return _load_profile_directory(configured_path)
        return _load_profile_file(configured_path)

    store = _load_profile_file(get_default_profiles_path())
    return _merge_store(store, _load_profile_directory(get_default_profiles_dir()))


def load_profile(name: str | None = None, *, profiles_path: Path | None = None) -> TranscribeProfile | None:
    return load_profile_store(profiles_path).get(name)
