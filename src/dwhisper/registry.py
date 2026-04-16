"""Model name registry for local and Hugging Face Whisper checkpoints."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import yaml

from dwhisper.config import REGISTRY_FILE, ensure_home, get_local_model_roots

BUILTIN_REGISTRY: dict[str, dict[str, str]] = {
    "whisper": {
        "default": "mlx-community/whisper-large-v3-turbo",
        "tiny": "mlx-community/whisper-tiny-mlx",
        "base": "mlx-community/whisper-base-mlx",
        "small": "mlx-community/whisper-small-mlx",
        "medium": "mlx-community/whisper-medium-mlx",
        "large-v3": "mlx-community/whisper-large-v3-mlx",
        "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    },
    "whisper-quantized": {
        "default": "mlx-community/whisper-large-v3-turbo-q4",
        "large-v3-4bit": "mlx-community/whisper-large-v3-mlx-q4",
        "large-v3-8bit": "mlx-community/whisper-large-v3-mlx-8bit",
    },
}

MODEL_CONFIG_FILES = ("config.json",)
MODEL_MARKER_FILES = (
    "preprocessor_config.json",
    "mel_filters.npz",
)
MODEL_WEIGHT_FILES = (
    "weights.npz",
    "*.safetensors",
    "model.safetensors.index.json",
)
MODEL_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "normalizer.json",
)
MULTIMODAL_MODEL_SUBDIRS = (
    "whisper",
    "whisper_model",
    "speech",
    "speech_model",
    "speech_encoder",
    "audio",
    "audio_model",
    "audio_encoder",
    "asr",
    "encoder",
)


def _save_user_registry(data: dict[str, dict[str, str]]) -> None:
    ensure_home()
    with REGISTRY_FILE.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=True)


def _format_short_name(family: str, variant: str, variants: dict[str, str]) -> Optional[str]:
    if variant == "default":
        return family
    return f"{family}:{variant}"


def _looks_like_local_path(name: str) -> bool:
    return (
        name.startswith(("~", ".", "/"))
        or name.startswith(f"..{os.sep}")
        or f"{os.sep}" in name
    )


def _is_local_target(value: str) -> bool:
    return value.startswith(("~", ".", "/"))


def normalize_hf_reference(name: str) -> str:
    value = name.strip()
    if value.startswith("hf.co/"):
        parts = [part for part in value[6:].split("/") if part]
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}"
        return value

    if value.startswith(
        (
            "https://hf.co/",
            "http://hf.co/",
            "https://huggingface.co/",
            "http://huggingface.co/",
        )
    ):
        parsed = urlparse(value)
        parts = [part for part in parsed.path.split("/") if part]
        if parts[:1] == ["models"]:
            parts = parts[1:]
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}"
    return value


def _sanitize_alias(name: str) -> str:
    alias = name.strip().lower().replace("_", "-").replace(" ", "-")
    alias = re.sub(r"[^a-z0-9.\-]+", "-", alias)
    alias = re.sub(r"-+", "-", alias).strip("-")

    stripped = True
    while stripped and alias:
        stripped = False
        for suffix in (
            "-mlx",
            "-4bit",
            "-8bit",
            "-fp16",
            "-bf16",
            "-int4",
            "-int8",
            "-q4",
            "-q8",
        ):
            if alias.endswith(suffix):
                alias = alias[: -len(suffix)].rstrip("-")
                stripped = True

    return alias or "local-model"


def _is_direct_local_model_dir(path: Path) -> bool:
    path = path.expanduser()
    if not path.is_dir():
        return False

    has_config = all((path / file_name).exists() for file_name in MODEL_CONFIG_FILES)
    if not has_config:
        return False

    has_markers = any((path / file_name).exists() for file_name in MODEL_MARKER_FILES)
    has_tokenizer = any((path / file_name).exists() for file_name in MODEL_TOKENIZER_FILES)
    has_weights = any((path / pattern).exists() for pattern in MODEL_WEIGHT_FILES if "*" not in pattern)
    has_weights = has_weights or any(path.glob("*.safetensors"))

    if (path / "daydream_fixture.json").exists():
        return True

    if not has_weights:
        return False

    if has_markers or has_tokenizer:
        return True

    try:
        with (path / "config.json").open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
    except Exception:
        return False
    if not isinstance(config, dict):
        return False
    return str(config.get("model_type", "")).lower() == "whisper"


def resolve_local_model_dir(path: Path) -> Path | None:
    expanded = path.expanduser()
    if _is_direct_local_model_dir(expanded):
        return expanded.resolve()
    if not expanded.is_dir():
        return None

    for child_name in MULTIMODAL_MODEL_SUBDIRS:
        candidate = expanded / child_name
        if _is_direct_local_model_dir(candidate):
            return candidate.resolve()

    try:
        children = sorted(
            child for child in expanded.iterdir()
            if child.is_dir() and child.name.lower() in MULTIMODAL_MODEL_SUBDIRS
        )
    except OSError:
        return None

    for child in children:
        resolved = resolve_local_model_dir(child)
        if resolved is not None:
            return resolved
    return None


def is_local_model_dir(path: Path) -> bool:
    return resolve_local_model_dir(path) is not None


def _iter_model_dirs(root: Path, *, max_depth: int = 3) -> list[Path]:
    if not root.exists() or not root.is_dir():
        return []

    discovered: list[Path] = []
    stack: list[tuple[Path, int]] = [(root, 0)]
    seen: set[Path] = set()

    while stack:
        current, depth = stack.pop()
        if current in seen:
            continue
        seen.add(current)

        if is_local_model_dir(current):
            discovered.append(current.resolve())
            continue

        if depth >= max_depth:
            continue

        try:
            children = [child for child in current.iterdir() if child.is_dir()]
        except OSError:
            continue

        for child in children:
            stack.append((child, depth + 1))

    return sorted(discovered)


def _load_user_registry() -> dict[str, dict[str, str]]:
    if not REGISTRY_FILE.exists():
        return {}
    try:
        with REGISTRY_FILE.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except Exception:
        return {}

    if not isinstance(data, dict):
        return {}

    normalized: dict[str, dict[str, str]] = {}
    for family, variants in data.items():
        if not isinstance(family, str) or not isinstance(variants, dict):
            continue
        normalized[family.lower()] = {
            str(variant).lower(): str(value)
            for variant, value in variants.items()
            if isinstance(value, str)
        }
    return normalized


def _get_merged_registry() -> dict[str, dict[str, str]]:
    merged = {family: dict(variants) for family, variants in BUILTIN_REGISTRY.items()}
    for family, variants in _load_user_registry().items():
        merged.setdefault(family, {}).update(variants)
    return merged


def _find_short_name_for_target(
    target: str,
    registry: Optional[dict[str, dict[str, str]]] = None,
) -> Optional[str]:
    current = registry or _get_merged_registry()
    for family, variants in current.items():
        for variant, value in variants.items():
            if value != target:
                continue
            return _format_short_name(family, variant, variants)
    return None


def register_local_model(path: str | Path) -> str:
    input_path = Path(path).expanduser()
    model_dir = resolve_local_model_dir(input_path)
    if model_dir is None:
        raise ValueError(f"Not a valid local Whisper model directory: {input_path}")

    target = str(model_dir)
    existing = _find_short_name_for_target(target)
    if existing:
        return existing

    user_registry = _load_user_registry()
    family = _sanitize_alias(input_path.name)
    merged = _get_merged_registry()

    if family in merged:
        if merged[family].get("default") == target and len(merged[family]) == 1:
            return family
        suffix = 2
        while f"{family}-{suffix}" in merged:
            suffix += 1
        family = f"{family}-{suffix}"

    user_registry[family] = {"default": target}
    _save_user_registry(user_registry)
    return family


def register_remote_model(repo_id: str) -> str:
    target = normalize_hf_reference(repo_id)
    existing = _find_short_name_for_target(target)
    if existing:
        return existing

    user_registry = _load_user_registry()
    family = _sanitize_alias(target.rsplit("/", 1)[-1])
    merged = _get_merged_registry()

    if family in merged:
        if merged[family].get("default") == target and len(merged[family]) == 1:
            return family
        suffix = 2
        while f"{family}-{suffix}" in merged:
            suffix += 1
        family = f"{family}-{suffix}"

    user_registry[family] = {"default": target}
    _save_user_registry(user_registry)
    return family


def scan_local_models(*, persist: bool = True) -> list[tuple[str, str]]:
    discovered: list[tuple[str, str]] = []
    seen_targets: set[str] = set()

    for root in get_local_model_roots():
        for model_dir in _iter_model_dirs(root):
            target = str(model_dir)
            if target in seen_targets:
                continue
            seen_targets.add(target)
            short_name = (
                register_local_model(model_dir)
                if persist
                else _find_short_name_for_target(target) or _sanitize_alias(model_dir.name)
            )
            discovered.append((short_name, target))

    return discovered


def resolve(name: str) -> str:
    value = normalize_hf_reference(name.strip())
    if not value:
        raise ValueError("Model name cannot be empty.")

    if _looks_like_local_path(value):
        local_path = Path(value).expanduser()
        if local_path.exists():
            runtime_dir = resolve_local_model_dir(local_path)
            if runtime_dir is None:
                raise ValueError(f"Not a valid local Whisper model directory: {local_path}")
            register_local_model(local_path)
            return str(runtime_dir)
        if value.startswith(("~", ".", "/")):
            raise ValueError(f"Local model path not found: {local_path}")

    if "/" in value:
        return value

    registry = _get_merged_registry()
    family, variant = (value.split(":", 1) + ["default"])[:2] if ":" in value else (value, "default")
    family = family.lower()
    variant = (variant or "default").lower()

    if family not in registry:
        scan_local_models(persist=True)
        registry = _get_merged_registry()

    if family not in registry:
        available = ", ".join(sorted(registry))
        raise ValueError(
            f"Unknown model family '{family}'. Available: {available}. "
            "You can also pass a Hugging Face repo ID or a local model path."
        )

    variants = registry[family]
    if variant not in variants:
        available = ", ".join(sorted(name for name in variants if name != "default"))
        raise ValueError(f"Unknown variant '{variant}' for '{family}'. Available: {available}")

    target = variants[variant]
    if _is_local_target(target):
        local_path = Path(target).expanduser()
        if not local_path.exists():
            raise ValueError(f"Local model alias '{name}' points to a missing path: {local_path}")
        return str(local_path.resolve())
    return target


def copy_alias(source: str, destination: str) -> str:
    repo_id = resolve(source)
    alias = destination.strip().lower()
    if not alias:
        raise ValueError("Destination alias cannot be empty.")
    if "/" in alias:
        raise ValueError("Alias cannot contain '/'. Use a short name like 'voice' or 'voice:fast'.")

    if ":" in alias:
        family, variant = alias.split(":", 1)
    else:
        family, variant = alias, "default"

    user_registry = _load_user_registry()
    merged = _get_merged_registry()
    if family in merged and variant in merged[family]:
        existing = merged[family][variant]
        if existing == repo_id:
            return repo_id
        raise ValueError(f"Alias '{destination}' already exists and points to '{existing}'.")

    user_registry.setdefault(family, {})[variant] = repo_id
    _save_user_registry(user_registry)
    return repo_id


def list_user_aliases() -> list[tuple[str, str]]:
    aliases: list[tuple[str, str]] = []
    for family, variants in _load_user_registry().items():
        for variant, repo_id in variants.items():
            aliases.append((family if variant == "default" else f"{family}:{variant}", repo_id))
    return sorted(aliases)


def remove_alias(alias: str) -> str:
    value = alias.strip().lower()
    family, variant = (value.split(":", 1) + ["default"])[:2] if ":" in value else (value, "default")

    registry = _load_user_registry()
    if family not in registry or variant not in registry[family]:
        raise ValueError(f"Alias '{alias}' not found in user registry.")

    if family in BUILTIN_REGISTRY and variant in BUILTIN_REGISTRY[family]:
        raise ValueError(f"'{alias}' is a built-in model name and cannot be removed.")

    repo_id = registry[family][variant]
    del registry[family][variant]
    if not registry[family]:
        del registry[family]
    _save_user_registry(registry)
    return repo_id


def reverse_lookup(repo_id: str) -> Optional[str]:
    return _find_short_name_for_target(repo_id)


def reverse_lookup_all(repo_id: str) -> list[str]:
    names: list[str] = []
    for family, variants in _get_merged_registry().items():
        for variant, value in variants.items():
            if value != repo_id:
                continue
            short_name = _format_short_name(family, variant, variants)
            if short_name:
                names.append(short_name)
    return names


def list_available() -> list[tuple[str, str, str]]:
    scan_local_models(persist=True)
    available: list[tuple[str, str, str]] = []
    for family, variants in sorted(_get_merged_registry().items()):
        for variant, target in sorted(variants.items()):
            short_name = _format_short_name(family, variant, variants)
            if short_name is None:
                continue
            available.append((family, variant, target))
    return available
