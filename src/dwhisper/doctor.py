"""Environment diagnostics for ``dwhisper doctor``.

Each check is a small, fast, side-effect-free function that returns a
:class:`DoctorCheck`. The CLI renders them in the order returned by
:func:`run_doctor`. Checks that require heavy imports (mlx, sounddevice)
use lazy imports so a failing environment still gets a useful report.
"""

from __future__ import annotations

import importlib
import shutil
import socket
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from dwhisper.config import (
    DAYDREAM_HOME,
    MODEL_CACHE_DIR,
    get_default_host,
    get_default_port,
    get_default_postprocess_base_url,
    get_default_postprocess_backend,
    get_default_postprocess_enabled,
    get_default_postprocess_model,
    get_local_model_roots,
)


@dataclass(slots=True)
class DoctorCheck:
    name: str
    status: str  # "ok" | "warn" | "error" | "info"
    message: str
    hint: str | None = None


_STATUS_ORDER = {"error": 0, "warn": 1, "ok": 2, "info": 3}


def _try_import(module: str) -> tuple[bool, str | None]:
    try:
        importlib.import_module(module)
    except Exception as exc:  # pragma: no cover - exercised via real import
        return False, str(exc)
    return True, None


def check_python_version() -> DoctorCheck:
    major, minor = sys.version_info[:2]
    version = f"{major}.{minor}.{sys.version_info.micro}"
    if (major, minor) < (3, 11):
        return DoctorCheck(
            name="Python version",
            status="error",
            message=f"Python {version} detected; need 3.11 or newer.",
            hint="Install a recent Python (3.13+ recommended) and recreate the venv.",
        )
    if (major, minor) < (3, 13):
        return DoctorCheck(
            name="Python version",
            status="warn",
            message=f"Python {version}. Works but 3.13+ is recommended for MLX.",
        )
    return DoctorCheck(name="Python version", status="ok", message=f"Python {version}")


def check_mlx_whisper() -> DoctorCheck:
    ok, err = _try_import("mlx_whisper")
    if ok:
        return DoctorCheck(name="mlx-whisper", status="ok", message="installed")
    return DoctorCheck(
        name="mlx-whisper",
        status="error",
        message=f"import failed: {err}",
        hint="Run `pip install mlx-whisper` inside the dwhisper environment.",
    )


def check_mlx_lm() -> DoctorCheck:
    ok, err = _try_import("mlx_lm")
    if ok:
        return DoctorCheck(
            name="mlx-lm (local post-process)",
            status="ok",
            message="installed",
        )
    return DoctorCheck(
        name="mlx-lm (local post-process)",
        status="warn",
        message=f"not installed: {err}",
        hint="Optional. Install with `pip install mlx-lm` to use "
        "postprocess_backend=mlx without an external OpenAI-compat server.",
    )


def check_mlx_audio() -> DoctorCheck:
    ok, err = _try_import("mlx_audio.stt.utils")
    if ok:
        return DoctorCheck(
            name="mlx-audio (non-Whisper ASR)",
            status="ok",
            message="installed",
        )
    return DoctorCheck(
        name="mlx-audio (non-Whisper ASR)",
        status="warn",
        message=f"not installed: {err}",
        hint="Optional. Install with `pip install mlx-audio` to use Qwen3-ASR, "
        "Parakeet, SenseVoice and other non-Whisper speech models.",
    )


def check_mlx_metal() -> DoctorCheck:
    ok, err = _try_import("mlx.core")
    if not ok:
        return DoctorCheck(
            name="MLX Metal device",
            status="warn",
            message=f"mlx.core import failed: {err}",
            hint="Reinstall MLX: `pip install --upgrade mlx mlx-whisper`.",
        )
    try:
        import mlx.core as mx  # type: ignore

        default_device = mx.default_device()
        name = getattr(default_device, "name", None) or str(default_device)
        if "gpu" in name.lower() or "metal" in name.lower():
            return DoctorCheck(
                name="MLX Metal device", status="ok", message=f"active: {name}"
            )
        return DoctorCheck(
            name="MLX Metal device",
            status="warn",
            message=f"default device is {name}; GPU/Metal not selected.",
            hint="Check `MLX_METAL_DEVICE` env var and that this is an Apple Silicon Mac.",
        )
    except Exception as exc:
        return DoctorCheck(
            name="MLX Metal device",
            status="warn",
            message=f"could not query device: {exc}",
        )


def check_sounddevice() -> DoctorCheck:
    ok, err = _try_import("sounddevice")
    if not ok:
        return DoctorCheck(
            name="sounddevice / PortAudio",
            status="error",
            message=f"import failed: {err}",
            hint="Install PortAudio (`brew install portaudio`) then reinstall dwhisper.",
        )
    try:
        import sounddevice as sd  # type: ignore

        sd.query_devices()
    except Exception as exc:
        return DoctorCheck(
            name="sounddevice / PortAudio",
            status="error",
            message=f"query_devices() failed: {exc}",
            hint="PortAudio is probably missing or mis-linked. "
            "Run `brew reinstall portaudio` and reinstall the dwhisper venv.",
        )
    return DoctorCheck(
        name="sounddevice / PortAudio",
        status="ok",
        message="sounddevice ready",
    )


def check_audio_input_devices() -> DoctorCheck:
    try:
        from dwhisper.audio import list_audio_devices
    except Exception as exc:
        return DoctorCheck(
            name="Audio input devices",
            status="error",
            message=f"unable to enumerate: {exc}",
        )
    try:
        devices = list_audio_devices()
    except Exception as exc:
        return DoctorCheck(
            name="Audio input devices",
            status="error",
            message=f"enumeration error: {exc}",
            hint="Grant Microphone permission to your terminal in System Settings → Privacy.",
        )
    if not devices:
        return DoctorCheck(
            name="Audio input devices",
            status="warn",
            message="no input devices visible",
            hint="macOS: System Settings → Privacy & Security → Microphone, "
            "enable access for your terminal (Terminal, iTerm, VS Code, etc.).",
        )
    default = next((d for d in devices if d.get("is_default")), devices[0])
    return DoctorCheck(
        name="Audio input devices",
        status="ok",
        message=f"{len(devices)} device(s); default: {default['name']}",
    )


def check_home_directory() -> DoctorCheck:
    path = Path(DAYDREAM_HOME).expanduser()
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        return DoctorCheck(
            name="DWHISPER_HOME",
            status="error",
            message=f"{path} not writable: {exc}",
        )
    probe = path / ".doctor_write_probe"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except Exception as exc:
        return DoctorCheck(
            name="DWHISPER_HOME",
            status="error",
            message=f"{path} write test failed: {exc}",
        )
    return DoctorCheck(
        name="DWHISPER_HOME",
        status="ok",
        message=str(path),
    )


def check_port_available() -> DoctorCheck:
    host = get_default_host()
    port = get_default_port()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(0.5)
    try:
        sock.bind((host, port))
    except OSError as exc:
        return DoctorCheck(
            name="Default serve port",
            status="warn",
            message=f"{host}:{port} is already in use ({exc.strerror or exc}).",
            hint="Pick another port with `dwhisper serve --port <N>` or set "
            "DWHISPER_PORT in the environment.",
        )
    finally:
        sock.close()
    return DoctorCheck(
        name="Default serve port",
        status="ok",
        message=f"{host}:{port} available",
    )


def check_disk_space() -> DoctorCheck:
    target = MODEL_CACHE_DIR
    try:
        target.mkdir(parents=True, exist_ok=True)
        usage = shutil.disk_usage(target)
    except Exception as exc:
        return DoctorCheck(
            name="Model cache disk",
            status="warn",
            message=f"unable to stat {target}: {exc}",
        )
    free_gb = usage.free / (1024**3)
    if free_gb < 2.0:
        return DoctorCheck(
            name="Model cache disk",
            status="error",
            message=f"only {free_gb:.1f} GB free at {target}",
            hint="Free some space before pulling large Whisper models (turbo ~1.6 GB).",
        )
    if free_gb < 10.0:
        return DoctorCheck(
            name="Model cache disk",
            status="warn",
            message=f"{free_gb:.1f} GB free at {target}",
        )
    return DoctorCheck(
        name="Model cache disk",
        status="ok",
        message=f"{free_gb:.1f} GB free at {target}",
    )


def check_cached_models() -> DoctorCheck:
    roots = get_local_model_roots()
    total = 0
    locations: list[str] = []
    for root in roots:
        expanded = Path(root).expanduser()
        if not expanded.exists():
            continue
        try:
            entries = [
                entry for entry in expanded.iterdir() if entry.is_dir() and not entry.name.startswith(".")
            ]
        except Exception:
            continue
        if entries:
            total += len(entries)
            locations.append(f"{expanded} ({len(entries)})")

    hub = Path(MODEL_CACHE_DIR).expanduser()
    hub_whisper = 0
    if hub.exists():
        try:
            hub_whisper = sum(
                1
                for entry in hub.iterdir()
                if entry.is_dir() and "whisper" in entry.name.lower()
            )
        except Exception:
            hub_whisper = 0
        if hub_whisper:
            locations.append(f"{hub} (whisper: {hub_whisper})")
            total += hub_whisper

    if total == 0:
        return DoctorCheck(
            name="Cached Whisper models",
            status="warn",
            message="none found",
            hint="Pull a default model: `dwhisper pull whisper:large-v3-turbo` "
            "(or `whisper:base` for a lighter first install).",
        )
    return DoctorCheck(
        name="Cached Whisper models",
        status="ok",
        message=f"{total} model dir(s): " + ", ".join(locations),
    )


def check_postprocess() -> DoctorCheck:
    enabled = get_default_postprocess_enabled()
    model = get_default_postprocess_model()
    backend = get_default_postprocess_backend()
    base_url = get_default_postprocess_base_url()

    if not enabled:
        return DoctorCheck(
            name="Post-process defaults",
            status="info",
            message="disabled (set postprocess.enabled=true in ~/.dwhisper/config.yaml to enable)",
        )
    if not model:
        return DoctorCheck(
            name="Post-process defaults",
            status="warn",
            message="enabled but no postprocess_model configured",
            hint="Set `postprocess.model` in config.yaml or pass --postprocess-model.",
        )

    resolved = "http" if backend == "auto" and base_url else ("mlx" if backend == "auto" else backend)
    if resolved == "http":
        if not base_url:
            return DoctorCheck(
                name="Post-process defaults",
                status="warn",
                message="http backend selected but no postprocess_base_url set",
                hint="Set postprocess.base_url to an OpenAI-compatible endpoint "
                "(e.g. http://127.0.0.1:11435/v1).",
            )
        return DoctorCheck(
            name="Post-process defaults",
            status="ok",
            message=f"http · {model} @ {base_url}",
        )
    # mlx
    ok, err = _try_import("mlx_lm")
    if not ok:
        return DoctorCheck(
            name="Post-process defaults",
            status="error",
            message=f"mlx backend configured ({model}) but mlx-lm import failed: {err}",
            hint="Run `pip install mlx-lm` or switch postprocess.backend to http.",
        )
    return DoctorCheck(
        name="Post-process defaults",
        status="ok",
        message=f"mlx · {model}",
    )


DEFAULT_CHECKS: tuple[Callable[[], DoctorCheck], ...] = (
    check_python_version,
    check_mlx_whisper,
    check_mlx_metal,
    check_sounddevice,
    check_audio_input_devices,
    check_home_directory,
    check_port_available,
    check_disk_space,
    check_cached_models,
    check_mlx_lm,
    check_mlx_audio,
    check_postprocess,
)


def run_doctor(
    checks: tuple[Callable[[], DoctorCheck], ...] | None = None,
) -> list[DoctorCheck]:
    """Run the full diagnostic sweep. Individual failures never abort."""

    results: list[DoctorCheck] = []
    for check in checks or DEFAULT_CHECKS:
        try:
            result = check()
        except Exception as exc:  # pragma: no cover - last-ditch safety
            result = DoctorCheck(
                name=getattr(check, "__name__", "check"),
                status="error",
                message=f"check raised: {exc}",
            )
        results.append(result)
    return results


def summarize(results: list[DoctorCheck]) -> dict[str, int]:
    summary = {"ok": 0, "warn": 0, "error": 0, "info": 0}
    for result in results:
        summary[result.status] = summary.get(result.status, 0) + 1
    return summary


def worst_status(results: list[DoctorCheck]) -> str:
    return min((r.status for r in results), key=lambda s: _STATUS_ORDER.get(s, 99), default="ok")
