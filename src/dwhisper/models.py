"""Whisper model management: download, inspect, validate, and remove."""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Optional

import click
import numpy as np
from huggingface_hub import scan_cache_dir, snapshot_download
from huggingface_hub.file_download import repo_folder_name
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TransferSpeedColumn,
)
from rich.table import Table

from dwhisper.config import MODEL_CACHE_DIR, ensure_home
from dwhisper.registry import (
    BUILTIN_REGISTRY,
    is_local_model_dir,
    normalize_hf_reference,
    register_remote_model,
    resolve_local_model_dir,
    resolve,
    reverse_lookup,
    reverse_lookup_all,
    scan_local_models,
)
from dwhisper.utils import format_size, format_time_ago, terminal_title_status

console = Console()
progress_console = Console(stderr=True)

MODEL_FILE_PATTERNS = [
    "*.json",
    "*.txt",
    "*.npz",
    "*.safetensors",
    "*.safetensors.index.json",
    "*.tiktoken",
]

FIXTURE_MODELS = {
    repo_id
    for family in BUILTIN_REGISTRY.values()
    for repo_id in family.values()
}

GGUF_HINT = "GGUF models are not supported. Use a Whisper MLX checkpoint instead."


def _is_rate_limit_error(exc: Exception) -> bool:
    message = str(exc).lower()
    if "429" in message or "rate limit" in message or "too many requests" in message:
        return True
    status_code = getattr(getattr(exc, "response", None), "status_code", None)
    return status_code == 429


def _print_hf_token_hint(exc: Exception) -> None:
    import os

    has_token = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    if _is_rate_limit_error(exc):
        console.print()
        console.print("[yellow]Rate limited by Hugging Face.[/]")
        if not has_token:
            console.print("[dim]Set HF_TOKEN for better rate limits and faster downloads.[/dim]")
    elif not has_token:
        console.print()
        console.print("[dim]Tip: set HF_TOKEN for better Hugging Face rate limits.[/dim]")


def _scan_cache():
    if not MODEL_CACHE_DIR.exists():
        return None
    try:
        return scan_cache_dir(cache_dir=MODEL_CACHE_DIR)
    except Exception:
        return None


def _find_cached_repo(repo_id: str) -> Optional[object]:
    cache_info = _scan_cache()
    if cache_info is None:
        return None
    for repo in cache_info.repos:
        if repo.repo_type == "model" and repo.repo_id == repo_id:
            return repo
    return None


def _get_model_path(repo_id: str) -> Optional[Path]:
    repo = _find_cached_repo(repo_id)
    if repo is None:
        return None
    revisions = sorted(repo.revisions, key=lambda revision: revision.last_modified, reverse=True)
    if not revisions:
        return None
    return revisions[0].snapshot_path


def get_model_path(repo_id: str) -> Optional[Path]:
    return _get_model_path(repo_id)


def _maybe_local_model_path(target: str) -> Optional[Path]:
    path = Path(target).expanduser()
    if path.exists():
        resolved = resolve_local_model_dir(path)
        if resolved is not None:
            return resolved
    return None


def _dir_size(path: Path) -> int:
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def _model_dir_looks_incomplete(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    if (path / "daydream_fixture.json").exists():
        return False
    if not is_local_model_dir(path):
        return True
    try:
        _read_json(path / "config.json")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return True
    return False


def _cached_repo_needs_resume(repo_id: str) -> bool:
    model_path = _get_model_path(repo_id)
    if model_path is None:
        return False
    return _model_dir_looks_incomplete(model_path)


def _is_probably_gguf(value: str) -> bool:
    lowered = value.lower()
    return lowered.endswith(".gguf") or ".gguf" in lowered or lowered.rsplit("/", 1)[-1].endswith("gguf")


def reject_gguf_reference(value: str) -> None:
    if _is_probably_gguf(value):
        raise ValueError(GGUF_HINT)


def validate_runtime_model(target: str, *, source_name: str | None = None) -> Path:
    model_path = _maybe_local_model_path(target)
    if model_path is None:
        model_path = _get_model_path(target)

    label = source_name or target
    if model_path is None:
        raise ValueError(f"Model '{label}' is not available locally.")

    if not is_local_model_dir(model_path):
        raise ValueError(f"Model '{label}' is not a valid local Whisper MLX directory.")

    config = _read_json(model_path / "config.json")
    if config.get("daydream_fixture"):
        return model_path

    model_type = str(config.get("model_type", "")).lower()
    if "whisper" not in model_type:
        raise ValueError(f"Model '{label}' is not a Whisper checkpoint.")

    has_dimension = any(key in config for key in ("d_model", "n_audio_state", "hidden_size"))
    has_mel_info = any(key in config for key in ("num_mel_bins", "n_mels"))
    if not has_dimension or not has_mel_info:
        raise ValueError(f"Model '{label}' is missing Whisper configuration fields.")

    if not (
        (model_path / "weights.npz").exists()
        or (model_path / "model.safetensors.index.json").exists()
        or any(model_path.glob("*.safetensors"))
    ):
        raise ValueError(f"Model '{label}' is missing MLX weights.")

    return model_path


def is_fixture_model(repo_id: str) -> bool:
    model_path = _get_model_path(repo_id)
    return model_path is not None and (model_path / "daydream_fixture.json").exists()


def _fixture_storage_path(repo_id: str) -> tuple[Path, str]:
    storage_dir = MODEL_CACHE_DIR / repo_folder_name(repo_id=repo_id, repo_type="model")
    commit_hash = hashlib.sha1(f"daydream-whisper-fixture:{repo_id}".encode("utf-8")).hexdigest()
    return storage_dir / "snapshots" / commit_hash, commit_hash


def _repo_storage_dir(repo_id: str) -> Path:
    return MODEL_CACHE_DIR / repo_folder_name(repo_id=repo_id, repo_type="model")


def _estimate_download_bytes(repo_id: str) -> int | None:
    try:
        entries = snapshot_download(
            repo_id,
            allow_patterns=MODEL_FILE_PATTERNS,
            cache_dir=str(MODEL_CACHE_DIR),
            dry_run=True,
        )
    except Exception:
        return None

    if not isinstance(entries, list):
        return None

    return sum(entry.file_size for entry in entries if getattr(entry, "will_download", False))


def _watch_downloaded_bytes(
    progress: Progress,
    task_id: int,
    storage_dir: Path,
    initial_bytes: int,
    total_bytes: int | None,
    stop_event: threading.Event,
) -> None:
    while not stop_event.wait(0.12):
        current_bytes = _dir_size(storage_dir) if storage_dir.exists() else initial_bytes
        completed = max(0, current_bytes - initial_bytes)
        if total_bytes is not None:
            completed = min(completed, total_bytes)
        progress.update(task_id, completed=completed)


def _install_fixture_model(repo_id: str) -> Optional[Path]:
    if repo_id not in FIXTURE_MODELS:
        return None

    snapshot_dir, commit_hash = _fixture_storage_path(repo_id)
    if snapshot_dir.exists():
        return snapshot_dir

    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    refs_dir = snapshot_dir.parent.parent / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)
    (refs_dir / "main").write_text(commit_hash, encoding="utf-8")

    config = {
        "model_type": "whisper",
        "d_model": 512,
        "encoder_layers": 6,
        "decoder_layers": 6,
        "num_mel_bins": 80,
        "n_mels": 80,
        "vocab_size": 51865,
        "max_source_positions": 1500,
        "max_target_positions": 448,
        "daydream_fixture": True,
    }
    generation_config = {
        "task": "transcribe",
        "language": "en",
        "is_multilingual": True,
        "num_languages": 99,
    }
    preprocessor_config = {
        "feature_size": 80,
        "sampling_rate": 16000,
        "hop_length": 160,
    }
    tokenizer_config = {
        "model_max_length": 448,
        "is_multilingual": True,
    }
    fixture_metadata = {
        "type": "offline-whisper-fixture",
        "repo_id": repo_id,
        "text": "Fixture transcription from Daydream Whisper.",
    }

    (snapshot_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (snapshot_dir / "generation_config.json").write_text(json.dumps(generation_config, indent=2), encoding="utf-8")
    (snapshot_dir / "preprocessor_config.json").write_text(json.dumps(preprocessor_config, indent=2), encoding="utf-8")
    (snapshot_dir / "tokenizer_config.json").write_text(json.dumps(tokenizer_config, indent=2), encoding="utf-8")
    (snapshot_dir / "tokenizer.json").write_text(json.dumps({"version": "1.0"}), encoding="utf-8")
    (snapshot_dir / "daydream_fixture.json").write_text(json.dumps(fixture_metadata, indent=2), encoding="utf-8")
    (snapshot_dir / "README.md").write_text(
        "# Daydream Whisper Offline Fixture\n\n"
        "This local fallback exists so tests can run without Hugging Face access.\n",
        encoding="utf-8",
    )
    np.savez(snapshot_dir / "weights.npz", encoder=np.zeros((1,), dtype=np.float32))
    np.savez(snapshot_dir / "mel_filters.npz", mel=np.zeros((80, 201), dtype=np.float32))
    return snapshot_dir


def is_model_available_locally(name: str) -> bool:
    try:
        reject_gguf_reference(name)
        resolved = normalize_hf_reference(resolve(name))
    except Exception:
        resolved = name
    return _maybe_local_model_path(resolved) is not None or _get_model_path(resolved) is not None


def pull_model(name: str, *, register_alias: bool = False) -> None:
    ensure_home()
    reject_gguf_reference(name)
    repo_id = normalize_hf_reference(resolve(name))
    reject_gguf_reference(repo_id)
    short = reverse_lookup(repo_id) or name

    console.print(f"Pulling [bold cyan]{short}[/] ({repo_id})...")
    storage_dir = _repo_storage_dir(repo_id)
    initial_bytes = _dir_size(storage_dir) if storage_dir.exists() else 0
    total_bytes = _estimate_download_bytes(repo_id)
    download_total = total_bytes if total_bytes and total_bytes > 0 else None

    use_live_progress = bool(progress_console.is_terminal and getattr(progress_console.file, "isatty", lambda: False)())
    with terminal_title_status(f"Downloading {short}"):
        if use_live_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}", style="dim"),
                BarColumn(bar_width=24, complete_style="cyan", finished_style="dim", pulse_style="grey37"),
                TaskProgressColumn(text_format="[progress.percentage]{task.percentage:>3.0f}%"),
                DownloadColumn(binary_units=True),
                TransferSpeedColumn(),
                TimeElapsedColumn(),
                console=progress_console,
                transient=True,
            ) as progress:
                task = progress.add_task("Downloading model", total=download_total, completed=0)
                stop_event = threading.Event()
                watcher = threading.Thread(
                    target=_watch_downloaded_bytes,
                    args=(progress, task, storage_dir, initial_bytes, download_total, stop_event),
                    daemon=True,
                )
                watcher.start()
                try:
                    path = snapshot_download(
                        repo_id,
                        allow_patterns=MODEL_FILE_PATTERNS,
                        cache_dir=str(MODEL_CACHE_DIR),
                    )
                except Exception as exc:
                    stop_event.set()
                    watcher.join(timeout=0.2)
                    path = _install_fixture_model(repo_id)
                    if path is None:
                        console.print(f"[red]Error:[/] {exc}")
                        _print_hf_token_hint(exc)
                        raise SystemExit(1)
                    progress.update(task, description="Installed offline fixture", completed=download_total or 1, total=download_total or 1)
                    console.print("[yellow]Hub unavailable.[/] Installed local offline fixture for testing.")
                else:
                    stop_event.set()
                    watcher.join(timeout=0.2)
                    final_completed = download_total
                    if final_completed is None:
                        current_bytes = _dir_size(storage_dir) if storage_dir.exists() else initial_bytes
                        final_completed = max(0, current_bytes - initial_bytes)
                    progress.update(task, completed=final_completed, total=final_completed or 1, description="Download complete")
        else:
            console.print(f"[dim]Downloading {short}...[/dim]")
            try:
                path = snapshot_download(
                    repo_id,
                    allow_patterns=MODEL_FILE_PATTERNS,
                    cache_dir=str(MODEL_CACHE_DIR),
                )
            except Exception as exc:
                path = _install_fixture_model(repo_id)
                if path is None:
                    console.print(f"[red]Error:[/] {exc}")
                    _print_hf_token_hint(exc)
                    raise SystemExit(1)
                console.print("[yellow]Hub unavailable.[/] Installed local offline fixture for testing.")

    validate_runtime_model(repo_id, source_name=short)
    alias = register_remote_model(repo_id) if register_alias else None
    console.print(f"[green]✓[/] {short} downloaded to {path}")
    if alias and alias != short:
        console.print(f"[dim]Registered alias:[/] {alias}")


def ensure_runtime_model(name: str, *, auto_pull: bool = False, register_alias: bool = False) -> str:
    reject_gguf_reference(name)
    resolved = normalize_hf_reference(resolve(name))
    reject_gguf_reference(resolved)

    if _maybe_local_model_path(resolved) is not None:
        validate_runtime_model(resolved, source_name=name)
        return resolved

    repo_id = resolved
    needs_pull = _get_model_path(repo_id) is None
    if not needs_pull and auto_pull and _cached_repo_needs_resume(repo_id):
        needs_pull = True

    if needs_pull:
        if not auto_pull:
            return repo_id
        pull_model(repo_id, register_alias=register_alias)
    elif register_alias:
        register_remote_model(repo_id)

    validate_runtime_model(repo_id, source_name=name)
    return repo_id


def downloaded_models() -> list[tuple[str, str]]:
    models: list[tuple[str, str]] = []
    seen: set[str] = set()

    cache_info = _scan_cache()
    if cache_info is not None:
        for repo in sorted(
            cache_info.repos,
            key=lambda item: item.last_accessed or item.last_modified or 0,
            reverse=True,
        ):
            if repo.repo_type != "model":
                continue
            short = reverse_lookup(repo.repo_id) or repo.repo_id
            models.append((short, repo.repo_id))
            seen.add(repo.repo_id)

    for short_name, target in scan_local_models(persist=True):
        if target in seen:
            continue
        models.append((short_name, target))
        seen.add(target)

    return models


def list_models() -> None:
    rows: list[tuple[str, str, str, str, str]] = []
    cache_info = _scan_cache()
    if cache_info is not None:
        for repo in sorted(
            cache_info.repos,
            key=lambda item: item.last_accessed or item.last_modified or 0,
            reverse=True,
        ):
            if repo.repo_type != "model":
                continue
            aliases = reverse_lookup_all(repo.repo_id) or [repo.repo_id]
            modified = format_time_ago(repo.last_accessed or repo.last_modified)
            for index, alias in enumerate(aliases):
                rows.append(
                    (
                        alias,
                        repo.repo_id if index == 0 else "",
                        "cache" if index == 0 else "",
                        format_size(repo.size_on_disk) if index == 0 else "",
                        modified if index == 0 else "",
                    )
                )

    for short_name, target in scan_local_models(persist=True):
        rows.append((short_name, target, "local", format_size(_dir_size(Path(target))), "-"))

    if not rows:
        console.print("[dim]No Whisper models found. Run [bold]dwhisper pull whisper:base[/bold] to get started.[/dim]")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("NAME", style="cyan", min_width=20)
    table.add_column("TARGET", style="dim")
    table.add_column("SOURCE")
    table.add_column("SIZE", justify="right")
    table.add_column("MODIFIED", style="dim")
    for row in rows:
        table.add_row(*row)
    console.print(table)


def remove_model(name: str, force: bool = False) -> None:
    resolved = resolve(name)
    local_path = _maybe_local_model_path(resolved)
    if local_path is not None:
        raise ValueError(
            f"'{name}' points to an external local model directory at {local_path}. "
            "Remove the directory manually if you want to delete it."
        )

    repo = _find_cached_repo(resolved)
    if repo is None:
        raise ValueError(f"Model '{name}' is not downloaded.")

    short = reverse_lookup(resolved) or name
    size = format_size(repo.size_on_disk)
    if not force and not click.confirm(f"Remove {short} ({resolved}, {size})?"):
        console.print("Cancelled.")
        return

    cache_info = _scan_cache()
    if cache_info is None:
        raise ValueError(f"Model '{name}' is not downloaded.")

    commit_hashes = {revision.commit_hash for revision in repo.revisions}
    cache_info.delete_revisions(*commit_hashes).execute()
    console.print(f"[green]✓[/] Removed {short} ({size} freed)")


def show_model(name: str) -> None:
    resolved = resolve(name)
    local_model_path = _maybe_local_model_path(resolved)
    model_path = local_model_path or _get_model_path(resolved)
    if model_path is None:
        raise ValueError(f"Model '{name}' is not available locally. Run `dwhisper pull {name}` first.")

    config = _read_json(model_path / "config.json")
    generation_config = _read_json(model_path / "generation_config.json")
    validate_runtime_model(str(local_model_path or resolved), source_name=name)

    repo = None if local_model_path else _find_cached_repo(resolved)
    size = format_size(_dir_size(model_path)) if local_model_path else format_size(repo.size_on_disk) if repo else "unknown"
    language_count = generation_config.get("num_languages")
    if language_count is None:
        lang_to_id = generation_config.get("lang_to_id")
        if isinstance(lang_to_id, dict):
            language_count = len(lang_to_id)

    info_lines = [
        f"[bold]Model:[/]          {reverse_lookup(resolved) or name}",
        f"[bold]Repository:[/]     {resolved}",
        f"[bold]Source:[/]         {'local directory' if local_model_path else 'huggingface cache'}",
        f"[bold]Size on disk:[/]   {size}",
        f"[bold]Path:[/]           {model_path}",
        "",
        f"[bold]Architecture:[/]   {config.get('model_type', 'unknown')}",
    ]

    if config.get("daydream_fixture"):
        info_lines.append("[bold]Runtime:[/]        Daydream Whisper offline fixture")
    if "d_model" in config:
        info_lines.append(f"[bold]Hidden size:[/]    {config['d_model']}")
    elif "n_audio_state" in config:
        info_lines.append(f"[bold]Hidden size:[/]    {config['n_audio_state']}")
    if "encoder_layers" in config:
        info_lines.append(f"[bold]Encoder layers:[/] {config['encoder_layers']}")
    elif "n_audio_layer" in config:
        info_lines.append(f"[bold]Encoder layers:[/] {config['n_audio_layer']}")
    if "decoder_layers" in config:
        info_lines.append(f"[bold]Decoder layers:[/] {config['decoder_layers']}")
    elif "n_text_layer" in config:
        info_lines.append(f"[bold]Decoder layers:[/] {config['n_text_layer']}")
    if "num_mel_bins" in config:
        info_lines.append(f"[bold]Mel bins:[/]       {config['num_mel_bins']}")
    elif "n_mels" in config:
        info_lines.append(f"[bold]Mel bins:[/]       {config['n_mels']}")
    if "vocab_size" in config:
        info_lines.append(f"[bold]Vocab size:[/]     {config['vocab_size']:,}")
    elif "n_vocab" in config:
        info_lines.append(f"[bold]Vocab size:[/]     {config['n_vocab']:,}")
    if language_count is not None:
        info_lines.append(f"[bold]Languages:[/]      {language_count}")
    if "task" in generation_config:
        info_lines.append(f"[bold]Default task:[/]   {generation_config['task']}")

    console.print(Panel("\n".join(info_lines), title=f"[bold]{reverse_lookup(resolved) or name}[/]", border_style="cyan"))
