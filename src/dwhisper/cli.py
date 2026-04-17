from __future__ import annotations

import functools
from pathlib import Path

import click
from click.core import ParameterSource
from rich.console import Console
from rich.table import Table

from dwhisper import __version__
from dwhisper.config import (
    get_default_corrections_path,
    get_default_audio_device,
    get_configured_postprocess_max_tokens,
    get_default_chunk_duration,
    get_default_language,
    get_default_model,
    get_default_output_format,
    get_default_overlap_duration,
    get_default_postprocess_api_key,
    get_default_postprocess_backend,
    get_default_postprocess_base_url,
    get_default_postprocess_enabled,
    get_default_postprocess_mode,
    get_default_postprocess_model,
    get_default_postprocess_timeout,
    get_default_profile,
    get_default_push_to_talk,
    get_default_sample_rate,
    get_default_serve_allow_origin,
    get_default_host,
    get_default_port,
    get_default_silence_threshold,
    get_default_serve_max_concurrency,
    get_default_serve_max_request_bytes,
    get_default_serve_preload,
    get_default_serve_request_timeout,
    get_default_task,
    get_default_vad_sensitivity,
    get_default_vocabulary_path,
    get_default_word_timestamps,
)
from dwhisper.utils import render_transcription_progress

console = Console()
err_console = Console(stderr=True)
DEFAULT_LOCAL_POSTPROCESS_BASE_URL = "http://127.0.0.1:11435/v1"
CONFIGURED_POSTPROCESS_MAX_TOKENS = get_configured_postprocess_max_tokens()
POSTPROCESS_MAX_TOKENS_SHOW_DEFAULT = (
    str(CONFIGURED_POSTPROCESS_MAX_TOKENS)
    if CONFIGURED_POSTPROCESS_MAX_TOKENS is not None
    else "mode-based"
)


def _handle_errors(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValueError as exc:
            err_console.print(f"[red]Error:[/] {exc}")
            raise SystemExit(1)
        except FileNotFoundError as exc:
            err_console.print(f"[red]Error:[/] {exc}")
            raise SystemExit(1)
        except KeyboardInterrupt:
            err_console.print()
            raise SystemExit(0)
        except SystemExit:
            raise
        except Exception as exc:
            err_console.print(f"[red]Error:[/] {exc}")
            raise SystemExit(1)

    return wrapper


def _write_output(content: str, output_path: str | None) -> None:
    if output_path:
        path = Path(output_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        console.print(f"[green]✓[/] Wrote output to {path}")
        return
    click.echo(content, nl=not content.endswith("\n"))


def _parameter_was_explicit(ctx: click.Context, name: str) -> bool:
    try:
        return ctx.get_parameter_source(name) == ParameterSource.COMMANDLINE
    except Exception:
        return False


def _profile_value(ctx: click.Context, name: str, current: object, profile_value: object) -> object:
    if _parameter_was_explicit(ctx, name) or profile_value is None:
        return current
    return profile_value


def _default_file_path(path: Path) -> str | None:
    expanded = path.expanduser()
    return str(expanded) if expanded.exists() else None


def _resolve_postprocess_shortcut(
    *,
    ctx: click.Context,
    profile_transcribe: dict[str, object],
    with_postprocess_model: str | None,
    postprocess: bool,
    postprocess_model: str | None,
    postprocess_base_url: str | None,
    postprocess_api_key: str | None,
    postprocess_mode: str,
    postprocess_prompt: str | None,
    postprocess_timeout: float,
    postprocess_backend: str = "auto",
    postprocess_max_tokens: int | None = None,
) -> tuple[bool, object, object, object, str, object, float, str, int | None]:
    shortcut_model = with_postprocess_model.strip() if with_postprocess_model else None
    shortcut_enabled = bool(shortcut_model)
    resolved_postprocess = bool(
        _profile_value(ctx, "postprocess", postprocess or shortcut_enabled, profile_transcribe.get("postprocess"))
    )
    resolved_postprocess_model = _profile_value(
        ctx,
        "postprocess_model",
        postprocess_model or shortcut_model,
        profile_transcribe.get("postprocess_model"),
    )
    base_url_current = postprocess_base_url
    if shortcut_enabled and not _parameter_was_explicit(ctx, "postprocess_base_url") and not base_url_current:
        base_url_current = DEFAULT_LOCAL_POSTPROCESS_BASE_URL
    resolved_postprocess_base_url = _profile_value(
        ctx,
        "postprocess_base_url",
        base_url_current,
        profile_transcribe.get("postprocess_base_url"),
    )
    resolved_postprocess_api_key = _profile_value(
        ctx,
        "postprocess_api_key",
        postprocess_api_key,
        profile_transcribe.get("postprocess_api_key"),
    )
    resolved_postprocess_mode = str(
        _profile_value(
            ctx,
            "postprocess_mode",
            postprocess_mode.lower(),
            profile_transcribe.get("postprocess_mode") or postprocess_mode.lower(),
        )
    ).lower()
    resolved_postprocess_prompt = _profile_value(
        ctx,
        "postprocess_prompt",
        postprocess_prompt,
        profile_transcribe.get("postprocess_prompt"),
    )
    resolved_postprocess_timeout = float(
        _profile_value(
            ctx,
            "postprocess_timeout",
            postprocess_timeout,
            profile_transcribe.get("postprocess_timeout"),
        )
    )
    resolved_postprocess_backend = str(
        _profile_value(
            ctx,
            "postprocess_backend",
            postprocess_backend.lower(),
            profile_transcribe.get("postprocess_backend") or postprocess_backend.lower(),
        )
    ).lower()
    if resolved_postprocess_backend not in {"auto", "http", "mlx"}:
        resolved_postprocess_backend = "auto"
    resolved_postprocess_max_tokens_value = _profile_value(
        ctx,
        "postprocess_max_tokens",
        postprocess_max_tokens,
        profile_transcribe.get("postprocess_max_tokens"),
    )
    resolved_postprocess_max_tokens = (
        int(resolved_postprocess_max_tokens_value)
        if resolved_postprocess_max_tokens_value is not None
        else None
    )

    return (
        resolved_postprocess,
        resolved_postprocess_model,
        resolved_postprocess_base_url,
        resolved_postprocess_api_key,
        resolved_postprocess_mode,
        resolved_postprocess_prompt,
        resolved_postprocess_timeout,
        resolved_postprocess_backend,
        resolved_postprocess_max_tokens,
    )


def _parse_vocabulary_entries(entries: tuple[str, ...]) -> dict[str, str]:
    vocabulary: dict[str, str] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Invalid vocabulary entry '{entry}'. Use SOURCE=TARGET.")
        source, target = entry.split("=", 1)
        source = source.strip()
        target = target.strip()
        if not source or not target:
            raise ValueError(f"Invalid vocabulary entry '{entry}'. Use SOURCE=TARGET.")
        vocabulary[source] = target
    return vocabulary


def _load_selected_profile(name: str | None):
    from dwhisper.profiles import load_profile

    return load_profile(name)


@click.group()
@click.version_option(version=__version__, prog_name="dwhisper")
def cli():
    """Daydream Whisper — Apple Silicon speech recognition CLI powered by MLX Whisper."""
    from dwhisper.config import ensure_home

    ensure_home()


@cli.command()
@click.argument("model")
@_handle_errors
def pull(model: str) -> None:
    """Download a Whisper model from Hugging Face."""
    from dwhisper.models import pull_model

    pull_model(model)


@cli.command(name="list")
@_handle_errors
def list_command() -> None:
    """List downloaded and discovered Whisper models."""
    from dwhisper.models import list_models

    list_models()


@cli.command()
@click.argument("model")
@click.option("--force", "-f", is_flag=True, help="Skip confirmation prompt.")
@_handle_errors
def rm(model: str, force: bool) -> None:
    """Remove a downloaded cached Whisper model."""
    from dwhisper.models import remove_model

    remove_model(model, force=force)


@cli.command()
@click.argument("model")
@_handle_errors
def show(model: str) -> None:
    """Show local metadata for a Whisper model."""
    from dwhisper.models import show_model

    show_model(model)


@cli.command(name="models")
@_handle_errors
def available_models() -> None:
    """List all available built-in and registered model names."""
    from dwhisper.registry import list_available

    table = Table(show_header=True, header_style="bold")
    table.add_column("NAME", style="cyan")
    table.add_column("TARGET", style="dim")
    table.add_column("SOURCE")

    for family, variant, target in list_available():
        short_name = family if variant == "default" else f"{family}:{variant}"
        source = "local" if Path(target).expanduser().exists() else "huggingface"
        table.add_row(short_name, target, source)

    console.print(table)


@cli.command(name="profiles")
@_handle_errors
def list_profiles_command() -> None:
    """List named transcription profiles."""
    from dwhisper.profiles import load_profile_store

    store = load_profile_store()
    profiles = store.list()
    if not profiles:
        console.print(
            "[dim]No profiles found. Create ~/.dwhisper/profiles.yaml or ~/.dwhisper/profiles/*.yaml to add reusable speech presets.[/dim]"
        )
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("NAME", style="cyan")
    table.add_column("MODEL")
    table.add_column("TASK")
    table.add_column("LANG")
    table.add_column("FORMAT")
    table.add_column("DESCRIPTION", style="dim")

    for profile in profiles:
        task = profile.transcribe.get("task", "")
        language = profile.transcribe.get("language", "")
        marker = " (default)" if profile.name == store.default_profile else ""
        table.add_row(
            f"{profile.name}{marker}",
            str(profile.model or ""),
            str(task or ""),
            str(language or ""),
            str(profile.output_format or profile.listen.get("output_format", "")),
            profile.description or "",
        )

    console.print(table)


@cli.command()
@_handle_errors
def devices() -> None:
    """List available microphone input devices."""
    from dwhisper.audio import list_audio_devices

    device_rows = list_audio_devices()
    if not device_rows:
        console.print("[dim]No audio input devices found.[/dim]")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("INDEX", justify="right")
    table.add_column("NAME", style="cyan")
    table.add_column("INPUTS", justify="right")
    table.add_column("RATE", justify="right")
    table.add_column("DEFAULT")
    for device in device_rows:
        table.add_row(
            str(device["index"]),
            device["name"],
            str(device["max_input_channels"]),
            str(device["default_samplerate"]),
            "yes" if device["is_default"] else "",
        )
    console.print(table)


@cli.command()
@click.argument("audio_file", type=click.Path(exists=True, dir_okay=False, path_type=str))
@click.option("--profile", default=None, help="Named transcription profile from ~/.dwhisper/profiles.yaml or ~/.dwhisper/profiles/*.yaml.")
@click.option("--model", "-m", default=get_default_model(), show_default=True, help="Whisper model to use.")
@click.option("--language", "-l", default=get_default_language(), help="Language code. Default: auto-detect.")
@click.option(
    "--task",
    type=click.Choice(["transcribe", "translate"], case_sensitive=False),
    default=get_default_task(),
    show_default=True,
    help="Recognition task.",
)
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["text", "json", "srt", "vtt"], case_sensitive=False),
    default=get_default_output_format(),
    show_default=True,
    help="Output format.",
)
@click.option(
    "--word-timestamps/--no-word-timestamps",
    default=get_default_word_timestamps(),
    show_default=True,
    help="Include word-level timestamps when supported by the model.",
)
@click.option("--initial-prompt", type=str, default=None, help="Optional initial prompt for domain-specific context.")
@click.option("--temperature", type=float, default=0.0, show_default=True, help="Sampling temperature for decoding.")
@click.option("--beam-size", type=int, default=None, help="Beam size override for decoding.")
@click.option("--best-of", type=int, default=None, help="Best-of sample count for non-beam decoding.")
@click.option("--no-speech-threshold", type=float, default=None, help="Drop segments above this no-speech threshold.")
@click.option("--hotword", "hotwords", multiple=True, help="Bias decoding toward a hotword. Repeat for multiple values.")
@click.option("--vocabulary-entry", "vocabulary_entries", multiple=True, help="Vocabulary correction entry in SOURCE=TARGET form.")
@click.option("--corrections-file", type=click.Path(exists=True, dir_okay=False, path_type=str), default=None, help="Correction rules YAML file.")
@click.option("--vocabulary-file", type=click.Path(exists=True, dir_okay=False, path_type=str), default=None, help="Vocabulary YAML file.")
@click.option(
    "--post-model",
    "--with-postprocess-model",
    "with_postprocess_model",
    default=None,
    help="Shortcut for optional local post-processing. Implies --postprocess and defaults the base URL to http://127.0.0.1:11435/v1.",
)
@click.option(
    "--postprocess/--no-postprocess",
    default=get_default_postprocess_enabled(),
    show_default=True,
    help="Send transcript through an optional local text or multimodal post-processor after Whisper.",
)
@click.option("--postprocess-model", default=get_default_postprocess_model(), help="Local text or multimodal model name for post-processing.")
@click.option("--post-url", "--postprocess-base-url", "postprocess_base_url", default=get_default_postprocess_base_url(), help="OpenAI-compatible local base URL for the post-process model.")
@click.option("--post-key", "--postprocess-api-key", "postprocess_api_key", default=get_default_postprocess_api_key(), show_default=False, help="API key sent to the local post-process endpoint.")
@click.option(
    "--post-mode",
    "--postprocess-mode",
    "postprocess_mode",
    type=click.Choice(["clean", "summary", "meeting-notes", "speaker-format"], case_sensitive=False),
    default=get_default_postprocess_mode(),
    show_default=True,
    help="How the local text or multimodal model should rewrite the transcript.",
)
@click.option("--post-prompt", "--postprocess-prompt", "postprocess_prompt", default=None, help="Custom prompt template for post-processing. Supports {transcript}, {language}, and {mode}.")
@click.option("--post-timeout", "--postprocess-timeout", "postprocess_timeout", type=float, default=get_default_postprocess_timeout(), show_default=True, help="Timeout in seconds for the optional local post-process request.")
@click.option(
    "--post-backend", "--postprocess-backend", "postprocess_backend",
    type=click.Choice(["auto", "http", "mlx"], case_sensitive=False),
    default=get_default_postprocess_backend(),
    show_default=True,
    help="Post-process backend. 'mlx' runs a local MLX LM in-process; 'http' calls an OpenAI-compatible server; 'auto' picks http when a base URL is set, otherwise mlx.",
)
@click.option(
    "--post-max-tokens", "--postprocess-max-tokens", "postprocess_max_tokens",
    type=int,
    default=CONFIGURED_POSTPROCESS_MAX_TOKENS,
    show_default=POSTPROCESS_MAX_TOKENS_SHOW_DEFAULT,
    help="Max generated tokens for MLX post-processing.",
)
@click.option("--output", "-o", "output_path", type=click.Path(dir_okay=False, path_type=str), default=None, help="Write output to a file.")
@click.option("--verbose", "-v", is_flag=True, help="Show timing and progress details.")
@click.pass_context
@_handle_errors
def transcribe(
    ctx: click.Context,
    audio_file: str,
    profile: str | None,
    model: str,
    language: str | None,
    task: str,
    output_format: str,
    word_timestamps: bool,
    initial_prompt: str | None,
    temperature: float,
    beam_size: int | None,
    best_of: int | None,
    no_speech_threshold: float | None,
    hotwords: tuple[str, ...],
    vocabulary_entries: tuple[str, ...],
    corrections_file: str | None,
    vocabulary_file: str | None,
    with_postprocess_model: str | None,
    postprocess: bool,
    postprocess_model: str | None,
    postprocess_base_url: str | None,
    postprocess_api_key: str,
    postprocess_mode: str,
    postprocess_prompt: str | None,
    postprocess_timeout: float,
    postprocess_backend: str,
    postprocess_max_tokens: int | None,
    output_path: str | None,
    verbose: bool,
) -> None:
    """Transcribe an audio file with MLX Whisper."""
    from dwhisper.transcriber import TranscribeOptions, WhisperTranscriber

    selected_profile = _load_selected_profile(profile or get_default_profile())
    profile_transcribe = selected_profile.transcribe if selected_profile is not None else {}
    model = str(_profile_value(ctx, "model", model, getattr(selected_profile, "model", None)))
    output_format = str(
        _profile_value(ctx, "output_format", output_format, getattr(selected_profile, "output_format", None))
    )
    hotword_values = list(hotwords) if _parameter_was_explicit(ctx, "hotwords") else list(profile_transcribe.get("hotwords", []) or [])
    vocabulary = (
        _parse_vocabulary_entries(vocabulary_entries)
        if _parameter_was_explicit(ctx, "vocabulary_entries")
        else dict(profile_transcribe.get("vocabulary", {}) or {})
    )
    resolved_corrections_file = corrections_file
    if not _parameter_was_explicit(ctx, "corrections_file"):
        resolved_corrections_file = profile_transcribe.get("corrections_path") or _default_file_path(get_default_corrections_path())
    resolved_vocabulary_file = vocabulary_file
    if not _parameter_was_explicit(ctx, "vocabulary_file"):
        resolved_vocabulary_file = profile_transcribe.get("vocabulary_path") or _default_file_path(get_default_vocabulary_path())
    (
        resolved_postprocess,
        resolved_postprocess_model,
        resolved_postprocess_base_url,
        resolved_postprocess_api_key,
        resolved_postprocess_mode,
        resolved_postprocess_prompt,
        resolved_postprocess_timeout,
        resolved_postprocess_backend,
        resolved_postprocess_max_tokens,
    ) = _resolve_postprocess_shortcut(
        ctx=ctx,
        profile_transcribe=profile_transcribe,
        with_postprocess_model=with_postprocess_model,
        postprocess=postprocess,
        postprocess_model=postprocess_model,
        postprocess_base_url=postprocess_base_url,
        postprocess_api_key=postprocess_api_key,
        postprocess_mode=postprocess_mode,
        postprocess_prompt=postprocess_prompt,
        postprocess_timeout=postprocess_timeout,
        postprocess_backend=postprocess_backend,
        postprocess_max_tokens=postprocess_max_tokens,
    )

    options = TranscribeOptions(
        profile=selected_profile.name if selected_profile is not None else profile,
        language=_profile_value(ctx, "language", language, profile_transcribe.get("language")),
        task=str(_profile_value(ctx, "task", task.lower(), profile_transcribe.get("task") or task.lower())).lower(),
        word_timestamps=bool(_profile_value(ctx, "word_timestamps", word_timestamps, profile_transcribe.get("word_timestamps"))),
        temperature=float(_profile_value(ctx, "temperature", temperature, profile_transcribe.get("temperature") if profile_transcribe.get("temperature") is not None else temperature)),
        initial_prompt=_profile_value(ctx, "initial_prompt", initial_prompt, profile_transcribe.get("initial_prompt")),
        verbose=verbose,
        beam_size=_profile_value(ctx, "beam_size", beam_size, profile_transcribe.get("beam_size")),
        best_of=_profile_value(ctx, "best_of", best_of, profile_transcribe.get("best_of")),
        no_speech_threshold=_profile_value(
            ctx,
            "no_speech_threshold",
            no_speech_threshold,
            profile_transcribe.get("no_speech_threshold"),
        ),
        hotwords=hotword_values,
        vocabulary=vocabulary,
        correction=profile_transcribe.get("correction"),
        corrections_path=resolved_corrections_file,
        vocabulary_path=resolved_vocabulary_file,
        postprocess=resolved_postprocess,
        postprocess_model=resolved_postprocess_model,
        postprocess_base_url=resolved_postprocess_base_url,
        postprocess_api_key=resolved_postprocess_api_key,
        postprocess_mode=resolved_postprocess_mode,
        postprocess_prompt=resolved_postprocess_prompt,
        postprocess_timeout=resolved_postprocess_timeout,
        postprocess_backend=resolved_postprocess_backend,
        postprocess_max_tokens=resolved_postprocess_max_tokens,
    )

    if verbose:
        err_console.print(render_transcription_progress(Path(audio_file).name))

    transcriber = WhisperTranscriber(model)
    try:
        result = transcriber.transcribe_file(audio_file, options=options)
        _write_output(result.render(output_format.lower()), output_path)

        if verbose:
            err_console.print(
                f"[dim]Processed {result.duration:.1f}s of audio in {result.processing_time:.2f}s "
                f"using {model}.[/dim]"
            )
    finally:
        transcriber.close()


@cli.command()
@click.option("--profile", default=None, help="Named transcription profile from ~/.dwhisper/profiles.yaml or ~/.dwhisper/profiles/*.yaml.")
@click.option("--model", "-m", default=get_default_model(), show_default=True, help="Whisper model to use.")
@click.option("--language", "-l", default=get_default_language(), help="Language code. Default: auto-detect.")
@click.option(
    "--task",
    type=click.Choice(["transcribe", "translate"], case_sensitive=False),
    default=get_default_task(),
    show_default=True,
    help="Recognition task.",
)
@click.option("--device", "-d", default=get_default_audio_device(), help="Audio input device index or name.")
@click.option("--sample-rate", type=int, default=get_default_sample_rate(), show_default=True, help="Microphone sample rate.")
@click.option("--chunk-duration", type=float, default=get_default_chunk_duration(), show_default=True, help="Seconds of speech to accumulate before forced flush.")
@click.option("--overlap", type=float, default=get_default_overlap_duration(), show_default=True, help="Overlap between chunks in seconds.")
@click.option("--silence-threshold", type=float, default=get_default_silence_threshold(), show_default=True, help="Silence duration that finalizes a segment.")
@click.option("--vad-sensitivity", type=float, default=get_default_vad_sensitivity(), show_default=True, help="RMS VAD sensitivity from 0.0 to 1.0.")
@click.option(
    "--push-to-talk/--no-push-to-talk",
    default=get_default_push_to_talk(),
    show_default=True,
    help="Gate microphone capture with the keyboard instead of VAD.",
)
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["text", "json", "srt", "vtt"], case_sensitive=False),
    default=get_default_output_format(),
    show_default=True,
    help="Output format for recognized segments.",
)
@click.option(
    "--word-timestamps/--no-word-timestamps",
    default=get_default_word_timestamps(),
    show_default=True,
    help="Include word-level timestamps when supported by the model.",
)
@click.option("--initial-prompt", type=str, default=None, help="Optional initial prompt for domain-specific context.")
@click.option("--temperature", type=float, default=0.0, show_default=True, help="Sampling temperature for decoding.")
@click.option("--beam-size", type=int, default=None, help="Beam size override for decoding.")
@click.option("--best-of", type=int, default=None, help="Best-of sample count for non-beam decoding.")
@click.option("--no-speech-threshold", type=float, default=None, help="Drop segments above this no-speech threshold.")
@click.option("--hotword", "hotwords", multiple=True, help="Bias decoding toward a hotword. Repeat for multiple values.")
@click.option("--vocabulary-entry", "vocabulary_entries", multiple=True, help="Vocabulary correction entry in SOURCE=TARGET form.")
@click.option("--corrections-file", type=click.Path(exists=True, dir_okay=False, path_type=str), default=None, help="Correction rules YAML file.")
@click.option("--vocabulary-file", type=click.Path(exists=True, dir_okay=False, path_type=str), default=None, help="Vocabulary YAML file.")
@click.option(
    "--post-model",
    "--with-postprocess-model",
    "with_postprocess_model",
    default=None,
    help="Shortcut for optional local post-processing. Implies --postprocess and defaults the base URL to http://127.0.0.1:11435/v1.",
)
@click.option(
    "--postprocess/--no-postprocess",
    default=get_default_postprocess_enabled(),
    show_default=True,
    help="Send finalized transcript text through an optional local text or multimodal post-processor.",
)
@click.option("--postprocess-model", default=get_default_postprocess_model(), help="Local text or multimodal model name for post-processing.")
@click.option("--post-url", "--postprocess-base-url", "postprocess_base_url", default=get_default_postprocess_base_url(), help="OpenAI-compatible local base URL for the post-process model.")
@click.option("--post-key", "--postprocess-api-key", "postprocess_api_key", default=get_default_postprocess_api_key(), show_default=False, help="API key sent to the local post-process endpoint.")
@click.option(
    "--post-mode",
    "--postprocess-mode",
    "postprocess_mode",
    type=click.Choice(["clean", "summary", "meeting-notes", "speaker-format"], case_sensitive=False),
    default=get_default_postprocess_mode(),
    show_default=True,
    help="How the local text or multimodal model should rewrite finalized transcript text.",
)
@click.option("--post-prompt", "--postprocess-prompt", "postprocess_prompt", default=None, help="Custom prompt template for post-processing. Supports {transcript}, {language}, and {mode}.")
@click.option("--post-timeout", "--postprocess-timeout", "postprocess_timeout", type=float, default=get_default_postprocess_timeout(), show_default=True, help="Timeout in seconds for the optional local post-process request.")
@click.option(
    "--post-backend", "--postprocess-backend", "postprocess_backend",
    type=click.Choice(["auto", "http", "mlx"], case_sensitive=False),
    default=get_default_postprocess_backend(),
    show_default=True,
    help="Post-process backend for live voice input. 'mlx' keeps the model loaded locally for low-latency correction; 'http' calls an OpenAI-compatible server.",
)
@click.option(
    "--post-max-tokens", "--postprocess-max-tokens", "postprocess_max_tokens",
    type=int,
    default=CONFIGURED_POSTPROCESS_MAX_TOKENS,
    show_default=POSTPROCESS_MAX_TOKENS_SHOW_DEFAULT,
    help="Max generated tokens for MLX post-processing (lower = faster per chunk).",
)
@click.option("--verbose", "-v", is_flag=True, help="Show additional realtime status.")
@click.pass_context
@_handle_errors
def listen(
    ctx: click.Context,
    profile: str | None,
    model: str,
    language: str | None,
    task: str,
    device: str | None,
    sample_rate: int,
    chunk_duration: float,
    overlap: float,
    silence_threshold: float,
    vad_sensitivity: float,
    push_to_talk: bool,
    output_format: str,
    word_timestamps: bool,
    initial_prompt: str | None,
    temperature: float,
    beam_size: int | None,
    best_of: int | None,
    no_speech_threshold: float | None,
    hotwords: tuple[str, ...],
    vocabulary_entries: tuple[str, ...],
    corrections_file: str | None,
    vocabulary_file: str | None,
    with_postprocess_model: str | None,
    postprocess: bool,
    postprocess_model: str | None,
    postprocess_base_url: str | None,
    postprocess_api_key: str,
    postprocess_mode: str,
    postprocess_prompt: str | None,
    postprocess_timeout: float,
    postprocess_backend: str,
    postprocess_max_tokens: int | None,
    verbose: bool,
) -> None:
    """Transcribe microphone input in realtime."""
    from dwhisper.realtime import RealtimeConfig, run_listen_session
    from dwhisper.transcriber import TranscribeOptions

    selected_profile = _load_selected_profile(profile or get_default_profile())
    profile_transcribe = selected_profile.transcribe if selected_profile is not None else {}
    profile_listen = selected_profile.listen if selected_profile is not None else {}
    model = str(_profile_value(ctx, "model", model, getattr(selected_profile, "model", None)))
    output_format = str(
        _profile_value(ctx, "output_format", output_format, getattr(selected_profile, "output_format", None) or profile_listen.get("output_format"))
    )
    hotword_values = list(hotwords) if _parameter_was_explicit(ctx, "hotwords") else list(profile_transcribe.get("hotwords", []) or [])
    vocabulary = (
        _parse_vocabulary_entries(vocabulary_entries)
        if _parameter_was_explicit(ctx, "vocabulary_entries")
        else dict(profile_transcribe.get("vocabulary", {}) or {})
    )
    resolved_corrections_file = corrections_file
    if not _parameter_was_explicit(ctx, "corrections_file"):
        resolved_corrections_file = profile_transcribe.get("corrections_path") or _default_file_path(get_default_corrections_path())
    resolved_vocabulary_file = vocabulary_file
    if not _parameter_was_explicit(ctx, "vocabulary_file"):
        resolved_vocabulary_file = profile_transcribe.get("vocabulary_path") or _default_file_path(get_default_vocabulary_path())
    (
        resolved_postprocess,
        resolved_postprocess_model,
        resolved_postprocess_base_url,
        resolved_postprocess_api_key,
        resolved_postprocess_mode,
        resolved_postprocess_prompt,
        resolved_postprocess_timeout,
        resolved_postprocess_backend,
        resolved_postprocess_max_tokens,
    ) = _resolve_postprocess_shortcut(
        ctx=ctx,
        profile_transcribe=profile_transcribe,
        with_postprocess_model=with_postprocess_model,
        postprocess=postprocess,
        postprocess_model=postprocess_model,
        postprocess_base_url=postprocess_base_url,
        postprocess_api_key=postprocess_api_key,
        postprocess_mode=postprocess_mode,
        postprocess_prompt=postprocess_prompt,
        postprocess_timeout=postprocess_timeout,
        postprocess_backend=postprocess_backend,
        postprocess_max_tokens=postprocess_max_tokens,
    )
    options = TranscribeOptions(
        profile=selected_profile.name if selected_profile is not None else profile,
        language=_profile_value(ctx, "language", language, profile_transcribe.get("language")),
        task=str(_profile_value(ctx, "task", task.lower(), profile_transcribe.get("task") or task.lower())).lower(),
        word_timestamps=bool(_profile_value(ctx, "word_timestamps", word_timestamps, profile_transcribe.get("word_timestamps"))),
        temperature=float(_profile_value(ctx, "temperature", temperature, profile_transcribe.get("temperature") if profile_transcribe.get("temperature") is not None else temperature)),
        initial_prompt=_profile_value(ctx, "initial_prompt", initial_prompt, profile_transcribe.get("initial_prompt")),
        verbose=verbose,
        beam_size=_profile_value(ctx, "beam_size", beam_size, profile_transcribe.get("beam_size")),
        best_of=_profile_value(ctx, "best_of", best_of, profile_transcribe.get("best_of")),
        no_speech_threshold=_profile_value(
            ctx,
            "no_speech_threshold",
            no_speech_threshold,
            profile_transcribe.get("no_speech_threshold"),
        ),
        hotwords=hotword_values,
        vocabulary=vocabulary,
        correction=profile_transcribe.get("correction"),
        corrections_path=resolved_corrections_file,
        vocabulary_path=resolved_vocabulary_file,
        postprocess=resolved_postprocess,
        postprocess_model=resolved_postprocess_model,
        postprocess_base_url=resolved_postprocess_base_url,
        postprocess_api_key=resolved_postprocess_api_key,
        postprocess_mode=resolved_postprocess_mode,
        postprocess_prompt=resolved_postprocess_prompt,
        postprocess_timeout=resolved_postprocess_timeout,
        postprocess_backend=resolved_postprocess_backend,
        postprocess_max_tokens=resolved_postprocess_max_tokens,
    )
    realtime_config = RealtimeConfig(
        sample_rate=int(_profile_value(ctx, "sample_rate", sample_rate, profile_listen.get("sample_rate"))),
        chunk_duration=float(_profile_value(ctx, "chunk_duration", chunk_duration, profile_listen.get("chunk_duration"))),
        overlap_duration=float(_profile_value(ctx, "overlap", overlap, profile_listen.get("overlap_duration"))),
        silence_threshold=float(_profile_value(ctx, "silence_threshold", silence_threshold, profile_listen.get("silence_threshold"))),
        vad_sensitivity=float(_profile_value(ctx, "vad_sensitivity", vad_sensitivity, profile_listen.get("vad_sensitivity"))),
        device=_profile_value(ctx, "device", device, profile_listen.get("device")),
        push_to_talk=bool(_profile_value(ctx, "push_to_talk", push_to_talk, profile_listen.get("push_to_talk"))),
    )

    run_listen_session(
        model=model,
        transcribe_options=options,
        realtime_config=realtime_config,
        output_format=output_format.lower(),
        verbose=verbose,
        console=console,
    )


@cli.command()
@click.option("--model", "-m", default=get_default_model(), show_default=True, help="Default Whisper model for API requests.")
@click.option("--host", default=get_default_host(), show_default=True, help="Bind address.")
@click.option("--port", "-p", type=int, default=get_default_port(), show_default=True, help="Port number.")
@click.option(
    "--auto-pull/--no-auto-pull",
    default=False,
    show_default=True,
    help="Pull the default model automatically if it is missing.",
)
@click.option(
    "--max-concurrency",
    type=int,
    default=get_default_serve_max_concurrency(),
    show_default=True,
    help="Maximum simultaneous transcription requests.",
)
@click.option(
    "--request-timeout",
    type=float,
    default=get_default_serve_request_timeout(),
    show_default=True,
    help="Seconds to wait for a free worker slot and transcription worker completion.",
)
@click.option(
    "--max-request-bytes",
    type=int,
    default=get_default_serve_max_request_bytes(),
    show_default=True,
    help="Maximum accepted HTTP request body size in bytes.",
)
@click.option(
    "--preload/--no-preload",
    default=get_default_serve_preload(),
    show_default=True,
    help="Preload the default model worker at startup for lower first-request latency.",
)
@click.option(
    "--allow-origin",
    default=get_default_serve_allow_origin(),
    show_default=True,
    help="CORS Access-Control-Allow-Origin value for browser-based local integrations.",
)
@click.option(
    "--post-model",
    "--with-postprocess-model",
    "with_postprocess_model",
    default=None,
    help="Shortcut to enable a local text or multimodal post-process model for all requests handled by this server.",
)
@click.option(
    "--postprocess/--no-postprocess",
    default=get_default_postprocess_enabled(),
    show_default=True,
    help="Enable default transcript post-processing for requests handled by this server.",
)
@click.option("--postprocess-model", default=get_default_postprocess_model(), help="Default local text or multimodal model used for post-processing.")
@click.option("--post-url", "--postprocess-base-url", "postprocess_base_url", default=get_default_postprocess_base_url(), help="Default OpenAI-compatible base URL used for transcript post-processing.")
@click.option("--post-key", "--postprocess-api-key", "postprocess_api_key", default=get_default_postprocess_api_key(), show_default=False, help="Default API key sent to the local post-process endpoint.")
@click.option(
    "--post-mode",
    "--postprocess-mode",
    "postprocess_mode",
    type=click.Choice(["clean", "summary", "meeting-notes", "speaker-format"], case_sensitive=False),
    default=get_default_postprocess_mode(),
    show_default=True,
    help="Default post-process mode applied by the local text or multimodal model.",
)
@click.option("--post-prompt", "--postprocess-prompt", "postprocess_prompt", default=None, help="Default custom prompt template for post-processing requests.")
@click.option("--post-timeout", "--postprocess-timeout", "postprocess_timeout", type=float, default=get_default_postprocess_timeout(), show_default=True, help="Default timeout in seconds for the local post-process request.")
@click.option(
    "--post-backend", "--postprocess-backend", "postprocess_backend",
    type=click.Choice(["auto", "http", "mlx"], case_sensitive=False),
    default=get_default_postprocess_backend(),
    show_default=True,
    help="Default backend for transcript post-processing. 'mlx' runs a local MLX LM in-process; 'http' uses an OpenAI-compatible endpoint; 'auto' picks http when a base URL is set, otherwise mlx.",
)
@click.option(
    "--post-max-tokens", "--postprocess-max-tokens", "postprocess_max_tokens",
    type=int,
    default=CONFIGURED_POSTPROCESS_MAX_TOKENS,
    show_default=POSTPROCESS_MAX_TOKENS_SHOW_DEFAULT,
    help="Default max generated tokens for MLX post-processing.",
)
@_handle_errors
def serve(
    model: str,
    host: str,
    port: int,
    auto_pull: bool,
    max_concurrency: int,
    request_timeout: float,
    max_request_bytes: int,
    preload: bool,
    allow_origin: str,
    with_postprocess_model: str | None,
    postprocess: bool,
    postprocess_model: str | None,
    postprocess_base_url: str | None,
    postprocess_api_key: str,
    postprocess_mode: str,
    postprocess_prompt: str | None,
    postprocess_timeout: float,
    postprocess_backend: str,
    postprocess_max_tokens: int | None,
) -> None:
    """Serve a speech-to-text HTTP API for local dictation and transcription apps."""
    from dwhisper.server import start_server

    shortcut_model = with_postprocess_model.strip() if with_postprocess_model else None
    resolved_postprocess = bool(postprocess or shortcut_model)
    resolved_postprocess_model = postprocess_model or shortcut_model
    resolved_postprocess_base_url = (
        postprocess_base_url
        or (DEFAULT_LOCAL_POSTPROCESS_BASE_URL if shortcut_model else None)
    )
    postprocess_defaults = {
        "postprocess": resolved_postprocess,
        "postprocess_model": resolved_postprocess_model,
        "postprocess_base_url": resolved_postprocess_base_url,
        "postprocess_api_key": postprocess_api_key,
        "postprocess_mode": postprocess_mode.lower(),
        "postprocess_prompt": postprocess_prompt,
        "postprocess_timeout": postprocess_timeout,
        "postprocess_backend": postprocess_backend.lower(),
    }
    if postprocess_max_tokens is not None:
        postprocess_defaults["postprocess_max_tokens"] = int(postprocess_max_tokens)

    start_server(
        model=model,
        host=host,
        port=port,
        auto_pull=auto_pull,
        max_concurrency=max_concurrency,
        request_timeout=request_timeout,
        max_request_bytes=max_request_bytes,
        preload=preload,
        allow_origin=allow_origin,
        postprocess_defaults=postprocess_defaults,
    )


@cli.command()
@click.option(
    "--strict/--no-strict",
    default=False,
    show_default=True,
    help="Exit non-zero when any check reports a warning or error.",
)
@_handle_errors
def doctor(strict: bool) -> None:
    """Check environment readiness: Python, MLX, PortAudio, cache, postprocess."""
    from dwhisper.doctor import run_doctor, summarize, worst_status

    results = run_doctor()

    style_map = {
        "ok": ("green", "[OK]"),
        "warn": ("yellow", "[WARN]"),
        "error": ("red", "[FAIL]"),
        "info": ("dim", "[INFO]"),
    }

    table = Table(show_header=True, header_style="bold")
    table.add_column("", style="bold", width=7)
    table.add_column("CHECK", style="cyan")
    table.add_column("DETAIL")

    for check in results:
        color, marker = style_map.get(check.status, ("", "[?]"))
        table.add_row(
            f"[{color}]{marker}[/{color}]" if color else marker,
            check.name,
            check.message,
        )
    console.print(table)

    for check in results:
        if check.hint and check.status in {"warn", "error"}:
            console.print(f"[dim]↳[/dim] [bold]{check.name}[/]: {check.hint}")

    summary = summarize(results)
    console.print(
        f"[dim]Summary:[/dim] [green]{summary['ok']} ok[/green] · "
        f"[yellow]{summary['warn']} warn[/yellow] · "
        f"[red]{summary['error']} error[/red] · "
        f"[dim]{summary['info']} info[/dim]"
    )

    overall = worst_status(results)
    if overall == "error" or (strict and overall in {"error", "warn"}):
        raise SystemExit(1)


cli.add_command(transcribe, name="run")
