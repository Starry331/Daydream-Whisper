from __future__ import annotations

import re
import shutil
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

from rich.console import Console
from rich.text import Text

DEFAULT_TERMINAL_TITLE = "Daydream Whisper"

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def format_size(size_bytes: int) -> str:
    value = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < 1024.0:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} PB"


def format_time_ago(dt: datetime | float | int) -> str:
    now = datetime.now(timezone.utc)
    if isinstance(dt, (int, float)):
        value = datetime.fromtimestamp(dt, tz=timezone.utc)
    else:
        value = dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)

    delta = now - value
    seconds = max(0, int(delta.total_seconds()))
    if seconds < 60:
        return "just now"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes} min ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours} hours ago"
    days = hours // 24
    if days < 30:
        return f"{days} days ago"
    months = days // 30
    return f"{months} months ago"


def is_interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def _display_width(text: str) -> int:
    return len(_strip_ansi(text))


def _fit_display_width(text: str, width: int) -> str:
    plain = _strip_ansi(text)
    if len(plain) <= width:
        return plain + (" " * (width - len(plain)))
    return plain[: max(0, width - 1)] + "…"


def set_terminal_title(title: str) -> None:
    if not sys.stdout.isatty():
        return
    sys.stdout.write(f"\033]0;{title}\007")
    sys.stdout.flush()


@contextmanager
def terminal_title_status(label: str):
    previous = DEFAULT_TERMINAL_TITLE
    if sys.stdout.isatty():
        set_terminal_title(label)
    try:
        yield
    finally:
        if sys.stdout.isatty():
            set_terminal_title(previous)


def format_timestamp(seconds: float, *, srt: bool = False) -> str:
    total_millis = max(0, int(round(seconds * 1000.0)))
    hours, remainder = divmod(total_millis, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    separator = "," if srt else "."
    return f"{hours:02}:{minutes:02}:{secs:02}{separator}{millis:03}"


def render_transcript_line(
    text: str,
    *,
    start: float | None = None,
    end: float | None = None,
    show_timestamps: bool = False,
) -> str:
    content = text.strip()
    if not show_timestamps or start is None:
        return content
    if end is None:
        return f"[{format_timestamp(start)}] {content}"
    return f"[{format_timestamp(start)} - {format_timestamp(end)}] {content}"


def _segment_text(segment: dict[str, Any]) -> str:
    text = str(segment.get("text", "")).strip()
    if text:
        return text
    words = segment.get("words")
    if isinstance(words, list):
        return " ".join(str(word.get("word", "")).strip() for word in words).strip()
    return ""


def _normalize_segments(segments: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, segment in enumerate(segments, start=1):
        normalized.append(
            {
                "id": int(segment.get("id", index)),
                "start": float(segment.get("start", 0.0) or 0.0),
                "end": float(segment.get("end", 0.0) or 0.0),
                "text": _segment_text(segment),
                "words": segment.get("words", []),
            }
        )
    return normalized


def format_srt(segments: Iterable[dict[str, Any]]) -> str:
    lines: list[str] = []
    for index, segment in enumerate(_normalize_segments(segments), start=1):
        lines.append(str(index))
        lines.append(
            f"{format_timestamp(segment['start'], srt=True)} --> "
            f"{format_timestamp(segment['end'], srt=True)}"
        )
        lines.append(segment["text"])
        lines.append("")
    return "\n".join(lines).rstrip() + ("\n" if lines else "")


def format_vtt(segments: Iterable[dict[str, Any]]) -> str:
    lines = ["WEBVTT", ""]
    for segment in _normalize_segments(segments):
        lines.append(
            f"{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}"
        )
        lines.append(segment["text"])
        lines.append("")
    return "\n".join(lines).rstrip() + ("\n" if len(lines) > 2 else "")


def render_listening_status(
    *,
    listening: bool,
    speech_detected: bool = False,
    push_to_talk: bool = False,
    device: str | None = None,
) -> Text:
    state = "listening" if listening else "idle"
    if speech_detected:
        state = "speech"
    label = Text()
    color = "green" if speech_detected else "cyan" if listening else "yellow"
    label.append("● ", style=color)
    label.append(state, style=f"bold {color}")
    if push_to_talk:
        label.append("  push-to-talk", style="dim")
    if device:
        label.append(f"  {device}", style="dim")
    return label


def render_transcription_progress(
    source_label: str,
    *,
    processed_seconds: float | None = None,
    total_seconds: float | None = None,
) -> str:
    if processed_seconds is None:
        return f"Transcribing {source_label}"
    if total_seconds is None or total_seconds <= 0:
        return f"Transcribing {source_label} [{processed_seconds:.1f}s]"
    ratio = max(0.0, min(1.0, processed_seconds / total_seconds))
    columns = shutil.get_terminal_size(fallback=(96, 24)).columns
    bar_width = max(12, min(32, columns // 4))
    filled = int(round(bar_width * ratio))
    bar = "#" * filled + "-" * (bar_width - filled)
    return (
        f"Transcribing {source_label} "
        f"[{bar}] {ratio * 100:>5.1f}% "
        f"({processed_seconds:.1f}s/{total_seconds:.1f}s)"
    )


@dataclass(slots=True)
class TranscriptionDisplay:
    console: Console = field(default_factory=Console)
    show_timestamps: bool = False
    output_format: str = "text"
    _status_line: str | None = None

    def status(self, message: str | Text) -> None:
        self.console.print(message)
        self._status_line = str(message)

    def emit_event(self, event: Any) -> None:
        kind = getattr(event, "kind", "")
        if kind == "error":
            message = getattr(event, "message", "unknown error")
            self.console.print(f"[red]Error:[/] {message}")
            return

        if kind in {"status", "silence"}:
            message = getattr(event, "message", None)
            if message:
                self.console.print(message)
            return

        result = getattr(event, "result", None)
        if result is not None:
            self.write_result(result)
            return

        text = str(getattr(event, "text", "")).strip()
        if text:
            self.console.print(
                render_transcript_line(
                    text,
                    start=getattr(event, "start", None),
                    end=getattr(event, "end", None),
                    show_timestamps=self.show_timestamps,
                )
            )

    def write_result(self, result: Any) -> None:
        output_format = self.output_format
        if output_format == "json":
            self.console.print_json(data=result.to_dict())
            return
        if output_format == "srt":
            self.console.print(format_srt(getattr(result, "segments", [])), end="")
            return
        if output_format == "vtt":
            self.console.print(format_vtt(getattr(result, "segments", [])), end="")
            return

        postprocess = getattr(result, "postprocess", {}) or {}
        text = str(getattr(result, "text", "")).strip()
        segments = getattr(result, "segments", [])
        if postprocess.get("applied") and text:
            start = end = None
            normalized_segments = _normalize_segments(segments)
            if normalized_segments:
                start = normalized_segments[0]["start"]
                end = normalized_segments[-1]["end"]
            self.console.print(
                render_transcript_line(
                    text,
                    start=start,
                    end=end,
                    show_timestamps=self.show_timestamps,
                )
            )
            return

        if segments:
            for segment in _normalize_segments(segments):
                self.console.print(
                    render_transcript_line(
                        segment["text"],
                        start=segment["start"],
                        end=segment["end"],
                        show_timestamps=self.show_timestamps,
                    )
                )
            return

        self.console.print(
            render_transcript_line(
                str(getattr(result, "text", "")).strip(),
                show_timestamps=False,
            )
        )
