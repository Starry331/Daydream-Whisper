from __future__ import annotations

import dataclasses
import queue
import select
import sys
import termios
import threading
import tty
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from rich.console import Console

from dwhisper.audio import AudioCapture, AudioConfig, VoiceActivityDetector
from dwhisper.transcriber import (
    TranscribeOptions,
    TranscribeResult,
    WhisperTranscriber,
    _safe_apply_postprocessor,
)
from dwhisper.utils import TranscriptionDisplay, render_listening_status


@dataclass(slots=True)
class RealtimeConfig:
    sample_rate: int = 16000
    chunk_duration: float = 3.0
    overlap_duration: float = 0.5
    silence_threshold: float = 1.0
    vad_sensitivity: float = 0.6
    device: str | int | None = None
    push_to_talk: bool = False
    poll_interval: float = 0.1
    capture_chunk_duration: float = 0.25

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be greater than 0.")
        if self.chunk_duration <= 0:
            raise ValueError("chunk_duration must be greater than 0.")
        if self.overlap_duration < 0:
            raise ValueError("overlap_duration cannot be negative.")
        if self.overlap_duration >= self.chunk_duration:
            raise ValueError("overlap_duration must be smaller than chunk_duration.")
        if self.silence_threshold < 0:
            raise ValueError("silence_threshold cannot be negative.")
        if not 0.0 <= self.vad_sensitivity <= 1.0:
            raise ValueError("vad_sensitivity must be between 0.0 and 1.0.")
        if self.poll_interval <= 0:
            raise ValueError("poll_interval must be greater than 0.")
        if self.capture_chunk_duration <= 0:
            raise ValueError("capture_chunk_duration must be greater than 0.")


@dataclass(slots=True)
class TranscriptionEvent:
    kind: str
    text: str = ""
    start: float | None = None
    end: float | None = None
    message: str | None = None
    result: TranscribeResult | None = None


@dataclass(slots=True)
class _QueuedRealtimeResult:
    result: TranscribeResult
    start: float
    end: float


@dataclass(slots=True)
class RealtimeSession:
    transcriber: WhisperTranscriber
    options: TranscribeOptions
    config: RealtimeConfig
    event_handler: Callable[[TranscriptionEvent], None] | None = None
    capture: AudioCapture | None = None
    vad: VoiceActivityDetector = field(init=False)
    running: bool = field(default=False, init=False)
    paused: bool = field(default=False, init=False)
    _push_to_talk_active: bool = field(default=False, init=False)
    _buffer_chunks: list[np.ndarray] = field(default_factory=list, init=False)
    _buffer_duration: float = field(default=0.0, init=False)
    _segment_start: float = field(default=0.0, init=False)
    _stream_cursor: float = field(default=0.0, init=False)
    _silence_duration: float = field(default=0.0, init=False)
    _had_speech: bool = field(default=False, init=False)
    _transcribe_options: TranscribeOptions = field(init=False, repr=False)
    _postprocessor: Any | None = field(default=None, init=False, repr=False)
    _postprocess_queue: queue.Queue[_QueuedRealtimeResult | None] | None = field(default=None, init=False, repr=False)
    _postprocess_thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _postprocess_queue_size: int = field(default=4, init=False, repr=False)

    def __post_init__(self) -> None:
        self.vad = VoiceActivityDetector(self.config.vad_sensitivity)
        if self.capture is None:
            capture_chunk = min(self.config.capture_chunk_duration, max(0.1, self.config.chunk_duration / 4.0))
            self.capture = AudioCapture(
                AudioConfig(
                    sample_rate=self.config.sample_rate,
                    channels=1,
                    chunk_duration=capture_chunk,
                    device=self.config.device,
                )
            )
        self._transcribe_options = self.options
        if self.options.postprocess:
            self._transcribe_options = dataclasses.replace(self.options, postprocess=False)
            self._postprocessor = self.options.build_postprocessor()
            self._postprocess_queue = queue.Queue(maxsize=self._postprocess_queue_size)
            self._postprocess_thread = threading.Thread(
                target=self._run_postprocess_loop,
                name="dwhisper-realtime-postprocess",
                daemon=True,
            )

    def _emit(self, event: TranscriptionEvent) -> None:
        if self.event_handler is not None:
            self.event_handler(event)

    def _run_postprocess_loop(self) -> None:
        if self._postprocess_queue is None:
            return

        while True:
            item = self._postprocess_queue.get()
            try:
                if item is None:
                    return
                if self._postprocessor is not None:
                    _safe_apply_postprocessor(self._postprocessor, item.result)
                self._emit(
                    TranscriptionEvent(
                        kind="final",
                        text=item.result.text,
                        start=item.start,
                        end=item.end,
                        result=item.result,
                    )
                )
            finally:
                self._postprocess_queue.task_done()

    def _submit_result(self, result: TranscribeResult, *, start: float, end: float) -> None:
        if self._postprocess_queue is None:
            self._emit(
                TranscriptionEvent(
                    kind="final",
                    text=result.text,
                    start=start,
                    end=end,
                    result=result,
                )
            )
            return

        queued = _QueuedRealtimeResult(result=result, start=start, end=end)
        while True:
            try:
                self._postprocess_queue.put(
                    queued,
                    timeout=max(self.config.poll_interval, 0.1),
                )
                return
            except queue.Full:
                continue

    def _drain_postprocess_loop(self) -> None:
        if self._postprocess_queue is None or self._postprocess_thread is None:
            return
        self._postprocess_queue.put(None)
        self._postprocess_thread.join()

    def start(self) -> None:
        if self.running:
            return
        self.capture.start()
        if self._postprocess_thread is not None and not self._postprocess_thread.is_alive():
            self._postprocess_thread.start()
        self.running = True
        self.paused = False
        self._push_to_talk_active = not self.config.push_to_talk
        self._emit(
            TranscriptionEvent(
                kind="status",
                message="Listening started.",
            )
        )

    def stop(self) -> None:
        if not self.running:
            return
        if self._buffer_chunks and self._had_speech:
            self._flush_buffer(retain_overlap=False)
        self.capture.stop()
        self.running = False
        self._drain_postprocess_loop()
        self._emit(TranscriptionEvent(kind="status", message="Listening stopped."))

    def pause(self) -> None:
        self.paused = True
        self._emit(TranscriptionEvent(kind="status", message="Listening paused."))

    def resume(self) -> None:
        self.paused = False
        self._emit(TranscriptionEvent(kind="status", message="Listening resumed."))

    def is_push_to_talk_active(self) -> bool:
        return self._push_to_talk_active

    def set_push_to_talk_active(self, active: bool) -> None:
        previous = self._push_to_talk_active
        self._push_to_talk_active = active
        if previous and not active and self._buffer_chunks and self._had_speech:
            self._flush_buffer(retain_overlap=False)

    def toggle_push_to_talk(self) -> bool:
        self.set_push_to_talk_active(not self._push_to_talk_active)
        return self._push_to_talk_active

    def poll_once(self, *, timeout: float | None = None) -> bool:
        if not self.running or self.paused:
            return False
        try:
            chunk = self.capture.read(timeout=timeout)
        except queue.Empty:
            return False
        self.feed_audio(chunk)
        return True

    def feed_audio(self, samples: np.ndarray) -> None:
        chunk = np.nan_to_num(np.asarray(samples, dtype=np.float32).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
        if chunk.size == 0:
            return

        chunk_duration = float(chunk.size) / float(self.config.sample_rate)
        gate_open = self._push_to_talk_active if self.config.push_to_talk else True
        if not gate_open:
            self._stream_cursor += chunk_duration
            return

        has_speech = True if self.config.push_to_talk else self.vad.is_speech(chunk)
        if has_speech:
            if not self._buffer_chunks:
                self._segment_start = self._stream_cursor
            self._buffer_chunks.append(chunk)
            self._buffer_duration += chunk_duration
            self._silence_duration = 0.0
            self._had_speech = True
            if self._buffer_duration >= self.config.chunk_duration:
                self._flush_buffer(retain_overlap=True)
        elif self._buffer_chunks:
            self._buffer_chunks.append(chunk)
            self._buffer_duration += chunk_duration
            self._silence_duration += chunk_duration
            if self._silence_duration >= self.config.silence_threshold:
                self._flush_buffer(retain_overlap=False)
                self._emit(TranscriptionEvent(kind="silence", message="Speech segment finalized."))

        self._stream_cursor += chunk_duration

    def _flush_buffer(self, *, retain_overlap: bool) -> None:
        if not self._buffer_chunks:
            return

        audio = np.concatenate(self._buffer_chunks)
        start = self._segment_start
        end = start + (audio.size / float(self.config.sample_rate))

        try:
            result = self.transcriber.transcribe_samples(
                audio,
                sample_rate=self.config.sample_rate,
                options=self._transcribe_options,
            )
        except Exception as exc:
            self._buffer_chunks = []
            self._buffer_duration = 0.0
            self._silence_duration = 0.0
            self._had_speech = False
            self._emit(TranscriptionEvent(kind="error", message=str(exc)))
            return

        self._submit_result(result, start=start, end=end)

        overlap_samples = int(round(self.config.overlap_duration * self.config.sample_rate))
        if retain_overlap and overlap_samples > 0 and audio.size > overlap_samples:
            overlap_audio = audio[-overlap_samples:].copy()
            self._buffer_chunks = [overlap_audio]
            self._buffer_duration = overlap_audio.size / float(self.config.sample_rate)
            self._segment_start = max(start, end - self._buffer_duration)
            self._silence_duration = 0.0
            self._had_speech = True
        else:
            self._buffer_chunks = []
            self._buffer_duration = 0.0
            self._segment_start = end
            self._silence_duration = 0.0
            self._had_speech = False


@contextmanager
def _raw_stdin():
    if not sys.stdin.isatty():
        yield False
        return

    file_descriptor = sys.stdin.fileno()
    previous = termios.tcgetattr(file_descriptor)
    try:
        tty.setcbreak(file_descriptor)
        yield True
    finally:
        termios.tcsetattr(file_descriptor, termios.TCSADRAIN, previous)


def _read_key(timeout: float) -> str | None:
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    if not ready:
        return None
    return sys.stdin.read(1)


def run_listen_session(
    *,
    model: str,
    transcribe_options: TranscribeOptions,
    realtime_config: RealtimeConfig,
    output_format: str = "text",
    verbose: bool = False,
    console: Console | None = None,
    transcriber: WhisperTranscriber | None = None,
    capture: AudioCapture | None = None,
) -> RealtimeSession:
    console = console or Console()
    owns_transcriber = transcriber is None
    display = TranscriptionDisplay(
        console=console,
        show_timestamps=output_format in {"srt", "vtt"},
        output_format=output_format,
    )
    session = RealtimeSession(
        transcriber=transcriber or WhisperTranscriber(model),
        options=transcribe_options,
        config=realtime_config,
        event_handler=display.emit_event,
        capture=capture,
    )

    device_label = str(realtime_config.device) if realtime_config.device is not None else None
    display.status(
        render_listening_status(
            listening=True,
            push_to_talk=realtime_config.push_to_talk,
            device=device_label,
        )
    )
    if verbose and realtime_config.push_to_talk:
        display.status("Push-to-talk mode: press space to toggle capture, q to quit.")

    session.start()
    try:
        if realtime_config.push_to_talk:
            if not sys.stdin.isatty():
                raise RuntimeError("Push-to-talk mode requires an interactive terminal.")
            with _raw_stdin():
                while True:
                    key = _read_key(realtime_config.poll_interval)
                    if key == " ":
                        active = session.toggle_push_to_talk()
                        display.status(
                            render_listening_status(
                                listening=active,
                                push_to_talk=True,
                                device=device_label,
                            )
                        )
                        session.poll_once(timeout=0.0)
                        continue
                    if key in {"q", "\x03"}:
                        break
                    session.poll_once(timeout=0.0)
        else:
            while True:
                session.poll_once(timeout=realtime_config.poll_interval)
    except KeyboardInterrupt:
        pass
    finally:
        session.stop()
        if owns_transcriber:
            session.transcriber.close()

    return session
