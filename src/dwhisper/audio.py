from __future__ import annotations

import queue
import subprocess
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    import sounddevice as sounddevice_module
except ImportError:  # pragma: no cover - exercised through dependency guards
    sounddevice_module = None


def _require_sounddevice():
    if sounddevice_module is None:
        raise RuntimeError(
            "sounddevice is not installed. Reinstall Daydream with audio capture support enabled."
        )
    return sounddevice_module


@dataclass(slots=True)
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    chunk_duration: float = 0.25
    device: str | int | None = None
    dtype: str = "float32"
    queue_size: int = 32

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be greater than 0.")
        if self.channels <= 0:
            raise ValueError("channels must be greater than 0.")
        if self.chunk_duration <= 0:
            raise ValueError("chunk_duration must be greater than 0.")
        if self.queue_size <= 0:
            raise ValueError("queue_size must be greater than 0.")

    @property
    def frames_per_chunk(self) -> int:
        return max(1, int(round(self.sample_rate * self.chunk_duration)))


@dataclass(slots=True)
class AudioCapture:
    config: AudioConfig
    input_queue: queue.Queue[np.ndarray] = field(default_factory=queue.Queue)
    _stream: Any | None = None

    def __post_init__(self) -> None:
        if self.input_queue.maxsize <= 0:
            self.input_queue = queue.Queue(maxsize=self.config.queue_size)

    def _callback(self, indata, frames, time_info, status) -> None:
        del frames, time_info
        if status:
            return
        samples = np.asarray(indata, dtype=np.float32)
        if samples.ndim == 2:
            samples = samples.mean(axis=1)
        try:
            self.input_queue.put_nowait(samples.copy())
        except queue.Full:
            try:
                self.input_queue.get_nowait()
            except queue.Empty:
                pass
            self.input_queue.put_nowait(samples.copy())

    def start(self) -> None:
        sounddevice = _require_sounddevice()
        self.input_queue = queue.Queue(maxsize=self.config.queue_size)
        resolved_device = resolve_audio_device(self.config.device)
        self._stream = sounddevice.InputStream(
            samplerate=self.config.sample_rate,
            channels=self.config.channels,
            blocksize=self.config.frames_per_chunk,
            device=resolved_device,
            dtype=self.config.dtype,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is None:
            return
        self._stream.stop()
        self._stream.close()
        self._stream = None

    def read(self, *, timeout: float | None = None) -> np.ndarray:
        return self.input_queue.get(timeout=timeout)


@dataclass(slots=True)
class VoiceActivityDetector:
    sensitivity: float = 0.6

    def threshold(self) -> float:
        sensitivity = min(max(self.sensitivity, 0.0), 1.0)
        return max(0.005, (1.0 - sensitivity) * 0.05)

    def rms(self, samples: np.ndarray) -> float:
        if samples.size == 0:
            return 0.0
        clean = np.nan_to_num(samples.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        return float(np.sqrt(np.mean(np.square(clean))))

    def is_speech(self, samples: np.ndarray) -> bool:
        return self.rms(samples) >= self.threshold()


def list_audio_devices() -> list[dict[str, Any]]:
    sounddevice = _require_sounddevice()
    default_input = None
    default_device = getattr(sounddevice, "default", None)
    if default_device is not None:
        raw_default = getattr(default_device, "device", None)
        if isinstance(raw_default, (list, tuple)) and raw_default:
            default_input = raw_default[0]

    devices: list[dict[str, Any]] = []
    try:
        queried_devices = sounddevice.query_devices()
    except Exception as exc:
        raise RuntimeError(f"Could not query audio devices: {exc}") from exc

    for index, device in enumerate(queried_devices):
        max_input = int(device.get("max_input_channels", 0) or 0)
        if max_input <= 0:
            continue
        devices.append(
            {
                "index": index,
                "name": str(device.get("name", f"Input {index}")),
                "max_input_channels": max_input,
                "default_samplerate": int(device.get("default_samplerate", 0) or 0),
                "is_default": index == default_input,
            }
        )
    return devices


def resolve_audio_device(device: str | int | None) -> str | int | None:
    if device is None:
        return None
    if isinstance(device, int):
        return device

    value = str(device).strip()
    if not value:
        return None
    if value.isdigit():
        return int(value)

    lowered = value.lower()
    exact_matches: list[int] = []
    partial_matches: list[int] = []
    for entry in list_audio_devices():
        name = str(entry["name"])
        if name.lower() == lowered:
            exact_matches.append(int(entry["index"]))
        elif lowered in name.lower():
            partial_matches.append(int(entry["index"]))

    matches = exact_matches or partial_matches
    if not matches:
        raise ValueError(f"No input audio device matches '{device}'.")
    if len(matches) > 1:
        raise ValueError(
            f"Audio device '{device}' is ambiguous. Use a numeric index from `dwhisper devices`."
        )
    return matches[0]


def _resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate or audio.size == 0:
        return audio.astype(np.float32, copy=False)

    target_length = max(1, int(round(audio.shape[0] * target_rate / source_rate)))
    source_positions = np.linspace(0.0, audio.shape[0] - 1, num=audio.shape[0], dtype=np.float64)
    target_positions = np.linspace(0.0, audio.shape[0] - 1, num=target_length, dtype=np.float64)
    return np.interp(target_positions, source_positions, audio).astype(np.float32)


def _pcm_bytes_to_float32(frames: bytes, sample_width: int) -> np.ndarray:
    if sample_width == 1:
        audio = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
        return (audio - 128.0) / 128.0
    if sample_width == 2:
        audio = np.frombuffer(frames, dtype="<i2").astype(np.float32)
        return audio / 32768.0
    if sample_width == 3:
        raw = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3)
        audio = (
            raw[:, 0].astype(np.int32)
            | (raw[:, 1].astype(np.int32) << 8)
            | (raw[:, 2].astype(np.int32) << 16)
        )
        sign_mask = 1 << 23
        audio = (audio ^ sign_mask) - sign_mask
        return audio.astype(np.float32) / float(1 << 23)
    if sample_width == 4:
        audio = np.frombuffer(frames, dtype="<i4").astype(np.float32)
        return audio / float(1 << 31)
    raise ValueError(f"Unsupported WAV sample width: {sample_width}")


def _collapse_channels(audio: np.ndarray, channels: int) -> np.ndarray:
    if channels <= 1:
        return audio
    return audio.reshape(-1, channels).mean(axis=1)


def _load_wave_file(path: Path, sample_rate: int) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        source_rate = handle.getframerate()
        sample_width = handle.getsampwidth()
        frames = handle.readframes(handle.getnframes())

    audio = _pcm_bytes_to_float32(frames, sample_width)
    audio = _collapse_channels(audio, channels)
    return _resample(audio, source_rate, sample_rate), sample_rate


def _load_ffmpeg_file(path: Path, sample_rate: int) -> tuple[np.ndarray, int]:
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-nostdin",
        "-i",
        str(path),
        "-f",
        "f32le",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-",
    ]
    try:
        result = subprocess.run(command, capture_output=True, check=True)
    except FileNotFoundError as exc:  # pragma: no cover - platform dependent
        raise RuntimeError("ffmpeg is required to decode this audio file format.") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="ignore").strip()
        raise RuntimeError(f"ffmpeg could not decode '{path}': {stderr or exc}") from exc

    audio = np.frombuffer(result.stdout, dtype=np.float32)
    return audio, sample_rate


def load_audio_file(path: str | Path, *, sample_rate: int = 16000) -> tuple[np.ndarray, int]:
    audio_path = Path(path).expanduser()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be greater than 0.")

    if audio_path.suffix.lower() in {".wav", ".wave"}:
        return _load_wave_file(audio_path, sample_rate)

    return _load_ffmpeg_file(audio_path, sample_rate)


def validate_audio_file(path: str | Path) -> Path:
    audio_path = Path(path).expanduser()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    load_audio_file(audio_path)
    return audio_path


def write_wav_file(path: str | Path, audio: np.ndarray, *, sample_rate: int = 16000) -> Path:
    audio_path = Path(path)
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    with wave.open(str(audio_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    return audio_path
