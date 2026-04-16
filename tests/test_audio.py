from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from dwhisper.audio import (
    AudioConfig,
    VoiceActivityDetector,
    load_audio_file,
    resolve_audio_device,
    write_wav_file,
)


class AudioTests(unittest.TestCase):
    def test_audio_config_computes_frames_per_chunk(self) -> None:
        config = AudioConfig(sample_rate=16000, chunk_duration=0.5)
        self.assertEqual(config.frames_per_chunk, 8000)

    def test_voice_activity_detector_distinguishes_speech_from_silence(self) -> None:
        vad = VoiceActivityDetector(sensitivity=0.6)
        silence = np.zeros(1600, dtype=np.float32)
        speech = np.full(1600, 0.2, dtype=np.float32)

        self.assertFalse(vad.is_speech(silence))
        self.assertTrue(vad.is_speech(speech))

    def test_load_audio_file_reads_and_resamples_wave(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tone.wav"
            samples = np.sin(np.linspace(0, np.pi * 4, num=8000, dtype=np.float32))
            write_wav_file(path, samples, sample_rate=8000)

            audio, sample_rate = load_audio_file(path, sample_rate=16000)

        self.assertEqual(sample_rate, 16000)
        self.assertGreater(audio.shape[0], samples.shape[0])

    def test_resolve_audio_device_supports_numeric_strings_and_name_matching(self) -> None:
        with mock.patch(
            "dwhisper.audio.list_audio_devices",
            return_value=[
                {"index": 0, "name": "Built-in Microphone"},
                {"index": 3, "name": "USB Audio Interface"},
            ],
        ):
            self.assertEqual(resolve_audio_device("3"), 3)
            self.assertEqual(resolve_audio_device("usb audio"), 3)

    def test_resolve_audio_device_rejects_ambiguous_names(self) -> None:
        with mock.patch(
            "dwhisper.audio.list_audio_devices",
            return_value=[
                {"index": 0, "name": "USB Mic A"},
                {"index": 1, "name": "USB Mic B"},
            ],
        ):
            with self.assertRaisesRegex(ValueError, "ambiguous"):
                resolve_audio_device("usb")


if __name__ == "__main__":
    unittest.main()
