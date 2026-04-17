from __future__ import annotations

import threading
import unittest
from unittest import mock

import numpy as np

from dwhisper.realtime import RealtimeConfig, RealtimeSession
from dwhisper.transcriber import TranscribeOptions, TranscribeResult


class FakeCapture:
    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None


class FakeTranscriber:
    def __init__(self) -> None:
        self.calls = 0

    def transcribe_samples(self, audio, *, sample_rate: int, options: TranscribeOptions):
        self.calls += 1
        duration = audio.size / sample_rate
        return TranscribeResult(
            text=f"segment-{self.calls}",
            segments=[{"start": 0.0, "end": duration, "text": f"segment-{self.calls}"}],
            duration=duration,
        )


class RealtimeTests(unittest.TestCase):
    def test_realtime_session_flushes_when_chunk_duration_is_reached(self) -> None:
        events = []
        session = RealtimeSession(
            transcriber=FakeTranscriber(),
            options=TranscribeOptions(),
            config=RealtimeConfig(sample_rate=4, chunk_duration=1.0, overlap_duration=0.0, silence_threshold=0.5),
            event_handler=events.append,
            capture=FakeCapture(),
        )
        session.start()
        session.feed_audio(np.ones(4, dtype=np.float32))

        self.assertEqual([event.kind for event in events if event.kind == "final"], ["final"])
        self.assertEqual(events[-1].text, "segment-1")

    def test_realtime_session_finalizes_after_silence(self) -> None:
        events = []
        session = RealtimeSession(
            transcriber=FakeTranscriber(),
            options=TranscribeOptions(),
            config=RealtimeConfig(sample_rate=4, chunk_duration=10.0, overlap_duration=0.0, silence_threshold=0.5),
            event_handler=events.append,
            capture=FakeCapture(),
        )
        session.start()
        session.feed_audio(np.ones(2, dtype=np.float32))
        session.feed_audio(np.zeros(2, dtype=np.float32))

        kinds = [event.kind for event in events]
        self.assertIn("final", kinds)
        self.assertIn("silence", kinds)

    def test_push_to_talk_blocks_audio_until_enabled(self) -> None:
        events = []
        session = RealtimeSession(
            transcriber=FakeTranscriber(),
            options=TranscribeOptions(),
            config=RealtimeConfig(sample_rate=4, chunk_duration=1.0, overlap_duration=0.0, push_to_talk=True),
            event_handler=events.append,
            capture=FakeCapture(),
        )
        session.start()
        session.feed_audio(np.ones(4, dtype=np.float32))
        self.assertFalse(any(event.kind == "final" for event in events))

        session.set_push_to_talk_active(True)
        session.feed_audio(np.ones(4, dtype=np.float32))
        self.assertTrue(any(event.kind == "final" for event in events))

    def test_push_to_talk_flushes_buffer_when_disabled(self) -> None:
        events = []
        session = RealtimeSession(
            transcriber=FakeTranscriber(),
            options=TranscribeOptions(),
            config=RealtimeConfig(sample_rate=4, chunk_duration=10.0, overlap_duration=0.0, push_to_talk=True),
            event_handler=events.append,
            capture=FakeCapture(),
        )
        session.start()
        session.set_push_to_talk_active(True)
        session.feed_audio(np.ones(2, dtype=np.float32))
        session.set_push_to_talk_active(False)

        self.assertTrue(any(event.kind == "final" for event in events))

    def test_realtime_config_rejects_invalid_overlap(self) -> None:
        with self.assertRaisesRegex(ValueError, "smaller than chunk_duration"):
            RealtimeConfig(chunk_duration=1.0, overlap_duration=1.0)

    def test_realtime_session_overlaps_postprocess_with_next_chunk(self) -> None:
        events = []
        transcriber = FakeTranscriber()
        gate = threading.Event()
        started = threading.Event()

        def fake_apply(_processor, result):
            started.set()
            gate.wait(timeout=1.0)
            result.text = f"{result.text}-clean"
            result.postprocess = {"applied": True}

        with mock.patch.object(
            TranscribeOptions,
            "build_postprocessor",
            return_value=object(),
        ), mock.patch("dwhisper.realtime._safe_apply_postprocessor", side_effect=fake_apply):
            session = RealtimeSession(
                transcriber=transcriber,
                options=TranscribeOptions(
                    postprocess=True,
                    postprocess_model="local-mm-model",
                    postprocess_base_url="http://127.0.0.1:11435/v1",
                ),
                config=RealtimeConfig(
                    sample_rate=4,
                    chunk_duration=1.0,
                    overlap_duration=0.0,
                    silence_threshold=0.5,
                ),
                event_handler=events.append,
                capture=FakeCapture(),
            )
            session.start()
            session.feed_audio(np.ones(4, dtype=np.float32))
            self.assertTrue(started.wait(timeout=1.0))
            session.feed_audio(np.ones(4, dtype=np.float32))
            self.assertEqual(transcriber.calls, 2)
            gate.set()
            session.stop()

        finals = [event for event in events if event.kind == "final"]
        self.assertEqual([event.text for event in finals], ["segment-1-clean", "segment-2-clean"])


if __name__ == "__main__":
    unittest.main()
