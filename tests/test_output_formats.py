from __future__ import annotations

import unittest

from dwhisper.utils import format_srt, format_vtt


SEGMENTS = [
    {"start": 0.0, "end": 1.23, "text": "hello"},
    {"start": 1.23, "end": 2.5, "text": "world"},
]


class OutputFormatTests(unittest.TestCase):
    def test_format_srt(self) -> None:
        rendered = format_srt(SEGMENTS)
        self.assertIn("00:00:00,000 --> 00:00:01,230", rendered)
        self.assertIn("hello", rendered)
        self.assertIn("2", rendered)

    def test_format_vtt(self) -> None:
        rendered = format_vtt(SEGMENTS)
        self.assertTrue(rendered.startswith("WEBVTT"))
        self.assertIn("00:00:01.230 --> 00:00:02.500", rendered)
        self.assertIn("world", rendered)


if __name__ == "__main__":
    unittest.main()
