from __future__ import annotations

from datetime import datetime, timedelta, timezone
import unittest

from dwhisper.utils import format_time_ago, render_transcript_line


class UtilsTests(unittest.TestCase):
    def test_format_time_ago_accepts_datetime(self) -> None:
        value = datetime.now(timezone.utc) - timedelta(minutes=5)
        self.assertEqual(format_time_ago(value), "5 min ago")

    def test_format_time_ago_accepts_timestamp(self) -> None:
        value = (datetime.now(timezone.utc) - timedelta(hours=2)).timestamp()
        self.assertEqual(format_time_ago(value), "2 hours ago")

    def test_render_transcript_line_includes_timestamps_when_requested(self) -> None:
        rendered = render_transcript_line("hello", start=1.25, end=2.0, show_timestamps=True)
        self.assertIn("[00:00:01.250 - 00:00:02.000]", rendered)
        self.assertTrue(rendered.endswith("hello"))


if __name__ == "__main__":
    unittest.main()
