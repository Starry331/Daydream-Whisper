from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from dwhisper.profiles import load_profile, load_profile_store


class ProfileTests(unittest.TestCase):
    def test_load_profile_store_reads_default_and_nested_options(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            profiles_path = Path(tmpdir) / "profiles.yaml"
            profiles_path.write_text(
                "default: meeting-zh\n"
                "profiles:\n"
                "  meeting-zh:\n"
                "    description: Chinese meeting preset\n"
                "    model: whisper:large-v3-turbo\n"
                "    output_format: srt\n"
                "    transcribe:\n"
                "      language: zh\n"
                "      task: transcribe\n"
                "      hotwords:\n"
                "        - Daydream\n"
                "      correction:\n"
                "        capitalize_sentences: true\n"
                "    listen:\n"
                "      chunk_duration: 2.5\n"
                "      push_to_talk: true\n",
                encoding="utf-8",
            )

            store = load_profile_store(profiles_path)

        self.assertEqual(store.default_profile, "meeting-zh")
        profile = store.get()
        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(profile.model, "whisper:large-v3-turbo")
        self.assertEqual(profile.output_format, "srt")
        self.assertEqual(profile.transcribe["language"], "zh")
        self.assertEqual(profile.listen["chunk_duration"], 2.5)

    def test_load_profile_supports_flat_profile_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            profiles_path = Path(tmpdir) / "profiles.yaml"
            profiles_path.write_text(
                "podcast:\n"
                "  model: whisper:medium\n"
                "  language: en\n"
                "  output_format: text\n"
                "  device: External Mic\n",
                encoding="utf-8",
            )

            profile = load_profile("podcast", profiles_path=profiles_path)

        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(profile.transcribe["language"], "en")
        self.assertEqual(profile.listen["device"], "External Mic")


if __name__ == "__main__":
    unittest.main()
