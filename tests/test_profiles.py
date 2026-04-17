from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dwhisper.profiles import load_profile, load_profile_store
from tests.helpers import reload_module


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

    def test_load_profile_store_reads_directory_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            profiles_dir = Path(tmpdir) / "profiles"
            profiles_dir.mkdir(parents=True, exist_ok=True)
            (profiles_dir / "meeting.yaml").write_text(
                "description: Meeting preset\n"
                "model: whisper:large-v3-turbo\n"
                "transcribe:\n"
                "  language: zh\n"
                "  postprocess: true\n"
                "  postprocess_mode: meeting-notes\n"
                "listen:\n"
                "  chunk_duration: 2.5\n",
                encoding="utf-8",
            )

            store = load_profile_store(profiles_dir)

        profile = store.get("meeting")
        assert profile is not None
        self.assertEqual(profile.name, "meeting")
        self.assertEqual(profile.transcribe["postprocess_mode"], "meeting-notes")
        self.assertEqual(profile.listen["chunk_duration"], 2.5)

    def test_default_loader_merges_profiles_file_and_directory_with_directory_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir) / "daydream-home"
            home.mkdir(parents=True, exist_ok=True)
            (home / "profiles.yaml").write_text(
                "profiles:\n"
                "  meeting:\n"
                "    model: whisper:base\n"
                "    transcribe:\n"
                "      language: en\n",
                encoding="utf-8",
            )
            profiles_dir = home / "profiles"
            profiles_dir.mkdir(parents=True, exist_ok=True)
            (profiles_dir / "meeting.yaml").write_text(
                "default: true\n"
                "model: whisper:large-v3-turbo\n"
                "transcribe:\n"
                "  language: zh\n",
                encoding="utf-8",
            )
            (profiles_dir / "podcast.yaml").write_text(
                "model: whisper:medium\n"
                "language: en\n",
                encoding="utf-8",
            )

            with mock.patch.dict(os.environ, {"DAYDREAM_HOME": str(home)}, clear=False):
                reload_module("dwhisper.config")
                profiles = reload_module("dwhisper.profiles")
                store = profiles.load_profile_store()

        self.assertEqual(store.default_profile, "meeting")
        meeting = store.get("meeting")
        podcast = store.get("podcast")
        assert meeting is not None
        assert podcast is not None
        self.assertEqual(meeting.model, "whisper:large-v3-turbo")
        self.assertEqual(meeting.transcribe["language"], "zh")
        self.assertEqual(podcast.name, "podcast")


if __name__ == "__main__":
    unittest.main()
