from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests.helpers import reload_module


class ConfigTests(unittest.TestCase):
    def test_config_reads_voice_defaults_from_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir) / "daydream-home"
            cache = Path(tmpdir) / "hf-cache"
            home.mkdir(parents=True, exist_ok=True)
            (home / "config.yaml").write_text(
                "model: whisper:base\n"
                "transcribe:\n"
                "  language: zh\n"
                "  profile: meeting-zh\n"
                "  task: translate\n"
                "  output_format: vtt\n"
                "  word_timestamps: true\n"
                "postprocess:\n"
                "  enabled: true\n"
                "  model: local-mm-model\n"
                "  base_url: http://127.0.0.1:11435/v1\n"
                "  api_key: test-key\n"
                "  mode: summary\n"
                "  timeout: 21\n"
                "correction:\n"
                "  corrections_file: /tmp/corrections.yaml\n"
                "  vocabulary_file: /tmp/vocabulary.yaml\n"
                "profiles:\n"
                "  file: /tmp/profiles.yaml\n"
                "audio:\n"
                "  sample_rate: 22050\n"
                "  device: Built-in Mic\n"
                "listen:\n"
                "  chunk_duration: 4.5\n"
                "  overlap_duration: 0.75\n"
                "  silence_threshold: 1.5\n"
                "  vad_sensitivity: 0.7\n"
                "  push_to_talk: true\n"
                "serve:\n"
                "  host: 0.0.0.0\n"
                "  port: 8080\n"
                "  max_concurrency: 4\n"
                "  request_timeout: 90\n"
                "  max_request_bytes: 123456\n"
                "  preload: true\n"
                "  allow_origin: http://localhost:3000\n",
                encoding="utf-8",
            )

            with mock.patch.dict(
                os.environ,
                {
                    "DAYDREAM_HOME": str(home),
                    "DAYDREAM_CACHE_DIR": str(cache),
                },
                clear=False,
            ):
                config = reload_module("dwhisper.config")
                self.assertEqual(config.DAYDREAM_HOME, home)
                self.assertEqual(config.MODEL_CACHE_DIR, cache)
                self.assertEqual(config.get_default_model(), "whisper:base")
                self.assertEqual(config.get_default_language(), "zh")
                self.assertEqual(config.get_default_profile(), "meeting-zh")
                self.assertEqual(config.get_default_task(), "translate")
                self.assertEqual(config.get_default_output_format(), "vtt")
                self.assertTrue(config.get_default_word_timestamps())
                self.assertTrue(config.get_default_postprocess_enabled())
                self.assertEqual(config.get_default_postprocess_model(), "local-mm-model")
                self.assertEqual(config.get_default_postprocess_base_url(), "http://127.0.0.1:11435/v1")
                self.assertEqual(config.get_default_postprocess_api_key(), "test-key")
                self.assertEqual(config.get_default_postprocess_mode(), "summary")
                self.assertEqual(config.get_default_postprocess_timeout(), 21.0)
                self.assertEqual(str(config.get_default_corrections_path()), "/tmp/corrections.yaml")
                self.assertEqual(str(config.get_default_vocabulary_path()), "/tmp/vocabulary.yaml")
                self.assertEqual(str(config.get_default_profiles_path()), "/tmp/profiles.yaml")
                self.assertEqual(config.get_default_audio_device(), "Built-in Mic")
                self.assertEqual(config.get_default_sample_rate(), 22050)
                self.assertEqual(config.get_default_chunk_duration(), 4.5)
                self.assertEqual(config.get_default_overlap_duration(), 0.75)
                self.assertEqual(config.get_default_silence_threshold(), 1.5)
                self.assertEqual(config.get_default_vad_sensitivity(), 0.7)
                self.assertTrue(config.get_default_push_to_talk())
                self.assertEqual(config.get_default_host(), "0.0.0.0")
                self.assertEqual(config.get_default_port(), 8080)
                self.assertEqual(config.get_default_serve_max_concurrency(), 4)
                self.assertEqual(config.get_default_serve_request_timeout(), 90.0)
                self.assertEqual(config.get_default_serve_max_request_bytes(), 123456)
                self.assertTrue(config.get_default_serve_preload())
                self.assertEqual(config.get_default_serve_allow_origin(), "http://localhost:3000")

    def test_environment_variables_override_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir) / "daydream-home"
            home.mkdir(parents=True, exist_ok=True)
            (home / "config.yaml").write_text(
                "model: whisper:base\ntranscribe:\n  language: en\n",
                encoding="utf-8",
            )

            with mock.patch.dict(
                os.environ,
                {
                    "DAYDREAM_HOME": str(home),
                    "DAYDREAM_MODEL": "whisper:large-v3",
                    "DAYDREAM_LANGUAGE": "ja",
                    "DAYDREAM_PUSH_TO_TALK": "true",
                    "DAYDREAM_POSTPROCESS": "true",
                    "DAYDREAM_POSTPROCESS_MODEL": "local-mm-model",
                },
                clear=False,
            ):
                config = reload_module("dwhisper.config")
                self.assertEqual(config.get_default_model(), "whisper:large-v3")
                self.assertEqual(config.get_default_language(), "ja")
                self.assertTrue(config.get_default_push_to_talk())
                self.assertTrue(config.get_default_postprocess_enabled())
                self.assertEqual(config.get_default_postprocess_model(), "local-mm-model")

    def test_ensure_home_creates_home_and_local_models_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir) / "daydream-home"
            with mock.patch.dict(os.environ, {"DAYDREAM_HOME": str(home)}, clear=False):
                config = reload_module("dwhisper.config")
                config.ensure_home()
                self.assertTrue(config.DAYDREAM_HOME.exists())
                self.assertTrue(config.LOCAL_MODELS_DIR.exists())


if __name__ == "__main__":
    unittest.main()
