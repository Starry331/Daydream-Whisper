from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests.helpers import reload_module, write_fake_local_whisper_model, write_fake_multimodal_bundle


class LocalRegistryTests(unittest.TestCase):
    def test_direct_local_path_auto_registers_alias(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir) / "home"
            local_model = Path(tmpdir) / "Whisper-Large-v3-MLX"
            home.mkdir(parents=True, exist_ok=True)
            write_fake_local_whisper_model(local_model)

            with mock.patch.dict(os.environ, {"DAYDREAM_HOME": str(home)}, clear=False):
                reload_module("dwhisper.config")
                registry = reload_module("dwhisper.registry")

                resolved = registry.resolve(str(local_model))
                self.assertEqual(resolved, str(local_model.resolve()))
                self.assertEqual(registry.resolve("whisper-large-v3"), str(local_model.resolve()))
                self.assertEqual(
                    registry.reverse_lookup(str(local_model.resolve())),
                    "whisper-large-v3",
                )

    def test_scan_local_model_roots_registers_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir) / "home"
            root = Path(tmpdir) / "models"
            local_model = root / "nested" / "Meeting-Voice-MLX"
            home.mkdir(parents=True, exist_ok=True)
            write_fake_local_whisper_model(local_model)

            with mock.patch.dict(
                os.environ,
                {
                    "DAYDREAM_HOME": str(home),
                    "DAYDREAM_MODELS_DIRS": str(root),
                },
                clear=False,
            ):
                reload_module("dwhisper.config")
                registry = reload_module("dwhisper.registry")

                discovered = registry.scan_local_models(persist=True)
                self.assertIn(("meeting-voice", str(local_model.resolve())), discovered)
                self.assertEqual(registry.resolve("meeting-voice"), str(local_model.resolve()))

    def test_bundle_root_resolves_to_embedded_speech_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir) / "home"
            bundle_root, runtime_dir = write_fake_multimodal_bundle(Path(tmpdir) / "Voice-Agent-Bundle")
            home.mkdir(parents=True, exist_ok=True)

            with mock.patch.dict(os.environ, {"DAYDREAM_HOME": str(home)}, clear=False):
                reload_module("dwhisper.config")
                registry = reload_module("dwhisper.registry")

                resolved = registry.resolve(str(bundle_root))
                self.assertEqual(resolved, str(runtime_dir.resolve()))
                self.assertEqual(registry.resolve("voice-agent-bundle"), str(runtime_dir.resolve()))


if __name__ == "__main__":
    unittest.main()
