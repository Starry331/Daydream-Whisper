from __future__ import annotations

import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rich.console import Console

from tests.helpers import (
    reload_module,
    write_fake_hf_style_whisper_model,
    write_fake_local_whisper_model,
    write_fake_multimodal_bundle,
)


class ModelTests(unittest.TestCase):
    def test_pull_installs_fixture_when_hub_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir) / "home"
            cache = Path(tmpdir) / "cache"
            output = io.StringIO()

            with mock.patch.dict(
                os.environ,
                {
                    "DAYDREAM_HOME": str(home),
                    "DAYDREAM_CACHE_DIR": str(cache),
                },
                clear=False,
            ):
                reload_module("dwhisper.config")
                reload_module("dwhisper.registry")
                models = reload_module("dwhisper.models")
                models.console = Console(file=output, force_terminal=False, color_system=None)
                models.progress_console = Console(file=io.StringIO(), force_terminal=False, color_system=None)

                def fake_snapshot_download(*args, dry_run=False, **kwargs):
                    if dry_run:
                        class Entry:
                            file_size = 1024
                            will_download = True

                        return [Entry()]
                    raise RuntimeError("offline")

                with mock.patch.object(models, "snapshot_download", side_effect=fake_snapshot_download):
                    models.pull_model("whisper:base")

                repo_id = "mlx-community/whisper-base-mlx"
                model_path = models.get_model_path(repo_id)
                self.assertIsNotNone(model_path)
                self.assertTrue(model_path.exists())
                self.assertTrue(models.is_fixture_model(repo_id))

                output.truncate(0)
                output.seek(0)
                models.list_models()
                rendered = output.getvalue()
                self.assertIn("whisper:base", rendered)
                self.assertIn("cache", rendered)

    def test_validate_runtime_model_rejects_non_whisper_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir) / "home"
            model_dir = Path(tmpdir) / "bad-model"
            home.mkdir(parents=True, exist_ok=True)
            write_fake_local_whisper_model(model_dir)
            (model_dir / "config.json").write_text('{"model_type":"qwen","d_model":512,"n_mels":80}', encoding="utf-8")

            with mock.patch.dict(os.environ, {"DAYDREAM_HOME": str(home)}, clear=False):
                reload_module("dwhisper.config")
                reload_module("dwhisper.registry")
                models = reload_module("dwhisper.models")

                with self.assertRaisesRegex(ValueError, "not a Whisper checkpoint"):
                    models.validate_runtime_model(str(model_dir), source_name="bad-model")

    def test_validate_runtime_model_accepts_hf_style_whisper_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir) / "home"
            model_dir = write_fake_hf_style_whisper_model(Path(tmpdir) / "hf-style-model")
            home.mkdir(parents=True, exist_ok=True)

            with mock.patch.dict(os.environ, {"DAYDREAM_HOME": str(home)}, clear=False):
                reload_module("dwhisper.config")
                reload_module("dwhisper.registry")
                models = reload_module("dwhisper.models")

                validated = models.validate_runtime_model(str(model_dir), source_name="hf-style-model")

        self.assertEqual(validated, model_dir.resolve())

    def test_validate_runtime_model_accepts_multimodal_bundle_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir) / "home"
            bundle_root, runtime_dir = write_fake_multimodal_bundle(Path(tmpdir) / "bundle-model")
            home.mkdir(parents=True, exist_ok=True)

            with mock.patch.dict(os.environ, {"DAYDREAM_HOME": str(home)}, clear=False):
                reload_module("dwhisper.config")
                reload_module("dwhisper.registry")
                models = reload_module("dwhisper.models")

                validated = models.validate_runtime_model(str(bundle_root), source_name="bundle-model")

        self.assertEqual(validated, runtime_dir.resolve())


if __name__ == "__main__":
    unittest.main()
