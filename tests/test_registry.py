from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests.helpers import reload_module


class RegistryTests(unittest.TestCase):
    def test_builtin_whisper_registry_and_user_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir) / "home"
            home.mkdir(parents=True, exist_ok=True)
            (home / "registry.yaml").write_text(
                "whisper:\n"
                "  base: acme/custom-whisper-base\n"
                "custom:\n"
                "  default: acme/custom-default\n"
                "  fast: acme/custom-fast\n",
                encoding="utf-8",
            )

            with mock.patch.dict(os.environ, {"DAYDREAM_HOME": str(home)}, clear=False):
                reload_module("dwhisper.config")
                registry = reload_module("dwhisper.registry")

                self.assertEqual(registry.resolve("whisper"), "mlx-community/whisper-large-v3-turbo")
                self.assertEqual(registry.resolve("whisper:base"), "acme/custom-whisper-base")
                self.assertEqual(registry.resolve("custom"), "acme/custom-default")
                self.assertEqual(registry.resolve("custom:fast"), "acme/custom-fast")
                self.assertEqual(registry.resolve("hf.co/acme/whisper-model"), "acme/whisper-model")


if __name__ == "__main__":
    unittest.main()
