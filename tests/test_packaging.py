from __future__ import annotations

import tomllib
import unittest
from pathlib import Path


class PackagingMetadataTests(unittest.TestCase):
    def test_pyproject_uses_isolated_dwhisper_metadata(self) -> None:
        data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

        self.assertEqual(data["project"]["name"], "daydream-whisper")
        self.assertEqual(data["project"]["scripts"], {"dwhisper": "dwhisper.cli:cli"})
        self.assertEqual(data["tool"]["setuptools"]["packages"]["find"]["include"], ["dwhisper*"])


if __name__ == "__main__":
    unittest.main()
