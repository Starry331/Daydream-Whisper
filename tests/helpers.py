from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np


def reload_module(name: str):
    return importlib.reload(importlib.import_module(name))


def write_fake_local_whisper_model(path: Path, *, fixture: bool = False) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    config = {
        "model_type": "whisper",
        "d_model": 512,
        "encoder_layers": 6,
        "decoder_layers": 6,
        "num_mel_bins": 80,
        "n_mels": 80,
        "vocab_size": 51865,
    }
    if fixture:
        config["daydream_fixture"] = True

    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (path / "preprocessor_config.json").write_text(
        json.dumps({"feature_size": 80, "sampling_rate": 16000}),
        encoding="utf-8",
    )
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")
    np.savez(path / "weights.npz", encoder=np.zeros((1,), dtype=np.float32))
    np.savez(path / "mel_filters.npz", mel=np.zeros((80, 201), dtype=np.float32))
    if fixture:
        (path / "daydream_fixture.json").write_text("{}", encoding="utf-8")
    return path


def write_fake_hf_style_whisper_model(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    config = {
        "model_type": "whisper",
        "n_audio_state": 512,
        "n_audio_layer": 6,
        "n_text_layer": 6,
        "n_mels": 80,
        "n_vocab": 51865,
    }
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    np.savez(path / "weights.npz", encoder=np.zeros((1,), dtype=np.float32))
    return path


def write_fake_multimodal_bundle(path: Path, *, subdir: str = "speech_encoder") -> tuple[Path, Path]:
    bundle_root = path
    runtime_dir = path / subdir
    bundle_root.mkdir(parents=True, exist_ok=True)
    (bundle_root / "config.json").write_text(
        json.dumps({"model_type": "multimodal-audio", "components": [subdir]}),
        encoding="utf-8",
    )
    write_fake_local_whisper_model(runtime_dir)
    return bundle_root, runtime_dir
