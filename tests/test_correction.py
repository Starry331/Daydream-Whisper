from __future__ import annotations

import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dwhisper.correction import CorrectionConfig, TranscriptCorrector, load_correction_config


@dataclass(slots=True)
class DummyResult:
    text: str
    segments: list[dict[str, Any]] = field(default_factory=list)


class CorrectionTests(unittest.TestCase):
    def test_hotword_biasing_mentions_each_hotword_without_duplication(self) -> None:
        corrector = TranscriptCorrector(
            config=CorrectionConfig(hotwords=["MLX", "Whisper"])
        )

        prompt = corrector.biased_initial_prompt("Existing prompt mentioning MLX already.")

        self.assertIsNotNone(prompt)
        assert prompt is not None
        self.assertIn("Existing prompt mentioning MLX already.", prompt)
        self.assertIn("Whisper", prompt)
        self.assertEqual(prompt.lower().count("mlx"), 1)

    def test_correct_text_collapses_character_runs_and_repeated_phrases(self) -> None:
        corrector = TranscriptCorrector(config=CorrectionConfig())

        corrected = corrector.correct_text("sooooo again again again again")

        self.assertEqual(corrected, "soooo again")

    def test_correct_text_applies_vocabulary_with_boundaries_and_case(self) -> None:
        corrector = TranscriptCorrector(
            config=CorrectionConfig(
                vocabulary={"helo": "hello"},
                collapse_phrase_repeats=False,
            )
        )

        corrected = corrector.correct_text("helo HELO Helo shelloworld")

        self.assertEqual(corrected, "hello HELLO Hello shelloworld")

    def test_correct_text_runs_regex_substitutions_and_skips_invalid_patterns(self) -> None:
        corrector = TranscriptCorrector(
            config=CorrectionConfig(
                regex_substitutions=[(r"\bteh\b", "the"), ("(", "ignored")]
            )
        )

        corrected = corrector.correct_text("teh cat")

        self.assertEqual(corrected, "the cat")

    def test_correct_text_normalizes_whitespace_and_optionally_capitalizes(self) -> None:
        corrector = TranscriptCorrector(
            config=CorrectionConfig(capitalize_sentences=True)
        )

        corrected = corrector.correct_text("hello  ,   world.  next sentence")

        self.assertEqual(corrected, "Hello, world. Next sentence")

    def test_correct_segments_drops_hallucinations_and_low_confidence_segments(self) -> None:
        corrector = TranscriptCorrector(
            config=CorrectionConfig(
                drop_low_confidence_segments=True,
                no_speech_drop_threshold=0.8,
                avg_logprob_drop_threshold=-1.5,
            )
        )

        corrected = corrector.correct_segments(
            [
                {"id": 1, "text": "Thanks for watching"},
                {
                    "id": 2,
                    "text": "Thanks for watching the MLX whisper demo in full.",
                    "no_speech_prob": 0.1,
                    "avg_logprob": -0.1,
                },
                {
                    "id": 3,
                    "text": "Real speech but likely silence.",
                    "no_speech_prob": 0.95,
                },
                {
                    "id": 4,
                    "text": "Real speech but uncertain.",
                    "avg_logprob": -2.0,
                },
            ]
        )

        self.assertEqual([segment["id"] for segment in corrected], [2])

    def test_apply_mutates_result_and_noop_returns_unchanged(self) -> None:
        result = DummyResult(text="helo", segments=[{"text": "helo"}])
        corrector = TranscriptCorrector(
            config=CorrectionConfig(vocabulary={"helo": "hello"})
        )

        corrected = corrector.apply(result)

        self.assertIs(corrected, result)
        self.assertEqual(result.text, "hello")
        self.assertEqual(result.segments[0]["text"], "hello")

        noop_result = DummyResult(text="helo", segments=[{"text": "helo"}])
        noop_corrector = TranscriptCorrector(config=CorrectionConfig(enabled=False))

        unchanged = noop_corrector.apply(noop_result)

        self.assertIs(unchanged, noop_result)
        self.assertEqual(noop_result.text, "helo")
        self.assertEqual(noop_result.segments[0]["text"], "helo")

    def test_load_correction_config_merges_separate_vocabulary_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            corrections_path = Path(tmpdir) / "corrections.yaml"
            vocabulary_path = Path(tmpdir) / "vocabulary.yaml"
            corrections_path.write_text(
                "hotwords:\n  - MLX\nvocabulary:\n  mlxwhspr: mlx-whisper\n",
                encoding="utf-8",
            )
            vocabulary_path.write_text(
                "vocabulary:\n  helo: hello\n",
                encoding="utf-8",
            )

            config = load_correction_config(
                corrections_path=corrections_path,
                vocabulary_path=vocabulary_path,
            )

        self.assertEqual(config.hotwords, ["MLX"])
        self.assertEqual(
            config.vocabulary,
            {"mlxwhspr": "mlx-whisper", "helo": "hello"},
        )


class AutoPunctuationTests(unittest.TestCase):
    def test_auto_punctuation_collapses_repeated_terminals(self) -> None:
        corrector = TranscriptCorrector(config=CorrectionConfig())

        self.assertEqual(corrector.correct_text("wait... really??"), "wait. really?")

    def test_auto_punctuation_inserts_space_after_latin_punct(self) -> None:
        corrector = TranscriptCorrector(config=CorrectionConfig())

        self.assertEqual(
            corrector.correct_text("hello,world.how are you"),
            "hello, world. how are you",
        )

    def test_auto_punctuation_leaves_cjk_punct_untouched(self) -> None:
        corrector = TranscriptCorrector(config=CorrectionConfig())

        # Three full-width periods should collapse to one; no Latin spacing
        # rules should fire on the CJK text.
        self.assertEqual(corrector.correct_text("你好。。。世界"), "你好。世界")

    def test_ensure_terminal_punctuation_prefers_full_width_for_cjk(self) -> None:
        corrector = TranscriptCorrector(
            config=CorrectionConfig(ensure_terminal_punctuation=True)
        )

        self.assertEqual(corrector.correct_text("你好世界"), "你好世界。")
        self.assertEqual(corrector.correct_text("hello world"), "hello world.")

    def test_auto_punctuation_can_be_disabled(self) -> None:
        corrector = TranscriptCorrector(
            config=CorrectionConfig(auto_punctuation=False)
        )

        # Without auto_punctuation the stray run and missing-space artefacts
        # come through exactly as Whisper produced them.
        self.assertEqual(
            corrector.correct_text("wait... really??"),
            "wait... really??",
        )


if __name__ == "__main__":
    unittest.main()
