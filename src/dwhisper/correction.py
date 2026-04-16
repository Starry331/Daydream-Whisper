"""Transcript auto-correction: hotword biasing, hallucination filtering, normalization.

This module is intentionally dependency-free beyond stdlib + PyYAML so it can run
inside the persistent Whisper worker, the API server, and the CLI without
requiring MLX runtime initialization.
"""

from __future__ import annotations

import re
import unicodedata
from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml

# Common Whisper hallucination phrases that frequently appear on near-silent or
# music-only audio segments. Lower-cased on both sides during comparison.
COMMON_HALLUCINATIONS: tuple[str, ...] = (
    "thanks for watching",
    "thank you for watching",
    "thank you so much for watching",
    "please subscribe",
    "don't forget to subscribe",
    "like and subscribe",
    "see you next time",
    "see you in the next video",
    "subtitles by",
    "subtitled by",
    "thanks for listening",
    "stay tuned",
    "请订阅",
    "感谢观看",
    "感谢收看",
    "字幕由",
    "字幕组",
    "ご視聴ありがとうございました",
    "최고의 영상",
)
DEFAULT_INITIAL_PROMPT_TEMPLATE = "Glossary terms that may appear: {hotwords}."

_PUNCT_SPACE_RE = re.compile(r"\s+([,.!?;:%。、！？；：])")
_DOUBLE_SPACE_RE = re.compile(r"[ \t]+")
_LEADING_PUNCT_RE = re.compile(r"^[\s,.;:!?，。、；：！？]+")
_TRAILING_TERMINAL_RE = re.compile(r"[.!?。！？]\s*$")
_SENTENCE_SPLIT_RE = re.compile(r"([.!?。！？]+\s+)")


@dataclass(slots=True)
class CorrectionConfig:
    """Knobs for the post-transcription :class:`TranscriptCorrector`.

    The defaults are conservative: we only do safe, non-destructive cleanup
    unless callers explicitly opt into stronger normalization.
    """

    enabled: bool = True

    # Hotword / vocabulary biasing -------------------------------------------------
    hotwords: list[str] = field(default_factory=list)
    vocabulary: dict[str, str] = field(default_factory=dict)
    regex_substitutions: list[tuple[str, str]] = field(default_factory=list)
    initial_prompt_template: str = DEFAULT_INITIAL_PROMPT_TEMPLATE
    hotword_case_insensitive: bool = True
    apply_vocabulary_to_segments: bool = True

    # Hallucination filtering ------------------------------------------------------
    drop_hallucinations: bool = True
    extra_hallucination_phrases: list[str] = field(default_factory=list)
    drop_low_confidence_segments: bool = False
    no_speech_drop_threshold: float = 0.85
    avg_logprob_drop_threshold: float = -1.5

    # Repetition collapse ----------------------------------------------------------
    collapse_character_repeats: bool = True
    max_repeat_chars: int = 4
    collapse_phrase_repeats: bool = True
    max_phrase_repeats: int = 2

    # Whitespace + punctuation -----------------------------------------------------
    normalize_whitespace: bool = True
    capitalize_sentences: bool = False
    ensure_terminal_punctuation: bool = False
    terminal_punctuation: str = "."

    # Profanity filter -------------------------------------------------------------
    profanity_filter: bool = False
    profanity_words: list[str] = field(default_factory=list)
    profanity_mask: str = "***"

    def is_noop(self) -> bool:
        if not self.enabled:
            return True
        return not any(
            (
                self.hotwords,
                self.vocabulary,
                self.regex_substitutions,
                self.drop_hallucinations,
                self.extra_hallucination_phrases,
                self.drop_low_confidence_segments,
                self.collapse_character_repeats,
                self.collapse_phrase_repeats,
                self.normalize_whitespace,
                self.capitalize_sentences,
                self.ensure_terminal_punctuation,
                self.profanity_filter,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "hotwords": list(self.hotwords),
            "vocabulary": dict(self.vocabulary),
            "regex_substitutions": [list(item) for item in self.regex_substitutions],
            "initial_prompt_template": self.initial_prompt_template,
            "hotword_case_insensitive": self.hotword_case_insensitive,
            "apply_vocabulary_to_segments": self.apply_vocabulary_to_segments,
            "drop_hallucinations": self.drop_hallucinations,
            "extra_hallucination_phrases": list(self.extra_hallucination_phrases),
            "drop_low_confidence_segments": self.drop_low_confidence_segments,
            "no_speech_drop_threshold": self.no_speech_drop_threshold,
            "avg_logprob_drop_threshold": self.avg_logprob_drop_threshold,
            "collapse_character_repeats": self.collapse_character_repeats,
            "max_repeat_chars": self.max_repeat_chars,
            "collapse_phrase_repeats": self.collapse_phrase_repeats,
            "max_phrase_repeats": self.max_phrase_repeats,
            "normalize_whitespace": self.normalize_whitespace,
            "capitalize_sentences": self.capitalize_sentences,
            "ensure_terminal_punctuation": self.ensure_terminal_punctuation,
            "terminal_punctuation": self.terminal_punctuation,
            "profanity_filter": self.profanity_filter,
            "profanity_words": list(self.profanity_words),
            "profanity_mask": self.profanity_mask,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "CorrectionConfig":
        if not payload:
            return cls()
        if not isinstance(payload, dict):
            raise ValueError("CorrectionConfig payload must be a mapping.")

        substitutions_raw = payload.get("regex_substitutions") or []
        regex_substitutions: list[tuple[str, str]] = []
        for entry in substitutions_raw:
            if isinstance(entry, dict):
                pattern = str(entry.get("pattern") or entry.get("from") or "")
                replacement = str(entry.get("replacement") or entry.get("to") or "")
            elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                pattern = str(entry[0])
                replacement = str(entry[1])
            else:
                continue
            if not pattern:
                continue
            regex_substitutions.append((pattern, replacement))

        vocabulary_raw = payload.get("vocabulary") or {}
        vocabulary: dict[str, str] = {}
        if isinstance(vocabulary_raw, dict):
            for key, value in vocabulary_raw.items():
                if value is None:
                    continue
                vocabulary[str(key)] = str(value)
        elif isinstance(vocabulary_raw, list):
            # Allow [{from: foo, to: bar}, ...] form.
            for entry in vocabulary_raw:
                if not isinstance(entry, dict):
                    continue
                src = entry.get("from") or entry.get("term")
                dst = entry.get("to") or entry.get("replacement")
                if src is None or dst is None:
                    continue
                vocabulary[str(src)] = str(dst)

        def _coerce_str_list(value: Any) -> list[str]:
            if value is None:
                return []
            if isinstance(value, str):
                return [value]
            if isinstance(value, Iterable):
                return [str(item).strip() for item in value if str(item).strip()]
            return []

        return cls(
            enabled=bool(payload.get("enabled", True)),
            hotwords=_coerce_str_list(payload.get("hotwords")),
            vocabulary=vocabulary,
            regex_substitutions=regex_substitutions,
            initial_prompt_template=str(
                payload.get("initial_prompt_template", DEFAULT_INITIAL_PROMPT_TEMPLATE)
            ),
            hotword_case_insensitive=bool(payload.get("hotword_case_insensitive", True)),
            apply_vocabulary_to_segments=bool(payload.get("apply_vocabulary_to_segments", True)),
            drop_hallucinations=bool(payload.get("drop_hallucinations", True)),
            extra_hallucination_phrases=_coerce_str_list(
                payload.get("extra_hallucination_phrases")
            ),
            drop_low_confidence_segments=bool(
                payload.get("drop_low_confidence_segments", False)
            ),
            no_speech_drop_threshold=float(payload.get("no_speech_drop_threshold", 0.85)),
            avg_logprob_drop_threshold=float(payload.get("avg_logprob_drop_threshold", -1.5)),
            collapse_character_repeats=bool(
                payload.get("collapse_character_repeats", True)
            ),
            max_repeat_chars=max(1, int(payload.get("max_repeat_chars", 4) or 4)),
            collapse_phrase_repeats=bool(payload.get("collapse_phrase_repeats", True)),
            max_phrase_repeats=max(1, int(payload.get("max_phrase_repeats", 2) or 2)),
            normalize_whitespace=bool(payload.get("normalize_whitespace", True)),
            capitalize_sentences=bool(payload.get("capitalize_sentences", False)),
            ensure_terminal_punctuation=bool(
                payload.get("ensure_terminal_punctuation", False)
            ),
            terminal_punctuation=str(payload.get("terminal_punctuation", ".")),
            profanity_filter=bool(payload.get("profanity_filter", False)),
            profanity_words=_coerce_str_list(payload.get("profanity_words")),
            profanity_mask=str(payload.get("profanity_mask", "***")),
        )

    def merged_with(self, other: "CorrectionConfig | None") -> "CorrectionConfig":
        if other is None:
            return self
        merged_vocabulary = dict(self.vocabulary)
        merged_vocabulary.update(other.vocabulary)
        merged_substitutions = list(self.regex_substitutions) + list(other.regex_substitutions)
        merged_hotwords = list(dict.fromkeys([*self.hotwords, *other.hotwords]))
        merged_extra = list(
            dict.fromkeys([*self.extra_hallucination_phrases, *other.extra_hallucination_phrases])
        )
        merged_profanity = list(dict.fromkeys([*self.profanity_words, *other.profanity_words]))
        return replace(
            other,
            hotwords=merged_hotwords,
            vocabulary=merged_vocabulary,
            regex_substitutions=merged_substitutions,
            extra_hallucination_phrases=merged_extra,
            profanity_words=merged_profanity,
        )


def _normalize_phrase(text: str) -> str:
    cleaned = unicodedata.normalize("NFKC", text or "")
    cleaned = re.sub(r"[\s\u3000]+", " ", cleaned).strip().lower()
    cleaned = cleaned.strip(" .!?,;:'\"()[]{}")
    return cleaned


def _is_hallucinated_segment(text: str, hallucinations: Sequence[str]) -> bool:
    normalized = _normalize_phrase(text)
    if not normalized:
        return False
    for phrase in hallucinations:
        candidate = _normalize_phrase(phrase)
        if not candidate:
            continue
        if normalized == candidate or normalized.startswith(candidate) or candidate in normalized:
            # Only drop when the hallucination dominates the segment to avoid
            # nuking real transcripts that happen to mention "thanks".
            if len(candidate) >= 0.6 * max(1, len(normalized)):
                return True
    return False


def _collapse_character_repeats(text: str, max_run: int) -> str:
    if max_run <= 0:
        return text
    pattern = re.compile(r"(\w)\1{" + str(max_run) + r",}")
    return pattern.sub(lambda match: match.group(1) * max_run, text)


def _collapse_phrase_repeats(text: str, max_repeats: int) -> str:
    if max_repeats < 1:
        return text
    pattern = re.compile(
        r"(\b[\w'\-]+(?:\s+[\w'\-]+){0,4}\b)(?:[\s,.;!?]+\1){" + str(max_repeats) + r",}",
        flags=re.IGNORECASE,
    )
    previous = None
    current = text
    while previous != current:
        previous = current
        current = pattern.sub(lambda match: match.group(1), current)
    return current


def _capitalize_sentences(text: str) -> str:
    if not text:
        return text
    parts = _SENTENCE_SPLIT_RE.split(text)
    rebuilt: list[str] = []
    for fragment in parts:
        if not fragment:
            rebuilt.append(fragment)
            continue
        stripped = fragment.lstrip()
        if not stripped:
            rebuilt.append(fragment)
            continue
        leading_ws = fragment[: len(fragment) - len(stripped)]
        rebuilt.append(leading_ws + stripped[0].upper() + stripped[1:])
    return "".join(rebuilt)


@dataclass(slots=True)
class TranscriptCorrector:
    """Apply :class:`CorrectionConfig` to text and structured Whisper results."""

    config: CorrectionConfig = field(default_factory=CorrectionConfig)

    @property
    def hallucination_phrases(self) -> tuple[str, ...]:
        if not self.config.drop_hallucinations:
            return tuple(self.config.extra_hallucination_phrases)
        return (*COMMON_HALLUCINATIONS, *self.config.extra_hallucination_phrases)

    def biased_initial_prompt(self, base_prompt: str | None) -> str | None:
        prompt = (base_prompt or "").strip()
        if not self.config.hotwords:
            return prompt or None
        unique_hotwords = list(
            dict.fromkeys(word.strip() for word in self.config.hotwords if word.strip())
        )
        missing_hotwords = [
            word
            for word in unique_hotwords
            if not prompt or re.search(re.escape(word), prompt, flags=re.IGNORECASE) is None
        ]
        formatted = ", ".join(missing_hotwords)
        if not formatted:
            return prompt or None
        try:
            hotword_clause = self.config.initial_prompt_template.format(hotwords=formatted)
        except (IndexError, KeyError):
            hotword_clause = f"Glossary: {formatted}."
        if not prompt:
            return hotword_clause
        return f"{prompt} {hotword_clause}".strip()

    def correct_text(self, text: str) -> str:
        if self.config.is_noop():
            return text or ""

        cleaned = (text or "").replace("\u200b", "").replace("\u200c", "")
        cleaned = unicodedata.normalize("NFC", cleaned)

        if self.config.collapse_character_repeats:
            cleaned = _collapse_character_repeats(cleaned, self.config.max_repeat_chars)

        if self.config.collapse_phrase_repeats:
            cleaned = _collapse_phrase_repeats(cleaned, self.config.max_phrase_repeats)

        if self.config.vocabulary:
            cleaned = self._apply_vocabulary(cleaned)

        if self.config.regex_substitutions:
            cleaned = self._apply_regex_substitutions(cleaned)

        if self.config.profanity_filter and self.config.profanity_words:
            cleaned = self._apply_profanity_filter(cleaned)

        if self.config.normalize_whitespace:
            cleaned = _DOUBLE_SPACE_RE.sub(" ", cleaned)
            cleaned = _PUNCT_SPACE_RE.sub(r"\1", cleaned)
            cleaned = _LEADING_PUNCT_RE.sub("", cleaned).strip()

        if self.config.capitalize_sentences:
            cleaned = _capitalize_sentences(cleaned)

        if (
            self.config.ensure_terminal_punctuation
            and cleaned
            and not _TRAILING_TERMINAL_RE.search(cleaned)
        ):
            cleaned = cleaned.rstrip() + (self.config.terminal_punctuation or ".")

        return cleaned

    def correct_segments(self, segments: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.config.is_noop():
            return [deepcopy(segment) for segment in segments]

        hallucinations = self.hallucination_phrases
        cleaned_segments: list[dict[str, Any]] = []
        for raw in segments:
            segment = deepcopy(raw)
            text = str(segment.get("text", ""))
            if self.config.drop_hallucinations and _is_hallucinated_segment(text, hallucinations):
                continue
            if self.config.drop_low_confidence_segments and self._segment_is_low_confidence(segment):
                continue
            if self.config.apply_vocabulary_to_segments:
                segment["text"] = self.correct_text(text)
            cleaned_segments.append(segment)
        return cleaned_segments

    def apply(self, result: Any) -> Any:
        """Mutate-and-return helper for any object exposing ``text`` and ``segments``."""

        if self.config.is_noop():
            return result

        try:
            segments = list(getattr(result, "segments", []) or [])
        except TypeError:
            segments = []
        cleaned_segments = self.correct_segments(segments)

        text = str(getattr(result, "text", "") or "")
        if not text and cleaned_segments:
            text = " ".join(str(segment.get("text", "")).strip() for segment in cleaned_segments).strip()
        cleaned_text = self.correct_text(text)

        try:
            result.text = cleaned_text
            result.segments = cleaned_segments
        except AttributeError:
            return result
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _segment_is_low_confidence(self, segment: dict[str, Any]) -> bool:
        no_speech = segment.get("no_speech_prob")
        if isinstance(no_speech, (int, float)) and float(no_speech) >= self.config.no_speech_drop_threshold:
            return True
        avg_logprob = segment.get("avg_logprob")
        if isinstance(avg_logprob, (int, float)) and float(avg_logprob) <= self.config.avg_logprob_drop_threshold:
            return True
        return False

    def _apply_vocabulary(self, text: str) -> str:
        if not self.config.vocabulary:
            return text
        # Replace longest keys first so that substrings do not steal the match.
        for term in sorted(self.config.vocabulary.keys(), key=len, reverse=True):
            replacement = self.config.vocabulary[term]
            if not term:
                continue
            try:
                pattern = re.compile(r"(?<!\w)" + re.escape(term) + r"(?!\w)", flags=re.IGNORECASE)
            except re.error:
                continue
            text = pattern.sub(
                lambda match: self._preserve_replacement_case(match.group(0), replacement),
                text,
            )
        return text

    def _apply_regex_substitutions(self, text: str) -> str:
        for pattern, replacement in self.config.regex_substitutions:
            try:
                text = re.sub(pattern, replacement, text)
            except re.error:
                continue
        return text

    def _apply_profanity_filter(self, text: str) -> str:
        mask = self.config.profanity_mask or "***"
        for word in self.config.profanity_words:
            cleaned_word = word.strip()
            if not cleaned_word:
                continue
            try:
                pattern = re.compile(r"(?<!\w)" + re.escape(cleaned_word) + r"(?!\w)", flags=re.IGNORECASE)
            except re.error:
                continue
            text = pattern.sub(mask, text)
        return text

    def _preserve_replacement_case(self, original: str, replacement: str) -> str:
        if not original:
            return replacement
        if original.isupper():
            return replacement.upper()
        if original.islower():
            return replacement.lower()
        if original.istitle():
            return replacement.title()
        return replacement


def _safe_load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def load_correction_config(
    *,
    corrections_path: Path | None = None,
    vocabulary_path: Path | None = None,
) -> CorrectionConfig:
    """Load correction settings from ``~/.dwhisper/corrections.yaml``.

    A separate ``vocabulary.yaml`` file (if present) is merged into the
    correction config so that vocabulary is easy to manage independently.
    """

    base_payload: dict[str, Any] = {}
    if corrections_path is not None:
        base_payload = _safe_load_yaml(corrections_path)

    config = CorrectionConfig.from_dict(base_payload)

    if vocabulary_path is not None and vocabulary_path.exists():
        vocab_payload = _safe_load_yaml(vocabulary_path)
        vocab_config = CorrectionConfig.from_dict({"vocabulary": vocab_payload.get("vocabulary", vocab_payload)})
        config = vocab_config.merged_with(config)
    return config
