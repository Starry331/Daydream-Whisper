from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request


POSTPROCESS_MODES = {
    "clean",
    "summary",
    "meeting-notes",
    "speaker-format",
}

POSTPROCESS_MODE_MAX_TOKENS = {
    "clean": 256,
    "summary": 768,
    "meeting-notes": 2048,
    "speaker-format": 1024,
}

DEFAULT_SYSTEM_PROMPTS = {
    "clean": (
        "You clean speech-to-text transcripts. "
        "Fix punctuation, casing, spacing, and obvious ASR mistakes. "
        "Restore sentence boundaries and insert appropriate commas, periods, "
        "question marks, and quotation marks so the text reads naturally. "
        "Use language-appropriate punctuation: full-width (。，？！：；“”) for "
        "Chinese/Japanese/Korean text, half-width (.,?!:;\"') for English and "
        "other Latin-script text; never mix the two within a single sentence. "
        "Break long runs into separate sentences and add paragraph breaks "
        "between clearly distinct topics. Capitalize proper nouns and the "
        "first word of each sentence in Latin-script output. "
        "Preserve the original language, meaning, and factual content. "
        "Do not summarize. Do not translate. Do not invent speakers or missing details. "
        "Return only the cleaned transcript with no commentary, no Markdown "
        "fences, and no prefix like 'Here is the cleaned text:'."
    ),
    "summary": (
        "You summarize transcripts. "
        "Return a concise factual summary in the same language as the transcript. "
        "Do not invent facts. Return only the summary."
    ),
    "meeting-notes": (
        "You turn transcripts into structured meeting notes. "
        "Return a short title, key points, decisions, and action items in the same language as the transcript. "
        "Do not invent facts. Return only the notes."
    ),
    "speaker-format": (
        "You normalize speaker formatting in transcripts. "
        "If speaker labels already exist, make them consistent and readable. "
        "Do not invent speaker identities or merge separate speakers without evidence. "
        "Return only the formatted transcript."
    ),
}


def default_max_tokens_for_mode(mode: str) -> int:
    normalized = (_normalize_text(mode) or "clean").lower()
    return POSTPROCESS_MODE_MAX_TOKENS.get(normalized, POSTPROCESS_MODE_MAX_TOKENS["clean"])


def _normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_base_url(base_url: str) -> str:
    parsed = urllib_parse.urlsplit(base_url.strip())
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("postprocess_base_url must be a valid http(s) URL.")

    path = parsed.path.rstrip("/")
    if path.endswith("/chat/completions"):
        final_path = path
    elif path.endswith("/v1"):
        final_path = f"{path}/chat/completions"
    elif not path:
        final_path = "/v1/chat/completions"
    else:
        final_path = f"{path}/v1/chat/completions"
    return urllib_parse.urlunsplit((parsed.scheme, parsed.netloc, final_path, "", ""))


def _extract_message_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()
                if isinstance(content, list):
                    parts: list[str] = []
                    for item in content:
                        if not isinstance(item, dict):
                            continue
                        text = item.get("text")
                        if text is not None and str(text).strip():
                            parts.append(str(text).strip())
                    if parts:
                        return "\n".join(parts).strip()
            text = first.get("text")
            if text is not None and str(text).strip():
                return str(text).strip()

    text = payload.get("text")
    if text is not None and str(text).strip():
        return str(text).strip()
    raise RuntimeError("Post-process endpoint returned no message content.")


POSTPROCESS_BACKENDS = {"auto", "http", "mlx"}


@dataclass(slots=True)
class PostProcessOptions:
    enabled: bool = False
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    mode: str = "clean"
    prompt: str | None = None
    timeout: float = 30.0
    backend: str = "auto"
    max_tokens: int | None = None
    max_tokens_explicit: bool = False

    def __post_init__(self) -> None:
        self.model = _normalize_text(self.model)
        self.base_url = _normalize_text(self.base_url)
        self.api_key = _normalize_text(self.api_key)
        self.prompt = _normalize_text(self.prompt)
        self.mode = (_normalize_text(self.mode) or "clean").lower()
        if self.mode not in POSTPROCESS_MODES:
            raise ValueError(
                "postprocess_mode must be one of: clean, summary, meeting-notes, speaker-format."
            )
        self.timeout = max(1.0, float(self.timeout))
        backend = (_normalize_text(self.backend) or "auto").lower()
        if backend not in POSTPROCESS_BACKENDS:
            raise ValueError("postprocess_backend must be one of: auto, http, mlx.")
        self.backend = backend
        if self.max_tokens is None:
            tokens = default_max_tokens_for_mode(self.mode)
            self.max_tokens_explicit = False
        else:
            try:
                tokens = int(self.max_tokens)
            except (TypeError, ValueError):
                tokens = default_max_tokens_for_mode(self.mode)
                self.max_tokens_explicit = False
        self.max_tokens = max(32, tokens)

    def resolved_backend(self) -> str:
        if self.backend != "auto":
            return self.backend
        return "http" if self.base_url else "mlx"

    def is_configured(self) -> bool:
        if not self.enabled or self.model is None:
            return False
        if self.resolved_backend() == "mlx":
            return True
        return self.base_url is not None


Requester = Callable[[str, dict[str, str], dict[str, Any], float], dict[str, Any]]


@dataclass(slots=True)
class OpenAICompatPostProcessor:
    options: PostProcessOptions
    requester: Requester | None = None

    def _request(
        self,
        endpoint: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout: float,
    ) -> dict[str, Any]:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib_request.Request(endpoint, data=data, headers=headers, method="POST")
        try:
            with urllib_request.urlopen(request, timeout=timeout) as response:
                raw_body = response.read()
        except urllib_error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"Post-process endpoint returned HTTP {exc.code}: {detail or exc.reason}"
            ) from exc
        except urllib_error.URLError as exc:
            raise RuntimeError(f"Could not reach post-process endpoint: {exc.reason}") from exc

        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except Exception as exc:
            raise RuntimeError("Post-process endpoint returned malformed JSON.") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("Post-process endpoint returned an invalid JSON payload.")
        return payload

    def _build_messages(self, *, transcript: str, language: str | None) -> list[dict[str, str]]:
        system_prompt = DEFAULT_SYSTEM_PROMPTS[self.options.mode]
        if self.options.prompt is not None:
            user_prompt = self.options.prompt.format(
                transcript=transcript,
                language=language or "auto",
                mode=self.options.mode,
            )
        else:
            user_prompt = (
                f"Mode: {self.options.mode}\n"
                f"Language: {language or 'auto'}\n\n"
                "Transcript:\n"
                f"{transcript}"
            )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def process_text(self, *, transcript: str, language: str | None = None) -> str:
        if not self.options.enabled:
            return transcript
        if self.options.model is None:
            raise ValueError("postprocess_model is required when post-processing is enabled.")
        if self.options.base_url is None:
            raise ValueError("postprocess_base_url is required when post-processing is enabled.")

        endpoint = _normalize_base_url(self.options.base_url)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.options.api_key or 'dwhisper-local'}",
        }
        payload = {
            "model": self.options.model,
            "messages": self._build_messages(transcript=transcript, language=language),
            "temperature": 0,
            "stream": False,
        }
        requester = self.requester or self._request
        response = requester(endpoint, headers, payload, self.options.timeout)
        return _extract_message_text(response)

    def stream_text(
        self, *, transcript: str, language: str | None = None
    ) -> Iterator[str]:
        """Yield incremental content deltas from an OpenAI-compatible server.

        Falls back to a single-chunk yield of ``process_text`` if the backend
        does not honor ``stream=true`` or if the caller injected a ``requester``
        (tests) that is not stream-aware.
        """

        if not self.options.enabled:
            if transcript:
                yield transcript
            return
        if self.options.model is None:
            raise ValueError("postprocess_model is required when post-processing is enabled.")
        if self.options.base_url is None:
            raise ValueError("postprocess_base_url is required when post-processing is enabled.")

        if self.requester is not None:
            # The test hook bypasses HTTP entirely — emulate streaming by
            # splitting on whitespace so tests exercise the SSE plumbing.
            full = self.process_text(transcript=transcript, language=language)
            for chunk in _split_for_fake_stream(full):
                yield chunk
            return

        endpoint = _normalize_base_url(self.options.base_url)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.options.api_key or 'dwhisper-local'}",
            "Accept": "text/event-stream",
        }
        payload = {
            "model": self.options.model,
            "messages": self._build_messages(transcript=transcript, language=language),
            "temperature": 0,
            "stream": True,
        }
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib_request.Request(endpoint, data=data, headers=headers, method="POST")
        try:
            response = urllib_request.urlopen(request, timeout=self.options.timeout)
        except urllib_error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"Post-process endpoint returned HTTP {exc.code}: {detail or exc.reason}"
            ) from exc
        except urllib_error.URLError as exc:
            raise RuntimeError(f"Could not reach post-process endpoint: {exc.reason}") from exc

        try:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                payload_part = line[5:].strip()
                if not payload_part or payload_part == "[DONE]":
                    if payload_part == "[DONE]":
                        break
                    continue
                try:
                    chunk = json.loads(payload_part)
                except Exception:
                    continue
                delta = _extract_stream_delta(chunk)
                if delta:
                    yield delta
        finally:
            response.close()

    def apply(self, result: Any) -> None:
        _apply_processor(self, result)


def _extract_stream_delta(payload: dict[str, Any]) -> str:
    """Pull the incremental text out of one OpenAI-compat SSE payload."""

    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    delta = first.get("delta") or first.get("message")
    if isinstance(delta, dict):
        content = delta.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            if parts:
                return "".join(parts)
    text = first.get("text")
    if isinstance(text, str):
        return text
    return ""


def _split_for_fake_stream(text: str) -> list[str]:
    """Break a full response into chunks so tests can observe multiple deltas."""

    if not text:
        return []
    chunks: list[str] = []
    buffer = ""
    for ch in text:
        buffer += ch
        if ch.isspace():
            chunks.append(buffer)
            buffer = ""
    if buffer:
        chunks.append(buffer)
    return chunks


MLXGenerator = Callable[[str, str, int, float], str]
MLXStreamGenerator = Callable[[str, str, int, float], Iterable[str]]


# Process-wide cache so realtime listen / repeated transcribe_file calls reuse
# the MLX LM instead of re-loading it per chunk.
_MLX_MODEL_CACHE: dict[str, tuple[Any, Any]] = {}
_MLX_MODEL_CACHE_LOCK = threading.Lock()


def _load_mlx_model(model_name: str) -> tuple[Any, Any]:
    with _MLX_MODEL_CACHE_LOCK:
        cached = _MLX_MODEL_CACHE.get(model_name)
        if cached is not None:
            return cached
    try:
        from mlx_lm import load as mlx_load  # type: ignore
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "mlx-lm is not installed. Install it with `pip install mlx-lm` to use the "
            "local MLX post-process backend, or set postprocess_backend=http."
        ) from exc
    loaded = mlx_load(model_name)
    with _MLX_MODEL_CACHE_LOCK:
        # Someone else may have loaded it while we were downloading; keep theirs.
        _MLX_MODEL_CACHE.setdefault(model_name, loaded)
        return _MLX_MODEL_CACHE[model_name]


def clear_mlx_model_cache() -> None:
    """Drop all cached MLX LM instances. Mostly used by tests."""
    with _MLX_MODEL_CACHE_LOCK:
        _MLX_MODEL_CACHE.clear()


@dataclass(slots=True)
class MLXLMPostProcessor:
    """In-process post-processor that drives a local MLX LM (e.g. Qwen3.5-0.8B-MLX)."""

    options: PostProcessOptions
    generator: MLXGenerator | None = None
    stream_generator: MLXStreamGenerator | None = None

    def _ensure_loaded(self) -> tuple[Any, Any]:
        if self.options.model is None:
            raise ValueError("postprocess_model is required when post-processing is enabled.")
        return _load_mlx_model(self.options.model)

    def _build_prompt(self, *, transcript: str, language: str | None) -> str:
        system_prompt = DEFAULT_SYSTEM_PROMPTS[self.options.mode]
        if self.options.prompt is not None:
            user_prompt = self.options.prompt.format(
                transcript=transcript,
                language=language or "auto",
                mode=self.options.mode,
            )
        else:
            user_prompt = (
                f"Mode: {self.options.mode}\n"
                f"Language: {language or 'auto'}\n\n"
                "Transcript:\n"
                f"{transcript}"
            )
        # Only touch the on-disk MLX LM when no test hook is wired up. Any
        # injected (stream_)generator is responsible for its own prompt format.
        if self.generator is None and self.stream_generator is None:
            model, tokenizer = self._ensure_loaded()
        else:
            model, tokenizer = None, None
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            try:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                return tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                pass
        return f"<|system|>\n{system_prompt}\n<|user|>\n{user_prompt}\n<|assistant|>\n"

    def _default_generate(self, prompt: str, model_name: str, max_tokens: int, timeout: float) -> str:
        try:
            from mlx_lm import generate as mlx_generate  # type: ignore
        except Exception as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "mlx-lm is not installed. Install it with `pip install mlx-lm`."
            ) from exc
        model, tokenizer = self._ensure_loaded()
        try:
            output = mlx_generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                verbose=False,
            )
        except TypeError:
            output = mlx_generate(model, tokenizer, prompt, max_tokens)
        return str(output or "").strip()

    def process_text(self, *, transcript: str, language: str | None = None) -> str:
        if not self.options.enabled:
            return transcript
        if self.options.model is None:
            raise ValueError("postprocess_model is required when post-processing is enabled.")

        prompt = self._build_prompt(transcript=transcript, language=language)
        generator = self.generator or self._default_generate
        text = generator(prompt, self.options.model, self.options.max_tokens, self.options.timeout)
        text = (text or "").strip()
        if not text:
            raise RuntimeError("MLX post-process backend returned no content.")
        return text

    def _default_stream_generate(
        self, prompt: str, model_name: str, max_tokens: int, timeout: float
    ) -> Iterator[str]:
        try:
            from mlx_lm import stream_generate as mlx_stream_generate  # type: ignore
        except Exception:
            # mlx-lm without stream_generate (older) — degrade to one big chunk.
            text = self._default_generate(prompt, model_name, max_tokens, timeout)
            if text:
                yield text
            return

        model, tokenizer = self._ensure_loaded()
        try:
            iterator = mlx_stream_generate(
                model, tokenizer, prompt=prompt, max_tokens=max_tokens
            )
        except TypeError:
            iterator = mlx_stream_generate(model, tokenizer, prompt, max_tokens)
        for response in iterator:
            text = getattr(response, "text", None)
            if text is None:
                text = str(response or "")
            if text:
                yield text

    def stream_text(
        self, *, transcript: str, language: str | None = None
    ) -> Iterator[str]:
        """Yield text deltas from the MLX LM token stream."""

        if not self.options.enabled:
            if transcript:
                yield transcript
            return
        if self.options.model is None:
            raise ValueError("postprocess_model is required when post-processing is enabled.")

        prompt = self._build_prompt(transcript=transcript, language=language)
        if self.stream_generator is not None:
            iterator = self.stream_generator(
                prompt, self.options.model, self.options.max_tokens, self.options.timeout
            )
        elif self.generator is not None:
            # When only a synchronous generator is wired (e.g. tests), break
            # its output into whitespace-delimited chunks so the SSE pathway
            # gets exercised.
            full = self.generator(
                prompt, self.options.model, self.options.max_tokens, self.options.timeout
            )
            iterator = iter(_split_for_fake_stream(full or ""))
        else:
            iterator = self._default_stream_generate(
                prompt, self.options.model, self.options.max_tokens, self.options.timeout
            )
        emitted = False
        for chunk in iterator:
            if chunk:
                emitted = True
                yield chunk
        if not emitted:
            raise RuntimeError("MLX post-process backend returned no content.")

    def apply(self, result: Any) -> None:
        _apply_processor(self, result)


def _apply_processor(processor: Any, result: Any) -> None:
    options: PostProcessOptions = processor.options
    backend = options.resolved_backend() if options is not None else None
    transcript = _normalize_text(getattr(result, "text", None))
    if not transcript:
        setattr(
            result,
            "postprocess",
            {
                "enabled": options.enabled,
                "applied": False,
                "mode": options.mode,
                "model": options.model,
                "backend": backend,
            },
        )
        return

    processed = processor.process_text(transcript=transcript, language=getattr(result, "language", None))
    applied = bool(processed and processed.strip() and processed.strip() != transcript)
    if applied:
        if getattr(result, "raw_text", None) is None:
            setattr(result, "raw_text", transcript)
        setattr(result, "text", processed.strip())
    payload: dict[str, Any] = {
        "enabled": options.enabled,
        "applied": applied,
        "mode": options.mode,
        "model": options.model,
        "backend": backend,
    }
    if options.base_url is not None:
        payload["base_url"] = options.base_url
    setattr(result, "postprocess", payload)


def build_postprocessor(
    options: PostProcessOptions,
    *,
    http_factory: Callable[[PostProcessOptions], Any] | None = None,
    mlx_factory: Callable[[PostProcessOptions], Any] | None = None,
) -> Any:
    """Return the right backend instance for ``options``.

    The factory indirection exists purely so tests can inject fakes. In production
    the defaults build :class:`OpenAICompatPostProcessor` / :class:`MLXLMPostProcessor`.
    """

    backend = options.resolved_backend()
    if backend == "mlx":
        factory = mlx_factory or MLXLMPostProcessor
        return factory(options)
    factory = http_factory or OpenAICompatPostProcessor
    return factory(options)
