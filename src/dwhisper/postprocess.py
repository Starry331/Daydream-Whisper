from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request


POSTPROCESS_MODES = {
    "clean",
    "summary",
    "meeting-notes",
    "speaker-format",
}

DEFAULT_SYSTEM_PROMPTS = {
    "clean": (
        "You clean speech-to-text transcripts. "
        "Fix punctuation, casing, spacing, and obvious ASR mistakes. "
        "Preserve the original language, meaning, and factual content. "
        "Do not summarize. Do not translate. Do not invent speakers or missing details. "
        "Return only the cleaned transcript."
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


@dataclass(slots=True)
class PostProcessOptions:
    enabled: bool = False
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    mode: str = "clean"
    prompt: str | None = None
    timeout: float = 30.0

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

    def is_configured(self) -> bool:
        return self.enabled and self.model is not None and self.base_url is not None


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

    def apply(self, result: Any) -> None:
        transcript = _normalize_text(getattr(result, "text", None))
        if not transcript:
            setattr(
                result,
                "postprocess",
                {
                    "enabled": self.options.enabled,
                    "applied": False,
                    "mode": self.options.mode,
                    "model": self.options.model,
                },
            )
            return

        processed = self.process_text(transcript=transcript, language=getattr(result, "language", None))
        applied = bool(processed and processed.strip() and processed.strip() != transcript)
        if applied:
            if getattr(result, "raw_text", None) is None:
                setattr(result, "raw_text", transcript)
            setattr(result, "text", processed.strip())
        setattr(
            result,
            "postprocess",
            {
                "enabled": self.options.enabled,
                "applied": applied,
                "mode": self.options.mode,
                "model": self.options.model,
                "base_url": self.options.base_url,
            },
        )
