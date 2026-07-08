"""LLM adapter.

Exposes a minimal :class:`LLM` protocol with ``chat(messages)`` and a single
implementation :class:`ChatLLM`. Keeping this behind a small interface means
tests can swap in a fake without touching the network, and swapping runners
later is a one-file change.

:class:`ChatLLM` talks to any HTTP endpoint that implements the OpenAI
chat-completions wire protocol (Ollama, LM Studio, vLLM, llama.cpp,
LocalAI, ...). By default it points at a local Ollama instance, so no
data leaves the machine. The class uses the ``openai`` Python SDK purely
as an HTTP client for that wire format — it does not imply we're calling
OpenAI's servers.

Messages use the OpenAI wire format (list of dicts with ``role`` and
``content``); ``chat`` returns the assistant reply as a plain string.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterator, List, Optional, Protocol, TypedDict

from app.config import get_settings


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_think_blocks(text: str) -> str:
    """Remove ``<think>...</think>`` reasoning blocks from a model reply.

    Some models (DeepSeek-R1, QwQ, etc.) emit an internal reasoning trace
    wrapped in ``<think>`` tags before the final answer. That's fine for
    the user-visible answer, but it must be stripped from any output that
    the app uses *internally* (e.g. as a retrieval query).

    Unclosed ``<think>`` (model ran out of tokens mid-reasoning) is also
    dropped: everything from ``<think>`` to end-of-string is removed.
    """
    if not text:
        return text
    cleaned = _THINK_BLOCK_RE.sub("", text)
    # Handle unclosed <think> from truncated output.
    idx = cleaned.lower().find("<think>")
    if idx != -1:
        cleaned = cleaned[:idx]
    return cleaned.strip()


class ChatMessage(TypedDict):
    role: str      # "system" | "user" | "assistant"
    content: str


class LLM(Protocol):
    """Interface implemented by every LLM backend."""

    def chat(self, messages: List[ChatMessage], *, temperature: float = 0.0, max_tokens: Optional[int] = None) -> str: ...

    def chat_stream(
        self,
        messages: List[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> Iterator[str]: ...


class LLMError(RuntimeError):
    """Raised when a backend fails after retries."""


# --------------------------------------------------------------------------- #
# ChatLLM                                                                     #
# --------------------------------------------------------------------------- #

class ChatLLM:
    """OpenAI-compatible chat completions adapter.

    Talks to any HTTP endpoint that speaks the OpenAI chat-completions
    protocol. The ``base_url`` decides where requests go (typically a
    local Ollama on ``http://localhost:11434/v1``).
    """

    def __init__(self, model: str, base_url: str, api_key: str = "unused") -> None:
        if not base_url:
            raise ValueError("LLM_BASE_URL is required for ChatLLM")
        if not model:
            raise ValueError("LLM_MODEL is required for ChatLLM")
        # Lazy import so tests without the openai dep don't fail at import time.
        from openai import OpenAI

        # ``api_key`` must be a non-empty string for the SDK; the local
        # runner ignores its value. Nothing is ever sent to OpenAI.
        self._client = OpenAI(api_key=api_key or "unused", base_url=base_url)
        self.model = model
        self.base_url = base_url

    def chat(
        self,
        messages: List[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> str:
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=list(messages),
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:  # pragma: no cover - network error passthrough
            raise LLMError(f"LLM call to {self.base_url} failed: {exc}") from exc

        content = resp.choices[0].message.content or ""
        return content.strip()

    def chat_stream(
        self,
        messages: List[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> Iterator[str]:
        """Yield content deltas as the model generates them.

        Uses the OpenAI-compatible ``stream=True`` mode. Each chunk is a
        plain string containing whatever new text arrived. Does **not**
        strip anything — the caller sees raw output including any
        ``<think>...</think>`` blocks from reasoning models.
        """
        try:
            stream = self._client.chat.completions.create(
                model=self.model,
                messages=list(messages),
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
        except Exception as exc:  # pragma: no cover - network error passthrough
            raise LLMError(f"LLM stream call to {self.base_url} failed: {exc}") from exc

        try:
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                # `delta.content` may be None on the first/last chunk.
                text = getattr(delta, "content", None) or ""
                if text:
                    yield text
        except Exception as exc:  # pragma: no cover
            raise LLMError(f"LLM stream call to {self.base_url} interrupted: {exc}") from exc


# --------------------------------------------------------------------------- #
# Factory                                                                     #
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=1)
def get_llm() -> LLM:
    """Return a cached LLM built from :func:`app.config.get_settings`."""
    s = get_settings()
    return ChatLLM(
        model=s.llm_model,
        base_url=s.llm_base_url,
        api_key=s.llm_api_key,
    )


def reset_llm_cache() -> None:
    """Reset the cached LLM (used in tests)."""
    get_llm.cache_clear()
