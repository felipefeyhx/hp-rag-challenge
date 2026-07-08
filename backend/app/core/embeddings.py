"""Embedding backend.

Thin adapter over :class:`langchain_openai.OpenAIEmbeddings` exposing two
methods (``embed_documents``, ``embed_query``) so the rest of the code
depends only on this small protocol and tests can inject a fake.

:class:`TextEmbedder` talks to any HTTP endpoint that implements the
OpenAI embeddings wire protocol (Ollama, LM Studio, LocalAI, ...). By
default it points at a local Ollama instance so no data leaves the
machine. The langchain-openai client is used purely as an HTTP client
for that wire format — it does not imply we're calling OpenAI's servers.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List, Protocol

from app.config import get_settings


class Embedder(Protocol):
    """Minimum interface every embedding backend implements."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]: ...

    def embed_query(self, text: str) -> List[float]: ...


class TextEmbedder:
    """OpenAI-compatible embeddings adapter.

    Points at any HTTP endpoint speaking the OpenAI embeddings protocol.
    We keep ``langchain_openai.OpenAIEmbeddings`` under the hood for
    request batching, retries, and a clean API surface.
    """

    def __init__(self, model: str, base_url: str, api_key: str = "unused") -> None:
        if not base_url:
            raise ValueError("EMBEDDING_BASE_URL is required for TextEmbedder")
        if not model:
            raise ValueError("EMBEDDING_MODEL is required for TextEmbedder")
        # Lazy import so tests can monkeypatch without pulling openai at module load.
        from langchain_openai import OpenAIEmbeddings

        # ``api_key`` must be a non-empty string for the SDK; local runners
        # ignore its value. Nothing is ever sent to OpenAI.
        self._impl = OpenAIEmbeddings(
            model=model,
            api_key=api_key or "unused",
            base_url=base_url,
            # Ollama and other local runners don't support the OpenAI-only
            # "embedding preflight" that langchain-openai does by default.
            check_embedding_ctx_length=False,
        )
        self.model = model
        self.base_url = base_url

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return self._impl.embed_documents(list(texts))

    def embed_query(self, text: str) -> List[float]:
        return self._impl.embed_query(text)


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    """Return a cached embedder instance built from application settings."""
    s = get_settings()
    return TextEmbedder(
        model=s.embedding_model,
        base_url=s.embedding_base_url,
        api_key=s.embedding_api_key,
    )


def reset_embedder_cache() -> None:
    """Reset the cached embedder (used in tests)."""
    get_embedder.cache_clear()
