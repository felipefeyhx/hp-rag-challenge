"""Tests for app.core.embeddings."""
from __future__ import annotations

from typing import List

import pytest

from app.core import embeddings as emb_mod
from app.core.embeddings import TextEmbedder, get_embedder, reset_embedder_cache


class _FakeOpenAIEmbeddings:
    """Stand-in matching the langchain_openai.OpenAIEmbeddings surface we use."""

    def __init__(self, model: str, api_key: str, base_url: str = "", **kwargs) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.embed_documents_calls: List[List[str]] = []
        self.embed_query_calls: List[str] = []

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        self.embed_documents_calls.append(list(texts))
        return [[float(len(t))] * 3 for t in texts]

    def embed_query(self, text: str) -> List[float]:
        self.embed_query_calls.append(text)
        return [float(len(text))] * 3


class _FakeLangchainOpenAI:
    OpenAIEmbeddings = _FakeOpenAIEmbeddings


@pytest.fixture()
def patched_lc(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "langchain_openai", _FakeLangchainOpenAI())


# --- TextEmbedder --------------------------------------------------------- #

def test_text_embedder_requires_base_url():
    with pytest.raises(ValueError):
        TextEmbedder(model="test", base_url="")


def test_text_embedder_requires_model():
    with pytest.raises(ValueError):
        TextEmbedder(model="", base_url="http://localhost:11434/v1")


def test_text_embedder_embed_documents(patched_lc):
    e = TextEmbedder(model="nomic-embed-text", base_url="http://localhost:11434/v1")
    result = e.embed_documents(["ab", "abc"])
    assert result == [[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]


def test_text_embedder_embed_query(patched_lc):
    e = TextEmbedder(model="nomic-embed-text", base_url="http://localhost:11434/v1")
    result = e.embed_query("hello")
    assert result == [5.0, 5.0, 5.0]


def test_text_embedder_empty_list_returns_empty(patched_lc):
    e = TextEmbedder(model="nomic-embed-text", base_url="http://localhost:11434/v1")
    assert e.embed_documents([]) == []


# --- get_embedder cache --------------------------------------------------- #

def test_get_embedder_caches_and_resets(patched_lc, monkeypatch):
    monkeypatch.setenv("EMBEDDING_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("EMBEDDING_API_KEY", "unused")
    from app.config import reset_settings_cache
    reset_settings_cache()
    reset_embedder_cache()

    e1 = get_embedder()
    e2 = get_embedder()
    assert e1 is e2

    reset_embedder_cache()
    e3 = get_embedder()
    assert e3 is not e1
