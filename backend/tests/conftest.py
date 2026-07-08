"""Shared pytest fixtures + fake implementations.

Every test in this suite depends on some subset of the fakes defined here.
Nothing hits the network or a real database.
"""
from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Iterator, List, Optional

import pytest

# ---- Set safe env vars BEFORE app.config is imported anywhere ----
# These point at a local/loopback URL that we never actually hit; every
# call is intercepted by fakes below.
os.environ.setdefault("LLM_BASE_URL", "http://localhost:11434/v1")
os.environ.setdefault("LLM_API_KEY", "test-not-real")
os.environ.setdefault("EMBEDDING_BASE_URL", "http://localhost:11434/v1")
os.environ.setdefault("EMBEDDING_API_KEY", "test-not-real")
# LLM_MODEL and EMBEDDING_MODEL are hardcoded in app.config — not read from env.
os.environ.setdefault("CHROMA_PERSIST_DIR", "/tmp/chroma-tests")
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session as DBSession, sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import get_settings, reset_settings_cache
from app.core.chunking import DocumentChunk
from app.core.embeddings import reset_embedder_cache
from app.core.llm import ChatMessage, reset_llm_cache
from app.core.vectorstore import RetrievedChunk, reset_vector_store_cache
from app.db import database as db_module
from app.db.models import Base


# --------------------------------------------------------------------------- #
# Cache resets                                                                #
# --------------------------------------------------------------------------- #

@pytest.fixture(autouse=True)
def _reset_caches() -> Iterator[None]:
    """Clear cached singletons between tests so env-var / monkeypatch changes stick."""
    reset_settings_cache()
    reset_embedder_cache()
    reset_llm_cache()
    reset_vector_store_cache()
    yield
    reset_settings_cache()
    reset_embedder_cache()
    reset_llm_cache()
    reset_vector_store_cache()


# --------------------------------------------------------------------------- #
# In-memory SQLite session                                                    #
# --------------------------------------------------------------------------- #

@pytest.fixture()
def db_engine():
    """A fresh in-memory SQLite engine with all tables created."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,       # share one connection so all sessions see the same in-memory DB
        future=True,
    )
    Base.metadata.create_all(bind=engine)
    yield engine
    engine.dispose()


@pytest.fixture()
def db_session(db_engine) -> Iterator[DBSession]:
    """A SQLAlchemy session bound to the in-memory engine."""
    factory = sessionmaker(bind=db_engine, expire_on_commit=False, future=True)
    session = factory()
    try:
        yield session
    finally:
        session.close()


# --------------------------------------------------------------------------- #
# Fake LLM                                                                    #
# --------------------------------------------------------------------------- #

class FakeLLM:
    """Deterministic LLM stand-in for tests."""

    def __init__(self, responses: Optional[List[str]] = None, default: str = "fake answer") -> None:
        self.responses = list(responses or [])
        self.default = default
        self.calls: List[List[ChatMessage]] = []
        self.stream_calls: List[List[ChatMessage]] = []

    def chat(self, messages, *, temperature: float = 0.0, max_tokens=None) -> str:
        self.calls.append(list(messages))
        if self.responses:
            return self.responses.pop(0)
        return self.default

    def chat_stream(self, messages, *, temperature: float = 0.0, max_tokens=None):
        """Yield the next scripted response character-by-character.

        Character-level chunking is intentional: it exercises the frontend's
        ``<think>``/``</think>`` splitter (and any server-side buffering)
        against the pathological case where tags are torn apart mid-tag.
        """
        self.stream_calls.append(list(messages))
        text = self.responses.pop(0) if self.responses else self.default
        for ch in text:
            yield ch


@pytest.fixture()
def fake_llm() -> FakeLLM:
    return FakeLLM()


# --------------------------------------------------------------------------- #
# Fake embedder                                                               #
# --------------------------------------------------------------------------- #

class FakeEmbedder:
    """Deterministic embedder: hashes text into a small fixed-dim vector."""

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim
        self.calls_documents: List[List[str]] = []
        self.calls_queries: List[str] = []

    def _vec(self, text: str) -> List[float]:
        # Deterministic pseudo-embedding based on character values.
        # Not semantic — just consistent for tests that compare identity.
        base = [0.0] * self.dim
        for i, ch in enumerate(text):
            base[i % self.dim] += (ord(ch) % 17) / 17.0
        # Normalize
        norm = sum(x * x for x in base) ** 0.5 or 1.0
        return [x / norm for x in base]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        self.calls_documents.append(list(texts))
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        self.calls_queries.append(text)
        return self._vec(text)


@pytest.fixture()
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


# --------------------------------------------------------------------------- #
# Fake vector store                                                           #
# --------------------------------------------------------------------------- #

class FakeVectorStore:
    """In-memory stand-in that satisfies the same surface as VectorStore."""

    def __init__(self, chunks: Optional[List[DocumentChunk]] = None) -> None:
        self._chunks = list(chunks or [])
        self.retrieve_calls: List[dict] = []

    def upsert(self, chunks) -> int:
        for c in chunks:
            self._chunks.append(c)
        return len(chunks)

    def count(self) -> int:
        return len(self._chunks)

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        fetch_k: int = 20,
        strategy: str = "mmr",
        mmr_lambda: float = 0.5,
    ) -> List[RetrievedChunk]:
        self.retrieve_calls.append({
            "query": query,
            "top_k": top_k,
            "fetch_k": fetch_k,
            "strategy": strategy,
            "mmr_lambda": mmr_lambda,
        })
        # Return the first `top_k` chunks as-is with a fake similarity score.
        out: List[RetrievedChunk] = []
        for i, c in enumerate(self._chunks[:top_k]):
            out.append(RetrievedChunk(
                text=c.text,
                source=c.source,
                page=c.page,
                chunk_index=c.chunk_index,
                score=1.0 - (0.1 * i),
            ))
        return out


@pytest.fixture()
def fake_vector_store() -> FakeVectorStore:
    return FakeVectorStore(chunks=[
        DocumentChunk(text="Alpha chunk about paper tray.", source="a.pdf", page=1, chunk_index=0),
        DocumentChunk(text="Beta chunk about network setup.", source="a.pdf", page=2, chunk_index=1),
        DocumentChunk(text="Gamma chunk about environmental specs.", source="b.pdf", page=3, chunk_index=0),
    ])


# --------------------------------------------------------------------------- #
# Fake OpenAI-compatible client (used by ChatLLM and TextEmbedder tests).     #
# The wire format is OpenAI's; the actual backend in prod is Ollama.          #
# --------------------------------------------------------------------------- #

class FakeOpenAIChatCompletions:
    def __init__(self, holder):
        self._holder = holder

    def create(self, **kwargs):
        self._holder.last_call = kwargs
        content = self._holder.responses.pop(0) if self._holder.responses else "fake reply"
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


class FakeOpenAIClient:
    """Mimics the parts of openai.OpenAI used by ChatLLM."""

    def __init__(
        self,
        api_key: str = "test",
        base_url: str = "http://localhost:11434/v1",
        responses: Optional[List[str]] = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.responses = list(responses or [])
        self.last_call: dict = {}
        self.chat = SimpleNamespace(completions=FakeOpenAIChatCompletions(self))


@pytest.fixture()
def fake_openai_factory():
    """Return a factory that produces FakeOpenAIClient instances.

    Usage in a test:
        def test_x(monkeypatch, fake_openai_factory):
            monkeypatch.setattr("openai.OpenAI", fake_openai_factory(["hi"]))
    """
    def _factory(responses=None):
        # openai.OpenAI is called with kwargs (api_key=..., base_url=...)
        def _ctor(**kwargs):
            return FakeOpenAIClient(
                api_key=kwargs.get("api_key", "test"),
                base_url=kwargs.get("base_url", "http://localhost:11434/v1"),
                responses=responses,
            )
        return _ctor
    return _factory


# --------------------------------------------------------------------------- #
# FastAPI TestClient with dependency overrides                                #
# --------------------------------------------------------------------------- #

@pytest.fixture()
def client(db_engine) -> Iterator[TestClient]:
    """FastAPI TestClient wired against the in-memory DB."""
    from app.main import create_app
    from app.db.database import get_db

    # Point the app's global engine at the test one (so init_db is a no-op).
    db_module._engine = db_engine
    db_module._SessionLocal = sessionmaker(bind=db_engine, expire_on_commit=False, future=True)

    app = create_app()

    def _override_get_db():
        session = db_module._SessionLocal()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = _override_get_db

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()
    db_module._engine = None
    db_module._SessionLocal = None
