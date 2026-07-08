"""Tests for app.core.vectorstore.

Uses Chroma's EphemeralClient (in-memory) via monkeypatch, and a fake embedder.
"""
from __future__ import annotations

import chromadb
import numpy as np
import pytest

from app.core import vectorstore as vs_mod
from app.core.chunking import DocumentChunk
from app.core.vectorstore import RetrievedChunk, VectorStore, _mmr, get_vector_store


@pytest.fixture(autouse=True)
def _use_ephemeral_chroma(monkeypatch):
    """Make VectorStore use Chroma's in-memory client."""
    monkeypatch.setattr(chromadb, "PersistentClient", lambda path: chromadb.EphemeralClient())


def _chunks_for_tests():
    return [
        DocumentChunk(text="Paper tray holds 80 sheets of plain paper.",
                      source="a.pdf", page=118, chunk_index=0),
        DocumentChunk(text="Connect the Ethernet cable to the middle port on the rear.",
                      source="a.pdf", page=35, chunk_index=0),
        DocumentChunk(text="Environmental specifications: 15-32C operating temperature.",
                      source="b.pdf", page=5, chunk_index=0),
        DocumentChunk(text="Toner cartridge replacement steps require lifting the top cover.",
                      source="a.pdf", page=60, chunk_index=0),
    ]


# --- MMR ------------------------------------------------------------------- #

def test_mmr_returns_top_k_indices():
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    candidates = np.array([
        [1.0, 0.0, 0.0],   # most similar
        [0.9, 0.1, 0.0],   # similar
        [0.0, 0.0, 1.0],   # dissimilar → good for diversity
    ], dtype=np.float32)
    selected = _mmr(query, candidates, top_k=2, mmr_lambda=0.5)
    assert selected[0] == 0                   # first pick is highest similarity
    assert len(selected) == 2
    assert 0 <= selected[1] <= 2 and selected[1] != 0


def test_mmr_empty_returns_empty():
    q = np.array([1.0, 0.0], dtype=np.float32)
    assert _mmr(q, np.zeros((0, 2), dtype=np.float32), top_k=3, mmr_lambda=0.5) == []
    assert _mmr(q, np.ones((3, 2), dtype=np.float32), top_k=0, mmr_lambda=0.5) == []


def test_mmr_top_k_capped_by_candidates():
    q = np.array([1.0, 0.0], dtype=np.float32)
    C = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    selected = _mmr(q, C, top_k=5, mmr_lambda=0.5)
    assert len(selected) == 2


def test_mmr_handles_zero_vectors():
    q = np.array([0.0, 0.0], dtype=np.float32)
    C = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    # Should not crash on the zero-norm normalization branch.
    selected = _mmr(q, C, top_k=2, mmr_lambda=0.5)
    assert len(selected) == 2


# --- VectorStore ---------------------------------------------------------- #

def test_vectorstore_upsert_and_count(fake_embedder, tmp_path):
    store = VectorStore(str(tmp_path), "coll", fake_embedder)
    n = store.upsert(_chunks_for_tests())
    assert n == 4
    assert store.count() == 4


def test_vectorstore_upsert_empty_returns_zero(fake_embedder, tmp_path):
    store = VectorStore(str(tmp_path), "coll2", fake_embedder)
    assert store.upsert([]) == 0
    assert store.count() == 0


def test_vectorstore_upsert_is_idempotent(fake_embedder, tmp_path):
    store = VectorStore(str(tmp_path), "coll3", fake_embedder)
    chunks = _chunks_for_tests()
    store.upsert(chunks)
    store.upsert(chunks)     # same deterministic ids → upsert, not duplicate
    assert store.count() == 4


def test_vectorstore_retrieve_similarity(fake_embedder, tmp_path):
    store = VectorStore(str(tmp_path), "coll4", fake_embedder)
    store.upsert(_chunks_for_tests())
    results = store.retrieve("paper tray", top_k=2, strategy="similarity")
    assert len(results) == 2
    assert all(isinstance(r, RetrievedChunk) for r in results)


def test_vectorstore_retrieve_mmr(fake_embedder, tmp_path):
    store = VectorStore(str(tmp_path), "coll5", fake_embedder)
    store.upsert(_chunks_for_tests())
    results = store.retrieve("paper tray", top_k=3, fetch_k=4, strategy="mmr", mmr_lambda=0.5)
    assert len(results) == 3


def test_vectorstore_retrieve_empty_query_returns_empty(fake_embedder, tmp_path):
    store = VectorStore(str(tmp_path), "coll6", fake_embedder)
    store.upsert(_chunks_for_tests())
    assert store.retrieve("", top_k=3) == []
    assert store.retrieve("   ", top_k=3) == []


def test_vectorstore_retrieve_no_data_returns_empty(fake_embedder, tmp_path):
    store = VectorStore(str(tmp_path), "coll7", fake_embedder)
    assert store.retrieve("anything", top_k=3) == []


def test_vectorstore_retrieve_rejects_bad_strategy(fake_embedder, tmp_path):
    store = VectorStore(str(tmp_path), "coll8", fake_embedder)
    store.upsert(_chunks_for_tests()[:1])
    with pytest.raises(ValueError):
        store.retrieve("x", strategy="banana")


# --- factory --------------------------------------------------------------- #

def test_retrieved_chunk_to_dict():
    r = RetrievedChunk(text="x", source="a.pdf", page=1, chunk_index=0, score=0.9)
    d = r.to_dict()
    assert d["text"] == "x"
    assert d["source"] == "a.pdf"
    assert d["page"] == 1
    assert d["score"] == 0.9


def test_get_vector_store_caches(fake_embedder):
    vs_mod.reset_vector_store_cache()
    a = get_vector_store(embedder=fake_embedder)
    b = get_vector_store(embedder=fake_embedder)
    assert a is b
    vs_mod.reset_vector_store_cache()
    c = get_vector_store(embedder=fake_embedder)
    assert c is not a
    vs_mod.reset_vector_store_cache()
