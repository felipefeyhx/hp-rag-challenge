"""Tests for app.config."""
from __future__ import annotations

import pytest

from app.config import Settings, get_settings, reset_settings_cache


def test_get_settings_returns_cached_instance():
    reset_settings_cache()
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2


def test_settings_have_sensible_defaults(monkeypatch):
    # Wipe env so we hit the field defaults.
    for k in (
        "LLM_BASE_URL", "LLM_API_KEY",
        "EMBEDDING_BASE_URL", "EMBEDDING_API_KEY",
        "CHUNK_SIZE", "CHUNK_OVERLAP", "RETRIEVAL_TOP_K",
        "RETRIEVAL_FETCH_K", "RETRIEVAL_STRATEGY", "HISTORY_WINDOW",
        "DATABASE_URL",
    ):
        monkeypatch.delenv(k, raising=False)
    reset_settings_cache()
    s = Settings(_env_file=None)  # ignore .env in the workdir
    assert s.llm_base_url == "http://localhost:11434/v1"
    assert s.llm_model == "qwen2.5:7b-instruct"          # hardcoded ClassVar
    assert s.embedding_base_url == "http://localhost:11434/v1"
    assert s.embedding_model == "nomic-embed-text"        # hardcoded ClassVar
    assert s.chunk_size == 800
    assert s.chunk_overlap == 120
    assert s.retrieval_top_k == 5
    assert s.retrieval_strategy == "mmr"
    assert s.history_window == 6


def test_llm_model_is_not_overridable_via_env(monkeypatch):
    """LLM_MODEL / EMBEDDING_MODEL are hardcoded — env vars must be ignored."""
    monkeypatch.setenv("LLM_MODEL", "some-other-model:latest")
    monkeypatch.setenv("EMBEDDING_MODEL", "some-other-embedder")
    reset_settings_cache()
    s = Settings(_env_file=None)
    assert s.llm_model == "qwen2.5:7b-instruct"
    assert s.embedding_model == "nomic-embed-text"


def test_settings_reads_env(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://ollama:11434/v1")
    monkeypatch.setenv("CHUNK_SIZE", "1200")
    monkeypatch.setenv("CHUNK_OVERLAP", "150")
    monkeypatch.setenv("RETRIEVAL_STRATEGY", "similarity")
    s = Settings(_env_file=None)
    assert s.llm_base_url == "http://ollama:11434/v1"
    assert s.chunk_size == 1200
    assert s.chunk_overlap == 150
    assert s.retrieval_strategy == "similarity"


def test_settings_normalizes_strategy_case(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_STRATEGY", "MMR")
    s = Settings(_env_file=None)
    assert s.retrieval_strategy == "mmr"


def test_settings_rejects_invalid_strategy(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_STRATEGY", "banana")
    with pytest.raises(Exception):
        Settings(_env_file=None)


def test_settings_rejects_overlap_ge_chunk_size(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "500")
    monkeypatch.setenv("CHUNK_OVERLAP", "500")
    with pytest.raises(Exception):
        Settings(_env_file=None)


def test_cors_origin_list_wildcard():
    s = Settings(_env_file=None, cors_origins="*")
    assert s.cors_origin_list() == ["*"]


def test_cors_origin_list_empty():
    s = Settings(_env_file=None, cors_origins="")
    assert s.cors_origin_list() == ["*"]


def test_cors_origin_list_specific():
    s = Settings(_env_file=None, cors_origins="http://a.com, http://b.com ,")
    assert s.cors_origin_list() == ["http://a.com", "http://b.com"]
