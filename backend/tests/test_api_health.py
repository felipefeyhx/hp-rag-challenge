"""Tests for GET /health."""
from __future__ import annotations


def test_health_shape(client, monkeypatch, fake_vector_store):
    monkeypatch.setattr("app.api.health.get_vector_store", lambda: fake_vector_store)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["version"]
    assert body["llm_model"]
    assert isinstance(body["vector_store_docs"], int)


def test_health_reports_vector_store_count(client, monkeypatch, fake_vector_store):
    monkeypatch.setattr("app.api.health.get_vector_store", lambda: fake_vector_store)
    r = client.get("/health")
    assert r.json()["vector_store_docs"] == fake_vector_store.count()


def test_health_zero_when_vector_store_unavailable(client, monkeypatch):
    def _boom():
        raise RuntimeError("no store")
    monkeypatch.setattr("app.api.health.get_vector_store", _boom)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["vector_store_docs"] == 0
