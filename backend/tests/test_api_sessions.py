"""Tests for /sessions endpoints."""
from __future__ import annotations


def test_create_session(client):
    r = client.post("/sessions")
    assert r.status_code == 201
    body = r.json()
    assert body["id"]
    assert body["title"] == "New chat"


def test_list_sessions_empty(client):
    r = client.get("/sessions")
    assert r.status_code == 200
    assert r.json() == []


def test_list_sessions_returns_created(client):
    client.post("/sessions")
    client.post("/sessions")
    r = client.get("/sessions")
    assert r.status_code == 200
    assert len(r.json()) == 2


def test_get_session_404(client):
    r = client.get("/sessions/does-not-exist")
    assert r.status_code == 404


def test_get_session_returns_messages(client, monkeypatch, fake_vector_store):
    from tests.conftest import FakeLLM

    class _FakeLLM(FakeLLM):
        pass

    fake_llm = _FakeLLM(responses=["hello there"])
    monkeypatch.setattr("app.api.chat.get_vector_store", lambda: fake_vector_store)
    monkeypatch.setattr("app.api.chat.get_llm", lambda: fake_llm)

    created = client.post("/chat", json={"message": "hi"})
    session_id = created.json()["session_id"]

    r = client.get(f"/sessions/{session_id}")
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == session_id
    assert len(body["messages"]) == 2
    assert body["messages"][0]["role"] == "user"
    assert body["messages"][1]["role"] == "assistant"


def test_delete_session_204(client):
    r = client.post("/sessions")
    sid = r.json()["id"]
    d = client.delete(f"/sessions/{sid}")
    assert d.status_code == 204
    assert client.get(f"/sessions/{sid}").status_code == 404


def test_delete_session_404_when_missing(client):
    r = client.delete("/sessions/missing-id")
    assert r.status_code == 404


def test_get_session_ignores_malformed_sources(client, monkeypatch, fake_vector_store, db_engine):
    """If the DB has a message with malformed sources JSON, /sessions still returns 200."""
    from sqlalchemy.orm import sessionmaker
    from app.db.models import Message, Session as ChatSession

    factory = sessionmaker(bind=db_engine, expire_on_commit=False, future=True)
    with factory() as db:
        s = ChatSession(id="sess-bad-json", title="x")
        db.add(s)
        db.commit()
        db.add(Message(session_id="sess-bad-json", role="assistant", content="hi", sources="not-json"))
        db.commit()

    r = client.get("/sessions/sess-bad-json")
    assert r.status_code == 200
    assert r.json()["messages"][0]["sources"] == []
