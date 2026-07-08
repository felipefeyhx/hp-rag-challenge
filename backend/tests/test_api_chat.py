"""Tests for POST /chat."""
from __future__ import annotations

import pytest

from tests.conftest import FakeLLM


@pytest.fixture()
def wired_llm(monkeypatch, fake_vector_store):
    """Inject a fake LLM + fake vector store into the chat router."""
    llm = FakeLLM(responses=["Assistant response."])
    monkeypatch.setattr("app.api.chat.get_vector_store", lambda: fake_vector_store)
    monkeypatch.setattr("app.api.chat.get_llm", lambda: llm)
    return llm


def test_chat_creates_session_and_answers(client, wired_llm):
    r = client.post("/chat", json={"message": "What is the paper tray capacity?"})
    assert r.status_code == 200
    body = r.json()
    assert body["session_id"]
    assert body["answer"] == "Assistant response."
    assert len(body["sources"]) >= 1
    assert body["sources"][0]["source"]
    assert isinstance(body["sources"][0]["page"], int)


def test_chat_continues_existing_session(client, wired_llm):
    first = client.post("/chat", json={"message": "hi"})
    sid = first.json()["session_id"]

    wired_llm.responses.append("second reply")
    second = client.post("/chat", json={"message": "again", "session_id": sid})
    assert second.status_code == 200
    assert second.json()["session_id"] == sid


def test_chat_rejects_empty_message(client):
    # Pydantic validates min_length=1; FastAPI returns 422 before our handler runs.
    r = client.post("/chat", json={"message": ""})
    assert r.status_code == 422


def test_chat_rejects_missing_message(client):
    r = client.post("/chat", json={})
    assert r.status_code == 422


def test_chat_returns_500_on_backend_failure(client, monkeypatch, fake_vector_store):
    """If the RAG service crashes, /chat should return a 500 with a message."""
    class _BoomLLM:
        def chat(self, *a, **kw): raise RuntimeError("simulated failure")

    monkeypatch.setattr("app.api.chat.get_vector_store", lambda: fake_vector_store)
    monkeypatch.setattr("app.api.chat.get_llm", lambda: _BoomLLM())
    r = client.post("/chat", json={"message": "trigger failure"})
    assert r.status_code == 500
    assert "chat failed" in r.json()["detail"].lower()


def test_chat_returns_400_on_invalid_call(client, monkeypatch, fake_vector_store):
    """Simulate the internal ValueError path (e.g. empty user message passed downstream)."""
    def _raise_value_error(*a, **kw):
        raise ValueError("bad input")
    monkeypatch.setattr("app.api.chat.chat_service.handle_chat_turn", _raise_value_error)
    monkeypatch.setattr("app.api.chat.get_vector_store", lambda: fake_vector_store)
    monkeypatch.setattr("app.api.chat.get_llm", lambda: FakeLLM())
    r = client.post("/chat", json={"message": "something"})
    assert r.status_code == 400
    assert "bad input" in r.json()["detail"]


# --- POST /chat/stream ---------------------------------------------------- #

def _parse_sse(text: str):
    """Parse an SSE response body into a list of (event, data) pairs."""
    import json as _json
    events = []
    for block in text.strip().split("\n\n"):
        event = "message"
        data_lines = []
        for line in block.splitlines():
            if line.startswith("event:"):
                event = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data_lines.append(line.split(":", 1)[1].strip())
        if data_lines:
            events.append((event, _json.loads("\n".join(data_lines))))
    return events


def test_chat_stream_emits_sse_events(client, wired_llm):
    r = client.post("/chat/stream", json={"message": "What is the paper tray capacity?"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse(r.text)
    kinds = [e for e, _ in events]
    assert kinds[0] == "session"
    assert kinds[-1] == "done"
    assert "chunk" in kinds

    reconstructed = "".join(data["text"] for evt, data in events if evt == "chunk")
    assert reconstructed == "Assistant response."

    done = [data for evt, data in events if evt == "done"][0]
    assert done["session_id"]
    assert done["sources"]


def test_chat_stream_emits_error_event_on_value_error(client, monkeypatch, fake_vector_store):
    def _raise(*a, **kw):
        raise ValueError("bad input")
        yield  # pragma: no cover - never reached

    monkeypatch.setattr("app.api.chat.chat_service.handle_chat_turn_stream", _raise)
    monkeypatch.setattr("app.api.chat.get_vector_store", lambda: fake_vector_store)
    monkeypatch.setattr("app.api.chat.get_llm", lambda: FakeLLM())

    r = client.post("/chat/stream", json={"message": "trigger"})
    assert r.status_code == 200
    events = _parse_sse(r.text)
    assert events[-1][0] == "error"
    assert "bad input" in events[-1][1]["message"]
