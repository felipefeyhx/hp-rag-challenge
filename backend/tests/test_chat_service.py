"""Tests for app.services.chat_service."""
from __future__ import annotations

import json

import pytest

from app.core.vectorstore import RetrievedChunk
from app.db.models import Message, Session as ChatSession
from app.services import chat_service


def _rc(text="ctx", source="a.pdf", page=1, score=0.9) -> RetrievedChunk:
    return RetrievedChunk(text=text, source=source, page=page, chunk_index=0, score=score)


# --- session CRUD -------------------------------------------------------- #

def test_create_session_defaults(db_session):
    s = chat_service.create_session(db_session)
    assert s.id
    assert s.title == "New chat"


def test_get_session_returns_none_for_missing(db_session):
    assert chat_service.get_session(db_session, "does-not-exist") is None


def test_get_or_create_session_creates_when_id_missing(db_session):
    s = chat_service.get_or_create_session(db_session, None)
    assert s.id


def test_get_or_create_session_creates_when_id_unknown(db_session):
    s = chat_service.get_or_create_session(db_session, "unknown")
    assert s.id != "unknown"


def test_get_or_create_session_returns_existing(db_session):
    s1 = chat_service.create_session(db_session)
    s2 = chat_service.get_or_create_session(db_session, s1.id)
    assert s2.id == s1.id


def test_list_sessions_orders_by_recent(db_session):
    a = chat_service.create_session(db_session)
    b = chat_service.create_session(db_session)
    listed = chat_service.list_sessions(db_session)
    assert {s.id for s in listed} == {a.id, b.id}


def test_delete_session_returns_false_when_missing(db_session):
    assert chat_service.delete_session(db_session, "missing") is False


def test_delete_session_removes(db_session):
    s = chat_service.create_session(db_session)
    assert chat_service.delete_session(db_session, s.id) is True
    assert chat_service.get_session(db_session, s.id) is None


# --- messages ------------------------------------------------------------ #

def test_append_message_persists(db_session):
    s = chat_service.create_session(db_session)
    m = chat_service.append_message(db_session, s, role="user", content="hi")
    assert m.id is not None
    assert m.role == "user"
    assert m.sources == ""


def test_append_message_with_sources(db_session):
    s = chat_service.create_session(db_session)
    m = chat_service.append_message(
        db_session, s, role="assistant", content="a",
        sources=[_rc("chunk", "a.pdf", 5, 0.7)],
    )
    parsed = json.loads(m.sources)
    assert parsed[0]["source"] == "a.pdf"
    assert parsed[0]["page"] == 5


def test_load_history_returns_chronological(db_session):
    s = chat_service.create_session(db_session)
    chat_service.append_message(db_session, s, role="user", content="q1")
    chat_service.append_message(db_session, s, role="assistant", content="a1")
    chat_service.append_message(db_session, s, role="user", content="q2")
    history = chat_service.load_history(db_session, s.id, window=3)
    assert [m["role"] for m in history] == ["user", "assistant", "user"]
    assert history[0]["content"] == "q1"
    assert history[-1]["content"] == "q2"


def test_load_history_zero_window_returns_empty(db_session):
    s = chat_service.create_session(db_session)
    chat_service.append_message(db_session, s, role="user", content="x")
    assert chat_service.load_history(db_session, s.id, window=0) == []


def test_parse_sources_variants():
    assert chat_service.parse_sources("") == []
    assert chat_service.parse_sources("not-json") == []
    assert chat_service.parse_sources("[]") == []
    assert chat_service.parse_sources('{"not":"a list"}') == []
    parsed = chat_service.parse_sources('[{"a": 1}]')
    assert parsed == [{"a": 1}]


# --- handle_chat_turn --------------------------------------------------- #

def test_handle_chat_turn_first_turn(db_session, fake_vector_store):
    from tests.conftest import FakeLLM
    llm = FakeLLM(responses=["This is the assistant answer."])

    session, result = chat_service.handle_chat_turn(
        db_session,
        session_id=None,
        user_message="What is the paper tray capacity?",
        vector_store=fake_vector_store,
        llm=llm,
    )

    assert session.id
    # Session was auto-titled from the first user message
    assert session.title == "What is the paper tray capacity?"
    assert result.answer == "This is the assistant answer."
    # Two messages persisted: user + assistant
    msgs = db_session.query(Message).filter_by(session_id=session.id).order_by(Message.id).all()
    assert [m.role for m in msgs] == ["user", "assistant"]
    assert msgs[1].sources != ""


def test_handle_chat_turn_reuses_existing_session(db_session, fake_vector_store):
    from tests.conftest import FakeLLM
    llm = FakeLLM(responses=["ok"])
    existing = chat_service.create_session(db_session, title="Existing chat")

    session, _ = chat_service.handle_chat_turn(
        db_session,
        session_id=existing.id,
        user_message="another message",
        vector_store=fake_vector_store,
        llm=llm,
    )
    assert session.id == existing.id
    # Existing title preserved (not overwritten)
    assert session.title == "Existing chat"


def test_make_title_short_returns_as_is():
    assert chat_service._make_title("hello world") == "hello world"


def test_make_title_collapses_whitespace():
    assert chat_service._make_title("hello    world\n\nfoo") == "hello world foo"


def test_make_title_truncates_long_message():
    long = "x" * 200
    result = chat_service._make_title(long, max_len=30)
    assert len(result) == 30
    assert result.endswith("…")


# --- handle_chat_turn_stream --------------------------------------------- #

def test_handle_chat_turn_stream_first_turn_persists_and_titles(db_session, fake_vector_store):
    from tests.conftest import FakeLLM
    llm = FakeLLM(responses=["Streamed assistant answer."])

    events = list(chat_service.handle_chat_turn_stream(
        db_session,
        session_id=None,
        user_message="What is the paper tray capacity?",
        vector_store=fake_vector_store,
        llm=llm,
    ))

    kinds = [k for k, _ in events]
    assert kinds[0] == "session"
    assert kinds[-1] == "done"
    assert kinds.count("chunk") >= 1

    session_payload = dict(events[0][1])
    done_payload = dict(events[-1][1])
    assert done_payload["session_id"] == session_payload["session_id"]
    assert isinstance(done_payload["sources"], list) and done_payload["sources"]

    # Full answer text arrived via chunks.
    streamed = "".join(payload for kind, payload in events if kind == "chunk")
    assert streamed == "Streamed assistant answer."

    # DB has user + assistant messages, session was auto-titled.
    session = chat_service.get_session(db_session, session_payload["session_id"])
    assert session.title == "What is the paper tray capacity?"
    msgs = db_session.query(Message).filter_by(session_id=session.id).order_by(Message.id).all()
    assert [m.role for m in msgs] == ["user", "assistant"]
    assert msgs[1].content == "Streamed assistant answer."
    assert msgs[1].sources != ""


def test_handle_chat_turn_stream_reuses_existing_session(db_session, fake_vector_store):
    from tests.conftest import FakeLLM
    llm = FakeLLM(responses=["ok"])
    existing = chat_service.create_session(db_session, title="Existing chat")

    events = list(chat_service.handle_chat_turn_stream(
        db_session,
        session_id=existing.id,
        user_message="hello again",
        vector_store=fake_vector_store,
        llm=llm,
    ))
    session_payload = dict(events[0][1])
    assert session_payload["session_id"] == existing.id

    session = chat_service.get_session(db_session, existing.id)
    assert session.title == "Existing chat"    # preserved
