"""Tests for app.db.models and app.db.database."""
from __future__ import annotations

import os

import pytest
from sqlalchemy import inspect, select

from app.db import database as db_module
from app.db.models import Message, Session as ChatSession


# --- Models --------------------------------------------------------------- #

def test_session_default_title(db_session):
    s = ChatSession(id="s1")
    db_session.add(s)
    db_session.commit()
    db_session.refresh(s)
    assert s.title == "New chat"
    assert s.created_at is not None
    assert s.updated_at is not None


def test_message_persist_and_cascade_delete(db_session):
    s = ChatSession(id="s2", title="Test")
    m1 = Message(session_id="s2", role="user", content="hi")
    m2 = Message(session_id="s2", role="assistant", content="hello", sources="[]")
    s.messages = [m1, m2]
    db_session.add(s)
    db_session.commit()

    stored = db_session.execute(select(Message).where(Message.session_id == "s2")).scalars().all()
    assert len(stored) == 2

    db_session.delete(s)
    db_session.commit()
    remaining = db_session.execute(select(Message).where(Message.session_id == "s2")).scalars().all()
    assert remaining == []


def test_messages_relationship_ordered_by_created(db_session):
    s = ChatSession(id="s3")
    db_session.add(s)
    db_session.commit()
    db_session.add(Message(session_id="s3", role="user", content="one"))
    db_session.add(Message(session_id="s3", role="assistant", content="two"))
    db_session.commit()
    db_session.refresh(s)
    assert [m.role for m in s.messages] == ["user", "assistant"]


# --- database helpers ----------------------------------------------------- #

def test_make_engine_creates_parent_dir_for_sqlite(tmp_path):
    target = tmp_path / "nested" / "sub" / "chat.db"
    url = f"sqlite:///{target}"
    engine = db_module._make_engine(url)
    assert engine is not None
    assert target.parent.exists()


def test_make_engine_ignores_memory_url():
    engine = db_module._make_engine("sqlite:///:memory:")
    assert engine is not None


def test_get_engine_caches(monkeypatch):
    db_module.reset_engine()
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    from app.config import reset_settings_cache
    reset_settings_cache()
    e1 = db_module.get_engine()
    e2 = db_module.get_engine()
    assert e1 is e2
    db_module.reset_engine()


def test_get_session_factory_returns_a_factory():
    db_module.reset_engine()
    factory = db_module.get_session_factory()
    session = factory()
    session.close()
    db_module.reset_engine()


def test_init_db_creates_tables():
    db_module.reset_engine()
    from app.config import reset_settings_cache
    reset_settings_cache()
    db_module.init_db()
    inspector = inspect(db_module.get_engine())
    names = set(inspector.get_table_names())
    assert {"sessions", "messages"}.issubset(names)
    db_module.reset_engine()


def test_get_db_yields_and_closes_session():
    db_module.reset_engine()
    from app.config import reset_settings_cache
    reset_settings_cache()
    db_module.init_db()
    gen = db_module.get_db()
    session = next(gen)
    assert session is not None
    with pytest.raises(StopIteration):
        next(gen)
    db_module.reset_engine()
