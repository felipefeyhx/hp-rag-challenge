"""Database engine, session factory and dependency helpers."""
from __future__ import annotations

import os
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings
from app.db.models import Base


def _make_engine(url: str) -> Engine:
    connect_args = {}
    if url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
        # Ensure parent directory exists for file-based SQLite
        # Format: sqlite:///relative/path or sqlite:////abs/path
        prefix = "sqlite:///"
        if url.startswith(prefix):
            path = url[len(prefix):]
            if path and not path.startswith(":memory:"):
                parent = os.path.dirname(path)
                if parent:
                    os.makedirs(parent, exist_ok=True)
    return create_engine(url, connect_args=connect_args, future=True)


_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = _make_engine(get_settings().database_url)
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), expire_on_commit=False, future=True)
    return _SessionLocal


def init_db() -> None:
    """Create all tables. Safe to call multiple times."""
    Base.metadata.create_all(bind=get_engine())


def reset_engine() -> None:
    """Reset cached engine/session factory (used in tests)."""
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionLocal = None


def get_db() -> Iterator[Session]:
    """FastAPI dependency yielding a database session."""
    factory = get_session_factory()
    db = factory()
    try:
        yield db
    finally:
        db.close()
