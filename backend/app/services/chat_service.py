"""Chat persistence + orchestration between DB and RAG.

This is the thin layer the API calls into. It:

* creates / fetches sessions,
* stores user + assistant messages,
* loads the last N turns for the RAG service,
* returns a hydrated response for the frontend.
"""
from __future__ import annotations

import json
import uuid
from typing import Any, Iterator, List, Optional, Sequence, Tuple

from sqlalchemy import select
from sqlalchemy.orm import Session as DBSession

from app.config import get_settings
from app.core.llm import ChatMessage, LLM
from app.core.vectorstore import RetrievedChunk, VectorStore
from app.db.models import Message, Session
from app.services.rag_service import RagAnswer, answer_question, answer_question_stream


# --------------------------------------------------------------------------- #
# Sessions                                                                    #
# --------------------------------------------------------------------------- #

def create_session(db: DBSession, *, title: str = "New chat") -> Session:
    session = Session(id=uuid.uuid4().hex, title=title)
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def get_session(db: DBSession, session_id: str) -> Optional[Session]:
    return db.get(Session, session_id)


def get_or_create_session(db: DBSession, session_id: Optional[str]) -> Session:
    if session_id:
        sess = get_session(db, session_id)
        if sess:
            return sess
    return create_session(db)


def list_sessions(db: DBSession, limit: int = 100) -> List[Session]:
    stmt = select(Session).order_by(Session.updated_at.desc()).limit(limit)
    return list(db.execute(stmt).scalars())


def delete_session(db: DBSession, session_id: str) -> bool:
    sess = db.get(Session, session_id)
    if sess is None:
        return False
    db.delete(sess)
    db.commit()
    return True


# --------------------------------------------------------------------------- #
# Messages                                                                    #
# --------------------------------------------------------------------------- #

def append_message(
    db: DBSession,
    session: Session,
    *,
    role: str,
    content: str,
    sources: Optional[Sequence[RetrievedChunk]] = None,
) -> Message:
    """Append a message to the session and commit."""
    payload = ""
    if sources:
        payload = json.dumps([s.to_dict() for s in sources], ensure_ascii=False)
    msg = Message(session_id=session.id, role=role, content=content, sources=payload)
    db.add(msg)
    db.commit()
    db.refresh(msg)
    return msg


def load_history(db: DBSession, session_id: str, window: int) -> List[ChatMessage]:
    """Load the last ``window`` (user + assistant) messages as OpenAI-wire dicts.

    Returned in chronological order (oldest first).
    """
    if window <= 0:
        return []
    stmt = (
        select(Message)
        .where(Message.session_id == session_id)
        .order_by(Message.created_at.desc(), Message.id.desc())
        .limit(window * 2)
    )
    rows = list(db.execute(stmt).scalars())
    rows.reverse()
    return [{"role": m.role, "content": m.content} for m in rows]


def parse_sources(raw: str) -> List[dict]:
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


# --------------------------------------------------------------------------- #
# High-level: answer + persist                                                #
# --------------------------------------------------------------------------- #

def handle_chat_turn(
    db: DBSession,
    *,
    session_id: Optional[str],
    user_message: str,
    vector_store: VectorStore,
    llm: LLM,
) -> Tuple[Session, RagAnswer]:
    """End-to-end handling of one user message.

    * Loads or creates the session.
    * Persists the user message.
    * Loads the last ``HISTORY_WINDOW`` turns.
    * Runs the RAG pipeline.
    * Persists the assistant reply (with sources JSON).
    * Returns ``(session, rag_answer)`` for the API layer to serialize.
    """
    s = get_settings()
    session = get_or_create_session(db, session_id)
    append_message(db, session, role="user", content=user_message)

    history = load_history(db, session.id, s.history_window)
    # Drop the just-inserted user turn from history so the RAG builder can add it back
    if history and history[-1]["role"] == "user" and history[-1]["content"] == user_message.strip():
        history = history[:-1]

    result = answer_question(
        user_message=user_message,
        history=history,
        vector_store=vector_store,
        llm=llm,
    )

    append_message(
        db,
        session,
        role="assistant",
        content=result.answer,
        sources=result.sources,
    )

    # Auto-title on the first exchange
    if session.title in ("", "New chat"):
        session.title = _make_title(user_message)
        db.commit()

    return session, result


def _make_title(first_user_message: str, max_len: int = 60) -> str:
    text = " ".join(first_user_message.split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"


def handle_chat_turn_stream(
    db: DBSession,
    *,
    session_id: Optional[str],
    user_message: str,
    vector_store: VectorStore,
    llm: LLM,
) -> Iterator[Tuple[str, Any]]:
    """Streaming variant of :func:`handle_chat_turn`.

    Persists the user message immediately, streams the assistant reply from
    the RAG service, and finally persists the accumulated assistant reply
    with its sources. Yields ``(event_type, payload)`` tuples:

    * ``("session", {"session_id": str, "title": str})`` — once at the start
      so the client knows which session it landed in.
    * ``("chunk", str)`` — one per LLM delta; contains raw model output
      (may include ``<think>...</think>`` for reasoning models).
    * ``("done", {"session_id": str, "sources": list[dict]})`` — once at the
      end, after the assistant message has been persisted.
    """
    s = get_settings()
    session = get_or_create_session(db, session_id)
    append_message(db, session, role="user", content=user_message)

    history = load_history(db, session.id, s.history_window)
    # Drop the just-inserted user turn from history.
    if history and history[-1]["role"] == "user" and history[-1]["content"] == user_message.strip():
        history = history[:-1]

    yield ("session", {"session_id": session.id, "title": session.title})

    final: Optional[RagAnswer] = None
    for event_type, payload in answer_question_stream(
        user_message=user_message,
        history=history,
        vector_store=vector_store,
        llm=llm,
    ):
        if event_type == "chunk":
            yield ("chunk", payload)
        elif event_type == "done":
            final = payload  # type: ignore[assignment]

    if final is not None:
        append_message(
            db,
            session,
            role="assistant",
            content=final.answer,
            sources=final.sources,
        )
        if session.title in ("", "New chat"):
            session.title = _make_title(user_message)
            db.commit()

    yield (
        "done",
        {
            "session_id": session.id,
            "sources": [c.to_dict() for c in (final.sources if final else [])],
        },
    )
