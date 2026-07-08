"""Session CRUD endpoints."""
from __future__ import annotations

import json
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.orm import Session as DBSession

from app.db.database import get_db
from app.schemas.chat import MessageOut, SessionDetail, SessionOut, SourceChunk
from app.services import chat_service

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.post("", response_model=SessionOut, status_code=status.HTTP_201_CREATED)
def create_session(db: DBSession = Depends(get_db)) -> SessionOut:
    sess = chat_service.create_session(db)
    return SessionOut.model_validate(sess, from_attributes=True)


@router.get("", response_model=List[SessionOut])
def list_sessions(db: DBSession = Depends(get_db)) -> List[SessionOut]:
    sessions = chat_service.list_sessions(db)
    return [SessionOut.model_validate(s, from_attributes=True) for s in sessions]


@router.get("/{session_id}", response_model=SessionDetail)
def get_session(session_id: str, db: DBSession = Depends(get_db)) -> SessionDetail:
    sess = chat_service.get_session(db, session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="Session not found")

    messages = [
        MessageOut(
            role=m.role,
            content=m.content,
            sources=_parse_sources(m.sources),
            created_at=m.created_at,
        )
        for m in sess.messages
    ]
    return SessionDetail(
        id=sess.id,
        title=sess.title,
        created_at=sess.created_at,
        updated_at=sess.updated_at,
        messages=messages,
    )


@router.delete(
    "/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
)
def delete_session(session_id: str, db: DBSession = Depends(get_db)) -> Response:
    ok = chat_service.delete_session(db, session_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Session not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


def _parse_sources(raw: str) -> List[SourceChunk]:
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    out: List[SourceChunk] = []
    if not isinstance(data, list):
        return out
    for item in data:
        try:
            out.append(SourceChunk(**item))
        except Exception:
            continue
    return out
