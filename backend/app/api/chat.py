"""Chat endpoint — the main /chat POST plus an SSE streaming variant."""
from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session as DBSession

from app.core.llm import get_llm
from app.core.vectorstore import get_vector_store
from app.db.database import get_db
from app.schemas.chat import ChatRequest, ChatResponse, SourceChunk
from app.services import chat_service

router = APIRouter(tags=["chat"])


def _sse(event: str, data) -> str:
    """Encode one SSE event. `data` is JSON-serialized."""
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, db: DBSession = Depends(get_db)) -> ChatResponse:
    try:
        session, result = chat_service.handle_chat_turn(
            db,
            session_id=payload.session_id,
            user_message=payload.message,
            vector_store=get_vector_store(),
            llm=get_llm(),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"chat failed: {exc}")

    return ChatResponse(
        session_id=session.id,
        answer=result.answer,
        sources=[
            SourceChunk(
                source=c.source,
                page=c.page,
                chunk_index=c.chunk_index,
                score=c.score,
                text=c.text,
            )
            for c in result.sources
        ],
    )


@router.post("/chat/stream")
def chat_stream(payload: ChatRequest, db: DBSession = Depends(get_db)) -> StreamingResponse:
    """Server-Sent Events stream of the assistant reply.

    Wire format:
        event: session
        data: {"session_id": "...", "title": "..."}

        event: chunk
        data: {"text": "…partial output…"}

        event: chunk
        data: {"text": "…more…"}

        event: done
        data: {"session_id": "...", "sources": [...]}

    On errors:
        event: error
        data: {"message": "..."}
    """

    def event_stream():
        try:
            for event_type, data in chat_service.handle_chat_turn_stream(
                db,
                session_id=payload.session_id,
                user_message=payload.message,
                vector_store=get_vector_store(),
                llm=get_llm(),
            ):
                if event_type == "chunk":
                    yield _sse("chunk", {"text": data})
                else:
                    yield _sse(event_type, data)
        except ValueError as exc:
            yield _sse("error", {"message": str(exc)})
        except Exception as exc:  # pragma: no cover - defensive
            yield _sse("error", {"message": f"chat stream failed: {exc}"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        # Ask any reverse proxy (nginx, etc.) not to buffer this response.
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
