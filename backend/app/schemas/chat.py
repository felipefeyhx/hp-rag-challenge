"""Pydantic schemas for the chat API."""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class SourceChunk(BaseModel):
    """A retrieved chunk returned alongside an answer."""
    source: str
    page: int
    chunk_index: int = 0
    score: Optional[float] = None
    text: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    session_id: Optional[str] = Field(
        default=None,
        description="Existing session id. Omit to start a new session.",
    )


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: List[SourceChunk] = []


class MessageOut(BaseModel):
    role: str
    content: str
    sources: List[SourceChunk] = []
    created_at: datetime


class SessionOut(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime


class SessionDetail(SessionOut):
    messages: List[MessageOut] = []


class HealthResponse(BaseModel):
    status: str
    version: str
    llm_model: str
    vector_store_docs: int
