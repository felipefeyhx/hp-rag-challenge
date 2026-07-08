"""Health endpoint."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from app import __version__
from app.config import Settings, get_settings
from app.core.vectorstore import get_vector_store
from app.schemas.chat import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def healthcheck(settings: Settings = Depends(get_settings)) -> HealthResponse:
    try:
        docs = get_vector_store().count()
    except Exception:
        docs = 0
    return HealthResponse(
        status="ok",
        version=__version__,
        llm_model=settings.llm_model,
        vector_store_docs=docs,
    )
