"""Application configuration loaded from environment variables."""
from __future__ import annotations

from functools import lru_cache
from typing import ClassVar, List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# Model choices are fixed and NOT overridable via environment.
# The ingest job pulls specific tags at container start, and switching the
# embedding model would invalidate the vector store. Changing the model
# requires a code change here (plus a matching update to docker-compose.yml).
LLM_MODEL: str = "qwen2.5:7b-instruct"
EMBEDDING_MODEL: str = "nomic-embed-text"


class Settings(BaseSettings):
    """Typed settings for the backend service.

    Every field below is overridable via the corresponding environment
    variable (see ``.env.example``). The **model choice** is intentionally
    fixed as a module constant (see ``LLM_MODEL``, ``EMBEDDING_MODEL``
    above) and exposed here as a ``ClassVar`` so ``settings.llm_model``
    still works but no env var will override it.

    LLM / embedding endpoints are provider-neutral: any HTTP endpoint that
    speaks the OpenAI chat-completions/embeddings wire protocol works
    (Ollama, LM Studio, vLLM, llama.cpp, LocalAI, etc.). Defaults point at
    a local Ollama instance so nothing leaves the machine.
    """

    # ------- LLM (chat completions) ------- #
    # Base URL of an OpenAI-compatible endpoint. For Ollama on the host,
    # keep the default. Inside Docker use http://ollama:11434/v1 .
    llm_base_url: str = Field(default="http://localhost:11434/v1")
    # Runners that don't require auth (Ollama) still expect a non-empty
    # string. This is never sent anywhere external.
    llm_api_key: str = Field(default="ollama")
    # Fixed — see LLM_MODEL constant above.
    llm_model: ClassVar[str] = LLM_MODEL

    # ------- Embeddings ------- #
    embedding_base_url: str = Field(default="http://localhost:11434/v1")
    embedding_api_key: str = Field(default="ollama")
    # Fixed — see EMBEDDING_MODEL constant above.
    embedding_model: ClassVar[str] = EMBEDDING_MODEL

    # ------- Vector store (Chroma embedded) ------- #
    chroma_persist_dir: str = Field(default="/app/data/chroma")
    chroma_collection: str = Field(default="hp_docs")

    # ------- Chunking ------- #
    chunk_size: int = Field(default=800, ge=100, le=4000)
    chunk_overlap: int = Field(default=120, ge=0, le=1000)

    # ------- Retrieval ------- #
    retrieval_strategy: str = Field(default="mmr")
    retrieval_top_k: int = Field(default=5, ge=1, le=20)
    retrieval_fetch_k: int = Field(default=20, ge=1, le=100)
    retrieval_mmr_lambda: float = Field(default=0.5, ge=0.0, le=1.0)

    # ------- Chat history window sent to the LLM (DB keeps everything) ------- #
    history_window: int = Field(default=6, ge=0, le=50)

    # ------- Persistence ------- #
    database_url: str = Field(default="sqlite:////app/data/db/chatbot.db")

    # ------- API ------- #
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    cors_origins: str = Field(default="*")

    # ------- Documents (used by the ingest script) ------- #
    documents_dir: str = Field(default="/app/data/documents")

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")

    @field_validator("retrieval_strategy")
    @classmethod
    def _validate_strategy(cls, v: str) -> str:
        v = v.lower().strip()
        if v not in {"similarity", "mmr"}:
            raise ValueError("RETRIEVAL_STRATEGY must be 'similarity' or 'mmr'")
        return v

    @field_validator("chunk_overlap")
    @classmethod
    def _validate_overlap(cls, v: int, info) -> int:
        size = info.data.get("chunk_size", 800)
        if v >= size:
            raise ValueError("CHUNK_OVERLAP must be strictly smaller than CHUNK_SIZE")
        return v

    def cors_origin_list(self) -> List[str]:
        raw = self.cors_origins.strip()
        if not raw or raw == "*":
            return ["*"]
        return [o.strip() for o in raw.split(",") if o.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached ``Settings`` instance."""
    return Settings()


def reset_settings_cache() -> None:
    """Reset the cached settings (used in tests)."""
    get_settings.cache_clear()
