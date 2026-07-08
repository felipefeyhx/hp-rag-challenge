"""RAG orchestration.

Explicit condense → retrieve → answer loop. This gives us the same behavior
as LangChain's ``ConversationalRetrievalChain`` but without depending on its
fast-moving internals, which makes it easier to test and reason about.

Public entry point: :func:`answer_question`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, List, Optional, Sequence, Tuple

from app.config import get_settings
from app.core.llm import LLM, ChatMessage, strip_think_blocks
from app.core.prompt import (
    build_answer_messages,
    build_condense_messages,
)


# Generous cap that fits both a reasoning-model <think> block and the
# subsequent answer for typical RAG questions. Non-reasoning models are
# unaffected — they simply won't emit that many tokens.
_ANSWER_MAX_TOKENS = 4096
_CONDENSE_MAX_TOKENS = 2048
from app.core.vectorstore import RetrievedChunk, VectorStore


@dataclass
class RagAnswer:
    """Answer + retrieved sources for a single turn."""
    answer: str
    sources: List[RetrievedChunk]
    condensed_question: str


def answer_question(
    *,
    user_message: str,
    history: Sequence[ChatMessage],
    vector_store: VectorStore,
    llm: LLM,
    top_k: Optional[int] = None,
    fetch_k: Optional[int] = None,
    strategy: Optional[str] = None,
    mmr_lambda: Optional[float] = None,
) -> RagAnswer:
    """Answer a user message using RAG on the provided vector store.

    Steps:

    1. If there is prior history, condense the latest message into a
       standalone question via a small LLM call. This improves retrieval
       for pronouns and follow-ups (*"and how much does it weigh?"*).
    2. Retrieve top-k chunks from the vector store using MMR.
    3. Build the answer prompt and call the LLM.

    Parameters
    ----------
    user_message:
        The latest user turn (raw, not condensed).
    history:
        Prior turns in OpenAI wire format, already truncated to the
        history window by the caller.
    vector_store:
        The vector store to query.
    llm:
        The chat LLM to use.
    top_k, fetch_k, strategy, mmr_lambda:
        Retrieval knobs; fall back to :func:`app.config.get_settings`.
    """
    s = get_settings()
    top_k = top_k if top_k is not None else s.retrieval_top_k
    fetch_k = fetch_k if fetch_k is not None else s.retrieval_fetch_k
    strategy = strategy or s.retrieval_strategy
    mmr_lambda = mmr_lambda if mmr_lambda is not None else s.retrieval_mmr_lambda

    user_message = (user_message or "").strip()
    if not user_message:
        raise ValueError("user_message must be non-empty")

    # 1. Condense (only if we have history and the message looks context-dependent)
    condensed = _maybe_condense(user_message, history, llm)

    # 2. Retrieve
    retrieved = vector_store.retrieve(
        condensed,
        top_k=top_k,
        fetch_k=fetch_k,
        strategy=strategy,
        mmr_lambda=mmr_lambda,
    )

    # 3. Answer. Cap tokens generously so a reasoning model has room for
    # both the <think> trace and the final answer.
    messages = build_answer_messages(history, user_message, retrieved)
    answer = llm.chat(messages, temperature=0.0, max_tokens=_ANSWER_MAX_TOKENS)

    return RagAnswer(answer=answer, sources=list(retrieved), condensed_question=condensed)


def answer_question_stream(
    *,
    user_message: str,
    history: Sequence[ChatMessage],
    vector_store: VectorStore,
    llm: LLM,
    top_k: Optional[int] = None,
    fetch_k: Optional[int] = None,
    strategy: Optional[str] = None,
    mmr_lambda: Optional[float] = None,
) -> Iterator[Tuple[str, Any]]:
    """Streaming variant of :func:`answer_question`.

    Runs the condense + retrieve steps non-streaming (they're quick and small),
    then streams the answer LLM call. Yields ``(event_type, payload)`` tuples:

    * ``("chunk", str)`` — one or more per turn, each carrying a fresh piece of
      the model's raw output (including any ``<think>...</think>`` text).
    * ``("done", RagAnswer)`` — exactly once, at the end, with the accumulated
      answer and the retrieved sources.
    """
    s = get_settings()
    top_k = top_k if top_k is not None else s.retrieval_top_k
    fetch_k = fetch_k if fetch_k is not None else s.retrieval_fetch_k
    strategy = strategy or s.retrieval_strategy
    mmr_lambda = mmr_lambda if mmr_lambda is not None else s.retrieval_mmr_lambda

    user_message = (user_message or "").strip()
    if not user_message:
        raise ValueError("user_message must be non-empty")

    condensed = _maybe_condense(user_message, history, llm)

    retrieved = vector_store.retrieve(
        condensed,
        top_k=top_k,
        fetch_k=fetch_k,
        strategy=strategy,
        mmr_lambda=mmr_lambda,
    )

    messages = build_answer_messages(history, user_message, retrieved)

    parts: List[str] = []
    for chunk in llm.chat_stream(messages, temperature=0.0, max_tokens=_ANSWER_MAX_TOKENS):
        parts.append(chunk)
        yield ("chunk", chunk)

    full_answer = "".join(parts).strip()
    yield ("done", RagAnswer(
        answer=full_answer,
        sources=list(retrieved),
        condensed_question=condensed,
    ))


# --------------------------------------------------------------------------- #
# Internals                                                                   #
# --------------------------------------------------------------------------- #

def _maybe_condense(user_message: str, history: Sequence[ChatMessage], llm: LLM) -> str:
    """Return a standalone question, using an LLM rewrite when it helps.

    We only pay for the rewrite when there is prior history — first-turn
    questions are already standalone.
    """
    if not history:
        return user_message

    messages = build_condense_messages(history, user_message)
    rewritten = llm.chat(messages, temperature=0.0, max_tokens=_CONDENSE_MAX_TOKENS)
    # Retrieval must never see reasoning-model <think> traces.
    rewritten = strip_think_blocks(rewritten)
    rewritten = rewritten.strip().strip('"').strip()
    return rewritten or user_message
