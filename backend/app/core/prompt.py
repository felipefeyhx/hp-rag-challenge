"""Prompt building for the RAG chain.

Two prompts:

1. **Condense** — turn a follow-up message into a self-contained question
   using the chat history. Used before retrieval so the query has enough
   context on its own.
2. **Answer** — the main system + context + question prompt sent to the LLM.

Both return message lists in OpenAI wire format.
"""
from __future__ import annotations

from typing import List, Sequence

from app.core.llm import ChatMessage
from app.core.vectorstore import RetrievedChunk


SYSTEM_PROMPT = """\
You are an HP support assistant. You answer questions about HP products
using ONLY the context provided below. Follow these rules strictly:

* If the context does not contain enough information to answer, say
  "I don't have enough information in the provided documents to answer that."
  Do not invent facts.
* Cite the source of each factual statement with the format
  [<document>, p.<page>], e.g. [pdf_10301542_en-US-1.pdf, p.12].
* Prefer concise, direct answers. Use bullet points for steps.
* Answer in the same language the user used.
"""


CONDENSE_PROMPT = """\
Given the conversation history and the user's latest message, rewrite the
latest message as a standalone question that includes any context needed
to answer it on its own. Do not answer the question — only rewrite it.
If the latest message is already self-contained, return it unchanged.

Return ONLY the rewritten question, no explanations, no quotes.
"""


def format_context(chunks: Sequence[RetrievedChunk]) -> str:
    """Format retrieved chunks as a numbered context block for the prompt."""
    if not chunks:
        return "(no context retrieved)"
    lines = []
    for i, c in enumerate(chunks, start=1):
        header = f"[{i}] ({c.source}, p.{c.page})"
        lines.append(f"{header}\n{c.text.strip()}")
    return "\n\n".join(lines)


def build_condense_messages(history: Sequence[ChatMessage], user_message: str) -> List[ChatMessage]:
    """Build the messages sent to the LLM to condense a follow-up question."""
    convo = _format_history_as_text(history)
    user_block = (
        f"Conversation so far:\n{convo}\n\n"
        f"Latest user message: {user_message.strip()}\n\n"
        f"Standalone question:"
    )
    return [
        {"role": "system", "content": CONDENSE_PROMPT},
        {"role": "user", "content": user_block},
    ]


def build_answer_messages(
    history: Sequence[ChatMessage],
    user_message: str,
    context_chunks: Sequence[RetrievedChunk],
) -> List[ChatMessage]:
    """Build the messages sent to the LLM to produce the final answer."""
    context_block = format_context(context_chunks)
    system = SYSTEM_PROMPT + "\n\nContext:\n" + context_block

    messages: List[ChatMessage] = [{"role": "system", "content": system}]
    for m in history:
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": user_message.strip()})
    return messages


def _format_history_as_text(history: Sequence[ChatMessage]) -> str:
    if not history:
        return "(empty)"
    lines = []
    for m in history:
        role = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"{role}: {m['content']}")
    return "\n".join(lines)
