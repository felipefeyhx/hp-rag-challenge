"""Tests for app.core.prompt."""
from __future__ import annotations

from app.core.prompt import (
    SYSTEM_PROMPT,
    CONDENSE_PROMPT,
    build_answer_messages,
    build_condense_messages,
    format_context,
)
from app.core.vectorstore import RetrievedChunk


def _chunk(text: str, source: str = "doc.pdf", page: int = 1, score: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(text=text, source=source, page=page, chunk_index=0, score=score)


# --- format_context -------------------------------------------------------- #

def test_format_context_empty_returns_placeholder():
    assert format_context([]) == "(no context retrieved)"


def test_format_context_numbers_and_labels_chunks():
    chunks = [
        _chunk("first bit of text", source="a.pdf", page=3),
        _chunk("second bit", source="b.pdf", page=7),
    ]
    out = format_context(chunks)
    assert "[1] (a.pdf, p.3)" in out
    assert "[2] (b.pdf, p.7)" in out
    assert "first bit of text" in out
    assert "second bit" in out


# --- build_answer_messages ------------------------------------------------- #

def test_build_answer_messages_shape_no_history():
    chunks = [_chunk("context text", source="a.pdf", page=1)]
    msgs = build_answer_messages([], "What is X?", chunks)
    assert msgs[0]["role"] == "system"
    assert SYSTEM_PROMPT.split("\n")[0] in msgs[0]["content"]
    assert "context text" in msgs[0]["content"]
    assert msgs[-1] == {"role": "user", "content": "What is X?"}


def test_build_answer_messages_includes_history_between_system_and_user():
    history = [
        {"role": "user", "content": "prior question"},
        {"role": "assistant", "content": "prior answer"},
    ]
    msgs = build_answer_messages(history, "next question", [_chunk("ctx")])
    roles = [m["role"] for m in msgs]
    assert roles == ["system", "user", "assistant", "user"]
    assert msgs[1]["content"] == "prior question"
    assert msgs[-1]["content"] == "next question"


def test_build_answer_messages_handles_empty_context():
    msgs = build_answer_messages([], "hello", [])
    # Still valid, and the system message declares "(no context retrieved)"
    assert "(no context retrieved)" in msgs[0]["content"]


# --- build_condense_messages ---------------------------------------------- #

def test_build_condense_messages_shape():
    history = [
        {"role": "user", "content": "prior"},
        {"role": "assistant", "content": "reply"},
    ]
    msgs = build_condense_messages(history, "follow-up question")
    assert msgs[0]["role"] == "system"
    assert CONDENSE_PROMPT.split("\n")[0] in msgs[0]["content"]
    assert msgs[1]["role"] == "user"
    assert "follow-up question" in msgs[1]["content"]
    assert "prior" in msgs[1]["content"]


def test_build_condense_messages_handles_empty_history():
    msgs = build_condense_messages([], "first question")
    assert "empty" in msgs[1]["content"].lower() or "(empty)" in msgs[1]["content"]
