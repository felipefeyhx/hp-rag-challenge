"""Tests for app.services.rag_service."""
from __future__ import annotations

import pytest

from app.services.rag_service import (
    RagAnswer,
    _maybe_condense,
    answer_question,
    answer_question_stream,
)


# --- _maybe_condense ------------------------------------------------------ #

def test_maybe_condense_returns_message_when_no_history(fake_llm):
    result = _maybe_condense("hello", [], fake_llm)
    assert result == "hello"
    assert fake_llm.calls == []           # no LLM call


def test_maybe_condense_calls_llm_with_history():
    from tests.conftest import FakeLLM
    llm = FakeLLM(responses=["Rewritten standalone question."])
    history = [
        {"role": "user", "content": "prior question"},
        {"role": "assistant", "content": "prior answer"},
    ]
    result = _maybe_condense("follow up", history, llm)
    assert result == "Rewritten standalone question."
    assert len(llm.calls) == 1


def test_maybe_condense_strips_quotes():
    from tests.conftest import FakeLLM
    llm = FakeLLM(responses=['"quoted result"'])
    history = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]
    result = _maybe_condense("q", history, llm)
    assert result == "quoted result"


def test_maybe_condense_falls_back_when_empty_rewrite():
    from tests.conftest import FakeLLM
    llm = FakeLLM(responses=[""])
    history = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    result = _maybe_condense("original", history, llm)
    assert result == "original"


# --- answer_question ------------------------------------------------------ #

def test_answer_question_rejects_empty_message(fake_llm, fake_vector_store):
    with pytest.raises(ValueError):
        answer_question(
            user_message="",
            history=[],
            vector_store=fake_vector_store,
            llm=fake_llm,
        )


def test_answer_question_first_turn_no_condense(fake_vector_store):
    from tests.conftest import FakeLLM
    llm = FakeLLM(responses=["The paper tray holds up to 80 sheets."])

    result = answer_question(
        user_message="What is the paper tray capacity?",
        history=[],
        vector_store=fake_vector_store,
        llm=llm,
        top_k=2,
        strategy="mmr",
    )

    assert isinstance(result, RagAnswer)
    assert "80 sheets" in result.answer
    assert result.condensed_question == "What is the paper tray capacity?"
    assert len(result.sources) == 2
    # No condense call → LLM called exactly once
    assert len(llm.calls) == 1


def test_answer_question_follow_up_triggers_condense(fake_vector_store):
    from tests.conftest import FakeLLM
    llm = FakeLLM(responses=[
        "How many envelopes fit in the paper tray?",   # condense output
        "Up to 10 envelopes.",                          # final answer
    ])

    history = [
        {"role": "user", "content": "paper tray capacity?"},
        {"role": "assistant", "content": "80 sheets."},
    ]
    result = answer_question(
        user_message="and envelopes?",
        history=history,
        vector_store=fake_vector_store,
        llm=llm,
    )
    assert result.answer == "Up to 10 envelopes."
    assert "envelopes" in result.condensed_question.lower()
    assert len(llm.calls) == 2

    # Retrieval used the condensed query, not the raw one
    assert fake_vector_store.retrieve_calls[0]["query"] == result.condensed_question


def test_answer_question_uses_config_defaults_when_kwargs_missing(fake_vector_store, monkeypatch):
    monkeypatch.setenv("RETRIEVAL_TOP_K", "3")
    monkeypatch.setenv("RETRIEVAL_STRATEGY", "similarity")
    from app.config import reset_settings_cache
    reset_settings_cache()

    from tests.conftest import FakeLLM
    llm = FakeLLM(responses=["ok"])

    answer_question(
        user_message="query",
        history=[],
        vector_store=fake_vector_store,
        llm=llm,
    )
    call = fake_vector_store.retrieve_calls[0]
    assert call["top_k"] == 3
    assert call["strategy"] == "similarity"


# --- answer_question_stream ---------------------------------------------- #

def test_answer_question_stream_rejects_empty_message(fake_llm, fake_vector_store):
    with pytest.raises(ValueError):
        list(answer_question_stream(
            user_message="   ",
            history=[],
            vector_store=fake_vector_store,
            llm=fake_llm,
        ))


def test_answer_question_stream_emits_chunks_then_done(fake_vector_store):
    from tests.conftest import FakeLLM
    llm = FakeLLM(responses=["The tray holds 80 sheets."])

    events = list(answer_question_stream(
        user_message="What is the paper tray capacity?",
        history=[],
        vector_store=fake_vector_store,
        llm=llm,
    ))

    chunk_events = [payload for kind, payload in events if kind == "chunk"]
    done_events = [payload for kind, payload in events if kind == "done"]

    assert "".join(chunk_events) == "The tray holds 80 sheets."
    assert len(done_events) == 1
    final = done_events[0]
    assert isinstance(final, RagAnswer)
    assert final.answer == "The tray holds 80 sheets."
    assert len(final.sources) >= 1


def test_answer_question_stream_runs_condense_on_followup(fake_vector_store):
    from tests.conftest import FakeLLM
    llm = FakeLLM(responses=[
        "How many envelopes fit?",   # non-streaming condense
        "Up to 10.",                  # streamed answer
    ])
    history = [
        {"role": "user", "content": "paper tray?"},
        {"role": "assistant", "content": "80 sheets."},
    ]

    events = list(answer_question_stream(
        user_message="and envelopes?",
        history=history,
        vector_store=fake_vector_store,
        llm=llm,
    ))
    _, final = [e for e in events if e[0] == "done"][0]
    assert final.condensed_question == "How many envelopes fit?"
    assert final.answer == "Up to 10."
    # 1 non-streaming call for condense + 1 streaming call for the answer
    assert len(llm.calls) == 1
    assert len(llm.stream_calls) == 1
