"""Tests for app.core.llm."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.core.llm import ChatLLM, LLMError, get_llm, reset_llm_cache


# --- Fake OpenAI SDK ------------------------------------------------------ #

class _FakeCompletions:
    def __init__(self, reply="hello", raise_exc=None) -> None:
        self.reply = reply
        self.raise_exc = raise_exc
        self.last_kwargs: dict = {}

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        if self.raise_exc:
            raise self.raise_exc
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.reply))]
        )


class _FakeClient:
    def __init__(self, api_key=None, base_url=None, reply="hello", raise_exc=None) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.chat = SimpleNamespace(completions=_FakeCompletions(reply, raise_exc))


def _patch_openai(monkeypatch, *, reply="hello", raise_exc=None):
    import sys

    class _FakeOpenAIModule:
        OpenAI = staticmethod(lambda **kw: _FakeClient(reply=reply, raise_exc=raise_exc, **kw))

    monkeypatch.setitem(sys.modules, "openai", _FakeOpenAIModule())


# --- ChatLLM -------------------------------------------------------------- #

def test_chat_llm_requires_base_url(monkeypatch):
    _patch_openai(monkeypatch)
    with pytest.raises(ValueError):
        ChatLLM(model="llama3.1:8b", base_url="")


def test_chat_llm_requires_model(monkeypatch):
    _patch_openai(monkeypatch)
    with pytest.raises(ValueError):
        ChatLLM(model="", base_url="http://localhost:11434/v1")


def test_chat_llm_returns_content(monkeypatch):
    _patch_openai(monkeypatch, reply="pong")
    llm = ChatLLM(model="llama3.1:8b", base_url="http://localhost:11434/v1")
    out = llm.chat([{"role": "user", "content": "ping"}])
    assert out == "pong"


def test_chat_llm_passes_kwargs(monkeypatch):
    _patch_openai(monkeypatch, reply="ok")
    llm = ChatLLM(model="qwen2.5:7b", base_url="http://localhost:11434/v1")
    llm.chat([{"role": "user", "content": "hi"}], temperature=0.7, max_tokens=64)
    kw = llm._client.chat.completions.last_kwargs
    assert kw["model"] == "qwen2.5:7b"
    assert kw["temperature"] == 0.7
    assert kw["max_tokens"] == 64
    assert kw["messages"] == [{"role": "user", "content": "hi"}]


def test_chat_llm_strips_whitespace(monkeypatch):
    _patch_openai(monkeypatch, reply="   hi   ")
    llm = ChatLLM(model="llama3.1:8b", base_url="http://localhost:11434/v1")
    assert llm.chat([{"role": "user", "content": "hi"}]) == "hi"


def test_chat_llm_returns_empty_for_none_content(monkeypatch):
    class _NoneCompletions:
        def create(self, **kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=None))]
            )

    class _NoneClient:
        def __init__(self, **kw): self.chat = SimpleNamespace(completions=_NoneCompletions())

    import sys
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=_NoneClient))
    llm = ChatLLM(model="llama3.1:8b", base_url="http://localhost:11434/v1")
    assert llm.chat([{"role": "user", "content": "x"}]) == ""


def test_chat_llm_wraps_errors(monkeypatch):
    _patch_openai(monkeypatch, raise_exc=RuntimeError("boom"))
    llm = ChatLLM(model="llama3.1:8b", base_url="http://localhost:11434/v1")
    with pytest.raises(LLMError):
        llm.chat([{"role": "user", "content": "x"}])


# --- ChatLLM.chat_stream -------------------------------------------------- #

class _FakeStreamCompletions:
    """Yields OpenAI-shape stream chunks with a `delta.content` string."""

    def __init__(self, deltas, raise_after=None):
        self.deltas = list(deltas)
        self.raise_after = raise_after
        self.last_kwargs: dict = {}

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        assert kwargs.get("stream") is True
        for i, text in enumerate(self.deltas):
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=text))]
            )


def _patch_openai_stream(monkeypatch, deltas):
    import sys

    class _StreamClient:
        def __init__(self, **_kw):
            self.chat = SimpleNamespace(completions=_FakeStreamCompletions(deltas))

    class _FakeOpenAIModule:
        OpenAI = staticmethod(lambda **kw: _StreamClient(**kw))

    monkeypatch.setitem(sys.modules, "openai", _FakeOpenAIModule())


def test_chat_stream_yields_content_deltas(monkeypatch):
    _patch_openai_stream(monkeypatch, ["Hello", ", ", "world!"])
    llm = ChatLLM(model="qwen2.5:7b-instruct", base_url="http://localhost:11434/v1")
    parts = list(llm.chat_stream([{"role": "user", "content": "hi"}]))
    assert parts == ["Hello", ", ", "world!"]


def test_chat_stream_skips_empty_deltas(monkeypatch):
    # The first/last SSE chunks in the OpenAI protocol carry `delta.content=None`
    # (role or finish_reason only). The adapter must swallow them.
    _patch_openai_stream(monkeypatch, ["", None, "real"])
    llm = ChatLLM(model="qwen2.5:7b-instruct", base_url="http://localhost:11434/v1")
    parts = list(llm.chat_stream([{"role": "user", "content": "hi"}]))
    assert parts == ["real"]


def test_chat_stream_skips_chunks_without_choices(monkeypatch):
    """OpenAI-compat servers may occasionally send heartbeat chunks with choices=[]."""

    class _MixedCompletions:
        def create(self, **kwargs):
            yield SimpleNamespace(choices=[])
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="ok"))])

    import sys
    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(OpenAI=lambda **_kw: SimpleNamespace(
            chat=SimpleNamespace(completions=_MixedCompletions())
        )),
    )
    llm = ChatLLM(model="qwen2.5:7b-instruct", base_url="http://localhost:11434/v1")
    assert list(llm.chat_stream([{"role": "user", "content": "hi"}])) == ["ok"]


def test_chat_stream_wraps_setup_errors(monkeypatch):
    class _BoomCompletions:
        def create(self, **_kw):
            raise RuntimeError("connect refused")

    import sys
    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(OpenAI=lambda **_kw: SimpleNamespace(
            chat=SimpleNamespace(completions=_BoomCompletions())
        )),
    )
    llm = ChatLLM(model="qwen2.5:7b-instruct", base_url="http://localhost:11434/v1")
    with pytest.raises(LLMError):
        list(llm.chat_stream([{"role": "user", "content": "hi"}]))


# --- get_llm cache -------------------------------------------------------- #

def test_get_llm_caches(monkeypatch):
    _patch_openai(monkeypatch)
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("LLM_API_KEY", "unused")
    from app.config import reset_settings_cache
    reset_settings_cache()
    reset_llm_cache()

    l1 = get_llm()
    l2 = get_llm()
    assert l1 is l2
    reset_llm_cache()
    l3 = get_llm()
    assert l3 is not l1
