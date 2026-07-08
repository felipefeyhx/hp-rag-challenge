"""Tests for app.core.pdf_loader.

We monkeypatch LangChain's PyPDFLoader so the tests don't need a real PDF on disk.
"""
from __future__ import annotations

import pytest

from app.core import pdf_loader
from app.core.pdf_loader import DocumentPage, _clean, load_pdf, load_pdfs


class _FakeDoc:
    def __init__(self, page_content: str, metadata=None) -> None:
        self.page_content = page_content
        self.metadata = metadata or {}


class _FakeLoader:
    def __init__(self, path: str) -> None:
        self.path = path

    def load(self):
        # Third doc has no metadata['page'], so 1-based enumeration is used.
        return [
            _FakeDoc("Page one\n\ntext with docu-\nment hyphen.", metadata={"page": 0}),
            _FakeDoc("   \n\n\nExtra\r\nlines\n\n\n\n\n", metadata={"page": 1}),
            _FakeDoc(""),
        ]


# --- _clean ---------------------------------------------------------------- #

def test_clean_empty_returns_empty():
    assert _clean("") == ""
    assert _clean(None) == ""  # type: ignore[arg-type]


def test_clean_dehyphenates():
    assert "document" in _clean("docu-\nment")


def test_clean_normalizes_line_endings():
    assert "\r\n" not in _clean("a\r\nb")
    assert "\r" not in _clean("a\rb")


def test_clean_collapses_many_newlines():
    result = _clean("a\n\n\n\n\nb")
    assert result == "a\n\nb"


def test_clean_trims_trailing_spaces_on_lines():
    assert _clean("line   \nnext") == "line\nnext"


# --- load_pdf -------------------------------------------------------------- #

def test_load_pdf_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_pdf(str(tmp_path / "does-not-exist.pdf"))


def test_load_pdf_uses_metadata_page_when_present(monkeypatch, tmp_path):
    p = tmp_path / "fake.pdf"
    p.write_bytes(b"not really a pdf but we monkeypatch the loader")
    monkeypatch.setattr(pdf_loader, "PyPDFLoader", _FakeLoader, raising=False)
    # Force the lazy import inside load_pdf to use our fake
    monkeypatch.setitem(
        __import__("sys").modules,
        "langchain_community.document_loaders",
        _FakeDocLoadersModule(),
    )
    pages = load_pdf(str(p))
    assert len(pages) == 3
    assert pages[0].source == "fake.pdf"
    # Metadata page 0 → 1-indexed
    assert pages[0].page == 1
    assert "document hyphen" in pages[0].text
    # Third page had no metadata → fallback to enumerate index 3
    assert pages[2].page == 3
    assert pages[2].text == ""


def test_load_pdfs_concatenates(monkeypatch, tmp_path):
    a = tmp_path / "a.pdf"
    b = tmp_path / "b.pdf"
    a.write_bytes(b"a"); b.write_bytes(b"b")
    monkeypatch.setitem(
        __import__("sys").modules,
        "langchain_community.document_loaders",
        _FakeDocLoadersModule(),
    )
    pages = load_pdfs([str(a), str(b)])
    assert len(pages) == 6  # 3 fake pages per file
    assert {p.source for p in pages} == {"a.pdf", "b.pdf"}


# --------------------------------------------------------------------------- #
# Utility: fake module exposing PyPDFLoader                                   #
# --------------------------------------------------------------------------- #

class _FakeDocLoadersModule:
    PyPDFLoader = _FakeLoader
