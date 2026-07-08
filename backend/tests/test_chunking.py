"""Tests for app.core.chunking."""
from __future__ import annotations

import pytest

from app.core.chunking import DocumentChunk, split_pages
from app.core.pdf_loader import DocumentPage


def _pg(text: str, page: int = 1, source: str = "doc.pdf") -> DocumentPage:
    return DocumentPage(source=source, page=page, text=text)


# --- DocumentChunk helpers ------------------------------------------------- #

def test_document_chunk_deterministic_id():
    c = DocumentChunk(text="x", source="a.pdf", page=3, chunk_index=7)
    assert c.deterministic_id() == "a.pdf::p3::c7"


def test_document_chunk_metadata_includes_extra():
    c = DocumentChunk(text="x", source="a.pdf", page=3, chunk_index=7, extra={"kind": "text"})
    md = c.as_metadata()
    assert md["source"] == "a.pdf"
    assert md["page"] == 3
    assert md["chunk_index"] == 7
    assert md["kind"] == "text"


# --- split_pages ----------------------------------------------------------- #

def test_split_pages_short_text_single_chunk():
    pg = _pg("Short paragraph.", page=1)
    chunks = split_pages([pg], chunk_size=800, chunk_overlap=120)
    assert len(chunks) == 1
    assert chunks[0].text == "Short paragraph."
    assert chunks[0].source == "doc.pdf"
    assert chunks[0].page == 1


def test_split_pages_skips_empty_pages():
    pgs = [_pg("", 1), _pg("   \n\n", 2), _pg("Real content here.", 3)]
    chunks = split_pages(pgs, chunk_size=800, chunk_overlap=120)
    assert len(chunks) == 1
    assert chunks[0].page == 3


def test_split_pages_produces_multiple_chunks_for_long_input():
    text = ("Paragraph one. " * 40) + "\n\n" + ("Paragraph two. " * 40)
    pg = _pg(text, page=5)
    chunks = split_pages([pg], chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1
    # Every chunk should stay under a reasonable overshoot of chunk_size
    for c in chunks:
        assert len(c.text) <= 260
        assert c.source == "doc.pdf"
        assert c.page == 5


def test_split_pages_chunk_index_resets_per_page():
    pgs = [
        _pg("Alpha content on page one.", page=1),
        _pg("Beta content on page two.", page=2),
    ]
    chunks = split_pages(pgs, chunk_size=800, chunk_overlap=120)
    assert [c.chunk_index for c in chunks] == [0, 0]  # each page starts at 0


def test_split_pages_rejects_bad_chunk_size():
    with pytest.raises(ValueError):
        split_pages([_pg("x")], chunk_size=0, chunk_overlap=0)


def test_split_pages_rejects_negative_overlap():
    with pytest.raises(ValueError):
        split_pages([_pg("x")], chunk_size=100, chunk_overlap=-1)


def test_split_pages_rejects_overlap_ge_size():
    with pytest.raises(ValueError):
        split_pages([_pg("x")], chunk_size=100, chunk_overlap=100)


def test_split_pages_accepts_custom_separators():
    pg = _pg("A|B|C", page=1)
    chunks = split_pages([pg], chunk_size=3, chunk_overlap=0, separators=["|", ""])
    assert len(chunks) >= 2
