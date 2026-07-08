"""Tests for app.services.ingest_service."""
from __future__ import annotations

import pytest

from app.core.chunking import DocumentChunk
from app.core.pdf_loader import DocumentPage
from app.services import ingest_service
from app.services.ingest_service import discover_pdfs, run_ingest


def _make_pdf_files(root, names=("a.pdf", "b.pdf")):
    root.mkdir(exist_ok=True)
    for n in names:
        (root / n).write_bytes(b"fake")
    # Non-PDF should be ignored by discover_pdfs
    (root / "readme.txt").write_text("noise")


def test_discover_pdfs_returns_sorted_pdfs(tmp_path):
    _make_pdf_files(tmp_path, names=("z.pdf", "a.pdf"))
    result = discover_pdfs(str(tmp_path))
    assert [r.split("/")[-1].split("\\")[-1] for r in result] == ["a.pdf", "z.pdf"]


def test_discover_pdfs_empty_folder(tmp_path):
    assert discover_pdfs(str(tmp_path)) == []


def test_run_ingest_raises_when_no_pdfs(tmp_path, fake_vector_store):
    with pytest.raises(FileNotFoundError):
        run_ingest(documents_dir=str(tmp_path), store=fake_vector_store)


def test_run_ingest_end_to_end(monkeypatch, tmp_path, fake_vector_store):
    _make_pdf_files(tmp_path, names=("hp.pdf",))

    def _fake_load(path):
        return [DocumentPage(source="hp.pdf", page=1, text="Sample text on the page.")]

    monkeypatch.setattr(ingest_service, "load_pdf", _fake_load)

    report = run_ingest(documents_dir=str(tmp_path), store=fake_vector_store)
    assert report.documents == ["hp.pdf"]
    assert report.pages == 1
    assert report.chunks >= 1
    assert report.added == report.chunks
    assert report.collection_size_after >= report.chunks


def test_run_ingest_uses_settings_defaults(monkeypatch, tmp_path, fake_vector_store):
    _make_pdf_files(tmp_path, names=("x.pdf",))
    monkeypatch.setenv("DOCUMENTS_DIR", str(tmp_path))
    monkeypatch.setenv("CHUNK_SIZE", "300")
    monkeypatch.setenv("CHUNK_OVERLAP", "20")
    from app.config import reset_settings_cache
    reset_settings_cache()

    def _fake_load(path):
        return [DocumentPage(source="x.pdf", page=1, text="Alpha content.")]

    monkeypatch.setattr(ingest_service, "load_pdf", _fake_load)

    report = run_ingest(store=fake_vector_store)
    assert report.chunks >= 1
