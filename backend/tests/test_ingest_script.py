"""Tests for scripts.ingest (CLI entrypoint)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pytest


@dataclass
class _FakeReport:
    documents: List[str]
    pages: int
    chunks: int
    added: int
    already_indexed: int
    collection_size_after: int


def test_ingest_main_success(monkeypatch, capsys):
    from scripts import ingest

    def _fake_run_ingest():
        return _FakeReport(
            documents=["a.pdf", "b.pdf"],
            pages=10, chunks=25, added=25,
            already_indexed=0, collection_size_after=25,
        )
    monkeypatch.setattr(ingest, "run_ingest", _fake_run_ingest)
    rc = ingest.main()
    assert rc == 0
    out = capsys.readouterr().out
    assert "chunks upserted" in out
    assert "25" in out


def test_ingest_main_reports_missing_pdfs(monkeypatch, capsys):
    from scripts import ingest

    def _fake_run_ingest():
        raise FileNotFoundError("no pdfs")
    monkeypatch.setattr(ingest, "run_ingest", _fake_run_ingest)
    rc = ingest.main()
    assert rc == 1


def test_ingest_main_generic_failure(monkeypatch, capsys):
    from scripts import ingest

    def _fake_run_ingest():
        raise RuntimeError("something else")
    monkeypatch.setattr(ingest, "run_ingest", _fake_run_ingest)
    rc = ingest.main()
    assert rc == 2
