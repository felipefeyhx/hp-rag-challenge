"""Ingest orchestration.

Called by ``scripts/ingest.py`` at container start. Idempotent: rerunning
does not duplicate chunks because we use deterministic IDs and Chroma's
``upsert``.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import List

from app.config import get_settings
from app.core.chunking import DocumentChunk, split_pages
from app.core.embeddings import get_embedder
from app.core.pdf_loader import load_pdf
from app.core.vectorstore import VectorStore, get_vector_store


@dataclass
class IngestReport:
    documents: List[str]
    pages: int
    chunks: int
    added: int  # chunks written to the store (== len(chunks) with upsert)
    already_indexed: int
    collection_size_after: int


def discover_pdfs(documents_dir: str) -> List[str]:
    """Return sorted PDF paths under ``documents_dir``."""
    pattern = os.path.join(documents_dir, "*.pdf")
    return sorted(glob.glob(pattern))


def run_ingest(
    *,
    documents_dir: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    store: VectorStore | None = None,
) -> IngestReport:
    """Ingest every PDF under ``documents_dir`` into the vector store."""
    s = get_settings()
    documents_dir = documents_dir or s.documents_dir
    chunk_size = chunk_size or s.chunk_size
    chunk_overlap = chunk_overlap or s.chunk_overlap

    paths = discover_pdfs(documents_dir)
    if not paths:
        raise FileNotFoundError(f"No PDFs found under {documents_dir}")

    store = store or get_vector_store(embedder=get_embedder())
    already = store.count()

    all_chunks: List[DocumentChunk] = []
    total_pages = 0
    for p in paths:
        pages = load_pdf(p)
        total_pages += len(pages)
        all_chunks.extend(split_pages(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap))

    added = store.upsert(all_chunks)
    return IngestReport(
        documents=[os.path.basename(p) for p in paths],
        pages=total_pages,
        chunks=len(all_chunks),
        added=added,
        already_indexed=already,
        collection_size_after=store.count(),
    )
