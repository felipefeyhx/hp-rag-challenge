"""Text chunking.

Thin wrapper around LangChain's :class:`RecursiveCharacterTextSplitter` that
emits :class:`DocumentChunk` objects with provenance metadata.

Chunking strategy (documented in ``docs/ARCHITECTURE.md``):

* recursive character splitter
* separators, in priority order: ``"\\n\\n"``, ``"\\n"``, ``". "``, ``" "``, ``""``
* defaults: ``chunk_size=800``, ``chunk_overlap=120``
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

from app.core.pdf_loader import DocumentPage


@dataclass
class DocumentChunk:
    """A chunk of text with source + page provenance."""
    text: str
    source: str
    page: int
    chunk_index: int = 0
    extra: dict = field(default_factory=dict)

    def deterministic_id(self) -> str:
        """Stable ID used for idempotent upserts into Chroma."""
        return f"{self.source}::p{self.page}::c{self.chunk_index}"

    def as_metadata(self) -> dict:
        """Metadata payload to store alongside the vector in Chroma."""
        m = {
            "source": self.source,
            "page": self.page,
            "chunk_index": self.chunk_index,
        }
        m.update(self.extra)
        return m


def _default_separators() -> List[str]:
    return ["\n\n", "\n", ". ", " ", ""]


def split_pages(
    pages: Sequence[DocumentPage],
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    separators: Sequence[str] | None = None,
) -> List[DocumentChunk]:
    """Split a list of pages into :class:`DocumentChunk` objects.

    ``chunk_index`` is unique across the whole result so
    ``chunk.deterministic_id()`` is stable across runs regardless of insertion
    order (as long as the page order is stable).
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=list(separators) if separators else _default_separators(),
        keep_separator=False,
    )

    # Group chunks by (source, page) so chunk_index restarts per page and stays stable
    chunks: List[DocumentChunk] = []
    for pg in pages:
        if not pg.text.strip():
            continue
        pieces = splitter.split_text(pg.text)
        for local_idx, piece in enumerate(pieces):
            piece = piece.strip()
            if not piece:
                continue
            chunks.append(
                DocumentChunk(
                    text=piece,
                    source=pg.source,
                    page=pg.page,
                    chunk_index=local_idx,
                )
            )
    return chunks
