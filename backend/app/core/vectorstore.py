"""ChromaDB adapter with MMR reranking.

The rest of the app talks only to :class:`VectorStore`. Chroma's Python API
is wrapped so we can swap implementations without touching consumers.

The store is intentionally read-only after ingest — building the index is
handled by ``scripts.ingest`` at container start.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from app.config import get_settings
from app.core.chunking import DocumentChunk
from app.core.embeddings import Embedder


@dataclass
class RetrievedChunk:
    """A chunk returned by :meth:`VectorStore.retrieve`."""
    text: str
    source: str
    page: int
    chunk_index: int
    score: float  # cosine similarity in [-1, 1]

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "source": self.source,
            "page": self.page,
            "chunk_index": self.chunk_index,
            "score": self.score,
        }


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #

class VectorStore:
    """Chroma-backed vector store with an MMR reranker on top."""

    def __init__(
        self,
        persist_dir: str,
        collection_name: str,
        embedder: Embedder,
    ) -> None:
        import chromadb
        from chromadb.config import Settings

        self.embedder = embedder
        self.collection_name = collection_name
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------ Write path -------------------------- #

    def upsert(self, chunks: Sequence[DocumentChunk]) -> int:
        """Embed and upsert chunks. Returns the number of chunks written.

        Uses deterministic IDs (``chunk.deterministic_id()``) so re-running the
        ingest is idempotent.
        """
        if not chunks:
            return 0
        ids = [c.deterministic_id() for c in chunks]
        texts = [c.text for c in chunks]
        metadatas = [c.as_metadata() for c in chunks]
        embeddings = self.embedder.embed_documents(texts)
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        return len(chunks)

    def count(self) -> int:
        """Number of vectors currently stored."""
        return int(self._collection.count())

    # ------------------------------ Read path --------------------------- #

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        fetch_k: int = 20,
        strategy: str = "mmr",
        mmr_lambda: float = 0.5,
    ) -> List[RetrievedChunk]:
        """Retrieve the top-`k` chunks for ``query``.

        Parameters
        ----------
        query:
            The user question (already contextualized by chat history if the
            caller wants that).
        top_k:
            Number of chunks to return.
        fetch_k:
            Number of candidates to fetch from Chroma before MMR reranking.
            Ignored when ``strategy="similarity"``.
        strategy:
            ``"similarity"`` for plain cosine top-`k`, ``"mmr"`` for
            Maximum Marginal Relevance.
        mmr_lambda:
            Trade-off between relevance and diversity for MMR
            (``1.0`` = pure relevance, ``0.0`` = pure diversity).
        """
        if not query or not query.strip():
            return []
        strategy = strategy.lower().strip()
        if strategy not in {"similarity", "mmr"}:
            raise ValueError("strategy must be 'similarity' or 'mmr'")

        query_vec = self.embedder.embed_query(query)
        n_results = max(top_k, fetch_k) if strategy == "mmr" else top_k

        raw = self._collection.query(
            query_embeddings=[query_vec],
            n_results=min(n_results, max(self.count(), 1)),
            include=["documents", "metadatas", "distances", "embeddings"],
        )

        docs = raw.get("documents", [[]])[0]
        metas = raw.get("metadatas", [[]])[0]
        dists = raw.get("distances", [[]])[0]
        embs = raw.get("embeddings", [[]])[0]

        if not docs:
            return []

        candidates: List[RetrievedChunk] = []
        for doc, meta, dist, _emb in zip(docs, metas, dists, embs):
            candidates.append(
                RetrievedChunk(
                    text=doc,
                    source=str(meta.get("source", "")),
                    page=int(meta.get("page", 0)),
                    chunk_index=int(meta.get("chunk_index", 0)),
                    score=1.0 - float(dist),  # cosine distance -> similarity
                )
            )

        if strategy == "similarity":
            return candidates[:top_k]

        # MMR reranking on the fetched candidates.
        selected_idx = _mmr(
            query_vec=np.asarray(query_vec, dtype=np.float32),
            candidate_vecs=np.asarray(embs, dtype=np.float32),
            top_k=top_k,
            mmr_lambda=mmr_lambda,
        )
        return [candidates[i] for i in selected_idx]


# --------------------------------------------------------------------------- #
# MMR                                                                         #
# --------------------------------------------------------------------------- #

def _mmr(
    query_vec: np.ndarray,
    candidate_vecs: np.ndarray,
    top_k: int,
    mmr_lambda: float,
) -> List[int]:
    """Return the indices of the ``top_k`` MMR-selected candidates.

    Uses cosine similarity throughout. Assumes vectors may or may not be
    normalized; we normalize on the fly so the math is scale-invariant.
    """
    if candidate_vecs.size == 0 or top_k <= 0:
        return []

    def _normalize(m: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(m, axis=-1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return m / norms

    q = _normalize(query_vec.reshape(1, -1))[0]
    C = _normalize(candidate_vecs)
    sim_to_query = C @ q  # (n,)

    selected: List[int] = []
    remaining = list(range(len(C)))
    top_k = min(top_k, len(remaining))

    while len(selected) < top_k and remaining:
        if not selected:
            best = int(np.argmax(sim_to_query[remaining]))
            chosen = remaining[best]
            selected.append(chosen)
            remaining.remove(chosen)
            continue

        # Redundancy: max cosine similarity to anything already selected.
        sel_mat = C[selected]
        sims_to_selected = C[remaining] @ sel_mat.T
        max_redundancy = sims_to_selected.max(axis=1)

        scores = mmr_lambda * sim_to_query[remaining] - (1.0 - mmr_lambda) * max_redundancy
        best_local = int(np.argmax(scores))
        chosen = remaining[best_local]
        selected.append(chosen)
        remaining.remove(chosen)

    return selected


# --------------------------------------------------------------------------- #
# Factory                                                                     #
# --------------------------------------------------------------------------- #

_store: Optional[VectorStore] = None


def get_vector_store(embedder: Optional[Embedder] = None) -> VectorStore:
    """Return a cached :class:`VectorStore` built from settings."""
    global _store
    if _store is None:
        from app.core.embeddings import get_embedder

        s = get_settings()
        _store = VectorStore(
            persist_dir=s.chroma_persist_dir,
            collection_name=s.chroma_collection,
            embedder=embedder or get_embedder(),
        )
    return _store


def reset_vector_store_cache() -> None:
    """Reset the cached vector store (used in tests / after re-ingest)."""
    global _store
    _store = None
