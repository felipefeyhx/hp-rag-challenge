"""Idempotent PDF ingest — runs at container start.

Reads every PDF under ``DOCUMENTS_DIR``, splits it, embeds it, and upserts
into ChromaDB. Safe to re-run: deterministic chunk IDs keep the collection
consistent.

Usage:
    python -m scripts.ingest
"""
from __future__ import annotations

import sys

from app.services.ingest_service import run_ingest


def main() -> int:
    print("=== HP RAG ingest ===")
    try:
        report = run_ingest()
    except FileNotFoundError as exc:
        print(f"[ingest] no documents to ingest: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[ingest] failed: {exc}", file=sys.stderr)
        return 2

    print(f"[ingest] documents:            {report.documents}")
    print(f"[ingest] pages:                {report.pages}")
    print(f"[ingest] chunks produced:      {report.chunks}")
    print(f"[ingest] already indexed:      {report.already_indexed}")
    print(f"[ingest] chunks upserted:      {report.added}")
    print(f"[ingest] collection size now:  {report.collection_size_after}")
    print("[ingest] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
