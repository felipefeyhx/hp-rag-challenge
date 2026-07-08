# HP RAG Chatbot — Architecture

Design of the current system.

## 1. One-liner

FastAPI backend answering questions about HP PDFs using RAG, backed by
embedded ChromaDB and a **local LLM via Ollama**. Same FastAPI process
serves the vanilla-JS frontend, so the whole app is a single Docker
Compose service. **No data leaves the machine.**

## 2. Stack

| Layer | Choice |
|---|---|
| Frontend | Static HTML + vanilla JS, served by FastAPI at `/` |
| Backend | FastAPI + Uvicorn (`--workers 4`) 🔒 |
| PDF parser | `pypdf` via `PyPDFLoader` |
| Chunking | `RecursiveCharacterTextSplitter`, size 800, overlap 120 |
| Embeddings | Ollama `nomic-embed-text` (768 dims, local) |
| Vector store | ChromaDB embedded (HNSW, cosine) |
| Retrieval | MMR, `k=5`, `fetch_k=20`, `λ=0.5` |
| LLM | Ollama `qwen2.5:7b-instruct` (OpenAI-compat wire protocol) |
| RAG loop | Custom condense → retrieve → answer in `rag_service.py` |
| Chat history | SQLite via SQLAlchemy |
| Deployment | Docker Compose 🔒 |

🔒 = mandated by the challenge.

## 3. Component diagram

```
             Browser (fetch → same origin, no CORS)
                              │
                              ▼
                ┌──────────────────────────────┐
                │       FastAPI + Uvicorn      │  :8000
                │  /  → index.html             │
                │  /chat  /sessions  /health   │
                └───┬───────────┬───────────┬──┘
                    │           │           │
                    ▼           ▼           ▼
       ┌────────────────┐  ┌──────────┐  ┌──────────────────┐
       │  RAG service   │  │  SQL ORM │  │   LLM adapter    │
       └───┬────────┬───┘  └────┬─────┘  └────────┬─────────┘
           ▼        ▼           ▼                 ▼
     ┌────────┐ ┌────────┐ ┌────────────┐  ┌─────────────┐
     │ Chroma │ │Embedder│ │  SQLite    │  │   Ollama    │
     │(HNSW)  │ │(Ollama)│ │chatbot.db  │  │  :11434     │
     └────────┘ └────────┘ └────────────┘  └─────────────┘

    Ingest (runs once at container start, idempotent):
    PDFs → chunks → embeddings → Chroma.upsert
```

## 4. Components

**Frontend** — Three static files in `backend/app/static/`. Talks to the
API on relative URLs (same origin, no CORS). Persists session id in
`localStorage`. Streaming replies via SSE; sources rendered as collapsible
citations under each answer.

**Backend** — Async FastAPI. Layout:
```
app/
  api/       # routes: chat, sessions, health
  core/      # RAG blocks: pdf_loader, chunking, embeddings, vectorstore, llm, prompt
  services/  # orchestration: chat, ingest, rag
  db/        # SQLAlchemy models + engine
  schemas/   # Pydantic types
```
4 Uvicorn workers, each with its own event loop and ~2 MB Chroma index copy.

**Vector store** — ChromaDB via `PersistentClient`, HNSW, cosine. Persists
to `/app/data/chroma` (Docker volume). Deterministic chunk IDs
(`{source}::p{page}::c{idx}`) make ingest idempotent.

**LLM adapter** — `ChatLLM` implements a small `LLM` protocol. Uses the
`openai` Python SDK as an HTTP client for the OpenAI-compatible wire
format Ollama exposes at `http://ollama:11434/v1`. Not calling OpenAI's
servers — cloud endpoints are out of scope. Same code works against LM
Studio, vLLM, llama.cpp `llama-server`, LocalAI — one env var swap.

**Embeddings** — `TextEmbedder` points at the same Ollama endpoint,
running `nomic-embed-text`. Changing the model requires re-ingesting
(dims differ; the collection is incompatible).

**Chat persistence** — Two SQLite tables:
- `sessions(id, title, created_at, updated_at)`
- `messages(id, session_id, role, content, sources, created_at)`


## 5. Data flows

**Ingest (at container start):**
```
PDFs → PyPDFLoader → clean text → RecursiveCharacterTextSplitter
     → Ollama embeddings → Chroma.upsert (deterministic IDs, idempotent)
```

**Query (per user message):**
```
POST /chat
  → load/create session, save user msg to DB
  → load last 6 turns from DB
  → condense: if history exists, LLM rewrites into standalone question
  → embed condensed question
  → Chroma.query(fetch_k=20) → MMR rerank → top 5 chunks
  → build prompt (system + context + history + user)
  → LLM.chat_stream → SSE tokens to browser
  → persist assistant msg + sources → DB
```

Wall-clock: ~99% of latency is the LLM call. Retrieval + DB = <10 ms.

## 6. Chunking

Recursive character splitter, separators in priority order:
`\n\n` → `\n` → `". "` → `" "` → `""`. Size 800, overlap 120.

- 800 chars: precise enough to keep 5 chunks in a ~2500-token context.
- 120 overlap: preserves context across boundaries.
- Recursive fallback: cuts near natural boundaries when possible.

## 7. Retrieval — MMR

```
score(c) = λ · sim(c, q) − (1 − λ) · maxₛ∈S sim(c, s)
```

Fetch 20, rerank to 5. `λ=0.5` balances relevance vs diversity. Critical
for HP docs (lots of repeated safety-notice boilerplate). Chroma has no
native MMR; ~40 lines of NumPy on top of `fetch_k` results.

Selectable via `RETRIEVAL_STRATEGY` (`mmr` or `similarity`).

## 8. Chat history

- SQL is the source of truth (durable, survives restarts).
- Every request loads the last `HISTORY_WINDOW=6` user+assistant pairs.
- Two uses: condense the current turn into a standalone question, and
  give the LLM short-term memory in the final prompt.
- DB keeps everything; `HISTORY_WINDOW` only bounds what's sent to the LLM.

## 9. Configuration (`.env`)

| Env var | Default | Purpose |
|---|---|---|
| `LLM_BASE_URL` | `http://ollama:11434/v1` | LLM endpoint |
| `EMBEDDING_BASE_URL` | `http://ollama:11434/v1` | Embeddings endpoint |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `120` | Splitter |
| `RETRIEVAL_TOP_K` / `FETCH_K` / `MMR_LAMBDA` | `5` / `20` / `0.5` | Retrieval |
| `HISTORY_WINDOW` | `6` | Turns injected in prompt |
| `DATABASE_URL` | `sqlite:////app/data/db/chatbot.db` | SQLAlchemy URL |
| `CORS_ORIGINS` | `*` | Comma-separated or `*` |

**Models are fixed in `backend/app/config.py`** (`qwen2.5:7b-instruct` +
`nomic-embed-text`) and NOT overridable via env — changing the embedding
model would invalidate the vector store.

Full env-var list in `.env.example`.

## 10. Deployment

Two Compose services:
- **`ollama`** — pulls model weights at boot, serves inference on port 11434 (internal only).
- **`backend`** — FastAPI + static frontend, runs ingest at start, uvicorn workers = 4.

```
docker compose up --build
# UI:       http://localhost:8000/
# API docs: http://localhost:8000/docs
```

Cold boot: ~2–5 min (image build + model pull + first ingest). Warm boot:
~15 s. GPU auto-detected if NVIDIA Container Toolkit is on the host.

## 11. Scalability

**Bottleneck:** local LLM inference (~99% of request latency). Everything
else is milliseconds: Chroma < 5 ms, SQLite reads ~1–3 ms, SQLite writes
5–15 ms.

**Concurrent-user ceilings, in order they hit:**
1. LLM inference throughput (single-threaded per generation unless
   `OLLAMA_NUM_PARALLEL` is set).
2. SQLite write throughput (~400 writes/s — switch to Postgres if you hit it).
3. Backend CPU cores (raise `--workers`, then horizontal replicas).

**Measured today:**
- Sustained **47 req/min at 0% failure rate** (GPU, `NUM_PARALLEL=2`, 10 users).
- Uncontended latency: **~1 s on GPU**, ~14 s on CPU.

Details in `docs/LOAD_TESTING.md` and `loadtests/reports/RUN_LOGS.md`.

## 12. Testing

- `pytest` + `pytest-cov`, gate at ≥ 90% (`--cov-fail-under=90` in `pytest.ini`).
- Adapters (LLM, Embedder, VectorStore) are `Protocol`-based → tests inject
  fakes, no network in CI. Chroma runs `EphemeralClient` in tests; SQLite
  uses `:memory:`.
- **Latest: 122 tests, 99.12% coverage.**

## 13. Load testing & benchmark

- **Load** (`docs/LOAD_TESTING.md`): Locust, 4 recorded runs, ~47 req/min
  sustained, 0% failures.
- **Quality** (`docs/BENCHMARK.md`): 18 hand-labeled Q&A. Retrieval 87.5%,
  refusal 100%, LLM-as-judge 4.06 / 5.

## 14. Repository layout

```
hp-rag-challenge/
├── docker-compose.yml
├── .env.example
├── README.md
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── app/
│   │   ├── api/          # routes
│   │   ├── core/         # RAG building blocks
│   │   ├── db/           # SQLAlchemy
│   │   ├── schemas/
│   │   ├── services/
│   │   ├── config.py
│   │   ├── main.py
│   │   └── static/       # frontend
│   ├── data/
│   │   ├── documents/    # source PDFs (read-only bind mount)
│   │   ├── chroma/       # vector store (volume)
│   │   └── db/           # sqlite (volume)
│   ├── scripts/ingest.py
│   └── tests/
├── benchmarks/
├── loadtests/
└── docs/
```

## 15. Requirement traceability

| Challenge requirement | Where | Measured result |
|---|---|---|
| Frontend with GUI | `backend/app/static/` | Served at `http://localhost:8000/` |
| Backend in Python + FastAPI | `backend/app/` | 4 uvicorn workers, healthchecked |
| Unit tests ≥ 90% coverage | `backend/tests/` | **122 tests, 99.12%** |
| Cloud or local LLM | `app/core/llm.py` | Local Ollama, `qwen2.5:7b-instruct` |
| Open-source vector DB | ChromaDB | 541 chunks, HNSW, cosine |
| Only attached PDFs | `backend/data/documents/` | Ingest limited to folder |
| Chunking strategy | `app/core/chunking.py` + §6 | Recursive char, 800/120 |
| Search strategy | `app/core/vectorstore.py` + §7 | MMR k=5 |
| Conversation with history | `services/chat_service.py` + §8 | Condense + last 6 turns |
| Chats stored server-side | `app/db/` | SQLite on Docker volume |
| Runs via Docker Compose | `docker-compose.yml` | `docker compose up --build` |
| Load tests + RPM | `docs/LOAD_TESTING.md` | **47 req/min, 0% failures (GPU)** |
| Quality benchmark | `docs/BENCHMARK.md` | **87.5% retrieval, 100% refusal, 4.06/5** |
| Easy deployment | `.env.example` + one command | Single `.env` + `docker compose up` |
| Code + docs in English | Repo-wide | Verified |

## 16. Deferred upgrade paths

- **Image handling** — OCR + VLM captioning at ingest (swap PyPDFLoader for
  PyMuPDF, add an `image_ingest` step, extend `SourceChunk` with `image_path`).
- **Hybrid retrieval** — BM25 + dense (swap Chroma for Qdrant, or add `rank_bm25`).
- **Cross-encoder reranker** — `bge-reranker-base` between `fetch_k` and `top_k`.
- **Postgres** — flip `DATABASE_URL` when SQLite writes saturate.
- **Multi-replica backend** — each replica loads its own Chroma copy; move
  to networked vector DB above ~300K vectors.
