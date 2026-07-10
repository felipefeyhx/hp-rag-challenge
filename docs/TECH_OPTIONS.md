# Technology Options per Requirement

For each requirement from `challenge.txt`, this document lists the realistic
technology options, their trade-offs, the chosen pick, and why. This is a
**decision log** — the final choices live in `ARCHITECTURE.md`.

**Legend:** ⭐ = final choice · 🔒 = mandated by the challenge

---

## 1. Frontend — Graphical User Interface

> Challenge: *"Frontend with Graphical Interface, of your own choosing."*

| Option | Pros | Cons |
|---|---|---|
| ⭐ **Static HTML + vanilla JS + CSS** served by FastAPI | Zero build step, zero extra container, zero JS toolchain, same origin as the API (no CORS), simple to reason about | Not framework-y — you write the DOM code yourself |
| **Streamlit** | Pure Python, chat widgets built in, ~50 LOC to a full UI | Extra container, extra port, separate origin |
| **Chainlit** | Purpose-built for LLM chat, streaming out of the box | Newer, smaller community |
| **Gradio** | Also Python, chat interface in one line | Layout flexibility limited |
| **React + Vite / Next.js** | Full control, production-grade | More code, adds JS toolchain |
| **Vue / Nuxt** | Clean templating, same power as React | Same overhead |

**Chosen:** **Static HTML + vanilla JS** served by FastAPI at `/`. Keeps the
whole app in a single container with no build step, and shows the reviewer
that the API surface is complete on its own (the "frontend" is just a thin
client that hits `/health`, `/chat`, `/sessions`).

---

## 2. Backend framework 🔒

> Challenge: *"Backend must run with Python code using FastAPI."*

Fixed: **Python + FastAPI**. No alternatives.

**ASGI server:** Uvicorn with `--workers 4`.

---

## 3. Testing & coverage

> Challenge: *"Backend must have unit tests with at least 90% of coverage."*

| Option | Pros | Cons |
|---|---|---|
| ⭐ **pytest + pytest-cov** | Industry standard, great fixtures, plays with FastAPI's `TestClient` | None material |
| **unittest + coverage.py** | Stdlib, no extra deps | More verbose, weaker fixtures |
| **nose2** | Simple | Community has moved on |

**Helpful add-ons:** `pytest-asyncio`, `respx` (mock httpx), `pytest-mock`, `hypothesis` (property-based).

**Chosen:** **pytest + pytest-cov + pytest-asyncio + respx**. Coverage gate
enforced via `--cov-fail-under=90` in `pytest.ini`.

---

## 4. LLM provider

> Challenge: *"You can use either cloud or local (open source) models, of your own choosing."*

### Local (open-source) — chosen

| Option | Pros | Cons |
|---|---|---|
| ⭐ **Ollama** + `qwen2.5:7b-instruct` | Fully local, free, offline. OpenAI-compatible HTTP endpoint drops into the same SDK. Strong multilingual (incl. PT-BR). Correct citation behavior at 7B. | Needs ~5 GB disk, ~6 GB RAM. Slow on CPU (2–8 s/response). |
| **Ollama** + `llama3.1:8b` | Similar quality, well-known model | Same footprint |
| **Ollama** + `qwen2.5:3b-instruct` | Half the size, ~3× faster on CPU | Weaker at citation instructions |
| **Ollama** + `deepseek-r1:1.5b` | Tiny, has reasoning traces | Hallucinates citation page numbers (tested); reasoning trace bloats every response |
| **vLLM / TGI** self-hosted | Higher throughput | Requires GPU + more ops |
| **HF Transformers pipeline** | Simple in Python | Slow, memory-heavy |
| **llama-cpp-python** | Quantized CPU-friendly | Setup pain in Docker |

### Cloud (paid API) — considered, rejected

| Option | Pros | Cons |
|---|---|---|
| **OpenAI** (`gpt-4o-mini`) | Fast, cheap, high quality | Data leaves the machine, requires API key + billing |
| **Azure OpenAI** | Enterprise-friendly | Same data-flow concern, more setup |
| **Anthropic Claude** (Haiku/Sonnet) | Strong reasoning | External data, key required |
| **Google Gemini** | Free tier | External data, rate-limited |
| **Google Vertex AI OpenAI-compat** | OpenAI SDK works via `base_url` | External data, GCP auth |
| **AWS Bedrock `bedrock-mantle`** | OpenAI SDK works via `base_url` | External data, AWS auth |

**Chosen:** **Ollama serving `qwen2.5:7b-instruct`** over its OpenAI-compatible
endpoint (`http://localhost:11434/v1`).

Rationale:
- **Nothing leaves the machine.** Full data-flow control, no key management,
  no rate limits, no per-request cost. Matches the primary product
  constraint we set for this iteration.
- **OpenAI-compatible wire format.** The `openai` Python SDK works
  unchanged; we just set `base_url`. Same code runs against LM Studio, vLLM,
  LocalAI, or a cloud OpenAI-compat endpoint (Vertex AI, Bedrock) if the
  constraint ever relaxes — no code change.
- **Qwen 2.5 7B** was validated against the corpus on the "Quiet Mode"
  procedural question. Verbatim step content, correct page numbers,
  concise formatting — clearly usable for HP support Q&A at this size.
  Smaller models (tested `deepseek-r1:1.5b`) hallucinate citations.
- The `LLM` adapter in `app/core/llm.py` remains behind a small `Protocol`,
  so swapping to a different runner or a cloud endpoint later is a one-env-
  var change without touching application code.

---

## 5. Embeddings model

> Free choice; needed for RAG.

| Option | Dim | Pros | Cons |
|---|---:|---|---|
| ⭐ **Ollama `nomic-embed-text`** | 768 | Local, ~275 MB, strong MTEB scores, same OpenAI-compat surface as the chat LLM | English-primary; weaker on multilingual queries |
| **Ollama `bge-m3`** | 1024 | Local, multilingual (PT-BR strong), MTEB leader for its class | ~1.2 GB, larger download |
| **Ollama `mxbai-embed-large`** | 1024 | Local, high quality | ~700 MB, English-primary |
| **Ollama `all-minilm`** | 384 | Tiny (~50 MB), fastest | Weakest of the local set |
| **OpenAI `text-embedding-3-small`** | 1536 | High quality, low cost | External data flow (rejected under current constraint) |
| **Cohere embed-multilingual-v3** | 1024 | Multilingual, cloud | External data flow (rejected) |
| **sentence-transformers/all-MiniLM-L6-v2** | 384 | Tiny, fast on CPU | Weaker; would require a second embedding stack outside Ollama |
| **BAAI/bge-base-en-v1.5** | 768 | Strong quality | Same "second stack" issue |

**Chosen:** **Ollama `nomic-embed-text`** (768 dims).

Rationale:
- Same OpenAI-compatible endpoint as the chat LLM — one HTTP surface, one
  container to manage, one adapter (`TextEmbedder`) in the code.
- 768 dims is a comfortable sweet spot for HNSW indexing on the ~541-chunk
  corpus.
- Local, so nothing leaves the machine.
- Measured 87.5% retrieval hit-rate on the 18-question benchmark. Two of the
  16 in-scope questions missed — both involving model-number/SKU queries
  that `nomic-embed-text` doesn't align well against natural-language queries.
  Documented upgrade path: swap to `bge-m3` if the corpus grows or PT-BR
  becomes a first-class requirement.
- **Small footprint keeps the stack lean.** At ~275 MB, `nomic-embed-text` is
  roughly a quarter the size of the multilingual alternative `bge-m3` (~1.2 GB).
  The embedding model isn't baked into the Docker image — it's pulled at
  runtime by the `ollama-pull` init container — so a smaller model means a
  **faster first boot** (less to download) and a lighter footprint on disk and
  in memory. Because this deployment targets **English-only** questions, the
  weaker multilingual coverage is an acceptable trade-off; if PT-BR or other
  languages become a requirement, `bge-m3` is the documented upgrade.

---

## 6. Vector Database

> Challenge: *"Must use an open-source Vector Database for performing RAG."*

| Option | Pros | Cons |
|---|---|---|
| ⭐ **ChromaDB** (embedded, `PersistentClient`) | RAG-native API, automatic persistence, native metadata filters, no extra container, HNSW under the hood | Slower than Qdrant on stress tests (irrelevant at this scale) |
| **Qdrant** | Fast (Rust), pre-filtered ANN, hybrid dense + sparse, snapshot API | Adds a container; overkill for a ~300-chunk corpus |
| **FAISS** | Fastest ANN library, tiny footprint | No native persistence (manual `save_local` / `load_local`), no metadata filters, more glue code to test |
| **Weaviate** | Hybrid BM25 + dense in one call, GraphQL | Heavier, GraphQL learning tax, hybrid unused here |
| **Milvus** | Scales to billions of vectors | 5+ services in Compose, massive overkill |
| **PGVector** (Postgres extension) | Reuses relational DB, transactional | Only wins if we already had Postgres; slower ANN with filters |
| **LanceDB** | Embedded, columnar, growing fast | Smaller ecosystem, no meaningful advantage at this scale |

**Chosen:** **ChromaDB embedded** (`PersistentClient`, HNSW, cosine).

**Rationale (full argument in the conversation log):**
- ~300 chunks total → any DB works; performance is not the tiebreaker.
- Read-only after ingest → HNSW's lock-free concurrent reads shine.
- Zero extra services in Compose (embedded).
- Automatic persistence to a Docker volume — no manual save/load code.
- RAG-native API (`collection.add`, `collection.query`) keeps the adapter thin.
- The one thing Chroma lacks (MMR) is a tiny Python function on top of `fetch_k` results.

---

## 7. PDF parsing

> Challenge: *"Must use only the attached documents for building the Vector Database."*

| Option | Pros | Cons |
|---|---|---|
| ⭐ **pypdf** (via LangChain's `PyPDFLoader` wrapper) | Pure Python, MIT, tiny | Weak on tables/complex layout |
| **pdfplumber** | Good text + table extraction | Slower |
| **PyMuPDF (fitz)** | Fastest, high-quality text | AGPL license |
| **Unstructured.io** | Handles many formats, layout-aware | Heavy dependency tree |
| **Docling** (IBM) | Layout-aware, tables, OCR fallback | Newer, ~500 MB install; overkill for these text-based PDFs |

**Chosen:** **pypdf** (via `PyPDFLoader`). Both HP PDFs are text-based, so
a single loader is enough — no fallback needed.

**Why not Docling:** Docling was the layout-aware candidate we'd otherwise
reach for (tables, OCR fallback, markdown output). But as discussed in earlier
meetings, its processing is **slow** — the extra layout/OCR passes add real
time to every ingest. For two already-text-based HP PDFs that pypdf handles
cleanly, that ingest-time cost buys us nothing, so Docling isn't worth it here.
It stays on the table only as an upgrade path if a future corpus has complex
tables or scanned pages that pypdf can't read.

---

## 7b. Image handling in PDFs — deferred

> The HP PDFs contain diagrams. Text-only ingest drops that information. We
> defer image handling as an explicit upgrade path if the quality benchmark
> shows a gap.

### Strategies (kept for reference)

| # | Strategy | What it captures | Cost / effort | Quality on manuals |
|---|---|---|---|---|
| ⭐ 1 | **Ignore images** (chosen for v1) | Nothing beyond text | Zero | Sufficient because HP manuals name icons with text labels (e.g., "Power connection") |
| 2 | **OCR only** (Tesseract / EasyOCR / PaddleOCR) | Text inside images: labels, callouts, screenshots | Low, free, CPU | Good for labeled diagrams and screenshots |
| 3 | **VLM captioning at ingest** (vision LLM once per image) | Semantic description of each image | Medium (one-time API cost) | High — captures meaning of diagrams |
| 4 | **Layout-aware parser with built-in OCR** (Docling / PyMuPDF4LLM / Unstructured) | Text + OCR'd image content in a single markdown output | Medium (heavier install) | Good, less code to maintain |
| 5 | **Full multimodal RAG** (CLIP image embeddings + hybrid retrieval + vision LLM at answer time) | Images as first-class citizens in the index | High — dual embedding pipelines | Highest, most complex |
| 6 | **Page-as-image** (render each page as PNG, send to VLM on every query) | Whole-page vision at query time | High recurring cost | High but expensive at scale |

**Chosen for v1:** **Strategy 1 (ignore images)**. Text labels in the corpus
make the diagrams self-describing.

**Upgrade path if the benchmark demands it:** escalate to **strategy 2 + 3**
(OCR + VLM captions stored as text chunks). Impact: switch `PyPDFLoader` for
`PyMuPDF`, add an `image_ingest` step, extend `SourceChunk` schema with
`image_path` and `kind`, render images in the static frontend.

---

## 8. Chunking strategy

> Challenge: *"Must use a Chunking Strategy to index the document on the Vector Database."*

| Option | Pros | Cons |
|---|---|---|
| ⭐ **Recursive character splitter** (`RecursiveCharacterTextSplitter`) | Respects paragraph/sentence boundaries, no extra deps | Character-based |
| **Token-based splitter** (tiktoken) | Precise for LLM context budgets | Tokenizer tied to model |
| **Sentence splitter** (nltk/spacy) | Semantic boundaries | Extra deps + model downloads |
| **Semantic chunking** (embed sentences, group by similarity) | Best semantic cohesion | Slower, more complex |
| **Layout-aware chunking** (via Unstructured / Docling) | Preserves headings/tables | Requires layout-aware loader |
| **Sliding window** | Simple | Blind splits |

**Chosen parameters:**
- `chunk_size = 800` characters
- `chunk_overlap = 120` characters (~15%)
- Separators (in order): `"\n\n"`, `"\n"`, `". "`, `" "`, `""`

**Why 800 / 120:** small enough that 5 retrieved chunks fit into a ~2500-token
context without eating the budget; overlap preserves context across boundaries;
recursive fallback keeps splits near natural language boundaries. Larger
chunks (2000+) help extraction-style Q&A but hurt chat with citations.

**Why recursive character splitting for these docs:** the two HP documents are
structured manuals / informational PDFs — headings, numbered steps, short
paragraphs, and bullet lists. The recursive splitter's separator order
(`"\n\n"` → `"\n"` → `". "` → `" "`) maps directly onto that structure, so it
tends to break on paragraph and sentence boundaries rather than mid-thought.
For this kind of well-formatted reference material it gives clean, coherent
chunks without the extra cost of semantic or layout-aware chunking, which is
why it's the right fit over the fancier options above.

Both values are env-configurable (`CHUNK_SIZE`, `CHUNK_OVERLAP`).

---

## 9. Retrieval / Search strategy

> Challenge: *"Must select a Search Strategy for retrieval."*

| Option | Pros | Cons |
|---|---|---|
| **Cosine similarity top-k** (`search_type="similarity"`) | Simplest baseline | Returns redundant chunks |
| ⭐ **MMR (Maximum Marginal Relevance)** (`k=5`, `fetch_k=20`, `λ=0.5`) | Diversifies results, avoids near-duplicates in boilerplate-heavy docs | One extra hyperparameter (λ) |
| **Hybrid (BM25 + dense)** | Best recall on keyword queries | Needs a keyword index |
| **Multi-query retrieval** | LLM rewrites the query into N variants | Extra LLM calls |
| **HyDE** (hypothetical answer embedding) | Good on vague queries | Extra LLM call |
| **Cross-encoder reranker** (`bge-reranker-base`, `ms-marco-MiniLM`) | Big quality boost | Slower, extra model |
| **Contextual compression** | Cuts irrelevant chunks before LLM | Extra step |

**Chosen:** **MMR** with `k=5`, `fetch_k=20`, `λ=0.5`.

- The HP PDFs contain lots of repeated boilerplate (safety notices, headers).
  MMR keeps the returned context varied instead of returning five copies of
  the same warning box.
- Chroma doesn't ship MMR natively — we implement it in Python on the top
  `fetch_k` candidates. Small function, easy to test.
- Strategy selectable via `RETRIEVAL_STRATEGY` env var (`mmr` | `similarity`).

---

## 10. Chat history storage

> Challenge: *"Must have support for conversation with chat history."* + *"Must store the user chats and history in the backend."*

| Option | Pros | Cons |
|---|---|---|
| ⭐ **SQLite via SQLAlchemy** | Zero-config, single file, plenty for this scope, easy to swap to Postgres | Not for multi-node scale |
| **PostgreSQL via SQLAlchemy** | Production-grade, easy in Compose | Extra service |
| **MongoDB** | Document-oriented fits chat messages | Extra service, more setup |
| **Redis** (as JSON store) | Very fast | Not durable by default |
| **TinyDB** | Trivial | Not for concurrency |
| **JSON files on disk** | Trivial | Race conditions, no queries |

**Chosen:** **SQLite via SQLAlchemy** — zero-config, one file on a Docker
volume, and swapping to Postgres later is a single `DATABASE_URL` env-var
change because we go through SQLAlchemy.

---

## 11. RAG orchestration framework

> Not explicitly required; free choice.

| Option | Pros | Cons |
|---|---|---|
| ⭐ **LangChain `ConversationalRetrievalChain`** | Handles the "condense follow-up question with history" step natively; well-integrated with Chroma and OpenAI | Heavy, fast-moving API |
| **Custom thin code** | Full control, no bloat, ~150 LOC | You write the glue |
| **LlamaIndex** | RAG-first design, good abstractions | Learning curve |
| **Haystack (deepset)** | Pipeline-based, production-oriented | More opinionated |
| **DSPy** | Programmatic prompt optimization | Overkill here |

**Chosen:** **LangChain `ConversationalRetrievalChain`**, with a custom
server-side history layer (SQL-backed) instead of `ConversationBufferMemory`.
LangChain gives us the condense-question-with-history behavior for free,
which is the trickiest part of chat+RAG to get right.

---

## 12. Docker Compose 🔒

> Challenge: *"Must run through Docker Compose."*

Fixed: **Docker Compose**. Service layout is our choice.

**Chosen services:**
- `backend` (FastAPI + Uvicorn workers=4, runs ingest at startup, also serves the static frontend)

Single-service compose. No dedicated frontend, vectorstore, or db services —
the static frontend is served by FastAPI, Chroma is embedded, SQLite is a
file on a volume.

---

## 13. Load testing

> Challenge: *"Scalability must be ensured with load tests (how many requests per minute the solution supports)."*

| Option | Pros | Cons |
|---|---|---|
| ⭐ **Locust** | Python, scripts read like tests, live web UI | Python threading limits (fine here) |
| **k6** (Grafana) | JS scripts, fast, great reports | Adds a non-Python tool |
| **JMeter** | GUI, mature | Heavy, XML config |
| **Vegeta** | CLI, simple, fast (Go) | Less scriptable |
| **wrk / hey / bombardier** | Ultra-simple HTTP hammers | Not scenario-based |
| **Artillery** | YAML scenarios | Node-based |

**Chosen:** **Locust** — Python-native fits the stack, easy to script
realistic chat traffic and expose an RPM number.

---

## 14. Quality benchmark

> Challenge: *"Must benchmark the quality of the LLM responses to the user."*

| Option | Pros | Cons |
|---|---|---|
| ⭐ **RAGAS** | Purpose-built for RAG (faithfulness, answer relevancy, context precision/recall) | Requires an LLM for judging |
| ⭐ **Custom LLM-as-judge + retrieval hit-rate** | Transparent, no framework lock-in | We write it |
| **DeepEval** | pytest-style eval, many metrics | Extra dep |
| **TruLens** | Rich observability | Heavier |
| **BLEU / ROUGE / BERTScore** | Deterministic | Weak signal for RAG quality |
| **Promptfoo** | YAML test cases, CI-friendly | Node-based |

**Chosen:** **RAGAS + custom retrieval hit-rate + LLM-as-judge** on a
hand-labeled Q&A file drawn from the two HP PDFs.

---

## 15. Reranker — deferred

> Not needed for v1. Documented as an upgrade path if the benchmark shows retrieval quality is the bottleneck.

- `cross-encoder/ms-marco-MiniLM-L-6-v2` — light, CPU-runnable.
- `BAAI/bge-reranker-base` — top open-source.
- Cohere Rerank — cloud, paid.

---

## Final locked-in stack

| Layer                    | Choice                                                                       |
| ------------------------ | ---------------------------------------------------------------------------- |
| Frontend                 | Static HTML + vanilla JS + CSS, served by FastAPI at `/`                    |
| Backend framework        | FastAPI (🔒)                                                                  |
| ASGI server              | Uvicorn `--workers 4`                                                        |
| Testing                  | pytest + pytest-cov, ≥ 90% coverage                                          |
| LLM                      | Ollama `qwen2.5:7b-instruct` (local, OpenAI-compat endpoint)                 |
| Embeddings               | Ollama `nomic-embed-text` (local, 768 dims)                                  |
| Vector DB                | ChromaDB embedded (`PersistentClient`, HNSW, cosine)                         |
| PDF parser               | pypdf via `PyPDFLoader`                                                      |
| Image handling           | None (v1) — upgrade to OCR + VLM captions if benchmark demands               |
| Chunking                 | Recursive character, `size=800`, `overlap=120`                               |
| Retrieval                | MMR, `k=5`, `fetch_k=20`, `λ=0.5`                                            |
| RAG orchestration        | LangChain `ConversationalRetrievalChain` + SQL history                       |
| Chat history storage     | SQLite via SQLAlchemy                                                        |
| Deployment               | Docker Compose (🔒) — single service (backend serves API + static frontend) |
| Load testing             | Locust                                                                       |
| Quality benchmark        | RAGAS + retrieval hit-rate + LLM-as-judge                                    |
| Reranker                 | Deferred                                                                     |

This is the full stack. The design flows from these choices in
`ARCHITECTURE.md`.
