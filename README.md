# HP RAG Chatbot

A Retrieval-Augmented Generation chatbot that answers questions about HP
documents. FastAPI backend serves both the API and a static HTML/CSS/JS
frontend from the same origin, backed by ChromaDB for retrieval and a
**local Ollama runtime** for chat and embeddings. Nothing leaves the machine.
One `docker compose up` and you're chatting.

Full design in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Prerequisites

Only one thing: **Docker Desktop** (or Docker Engine + Compose v2).

Ollama, the LLM runtime, runs as a container inside the compose stack —
you don't need to install it on the host. Nothing leaves your machine.

**Hardware:** ~8 GB free RAM is comfortable for `qwen2.5:7b-instruct` on CPU.
GPU is optional; uncomment the `deploy` block in `docker-compose.yml` if you
have the NVIDIA Container Toolkit installed.

---

## Quick start (Docker Compose)

```bash
git clone <repo> && cd hp-rag-challenge
cp .env.example .env
docker compose up --build
```

That's it. Open the UI at **http://localhost:8000/**.
FastAPI docs at **http://localhost:8000/docs**.

> **Windows cmd users:** replace `cp` with `copy`:
> ```cmd
> copy .env.example .env
> docker compose up --build
> ```
> PowerShell users can use either `cp` (aliased) or `Copy-Item`.

> **Linux users:** if `docker` requires `sudo`, either prefix commands with
> `sudo` or add your user to the `docker` group once:
> ```bash
> sudo usermod -aG docker $USER
> # log out and back in for the group change to take effect
> ```

> **First boot takes ~5–10 minutes** because Compose pulls two Ollama models
> (~5 GB total: `qwen2.5:7b-instruct` + `nomic-embed-text`) into a named
> Docker volume, then ingests the two HP PDFs into Chroma (~30 s once models
> are ready). Subsequent starts finish in ~15 s — models and the vector
> store are cached.

---

## Repository layout

See [`docs/ARCHITECTURE.md §15`](docs/ARCHITECTURE.md).

---

## Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | Static frontend (`index.html`) |
| GET | `/static/*` | JS / CSS assets |
| GET | `/health` | Liveness + model name + indexed chunk count |
| POST | `/chat` | Send a message. Body: `{message, session_id?}` |
| GET | `/sessions` | List chat sessions |
| POST | `/sessions` | Create a new session |
| GET | `/sessions/{id}` | Get a session with its messages |
| DELETE | `/sessions/{id}` | Delete a session |

---

## Configuration

See [`.env.example`](.env.example) for the full list. Highlights:

| Variable | Default | Purpose |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:11434/v1` | Ollama URL. Compose overrides to `http://ollama:11434/v1`. |
| `LLM_API_KEY` | `ollama` | Placeholder; never sent externally |
| `EMBEDDING_BASE_URL` | `http://localhost:11434/v1` | Embeddings endpoint |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `120` | Chunking |
| `RETRIEVAL_TOP_K` | `5` | Number of chunks in the prompt |
| `RETRIEVAL_STRATEGY` | `mmr` | `mmr` or `similarity` |
| `HISTORY_WINDOW` | `6` | Turns of history sent to the LLM |
| `DATABASE_URL` | `sqlite:///.../chatbot.db` | Any SQLAlchemy URL |

**Models are fixed** — `qwen2.5:7b-instruct` (chat) and `nomic-embed-text`
(embeddings) are hardcoded in `backend/app/config.py`. Changing them requires
editing that file and the `ollama-pull` step in `docker-compose.yml`.

---

## Tuning performance with Docker

The default `qwen2.5:7b-instruct` runs on **CPU**, so the first token can take
a while to appear (the reply bubble stays empty until generation starts).
Enable GPU below for the biggest single speedup — no code changes.

### Enable GPU acceleration (biggest speedup)

If you have an NVIDIA GPU and the NVIDIA Container Toolkit installed, uncomment
the `deploy` block under the `ollama` service in `docker-compose.yml`:

```yaml
  ollama:
    image: ollama/ollama:latest
    # ...
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

Then recreate the container:

```bash
docker compose up -d
docker compose logs -f backend   # wait until healthy
```

> **Note:** the first message in a session is always slowest — Ollama loads the
> model into memory on first use. Later messages are faster because it's warm.

Tuning knob for GPU concurrency: `OLLAMA_NUM_PARALLEL` in `docker-compose.yml`
(default `2`). Each parallel slot allocates its own KV cache — raise on
larger-VRAM GPUs, lower if you OOM.

---

## Tests, load tests, quality benchmark

- **Unit tests:** `cd backend && pytest` — **122 tests, 99% coverage** (≥ 90% required).
- **Load test:** `locust -f loadtests/locustfile.py --host http://localhost:8000 --headless -u 10 -r 2 -t 2m`
  — baseline on RTX 2000 Ada: **~47 req/min sustained at 0% failure rate** (see [`docs/LOAD_TESTING.md`](docs/LOAD_TESTING.md)).
- **Quality benchmark:** `python -m benchmarks.benchmark_quality`
  — 18 curated questions. Uses the local LLM as judge by default (override with `--judge-model` / `JUDGE_MODEL`). See [`docs/BENCHMARK.md`](docs/BENCHMARK.md).

---

## Requirement traceability

See [`docs/ARCHITECTURE.md §16`](docs/ARCHITECTURE.md) for the mapping
from each bullet in `challenge.txt` to the file that satisfies it.
