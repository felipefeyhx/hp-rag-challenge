# Load Testing

Locust-based load tests against a live backend. Scenarios in
`loadtests/locustfile.py` simulate realistic chat traffic.

## TL;DR

Three runs on the same host (RTX 2000 Ada, 8 GB VRAM). All numbers are
`POST /chat`; other endpoints are sub-30 ms and irrelevant.

| Config | Users | Chats / 2 min | RPM | p50 | p95 | Failures |
|---|---:|---:|---:|---:|---:|---:|
| CPU, serial | 5 | 32 | 28 | 14 s | 26 s | 0% |
| CPU, `NP=2` | 10 | 2 | 14 | 93 s | 93 s | 0% |
| **GPU, `NP=2`** | **10** | **49** | **47** | **19 s** | **24 s** | **0%** |

Single-user (uncontended): **CPU 14 s → GPU 1 s** (14× faster).

## How to run

```bash
pip install -r loadtests/requirements.txt

locust -f loadtests/locustfile.py --host http://localhost:8000 \
       --headless -u 10 -r 2 -t 2m --csv loadtests/reports/run
```

- `-u N` concurrent users · `-r N` ramp/s · `-t Nm` duration

## Scenario weights

| Task | Weight | Endpoint |
|---|---:|---|
| Send chat | 15 | `POST /chat` |
| View history | 3 | `GET /sessions/{id}` |
| List sessions | 2 | `GET /sessions` |
| Health check | 1 | `GET /health` |
| Create session | once | `POST /sessions` |

Users wait 1–3 s between tasks. Questions rotate through 24 realistic
HP-support prompts.

Full per-endpoint numbers, pre-run GPU verification, and raw Locust output
are preserved in [`loadtests/reports/RUN_LOGS.md`](../loadtests/reports/RUN_LOGS.md).
CSVs from every run are at `loadtests/reports/*_stats.csv`.

## What was tested

Nothing is mocked — every request hits the live backend end-to-end:

- **API endpoints:** `POST /chat`, `POST /sessions`, `GET /sessions`,
  `GET /sessions/{id}`, `GET /health`.
- **RAG pipeline per chat:** query embedding, Chroma retrieval, MMR rerank,
  prompt build, LLM inference (streaming), assistant reply.
- **Database:** SELECT session, INSERT user message, SELECT history (last 6
  turns), INSERT assistant message + sources JSON, occasional UPDATE for
  session title.
- **Concurrency:** 5 and 10 simultaneous users with realistic 1–3 s think
  time between actions.
- **Inference paths:** CPU vs GPU, serial (`NP=1`) vs parallel (`NP=2`).
- **Metrics captured:** total requests, failure rate, throughput (RPM),
  latency p50 / p95 / max per endpoint, uncontended single-user latency.

## What the ceiling is

Local LLM inference — 99% of chat-request time. Everything else is
milliseconds: retrieval + MMR (< 5 ms), prompt build (< 1 ms), DB write
(< 10 ms), Uvicorn (~200 concurrent-request headroom), SQLite (150+
chats/s capacity vs 0.4 measured).

## What could be escalated

Ordered from cheapest to most involved.

| # | Change | Expected gain | Effort | When to do it |
|---:|---|---|---|---|
| 1 | Raise `OLLAMA_NUM_PARALLEL` (`2 → 3`, `4`, …) | ~1.5× per slot | Env var | You have spare VRAM |
| 2 | Smaller chat model (`qwen2.5:3b`) | ~3× tok/s | Edit `config.py` + compose | Latency-critical, quality trade-off acceptable |
| 3 | Bigger GPU (A10 / RTX 4090 / A100) | 5–10× tok/s + higher `NP` | Hardware swap, no app change | Single biggest lever |
| 4 | Swap Ollama → vLLM or TGI | 5–20× at high concurrency | One env var (`LLM_BASE_URL`) | Many users, batching helps |
| 5 | Multiple Ollama replicas + LB | Linear per replica | Compose + LB config | Multi-GPU host |
| 6 | Horizontal backend replicas | Linear per replica | Compose scale + LB | API-side load rises |
| 7 | Swap SQLite → Postgres | Removes DB write ceiling | Change `DATABASE_URL` | Writes > 300/s (not close) |
| 8 | Swap Chroma → Qdrant / pgvector | Networked, sharded index | ~50 LOC adapter | Corpus > millions of chunks |

All items 1–7 are config/env changes — no code rewrites — because every
I/O boundary is behind a `Protocol`. Item 8 needs a small adapter swap.

## Verifying that failures ARE detected

The "0% failure" number in every real run is a real zero — not a false
zero from a broken test. Proof: `loadtests/failure_probe.py` fires only
requests the API is guaranteed to reject.

```bash
locust -f loadtests/failure_probe.py --host http://localhost:8000 \
       --headless -u 5 -r 5 -t 30s --csv loadtests/reports/failure_probe
```

Result (30 s, 5 users):

| Endpoint | # reqs | # fails | Failure rate | Error |
|---|---:|---:|---:|---|
| `POST /chat (empty)` | 411 | 411 | **100%** | 422 Unprocessable Entity (Pydantic `min_length=1`) |
| `GET /sessions/{missing}` | 200 | 200 | **100%** | 404 Not Found |
| **Aggregated** | **611** | **611** | **100%** | — |

Locust records every HTTP >= 400 as a failure by default. The counter works.

## Stress test — 50 users, no think time

Not a realistic user model — a ceiling probe. `loadtests/stress_test.py`
fires requests back-to-back with `wait_time = between(0, 0)`.

```bash
locust -f loadtests/stress_test.py --host http://localhost:8000 \
       --headless -u 50 -r 25 -t 1m --csv loadtests/reports/stress
```

Result (1 min, 50 users, GPU + `NP=2`):

| Endpoint | # reqs | Fails | p50 | p95 | Max |
|---|---:|---:|---:|---:|---:|
| `POST /chat` | 37 | 2 (5.41%) | 31 s | 57 s | 58 s |
| `GET /health` | 7 | 0 (0%) | 11 ms | 1.3 s | 1.3 s |

**Degradation is graceful, not catastrophic:**

- **Cheap endpoints stay fast.** `/health` p50 = 11 ms while `/chat`
  queued at 30+ s. Uvicorn didn't tip over — only the LLM queue backed up.
- **First failures are LLM-related.** The 2 errors were HTTP 500s from
  the chat pipeline (Ollama slot exhaustion / stream timeout). No 502s,
  no 503s, no connection resets.
- **95.5% still succeeded** with real answers and real sources — no
  corruption, no half-persisted messages.

The pattern matches a 2-slot bottleneck: 50 clients queueing behind 2
Ollama parallel slots, each ~15 s per generation. If Uvicorn or the app
were saturated, `/health` would have slowed down too.

## Files

| Path | Purpose |
|---|---|
| `loadtests/locustfile.py` | Main scenarios + weights (realistic traffic) |
| `loadtests/failure_probe.py` | Deliberately failing requests (verifies Locust catches errors) |
| `loadtests/stress_test.py` | Zero think-time ceiling probe (50 users hammering) |
| `loadtests/questions.txt` | 24 rotating questions |
| `loadtests/reports/RUN_LOGS.md` | Detailed per-run outputs |
| `loadtests/reports/*.csv` | Raw stats from every run |
| `loadtests/reports/single_user_probe.py` | Uncontended latency probe |
