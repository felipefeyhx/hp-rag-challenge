# Quality Benchmark

Evaluates whether the chatbot answers correctly using the HP corpus.
18 hand-labeled questions in `benchmarks/questions.json`.

## Metrics

| Metric | What it measures | Type |
|---|---|---|
| **Retrieval hit-rate** | Did retrieval surface the right source page? (`±2` page tolerance) | Mechanical |
| **Refusal correctness** | Does the bot refuse off-topic questions? | Mechanical |
| **LLM-as-judge (1–5)** | Does the answer match the reference? Local LLM scores it. | Subjective |

RAGAS was considered but not adopted — the three metrics cover the same
axes (context precision, answer relevancy, faithfulness) without extra
dependencies.

## How to run

```bash
# Backend must be running at http://localhost:8000
pip install -r benchmarks/requirements.txt
python -m benchmarks.benchmark_quality
```

Flags: `--judge-model`, `--judge-base-url`, `--no-judge`, `--output`.

**Cost:** $0. Judge runs against the same local Ollama. Wall-clock: ~10–20
minutes on CPU (much less on GPU).

**Self-judging caveat:** using the same model as generator and judge tends
to inflate scores. Point `--judge-model` at a larger local model for a
stricter signal.

## Q&A set

| Category | Count |
|---|---:|
| factual | 8 |
| instruction | 5 |
| cross-doc | 1 |
| follow-up | 1 |
| off-topic (refusal expected) | 3 |

Each item: question, expected source doc + page(s), reference answer,
optional `should_refuse: true`.

## Latest results

Setup: `qwen2.5:7b-instruct` + `nomic-embed-text`, 541 chunks, same-model
judge, single run.

| Metric | Value |
|---|---|
| Retrieval hit-rate | **87.5% (14/16 in-scope)** |
| Refusal correctness | **100.0% (3/3)** |
| LLM-as-judge mean | 4.06 / 5 |
| LLM-as-judge median | 4.5 / 5 |
| Answer latency p50 | 2362 ms |
| Answer latency p95 | 7905 ms |

Full JSON at `benchmarks/results/<timestamp>.json`.

## Retrieval misses

`q03` ("model number") and `q08` ("CPU options for OMEN 17.3") — both
involve model numbers/SKUs where `nomic-embed-text` doesn't align well
against natural-language queries. A hybrid retriever (BM25 + dense) would
catch these; alternative embedding models like `bge-m3` (multilingual,
1024 dims) would likely reclaim most of the lost recall.

## Reading the metrics

| Metric | Trust for | Watch out for |
|---|---|---|
| Retrieval hit-rate | Retrieval regressions | Reference label errors |
| Refusal correctness | Off-topic guardrails | Small sample size |
| LLM-as-judge | Head-to-head version comparison | Self-judging inflates scores |

**Updating the Q&A set:** if a regression slips through with all metrics
green, add the failing question. Don't fix reference answers to hide bot
bugs — fix the bot instead.

## Files

| Path                              | Purpose                 |
| -----------------------------------| -------------------------|
| `benchmarks/questions.json`       | 18 labeled Q&A items    |
| `benchmarks/benchmark_quality.py` | Runner                  |
| `benchmarks/results/*.json`       | Timestamped run outputs |
