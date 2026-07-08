"""Quality benchmark for the HP RAG Chatbot.

Runs every item in ``benchmarks/questions.json`` against a live backend, then
computes three metrics per item:

1. **Retrieval hit-rate** — was the labeled source document retrieved, on a
   page within ``page_tolerance`` of any expected page? (mechanical, no LLM)
2. **Refusal correctness** — for items marked ``should_refuse: true``, does
   the answer contain a refusal phrase? (mechanical)
3. **LLM-as-judge** — a local LLM (same Ollama endpoint as the chat model
   by default) scores the answer against the reference answer on a 1-5
   rubric. (subjective, uses an eval LLM)

Aggregated metrics are printed as a table and written to
``benchmarks/results/<timestamp>.json`` for evidence.

Usage:
    # Backend must be running at http://localhost:8000
    cp .env.example .env        # if not already
    python -m benchmarks.benchmark_quality
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx


ROOT = Path(__file__).parent
QUESTIONS_FILE = ROOT / "questions.json"
RESULTS_DIR = ROOT / "results"


# --------------------------------------------------------------------------- #
# Config                                                                      #
# --------------------------------------------------------------------------- #

def _load_env() -> None:
    """Load .env from the repo root so LLM_BASE_URL/JUDGE_* are available."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    candidates = [
        ROOT.parent / ".env",           # hp-rag-challenge/.env
        Path(".env"),                    # cwd
    ]
    for c in candidates:
        if c.exists():
            load_dotenv(str(c), override=False)
            return


_load_env()


# --------------------------------------------------------------------------- #
# Data classes                                                                #
# --------------------------------------------------------------------------- #

@dataclass
class BenchmarkItem:
    id: str
    category: str
    question: str
    reference_answer: str
    expected_source: Optional[str] = None
    expected_pages: List[int] = field(default_factory=list)
    should_refuse: bool = False
    note: str = ""


@dataclass
class ChatResult:
    answer: str
    sources: List[Dict[str, Any]]
    latency_ms: float
    session_id: str


@dataclass
class ItemMetrics:
    id: str
    category: str
    question: str
    latency_ms: float
    retrieval_hit: Optional[bool]       # None for off-topic / should_refuse
    refusal_expected: bool
    refusal_seen: Optional[bool]
    judge_score: Optional[int]          # 1-5, or None if judge was skipped
    judge_reason: str
    answer_preview: str


# --------------------------------------------------------------------------- #
# Loading                                                                     #
# --------------------------------------------------------------------------- #

def load_items(path: Path = QUESTIONS_FILE) -> tuple[list[BenchmarkItem], int]:
    """Return (items, page_tolerance)."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    items = []
    for it in data["items"]:
        items.append(BenchmarkItem(
            id=it["id"],
            category=it.get("category", "unknown"),
            question=it["question"],
            reference_answer=it.get("reference_answer", ""),
            expected_source=it.get("expected_source"),
            expected_pages=list(it.get("expected_pages", [])),
            should_refuse=bool(it.get("should_refuse", False)),
            note=it.get("note", ""),
        ))
    return items, int(data.get("page_tolerance", 2))


# --------------------------------------------------------------------------- #
# Backend calls                                                               #
# --------------------------------------------------------------------------- #

def call_chat(client: httpx.Client, question: str) -> ChatResult:
    t0 = time.perf_counter()
    r = client.post("/chat", json={"message": question})
    r.raise_for_status()
    latency = (time.perf_counter() - t0) * 1000
    body = r.json()
    return ChatResult(
        answer=body.get("answer", ""),
        sources=body.get("sources", []),
        latency_ms=latency,
        session_id=body.get("session_id", ""),
    )


# --------------------------------------------------------------------------- #
# Metrics                                                                     #
# --------------------------------------------------------------------------- #

REFUSAL_PATTERNS = [
    "not enough information",
    "don't have enough",
    "do not have enough",
    "cannot find",
    "can't find",
    "does not contain",
    "no information",
    "unable to answer",
]


def check_retrieval_hit(item: BenchmarkItem, sources: list[dict], tolerance: int) -> Optional[bool]:
    """Return True/False for hit, or None when there is no expected source (off-topic)."""
    if item.should_refuse or not item.expected_source or not item.expected_pages:
        return None
    for s in sources:
        if s.get("source") != item.expected_source:
            continue
        page = int(s.get("page", -1))
        for target in item.expected_pages:
            if abs(page - target) <= tolerance:
                return True
    return False


def check_refusal(item: BenchmarkItem, answer: str) -> Optional[bool]:
    """True if the answer contains a refusal phrase; None if refusal wasn't expected."""
    if not item.should_refuse:
        return None
    lower = answer.lower()
    return any(p in lower for p in REFUSAL_PATTERNS)


# --------------------------------------------------------------------------- #
# LLM-as-judge                                                                #
# --------------------------------------------------------------------------- #

JUDGE_SYSTEM = """\
You are an evaluator for a support chatbot that answers questions using two HP
product documents. Score the actual answer against the reference answer.

Rubric (integer 1 to 5):
5 = Fully correct and complete for the question.
4 = Correct on the main point but incomplete.
3 = Partially correct or contains minor errors.
2 = Largely incorrect or missing the point.
1 = Wrong, hallucinated, or off-topic.

If the question was expected to be refused and the answer is a polite refusal,
score 5. If it was refused but should have answered, score based on the miss.

Respond with a single JSON object of the form:
{"score": <integer 1-5>, "reason": "<one short sentence>"}
"""


def _make_judge_client(api_key: str, base_url: str):
    """Build an OpenAI-compatible client pointed at a local runner.

    ``base_url`` decides where the requests go; ``api_key`` is only used
    to satisfy the SDK contract when talking to Ollama or similar.
    """
    from openai import OpenAI
    return OpenAI(api_key=api_key or "unused", base_url=base_url)


def judge_answer(
    openai_client,
    model: str,
    item: BenchmarkItem,
    answer: str,
) -> tuple[Optional[int], str]:
    """Call the judge LLM. Returns (score, reason)."""
    user = (
        f"Question: {item.question}\n\n"
        f"Reference answer: {item.reference_answer}\n\n"
        f"Actual answer: {answer}\n\n"
        f"Expected to refuse: {item.should_refuse}\n"
    )
    try:
        resp = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
    except Exception as exc:
        return None, f"judge error: {exc}"

    raw = resp.choices[0].message.content or "{}"
    try:
        parsed = json.loads(raw)
        score = int(parsed.get("score", 0))
        reason = str(parsed.get("reason", "")).strip()
        if not (1 <= score <= 5):
            return None, f"invalid score: {score}"
        return score, reason
    except (ValueError, json.JSONDecodeError):
        return None, f"unparseable judge output: {raw[:120]}"


# --------------------------------------------------------------------------- #
# Runner                                                                      #
# --------------------------------------------------------------------------- #

def run_benchmark(
    *,
    host: str,
    questions_path: Path,
    judge_model: str,
    judge_base_url: str,
    judge_api_key: str,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    items, tolerance = load_items(questions_path)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with httpx.Client(base_url=host, timeout=120.0) as api:
        # Sanity: backend is up
        h = api.get("/health")
        h.raise_for_status()
        health = h.json()
        print(f"[bench] backend: {health}")

    judge = _make_judge_client(judge_api_key, judge_base_url) if judge_base_url else None

    per_item: List[ItemMetrics] = []
    print()
    print(f"{'id':<5} {'cat':<12} {'ret':<4} {'refuse':<6} {'judge':<5} {'ms':>7}  question")
    print("-" * 100)

    with httpx.Client(base_url=host, timeout=120.0) as api:
        for it in items:
            try:
                result = call_chat(api, it.question)
            except Exception as exc:
                print(f"{it.id:<5} ERROR: {exc}")
                per_item.append(ItemMetrics(
                    id=it.id, category=it.category, question=it.question,
                    latency_ms=0.0, retrieval_hit=None,
                    refusal_expected=it.should_refuse, refusal_seen=None,
                    judge_score=None, judge_reason=f"api error: {exc}",
                    answer_preview="",
                ))
                continue

            ret_hit = check_retrieval_hit(it, result.sources, tolerance)
            refusal_seen = check_refusal(it, result.answer)

            judge_score, judge_reason = (None, "judge disabled")
            if judge is not None:
                judge_score, judge_reason = judge_answer(judge, judge_model, it, result.answer)

            per_item.append(ItemMetrics(
                id=it.id,
                category=it.category,
                question=it.question,
                latency_ms=result.latency_ms,
                retrieval_hit=ret_hit,
                refusal_expected=it.should_refuse,
                refusal_seen=refusal_seen,
                judge_score=judge_score,
                judge_reason=judge_reason,
                answer_preview=result.answer[:200].replace("\n", " "),
            ))

            ret_str = "-" if ret_hit is None else ("hit" if ret_hit else "miss")
            ref_str = "-" if refusal_seen is None else ("ok" if refusal_seen else "no")
            js_str = "-" if judge_score is None else str(judge_score)
            print(f"{it.id:<5} {it.category:<12} {ret_str:<4} {ref_str:<6} {js_str:<5} {result.latency_ms:7.0f}  {it.question[:60]}")

    summary = _summarize(per_item)
    _print_summary(summary)

    # Persist results
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = output_path or (RESULTS_DIR / f"{ts}.json")
    report = {
        "timestamp_utc": ts,
        "host": host,
        "judge_model": judge_model,
        "page_tolerance": tolerance,
        "summary": summary,
        "items": [it.__dict__ for it in per_item],
    }
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[bench] results written to {output_path}")
    return report


def _summarize(items: List[ItemMetrics]) -> Dict[str, Any]:
    def _rate(lst: list, positive) -> Optional[float]:
        if not lst: return None
        return sum(1 for v in lst if v == positive) / len(lst)

    retrieval_items = [i for i in items if i.retrieval_hit is not None]
    refusal_items = [i for i in items if i.refusal_expected]
    judge_items = [i for i in items if i.judge_score is not None]

    latencies = [i.latency_ms for i in items if i.latency_ms > 0]

    return {
        "total_items": len(items),
        "retrieval_evaluated": len(retrieval_items),
        "retrieval_hit_rate": _rate([i.retrieval_hit for i in retrieval_items], True),
        "refusal_items": len(refusal_items),
        "refusal_correct_rate": _rate([i.refusal_seen for i in refusal_items], True),
        "judge_scored": len(judge_items),
        "judge_mean": (statistics.mean([i.judge_score for i in judge_items]) if judge_items else None),
        "judge_median": (statistics.median([i.judge_score for i in judge_items]) if judge_items else None),
        "latency_p50_ms": statistics.median(latencies) if latencies else None,
        "latency_p95_ms": _p(latencies, 0.95) if latencies else None,
        "latency_mean_ms": statistics.mean(latencies) if latencies else None,
    }


def _p(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    k = max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))
    return s[k]


def _print_summary(s: Dict[str, Any]) -> None:
    print()
    print("=" * 64)
    print("  Quality benchmark summary")
    print("=" * 64)
    print(f"  Items evaluated              : {s['total_items']}")
    if s["retrieval_hit_rate"] is not None:
        print(f"  Retrieval hit-rate           : {s['retrieval_hit_rate']*100:.1f}%  "
              f"({s['retrieval_evaluated']} in-scope items)")
    if s["refusal_correct_rate"] is not None:
        print(f"  Refusal correctness          : {s['refusal_correct_rate']*100:.1f}%  "
              f"({s['refusal_items']} off-topic items)")
    if s["judge_mean"] is not None:
        print(f"  LLM-as-judge mean score      : {s['judge_mean']:.2f} / 5")
        print(f"  LLM-as-judge median score    : {s['judge_median']} / 5")
    if s["latency_p50_ms"] is not None:
        print(f"  Answer latency p50           : {s['latency_p50_ms']:.0f} ms")
        print(f"  Answer latency p95           : {s['latency_p95_ms']:.0f} ms")
    print("=" * 64)


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #

def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("BENCHMARK_HOST", "http://localhost:8000"))
    parser.add_argument("--questions", default=str(QUESTIONS_FILE))
    # Judge defaults: reuse the app's LLM endpoint so the benchmark
    # stays fully local. Override via JUDGE_* env vars if you want a
    # different (still local) judge model.
    default_judge_url = os.environ.get("JUDGE_BASE_URL") or os.environ.get(
        "LLM_BASE_URL", "http://localhost:11434/v1"
    )
    default_judge_key = os.environ.get("JUDGE_API_KEY") or os.environ.get("LLM_API_KEY", "ollama")
    default_judge_model = (
        os.environ.get("JUDGE_MODEL")
        or os.environ.get("BENCHMARK_JUDGE_MODEL")
        or "qwen2.5:7b-instruct"      # matches app/config.LLM_MODEL by default
    )

    parser.add_argument("--judge-model", default=default_judge_model)
    parser.add_argument("--judge-base-url", default=default_judge_url)
    parser.add_argument("--no-judge", action="store_true", help="Skip the LLM-as-judge step.")
    parser.add_argument("--output", default=None, help="Explicit output path for the JSON report.")
    args = parser.parse_args(argv)

    judge_base_url = "" if args.no_judge else args.judge_base_url
    judge_api_key = "" if args.no_judge else default_judge_key
    if args.no_judge:
        print("[bench] --no-judge set; skipping LLM-as-judge.")

    run_benchmark(
        host=args.host,
        questions_path=Path(args.questions),
        judge_model=args.judge_model,
        judge_base_url=judge_base_url,
        judge_api_key=judge_api_key,
        output_path=Path(args.output) if args.output else None,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(_main())
