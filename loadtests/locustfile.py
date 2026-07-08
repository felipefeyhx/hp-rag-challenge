"""Locust load-test scenarios for the HP RAG Chatbot.

Realistic user behavior:
- Every simulated user creates one session on startup.
- They then send chat messages in that session (weighted heaviest).
- Occasionally they list sessions or view the session detail (like a real UI would).
- The pool of questions rotates so caches / rate limits get exercised.

Run headless (no browser):
    locust -f loadtests/locustfile.py --host http://localhost:8000 \\
           --headless -u 10 -r 2 -t 3m --csv loadtests/reports/run

Run with the web UI:
    locust -f loadtests/locustfile.py --host http://localhost:8000
    # → open http://localhost:8089
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import List

from locust import HttpUser, between, events, task


QUESTIONS_FILE = Path(__file__).parent / "questions.txt"


def _load_questions() -> List[str]:
    lines = QUESTIONS_FILE.read_text(encoding="utf-8").splitlines()
    return [q.strip() for q in lines if q.strip()]


QUESTIONS = _load_questions()


class HpChatbotUser(HttpUser):
    """Simulates one browser tab chatting with the bot."""

    # Real users don't send messages back-to-back — leave 1–3 seconds between.
    wait_time = between(1.0, 3.0)

    def on_start(self) -> None:
        """Create a session once per virtual user."""
        with self.client.post("/sessions", catch_response=True, name="POST /sessions") as r:
            if r.status_code != 201:
                r.failure(f"session create failed: {r.status_code}")
                self.session_id = None
                return
            self.session_id = r.json()["id"]

    # ---------------- Tasks (weighted) ---------------- #

    @task(15)
    def send_chat_message(self) -> None:
        """Most common action: send a question and read the answer."""
        if not getattr(self, "session_id", None):
            return
        payload = {
            "message": random.choice(QUESTIONS),
            "session_id": self.session_id,
        }
        with self.client.post(
            "/chat",
            json=payload,
            name="POST /chat",
            catch_response=True,
        ) as r:
            if r.status_code >= 500:
                r.failure(f"server error: {r.status_code}")
            elif r.status_code == 200 and not r.json().get("answer"):
                r.failure("empty answer")

    @task(3)
    def view_session_history(self) -> None:
        if not getattr(self, "session_id", None):
            return
        self.client.get(f"/sessions/{self.session_id}", name="GET /sessions/{id}")

    @task(2)
    def list_sessions(self) -> None:
        self.client.get("/sessions", name="GET /sessions")

    @task(1)
    def health_check(self) -> None:
        self.client.get("/health", name="GET /health")


# --------------------------------------------------------------------------- #
# Report hooks                                                                #
# --------------------------------------------------------------------------- #

@events.quitting.add_listener
def _print_stats_snapshot(environment, **kwargs):  # pragma: no cover
    """Print a concise summary at the end of a headless run."""
    stats = environment.stats.total
    if stats.num_requests == 0:
        print("[loadtest] no requests were made.")
        return
    rps = stats.total_rps
    print("=" * 60)
    print(f"Total requests           : {stats.num_requests}")
    print(f"Total failures           : {stats.num_failures}")
    print(f"Failure rate             : {stats.fail_ratio * 100:.2f}%")
    print(f"Requests/sec (avg)       : {rps:.2f}")
    print(f"Requests/min             : {rps * 60:.0f}")
    print(f"Median response time (ms): {stats.median_response_time}")
    print(f"p95 response time (ms)   : {stats.get_response_time_percentile(0.95):.0f}")
    print(f"Max response time (ms)   : {stats.max_response_time:.0f}")
    print("=" * 60)
