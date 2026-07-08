"""Failure-probe scenarios.

Deliberately sends requests the API is guaranteed to reject, so Locust
records non-zero failures. Used to prove that the load test actually
detects failures (they were 0% in the real runs).

Run:
    locust -f loadtests/failure_probe.py --host http://localhost:8000 \
           --headless -u 5 -r 5 -t 30s --csv loadtests/reports/failure_probe

Expected: ~100% failure rate on both endpoints.
"""
from __future__ import annotations

from locust import HttpUser, between, task


class FailureProbeUser(HttpUser):
    """One virtual user firing invalid requests as fast as possible."""

    wait_time = between(0.1, 0.3)

    @task(2)
    def empty_message(self) -> None:
        """POST /chat with empty message → 422 (Pydantic min_length=1)."""
        # Locust marks HTTP >= 400 as a failure automatically.
        self.client.post("/chat", json={"message": ""}, name="POST /chat (empty)")

    @task(1)
    def missing_session(self) -> None:
        """GET /sessions/{id} for a non-existent id → 404."""
        self.client.get(
            "/sessions/nonexistent-session-1234567890",
            name="GET /sessions/{missing}",
        )
