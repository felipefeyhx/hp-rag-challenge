"""Stress test scenarios.

Hammers the backend with zero think time between requests to find where
the app degrades or refuses connections. This is NOT a realistic user
model — it's the ceiling probe.

Run:
    locust -f loadtests/stress_test.py --host http://localhost:8000 \
           --headless -u 50 -r 25 -t 1m --csv loadtests/reports/stress
"""
from __future__ import annotations

import random

from locust import HttpUser, between, task


QUESTIONS = [
    "What is the paper tray capacity?",
    "How do I connect the printer to Wi-Fi?",
    "What is the printer weight?",
    "How much RAM does the OMEN 17.3 have?",
    "What are the CPU options for the laptop?",
    "How do I enable Quiet Mode?",
    "What is the recommended paper type?",
    "What is the warranty period?",
]


class StressUser(HttpUser):
    """Zero think-time user — attacks the server as fast as possible."""

    # No wait time; fire the next request the instant this one returns.
    wait_time = between(0.0, 0.0)

    @task(20)
    def chat(self) -> None:
        """The expensive endpoint — heavy hitter."""
        # 60 s timeout so hung requests eventually surface as failures.
        self.client.post(
            "/chat",
            json={"message": random.choice(QUESTIONS)},
            name="POST /chat",
            timeout=60,
        )

    @task(1)
    def health(self) -> None:
        """Cheap endpoint — control group, should stay fast."""
        self.client.get("/health", name="GET /health", timeout=10)
