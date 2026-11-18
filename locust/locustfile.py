"""Flood the API with prediction requests.

Two ways to run it:
  1) One address (e.g. nginx / Docker / Hugging Face):
       locust -f locust/locustfile.py --host http://localhost:8080 --users 50 --spawn-rate 10 \
              --run-time 1m --headless --csv locust/results/containers_1
  2) Several copies of the API, no Docker needed: set BACKENDS and every simulated
     user is assigned to one copy in turn (round-robin), like a load balancer would do.
       BACKENDS=http://127.0.0.1:8001,http://127.0.0.1:8002 locust -f locust/locustfile.py ...
     (locust/run_scaling.py does all of this automatically.)
"""
import itertools
import os
import random
from pathlib import Path

from locust import HttpUser, between, task

ROOT = Path(__file__).resolve().parent.parent
IMAGES = [p.read_bytes() for p in sorted((ROOT / "data" / "test").rglob("*.jpg"))[:30]]
if not IMAGES:
    raise SystemExit("Put some test images in data/test/<class>/ first (the notebook does this).")

BACKENDS = [b.strip() for b in os.getenv("BACKENDS", "").split(",") if b.strip()]
_next_backend = itertools.cycle(BACKENDS) if BACKENDS else None


class FarmerUser(HttpUser):
    wait_time = between(0.5, 2)

    def on_start(self):
        if _next_backend:
            self.client.base_url = next(_next_backend)

    @task(5)
    def predict(self):
        self.client.post("/predict", files={"file": ("leaf.jpg", random.choice(IMAGES), "image/jpeg")})

    @task(1)
    def health(self):
        self.client.get("/health")
