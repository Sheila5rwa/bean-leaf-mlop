"""Flood the API with prediction requests.

Run (with the app already running):
  locust -f locust/locustfile.py --host http://localhost:8080 \
         --users 50 --spawn-rate 10 --run-time 1m --headless \
         --csv locust/results/containers_1 --html locust/results/containers_1.html
"""
import random
from pathlib import Path

from locust import HttpUser, between, task

ROOT = Path(__file__).resolve().parent.parent
IMAGES = [p.read_bytes() for p in sorted((ROOT / "data" / "test").rglob("*.jpg"))[:30]]
if not IMAGES:
    raise SystemExit("Put some test images in data/test/<class>/ first (the notebook does this).")


class FarmerUser(HttpUser):
    wait_time = between(0.5, 2)

    @task(5)
    def predict(self):
        self.client.post("/predict", files={"file": ("leaf.jpg", random.choice(IMAGES), "image/jpeg")})

    @task(1)
    def health(self):
        self.client.get("/health")
