"""Run the flood test with 1, 2 and 3 copies of the API, then print the README table.

No Docker needed: each "copy" is a separate uvicorn process with its own model in memory,
limited to one CPU thread (like a container limited to 1 CPU).

Usage (from the repository root, with requirements.txt and locust/requirements.txt installed):
    python locust/run_scaling.py                 # copies 1,2,3 / 50 users / 1 minute each
    python locust/run_scaling.py --users 30 --time 45s --copies 1 2 4
"""
import argparse
import csv
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "locust" / "results"
BASE_PORT = 8001


def wait_until_ready(url, timeout=180):
    end = time.time() + timeout
    while time.time() < end:
        try:
            with urllib.request.urlopen(url + "/health", timeout=3) as r:
                if b'"model_loaded":true' in r.read().replace(b" ", b""):
                    return
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError(f"{url} did not become ready in {timeout}s")


def run_once(copies, users, spawn, run_time):
    env = {**os.environ, "OMP_NUM_THREADS": "1", "TF_NUM_INTRAOP_THREADS": "1",
           "TF_NUM_INTEROP_THREADS": "1", "TF_CPP_MIN_LOG_LEVEL": "2"}
    urls = [f"http://127.0.0.1:{BASE_PORT + i}" for i in range(copies)]
    procs = [subprocess.Popen([sys.executable, "-m", "uvicorn", "api.main:app", "--port", str(BASE_PORT + i)],
                              cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
             for i in range(copies)]
    try:
        for u in urls:
            wait_until_ready(u)
        prefix = RESULTS / f"copies_{copies}"
        subprocess.run([sys.executable, "-m", "locust", "-f", str(ROOT / "locust" / "locustfile.py"),
                        "--headless", "--users", str(users), "--spawn-rate", str(spawn),
                        "--run-time", run_time, "--csv", str(prefix), "--html", f"{prefix}.html",
                        "--host", urls[0]],
                       cwd=ROOT, env={**env, "BACKENDS": ",".join(urls)}, check=False)
        with open(f"{prefix}_stats.csv", newline="") as f:
            for row in csv.DictReader(f):
                if row["Name"] == "Aggregated":
                    return row
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            p.wait(timeout=20)
    raise RuntimeError("No Aggregated row found in the Locust CSV")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--copies", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--users", type=int, default=50)
    ap.add_argument("--spawn", type=int, default=10)
    ap.add_argument("--time", default="1m")
    a = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)

    lines = ["| Copies of the API | Users | Requests/s | Median (ms) | 95th percentile (ms) | Failures |",
             "|---|---|---|---|---|---|"]
    for n in a.copies:
        print(f"\n=== {n} cop{'y' if n == 1 else 'ies'} ===", flush=True)
        r = run_once(n, a.users, a.spawn, a.time)
        lines.append(f"| {n} | {a.users} | {float(r['Requests/s']):.1f} | {r['Median Response Time']} | "
                     f"{r['95%']} | {r['Failure Count']} of {r['Request Count']} |")
    table = "\n".join(lines)
    (RESULTS / "summary_table.md").write_text(table + "\n")
    print("\nPaste this into the README:\n\n" + table)


if __name__ == "__main__":
    main()
