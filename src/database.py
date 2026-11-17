"""A small SQLite database that records every uploaded picture.

Table `uploads` keeps: file path, label, original filename, upload time and
whether the picture has already been used in a retraining run.
Table `retrain_runs` keeps a history of every retraining with its metrics.
"""
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.getenv("DB_PATH", ROOT / "data" / "pipeline.db"))


def _conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH, timeout=30)
    c.row_factory = sqlite3.Row
    return c


def init_db():
    with _conn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS uploads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            original_name TEXT, stored_path TEXT NOT NULL, label TEXT NOT NULL,
            uploaded_at TEXT NOT NULL, used_in_training INTEGER NOT NULL DEFAULT 0);
        CREATE TABLE IF NOT EXISTS retrain_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at TEXT NOT NULL, status TEXT NOT NULL, report TEXT);
        """)


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def add_upload(original_name, stored_path, label):
    with _conn() as c:
        cur = c.execute("INSERT INTO uploads (original_name, stored_path, label, uploaded_at) VALUES (?,?,?,?)",
                        (original_name, str(stored_path), label, now()))
        return cur.lastrowid


def pending_uploads():
    with _conn() as c:
        return [dict(r) for r in c.execute("SELECT * FROM uploads WHERE used_in_training = 0")]


def mark_used(ids):
    if not ids:
        return
    with _conn() as c:
        c.executemany("UPDATE uploads SET used_in_training = 1 WHERE id = ?", [(i,) for i in ids])


def upload_counts():
    with _conn() as c:
        rows = c.execute("SELECT label, used_in_training, COUNT(*) n FROM uploads GROUP BY label, used_in_training")
        return [dict(r) for r in rows]


def start_run():
    with _conn() as c:
        return c.execute("INSERT INTO retrain_runs (started_at, status) VALUES (?, 'running')", (now(),)).lastrowid


def finish_run(run_id, status, report):
    with _conn() as c:
        c.execute("UPDATE retrain_runs SET status = ?, report = ? WHERE id = ?", (status, json.dumps(report), run_id))


def runs(limit=10):
    with _conn() as c:
        out = []
        for r in c.execute("SELECT * FROM retrain_runs ORDER BY id DESC LIMIT ?", (limit,)):
            d = dict(r); d["report"] = json.loads(d["report"]) if d["report"] else None
            out.append(d)
        return out
