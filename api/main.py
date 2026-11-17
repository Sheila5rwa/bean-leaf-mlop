"""FastAPI app: the bridge between the website and the model.

Endpoints
  GET  /                -> the web dashboard (api/static/index.html)
  GET  /health          -> model up-time and status
  GET  /metrics         -> evaluation metrics of the current model
  GET  /insights        -> data for the 3 feature visualisations
  POST /predict         -> classify ONE leaf image
  POST /upload          -> bulk upload many images (+ their label), saved to the database
  POST /retrain         -> start retraining with the uploaded images
  GET  /retrain/status  -> progress / result of retraining
Interactive docs are automatically available at /docs (Swagger UI).
"""
import json
import socket
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src import database as db
from src.model import MODEL_PATH, read_metrics, retrain
from src.prediction import Predictor
from src.preprocessing import (CLASS_NAMES, TRAIN_DIR, UPLOAD_DIR, colour_features,
                               list_images, load_image, save_clean_image)

APP_DIR = Path(__file__).resolve().parent
STARTED = time.time()
STARTED_AT = datetime.now(timezone.utc).isoformat(timespec="seconds")
MAX_FILE_BYTES = 10 * 1024 * 1024

app = FastAPI(title="Bean Leaf Disease Classifier", version="1.0")
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")

predictor = Predictor()
stats = {"predictions": 0}
retrain_state = {"status": "idle", "run_id": None, "report": None, "error": None}
retrain_lock = threading.Lock()
_insights_cache = None


@app.on_event("startup")
def startup():
    db.init_db()
    if MODEL_PATH.exists():
        predictor.load()


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(APP_DIR / "static" / "index.html")


@app.get("/health")
def health():
    up = time.time() - STARTED
    return {
        "status": "ok" if predictor.ready else "model_missing",
        "model_loaded": predictor.ready,
        "uptime_seconds": round(up, 1),
        "started_at": STARTED_AT,
        "container": socket.gethostname(),
        "predictions_served": stats["predictions"],
        "retrain_status": retrain_state["status"],
    }


@app.get("/metrics")
def metrics():
    return read_metrics()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    if not data or len(data) > MAX_FILE_BYTES:
        raise HTTPException(400, "Please upload one image under 10 MB.")
    if not predictor.ready:
        raise HTTPException(503, "Model file not found. Put bean_model.keras in models/.")
    try:
        result = predictor.predict_bytes(data)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    stats["predictions"] += 1
    result["filename"] = file.filename
    return result


@app.post("/upload")
async def upload(label: str = Form(...), files: List[UploadFile] = File(...)):
    if label not in CLASS_NAMES:
        raise HTTPException(400, f"label must be one of {CLASS_NAMES}")
    saved, rejected = [], []
    for f in files:
        data = await f.read()
        if not data or len(data) > MAX_FILE_BYTES:
            rejected.append({"file": f.filename, "reason": "empty or larger than 10 MB"}); continue
        try:
            # preprocessing of the uploaded data: validate, RGB, resize to 224x224, save as JPEG
            path = save_clean_image(data, UPLOAD_DIR / label / uuid.uuid4().hex)
        except ValueError as exc:
            rejected.append({"file": f.filename, "reason": str(exc)}); continue
        row_id = db.add_upload(f.filename, path, label)  # <- saved to the database
        saved.append({"id": row_id, "file": f.filename})
    return {"label": label, "saved": len(saved), "rejected": rejected,
            "pending_for_retraining": len(db.pending_uploads())}


@app.get("/uploads")
def uploads():
    return {"counts": db.upload_counts(), "pending": len(db.pending_uploads())}


def _run_retrain(run_id, rows):
    global _insights_cache
    try:
        items = [(Path(r["stored_path"]), r["label"]) for r in rows if Path(r["stored_path"]).exists()]
        report = retrain(items)
        db.mark_used([r["id"] for r in rows])
        if report["kept_new_model"]:
            predictor.load()  # hot-swap: the website now uses the retrained model
        retrain_state.update(status="finished", report=report)
        db.finish_run(run_id, "finished", report)
    except Exception as exc:  # report the error to the UI instead of crashing
        retrain_state.update(status="failed", error=str(exc))
        db.finish_run(run_id, "failed", {"error": str(exc)})
    finally:
        retrain_lock.release()


@app.post("/retrain")
def trigger_retrain():
    rows = db.pending_uploads()
    if not rows:
        raise HTTPException(400, "Upload some new labelled images first.")
    if not retrain_lock.acquire(blocking=False):
        raise HTTPException(409, "Retraining is already running.")
    run_id = db.start_run()
    retrain_state.update(status="running", run_id=run_id, report=None, error=None)
    threading.Thread(target=_run_retrain, args=(run_id, rows), daemon=True).start()
    return {"message": "Retraining started", "run_id": run_id, "images": len(rows)}


@app.get("/retrain/status")
def retrain_status():
    return {**retrain_state, "history": db.runs(5)}


@app.get("/insights")
def insights():
    """Class balance + 3 colour features per class, computed from the training images."""
    global _insights_cache
    if _insights_cache is None:
        per_class = {c: [] for c in CLASS_NAMES}
        for path, label in list_images(TRAIN_DIR):
            per_class[label].append(colour_features(load_image(path)))
        def summary(vals):
            if not vals:
                return {}
            keys = vals[0].keys()
            return {k: {"mean": round(sum(v[k] for v in vals) / len(vals), 4),
                        "values": [v[k] for v in vals]} for k in keys}
        _insights_cache = {
            "class_counts": {c: len(v) for c, v in per_class.items()},
            "features": {c: summary(v) for c, v in per_class.items()},
        }
    return _insights_cache
