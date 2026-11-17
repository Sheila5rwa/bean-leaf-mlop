"""Prediction: load the saved model once and classify one picture."""
import threading
import numpy as np

from src.preprocessing import CLASS_NAMES, preprocess_bytes

ADVICE = {
    "healthy": "The leaf looks healthy. Keep monitoring the field weekly.",
    "bean_rust": "Signs of bean rust (small rust-coloured pustules). Remove badly infected leaves, avoid working in wet fields, and ask an agronomist about a suitable fungicide.",
    "angular_leaf_spot": "Signs of angular leaf spot (angular brown patches). Use clean, certified seed, rotate crops, and remove crop debris after harvest.",
}


class Predictor:
    def __init__(self):
        self._model = None
        self._lock = threading.Lock()

    def load(self):
        from src.model import load_model
        with self._lock:
            self._model = load_model()
        return self

    @property
    def ready(self) -> bool:
        return self._model is not None

    def predict_bytes(self, data: bytes) -> dict:
        if self._model is None:
            self.load()
        x = preprocess_bytes(data)
        probs = self._model.predict(x, verbose=0)[0]
        idx = int(np.argmax(probs))
        label = CLASS_NAMES[idx]
        return {
            "label": label,
            "confidence": round(float(probs[idx]), 4),
            "probabilities": {c: round(float(p), 4) for c, p in zip(CLASS_NAMES, probs)},
            "advice": ADVICE[label],
        }
