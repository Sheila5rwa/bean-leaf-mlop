"""Model creation, training, evaluation and RETRAINING.

The first model is built on MobileNetV2, a network Google pre-trained on
1.4 million everyday photos (ImageNet). We freeze it, put a small new
"head" on top that learns our 3 bean classes, then fine-tune a little.

Retraining does NOT start from scratch: it loads OUR saved bean model
(the custom model we trained in the notebook) and continues training it
on the newly uploaded pictures plus a sample of the original training
pictures (so it does not forget what it already knew).
"""
import json
import os
import random
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.preprocessing import (CLASS_NAMES, IMG_SIZE, TEST_DIR, TRAIN_DIR,
                               UPLOAD_DIR, list_images, load_folder_as_arrays)

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.getenv("MODELS_DIR", ROOT / "models"))
MODEL_PATH = MODELS_DIR / "bean_model.keras"
METRICS_PATH = MODELS_DIR / "metrics.json"
SEED = 42


def build_model(num_classes: int = len(CLASS_NAMES), dropout: float = 0.3, l2: float = 1e-4):
    import tensorflow as tf
    from tensorflow import keras

    base = keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet")
    base.trainable = False  # step 1: keep the pre-trained knowledge frozen

    augment = keras.Sequential([
        keras.layers.RandomFlip("horizontal_and_vertical"),
        keras.layers.RandomRotation(0.15),
        keras.layers.RandomZoom(0.15),
        keras.layers.RandomContrast(0.15),
    ], name="augmentation")

    inputs = keras.Input(shape=(*IMG_SIZE, 3), name="image")
    x = augment(inputs)
    x = keras.layers.Rescaling(1.0 / 127.5, offset=-1.0, name="scale_to_minus1_1")(x)
    x = base(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(dropout)(x)                              # regularisation
    outputs = keras.layers.Dense(num_classes, activation="softmax",
                                 kernel_regularizer=keras.regularizers.l2(l2),  # regularisation
                                 name="prediction")(x)
    model = keras.Model(inputs, outputs, name="bean_leaf_mobilenetv2")
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def unfreeze_top(model, n_layers: int = 30, lr: float = 1e-5):
    """Fine-tuning: let the last few MobileNetV2 layers adjust to bean leaves."""
    from tensorflow import keras
    base = next(l for l in model.layers if l.name.startswith("mobilenetv2"))
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False
    for layer in base.layers:  # keep BatchNorm frozen, standard practice
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def callbacks(patience: int = 4):
    from tensorflow import keras
    return [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-7),
    ]


def evaluate(model, X, y) -> dict:
    """Accuracy, loss, precision, recall, F1 and the confusion matrix."""
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                                 log_loss, precision_score, recall_score)
    probs = model.predict(X, verbose=0)
    preds = probs.argmax(axis=1)
    labels = list(range(len(CLASS_NAMES)))
    return {
        "accuracy": round(float(accuracy_score(y, preds)), 4),
        "loss": round(float(log_loss(y, probs, labels=labels)), 4),
        "precision": round(float(precision_score(y, preds, average="macro", zero_division=0)), 4),
        "recall": round(float(recall_score(y, preds, average="macro", zero_division=0)), 4),
        "f1": round(float(f1_score(y, preds, average="macro", zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y, preds, labels=labels).tolist(),
        "n_samples": int(len(y)),
    }


def load_model(path: Path = MODEL_PATH):
    from tensorflow import keras
    return keras.models.load_model(path)


def read_metrics() -> dict:
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text())
    return {}


def write_metrics(metrics: dict):
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))


def retrain(upload_items, epochs: int = 5, replay_per_class: int = 60, lr: float = 1e-5) -> dict:
    """Retrain the EXISTING saved model with newly uploaded images.

    upload_items: list of (path, label) for the uploaded, already preprocessed pictures.
    Returns a report with metrics before and after, and whether the new model was kept.
    """
    import tensorflow as tf
    from tensorflow import keras

    if not upload_items:
        raise ValueError("No new uploaded images to retrain on.")
    random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
    started = time.time()

    model = load_model()  # <- our own custom model is the pre-trained starting point

    # mix new data with a small "replay" sample of old data so the model does not forget
    replay = []
    for label in CLASS_NAMES:
        old = [it for it in list_images(TRAIN_DIR) if it[1] == label]
        replay += random.sample(old, min(replay_per_class, len(old)))
    items = list(upload_items) + replay
    random.shuffle(items)
    X, y = load_folder_as_arrays(items)

    test_items = list_images(TEST_DIR)
    X_test, y_test = load_folder_as_arrays(test_items)
    before = evaluate(model, X_test, y_test) if len(y_test) else {}

    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    val_split = 0.15 if len(y) >= 20 else 0.0
    history = model.fit(X, y, epochs=epochs, batch_size=16, validation_split=val_split,
                        callbacks=callbacks(patience=2) if val_split else None, verbose=0)

    after = evaluate(model, X_test, y_test) if len(y_test) else {}
    # keep the new model only if it is not clearly worse on the fixed test set
    kept = (not before) or after["accuracy"] >= before["accuracy"] - 0.02
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if kept:
        if MODEL_PATH.exists():
            shutil.copy(MODEL_PATH, MODELS_DIR / f"bean_model_backup_{stamp}.keras")
        model.save(MODEL_PATH)
        m = read_metrics()
        m.update({"test": after, "last_retrained_at": stamp,
                  "retrain_count": int(m.get("retrain_count", 0)) + 1})
        write_metrics(m)

    return {
        "kept_new_model": bool(kept),
        "trained_on_new_images": len(upload_items),
        "replay_images": len(replay),
        "epochs_run": len(history.history["loss"]),
        "final_train_loss": round(float(history.history["loss"][-1]), 4),
        "before": before, "after": after,
        "seconds": round(time.time() - started, 1),
        "finished_at": stamp,
    }
