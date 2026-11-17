"""Preprocessing: turn any uploaded picture into the exact shape the model expects.

Every image (from the dataset, from a single prediction, or from a bulk upload)
goes through the SAME steps, so the model always sees consistent input:
  1. open the file and make sure it really is an image
  2. convert to RGB (removes transparency / greyscale differences)
  3. resize to 224 x 224 pixels (the size MobileNetV2 was built for)
  4. turn it into numbers (a NumPy array of shape 224 x 224 x 3)
Scaling the pixel values to the range [-1, 1] happens INSIDE the model
(a Rescaling layer), so it can never be forgotten at prediction time.
"""
import io
import os
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

IMG_SIZE = (224, 224)
CLASS_NAMES = ["angular_leaf_spot", "bean_rust", "healthy"]
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
UPLOAD_DIR = DATA_DIR / "uploads"


def load_image(source) -> Image.Image:
    """Open an image from a path, raw bytes or a file object, as RGB."""
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    try:
        img = Image.open(source)
        img.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("The file is not a readable image.") from exc
    return img.convert("RGB")


def resize_image(img: Image.Image) -> Image.Image:
    return img.resize(IMG_SIZE, Image.BILINEAR)


def image_to_array(img: Image.Image) -> np.ndarray:
    """224x224 RGB picture -> float32 array (224, 224, 3), values 0-255."""
    return np.asarray(resize_image(img), dtype=np.float32)


def preprocess_bytes(data: bytes) -> np.ndarray:
    """Raw uploaded bytes -> batch of one image, shape (1, 224, 224, 3)."""
    return np.expand_dims(image_to_array(load_image(data)), axis=0)


def save_clean_image(data: bytes, dest: Path) -> Path:
    """Validate, convert and resize an uploaded image, then save it as JPEG."""
    img = resize_image(load_image(data))
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest = dest.with_suffix(".jpg")
    img.save(dest, format="JPEG", quality=92)
    return dest


def list_images(folder: Path):
    """Return [(path, label)] for a folder laid out as folder/<class_name>/*.jpg"""
    items = []
    for label in CLASS_NAMES:
        class_dir = Path(folder) / label
        if class_dir.is_dir():
            for p in sorted(class_dir.iterdir()):
                if p.suffix.lower() in ALLOWED_EXTENSIONS:
                    items.append((p, label))
    return items


def load_folder_as_arrays(items):
    """[(path, label)] -> (X, y) arrays ready for model.fit."""
    X = np.stack([image_to_array(load_image(p)) for p, _ in items]) if items else np.zeros((0, *IMG_SIZE, 3), np.float32)
    y = np.array([CLASS_NAMES.index(lbl) for _, lbl in items], dtype=np.int64)
    return X, y


# ---------- simple colour features used for the visualisations ----------

def colour_features(img: Image.Image) -> dict:
    """Three easy-to-explain features of a leaf picture.

    brightness  : how light the photo is overall (0 = black, 255 = white)
    greenness   : share of pixels where green clearly dominates (healthy tissue)
    lesion_share: share of pixels that are brown / yellow / rust coloured (disease spots)
    """
    arr = np.asarray(img.convert("RGB").resize((128, 128)), dtype=np.float32)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    brightness = float(arr.mean())
    green = (g > r * 1.05) & (g > b * 1.05)
    # brown / yellow / orange: red is high, red >= green, blue is low
    lesion = (r > 90) & (r >= g * 0.95) & (b < g) & (b < r * 0.75)
    return {
        "brightness": round(brightness, 2),
        "greenness": round(float(green.mean()), 4),
        "lesion_share": round(float(lesion.mean()), 4),
    }
