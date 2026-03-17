"""
app/preprocessing/image_preprocess.py
======================================
Image preprocessing pipeline for CNN (EfficientNetB0) inference.

Key design decisions (from RICE_DISEASE_EfficientNetB0_FINAL.ipynb):
  - Target size : 224 × 224 pixels  (EfficientNetB0 default)
  - Dtype       : float32, values in [0, 255]
  - NO manual normalisation — EfficientNetB0 has built-in preprocessing
    that maps 0–255 to the correct internal scale automatically.
  - RGB conversion : always forced (some field images may be RGBA or grayscale)
  - Batch dim      : always added (model expects shape (1, 224, 224, 3))

Usage
-----
    from app.preprocessing.image_preprocess import preprocess_image_file
    from app.preprocessing.image_preprocess import preprocess_image_bytes

    # From a file path (local testing)
    img_array = preprocess_image_file("leaf.jpg")

    # From raw bytes (API / Hugging Face upload)
    img_array = preprocess_image_bytes(file_bytes)
"""

import io
from pathlib import Path

import numpy as np
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
IMG_SIZE = (224, 224)          # (width, height) for PIL.Image.resize
SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Pillow >= 9.1 uses Image.Resampling.LANCZOS; older versions use Image.LANCZOS.
# This guard keeps the code working on Colab which may have Pillow < 9.1.
try:
    _LANCZOS = Image.Resampling.LANCZOS
    _BILINEAR = Image.Resampling.BILINEAR
except AttributeError:
    _LANCZOS = Image.LANCZOS      # type: ignore[attr-defined]
    _BILINEAR = Image.BILINEAR    # type: ignore[attr-defined]


# ─────────────────────────────────────────────────────────────────────────────
# Core preprocessing function
# ─────────────────────────────────────────────────────────────────────────────
def _pil_to_model_input(pil_image: Image.Image) -> np.ndarray:
    """
    Convert a PIL Image to a model-ready numpy array.

    Steps:
        1. Convert to RGB (handles RGBA, grayscale, palette modes)
        2. Resize to 224 × 224 using LANCZOS resampling (best quality)
        3. Cast to float32  →  values in [0, 255]
        4. Add batch dimension → shape (1, 224, 224, 3)

    Returns:
        np.ndarray of shape (1, 224, 224, 3), dtype float32.
    """
    pil_image = pil_image.convert("RGB")
    pil_image = pil_image.resize(IMG_SIZE, resample=_LANCZOS)
    arr = np.array(pil_image, dtype=np.float32)     # (224, 224, 3), 0–255
    return np.expand_dims(arr, axis=0)               # (1, 224, 224, 3)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_image_file(image_path: str | Path) -> np.ndarray:
    """
    Load an image from disk and preprocess it for CNN inference.

    Args:
        image_path: Path to the image file (.jpg, .jpeg, .png, etc.)

    Returns:
        np.ndarray of shape (1, 224, 224, 3), dtype float32.

    Raises:
        FileNotFoundError : If the file does not exist.
        ValueError        : If the file extension is not supported.
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if image_path.suffix.lower() not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported image format '{image_path.suffix}'. "
            f"Supported: {SUPPORTED_FORMATS}"
        )

    with Image.open(image_path) as img:
        return _pil_to_model_input(img)


def preprocess_image_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Preprocess an image from raw bytes (e.g., uploaded via HTTP/Gradio).

    Args:
        image_bytes: Raw image bytes (JPEG, PNG, etc.).

    Returns:
        np.ndarray of shape (1, 224, 224, 3), dtype float32.

    Raises:
        ValueError: If the bytes cannot be decoded as an image.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        return _pil_to_model_input(img)
    except Exception as exc:
        raise ValueError(f"Cannot decode image bytes: {exc}") from exc


def preprocess_pil_image(pil_image: Image.Image) -> np.ndarray:
    """
    Preprocess a PIL Image directly (e.g., from Gradio interface).

    Args:
        pil_image: A PIL.Image.Image object (any mode).

    Returns:
        np.ndarray of shape (1, 224, 224, 3), dtype float32.
    """
    return _pil_to_model_input(pil_image)


def get_raw_array(image_path: str | Path) -> np.ndarray:
    """
    Load image as a raw uint8 array without adding the batch dimension.
    Used by Grad-CAM visualisation (needs the original pixel values).

    Args:
        image_path: Path to the image file.

    Returns:
        np.ndarray of shape (224, 224, 3), dtype uint8.
    """
    image_path = Path(image_path)
    with Image.open(image_path) as img:
        img = img.convert("RGB").resize(IMG_SIZE, resample=_LANCZOS)
        return np.array(img, dtype=np.uint8)


def preprocess_batch(image_paths: list[str | Path]) -> np.ndarray:
    """
    Preprocess a list of images into a single batch.

    Args:
        image_paths: List of paths to image files.

    Returns:
        np.ndarray of shape (N, 224, 224, 3), dtype float32.
    """
    arrays = [preprocess_image_file(p)[0] for p in image_paths]  # drop batch dim per item
    return np.stack(arrays, axis=0)
