"""
app/models/load_model.py
========================
Model loading with automatic download from Hugging Face Hub.

How it works
------------
1. At startup, models are downloaded from HF Hub into /tmp/rice_models/
2. Models are cached — if already downloaded, skip download
3. Local trained_models/ folder is NOT needed on HF Space at all

HF Hub model repo : mlresearcher05/rice-disease-models
Files in repo     :
    rice_cnn_model.keras
    rice_lstm_model.pth
    scaler.pkl
"""

import os
import pickle
import warnings
from pathlib import Path

import torch
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

from app.models.lstm_model import WeatherRiskLSTM

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# HF Hub config
# ─────────────────────────────────────────────────────────────────────────────
HF_MODEL_REPO  = "mlresearcher05/rice-disease-models"
MODEL_CACHE_DIR = Path("/tmp/rice_models")   # writable on HF Space


def _download_from_hub(filename: str) -> Path:
    """
    Download a single file from HF Hub model repo into cache dir.
    If already cached, skip download and return cached path.

    Args:
        filename: e.g. "rice_cnn_model.keras"

    Returns:
        Local path to the downloaded file.
    """
    from huggingface_hub import hf_hub_download

    cached_path = MODEL_CACHE_DIR / filename
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if cached_path.exists():
        print(f"[load_model] Using cached: {cached_path}")
        return cached_path

    print(f"[load_model] Downloading {filename} from {HF_MODEL_REPO} ...")
    downloaded = hf_hub_download(
        repo_id   = HF_MODEL_REPO,
        filename  = filename,
        repo_type = "model",
        local_dir = str(MODEL_CACHE_DIR),
    )
    print(f"[load_model] Downloaded → {downloaded}")
    return Path(downloaded)


def _resolve_path(model_path: str | Path, hf_filename: str) -> Path:
    """
    Resolve model path:
    - If local path exists → use it directly (local dev / Colab)
    - If not → download from HF Hub (HF Space deployment)

    Args:
        model_path  : Path from config.yaml (e.g. trained_models/rice_cnn_model.keras)
        hf_filename : Filename in HF Hub repo (e.g. rice_cnn_model.keras)

    Returns:
        Resolved local Path ready to load.
    """
    local = Path(model_path)
    if local.exists():
        print(f"[load_model] Using local file: {local}")
        return local

    print(f"[load_model] Local path not found: {local} — downloading from HF Hub")
    return _download_from_hub(hf_filename)


# ─────────────────────────────────────────────────────────────────────────────
# Device selection
# ─────────────────────────────────────────────────────────────────────────────
def get_device() -> torch.device:
    """Return CUDA device if available, else CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[load_model] PyTorch device: {device}")
    return device


# ─────────────────────────────────────────────────────────────────────────────
# CNN loader
# ─────────────────────────────────────────────────────────────────────────────
def load_cnn_model(model_path: str | Path) -> keras.Model:
    """
    Load EfficientNetB0 CNN.
    Downloads from HF Hub automatically if not found locally.
    """
    resolved = _resolve_path(model_path, "rice_cnn_model.keras")
    print(f"[load_model] Loading CNN from: {resolved}")

    model = None

    # Method 1 — normal load
    try:
        model = keras.models.load_model(str(resolved))
        print("[load_model] CNN loaded via method 1 (normal)")
    except Exception as e1:
        print(f"[load_model] Method 1 failed: {e1.__class__.__name__}: {str(e1)[:80]}")

    # Method 2 — safe_mode=False
    if model is None:
        try:
            model = keras.models.load_model(str(resolved), safe_mode=False)
            print("[load_model] CNN loaded via method 2 (safe_mode=False)")
        except Exception as e2:
            print(f"[load_model] Method 2 failed: {e2.__class__.__name__}: {str(e2)[:80]}")

    # Method 3 — rebuild architecture + load weights only
    if model is None:
        try:
            print("[load_model] Trying method 3: rebuild architecture + load weights only")
            import tensorflow as tf
            from tensorflow.keras import layers
            from tensorflow.keras.applications import EfficientNetB0
            from app.models.cnn_model import build_augmentation_layer

            data_augmentation = build_augmentation_layer()
            base_model = EfficientNetB0(
                input_shape=(224, 224, 3),
                include_top=False,
                weights=None,
            )
            base_model.trainable = True
            inputs = keras.Input(shape=(224, 224, 3), name="image_input")
            x = data_augmentation(inputs)
            x = base_model(x, training=False)
            x = layers.GlobalAveragePooling2D(name="gap")(x)
            x = layers.Dropout(0.3, name="dropout_1")(x)
            x = layers.Dense(
                256, activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                name="dense_256",
            )(x)
            x = layers.Dropout(0.2, name="dropout_2")(x)
            outputs = layers.Dense(4, activation="softmax", name="classifier")(x)
            model = keras.Model(inputs, outputs, name="RiceLeaf_v3_EfficientNetB0")
            model.load_weights(str(resolved))
            print("[load_model] CNN loaded via method 3 (weights only)")
        except Exception as e3:
            raise RuntimeError(
                f"All 3 CNN load methods failed.\n"
                f"Keras version mismatch between training and deployment.\n"
                f"Last error: {e3}"
            ) from e3

    print(
        f"[load_model] CNN ready — "
        f"input: {model.input_shape}  output: {model.output_shape}  "
        f"params: {model.count_params():,}"
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# LSTM loader
# ─────────────────────────────────────────────────────────────────────────────
def load_lstm_model(
    model_path: str | Path,
    device: torch.device | None = None,
) -> tuple[WeatherRiskLSTM, torch.device]:
    """
    Load WeatherRiskLSTM.
    Downloads from HF Hub automatically if not found locally.
    """
    resolved = _resolve_path(model_path, "rice_lstm_model.pth")

    if device is None:
        device = get_device()

    print(f"[load_model] Loading LSTM from: {resolved}")
    model = WeatherRiskLSTM(input_size=20, hidden_size=64, num_layers=2, dropout=0.5)
    state_dict = torch.load(str(resolved), map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[load_model] LSTM ready — params: {n_params:,}  device: {device}")
    return model, device


# ─────────────────────────────────────────────────────────────────────────────
# Scaler loader
# ─────────────────────────────────────────────────────────────────────────────
def load_scaler(scaler_path: str | Path) -> MinMaxScaler:
    """
    Load MinMaxScaler.
    Downloads from HF Hub automatically if not found locally.
    """
    resolved = _resolve_path(scaler_path, "scaler.pkl")

    print(f"[load_model] Loading scaler from: {resolved}")
    with open(resolved, "rb") as f:
        scaler = pickle.load(f)

    print(f"[load_model] Scaler ready")
    return scaler


# ─────────────────────────────────────────────────────────────────────────────
# Combined loader
# ─────────────────────────────────────────────────────────────────────────────
def load_all_models(
    cnn_path: str | Path,
    lstm_path: str | Path,
    scaler_path: str | Path,
) -> dict:
    """
    Load all 3 models at startup.
    Automatically downloads from HF Hub if not found locally.
    """
    device = get_device()
    cnn            = load_cnn_model(cnn_path)
    lstm, device   = load_lstm_model(lstm_path, device)
    scaler         = load_scaler(scaler_path)

    print("[load_model] All models loaded successfully.")
    return {"cnn": cnn, "lstm": lstm, "scaler": scaler, "device": device}