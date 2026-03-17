"""
app/models/load_model.py
========================
Centralised model loading for CNN (TensorFlow/Keras) and LSTM (PyTorch).

Usage
-----
    from app.models.load_model import load_cnn_model, load_lstm_model, get_device

    cnn  = load_cnn_model("trained_models/rice_cnn_model.keras")
    lstm, device = load_lstm_model("trained_models/rice_lstm_model.pth")

Models are loaded once at startup and reused across all requests.
Do NOT reload models per request — it is slow and will crash on Hugging Face Spaces.
"""

import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

from app.models.lstm_model import WeatherRiskLSTM

warnings.filterwarnings("ignore")


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
    Load the trained EfficientNetB0 Keras model from disk.

    Supports both .keras and legacy .h5 formats.
    The model contains the CategoricalFocalCrossentropy loss inside it —
    custom_objects are NOT needed because TF 2.x serialises built-in losses.

    Args:
        model_path: Path to the saved .keras or .h5 file.

    Returns:
        Compiled keras.Model ready for inference.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"CNN model not found at: {model_path}\n"
            "Train the model first using notebooks/training_experiments.ipynb "
            "and copy the output to trained_models/."
        )

    print(f"[load_model] Loading CNN from: {model_path}")
    model = keras.models.load_model(str(model_path))
    print(
        f"[load_model] CNN loaded — "
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
    Load the trained WeatherRiskLSTM from a PyTorch state dict.

    The architecture (input_size=20) is hard-coded to match the saved weights.
    Changing any architecture hyperparameter without retraining will raise a
    RuntimeError at load time.

    Args:
        model_path: Path to the .pth state dict file (best_lstm_v2.pth).
        device    : Target device. Auto-detected if None.

    Returns:
        (model, device) tuple — model is set to eval() mode.

    Raises:
        FileNotFoundError: If the state dict file does not exist.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"LSTM model not found at: {model_path}\n"
            "Train the model first using notebooks/training_experiments.ipynb "
            "and copy the output to trained_models/."
        )

    if device is None:
        device = get_device()

    print(f"[load_model] Loading LSTM from: {model_path}")
    model = WeatherRiskLSTM(input_size=20, hidden_size=64, num_layers=2, dropout=0.5)
    state_dict = torch.load(str(model_path), map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[load_model] LSTM loaded — trainable params: {n_params:,}  device: {device}")
    return model, device


# ─────────────────────────────────────────────────────────────────────────────
# Scaler loader
# ─────────────────────────────────────────────────────────────────────────────
def load_scaler(scaler_path: str | Path) -> MinMaxScaler:
    """
    Load the MinMaxScaler fitted on the training weather data.

    The scaler must be the SAME one used during preprocessing to avoid
    data distribution mismatch at inference time.

    Args:
        scaler_path: Path to scaler.pkl.

    Returns:
        Fitted sklearn MinMaxScaler.

    Raises:
        FileNotFoundError: If the scaler file does not exist.
    """
    scaler_path = Path(scaler_path)
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Scaler not found at: {scaler_path}\n"
            "Run the preprocessing pipeline in the LSTM notebook to generate scaler.pkl."
        )

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    print(f"[load_model] Scaler loaded from: {scaler_path}")
    return scaler


# ─────────────────────────────────────────────────────────────────────────────
# Combined loader (convenience for app startup)
# ─────────────────────────────────────────────────────────────────────────────
def load_all_models(
    cnn_path: str | Path,
    lstm_path: str | Path,
    scaler_path: str | Path,
) -> dict:
    """
    Load CNN, LSTM, and scaler in one call.  Intended for app startup.

    Args:
        cnn_path   : Path to .keras CNN model.
        lstm_path  : Path to .pth LSTM state dict.
        scaler_path: Path to scaler.pkl.

    Returns:
        dict with keys: "cnn", "lstm", "scaler", "device"

    Example (in main.py):
        models = load_all_models(
            "trained_models/rice_cnn_model.keras",
            "trained_models/rice_lstm_model.pth",
            "trained_models/scaler.pkl",
        )
    """
    device = get_device()
    cnn = load_cnn_model(cnn_path)
    lstm, device = load_lstm_model(lstm_path, device)
    scaler = load_scaler(scaler_path)

    print("[load_model] All models loaded successfully.")
    return {"cnn": cnn, "lstm": lstm, "scaler": scaler, "device": device}
