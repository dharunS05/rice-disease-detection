"""
app/utils/helper.py
====================
Utility functions shared across the application.

Sections:
  1. Grad-CAM          — heatmap generation for CNN explainability
  2. Weather Sequence  — build scaled LSTM input from raw daily weather data
  3. Config Loader     — load config.yaml into a dict
  4. Risk Band         — interpret fused score as Low / Moderate / High

Grad-CAM implementation follows:
  Selvaraju et al. 2017 (ICCV) — Grad-CAM: Visual Explanations from Deep Networks.
  The fix from RICE_DISEASE_EfficientNetB0_FINAL.ipynb (Cell 12):
    Build ONE dual-output model so both conv_maps and predictions share a SINGLE
    forward pass → GradientTape can correctly trace gradients.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import yaml
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras


# ─────────────────────────────────────────────────────────────────────────────
# 1. Grad-CAM
# ─────────────────────────────────────────────────────────────────────────────
LAST_CONV_LAYER = "top_conv"   # EfficientNetB0 last conv — output shape (7, 7, 1280)


def build_gradcam_model(cnn_model: keras.Model) -> keras.Model:
    """
    Build a dual-output Grad-CAM model from the trained CNN.

    Both outputs (conv feature maps + final predictions) share ONE forward pass,
    which allows GradientTape to trace gradients correctly.

    Args:
        cnn_model: The loaded EfficientNetB0 Keras model.

    Returns:
        A keras.Model with:
            input : (1, 224, 224, 3)
            output: [(1, 7, 7, 1280), (1, 4)]  — [conv_maps, predictions]
    """
    base_model = cnn_model.get_layer("efficientnetb0")
    conv_layer = base_model.get_layer(LAST_CONV_LAYER)

    # Dual-output backbone: raw input → [conv_maps, final backbone features]
    backbone_dual = tf.keras.Model(
        inputs=base_model.input,
        outputs=[conv_layer.output, base_model.output],
        name="backbone_dual",
    )

    img_input = cnn_model.input
    # Disable augmentation during Grad-CAM (training=False → no randomness)
    aug_out = cnn_model.get_layer("data_augmentation")(img_input, training=False)
    conv_maps_out, bb_out = backbone_dual(aug_out, training=False)

    # Rebuild head using the same layer references (shared weights)
    x = cnn_model.get_layer("gap")(bb_out)
    x = cnn_model.get_layer("dropout_1")(x, training=False)
    x = cnn_model.get_layer("dense_256")(x)
    x = cnn_model.get_layer("dropout_2")(x, training=False)
    preds_out = cnn_model.get_layer("classifier")(x)

    gradcam_model = tf.keras.Model(
        inputs=img_input,
        outputs=[conv_maps_out, preds_out],
        name="gradcam_model",
    )
    return gradcam_model


def compute_gradcam(
    gradcam_model: keras.Model,
    img_array: np.ndarray,
    class_idx: int,
) -> np.ndarray:
    """
    Compute a Grad-CAM heatmap for the given class.

    Args:
        gradcam_model: Dual-output model from build_gradcam_model().
        img_array    : float32 array of shape (1, 224, 224, 3), values 0–255.
        class_idx    : Index of the class to explain (0–3).

    Returns:
        np.ndarray of shape (7, 7), values in [0, 1].
        If heatmap max is 0 (degenerate case), returns zero array.
    """
    img_tensor = tf.cast(img_array, tf.float32)

    with tf.GradientTape(persistent=True) as tape:
        conv_maps, preds = gradcam_model(img_tensor, training=False)
        tape.watch(conv_maps)          # explicit watch — required for correct gradients
        class_score = preds[0, class_idx]

    grads = tape.gradient(class_score, conv_maps)  # (1, 7, 7, 1280)

    if grads is None:
        raise RuntimeError(
            "Grad-CAM gradient is None. Ensure gradcam_model was built with "
            "build_gradcam_model() so both outputs share a single forward pass."
        )

    weights = tf.reduce_mean(grads, axis=(0, 1, 2))    # (1280,)
    maps = conv_maps[0]                                  # (7, 7, 1280)
    heatmap = tf.nn.relu(maps @ weights[..., tf.newaxis])  # (7, 7, 1)
    heatmap = tf.squeeze(heatmap).numpy()

    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    return heatmap.astype(np.float32)


def overlay_gradcam(
    raw_img_array: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Blend the Grad-CAM heatmap onto the original image.

    Args:
        raw_img_array: uint8 or float32 array of shape (224, 224, 3).
        heatmap      : float32 array of shape (7, 7), values in [0, 1].
        alpha        : Blend weight for the heatmap overlay.

    Returns:
        uint8 array of shape (224, 224, 3) — jet-coloured overlay.
    """
    hm_pil = Image.fromarray(np.uint8(255 * heatmap), mode="L")
    # Pillow >= 9.1 uses Image.Resampling.BILINEAR; older uses Image.BILINEAR
    try:
        _bilinear = Image.Resampling.BILINEAR
    except AttributeError:
        _bilinear = Image.BILINEAR  # type: ignore[attr-defined]
    hm_big = np.array(hm_pil.resize((224, 224), _bilinear)) / 255.0
    # matplotlib >= 3.7: get_cmap is deprecated — use matplotlib.colormaps instead
    try:
        import matplotlib
        jet = matplotlib.colormaps["jet"]
    except AttributeError:
        jet = plt.cm.get_cmap("jet")   # fallback for older matplotlib
    colored = np.uint8(255 * jet(hm_big)[:, :, :3])
    base = np.array(raw_img_array, dtype=np.float32)
    return np.uint8(base * (1 - alpha) + colored * alpha)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Weather Sequence Builder
# ─────────────────────────────────────────────────────────────────────────────

# 20 feature columns — must match the order used during training
FEATURE_COLS = [
    "weather_code",
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "rain_sum",
    "precipitation_hours",
    "et0_fao_evapotranspiration",
    "sunshine_duration",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "relative_humidity_2m_mean",
    "pressure_msl_mean",
    "soil_moisture_0_to_7cm_mean",
    "month_sin",
    "month_cos",
    "day_sin",
    "day_cos",
    "season",
    "precip_7d_sum",
    "district_id_norm",   # H3 FIX: district_id / 5.0
]

LOOKBACK = 14


def engineer_weather_features(df) -> object:
    """
    Add cyclical time, season, rolling, and district_id_norm features to a raw
    weather DataFrame (mirrors the preprocessing notebook Steps 4–6).

    Args:
        df: pandas DataFrame with columns matching WEATHER_VARS and a 'date'
            column (datetime), plus 'district_id' (int, 1–5).

    Returns:
        The same DataFrame with additional feature columns added in-place.
    """
    import pandas as pd

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Cyclical time features
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["day_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["day_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
    df.drop(columns=["month", "day_of_year"], inplace=True)

    # Tamil Nadu season (matches training notebook Step 5)
    def _get_season(month: int) -> int:
        if month in [6, 7, 8, 9]:
            return 0   # SW Monsoon
        if month in [10, 11, 12]:
            return 1   # NE Monsoon (peak disease risk)
        if month in [1, 2]:
            return 2   # Winter / dry
        return 3        # Summer

    df["season"] = df["date"].dt.month.apply(_get_season)

    # District ID normalisation (H3 FIX)
    df["district_id_norm"] = df["district_id"] / 5.0

    # 7-day rolling precipitation sum
    df["precip_7d_sum"] = df["precipitation_sum"].rolling(window=7, min_periods=1).sum()

    return df


def build_lstm_sequence(
    df,
    scaler: MinMaxScaler,
    district_id: int | None = None,
) -> np.ndarray:
    """
    Build a scaled 14-day LSTM input sequence from a preprocessed weather DataFrame.

    The last 14 rows of `df` are used as the sequence.  Pass only rows from a
    single district to avoid cross-district data leakage.

    Args:
        df         : pandas DataFrame (at least 14 rows) with all FEATURE_COLS.
                     Call engineer_weather_features() first if raw weather data.
        scaler     : Fitted MinMaxScaler loaded from scaler.pkl.
        district_id: If provided, filter df to only this district.

    Returns:
        float32 array of shape (14, 20) — scaled and ready for LSTM inference.

    Raises:
        ValueError: If fewer than 14 rows are available after optional filtering.
    """
    if district_id is not None:
        df = df[df["district_id"] == district_id].reset_index(drop=True)

    if len(df) < LOOKBACK:
        raise ValueError(
            f"Need at least {LOOKBACK} days of weather data, got {len(df)}."
        )

    window = df[FEATURE_COLS].tail(LOOKBACK).values          # (14, 20) raw
    scaled = scaler.transform(window)                          # (14, 20) scaled
    return scaled.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Config Loader
# ─────────────────────────────────────────────────────────────────────────────
def load_config(config_path: str | Path = "config.yaml") -> dict:
    """
    Load the project configuration from config.yaml.

    Args:
        config_path: Path to config.yaml (default: project root).

    Returns:
        Nested dict matching the YAML structure.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Risk Band Interpreter
# ─────────────────────────────────────────────────────────────────────────────
def interpret_risk(fused_score: float) -> dict:
    """
    Map a fused risk score to a human-readable risk band.

    Bands (from config.yaml):
        Low      : [0.00, 0.35)
        Moderate : [0.35, 0.65)
        High     : [0.65, 1.00]

    Args:
        fused_score: float in [0, 1] from prediction_service.py.

    Returns:
        dict with keys:
            score      (float) : raw fused score
            band       (str)   : "Low" / "Moderate" / "High"
            message    (str)   : short advisory message
            emoji      (str)   : visual indicator for UI
    """
    if fused_score < 0.35:
        return {
            "score": round(fused_score, 4),
            "band": "Low",
            "message": "Conditions are currently favourable. Routine monitoring recommended.",
            "emoji": "🟢",
        }
    elif fused_score < 0.65:
        return {
            "score": round(fused_score, 4),
            "band": "Moderate",
            "message": "Moderate disease risk detected. Inspect crops and consider preventive treatment.",
            "emoji": "🟡",
        }
    else:
        return {
            "score": round(fused_score, 4),
            "band": "High",
            "message": "High disease risk. Immediate field inspection and treatment advised.",
            "emoji": "🔴",
        }
