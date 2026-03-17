"""
app/services/prediction_service.py
====================================
Prediction service — the central business logic of the application.

This module combines:
  1. CNN (EfficientNetB0) — visual disease classification from leaf images
  2. LSTM (WeatherRiskLSTM) — weather-driven binary risk score
  3. Fusion — validated weighted combination of both signals

Fusion formula (from Rice_Disease_Ablation_Study.ipynb):
    fused_score = alpha * cnn_risk + (1 - alpha) * lstm_sigmoid_score
    where:
        cnn_risk         = 1 - P(Healthy Rice Leaf)     [CNN softmax output]
        lstm_sigmoid_score = sigmoid(LSTM logit)         [weather risk signal]
        alpha            = validated via grid search on val set (config.yaml)

NO threshold is applied at this layer.
The raw fused score [0, 1] is returned and interpreted via interpret_risk().

Usage
-----
    from app.services.prediction_service import PredictionService

    # One-time setup at app startup
    service = PredictionService.from_config("config.yaml")

    # CNN-only prediction (image path or PIL image)
    result = service.predict_from_image("leaf.jpg")

    # LSTM-only prediction (14-day weather DataFrame)
    result = service.predict_from_weather(df, district_id=1)

    # Full fusion prediction
    result = service.predict_fused("leaf.jpg", weather_df, district_id=1)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from app.models.cnn_model import predict_single
from app.models.load_model import load_all_models
from app.models.lstm_model import predict_risk_score
from app.preprocessing.image_preprocess import (
    preprocess_image_bytes,
    preprocess_image_file,
    preprocess_pil_image,
)
from app.utils.helper import (
    build_gradcam_model,
    build_lstm_sequence,
    compute_gradcam,
    engineer_weather_features,
    interpret_risk,
    load_config,
    overlay_gradcam,
)


class PredictionService:
    """
    Singleton-style service that holds loaded models and exposes prediction methods.

    Attributes:
        cnn_model  : Loaded Keras EfficientNetB0 model.
        lstm_model : Loaded WeatherRiskLSTM (PyTorch).
        scaler     : Fitted MinMaxScaler for weather features.
        device     : torch.device in use.
        alpha      : Fusion weight for CNN signal (1-alpha for LSTM).
        config     : Full config.yaml dict.
    """

    def __init__(self, models: dict, alpha: float, config: dict) -> None:
        self.cnn_model = models["cnn"]
        self.lstm_model = models["lstm"]
        self.scaler = models["scaler"]
        self.device = models["device"]
        self.alpha = alpha
        self.config = config
        self._gradcam_model: Optional[object] = None   # lazy build

    # ─────────────────────────────────────────────────────────────────────────
    # Factory
    # ─────────────────────────────────────────────────────────────────────────
    @classmethod
    def from_config(cls, config_path: str | Path = "config.yaml") -> "PredictionService":
        """
        Initialise the service by loading all models referenced in config.yaml.

        Args:
            config_path: Path to config.yaml.

        Returns:
            Ready-to-use PredictionService instance.
        """
        config = load_config(config_path)
        models_dir = Path(config["paths"]["trained_models_dir"])

        cnn_path = models_dir / config["cnn"]["model_file"]
        lstm_path = models_dir / config["lstm"]["model_file"]
        scaler_path = models_dir / config["lstm"]["scaler_file"]

        models = load_all_models(cnn_path, lstm_path, scaler_path)
        alpha = float(config["fusion"]["alpha"])
        return cls(models=models, alpha=alpha, config=config)

    # ─────────────────────────────────────────────────────────────────────────
    # CNN Prediction
    # ─────────────────────────────────────────────────────────────────────────
    def predict_from_image(
        self,
        image: str | Path | bytes | Image.Image,
        include_gradcam: bool = False,
    ) -> dict:
        """
        Classify a rice leaf image using the CNN.

        Args:
            image         : One of:
                              - str / Path  → local file path
                              - bytes       → raw image bytes (from HTTP upload)
                              - PIL.Image   → Gradio / direct PIL object
            include_gradcam: If True, compute and attach Grad-CAM heatmap.

        Returns:
            dict with keys:
                predicted_class  (str)   : e.g. "Leaf Blast"
                predicted_index  (int)   : e.g. 3
                probabilities    (dict)  : {class_name: softmax_score}
                cnn_risk_score   (float) : 1 - P(Healthy), fusion input
                gradcam_heatmap  (np.ndarray | None) : shape (7,7) if requested
                gradcam_overlay  (np.ndarray | None) : shape (224,224,3) if requested
        """
        # Preprocess image
        if isinstance(image, (str, Path)):
            img_array = preprocess_image_file(image)
        elif isinstance(image, bytes):
            img_array = preprocess_image_bytes(image)
        elif isinstance(image, Image.Image):
            img_array = preprocess_pil_image(image)
        else:
            raise TypeError(
                f"Unsupported image type: {type(image)}. "
                "Pass a file path, bytes, or PIL.Image."
            )

        # CNN inference
        result = predict_single(self.cnn_model, img_array)

        # Grad-CAM (optional)
        if include_gradcam:
            if self._gradcam_model is None:
                self._gradcam_model = build_gradcam_model(self.cnn_model)
            heatmap = compute_gradcam(
                self._gradcam_model, img_array, result["predicted_index"]
            )
            raw_arr = img_array[0].astype(np.uint8)    # (224, 224, 3)
            alpha_cam = float(self.config["cnn"].get("gradcam_alpha", 0.45))
            overlay = overlay_gradcam(raw_arr, heatmap, alpha=alpha_cam)
            result["gradcam_heatmap"] = heatmap
            result["gradcam_overlay"] = overlay
        else:
            result["gradcam_heatmap"] = None
            result["gradcam_overlay"] = None

        return result

    # ─────────────────────────────────────────────────────────────────────────
    # LSTM Prediction
    # ─────────────────────────────────────────────────────────────────────────
    def predict_from_weather(
        self,
        weather_df,
        district_id: Optional[int] = None,
        apply_feature_engineering: bool = True,
    ) -> dict:
        """
        Predict disease risk from 14 days of raw weather data using the LSTM.

        Args:
            weather_df             : pandas DataFrame with at least 14 rows.
                                     Must have columns matching WEATHER_VARS
                                     (weather_code, temperature_2m_max, etc.)
            district_id            : Filter to this district (1–5) before building
                                     the sequence. None = use df as-is.
            apply_feature_engineering: If True, call engineer_weather_features()
                                       to add cyclical/season/rolling columns.
                                       Set False if df is already fully engineered.

        Returns:
            dict with keys:
                lstm_risk_score  (float) : sigmoid probability in [0, 1]
                risk_band        (dict)  : from interpret_risk() — band, message, emoji
        """
        if apply_feature_engineering:
            weather_df = engineer_weather_features(weather_df)

        sequence = build_lstm_sequence(weather_df, self.scaler, district_id=district_id)
        lstm_score = predict_risk_score(self.lstm_model, sequence, self.device)

        return {
            "lstm_risk_score": round(lstm_score, 4),
            "risk_band": interpret_risk(lstm_score),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Fused Prediction
    # ─────────────────────────────────────────────────────────────────────────
    def predict_fused(
        self,
        image: str | Path | bytes | Image.Image,
        weather_df,
        district_id: Optional[int] = None,
        apply_feature_engineering: bool = True,
        include_gradcam: bool = False,
    ) -> dict:
        """
        Full fusion prediction: CNN + LSTM → combined disease risk.

        Fusion formula:
            fused_score = alpha * cnn_risk_score + (1 - alpha) * lstm_risk_score
            where alpha is set in config.yaml (validated via ablation study).

        Args:
            image                  : Leaf image (file path, bytes, or PIL.Image).
            weather_df             : 14-day weather DataFrame for the field.
            district_id            : Filter weather_df to this district.
            apply_feature_engineering: Engineer features before LSTM inference.
            include_gradcam        : Attach Grad-CAM heatmap to result.

        Returns:
            dict with keys:
                --- CNN ---
                predicted_class  (str)
                predicted_index  (int)
                probabilities    (dict)
                cnn_risk_score   (float)
                gradcam_heatmap  (np.ndarray | None)
                gradcam_overlay  (np.ndarray | None)
                --- LSTM ---
                lstm_risk_score  (float)
                --- Fusion ---
                fused_score      (float)
                alpha_used       (float)
                risk_band        (dict)   : final risk interpretation
        """
        # CNN
        cnn_result = self.predict_from_image(image, include_gradcam=include_gradcam)

        # LSTM
        lstm_result = self.predict_from_weather(
            weather_df,
            district_id=district_id,
            apply_feature_engineering=apply_feature_engineering,
        )

        # Fusion
        cnn_risk = cnn_result["cnn_risk_score"]
        lstm_risk = lstm_result["lstm_risk_score"]
        fused_score = self.alpha * cnn_risk + (1.0 - self.alpha) * lstm_risk

        return {
            # CNN outputs
            "predicted_class": cnn_result["predicted_class"],
            "predicted_index": cnn_result["predicted_index"],
            "probabilities": cnn_result["probabilities"],
            "cnn_risk_score": round(cnn_risk, 4),
            "gradcam_heatmap": cnn_result["gradcam_heatmap"],
            "gradcam_overlay": cnn_result["gradcam_overlay"],
            # LSTM output
            "lstm_risk_score": lstm_result["lstm_risk_score"],
            # Fusion output
            "fused_score": round(float(fused_score), 4),
            "alpha_used": self.alpha,
            "risk_band": interpret_risk(float(fused_score)),
        }
