"""
app/api/routes.py
==================
FastAPI APIRouter — all REST prediction endpoints.

Endpoints
---------
GET  /health                     → liveness check
POST /api/v1/predict/image       → CNN disease classification
POST /api/v1/predict/gradcam     → CNN + Grad-CAM heatmap overlay (base64 PNG)
POST /api/v1/predict/weather     → LSTM weather risk (JSON weather payload)
POST /api/v1/predict/fused       → Full CNN + LSTM fusion (image + district)

All endpoints share the single PredictionService instance injected via
FastAPI's dependency system from app state (loaded once at startup in main.py).

Image endpoints accept multipart/form-data file uploads.
Weather endpoint accepts a JSON body with a 14-row weather records array.
Fused endpoint accepts multipart/form-data (image file + district_id field).
"""

from __future__ import annotations

import base64
import io
import json
from typing import Annotated, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from PIL import Image
from pydantic import BaseModel, Field

router = APIRouter()


# ─────────────────────────────────────────────────────────────────────────────
# Dependency: get the shared PredictionService from app state
# ─────────────────────────────────────────────────────────────────────────────
def get_service(request: Request):
    """
    Retrieve the PredictionService loaded at startup.
    Raises 503 if models are not ready (e.g. trained_models/ is empty).
    """
    service = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Models not loaded. Place trained model files in trained_models/ "
                "and restart the server."
            ),
        )
    return service


ServiceDep = Annotated[object, Depends(get_service)]


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic schemas
# ─────────────────────────────────────────────────────────────────────────────
class WeatherRecord(BaseModel):
    """One row of raw daily weather data (matches Open-Meteo archive variables)."""

    date: str = Field(..., example="2024-11-01")
    district_id: int = Field(..., ge=1, le=5, example=1)
    weather_code: float = Field(..., example=61.0)
    temperature_2m_max: float = Field(..., example=32.5)
    temperature_2m_min: float = Field(..., example=24.1)
    precipitation_sum: float = Field(..., example=12.3)
    rain_sum: float = Field(..., example=11.0)
    precipitation_hours: float = Field(..., example=4.0)
    et0_fao_evapotranspiration: float = Field(..., example=4.2)
    sunshine_duration: float = Field(..., example=28000.0)
    wind_speed_10m_max: float = Field(..., example=18.0)
    wind_gusts_10m_max: float = Field(..., example=32.0)
    relative_humidity_2m_mean: float = Field(..., example=88.0)
    pressure_msl_mean: float = Field(..., example=1010.5)
    soil_moisture_0_to_7cm_mean: float = Field(..., example=0.25)


class WeatherPayload(BaseModel):
    """14 consecutive daily weather records for one district."""

    records: list[WeatherRecord] = Field(
        ...,
        description="Exactly 14 days of weather data, ordered oldest → newest.",
    )

    # Use model_validator (Pydantic v2) with v1 fallback for length check.
    # Avoids using min_length/max_length on Field which behaves differently
    # between Pydantic v1 (min_items/max_items) and v2 (min_length/max_length).
    @classmethod
    def __get_validators__(cls):
        yield cls._validate

    @classmethod
    def _validate(cls, v):
        return v

    def model_post_init(self, __context):  # Pydantic v2
        if len(self.records) != 14:
            raise ValueError(f"Exactly 14 weather records required, got {len(self.records)}.")

    def __init__(self, **data):
        super().__init__(**data)
        if len(self.records) != 14:
            raise ValueError(f"Exactly 14 weather records required, got {len(self.records)}.")


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    cnn_classes: list[str]
    fusion_alpha: float


class ProbabilityEntry(BaseModel):
    class_name: str
    probability: float


class CNNResponse(BaseModel):
    predicted_class: str
    predicted_index: int
    confidence: float
    probabilities: list[ProbabilityEntry]
    cnn_risk_score: float


class GradCAMResponse(BaseModel):
    predicted_class: str
    confidence: float
    cnn_risk_score: float
    overlay_png_b64: str   # base64-encoded PNG of the 224×224 overlay image


class WeatherRiskResponse(BaseModel):
    lstm_risk_score: float
    risk_band: str
    risk_message: str
    risk_emoji: str


class FusedResponse(BaseModel):
    predicted_class: str
    predicted_index: int
    confidence: float
    probabilities: list[ProbabilityEntry]
    cnn_risk_score: float
    lstm_risk_score: float
    fused_score: float
    alpha_used: float
    risk_band: str
    risk_message: str
    risk_emoji: str
    overlay_png_b64: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _probs_to_list(probs_dict: dict) -> list[ProbabilityEntry]:
    return [
        ProbabilityEntry(class_name=k, probability=round(v, 6))
        for k, v in probs_dict.items()
    ]


def _ndarray_to_png_b64(arr: np.ndarray) -> str:
    """Encode a uint8 (H, W, 3) numpy array as a base64 PNG string."""
    img = Image.fromarray(arr.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


async def _read_image_bytes(file: UploadFile) -> bytes:
    """Read and validate an uploaded image file."""
    allowed = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
    if file.content_type not in allowed:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}'. Upload JPEG or PNG.",
        )
    return await file.read()


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────
@router.get("/health", response_model=HealthResponse, tags=["Utility"])
def health_check(request: Request):
    """
    Liveness + readiness check.
    Returns model load status, class names, and current fusion alpha.
    """
    service = getattr(request.app.state, "service", None)
    loaded = service is not None
    return HealthResponse(
        status="ok",
        models_loaded=loaded,
        cnn_classes=list(service.config["cnn"]["class_names"]) if loaded else [],
        fusion_alpha=float(service.config["fusion"]["alpha"]) if loaded else 0.0,
    )


@router.post(
    "/api/v1/predict/image",
    response_model=CNNResponse,
    tags=["Prediction"],
    summary="CNN-only disease classification",
)
async def predict_image(
    service: ServiceDep,
    file: UploadFile = File(..., description="Rice leaf image (JPEG or PNG)"),
):
    """
    Classify a rice leaf image using the EfficientNetB0 CNN.

    Returns the predicted disease class, softmax probabilities for all 4 classes,
    and the CNN risk score (1 − P(Healthy)) used in fusion.

    No confidence threshold is applied — prediction is always argmax of softmax.
    """
    try:
        image_bytes = await _read_image_bytes(file)
        result = service.predict_from_image(image_bytes, include_gradcam=False)
        probs = result["probabilities"]
        max_prob = max(probs.values())
        return CNNResponse(
            predicted_class=result["predicted_class"],
            predicted_index=result["predicted_index"],
            confidence=round(max_prob, 6),
            probabilities=_probs_to_list(probs),
            cnn_risk_score=result["cnn_risk_score"],
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc


@router.post(
    "/api/v1/predict/gradcam",
    response_model=GradCAMResponse,
    tags=["Prediction"],
    summary="CNN prediction + Grad-CAM heatmap overlay",
)
async def predict_gradcam(
    service: ServiceDep,
    file: UploadFile = File(..., description="Rice leaf image (JPEG or PNG)"),
):
    """
    Classify a rice leaf image and return a Grad-CAM heatmap overlay.

    The overlay is a 224×224 RGB image (jet colormap blended at alpha=0.45)
    returned as a base64-encoded PNG string.

    Grad-CAM implementation: Selvaraju et al. 2017 (single dual-output forward pass).
    The predicted class is used as the Grad-CAM target class.
    """
    try:
        image_bytes = await _read_image_bytes(file)
        result = service.predict_from_image(image_bytes, include_gradcam=True)
        probs = result["probabilities"]
        max_prob = max(probs.values())
        overlay_b64 = _ndarray_to_png_b64(result["gradcam_overlay"])
        return GradCAMResponse(
            predicted_class=result["predicted_class"],
            confidence=round(max_prob, 6),
            cnn_risk_score=result["cnn_risk_score"],
            overlay_png_b64=overlay_b64,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Grad-CAM generation failed: {exc}"
        ) from exc


@router.post(
    "/api/v1/predict/weather",
    response_model=WeatherRiskResponse,
    tags=["Prediction"],
    summary="LSTM weather-driven risk prediction",
)
async def predict_weather(
    service: ServiceDep,
    payload: WeatherPayload,
):
    """
    Predict disease outbreak risk from 14 consecutive days of weather data.

    Accepts a JSON body with exactly 14 weather records (oldest → newest).
    The LSTM returns a sigmoid probability in [0, 1].
    No hard threshold is applied — the raw probability is returned.

    Risk bands:
      Low      [0.00 – 0.35) : routine monitoring
      Moderate [0.35 – 0.65) : consider preventive treatment
      High     [0.65 – 1.00] : immediate action advised
    """
    try:
        df = pd.DataFrame([r.model_dump() for r in payload.records])
        df["date"] = pd.to_datetime(df["date"])
        district_id = int(df["district_id"].iloc[0])
        result = service.predict_from_weather(
            df, district_id=district_id, apply_feature_engineering=True,
        )
        band = result["risk_band"]
        return WeatherRiskResponse(
            lstm_risk_score=result["lstm_risk_score"],
            risk_band=band["band"],
            risk_message=band["message"],
            risk_emoji=band["emoji"],
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Weather prediction failed: {exc}"
        ) from exc


@router.post(
    "/api/v1/predict/fused",
    response_model=FusedResponse,
    tags=["Prediction"],
    summary="Full fusion: CNN image + LSTM weather → combined risk",
)
async def predict_fused(
    service: ServiceDep,
    file: UploadFile = File(..., description="Rice leaf image (JPEG or PNG)"),
    weather_json: str = Form(
        ...,
        description="JSON string — array of 14 weather record objects.",
    ),
    district_id: int = Form(..., ge=1, le=5, description="District ID (1–5)"),
    include_gradcam: bool = Form(False, description="Include Grad-CAM overlay in response"),
):
    """
    Full pipeline: classify leaf image (CNN) + assess weather risk (LSTM) →
    combine into a single fused risk score.

    Fusion formula:
        fused = alpha × cnn_risk + (1−alpha) × lstm_risk
    where alpha is set in config.yaml (validated via ablation study).

    Submit as multipart/form-data:
      - file          : image file
      - weather_json  : JSON string of 14 weather records
      - district_id   : int (1–5)
      - include_gradcam : bool (optional, default False)
    """
    try:
        image_bytes = await _read_image_bytes(file)
        records_raw = json.loads(weather_json)
        records = [WeatherRecord(**r) for r in records_raw]
        df = pd.DataFrame([r.model_dump() for r in records])
        df["date"] = pd.to_datetime(df["date"])

        result = service.predict_fused(
            image=image_bytes,
            weather_df=df,
            district_id=district_id,
            apply_feature_engineering=True,
            include_gradcam=include_gradcam,
        )

        probs = result["probabilities"]
        max_prob = max(probs.values())
        band = result["risk_band"]
        overlay_b64 = None
        if include_gradcam and result.get("gradcam_overlay") is not None:
            overlay_b64 = _ndarray_to_png_b64(result["gradcam_overlay"])

        return FusedResponse(
            predicted_class=result["predicted_class"],
            predicted_index=result["predicted_index"],
            confidence=round(max_prob, 6),
            probabilities=_probs_to_list(probs),
            cnn_risk_score=result["cnn_risk_score"],
            lstm_risk_score=result["lstm_risk_score"],
            fused_score=result["fused_score"],
            alpha_used=result["alpha_used"],
            risk_band=band["band"],
            risk_message=band["message"],
            risk_emoji=band["emoji"],
            overlay_png_b64=overlay_b64,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Fused prediction failed: {exc}"
        ) from exc
