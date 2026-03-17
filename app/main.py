"""
app/main.py
============
Entry point for the Rice Disease Detection application.

What this file does
-------------------
1. Creates a FastAPI app with a lifespan context that loads all models once
   at startup and stores them in app.state.service.

2. Mounts all REST API routes from app/api/routes.py.

3. Builds a 4-tab Gradio UI and mounts it at the root path "/" so that
   Hugging Face Spaces shows the UI directly when the Space is opened.

4. Also serves full FastAPI REST docs at /docs (Swagger) and /redoc.

Tabs in the Gradio UI
---------------------
  Tab 1 — 🌿 Disease Classifier   : upload leaf image → CNN prediction + probability chart
  Tab 2 — 🌦️ Weather Risk         : select district → auto-fetch 14 days → LSTM score
  Tab 3 — 🔬 Grad-CAM             : upload leaf image → original + heatmap overlay
  Tab 4 — 🔗 Full Fusion          : image + district → CNN + LSTM + fused risk

Weather auto-fetch
------------------
Tabs 2 and 4 call the Open-Meteo historical archive API for the last 14 completed
days for the chosen Tamil Nadu district. This avoids asking users to enter 20
raw weather values manually.

Hugging Face Spaces
-------------------
  - Docker SDK Space: port 7860 (set in Dockerfile CMD)
  - The Gradio UI is at "/"  — this is what Spaces renders
  - REST API is at "/api/v1/..."  — available for external tools/curl

Running locally
---------------
    uvicorn app.main:app --reload --port 7860
"""

from __future__ import annotations

import io
import warnings
from contextlib import asynccontextmanager
from datetime import date, timedelta
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from app.api.routes import router
from app.services.prediction_service import PredictionService
from app.utils.helper import load_config

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
CONFIG = load_config("config.yaml")

CLASS_NAMES: list[str] = CONFIG["cnn"]["class_names"]

DISTRICTS: list[dict] = CONFIG["lstm"]["districts"]
DISTRICT_NAMES = [d["name"] for d in DISTRICTS]
DISTRICT_MAP = {d["name"]: d for d in DISTRICTS}   # name → {lat, lon, district_id}

WEATHER_VARS = [
    "weather_code", "temperature_2m_max", "temperature_2m_min",
    "precipitation_sum", "rain_sum", "precipitation_hours",
    "et0_fao_evapotranspiration", "sunshine_duration",
    "wind_speed_10m_max", "wind_gusts_10m_max",
    "relative_humidity_2m_mean", "pressure_msl_mean",
    "soil_moisture_0_to_7cm_mean",
]

# Disease descriptions for UI display
DISEASE_INFO = {
    "Bacterial Leaf Blight": (
        "Caused by Xanthomonas oryzae pv. oryzae. "
        "Appears as water-soaked lesions along leaf margins that turn yellow then brown. "
        "Spreads rapidly in warm, humid, windy conditions."
    ),
    "Brown Spot": (
        "Caused by Bipolaris oryzae. "
        "Oval brown lesions with a grey centre on leaves. "
        "Associated with nutrient deficiency and prolonged leaf wetness."
    ),
    "Healthy Rice Leaf": (
        "No disease detected. The leaf shows normal green colouration "
        "with no visible lesions or discolouration."
    ),
    "Leaf Blast": (
        "Caused by Magnaporthe oryzae. "
        "Diamond-shaped lesions with brown borders. "
        "Most severe in cool, cloudy, humid conditions with night dew."
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Weather auto-fetch helper (Open-Meteo archive API)
# ─────────────────────────────────────────────────────────────────────────────
def fetch_last_14_days(district_name: str) -> pd.DataFrame | None:
    """
    Fetch the last 14 completed days of weather data for the given district
    from the Open-Meteo historical archive API.

    Args:
        district_name: One of the 5 Tamil Nadu districts in config.yaml.

    Returns:
        pandas DataFrame with 14 rows and all raw weather columns,
        plus 'district_id' column.  Returns None on network failure.
    """
    dist = DISTRICT_MAP[district_name]
    end_date = date.today() - timedelta(days=1)        # yesterday = last complete day
    start_date = end_date - timedelta(days=13)          # 14 days total

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": dist["lat"],
        "longitude": dist["lon"],
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "daily": ",".join(WEATHER_VARS),
        "timezone": "Asia/Kolkata",
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        daily = data.get("daily", {})

        df = pd.DataFrame({"date": pd.to_datetime(daily["time"])})
        for var in WEATHER_VARS:
            df[var] = daily.get(var, [None] * len(df))

        df["district_id"] = dist["district_id"]
        df["district_name"] = district_name
        return df

    except Exception as exc:
        print(f"[weather_fetch] Failed for {district_name}: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI lifespan — load models once at startup
# ─────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load PredictionService at startup; release on shutdown."""
    print("[startup] Loading models...")
    try:
        app.state.service = PredictionService.from_config("config.yaml")
        print("[startup] All models loaded successfully.")
    except FileNotFoundError as exc:
        print(f"[startup] WARNING: {exc}")
        print("[startup] App will start but predictions will fail until models are placed in trained_models/.")
        app.state.service = None
    yield
    print("[shutdown] Releasing resources.")
    app.state.service = None


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Rice Disease Detection API",
    description=(
        "EfficientNetB0 CNN + WeatherRiskLSTM hybrid system for rice leaf disease "
        "classification and weather-driven risk prediction. "
        "Anna University Final Year Project 2024–25."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


# ─────────────────────────────────────────────────────────────────────────────
# Gradio helper: get service from FastAPI state
# ─────────────────────────────────────────────────────────────────────────────
def _get_service():
    service = getattr(app.state, "service", None)
    if service is None:
        raise gr.Error(
            "Models not loaded. Copy trained model files into trained_models/ "
            "and restart the app."
        )
    return service


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — Disease Classifier
# ─────────────────────────────────────────────────────────────────────────────
def tab1_classify(pil_image: Image.Image):
    """
    CNN-only disease classification.
    Returns: (result_markdown, bar_chart_dict)
    """
    if pil_image is None:
        return "⚠️ Please upload a leaf image.", None

    try:
        service = _get_service()
        result = service.predict_from_image(pil_image, include_gradcam=False)

        cls = result["predicted_class"]
        idx = result["predicted_index"]
        probs = result["probabilities"]
        confidence = probs[cls] * 100
        cnn_risk = result["cnn_risk_score"]
        info = DISEASE_INFO.get(cls, "")

        # Markdown summary
        icon = "🟢" if cls == "Healthy Rice Leaf" else "🔴"
        md = (
            f"## {icon} {cls}\n\n"
            f"**Confidence:** {confidence:.1f}%  |  "
            f"**CNN Risk Score:** {cnn_risk:.4f}\n\n"
            f"_{info}_\n\n"
            f"---\n"
            f"*Prediction uses argmax of softmax probabilities. No confidence threshold applied.*"
        )

        # Gradio Label dict format: {class_name: probability}
        label_dict = {k: float(v) for k, v in probs.items()}

        return md, label_dict

    except gr.Error:
        raise
    except Exception as exc:
        return f"❌ Error: {exc}", None


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 — Weather Risk (LSTM)
# ─────────────────────────────────────────────────────────────────────────────
def tab2_weather_risk(district_name: str):
    """
    Fetch last 14 days of weather for the district → LSTM risk prediction.
    Returns: (status_md, risk_md, weather_preview_df)
    """
    try:
        service = _get_service()

        status_md = f"⏳ Fetching weather for **{district_name}**..."

        df = fetch_last_14_days(district_name)
        if df is None:
            return (
                "❌ Weather fetch failed. Check your internet connection.",
                "",
                None,
            )

        dist = DISTRICT_MAP[district_name]
        date_range = f"{df['date'].min().date()} → {df['date'].max().date()}"
        status_md = (
            f"✅ Weather fetched for **{district_name}** "
            f"(lat {dist['lat']}°N, lon {dist['lon']}°E)\n\n"
            f"📅 Date range: {date_range}"
        )

        result = service.predict_from_weather(
            df, district_id=dist["district_id"], apply_feature_engineering=True
        )
        band = result["risk_band"]
        score = result["lstm_risk_score"]

        risk_md = (
            f"## {band['emoji']} {band['band']} Risk\n\n"
            f"**LSTM Risk Score:** `{score:.4f}` / 1.0\n\n"
            f"**Advisory:** {band['message']}\n\n"
            f"---\n"
            f"*Score is raw sigmoid output. No hard threshold applied.*"
        )

        # Preview table — show 3 key columns + last 5 rows
        preview = df[["date", "temperature_2m_max", "relative_humidity_2m_mean",
                       "precipitation_sum"]].tail(5).copy()
        preview.columns = ["Date", "Temp Max (°C)", "Humidity (%)", "Precip (mm)"]
        preview["Date"] = preview["Date"].dt.strftime("%Y-%m-%d")

        return status_md, risk_md, preview

    except gr.Error:
        raise
    except Exception as exc:
        return f"❌ Error: {exc}", "", None


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3 — Grad-CAM
# ─────────────────────────────────────────────────────────────────────────────
def tab3_gradcam(pil_image: Image.Image):
    """
    CNN prediction + Grad-CAM heatmap overlay.
    Returns: (info_md, original_image, overlay_image)
    """
    if pil_image is None:
        return "⚠️ Please upload a leaf image.", None, None

    try:
        service = _get_service()
        result = service.predict_from_image(pil_image, include_gradcam=True)

        cls = result["predicted_class"]
        probs = result["probabilities"]
        confidence = probs[cls] * 100

        info_md = (
            f"### Predicted: **{cls}** ({confidence:.1f}%)\n\n"
            f"The red/yellow regions in the overlay show which leaf areas the model "
            f"focused on most to make this prediction.\n\n"
            f"*Grad-CAM: Selvaraju et al. 2017 — single dual-output forward pass.*"
        )

        # Resize original to 224×224 for consistent display side-by-side
        original_resized = pil_image.convert("RGB").resize((224, 224))

        overlay_arr = result["gradcam_overlay"]    # uint8 (224, 224, 3)
        overlay_pil = Image.fromarray(overlay_arr)

        return info_md, original_resized, overlay_pil

    except gr.Error:
        raise
    except Exception as exc:
        return f"❌ Error: {exc}", None, None


# ─────────────────────────────────────────────────────────────────────────────
# Tab 4 — Full Fusion
# ─────────────────────────────────────────────────────────────────────────────
def tab4_fused(pil_image: Image.Image, district_name: str, include_gradcam: bool):
    """
    CNN leaf classification + LSTM weather risk → fused prediction.
    Returns: (cnn_md, weather_md, fusion_md, overlay_image_or_None)
    """
    if pil_image is None:
        return "⚠️ Please upload a leaf image.", "", "", None

    try:
        service = _get_service()

        # Fetch weather
        df = fetch_last_14_days(district_name)
        if df is None:
            return (
                "⚠️ Weather fetch failed — showing CNN result only.",
                "❌ Could not fetch weather data.",
                "",
                None,
            )

        dist = DISTRICT_MAP[district_name]
        result = service.predict_fused(
            image=pil_image,
            weather_df=df,
            district_id=dist["district_id"],
            apply_feature_engineering=True,
            include_gradcam=include_gradcam,
        )

        # CNN section
        cls = result["predicted_class"]
        probs = result["probabilities"]
        confidence = probs[cls] * 100
        icon = "🟢" if cls == "Healthy Rice Leaf" else "🔴"
        cnn_md = (
            f"### {icon} {cls}\n"
            f"Confidence: **{confidence:.1f}%**  |  "
            f"CNN Risk Score: `{result['cnn_risk_score']:.4f}`"
        )

        # LSTM section
        weather_md = (
            f"### {district_name} Weather (last 14 days)\n"
            f"LSTM Risk Score: `{result['lstm_risk_score']:.4f}`"
        )

        # Fusion section
        band = result["risk_band"]
        fused = result["fused_score"]
        alpha = result["alpha_used"]
        # Build a simple text progress bar
        filled = int(fused * 20)
        bar = "█" * filled + "░" * (20 - filled)
        fusion_md = (
            f"## {band['emoji']} Fused Risk: {band['band']}\n\n"
            f"`{bar}` **{fused:.4f}**\n\n"
            f"**{band['message']}**\n\n"
            f"---\n"
            f"Formula: `fused = {alpha} × CNN_risk + {1-alpha:.1f} × LSTM_risk`  "
            f"= `{alpha} × {result['cnn_risk_score']:.4f} + "
            f"{1-alpha:.1f} × {result['lstm_risk_score']:.4f}`"
        )

        overlay_out = None
        if include_gradcam and result.get("gradcam_overlay") is not None:
            overlay_out = Image.fromarray(result["gradcam_overlay"])

        return cnn_md, weather_md, fusion_md, overlay_out

    except gr.Error:
        raise
    except Exception as exc:
        return f"❌ Error: {exc}", "", "", None


# ─────────────────────────────────────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────────────────────────────────────
_CSS = """
.tab-title { font-size: 1.1em; font-weight: 600; }
.risk-box   { border-radius: 8px; padding: 12px; }
footer { display: none !important; }
"""

with gr.Blocks(
    title="Rice Disease Detection",
    theme=gr.themes.Soft(primary_hue="green", secondary_hue="emerald"),
    css=_CSS,
) as demo:

    gr.Markdown(
        """
        # 🌾 Rice Disease Detection System
        **EfficientNetB0 CNN + WeatherRiskLSTM | Anna University Final Year Project 2024–25**

        Upload a rice leaf image and/or select your district to detect disease
        and assess weather-driven outbreak risk.
        """
    )

    # ── Tab 1 — Disease Classifier ──────────────────────────────────────────
    with gr.Tab("🌿 Disease Classifier"):
        gr.Markdown("### Upload a rice leaf image to classify the disease.")
        with gr.Row():
            with gr.Column(scale=1):
                t1_image = gr.Image(
                    type="pil", label="Upload Leaf Image", height=280
                )
                t1_btn = gr.Button("🔍 Classify", variant="primary", size="lg")
            with gr.Column(scale=1):
                t1_result = gr.Markdown(label="Prediction")
                t1_probs = gr.Label(
                    num_top_classes=4, label="Class Probabilities"
                )

        t1_btn.click(
            fn=tab1_classify,
            inputs=[t1_image],
            outputs=[t1_result, t1_probs],
        )

        gr.Markdown(
            """
            **Classes:** Bacterial Leaf Blight · Brown Spot · Healthy Rice Leaf · Leaf Blast

            *Model: EfficientNetB0 · Loss: CategoricalFocalCrossentropy (γ=2.0) · 
            No confidence threshold — argmax only.*
            """
        )

    # ── Tab 2 — Weather Risk ────────────────────────────────────────────────
    with gr.Tab("🌦️ Weather Risk"):
        gr.Markdown(
            "### Select your district to auto-fetch the last 14 days of weather "
            "and assess disease outbreak risk."
        )
        with gr.Row():
            with gr.Column(scale=1):
                t2_district = gr.Dropdown(
                    choices=DISTRICT_NAMES,
                    value=DISTRICT_NAMES[0],
                    label="Tamil Nadu District",
                )
                t2_btn = gr.Button("🌧️ Fetch & Predict", variant="primary", size="lg")
                t2_status = gr.Markdown()
                t2_preview = gr.Dataframe(
                    label="Last 5 Days Weather Preview", interactive=False
                )
            with gr.Column(scale=1):
                t2_risk = gr.Markdown(label="LSTM Risk Assessment")

        t2_btn.click(
            fn=tab2_weather_risk,
            inputs=[t2_district],
            outputs=[t2_status, t2_risk, t2_preview],
        )

        gr.Markdown(
            """
            **Districts:** Thanjavur · Nilgiris · Chennai · Virudhunagar · Nagapattinam

            *Weather source: Open-Meteo Historical Archive API.
            LSTM input: 14-day window, 20 features.
            Output: raw sigmoid probability — no hard threshold.*
            """
        )

    # ── Tab 3 — Grad-CAM ───────────────────────────────────────────────────
    with gr.Tab("🔬 Grad-CAM"):
        gr.Markdown(
            "### Visualise which leaf regions the model uses to make its prediction."
        )
        with gr.Row():
            with gr.Column(scale=1):
                t3_image = gr.Image(
                    type="pil", label="Upload Leaf Image", height=280
                )
                t3_btn = gr.Button("🔬 Generate Grad-CAM", variant="primary", size="lg")
                t3_info = gr.Markdown()
            with gr.Column(scale=1):
                with gr.Row():
                    t3_orig = gr.Image(label="Original (224×224)", height=224)
                    t3_over = gr.Image(label="Grad-CAM Overlay", height=224)

        t3_btn.click(
            fn=tab3_gradcam,
            inputs=[t3_image],
            outputs=[t3_info, t3_orig, t3_over],
        )

        gr.Markdown(
            """
            *Grad-CAM (Selvaraju et al. 2017): single dual-output forward pass through
            EfficientNetB0. Last conv layer: `top_conv` (7×7×1280).
            Heatmap upsampled to 224×224. Jet colormap, alpha blend = 0.45.*
            """
        )

    # ── Tab 4 — Full Fusion ─────────────────────────────────────────────────
    with gr.Tab("🔗 Full Fusion"):
        gr.Markdown(
            "### Combine CNN leaf classification with LSTM weather risk "
            "for a single fused disease risk score."
        )
        with gr.Row():
            with gr.Column(scale=1):
                t4_image = gr.Image(
                    type="pil", label="Upload Leaf Image", height=250
                )
                t4_district = gr.Dropdown(
                    choices=DISTRICT_NAMES,
                    value=DISTRICT_NAMES[0],
                    label="Tamil Nadu District",
                )
                t4_gradcam = gr.Checkbox(
                    label="Include Grad-CAM overlay", value=False
                )
                t4_btn = gr.Button("⚡ Run Full Analysis", variant="primary", size="lg")

            with gr.Column(scale=2):
                t4_cnn_out = gr.Markdown(label="CNN Result")
                t4_wx_out  = gr.Markdown(label="Weather Risk")
                t4_fuse_out = gr.Markdown(label="Fused Risk")
                t4_overlay = gr.Image(label="Grad-CAM Overlay (optional)", height=224, visible=True)

        t4_btn.click(
            fn=tab4_fused,
            inputs=[t4_image, t4_district, t4_gradcam],
            outputs=[t4_cnn_out, t4_wx_out, t4_fuse_out, t4_overlay],
        )

        gr.Markdown(
            """
            **Fusion formula:** `fused = α × CNN_risk + (1−α) × LSTM_risk`

            where `CNN_risk = 1 − P(Healthy)` and `α` is validated via ablation
            grid search (see `notebooks/training_experiments.ipynb`).

            | Risk Band | Score Range | Action |
            |-----------|-------------|--------|
            | 🟢 Low      | 0.00 – 0.35 | Routine monitoring |
            | 🟡 Moderate | 0.35 – 0.65 | Consider preventive treatment |
            | 🔴 High     | 0.65 – 1.00 | Immediate inspection & treatment |
            """
        )

    gr.Markdown(
        """
        ---
        **Anna University B.E. CSE Final Year Project 2024–25** |
        EfficientNetB0 · WeatherRiskLSTM · Focal Loss · Grad-CAM |
        Weather data: [Open-Meteo](https://open-meteo.com)
        """
    )


# ─────────────────────────────────────────────────────────────────────────────
# Mount Gradio onto FastAPI at root "/"
# ─────────────────────────────────────────────────────────────────────────────
app = gr.mount_gradio_app(app, demo, path="/")
