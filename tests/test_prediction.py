"""
tests/test_prediction.py
=========================
Unit and integration tests for the rice disease detection pipeline.

Test groups:
  1. CNN Model Tests     — architecture, preprocessing, inference shape
  2. LSTM Model Tests    — architecture, sequence shape, inference range
  3. Preprocessing Tests — image loading, resizing, dtype
  4. Helper Tests        — Grad-CAM build, risk band interpretation
  5. Fusion Tests        — fused score formula, alpha boundary values
  6. Config Tests        — config.yaml loads correctly and class order is valid

Run all tests:
    pytest tests/test_prediction.py -v

Run a specific group:
    pytest tests/test_prediction.py -v -k "CNN"
"""

import io
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

# ── Make sure project root is on sys.path ──────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.models.cnn_model import (
    CLASS_NAMES,
    HEALTHY_IDX,
    IMG_SIZE,
    NUM_CLASSES,
    build_cnn_model,
    predict_single,
    predict_batch,
)
from app.models.lstm_model import WeatherRiskLSTM, predict_risk_score, predict_risk_batch
from app.preprocessing.image_preprocess import (
    preprocess_image_bytes,
    preprocess_pil_image,
    preprocess_batch,
)
from app.utils.helper import (
    build_lstm_sequence,
    engineer_weather_features,
    interpret_risk,
    load_config,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def dummy_cnn_model():
    """Build an untrained CNN model for structural tests (fast, no weights)."""
    return build_cnn_model()


@pytest.fixture(scope="module")
def dummy_lstm_model():
    """Instantiate an untrained LSTM for structural tests."""
    model = WeatherRiskLSTM(input_size=20, hidden_size=64, num_layers=2, dropout=0.5)
    model.eval()
    return model


@pytest.fixture
def dummy_image_pil():
    """224×224 green PIL image — simulates a rice leaf photo."""
    return Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8) + 80)


@pytest.fixture
def dummy_image_bytes(dummy_image_pil):
    """JPEG-encoded bytes of the dummy PIL image."""
    buf = io.BytesIO()
    dummy_image_pil.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def dummy_img_array(dummy_image_pil):
    """Preprocessed image array — shape (1, 224, 224, 3), float32."""
    return preprocess_pil_image(dummy_image_pil)


@pytest.fixture
def dummy_lstm_sequence():
    """Random float32 sequence — shape (14, 20), values in [0, 1]."""
    np.random.seed(42)
    return np.random.rand(14, 20).astype(np.float32)


@pytest.fixture
def dummy_device():
    return torch.device("cpu")


@pytest.fixture
def dummy_weather_df():
    """Minimal weather DataFrame with all required raw columns (20 rows)."""
    import pandas as pd

    n = 20
    np.random.seed(42)
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n),
            "district_id": [1] * n,
            "weather_code": np.random.randint(0, 90, n),
            "temperature_2m_max": np.random.uniform(25, 35, n),
            "temperature_2m_min": np.random.uniform(18, 25, n),
            "precipitation_sum": np.random.uniform(0, 20, n),
            "rain_sum": np.random.uniform(0, 20, n),
            "precipitation_hours": np.random.uniform(0, 12, n),
            "et0_fao_evapotranspiration": np.random.uniform(2, 6, n),
            "sunshine_duration": np.random.uniform(10000, 50000, n),
            "wind_speed_10m_max": np.random.uniform(5, 30, n),
            "wind_gusts_10m_max": np.random.uniform(10, 50, n),
            "relative_humidity_2m_mean": np.random.uniform(60, 95, n),
            "pressure_msl_mean": np.random.uniform(1005, 1020, n),
            "soil_moisture_0_to_7cm_mean": np.random.uniform(0.1, 0.4, n),
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. CNN Model Tests
# ─────────────────────────────────────────────────────────────────────────────
class TestCNNModel:

    def test_class_count(self):
        """Model must have exactly 4 output classes."""
        assert NUM_CLASSES == 4
        assert len(CLASS_NAMES) == 4

    def test_class_order_is_alphabetical(self):
        """Class names must be in alphabetical order (TF dataset loading order)."""
        assert CLASS_NAMES == sorted(CLASS_NAMES), (
            "CLASS_NAMES must be alphabetically sorted to match TF dataset_from_directory."
        )

    def test_healthy_idx(self):
        """HEALTHY_IDX must be 2 — used in fusion formula."""
        assert CLASS_NAMES[HEALTHY_IDX] == "Healthy Rice Leaf"
        assert HEALTHY_IDX == 2

    def test_model_output_shape(self, dummy_cnn_model, dummy_img_array):
        """Model must output (1, 4) for a single image."""
        probs = dummy_cnn_model.predict(dummy_img_array, verbose=0)
        assert probs.shape == (1, NUM_CLASSES), f"Expected (1, 4), got {probs.shape}"

    def test_model_output_sums_to_one(self, dummy_cnn_model, dummy_img_array):
        """Softmax probabilities must sum to ~1.0."""
        probs = dummy_cnn_model.predict(dummy_img_array, verbose=0)
        assert abs(probs[0].sum() - 1.0) < 1e-5, f"Probabilities sum to {probs[0].sum()}"

    def test_predict_single_keys(self, dummy_cnn_model, dummy_img_array):
        """predict_single() must return all expected keys."""
        result = predict_single(dummy_cnn_model, dummy_img_array)
        for key in ["predicted_class", "predicted_index", "probabilities", "cnn_risk_score"]:
            assert key in result, f"Missing key: {key}"

    def test_cnn_risk_score_range(self, dummy_cnn_model, dummy_img_array):
        """cnn_risk_score = 1 - P(Healthy) must be in [0, 1]."""
        result = predict_single(dummy_cnn_model, dummy_img_array)
        assert 0.0 <= result["cnn_risk_score"] <= 1.0

    def test_predict_batch_shape(self, dummy_cnn_model, dummy_img_array):
        """predict_batch() on N images must return N result dicts."""
        batch = np.tile(dummy_img_array, (3, 1, 1, 1))   # (3, 224, 224, 3)
        results = predict_batch(dummy_cnn_model, batch)
        assert len(results) == 3

    def test_no_threshold_applied(self, dummy_cnn_model, dummy_img_array):
        """Prediction must use argmax only — no confidence threshold."""
        result = predict_single(dummy_cnn_model, dummy_img_array)
        probs = list(result["probabilities"].values())
        max_prob = max(probs)
        # predicted class must be the argmax class regardless of its probability value
        expected_idx = int(np.argmax(probs))
        assert result["predicted_index"] == expected_idx


# ─────────────────────────────────────────────────────────────────────────────
# 2. LSTM Model Tests
# ─────────────────────────────────────────────────────────────────────────────
class TestLSTMModel:

    def test_architecture_input_size(self, dummy_lstm_model):
        """LSTM must have input_size=20 (H3 FIX)."""
        assert dummy_lstm_model.lstm.input_size == 20

    def test_architecture_hidden_size(self, dummy_lstm_model):
        assert dummy_lstm_model.lstm.hidden_size == 64

    def test_architecture_num_layers(self, dummy_lstm_model):
        assert dummy_lstm_model.lstm.num_layers == 2

    def test_forward_output_shape(self, dummy_lstm_model, dummy_lstm_sequence, dummy_device):
        """Single sequence forward pass must return a scalar logit."""
        dummy_lstm_model.to(dummy_device)
        tensor = torch.tensor(
            dummy_lstm_sequence[np.newaxis, ...], dtype=torch.float32
        ).to(dummy_device)
        with torch.no_grad():
            output = dummy_lstm_model(tensor)
        assert output.shape == torch.Size([1]), f"Expected shape (1,), got {output.shape}"

    def test_sigmoid_score_in_range(self, dummy_lstm_model, dummy_lstm_sequence, dummy_device):
        """Sigmoid risk score must be in [0, 1]."""
        dummy_lstm_model.to(dummy_device)
        score = predict_risk_score(dummy_lstm_model, dummy_lstm_sequence, dummy_device)
        assert 0.0 <= score <= 1.0, f"LSTM score out of range: {score}"

    def test_batch_inference_shape(self, dummy_lstm_model, dummy_device):
        """predict_risk_batch on N sequences must return N scores."""
        dummy_lstm_model.to(dummy_device)
        sequences = np.random.rand(5, 14, 20).astype(np.float32)
        scores = predict_risk_batch(dummy_lstm_model, sequences, dummy_device)
        assert scores.shape == (5,)
        assert np.all((scores >= 0) & (scores <= 1))

    def test_sequence_lookback_dimension(self, dummy_lstm_sequence):
        """Sequence must have lookback=14 timesteps."""
        assert dummy_lstm_sequence.shape == (14, 20), (
            f"Expected (14, 20), got {dummy_lstm_sequence.shape}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Preprocessing Tests
# ─────────────────────────────────────────────────────────────────────────────
class TestPreprocessing:

    def test_pil_output_shape(self, dummy_img_array):
        """Preprocessed array must have shape (1, 224, 224, 3)."""
        assert dummy_img_array.shape == (1, 224, 224, 3), (
            f"Expected (1, 224, 224, 3), got {dummy_img_array.shape}"
        )

    def test_pil_output_dtype(self, dummy_img_array):
        """Preprocessed array must be float32 (EfficientNetB0 expects 0-255 float)."""
        assert dummy_img_array.dtype == np.float32

    def test_pil_value_range(self, dummy_img_array):
        """Values must be in [0, 255] — no manual normalisation applied."""
        assert dummy_img_array.min() >= 0.0
        assert dummy_img_array.max() <= 255.0

    def test_bytes_output_shape(self, dummy_image_bytes):
        """Preprocessing from bytes must return (1, 224, 224, 3)."""
        arr = preprocess_image_bytes(dummy_image_bytes)
        assert arr.shape == (1, 224, 224, 3)

    def test_bytes_output_dtype(self, dummy_image_bytes):
        arr = preprocess_image_bytes(dummy_image_bytes)
        assert arr.dtype == np.float32

    def test_non_square_image_resized(self):
        """Non-square images must be resized to (224, 224)."""
        img = Image.fromarray(np.zeros((300, 400, 3), dtype=np.uint8))
        arr = preprocess_pil_image(img)
        assert arr.shape == (1, 224, 224, 3)

    def test_rgba_converted_to_rgb(self):
        """RGBA images must be converted to RGB (alpha channel dropped)."""
        img = Image.fromarray(np.zeros((224, 224, 4), dtype=np.uint8), mode="RGBA")
        arr = preprocess_pil_image(img)
        assert arr.shape == (1, 224, 224, 3)

    def test_batch_preprocessing(self, tmp_path, dummy_image_pil):
        """Batch preprocessing must return (N, 224, 224, 3)."""
        paths = []
        for i in range(3):
            p = tmp_path / f"img_{i}.jpg"
            dummy_image_pil.save(p)
            paths.append(p)
        batch = preprocess_batch(paths)
        assert batch.shape == (3, 224, 224, 3)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Helper Tests
# ─────────────────────────────────────────────────────────────────────────────
class TestHelpers:

    def test_interpret_risk_low(self):
        result = interpret_risk(0.10)
        assert result["band"] == "Low"
        assert result["emoji"] == "🟢"

    def test_interpret_risk_moderate(self):
        result = interpret_risk(0.50)
        assert result["band"] == "Moderate"
        assert result["emoji"] == "🟡"

    def test_interpret_risk_high(self):
        result = interpret_risk(0.80)
        assert result["band"] == "High"
        assert result["emoji"] == "🔴"

    def test_interpret_risk_boundary_low_moderate(self):
        """Score of 0.35 should be Moderate (boundary is inclusive on lower band)."""
        result = interpret_risk(0.35)
        assert result["band"] == "Moderate"

    def test_interpret_risk_boundary_moderate_high(self):
        result = interpret_risk(0.65)
        assert result["band"] == "High"

    def test_load_config_returns_dict(self):
        config = load_config("config.yaml")
        assert isinstance(config, dict)
        for section in ["cnn", "lstm", "fusion", "paths"]:
            assert section in config, f"Missing config section: {section}"

    def test_engineer_weather_features_adds_columns(self, dummy_weather_df):
        """Feature engineering must add all required cyclical and seasonal columns."""
        df_eng = engineer_weather_features(dummy_weather_df)
        for col in ["month_sin", "month_cos", "day_sin", "day_cos",
                    "season", "district_id_norm", "precip_7d_sum"]:
            assert col in df_eng.columns, f"Missing engineered column: {col}"

    def test_build_lstm_sequence_shape(self, dummy_weather_df):
        """build_lstm_sequence must return (14, 20) after feature engineering."""
        from sklearn.preprocessing import MinMaxScaler
        from app.utils.helper import FEATURE_COLS

        df_eng = engineer_weather_features(dummy_weather_df)
        scaler = MinMaxScaler()
        scaler.fit(df_eng[FEATURE_COLS].values)
        seq = build_lstm_sequence(df_eng, scaler, district_id=1)
        assert seq.shape == (14, 20), f"Expected (14, 20), got {seq.shape}"
        assert seq.dtype == np.float32

    def test_build_lstm_sequence_scaled_range(self, dummy_weather_df):
        """Scaled sequence values must be in [0, 1]."""
        from sklearn.preprocessing import MinMaxScaler
        from app.utils.helper import FEATURE_COLS

        df_eng = engineer_weather_features(dummy_weather_df)
        scaler = MinMaxScaler()
        scaler.fit(df_eng[FEATURE_COLS].values)
        seq = build_lstm_sequence(df_eng, scaler, district_id=1)
        # Scaler trained on same data → expect values tightly in [0, 1]
        assert seq.min() >= -0.01
        assert seq.max() <= 1.01


# ─────────────────────────────────────────────────────────────────────────────
# 5. Fusion Tests
# ─────────────────────────────────────────────────────────────────────────────
class TestFusion:

    def _fuse(self, cnn_risk: float, lstm_risk: float, alpha: float) -> float:
        return alpha * cnn_risk + (1.0 - alpha) * lstm_risk

    def test_fusion_alpha_0_is_pure_lstm(self):
        """When alpha=0, fused score == lstm_risk_score."""
        fused = self._fuse(cnn_risk=0.8, lstm_risk=0.3, alpha=0.0)
        assert abs(fused - 0.3) < 1e-6

    def test_fusion_alpha_1_is_pure_cnn(self):
        """When alpha=1, fused score == cnn_risk_score."""
        fused = self._fuse(cnn_risk=0.8, lstm_risk=0.3, alpha=1.0)
        assert abs(fused - 0.8) < 1e-6

    def test_fusion_output_in_range(self):
        """Fused score must always be in [0, 1]."""
        for alpha in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            for cnn_risk in [0.0, 0.5, 1.0]:
                for lstm_risk in [0.0, 0.5, 1.0]:
                    fused = self._fuse(cnn_risk, lstm_risk, alpha)
                    assert 0.0 <= fused <= 1.0, (
                        f"Fused out of range: {fused}  "
                        f"(cnn={cnn_risk}, lstm={lstm_risk}, alpha={alpha})"
                    )

    def test_fusion_config_alpha_valid(self):
        """config.yaml fusion.alpha must be in [0, 1]."""
        config = load_config("config.yaml")
        alpha = config["fusion"]["alpha"]
        assert 0.0 <= alpha <= 1.0, f"Invalid alpha in config: {alpha}"

    def test_cnn_risk_formula(self):
        """cnn_risk = 1 - P(Healthy) must be in [0, 1] for any valid softmax."""
        for p_healthy in [0.0, 0.1, 0.5, 0.9, 1.0]:
            cnn_risk = 1.0 - p_healthy
            assert 0.0 <= cnn_risk <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 6. Config Tests
# ─────────────────────────────────────────────────────────────────────────────
class TestConfig:

    def test_class_names_in_config(self):
        config = load_config("config.yaml")
        names = config["cnn"]["class_names"]
        assert len(names) == 4

    def test_class_names_alphabetical_in_config(self):
        config = load_config("config.yaml")
        names = config["cnn"]["class_names"]
        assert names == sorted(names), "config.yaml class_names must be alphabetically sorted"

    def test_healthy_idx_in_config(self):
        config = load_config("config.yaml")
        names = config["cnn"]["class_names"]
        healthy_idx = config["cnn"]["healthy_idx"]
        assert names[healthy_idx] == "Healthy Rice Leaf"

    def test_feature_cols_count(self):
        config = load_config("config.yaml")
        assert len(config["lstm"]["feature_cols"]) == 20, (
            "LSTM must have exactly 20 feature columns (H3 FIX)"
        )

    def test_lookback_value(self):
        config = load_config("config.yaml")
        assert config["lstm"]["lookback"] == 14

    def test_img_size_value(self):
        config = load_config("config.yaml")
        assert config["cnn"]["img_size"] == [224, 224]
