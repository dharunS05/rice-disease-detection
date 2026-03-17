"""
app/models/lstm_model.py
========================
WeatherRiskLSTM — binary weather-risk classifier for rice disease.

Architecture (from Weather_data_collection_DAY3_FIXED.ipynb, Cell 10):
  - Input  : (batch, 14, 20)  — 14-day rolling window, 20 weather features
  - LSTM   : input_size=20, hidden_size=64, num_layers=2, dropout=0.5
  - Head   : Linear(64→32) → ReLU → Dropout(0.3) → Linear(32→1)
  - Output : raw logit — apply sigmoid for risk probability

Training details:
  - Loss      : BCEWithLogitsLoss with pos_weight (class-imbalance correction)
  - Optimizer : Adam (lr=0.001, weight_decay=1e-4)
  - Sampler   : WeightedRandomSampler (oversample risk days)
  - Scheduler : ReduceLROnPlateau (patience=5, factor=0.5)
  - Saved as  : best_lstm_v2.pth  (5-district Tamil Nadu model)

H3 FIX (applied in notebook):
  - input_size increased 19 → 20 (added district_id_norm feature)
  - Sequences created per-district to prevent cross-district data leakage

Inference:
  - Returns raw sigmoid score in [0, 1] (NO hard threshold applied here)
  - Threshold was 0.40 inside the training loop ONLY for F1 tracking
  - Final risk decision is made by the fusion layer in prediction_service.py
"""

import numpy as np
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Model definition
# ─────────────────────────────────────────────────────────────────────────────
class WeatherRiskLSTM(nn.Module):
    """
    LSTM-based binary classifier that predicts daily weather-driven disease risk.

    Exact same architecture as the training notebook — do NOT modify without
    retraining the model, as the saved state dict must match this definition.

    Args:
        input_size  : Number of weather features per timestep. Must be 20
                      (H3 FIX — includes district_id_norm).
        hidden_size : LSTM hidden units.
        num_layers  : Stacked LSTM layers.
        dropout     : Dropout between LSTM layers (ignored when num_layers=1).
    """

    def __init__(
        self,
        input_size: int = 20,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, lookback, input_size) — scaled weather sequences.

        Returns:
            (batch,) raw logits — pass through sigmoid for probabilities.
        """
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]          # take final timestep hidden state
        return self.classifier(last_out).squeeze(1)


# ─────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────────────────────────────────────
def predict_risk_score(
    model: WeatherRiskLSTM,
    sequence: np.ndarray,
    device: torch.device,
) -> float:
    """
    Run inference on a single 14-day weather sequence.

    Args:
        model   : Loaded WeatherRiskLSTM (from load_model.py).
        sequence: float32 array of shape (14, 20) — one district's rolling window,
                  already MinMaxScaled using the saved scaler.pkl.
        device  : torch.device("cuda") or torch.device("cpu").

    Returns:
        float in [0, 1] — sigmoid risk probability.
        Higher = more weather conditions favour disease outbreak.
        No threshold is applied here; the fusion layer decides.
    """
    model.eval()
    # Add batch dimension: (14, 20) → (1, 14, 20)
    tensor = torch.tensor(sequence[np.newaxis, ...], dtype=torch.float32).to(device)
    with torch.no_grad():
        logit = model(tensor)
        score = torch.sigmoid(logit).item()
    return float(score)


def predict_risk_batch(
    model: WeatherRiskLSTM,
    sequences: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """
    Run inference on a batch of weather sequences (for evaluation / ablation).

    Args:
        model     : Loaded WeatherRiskLSTM.
        sequences : float32 array of shape (N, 14, 20).
        device    : Compute device.
        batch_size: Mini-batch size for inference (avoids OOM on large sets).

    Returns:
        float32 array of shape (N,) — sigmoid risk scores in [0, 1].
    """
    model.eval()
    scores = []
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            batch = torch.tensor(
                sequences[start : start + batch_size], dtype=torch.float32
            ).to(device)
            logits = model(batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            scores.extend(probs.tolist())
    return np.array(scores, dtype=np.float32)
