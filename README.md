# Rice Disease Detection
**Anna University Final Year Undergraduate Project 2024–25**

EfficientNetB0 CNN + WeatherRiskLSTM hybrid system for rice leaf disease classification and weather-driven risk prediction across Tamil Nadu districts.

---

## Project Overview

| Component | Details |
|---|---|
| **CNN** | EfficientNetB0 · 4 classes · CategoricalFocalCrossentropy |
| **LSTM** | WeatherRiskLSTM · 20 weather features · 14-day lookback |
| **Fusion** | `alpha × cnn_risk + (1−alpha) × lstm_risk` · alpha validated via ablation |
| **Dataset** | Kaggle rice-disease-dataset · 4 classes · 70/15/15 split |
| **Weather** | Open-Meteo archive API · 5 Tamil Nadu districts · 2015–2024 |

### Disease Classes (CNN)
1. Bacterial Leaf Blight
2. Brown Spot
3. Healthy Rice Leaf
4. Leaf Blast ← hardest class (gets highest focal loss alpha)

### Tamil Nadu Districts (LSTM)
Thanjavur · Nilgiris · Chennai · Virudhunagar · Nagapattinam

---

## Folder Structure

```
rice-disease-detection/
│
├── app/
│   ├── api/
│   │   └── routes.py              ← FastAPI routes (fill after validation)
│   │
│   ├── models/
│   │   ├── cnn_model.py           ← EfficientNetB0 architecture + inference
│   │   ├── lstm_model.py          ← WeatherRiskLSTM architecture + inference
│   │   └── load_model.py          ← Model loading utilities
│   │
│   ├── preprocessing/
│   │   └── image_preprocess.py    ← Image resize / dtype / batch handling
│   │
│   ├── services/
│   │   └── prediction_service.py  ← CNN + LSTM + Fusion logic
│   │
│   ├── utils/
│   │   └── helper.py              ← Grad-CAM, weather features, risk bands
│   │
│   └── main.py                    ← FastAPI app entry point (fill after validation)
│
├── trained_models/
│   ├── rice_cnn_model.keras       ← EfficientNetB0 saved model (copy from Drive)
│   ├── rice_lstm_model.pth        ← WeatherRiskLSTM state dict
│   └── scaler.pkl                 ← MinMaxScaler fitted on training weather data
│
├── tests/
│   └── test_prediction.py         ← pytest tests (CNN, LSTM, fusion, config)
│
├── notebooks/
│   └── training_experiments.ipynb ← Reference notebook
│
├── requirements.txt
├── Dockerfile                     ← (fill after validation)
├── README.md
└── config.yaml                    ← All hyperparameters and paths
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Copy trained models
After training in Google Colab, copy:
```
rice_v3_efficientnet.keras  →  trained_models/rice_cnn_model.keras
best_lstm_v2.pth            →  trained_models/rice_lstm_model.pth
scaler.pkl                  →  trained_models/scaler.pkl
```

### 3. Run tests
```bash
pytest tests/test_prediction.py -v
```

### 4. Quick inference example
```python
from app.services.prediction_service import PredictionService

service = PredictionService.from_config("config.yaml")

# CNN only
result = service.predict_from_image("leaf.jpg")
print(result["predicted_class"], result["cnn_risk_score"])

# CNN + LSTM fusion
import pandas as pd
weather_df = pd.read_csv("14_day_weather.csv")
result = service.predict_fused("leaf.jpg", weather_df, district_id=1)
print(result["risk_band"])
```

---

## Key Design Decisions

### No Confidence Threshold
The CNN uses `argmax` on softmax probabilities only. No minimum-confidence threshold is applied (removed in the final notebook version). The Focal Loss (gamma=2) already penalises uncertain predictions during training.

### No LSTM Hard Threshold
The LSTM returns raw `sigmoid(logit)` scores. The threshold=0.40 in the training notebook was only for computing training-time F1 scores. Inference returns the raw probability.

### Fusion Alpha
The `alpha` in `config.yaml` is a placeholder set to 0.4. Run the ablation grid search in `notebooks/training_experiments.ipynb` on your validation set and update this value.

### Class Order (Critical)
`CLASS_NAMES` must always be:
```python
["Bacterial Leaf Blight", "Brown Spot", "Healthy Rice Leaf", "Leaf Blast"]
```
TensorFlow's `image_dataset_from_directory` loads classes **alphabetically**. Changing this order without retraining will silently corrupt all predictions.

### H3 Fix (LSTM input_size)
The LSTM `input_size=20` (not 19) because `district_id_norm` was added as feature 20. Never change this without retraining.

---

## Running Tests

```bash
# All tests
pytest tests/test_prediction.py -v

# Only CNN tests
pytest tests/test_prediction.py -v -k "CNN"

# Only fusion tests
pytest tests/test_prediction.py -v -k "Fusion"

# With coverage report
pytest tests/test_prediction.py -v --cov=app --cov-report=term-missing
```

---

## Hugging Face Deployment

The demo app is deployed to Hugging Face Spaces via Gradio.
- Upload `trained_models/` files to the Space repository
- Set `config.yaml` paths to match the Space filesystem
- `app/main.py` will contain the Gradio interface (fill after validation)

---

## References

- Tan & Le 2019 — EfficientNet: Rethinking Model Scaling for CNNs
- Lin et al. 2018 — Focal Loss for Dense Object Detection
- Selvaraju et al. 2017 — Grad-CAM: Visual Explanations from Deep Networks
- IRRI — Brown Spot disease thresholds
- Kim et al. 2018 — Leaf Blast environmental conditions
- EPIRICE-SB PMC 2020 — Bacterial Blight epidemiology
