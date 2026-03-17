---
title: Rice Disease Detection
emoji: 🌾
colorFrom: green
colorTo: yellow
sdk: docker
pinned: false
---

# Rice Disease Detection
**EfficientNetB0 CNN + WeatherRiskLSTM | Anna University Final Year Project 2025–26**

EfficientNetB0 CNN + WeatherRiskLSTM hybrid system for rice leaf disease
classification and weather-driven risk prediction across Tamil Nadu districts.

---

## Disease Classes
1. Bacterial Leaf Blight
2. Brown Spot
3. Healthy Rice Leaf
4. Leaf Blast

## Tamil Nadu Districts (LSTM)
Thanjavur · Nilgiris · Chennai · Virudhunagar · Nagapattinam

---

## How to Use

**Tab 1 — Disease Classifier** : Upload a rice leaf image → get disease prediction

**Tab 2 — Weather Risk** : Select district → auto fetch 14 days weather → LSTM risk score

**Tab 3 — Grad-CAM** : Upload leaf image → see which area model focused on

**Tab 4 — Full Fusion** : Image + district → CNN + LSTM combined risk score

---

## Model Details

| Component | Details |
|---|---|
| CNN | EfficientNetB0 · 4 classes · CategoricalFocalCrossentropy |
| LSTM | WeatherRiskLSTM · 20 features · 14 day lookback |
| Fusion | alpha × cnn_risk + (1−alpha) × lstm_risk |
| Weather API | Open-Meteo Historical Archive |