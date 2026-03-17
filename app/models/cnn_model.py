"""
app/models/cnn_model.py
=======================
EfficientNetB0-based CNN for rice disease classification.

Architecture (from RICE_DISEASE_EfficientNetB0_FINAL.ipynb):
  - Backbone  : EfficientNetB0 (ImageNet pretrained, input 0–255 float)
  - Head      : GlobalAveragePooling2D → Dropout(0.3) → Dense(256, relu, L2)
                → Dropout(0.2) → Dense(4, softmax)
  - Loss      : CategoricalFocalCrossentropy (gamma=2, label_smoothing=0.1)
  - Classes   : Bacterial Leaf Blight / Brown Spot / Healthy Rice Leaf / Leaf Blast
  - Prediction: argmax of softmax probabilities (NO confidence threshold applied)

Training phases:
  Phase 1 — Backbone frozen,  head trained  (LR=1e-4, up to 30 epochs)
  Phase 2 — Top 20 layers unfrozen          (LR=1e-5, up to 20 more epochs)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0


# ─────────────────────────────────────────────────────────────────────────────
# Constants (kept in sync with config.yaml)
# ─────────────────────────────────────────────────────────────────────────────
IMG_SIZE = (224, 224)
NUM_CLASSES = 4
SEED = 42

CLASS_NAMES = [
    "Bacterial Leaf Blight",   # index 0
    "Brown Spot",              # index 1
    "Healthy Rice Leaf",       # index 2
    "Leaf Blast",              # index 3
]
HEALTHY_IDX = 2
LEAF_BLAST_IDX = 3

FOCAL_GAMMA = 2.0
FOCAL_LABEL_SMOOTHING = 0.1
# Per-class alpha — same order as CLASS_NAMES (alphabetical / TF load order)
FOCAL_ALPHA = [0.15, 0.20, 0.10, 0.35]

LR_PHASE1 = 1e-4
LR_PHASE2 = 1e-5
UNFREEZE_TOP_N = 20    # top N layers of EfficientNetB0 unfrozen in Phase 2


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation layer (used during training only)
# ─────────────────────────────────────────────────────────────────────────────
def build_augmentation_layer(seed: int = SEED) -> keras.Sequential:
    """
    Data augmentation applied during training.
    Stronger augmentation for hard classes (Brown Spot / Leaf Blast)
    is handled by the augmentation pipeline itself — same layer for all.

    Used in Phase 1 and Phase 2 training (training=True enables randomness).
    Called with training=False inside GradCAM model to disable randomness.
    """
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical", seed=seed),
            layers.RandomRotation(0.15, seed=seed),       # ±54°
            layers.RandomZoom(0.10, seed=seed),
            layers.RandomTranslation(0.05, 0.05, seed=seed),
            layers.RandomContrast(0.15, seed=seed),
            layers.RandomBrightness(0.15, seed=seed),
        ],
        name="data_augmentation",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Model builder
# ─────────────────────────────────────────────────────────────────────────────
def build_cnn_model(lr: float = LR_PHASE1) -> keras.Model:
    """
    Build and compile the EfficientNetB0 classification model.

    EfficientNetB0 built-in preprocessing handles raw 0–255 float input
    (no manual rescaling needed).

    Args:
        lr: Learning rate for the Adam optimizer.

    Returns:
        Compiled keras.Model ready for Phase 1 training (backbone frozen).
    """
    data_augmentation = build_augmentation_layer()

    base_model = EfficientNetB0(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False   # Phase 1: freeze backbone

    inputs = keras.Input(shape=(*IMG_SIZE, 3), name="image_input")
    x = data_augmentation(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.3, name="dropout_1")(x)
    x = layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name="dense_256",
    )(x)
    x = layers.Dropout(0.2, name="dropout_2")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="classifier")(x)

    model = keras.Model(inputs, outputs, name="RiceLeaf_v3_EfficientNetB0")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.CategoricalFocalCrossentropy(
            gamma=FOCAL_GAMMA,
            alpha=FOCAL_ALPHA,
            label_smoothing=FOCAL_LABEL_SMOOTHING,
            from_logits=False,
        ),
        metrics=["accuracy"],
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — unfreeze top N layers for fine-tuning
# ─────────────────────────────────────────────────────────────────────────────
def unfreeze_for_finetuning(model: keras.Model, unfreeze_n: int = UNFREEZE_TOP_N) -> keras.Model:
    """
    Unfreeze the top `unfreeze_n` layers of the EfficientNetB0 backbone
    for Phase 2 fine-tuning.

    BatchNormalization layers are always kept frozen (critical for small datasets).

    Args:
        model     : The compiled model returned by build_cnn_model().
        unfreeze_n: Number of top backbone layers to unfreeze.

    Returns:
        Recompiled model ready for Phase 2 training.
    """
    base_model = model.get_layer("efficientnetb0")
    base_model.trainable = True

    freeze_until = len(base_model.layers) - unfreeze_n
    for i, layer in enumerate(base_model.layers):
        layer.trainable = i >= freeze_until

    # Always freeze BatchNorm — critical for small datasets
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    n_trainable = sum(np.prod(v.shape) for v in model.trainable_weights)
    print(f"[Phase 2] Fine-tuning {n_trainable:,} parameters")
    print(f"[Phase 2] Unfreezing layers [{freeze_until} : {len(base_model.layers) - 1}]")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LR_PHASE2),
        loss=keras.losses.CategoricalFocalCrossentropy(
            gamma=FOCAL_GAMMA,
            alpha=FOCAL_ALPHA,
            label_smoothing=FOCAL_LABEL_SMOOTHING,
            from_logits=False,
        ),
        metrics=["accuracy"],
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Prediction helpers
# ─────────────────────────────────────────────────────────────────────────────
def predict_single(model: keras.Model, img_array: np.ndarray) -> dict:
    """
    Run inference on a single preprocessed image.

    Args:
        model    : Loaded keras model (from load_model.py).
        img_array: float32 array of shape (1, 224, 224, 3), values 0–255.

    Returns:
        dict with keys:
            predicted_class (str)   : e.g. "Leaf Blast"
            predicted_index (int)   : e.g. 3
            probabilities   (list)  : softmax scores for all 4 classes
            cnn_risk_score  (float) : 1 - P(Healthy), used in fusion
    """
    probs = model.predict(img_array, verbose=0)[0]         # shape (4,)
    pred_idx = int(np.argmax(probs))
    return {
        "predicted_class": CLASS_NAMES[pred_idx],
        "predicted_index": pred_idx,
        "probabilities": {name: float(p) for name, p in zip(CLASS_NAMES, probs)},
        "cnn_risk_score": float(1.0 - probs[HEALTHY_IDX]),  # fusion input
    }


def predict_batch(model: keras.Model, img_batch: np.ndarray) -> list[dict]:
    """
    Run inference on a batch of preprocessed images.

    Args:
        model    : Loaded keras model.
        img_batch: float32 array of shape (N, 224, 224, 3), values 0–255.

    Returns:
        List of dicts (same structure as predict_single) — one per image.
    """
    probs_batch = model.predict(img_batch, verbose=0)      # (N, 4)
    results = []
    for probs in probs_batch:
        pred_idx = int(np.argmax(probs))
        results.append(
            {
                "predicted_class": CLASS_NAMES[pred_idx],
                "predicted_index": pred_idx,
                "probabilities": {name: float(p) for name, p in zip(CLASS_NAMES, probs)},
                "cnn_risk_score": float(1.0 - probs[HEALTHY_IDX]),
            }
        )
    return results
