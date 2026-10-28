"""Load the saved pipeline and score one customer record."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from churn.config import MODEL_PATH
from churn.data import clean_features
from churn.risk import churn_label, risk_tier


def load_model(path: Path = MODEL_PATH):
    """Load the joblib pipeline saved by ``python -m churn.train``."""
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Run `python -m churn.train` first."
        )
    return joblib.load(model_path)


def positive_probabilities(model, features: pd.DataFrame) -> np.ndarray:
    """Return P(Churn) using the classifier class index for label 1."""
    classes = [int(label) for label in model.named_steps["classifier"].classes_]
    try:
        index = classes.index(1)
    except ValueError as exc:
        raise RuntimeError("Fitted classifier has no Churn class (label 1)") from exc
    probabilities = model.predict_proba(features)[:, index]
    return np.asarray(probabilities, dtype=float)


def predict_customer(model, record: dict) -> dict:
    """Score one customer and return probability, tier, and Churn/Stay label.

    The probability is rounded to 4 decimals. The tier and the label use that
    same rounded value, so the three fields agree with each other.
    """
    features = clean_features(pd.DataFrame([record]))
    probability = float(positive_probabilities(model, features)[0])
    probability = min(1.0, max(0.0, probability))
    shown = round(probability, 4)
    return {
        "churn_probability": shown,
        "risk_tier": risk_tier(shown),
        "prediction": churn_label(shown),
    }
