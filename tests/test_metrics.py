"""Tests for the saved comparison and the joblib pipeline."""

import json

from churn.config import METRICS_PATH, MODEL_PATH
from churn.pipeline import build_estimator
from churn.predict import load_model, predict_customer
from churn.train import select_best_model
from app.schemas import EXAMPLES


def test_saved_metrics_cover_both_models():
    assert METRICS_PATH.exists(), "Run python -m churn.train before pytest"
    report = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    assert set(report["models"]) == {"logistic_regression", "gradient_boosting"}
    assert report["best_model"] == select_best_model(report["models"])
    assert report["positive_class"] == "Churn"
    assert report["imbalance"]["method"] == "SMOTE"

    for block in report["models"].values():
        for split in ("cv", "holdout"):
            for metric in ("precision", "recall", "f1", "roc_auc"):
                raw = block[split][metric]
                value = raw["mean"] if isinstance(raw, dict) else raw
                assert 0.0 <= value <= 1.0
                if isinstance(raw, dict):
                    assert 0.0 <= raw["std"] <= 1.0


def test_saved_pipeline_scores_an_example():
    assert MODEL_PATH.exists(), "Run python -m churn.train before pytest"
    model = load_model()
    report = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    estimator = model.named_steps["classifier"]
    assert isinstance(estimator, type(build_estimator(report["best_model"])))
    assert "smote" in model.named_steps

    scored = predict_customer(model, EXAMPLES["low_risk"])
    assert 0.0 <= scored["churn_probability"] <= 1.0
    assert scored["risk_tier"] in {"Low", "Medium", "High"}
