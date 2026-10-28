"""Tests for probability rounding without the saved model."""

import json

import numpy as np

from churn.config import EXAMPLES_PATH
from churn.predict import predict_customer


class _Classifier:
    classes_ = [0, 1]


class _Model:
    def __init__(self, probability):
        self.probability = probability
        self.named_steps = {"classifier": _Classifier()}

    def predict_proba(self, _features):
        stay = 1.0 - self.probability
        return np.array([[stay, self.probability]])


def _record():
    return json.loads(EXAMPLES_PATH.read_text(encoding="utf-8"))["high_risk"]


def test_score_uses_the_rounded_probability_for_tier_and_label():
    result = predict_customer(_Model(0.75432), _record())
    assert result["churn_probability"] == round(0.75432, 4)
    assert result["risk_tier"] == "High"
    assert result["prediction"] == "Churn"


def test_rounding_can_move_a_score_across_the_tier_boundary():
    result = predict_customer(_Model(0.29996), _record())
    assert result["churn_probability"] == round(0.29996, 4)
    assert result["risk_tier"] == "Medium"
    assert result["prediction"] == "Stay"
