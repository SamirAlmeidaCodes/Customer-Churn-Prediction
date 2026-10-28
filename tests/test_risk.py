"""Tests for risk tiers and the Churn/Stay cutoff."""

import pytest

from churn.risk import churn_label, risk_tier


@pytest.mark.parametrize(
    ("probability", "tier"),
    [
        (0.0, "Low"),
        (0.2999, "Low"),
        (0.30, "Medium"),
        (0.5999, "Medium"),
        (0.60, "High"),
        (1.0, "High"),
    ],
)
def test_risk_tier_boundaries(probability, tier):
    assert risk_tier(probability) == tier


@pytest.mark.parametrize("probability", [-0.01, 1.01])
def test_risk_tier_rejects_out_of_range(probability):
    with pytest.raises(ValueError):
        risk_tier(probability)


def test_churn_label_uses_half_cutoff():
    assert churn_label(0.4999) == "Stay"
    assert churn_label(0.5) == "Churn"
    assert churn_label(1) == "Churn"
