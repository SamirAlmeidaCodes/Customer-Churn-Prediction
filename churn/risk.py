"""Map a churn probability onto a label and an outreach tier."""

from churn.config import (
    CLASSIFICATION_THRESHOLD,
    LOW_RISK_BELOW,
    MEDIUM_RISK_BELOW,
)


def _check_probability(probability: float) -> float:
    value = float(probability)
    if value < 0.0 or value > 1.0:
        raise ValueError(f"probability must be between 0 and 1, got {probability}")
    return value


def risk_tier(probability: float) -> str:
    """Return Low, Medium, or High for an outreach priority.

    Low is below 0.30, Medium is below 0.60, and High is 0.60 and above.
    """
    value = _check_probability(probability)
    if value < LOW_RISK_BELOW:
        return "Low"
    if value < MEDIUM_RISK_BELOW:
        return "Medium"
    return "High"


def churn_label(probability: float) -> str:
    """Return Churn when the probability is at least the classification cutoff."""
    value = _check_probability(probability)
    if value >= CLASSIFICATION_THRESHOLD:
        return "Churn"
    return "Stay"
