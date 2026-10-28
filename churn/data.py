"""Load and clean the IBM Telco Customer Churn table."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from churn.config import (
    CATEGORICAL_FEATURES,
    DATA_PATH,
    DATA_URL,
    FEATURE_COLUMNS,
    ID_COLUMN,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)


def ensure_dataset(path: Path = DATA_PATH, url: str = DATA_URL) -> Path:
    """Return the CSV path, downloading the public file when it is missing."""
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": "churn-prediction"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            temporary.write_bytes(response.read())
        temporary.replace(path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return path


def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the churn table and reject a file with a different column set."""
    ensure_dataset(path)
    frame = pd.read_csv(path)
    expected = set(FEATURE_COLUMNS + [TARGET_COLUMN, ID_COLUMN])
    if set(frame.columns) != expected:
        missing = sorted(expected - set(frame.columns))
        extra = sorted(set(frame.columns) - expected)
        raise ValueError(f"Unexpected columns. missing={missing} extra={extra}")
    return frame


def _normalize_senior(value: object) -> str:
    if value is None or value is pd.NA:
        raise ValueError("SeniorCitizen is missing")
    try:
        if pd.isna(value):
            raise ValueError("SeniorCitizen is missing")
    except TypeError:
        pass
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if text not in {"0", "1"}:
        raise ValueError(f"SeniorCitizen must be 0 or 1, got {value!r}")
    return text


def clean_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return model features with canonical dtypes.

    Blank TotalCharges values in this dataset belong to tenure-0 accounts.
    They are missing zeros, so they are filled with 0 before the median
    imputer sees the column. Any other numeric gap stays missing for the
    pipeline imputer.
    """
    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    features = frame.loc[:, FEATURE_COLUMNS].copy()
    for column in CATEGORICAL_FEATURES:
        if column == "SeniorCitizen":
            continue
        features[column] = features[column].astype(str).str.strip()
    features["SeniorCitizen"] = features["SeniorCitizen"].map(_normalize_senior)

    for column in NUMERIC_FEATURES:
        if features[column].dtype == object:
            stripped = features[column].astype(str).str.strip()
            features[column] = stripped.replace({"": np.nan})
        features[column] = pd.to_numeric(features[column], errors="coerce")

    missing_total = features["TotalCharges"].isna() & features["tenure"].eq(0)
    features.loc[missing_total, "TotalCharges"] = 0.0
    return features


def prepare_dataset(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return features and a 1/0 churn target aligned to the same index."""
    features = clean_features(frame)
    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"Missing target column {TARGET_COLUMN}")
    labels = frame[TARGET_COLUMN].astype(str).str.strip()
    unexpected = sorted(set(labels.unique()) - {"Yes", "No"})
    if unexpected:
        raise ValueError(f"Unexpected Churn labels: {unexpected}")
    target = labels.map({"Yes": 1, "No": 0}).astype(int)
    target.name = TARGET_COLUMN
    return features, target
