"""Tests for dataset loading and cleaning."""

import pandas as pd
import pytest

from churn.config import FEATURE_COLUMNS, ID_COLUMN
from churn.data import clean_features, ensure_dataset, load_raw, prepare_dataset


def test_local_dataset_is_present():
    path = ensure_dataset()
    assert path.exists()
    assert path.stat().st_size > 0


def test_prepare_dataset_encodes_the_public_table():
    raw = load_raw()
    features, target = prepare_dataset(raw)
    assert len(raw) == 7043
    assert len(features) == len(raw)
    assert list(features.columns) == FEATURE_COLUMNS
    assert ID_COLUMN not in features.columns
    assert set(target.unique()) == {0, 1}
    assert int(target.sum()) == 1869
    assert set(features["SeniorCitizen"].unique()) == {"0", "1"}


def _missing_total_charges(raw: pd.DataFrame) -> pd.Series:
    totals = raw["TotalCharges"]
    if pd.api.types.is_numeric_dtype(totals):
        return totals.isna()
    text = totals.astype(str).str.strip()
    return totals.isna() | text.eq("") | text.str.lower().eq("nan")


def test_blank_total_charges_on_new_accounts_become_zero():
    raw = load_raw()
    blank = _missing_total_charges(raw)
    assert int(blank.sum()) == 11
    assert set(raw.loc[blank, "tenure"].astype(int)) == {0}

    features, _target = prepare_dataset(raw)
    assert int(features["TotalCharges"].isna().sum()) == 0
    tenure_zero = features["tenure"].eq(0)
    assert bool((features.loc[tenure_zero, "TotalCharges"] == 0).all())


def test_numeric_text_is_stripped():
    raw = load_raw().head(1).copy()
    raw["TotalCharges"] = " 29.85 "
    features = clean_features(raw)
    assert features["TotalCharges"].iloc[0] == pytest.approx(29.85)


def test_unexpected_churn_label_raises():
    raw = load_raw().head(3).copy()
    raw.loc[raw.index[0], "Churn"] = "Maybe"
    with pytest.raises(ValueError, match="Unexpected Churn labels"):
        prepare_dataset(raw)


def test_missing_feature_column_raises():
    raw = load_raw().head(2).drop(columns=["Partner"])
    with pytest.raises(ValueError, match="Missing feature columns"):
        clean_features(raw)


def test_senior_citizen_rejects_other_codes():
    raw = load_raw().head(1).copy()
    raw["SeniorCitizen"] = 2
    with pytest.raises(ValueError, match="SeniorCitizen"):
        clean_features(raw)


def test_clean_features_keeps_median_gap_when_tenure_is_not_zero():
    raw = load_raw().head(1).copy()
    raw["tenure"] = 5
    raw["TotalCharges"] = " "
    features = clean_features(raw)
    assert pd.isna(features["TotalCharges"].iloc[0])
