"""Tests for the request schema and the form field list."""

import json

import pytest
from pydantic import ValidationError

from app.schemas import FORM_FIELDS, EXAMPLES, CustomerRecord
from churn.config import FEATURE_COLUMNS


def test_form_fields_follow_the_feature_order():
    assert [field["name"] for field in FORM_FIELDS] == FEATURE_COLUMNS


def test_examples_match_the_schema():
    for payload in EXAMPLES.values():
        record = CustomerRecord.model_validate(payload)
        dumped = record.model_dump(mode="json")
        assert dumped["gender"] in {"Female", "Male"}
        assert dumped["SeniorCitizen"] in {0, 1}
        assert isinstance(dumped["MonthlyCharges"], float)


def test_schema_rejects_unknown_gender_and_bool_senior():
    payload = dict(EXAMPLES["high_risk"])
    payload["gender"] = "Other"
    with pytest.raises(ValidationError):
        CustomerRecord.model_validate(payload)

    payload = dict(EXAMPLES["high_risk"])
    payload["SeniorCitizen"] = True
    with pytest.raises(ValidationError):
        CustomerRecord.model_validate(payload)


def test_schema_rejects_tenure_above_the_limit():
    payload = dict(EXAMPLES["low_risk"])
    payload["tenure"] = 121
    with pytest.raises(ValidationError):
        CustomerRecord.model_validate(payload)


def test_schema_accepts_numeric_strings():
    payload = dict(EXAMPLES["low_risk"])
    payload["tenure"] = "72"
    payload["SeniorCitizen"] = "0"
    payload["MonthlyCharges"] = "60.0"
    record = CustomerRecord.model_validate(payload)
    assert record.tenure == 72
    assert record.SeniorCitizen == 0
    assert record.MonthlyCharges == pytest.approx(60.0)


def test_training_categories_are_all_allowed_by_the_schema():
    from churn.data import clean_features, load_raw

    features = clean_features(load_raw())
    allowed = {
        field["name"]: set(field["options"])
        for field in FORM_FIELDS
        if field["kind"] == "select"
    }
    for column, values in allowed.items():
        observed = set(features[column].astype(str).unique())
        assert observed <= values, (column, observed - values)
    assert json.dumps(EXAMPLES["high_risk"])
