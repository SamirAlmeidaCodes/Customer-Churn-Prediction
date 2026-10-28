"""Tests for encoding, scaling, and SMOTE inside the model pipeline."""

import numpy as np
import pandas as pd
import pytest
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn.data import load_raw, prepare_dataset
from churn.pipeline import build_pipeline, build_preprocessor


@pytest.fixture(scope="module")
def sample_xy():
    features, target = prepare_dataset(load_raw())
    sample_x, _rest_x, sample_y, _rest_y = train_test_split(
        features,
        target,
        train_size=800,
        stratify=target,
        random_state=0,
    )
    return sample_x, sample_y


def test_pipeline_order_is_encode_scale_then_smote():
    pipeline = build_pipeline("logistic_regression")
    assert list(pipeline.named_steps) == ["preprocessor", "smote", "classifier"]
    assert isinstance(pipeline.named_steps["smote"], SMOTE)

    transformers = {
        name: transformer
        for name, transformer, _columns in pipeline.named_steps["preprocessor"].transformers
    }
    assert list(transformers["num"].named_steps) == ["imputer", "scaler"]
    assert list(transformers["cat"].named_steps) == ["imputer", "encoder"]
    assert isinstance(transformers["num"].named_steps["scaler"], StandardScaler)
    assert isinstance(transformers["cat"].named_steps["encoder"], OneHotEncoder)
    assert transformers["cat"].named_steps["encoder"].sparse_output is False


def test_unknown_model_name_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        build_pipeline("random_forest")


def test_preprocessor_outputs_finite_numbers():
    features, _target = prepare_dataset(load_raw())
    transformed = build_preprocessor().fit_transform(features)
    assert transformed.shape[0] == len(features)
    assert transformed.shape[1] > features.shape[1]
    assert not np.isnan(transformed).any()


def test_smote_balances_the_training_labels(sample_xy):
    features, target = sample_xy
    transformed = build_preprocessor().fit_transform(features)
    _resampled_x, resampled_y = SMOTE(random_state=42, k_neighbors=5).fit_resample(
        transformed,
        target,
    )
    counts = pd.Series(resampled_y).value_counts()
    assert counts.loc[0] == counts.loc[1]
    assert len(resampled_y) > len(target)


def test_single_row_predict_does_not_rerun_smote(sample_xy):
    features, target = sample_xy
    pipeline = build_pipeline("logistic_regression")
    pipeline.fit(features, target)
    assert hasattr(pipeline.named_steps["smote"], "sampling_strategy_")

    probability = pipeline.predict_proba(features.iloc[[0]])
    assert probability.shape == (1, 2)
    assert 0.0 <= probability[0, 1] <= 1.0


def test_unknown_category_is_ignored(sample_xy):
    features, target = sample_xy
    pipeline = build_pipeline("logistic_regression")
    pipeline.fit(features, target)
    row = features.iloc[[0]].copy()
    row["gender"] = "Unknown"
    probability = pipeline.predict_proba(row)
    assert probability.shape == (1, 2)
    assert np.isfinite(probability).all()
