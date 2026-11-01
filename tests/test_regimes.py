"""Tests for size-based training regimes and the demo tables."""

import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier

from churn.config import (
    FEATURE_COLUMNS,
    GRADIENT_BOOSTING_PARAMS,
    ID_COLUMN,
    LOGISTIC_REGRESSION_PARAMS,
    TARGET_COLUMN,
)
from churn.data import load_raw, prepare_dataset
from churn.demo_data import (
    DEMO_LARGE_PATH,
    DEMO_LARGE_ROWS,
    DEMO_SMALL_PATH,
    DEMO_SMALL_ROWS,
)
from churn.pipeline import (
    BalancedSampleWeightClassifier,
    build_pipeline,
    build_pipeline_for_regime,
)
from churn.regimes import choose_regime, fixed_regime


def test_fixed_regime_matches_the_original_settings():
    regime = fixed_regime()
    assert regime.name == "fixed"
    assert regime.test_size == 0.2
    assert regime.cv_folds == 5
    assert regime.use_smote is True
    assert regime.smote_k_neighbors == 5
    assert regime.balance_classes is False
    assert regime.tree_model == "gradient_boosting"
    assert regime.tree_params == GRADIENT_BOOSTING_PARAMS
    assert regime.logistic_params == LOGISTIC_REGRESSION_PARAMS


def test_boundaries_between_the_three_regimes():
    small = choose_regime(999, 200)
    assert small.name == "small"
    assert small.test_size == 0.3
    assert small.cv_folds == 5
    assert small.use_smote is True
    assert small.smote_k_neighbors == 3
    assert small.tree_params["n_estimators"] == 100
    assert small.tree_params["max_depth"] == 2
    assert small.tree_params["min_samples_leaf"] == 5
    assert "n_iter_no_change" not in small.tree_params

    medium = choose_regime(1_000, 200)
    assert medium.name == "medium"
    assert medium.test_size == 0.2
    assert medium.cv_folds == 5
    assert medium.use_smote is True
    assert medium.smote_k_neighbors == 5
    assert medium.tree_params["n_estimators"] == 200
    assert medium.tree_params["n_iter_no_change"] == 10

    upper = choose_regime(10_000, 2_000)
    assert upper.name == "medium"
    assert upper.test_size == 0.2

    large = choose_regime(10_001, 3_000)
    assert large.name == "large"
    assert large.test_size == 0.1
    assert large.cv_folds == 3
    assert large.use_smote is False
    assert large.balance_classes is True
    assert large.tree_model == "hist_gradient_boosting"
    assert large.tree_params["max_iter"] == 300
    assert large.tree_params["class_weight"] == "balanced"
    assert large.logistic_params["class_weight"] == "balanced"


def test_few_churners_use_the_small_regime_even_on_a_long_table():
    regime = choose_regime(5_000, 149)
    assert regime.name == "small"
    assert regime.test_size == 0.2


def test_large_table_keeps_80_20_when_the_10_percent_holdout_is_thin():
    regime = choose_regime(20_000, 400)
    assert regime.name == "large"
    assert regime.test_size == 0.2
    assert regime.cv_folds == 3
    assert regime.use_smote is True
    assert regime.balance_classes is False


def test_very_few_churners_skip_smote_and_clamp_the_folds():
    regime = choose_regime(400, 25)
    assert regime.use_smote is False
    assert regime.balance_classes is True
    assert regime.tree_model == "gradient_boosting"

    clamped = choose_regime(20, 6)
    assert clamped.cv_folds == 4

    with pytest.raises(ValueError, match="at least 2 customers"):
        choose_regime(6, 2)


def test_public_table_selects_the_medium_regime():
    raw = load_raw()
    regime = choose_regime(len(raw), int(raw[TARGET_COLUMN].eq("Yes").sum()))
    assert regime.name == "medium"
    assert regime.test_size == 0.2
    assert regime.use_smote is True
    assert regime.smote_k_neighbors == 5


def test_class_weight_pipeline_scores_one_row():
    raw = load_raw().head(120)
    features, target = prepare_dataset(raw)
    pipeline = build_pipeline(
        "gradient_boosting",
        use_smote=False,
        balance_classes=True,
        estimator_params={
            "n_estimators": 8,
            "learning_rate": 0.1,
            "max_depth": 2,
            "min_samples_leaf": 5,
            "random_state": 0,
        },
    )
    assert isinstance(
        pipeline.named_steps["classifier"],
        BalancedSampleWeightClassifier,
    )
    assert "smote" not in pipeline.named_steps
    pipeline.fit(features, target)
    probability = pipeline.predict_proba(features.iloc[[0]])
    assert probability.shape == (1, 2)
    assert 0.0 <= probability[0, 1] <= 1.0


def test_large_regime_pipeline_uses_histogram_boosting_without_smote():
    regime = choose_regime(12_000, 3_184)
    pipeline = build_pipeline_for_regime(regime.tree_model, regime)
    assert list(pipeline.named_steps) == ["preprocessor", "classifier"]
    assert isinstance(
        pipeline.named_steps["classifier"],
        HistGradientBoostingClassifier,
    )
    logistic = build_pipeline_for_regime("logistic_regression", regime)
    assert logistic.named_steps["classifier"].get_params()["class_weight"] == "balanced"


def test_demo_tables_have_the_expected_size_and_regime():
    assert DEMO_SMALL_PATH.exists(), "Run python -m churn.demo_data first"
    assert DEMO_LARGE_PATH.exists(), "Run python -m churn.demo_data first"
    source = load_raw()
    small = pd.read_csv(DEMO_SMALL_PATH)
    large = pd.read_csv(DEMO_LARGE_PATH)

    assert len(small) == DEMO_SMALL_ROWS
    assert len(large) == DEMO_LARGE_ROWS
    assert list(small.columns) == list(source.columns)
    assert list(large.columns) == list(source.columns)
    assert set(small[ID_COLUMN]).issubset(set(source[ID_COLUMN]))
    assert large[ID_COLUMN].is_unique
    assert int(large[ID_COLUMN].astype(str).str.startswith("L").sum()) == len(large)

    small_features, small_target = prepare_dataset(small)
    large_features, large_target = prepare_dataset(large)
    assert list(small_features.columns) == FEATURE_COLUMNS
    assert list(large_features.columns) == FEATURE_COLUMNS
    assert set(small_target.unique()) == {0, 1}
    assert set(large_target.unique()) == {0, 1}

    small_regime = choose_regime(len(small), int(small_target.sum()))
    large_regime = choose_regime(len(large), int(large_target.sum()))
    assert small_regime.name == "small"
    assert small_regime.test_size == 0.3
    assert large_regime.name == "large"
    assert large_regime.test_size == 0.1
    assert large_regime.use_smote is False
