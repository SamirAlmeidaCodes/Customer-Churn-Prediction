"""Tests for model selection, independent of a full training run."""

from churn.train import select_best_model


def _model(f1, roc_auc):
    return {"cv": {"f1": {"mean": f1}, "roc_auc": {"mean": roc_auc}}}


def test_selection_prefers_higher_cv_f1():
    models = {
        "logistic_regression": _model(0.61, 0.90),
        "gradient_boosting": _model(0.64, 0.80),
    }
    assert select_best_model(models) == "gradient_boosting"


def test_selection_breaks_f1_ties_with_auc():
    models = {
        "logistic_regression": _model(0.62, 0.84),
        "gradient_boosting": _model(0.62, 0.81),
    }
    assert select_best_model(models) == "logistic_regression"
