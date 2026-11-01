"""Feature pipeline: impute, scale numerics, one-hot encode, then balance classes."""

from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from churn.config import (
    CATEGORICAL_FEATURES,
    GRADIENT_BOOSTING_PARAMS,
    LOGISTIC_REGRESSION_PARAMS,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    SMOTE_PARAMS,
)
from churn.regimes import TrainingRegime

KNOWN_MODELS = (
    "logistic_regression",
    "gradient_boosting",
    "hist_gradient_boosting",
)


def build_preprocessor() -> ColumnTransformer:
    """Impute, scale numeric columns, and one-hot encode categorical columns."""
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            # SMOTE needs a dense numeric matrix. Unknown levels become zeros.
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric, list(NUMERIC_FEATURES)),
            ("cat", categorical, list(CATEGORICAL_FEATURES)),
        ]
    )


class BalancedSampleWeightClassifier(ClassifierMixin, BaseEstimator):
    """Fit an inner classifier with balanced sample weights.

    Gradient boosting has no class_weight argument. The weights follow
    class_weight='balanced': each class contributes the same total weight.
    """

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        estimator = clone(self.estimator)
        y_array = np.asarray(y)
        classes, counts = np.unique(y_array, return_counts=True)
        n_samples = int(y_array.shape[0])
        n_classes = int(classes.shape[0])
        weight_of = {
            label: n_samples / (n_classes * int(count))
            for label, count in zip(classes, counts)
        }
        sample_weight = np.array([weight_of[label] for label in y_array], dtype=float)
        estimator.fit(X, y_array, sample_weight=sample_weight)
        self.estimator_ = estimator
        self.classes_ = estimator.classes_
        return self

    def predict(self, X):
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict_proba(X)

    @property
    def feature_importances_(self):
        check_is_fitted(self, "estimator_")
        return self.estimator_.feature_importances_


def build_estimator(
    model_name: str,
    params: dict | None = None,
    *,
    balance_classes: bool = False,
):
    """Return an unfitted classifier for one of the supported model names."""
    if model_name == "logistic_regression":
        resolved = dict(LOGISTIC_REGRESSION_PARAMS if params is None else params)
        if balance_classes:
            resolved["class_weight"] = "balanced"
        return LogisticRegression(**resolved)
    if model_name == "gradient_boosting":
        resolved = dict(GRADIENT_BOOSTING_PARAMS if params is None else params)
        resolved.pop("class_weight", None)
        estimator = GradientBoostingClassifier(**resolved)
        if balance_classes:
            return BalancedSampleWeightClassifier(estimator)
        return estimator
    if model_name == "hist_gradient_boosting":
        if not params:
            raise ValueError("Histogram boosting parameters are required")
        resolved = dict(params)
        if balance_classes:
            resolved["class_weight"] = "balanced"
        return HistGradientBoostingClassifier(**resolved)
    known = ", ".join(KNOWN_MODELS)
    raise ValueError(f"Unknown model {model_name!r}. Expected one of: {known}")


def build_pipeline(
    model_name: str,
    *,
    use_smote: bool = True,
    smote_k_neighbors: int = SMOTE_PARAMS["k_neighbors"],
    estimator_params: dict | None = None,
    balance_classes: bool = False,
) -> Pipeline:
    """Stack preprocessing, optional SMOTE, and a classifier.

    The sampler runs during fit only, so validation rows and live scoring keep
    the original class balance. With SMOTE off, the classifier uses class
    weights instead of synthetic rows.
    """
    estimator = build_estimator(
        model_name,
        estimator_params,
        balance_classes=balance_classes and not use_smote,
    )
    steps: list = [("preprocessor", build_preprocessor())]
    if use_smote:
        steps.append(
            (
                "smote",
                SMOTE(random_state=RANDOM_STATE, k_neighbors=smote_k_neighbors),
            )
        )
    steps.append(("classifier", estimator))
    if use_smote:
        return ImbPipeline(steps=steps)
    return Pipeline(steps=steps)


def build_pipeline_for_regime(model_name: str, regime: TrainingRegime) -> Pipeline:
    """Build the logistic model or the regime's tree model."""
    if model_name == "logistic_regression":
        params = regime.logistic_params
    elif model_name == regime.tree_model:
        params = regime.tree_params
    else:
        raise ValueError(
            f"Regime {regime.name} does not train {model_name!r}. "
            f"Expected one of: {', '.join(regime.model_names)}"
        )
    return build_pipeline(
        model_name,
        use_smote=regime.use_smote,
        smote_k_neighbors=max(regime.smote_k_neighbors, 1),
        estimator_params=params,
        balance_classes=regime.balance_classes,
    )


def feature_importance_frame(pipeline: ImbPipeline) -> pd.DataFrame:
    """Return gradient-boosting importances aligned to transformed feature names."""
    classifier = pipeline.named_steps["classifier"]
    if not hasattr(classifier, "feature_importances_"):
        raise AttributeError("This classifier does not expose feature_importances_")
    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    frame = pd.DataFrame(
        {"feature": names, "importance": classifier.feature_importances_}
    )
    return frame.sort_values("importance", ascending=False).reset_index(drop=True)


def logistic_coefficient_frame(pipeline: ImbPipeline) -> pd.DataFrame:
    """Return logistic coefficients aligned to transformed feature names."""
    classifier = pipeline.named_steps["classifier"]
    if not hasattr(classifier, "coef_"):
        raise AttributeError("This classifier does not expose coef_")
    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    frame = pd.DataFrame(
        {"feature": names, "coefficient": classifier.coef_.ravel()}
    )
    frame["abs_coefficient"] = frame["coefficient"].abs()
    return frame.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)
