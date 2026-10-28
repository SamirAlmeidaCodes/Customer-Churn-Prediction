"""Feature pipeline: impute, scale numerics, one-hot encode, then SMOTE."""

from __future__ import annotations

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn.config import (
    CATEGORICAL_FEATURES,
    GRADIENT_BOOSTING_PARAMS,
    LOGISTIC_REGRESSION_PARAMS,
    MODEL_NAMES,
    NUMERIC_FEATURES,
    SMOTE_PARAMS,
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


def build_estimator(model_name: str):
    """Return an unfitted classifier for one of the supported model names."""
    if model_name == "logistic_regression":
        return LogisticRegression(**LOGISTIC_REGRESSION_PARAMS)
    if model_name == "gradient_boosting":
        return GradientBoostingClassifier(**GRADIENT_BOOSTING_PARAMS)
    known = ", ".join(MODEL_NAMES)
    raise ValueError(f"Unknown model {model_name!r}. Expected one of: {known}")


def build_pipeline(model_name: str) -> ImbPipeline:
    """Stack preprocessing, SMOTE, and a classifier.

    imblearn's Pipeline runs the sampler during fit only, so validation rows
    and live scoring keep the original class balance.
    """
    return ImbPipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("smote", SMOTE(**SMOTE_PARAMS)),
            ("classifier", build_estimator(model_name)),
        ]
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
