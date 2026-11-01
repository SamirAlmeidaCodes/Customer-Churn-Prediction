"""Train logistic regression and a tree model, then save the better one.

The table size picks a small, medium, or large regime: the holdout fraction,
the number of folds, and how the classes are balanced. Selection uses mean F1
from stratified cross-validation on the training split. ROC-AUC breaks a tie.
The holdout split is scored once and is not used to choose the saved model.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import imblearn
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)

from churn.config import (
    CATEGORICAL_FEATURES,
    CLASSIFICATION_THRESHOLD,
    DATA_PATH,
    FEATURE_COLUMNS,
    LOW_RISK_BELOW,
    MEDIUM_RISK_BELOW,
    METRICS_PATH,
    MODEL_PATH,
    NUMERIC_FEATURES,
    RANDOM_STATE,
)
from churn.data import load_raw, prepare_dataset
from churn.pipeline import build_pipeline_for_regime
from churn.predict import positive_probabilities
from churn.regimes import TrainingRegime, choose_regime


def class_balance(target: pd.Series) -> dict:
    """Summarize how many customers churn in a split."""
    counts = {int(label): int(count) for label, count in target.value_counts().items()}
    total = int(len(target))
    churners = counts.get(1, 0)
    return {
        "n": total,
        "churn": churners,
        "stay": counts.get(0, 0),
        "churn_rate": float(churners / total) if total else 0.0,
    }


def holdout_metrics(pipeline, features: pd.DataFrame, target: pd.Series) -> dict:
    """Score precision, recall, F1, and ROC-AUC on an untouched split."""
    predicted = pipeline.predict(features)
    probability = positive_probabilities(pipeline, features)
    return {
        "precision": float(precision_score(target, predicted, pos_label=1, zero_division=0)),
        "recall": float(recall_score(target, predicted, pos_label=1, zero_division=0)),
        "f1": float(f1_score(target, predicted, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(target, probability)),
    }


def cross_validated_metrics(
    pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    cv_folds: int,
) -> dict:
    """Return mean and std of the four metrics across stratified training folds."""
    scoring = {
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }
    splitter = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    scores = cross_validate(
        pipeline,
        features,
        target,
        cv=splitter,
        scoring=scoring,
        n_jobs=1,
        error_score="raise",
    )
    summary = {}
    for metric in ("precision", "recall", "f1", "roc_auc"):
        values = np.asarray(scores[f"test_{metric}"], dtype=float)
        summary[metric] = {"mean": float(values.mean()), "std": float(values.std(ddof=0))}
    return summary


def select_best_model(models: dict) -> str:
    """Pick the model with the higher mean CV F1, then the higher mean CV AUC."""
    return max(
        models,
        key=lambda name: (
            models[name]["cv"]["f1"]["mean"],
            models[name]["cv"]["roc_auc"]["mean"],
        ),
    )


def _fitted_iterations(pipeline) -> int | None:
    """Return how many trees the fitted model actually kept."""
    classifier = pipeline.named_steps["classifier"]
    inner = getattr(classifier, "estimator_", classifier)
    if type(inner).__name__ == "GradientBoostingClassifier":
        return int(inner.n_estimators_)
    if type(inner).__name__ == "HistGradientBoostingClassifier":
        return int(inner.n_iter_)
    return None


def _imbalance_block(regime: TrainingRegime) -> dict:
    if regime.use_smote:
        return {
            "method": "SMOTE",
            "applied_to": "training folds only, after encoding and scaling",
            "sampling_strategy": "auto",
            "params": {
                "random_state": RANDOM_STATE,
                "k_neighbors": regime.smote_k_neighbors,
            },
        }
    return {
        "method": "class_weight",
        "applied_to": "training folds only, inside the classifier",
        "sampling_strategy": "balanced",
        "params": {},
    }


def run_experiment(
    frame: pd.DataFrame,
    regime: TrainingRegime | None = None,
) -> dict:
    """Fit both candidates and return pipelines plus the comparison report.

    With no regime, the row count and the churner count choose one. Pass a
    regime to replay a specific recipe, including the historical fixed one.
    """
    features, target = prepare_dataset(frame)
    if target.nunique() < 2:
        raise ValueError("Training data must include both churn and stay customers")
    n_minority = int(target.sum())
    if regime is None:
        regime = choose_regime(len(target), n_minority)

    train_features, test_features, train_target, test_target = train_test_split(
        features,
        target,
        test_size=regime.test_size,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    fitted = {}
    model_reports = {}
    for name in regime.model_names:
        pipeline = build_pipeline_for_regime(name, regime)
        started = time.perf_counter()
        cv_summary = cross_validated_metrics(
            pipeline,
            train_features,
            train_target,
            regime.cv_folds,
        )
        pipeline.fit(train_features, train_target)
        fit_seconds = time.perf_counter() - started
        fitted[name] = pipeline
        model_reports[name] = {
            "cv": cv_summary,
            "holdout": holdout_metrics(pipeline, test_features, test_target),
            "fit_seconds": fit_seconds,
            "fitted_iterations": _fitted_iterations(pipeline),
        }

    best_name = select_best_model(model_reports)
    regime_report = regime.as_dict()
    regime_report["n_rows"] = int(len(frame))
    regime_report["n_minority"] = n_minority
    report = {
        "dataset": "Telco Customer Churn",
        "n_rows": int(len(frame)),
        "n_features": len(FEATURE_COLUMNS),
        "positive_class": "Churn",
        "test_size": regime.test_size,
        "random_state": RANDOM_STATE,
        "cv_folds": regime.cv_folds,
        "selection_metric": "cv_f1_mean",
        "selection_tie_break": "cv_roc_auc_mean",
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "risk_tiers": {
            "Low": f"probability < {LOW_RISK_BELOW}",
            "Medium": f"{LOW_RISK_BELOW} <= probability < {MEDIUM_RISK_BELOW}",
            "High": f"probability >= {MEDIUM_RISK_BELOW}",
        },
        "regime": regime_report,
        "imbalance": _imbalance_block(regime),
        "features": {
            "numeric": list(NUMERIC_FEATURES),
            "categorical": list(CATEGORICAL_FEATURES),
        },
        "hyperparameters": {
            "logistic_regression": dict(regime.logistic_params),
            regime.tree_model: dict(regime.tree_params),
        },
        "class_balance": {
            "all": class_balance(target),
            "train": class_balance(train_target),
            "test": class_balance(test_target),
        },
        "library_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
            "imbalanced-learn": imblearn.__version__,
        },
        "models": model_reports,
        "best_model": best_name,
    }
    return {
        "report": report,
        "pipelines": fitted,
        "best_name": best_name,
        "best_pipeline": fitted[best_name],
        "regime": regime,
        "X_train": train_features,
        "X_test": test_features,
        "y_train": train_target,
        "y_test": test_target,
    }


def _round_numbers(value):
    if isinstance(value, dict):
        return {key: _round_numbers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_round_numbers(item) for item in value]
    if isinstance(value, float):
        return round(value, 6)
    return value


def write_report(report: dict, path: Path) -> None:
    """Write one comparison report as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _round_numbers(report)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def save_artifacts(
    result: dict,
    model_path: Path = MODEL_PATH,
    metrics_path: Path = METRICS_PATH,
) -> None:
    """Write the winning pipeline with joblib and the comparison as JSON."""
    import joblib

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result["best_pipeline"], model_path)
    write_report(result["report"], metrics_path)


def format_report(report: dict) -> str:
    """Render the regime and the CV and holdout metrics as a fixed-width table."""
    regime = report.get("regime", {})
    lines = []
    if regime:
        lines.append(f"Regime: {regime['name']}. {regime['reason']}")
    lines.extend(
        [
            f"Best model: {report['best_model']} (highest mean CV F1, AUC as tie-break)",
            f"{'model':<24} {'split':<8} {'precision':>10} {'recall':>10} {'f1':>10} {'roc_auc':>10}",
        ]
    )
    for name, metrics in report["models"].items():
        for split in ("cv", "holdout"):
            block = metrics[split]

            def _value(metric: str, block: dict = block) -> float:
                raw = block[metric]
                if isinstance(raw, dict):
                    return float(raw["mean"])
                return float(raw)

            lines.append(
                f"{name:<24} {split:<8} "
                f"{_value('precision'):10.4f} {_value('recall'):10.4f} "
                f"{_value('f1'):10.4f} {_value('roc_auc'):10.4f}"
            )
        iterations = metrics.get("fitted_iterations")
        seconds = metrics.get("fit_seconds")
        if iterations is not None and seconds is not None:
            lines.append(
                f"{name}: fitted iterations {iterations}, "
                f"cross-validation and fit {seconds:.1f}s"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) > 1:
        raise SystemExit("Usage: python -m churn.train [data.csv]")
    data_path = Path(args[0]) if args else DATA_PATH
    frame = load_raw(data_path)
    result = run_experiment(frame)
    print(format_report(result["report"]))
    if data_path.resolve() == DATA_PATH.resolve():
        save_artifacts(result)
        print(f"Saved model: {MODEL_PATH}")
        print(f"Saved metrics: {METRICS_PATH}")
        return
    metrics_path = METRICS_PATH.with_name(f"{data_path.stem}_metrics.json")
    write_report(result["report"], metrics_path)
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
