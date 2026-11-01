"""Choose the training split and model settings from the table size.

Three regimes share one rule of thumb. A small table spends more of its rows
on the holdout and uses a smaller tree model. The current telco table stays
on the original recipe, with early stopping added. A large table spends more
rows on training, uses fewer cross-validation folds, and drops SMOTE once
thousands of real churners are already present.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from churn.config import GRADIENT_BOOSTING_PARAMS, LOGISTIC_REGRESSION_PARAMS, RANDOM_STATE

SMALL_ROW_BELOW = 1_000
LARGE_ROW_ABOVE = 10_000
SMALL_MINORITY_BELOW = 150
# A 10% holdout is used only when that holdout is still a real report card.
HOLDOUT_ROW_FLOOR = 1_000
HOLDOUT_MINORITY_FLOOR = 200
# SMOTE needs several real churners in each training fold.
SMOTE_FOLD_MINORITY_FLOOR = 30
# Above this many churners in the training split, class weights replace SMOTE.
LARGE_TRAIN_MINORITY_FOR_CLASS_WEIGHT = 2_000


@dataclass(frozen=True)
class TrainingRegime:
    """One resolved training recipe."""

    name: str
    test_size: float
    cv_folds: int
    use_smote: bool
    smote_k_neighbors: int
    balance_classes: bool
    tree_model: str
    tree_params: dict
    logistic_params: dict
    reason: str
    n_rows: int | None = None
    n_minority: int | None = None

    def as_dict(self) -> dict:
        payload = {
            "name": self.name,
            "test_size": self.test_size,
            "cv_folds": self.cv_folds,
            "use_smote": self.use_smote,
            "smote_k_neighbors": self.smote_k_neighbors if self.use_smote else 0,
            "balance_classes": self.balance_classes,
            "tree_model": self.tree_model,
            "tree_params": dict(self.tree_params),
            "logistic_params": dict(self.logistic_params),
            "reason": self.reason,
        }
        if self.n_rows is not None:
            payload["n_rows"] = self.n_rows
        if self.n_minority is not None:
            payload["n_minority"] = self.n_minority
        return payload

    @property
    def model_names(self) -> tuple[str, str]:
        return ("logistic_regression", self.tree_model)


def fixed_regime() -> TrainingRegime:
    """Return the recipe this project used before the size-based regimes.

    The comparison runs use this so the adaptive path has a fixed baseline:
    an 80/20 split, 5 folds, SMOTE with 5 neighbors, and 200 boosting trees.
    """
    return TrainingRegime(
        name="fixed",
        test_size=0.2,
        cv_folds=5,
        use_smote=True,
        smote_k_neighbors=5,
        balance_classes=False,
        tree_model="gradient_boosting",
        tree_params=dict(GRADIENT_BOOSTING_PARAMS),
        logistic_params=dict(LOGISTIC_REGRESSION_PARAMS),
        reason=(
            "Fixed historical settings: an 80/20 split, 5 folds, "
            "SMOTE with 5 neighbors, and 200 gradient-boosting trees."
        ),
    )


def choose_regime(n_rows: int, n_minority: int) -> TrainingRegime:
    """Pick small, medium, or large settings from the row and churner counts."""
    if n_rows < 2:
        raise ValueError(f"Need at least 2 rows to train, got {n_rows}")
    if n_minority < 0 or n_minority > n_rows:
        raise ValueError(f"Churner count {n_minority} is outside 0..{n_rows}")

    if n_rows < SMALL_ROW_BELOW or n_minority < SMALL_MINORITY_BELOW:
        name = "small"
    elif n_rows <= LARGE_ROW_ABOVE:
        name = "medium"
    else:
        name = "large"

    test_size = _test_size(name, n_rows, n_minority)
    requested_folds = 3 if name == "large" else 5
    cv_folds = _clamp_folds(n_rows, n_minority, test_size, requested_folds)
    use_smote, k_neighbors, balance_classes = _sampling(
        name, n_minority, test_size, cv_folds
    )
    tree_model, tree_params = _tree_settings(name)
    logistic_params = dict(LOGISTIC_REGRESSION_PARAMS)
    if balance_classes and tree_model == "hist_gradient_boosting":
        tree_params = {**tree_params, "class_weight": "balanced"}
    if balance_classes:
        logistic_params["class_weight"] = "balanced"

    reason = _reason(
        name=name,
        n_rows=n_rows,
        n_minority=n_minority,
        test_size=test_size,
        cv_folds=cv_folds,
        requested_folds=requested_folds,
        use_smote=use_smote,
        k_neighbors=k_neighbors,
    )
    return TrainingRegime(
        name=name,
        test_size=test_size,
        cv_folds=cv_folds,
        use_smote=use_smote,
        smote_k_neighbors=k_neighbors,
        balance_classes=balance_classes,
        tree_model=tree_model,
        tree_params=tree_params,
        logistic_params=logistic_params,
        reason=reason,
        n_rows=n_rows,
        n_minority=n_minority,
    )


def _test_size(name: str, n_rows: int, n_minority: int) -> float:
    """Return the holdout fraction. Large tables keep 80/20 when 10% is too thin."""
    if n_rows < SMALL_ROW_BELOW:
        return 0.3
    if n_rows <= LARGE_ROW_ABOVE:
        return 0.2
    test_rows = n_rows * 0.1
    test_minority = n_minority * 0.1
    if test_rows >= HOLDOUT_ROW_FLOOR and test_minority >= HOLDOUT_MINORITY_FLOOR:
        return 0.1
    return 0.2


def _clamp_folds(n_rows: int, n_minority: int, test_size: float, requested: int) -> int:
    """Keep each fold from asking for more classes than the training split holds."""
    train_minority = n_minority * (1.0 - test_size)
    train_majority = (n_rows - n_minority) * (1.0 - test_size)
    fold_cap = math.floor(min(train_minority, train_majority))
    if fold_cap < 2:
        raise ValueError(
            "Need at least 2 customers of each class in the training split. "
            f"Estimated training counts are {train_minority:.1f} churners and "
            f"{train_majority:.1f} stayers."
        )
    return min(requested, fold_cap)


def _sampling(
    name: str, n_minority: int, test_size: float, cv_folds: int
) -> tuple[bool, int, bool]:
    """Decide SMOTE versus class weights, and the SMOTE neighbor count."""
    train_minority = n_minority * (1.0 - test_size)
    fold_fit_minority = train_minority * (cv_folds - 1) / cv_folds
    desired_k = 3 if name == "small" else 5
    if fold_fit_minority < SMOTE_FOLD_MINORITY_FLOOR:
        return False, 0, True
    if name == "large" and train_minority >= LARGE_TRAIN_MINORITY_FOR_CLASS_WEIGHT:
        return False, 0, True
    max_k = math.floor(fold_fit_minority) - 1
    k_neighbors = min(desired_k, max_k)
    if k_neighbors < 1:
        return False, 0, True
    return True, k_neighbors, False


def _tree_settings(name: str) -> tuple[str, dict]:
    if name == "small":
        return "gradient_boosting", {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 2,
            "subsample": 0.8,
            "min_samples_leaf": 5,
            "random_state": RANDOM_STATE,
        }
    if name == "medium":
        return "gradient_boosting", {
            **GRADIENT_BOOSTING_PARAMS,
            "n_iter_no_change": 10,
            "validation_fraction": 0.1,
            "tol": 1e-4,
        }
    return "hist_gradient_boosting", {
        "learning_rate": 0.05,
        "max_iter": 300,
        "max_depth": 3,
        "early_stopping": True,
        "n_iter_no_change": 10,
        "validation_fraction": 0.1,
        "random_state": RANDOM_STATE,
    }


def _reason(
    *,
    name: str,
    n_rows: int,
    n_minority: int,
    test_size: float,
    cv_folds: int,
    requested_folds: int,
    use_smote: bool,
    k_neighbors: int,
) -> str:
    train_pct = int(round((1.0 - test_size) * 100))
    test_pct = int(round(test_size * 100))
    split = f"a {train_pct}/{test_pct} split and {cv_folds} folds"
    if cv_folds != requested_folds:
        split += f" ({requested_folds} folds would be larger than the smaller class)"
    if use_smote:
        balance = f"SMOTE with {k_neighbors} neighbors"
    else:
        balance = "class weights instead of SMOTE"

    if name == "small":
        if n_rows < SMALL_ROW_BELOW:
            trigger = f"{n_rows:,} rows is under {SMALL_ROW_BELOW:,}"
        else:
            trigger = f"{n_minority:,} churners is under {SMALL_MINORITY_BELOW}"
        return (
            f"{trigger}, so the small regime uses {split}, "
            f"100 trees of depth 2, and {balance}."
        )
    if name == "medium":
        return (
            f"{n_rows:,} rows and {n_minority:,} churners sit between "
            f"{SMALL_ROW_BELOW:,} and {LARGE_ROW_ABOVE:,}, so the medium regime "
            f"uses {split}, 200 trees of depth 3 with early stopping, and {balance}."
        )

    thin_holdout = ""
    projected_minority = int(round(n_minority * 0.1))
    if test_size != 0.1:
        thin_holdout = (
            f" A 10% holdout would keep about {projected_minority:,} churners, "
            f"under {HOLDOUT_MINORITY_FLOOR}, so the split stays {train_pct}/{test_pct}."
        )
    return (
        f"{n_rows:,} rows and {n_minority:,} churners are above {LARGE_ROW_ABOVE:,}, "
        f"so the large regime uses {split}, histogram boosting capped at 300 trees, "
        f"and {balance}.{thin_holdout}"
    )
