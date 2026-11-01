"""Build the small and large demo tables from the IBM telco extract.

The small file is a stratified sample of real rows. The large file draws
real rows with replacement and nudges tenure and charges, so it clears the
10,000-row regime line without being a stack of exact copies. Labels stay
with the row they were copied from.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from churn.config import DATA_PATH, ID_COLUMN, TARGET_COLUMN
from churn.data import load_raw

DEMO_SEED = 42
DEMO_SMALL_ROWS = 800
DEMO_LARGE_ROWS = 12_000
DEMO_SMALL_PATH = DATA_PATH.with_name("demo_small.csv")
DEMO_LARGE_PATH = DATA_PATH.with_name("demo_large.csv")


def make_small(
    raw: pd.DataFrame,
    n_rows: int = DEMO_SMALL_ROWS,
    seed: int = DEMO_SEED,
) -> pd.DataFrame:
    """Return a stratified sample of real customers."""
    if n_rows >= len(raw):
        raise ValueError(f"Small demo must be smaller than the source, got {n_rows}")
    sample, _rest = train_test_split(
        raw,
        train_size=n_rows,
        stratify=raw[TARGET_COLUMN],
        random_state=seed,
    )
    return sample.reset_index(drop=True)


def make_large(
    raw: pd.DataFrame,
    n_rows: int = DEMO_LARGE_ROWS,
    seed: int = DEMO_SEED,
) -> pd.DataFrame:
    """Return a perturbed expansion that keeps the source churn rate."""
    if n_rows <= len(raw):
        raise ValueError(f"Large demo must be larger than the source, got {n_rows}")
    rng = np.random.default_rng(seed)
    labels = raw[TARGET_COLUMN].astype(str)
    groups = {
        label: raw.index[labels.eq(label)].to_numpy()
        for label in ("Yes", "No")
    }
    n_churn = int(round(n_rows * len(groups["Yes"]) / len(raw)))
    n_stay = n_rows - n_churn
    chosen = np.concatenate(
        [
            rng.choice(groups["Yes"], size=n_churn, replace=True),
            rng.choice(groups["No"], size=n_stay, replace=True),
        ]
    )
    frame = raw.loc[chosen].copy().reset_index(drop=True)
    order = rng.permutation(len(frame))
    frame = frame.iloc[order].reset_index(drop=True)

    tenure = frame["tenure"].to_numpy(dtype=float)
    monthly = frame["MonthlyCharges"].to_numpy(dtype=float)
    total = pd.to_numeric(frame["TotalCharges"], errors="coerce").to_numpy(dtype=float)
    tenure_out = np.clip(np.rint(tenure + rng.normal(0, 2, size=len(frame))), 0, 72)
    monthly_out = np.clip(monthly * rng.uniform(0.93, 1.07, size=len(frame)), 0, None)
    total_out = total * rng.uniform(0.93, 1.07, size=len(frame))
    missing_total = np.isnan(total_out)
    total_out = np.where(missing_total, monthly_out * tenure_out, total_out)
    total_out = np.where(tenure_out == 0, 0.0, np.clip(total_out, 0, None))

    frame["tenure"] = tenure_out.astype(int)
    frame["MonthlyCharges"] = np.round(monthly_out, 2)
    frame["TotalCharges"] = np.round(total_out, 2)
    frame[ID_COLUMN] = [f"L{index:05d}-DEMO" for index in range(len(frame))]
    return frame.loc[:, list(raw.columns)]


def write_demo_datasets(
    source: Path = DATA_PATH,
    small_path: Path = DEMO_SMALL_PATH,
    large_path: Path = DEMO_LARGE_PATH,
) -> tuple[Path, Path]:
    """Write the small and large demo CSVs next to the source extract."""
    raw = load_raw(source)
    small_path.parent.mkdir(parents=True, exist_ok=True)
    make_small(raw).to_csv(small_path, index=False)
    make_large(raw).to_csv(large_path, index=False)
    return small_path, large_path


def main() -> None:
    small_path, large_path = write_demo_datasets()
    small = pd.read_csv(small_path)
    large = pd.read_csv(large_path)
    print(f"Wrote {small_path} ({len(small):,} rows)")
    print(f"Wrote {large_path} ({len(large):,} rows)")


if __name__ == "__main__":
    main()
