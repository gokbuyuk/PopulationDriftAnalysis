"""Example only: write a fake labelled table, a shifted unlabelled table, and a LightGBM model.

This script is not part of the population-drift analysis. ``population_drift`` does not
import it. Files are written under ``demo_data/`` by default, and that directory is gitignored.

The labelled draw is cut in generation order. The first 80% is ``train.csv`` and is the
table the example model is fitted on. The last 20% is ``evaluation.csv``. ``new.csv``
has the same columns and no label. Pass ``train.csv`` as ``--train``. Pass
``evaluation.csv`` as ``--test`` only when you want the two-file example.

Planted shifts in ``new.csv`` (absent with ``--no-shift``):
    f01  mean +0.5 standard deviations
    f02  variance x1.8
    f05/f06  correlation +0.6 to -0.2
    f07  missing rate 2% to 15%
    f10  mean +0.05 standard deviations
    c01  level frequencies change
    c02  new level "Z" in 6% of rows

Usage:
    python examples/make_synthetic_data.py --out demo_data
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

logger = logging.getLogger("example.synthetic")

CONTINUOUS = [f"f{i:02d}" for i in range(1, 17)]
CATEGORICAL = ["c01", "c02", "c03", "c04"]
FIT_FRACTION = 0.8
MIN_ROWS = 100


def make_population(n_rows: int, shifted: bool, rng: np.random.Generator) -> pd.DataFrame:
    """Draw one population with 16 continuous and 4 categorical features plus a binary label.

    Args:
        n_rows: Number of rows to draw.
        shifted: Whether to apply the planted shifts.
        rng: Seeded NumPy generator.

    Returns:
        One population, including a ``target`` column.

    Raises:
        ValueError: If ``n_rows`` is below 100.
    """
    if n_rows < MIN_ROWS:
        raise ValueError(f"n_rows must be at least {MIN_ROWS}.")
    frame = _draw_features(n_rows, shifted, rng)
    _assign_label(frame, n_rows, rng)
    return frame


def train_model(train: pd.DataFrame, seed: int) -> lgb.Booster:
    """Fit a LightGBM binary classifier on the example fit rows.

    Args:
        train: Fit rows, including ``target`` and excluding any non-feature columns.
        seed: LightGBM ``random_state``.

    Returns:
        The fitted booster.

    Raises:
        ValueError: If ``target`` is missing.
    """
    if "target" not in train.columns:
        raise ValueError("The fit table needs a target column.")
    features = train[CONTINUOUS + CATEGORICAL].copy()
    for col in CATEGORICAL:
        features[col] = features[col].astype("category")
    model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=15, random_state=seed, verbose=-1,
    )
    model.fit(features, train["target"])
    return model.booster_


def main() -> None:
    """Write the example files from the command line."""
    args = _parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    rng = np.random.default_rng(args.seed)
    labeled = make_population(args.n_train, shifted=False, rng=rng)
    new = make_population(args.n_new, shifted=not args.no_shift, rng=rng).drop(columns="target")
    _write_example(Path(args.out), labeled, new, args.seed)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python examples/make_synthetic_data.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--out", default="demo_data")
    parser.add_argument("--n-train", type=int, default=5000, help="Labelled rows before the 80/20 cut.")
    parser.add_argument("--n-new", type=int, default=700, help="Unlabelled rows written to new.csv.")
    parser.add_argument("--no-shift", action="store_true", help="Draw the new population from the same distribution.")
    parser.add_argument("--seed", type=int, default=7)
    return parser


def _draw_features(n_rows: int, shifted: bool, rng: np.random.Generator) -> pd.DataFrame:
    cov = np.eye(len(CONTINUOUS))
    cov[4, 5] = cov[5, 4] = -0.2 if shifted else 0.6
    cov[0, 7] = cov[7, 0] = 0.4
    frame = pd.DataFrame(rng.multivariate_normal(np.zeros(len(CONTINUOUS)), cov, size=n_rows), columns=CONTINUOUS)
    if shifted:
        frame["f01"] += 0.5
        frame["f02"] *= np.sqrt(1.8)
        frame["f10"] += 0.05
    frame["f03"] = np.exp(frame["f03"] * 0.5)
    frame["f04"] = np.where(rng.random(n_rows) < 0.3, 0.0, np.abs(frame["f04"]) * 10)
    frame.loc[rng.random(n_rows) < (0.15 if shifted else 0.02), "f07"] = np.nan
    frame["c01"] = rng.choice(["A", "B", "C"], size=n_rows, p=[0.2, 0.3, 0.5] if shifted else [0.5, 0.3, 0.2])
    c02 = rng.choice(["X", "Y"], size=n_rows, p=[0.6, 0.4])
    frame["c02"] = np.where(rng.random(n_rows) < 0.06, "Z", c02) if shifted else c02
    frame["c03"] = rng.choice(["low", "mid", "high"], size=n_rows, p=[0.3, 0.4, 0.3])
    frame["c04"] = rng.choice(["k1", "k2", "k3", "k4", "k5"], size=n_rows)
    return frame


def _assign_label(frame: pd.DataFrame, n_rows: int, rng: np.random.Generator) -> None:
    logit = (-1.0 + 0.9 * frame["f01"] - 0.7 * frame["f02"] + 0.5 * frame["f05"] * frame["f06"]
             + 0.4 * frame["f08"].clip(-2, 2) + frame["c01"].map({"A": 0.0, "B": 0.4, "C": 0.8})
             + 0.3 * np.log1p(frame["f04"]) / 2)
    frame["target"] = (rng.random(n_rows) < 1 / (1 + np.exp(-logit))).astype(int)


def _write_example(out: Path, labeled: pd.DataFrame, new: pd.DataFrame, seed: int) -> None:
    """Write the fit table, the later labelled slice, the new table, the feature list, and the model."""
    out.mkdir(parents=True, exist_ok=True)
    cut = int(round(len(labeled) * FIT_FRACTION))
    fit_rows, evaluation = labeled.iloc[:cut], labeled.iloc[cut:]
    if len(evaluation) < len(new) + 30:
        logger.warning(
            "evaluation.csv has %d rows and new.csv has %d. --test needs at least n_new + 30 evaluation rows.",
            len(evaluation), len(new),
        )
    fit_rows.to_csv(out / "train.csv", index=False)
    evaluation.to_csv(out / "evaluation.csv", index=False)
    new.to_csv(out / "new.csv", index=False)
    (out / "features.txt").write_text("\n".join(CONTINUOUS + CATEGORICAL) + "\n")
    train_model(fit_rows, seed).save_model(str(out / "model.txt"))
    logger.info(
        "Example files in %s: train.csv (%d fit rows), evaluation.csv (%d rows), new.csv (%d rows). "
        "These files are not part of the toolkit.",
        out, len(fit_rows), len(evaluation), len(new),
    )


if __name__ == "__main__":
    main()
