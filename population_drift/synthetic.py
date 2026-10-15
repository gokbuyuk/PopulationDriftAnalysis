"""Generate a synthetic training population, a shifted new population and a LightGBM model.

The model is fitted on a separate sample, so ``train.csv`` plays the labelled holdout of the
training population (the recommended input for label-based checks).

Planted shifts in the new population (absent with --no-shift):
    f01  mean shift of +0.5 SD (important feature)
    f02  variance x1.8 (important feature)
    f05/f06  correlation changes from +0.6 to -0.2
    f07  missing rate 2% -> 15%
    f10  negligible mean shift (+0.05 SD): detectable with large n, practically irrelevant
    c01  level frequencies change
    c02  new level "Z" in 6% of rows

Usage:
    python -m population_drift.synthetic --out demo_data
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

logger = logging.getLogger("make_synthetic_data")

CONTINUOUS = [f"f{i:02d}" for i in range(1, 17)]
CATEGORICAL = ["c01", "c02", "c03", "c04"]


def make_population(n_rows: int, shifted: bool, rng: np.random.Generator) -> pd.DataFrame:
    """Draw one population with 16 continuous and 4 categorical features plus a binary label."""
    if n_rows < 100:
        raise ValueError("n_rows must be at least 100.")
    cov = np.eye(len(CONTINUOUS))
    rho_56 = -0.2 if shifted else 0.6
    cov[4, 5] = cov[5, 4] = rho_56
    cov[0, 7] = cov[7, 0] = 0.4
    x = rng.multivariate_normal(np.zeros(len(CONTINUOUS)), cov, size=n_rows)
    df = pd.DataFrame(x, columns=CONTINUOUS)
    if shifted:
        df["f01"] += 0.5
        df["f02"] *= np.sqrt(1.8)
        df["f10"] += 0.05
    df["f03"] = np.exp(df["f03"] * 0.5)
    df["f04"] = np.where(rng.random(n_rows) < 0.3, 0.0, np.abs(df["f04"]) * 10)
    missing_rate = 0.15 if shifted else 0.02
    df.loc[rng.random(n_rows) < missing_rate, "f07"] = np.nan

    c01_probs = [0.2, 0.3, 0.5] if shifted else [0.5, 0.3, 0.2]
    df["c01"] = rng.choice(["A", "B", "C"], size=n_rows, p=c01_probs)
    c02 = rng.choice(["X", "Y"], size=n_rows, p=[0.6, 0.4])
    if shifted:
        c02 = np.where(rng.random(n_rows) < 0.06, "Z", c02)
    df["c02"] = c02
    df["c03"] = rng.choice(["low", "mid", "high"], size=n_rows, p=[0.3, 0.4, 0.3])
    df["c04"] = rng.choice(["k1", "k2", "k3", "k4", "k5"], size=n_rows)

    logit = (-1.0 + 0.9 * df["f01"] - 0.7 * df["f02"] + 0.5 * df["f05"] * df["f06"]
             + 0.4 * df["f08"].clip(-2, 2) + df["c01"].map({"A": 0.0, "B": 0.4, "C": 0.8})
             + 0.3 * np.log1p(df["f04"]) / 2)
    df["target"] = (rng.random(n_rows) < 1 / (1 + np.exp(-logit))).astype(int)
    return df


def train_model(train: pd.DataFrame, seed: int) -> lgb.Booster:
    """Train a LightGBM binary classifier on the training population."""
    features = train[CONTINUOUS + CATEGORICAL].copy()
    for col in CATEGORICAL:
        features[col] = features[col].astype("category")
    model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, random_state=seed, verbose=-1)
    model.fit(features, train["target"])
    return model.booster_


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m population_drift.synthetic",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--out", default="demo_data")
    parser.add_argument("--n-train", type=int, default=6000)
    parser.add_argument("--n-new", type=int, default=2500)
    parser.add_argument("--no-shift", action="store_true", help="Draw the new population from the same distribution.")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model_rows = make_population(args.n_train, shifted=False, rng=rng)
    train = make_population(args.n_train, shifted=False, rng=rng)
    new = make_population(args.n_new, shifted=not args.no_shift, rng=rng).drop(columns="target")
    train.to_csv(out / "train.csv", index=False)
    new.to_csv(out / "new.csv", index=False)
    (out / "features.txt").write_text("\n".join(CONTINUOUS + CATEGORICAL) + "\n")
    train_model(model_rows, args.seed).save_model(str(out / "model.txt"))
    logger.info("Wrote train.csv (%d labelled holdout rows of the training population), new.csv (%d rows), "
                "features.txt and model.txt to %s", len(train), len(new), out)


if __name__ == "__main__":
    main()
