"""Unit tests for loading a pickled LightGBM estimator from a ``.pkl`` file."""

from __future__ import annotations

import pickle
from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd
import pytest

from population_drift.analysis import load_model, predict_contributions, predict_scores


def _fitted_classifier() -> tuple[lgb.LGBMClassifier, pd.DataFrame]:
    """A tiny binary LightGBM estimator and the frame it was fitted on."""
    frame = pd.DataFrame({
        "x": [0.0, 1.0, 0.2, 0.8] * 15,
        "y": [0, 1, 0, 1] * 15,
    })
    model = lgb.LGBMClassifier(n_estimators=8, num_leaves=4, verbose=-1, random_state=0)
    model.fit(frame[["x"]], frame["y"])
    return model, frame


def _check(actual: object, expected: object, what: str) -> None:
    """Fail with the comparison that did not hold."""
    if actual != expected:
        pytest.fail(f"{what}: got {actual!r}, expected {expected!r}.")


def test_load_model_reads_a_pickle_of_a_sklearn_lightgbm_estimator(tmp_path: Path) -> None:
    """A ``.pkl`` written with pickle.dump loads and scores the same rows."""
    model, frame = _fitted_classifier()
    path = tmp_path / "model.pkl"
    with path.open("wb") as handle:
        pickle.dump(model, handle)
    loaded = load_model(path)
    scores = predict_scores(loaded, frame[["x"]], "binary")
    contrib = predict_contributions(loaded, frame[["x"]])
    _check(len(scores), len(frame), "score count from a pickled LGBMClassifier")
    _check(contrib.shape, (len(frame), 1), "TreeSHAP shape from a pickled LGBMClassifier")


def test_load_model_reads_a_joblib_dump_saved_with_a_pkl_suffix(tmp_path: Path) -> None:
    """A joblib dump named ``.pkl`` still loads after pickle.load fails."""
    model, frame = _fitted_classifier()
    path = tmp_path / "model.pkl"
    joblib.dump(model, path)
    loaded = load_model(path)
    scores = predict_scores(loaded, frame[["x"]], "binary")
    _check(len(scores), len(frame), "score count from a joblib dump named .pkl")


def test_load_model_reads_a_pickle_of_a_booster(tmp_path: Path) -> None:
    """A pickled native Booster is accepted as a ``.pkl`` model file."""
    model, frame = _fitted_classifier()
    path = tmp_path / "booster.pkl"
    with path.open("wb") as handle:
        pickle.dump(model.booster_, handle)
    loaded = load_model(path)
    scores = predict_scores(loaded, frame[["x"]], "binary")
    _check(len(scores), len(frame), "score count from a pickled Booster")


def test_load_model_rejects_an_unknown_suffix(tmp_path: Path) -> None:
    """A model file that is not text or pickle is rejected by suffix."""
    path = tmp_path / "model.bin"
    path.write_bytes(b"not-a-model")
    with pytest.raises(ValueError, match="Unsupported model file"):
        load_model(path)
