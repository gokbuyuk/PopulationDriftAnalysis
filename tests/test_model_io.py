"""Unit tests for loading a pickled LightGBM estimator from a ``.pkl`` file."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
import shap

import population_drift.analysis as analysis
from population_drift.analysis import (
    _copy_warnings,
    load_model,
    load_model_or_skip,
    log_input_features,
    main,
    predict_contributions,
    predict_scores,
    prepare_model_input,
)


class Model:
    """Scoring wrapper whose ``predict`` does not accept ``pred_contrib``."""

    def __init__(self, estimator: lgb.LGBMClassifier) -> None:
        """Store a fitted LightGBM estimator as ``model``.

        Args:
            estimator: Fitted ``LGBMClassifier``.

        Raises:
            TypeError: If ``estimator`` is None.
        """
        if estimator is None:
            raise TypeError("estimator is required.")
        self.model = estimator

    def predict(self, features: pd.DataFrame) -> object:
        """Return class labels. Extra LightGBM predict kwargs are not accepted.

        Args:
            features: Model input rows.

        Returns:
            Predicted class labels from the inner estimator.
        """
        return self.model.predict(features)

    def predict_proba(self, features: pd.DataFrame) -> object:
        """Return class probabilities from the inner estimator.

        Args:
            features: Model input rows.

        Returns:
            Class probabilities from the inner estimator.
        """
        return self.model.predict_proba(features)


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


def test_predict_contributions_uses_the_nested_lightgbm_estimator() -> None:
    """TreeSHAP is taken from the inner LightGBM estimator, not wrapper.predict()."""
    estimator, frame = _fitted_classifier()
    contrib = predict_contributions(Model(estimator), frame[["x"]])
    _check(contrib.shape, (len(frame), 1), "TreeSHAP shape from a Model wrapper")


def test_predict_contributions_reads_a_native_booster() -> None:
    """A native Booster yields TreeSHAP without sklearn predict()."""
    estimator, frame = _fitted_classifier()
    contrib = predict_contributions(estimator.booster_, frame[["x"]])
    _check(contrib.shape, (len(frame), 1), "TreeSHAP shape from a Booster")


class _FakeTreeExplainer:
    """Test double for ``shap.TreeExplainer`` that records the explained model."""

    models: list[object] = []
    values_for: Callable[[pd.DataFrame], object] | None = None

    def __init__(self, model: object) -> None:
        """Record ``model``; ``values_for`` must already be set on the class."""
        if model is None:
            raise TypeError("model is required.")
        if _FakeTreeExplainer.values_for is None:
            raise TypeError("values_for is required.")
        _FakeTreeExplainer.models.append(model)

    def shap_values(self, features: pd.DataFrame) -> object:
        """Return the scripted contribution matrix.

        Args:
            features: Rows passed to the explainer.

        Returns:
            Scripted SHAP values.

        Raises:
            TypeError: If ``features`` is not a DataFrame.
        """
        if not isinstance(features, pd.DataFrame):
            raise TypeError("features must be a DataFrame.")
        values_for = _FakeTreeExplainer.values_for
        if values_for is None:
            raise TypeError("values_for is required.")
        return values_for(features)


def _install_tree_explainer(
    monkeypatch: pytest.MonkeyPatch,
    values_for: Callable[[pd.DataFrame], object],
) -> list[object]:
    """Replace ``analysis.shap.TreeExplainer`` with ``_FakeTreeExplainer``.

    Args:
        monkeypatch: Pytest patcher.
        values_for: Returns the fake ``shap_values`` matrix for the features.

    Returns:
        Models passed to ``TreeExplainer``.

    Raises:
        TypeError: If ``values_for`` is None.
    """
    if values_for is None:
        raise TypeError("values_for is required.")
    _FakeTreeExplainer.models = []
    _FakeTreeExplainer.values_for = values_for
    monkeypatch.setattr(
        analysis, "shap", SimpleNamespace(TreeExplainer=_FakeTreeExplainer), raising=False)
    return _FakeTreeExplainer.models


def test_predict_contributions_uses_shap_tree_explainer(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Contributions come from ``shap.TreeExplainer.shap_values``, not ``pred_contrib``."""
    estimator, frame = _fitted_classifier()
    features = frame[["x"]]
    _install_tree_explainer(monkeypatch, lambda data: np.full((len(data), data.shape[1]), 0.25))
    contrib = predict_contributions(estimator, features)
    _check(contrib.tolist(), np.full((len(features), 1), 0.25).tolist(), "TreeExplainer matrix")


def test_predict_contributions_explains_the_nested_estimator(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """TreeExplainer receives the inner LightGBM estimator, not the scoring wrapper."""
    estimator, frame = _fitted_classifier()
    seen = _install_tree_explainer(
        monkeypatch, lambda data: np.zeros((len(data), data.shape[1])))
    predict_contributions(Model(estimator), frame[["x"]])
    _check([type(item).__name__ for item in seen], ["LGBMClassifier"], "explained model")


def test_predict_contributions_uses_the_positive_class_matrix(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """A class-list from TreeExplainer keeps the last (positive-class) block."""
    estimator, frame = _fitted_classifier()
    _install_tree_explainer(
        monkeypatch,
        lambda data: [np.zeros((len(data), 1)), np.ones((len(data), 1))],
    )
    contrib = predict_contributions(estimator, frame[["x"]])
    _check(contrib.tolist(), np.ones((len(frame), 1)).tolist(), "positive-class SHAP")


def test_predict_contributions_matches_shap_tree_explainer() -> None:
    """A live TreeExplainer on the estimator matches ``predict_contributions``."""
    estimator, frame = _fitted_classifier()
    features = frame[["x"]]
    raw = shap.TreeExplainer(estimator).shap_values(features)
    expected = raw[-1] if isinstance(raw, list) else np.asarray(raw)
    if getattr(expected, "ndim", 0) == 3:
        expected = expected[:, :, -1]
    actual = predict_contributions(estimator, features)
    if not np.allclose(actual, expected):
        pytest.fail("TreeExplainer values diverged from predict_contributions.")


def test_predict_contributions_names_a_scorer_without_a_booster() -> None:
    """A predict-only object is rejected by type, not by pred_contrib."""

    class Scorer:
        """Object with predict() and no LightGBM booster."""

        def predict(self, features: pd.DataFrame) -> list[int]:
            """Return a zero label per row.

            Args:
                features: Unused rows.

            Returns:
                A zero label per row.
            """
            return [0] * len(features)

    with pytest.raises(TypeError, match="Got Scorer"):
        predict_contributions(Scorer(), pd.DataFrame({"x": [0.0]}))


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


def test_load_model_or_skip_returns_none_for_a_corrupt_pickle(tmp_path: Path) -> None:
    """A set but unreadable pickle does not raise; the error is returned."""
    path = tmp_path / "model.pkl"
    path.write_bytes(b"not-a-pickle")
    model, error = load_model_or_skip(path)
    _check(model is None, True, "corrupt pickle yields no model")
    assert error is not None, "corrupt pickle must return an error message."
    _check("could not be loaded" in error, True, "error names a load failure")
    _check("Feature-distribution tests will still run" in error, True, "error says feature tests continue")


def test_load_model_or_skip_is_silent_when_no_path_is_set() -> None:
    """A blank path is a deliberate feature-only run."""
    model, error = load_model_or_skip("")
    _check(model is None, True, "blank path yields no model")
    _check(error is None, True, "blank path is not a load error")


def test_copy_warnings_rejects_a_bare_string() -> None:
    """A string is not treated as a sequence of one-character messages."""
    with pytest.raises(TypeError, match="extra_warnings must be a sequence"):
        _copy_warnings("not a list")


def test_copy_warnings_returns_an_independent_list() -> None:
    """The caller list is copied so later appends do not mutate the original."""
    original = ["first"]
    copied = _copy_warnings(original)
    original.append("second")
    _check(copied, ["first"], "copied warnings stay independent")


def test_compare_cli_still_runs_when_the_model_cannot_be_loaded(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Feature comparison is invoked with model=None and a load warning."""
    seen: dict[str, object] = {}

    def fake_compare(*args: object, extra_warnings: list[str] | None = None, **kwargs: object) -> dict[str, str]:
        seen["model"] = args[4]
        seen["warnings"] = extra_warnings
        return {"verdict_level": "mild"}

    monkeypatch.setattr("population_drift.analysis.run_comparison", fake_compare)
    train = tmp_path / "train.csv"
    new = tmp_path / "new.csv"
    pd.DataFrame({"x": [0.0, 1.0], "target": [0, 1]}).to_csv(train, index=False)
    pd.DataFrame({"x": [0.5, 1.5]}).to_csv(new, index=False)
    model_path = tmp_path / "model.pkl"
    model_path.write_bytes(b"not-a-pickle")
    code = main([
        "compare", "--train", str(train), "--new", str(new), "--model", str(model_path),
        "--out", str(tmp_path / "out"),
    ])
    _check(code, 0, "compare exit code after a model load failure")
    _check(seen["model"] is None, True, "compare is called without a model")
    warnings = seen["warnings"]
    assert isinstance(warnings, list) and warnings, "load warning was not passed to compare."
    _check("could not be loaded" in warnings[0], True, "compare receives the load warning")


def test_paired_cli_still_runs_when_the_model_cannot_be_loaded(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A paired compare also drops the model and keeps the load warning."""
    seen: dict[str, object] = {}

    def fake_paired(*args: object, extra_warnings: list[str] | None = None, **kwargs: object) -> dict[str, str]:
        seen["model"] = args[5]
        seen["warnings"] = extra_warnings
        return {"verdict_level": "mild"}

    monkeypatch.setattr("population_drift.analysis.run_paired_comparison", fake_paired)
    frame = pd.DataFrame({"x": [0.0, 1.0], "target": [0, 1]})
    train = tmp_path / "train.csv"
    test = tmp_path / "test.csv"
    new = tmp_path / "new.csv"
    frame.to_csv(train, index=False)
    frame.to_csv(test, index=False)
    frame[["x"]].to_csv(new, index=False)
    model_path = tmp_path / "model.pkl"
    model_path.write_bytes(b"not-a-pickle")
    code = main([
        "compare", "--train", str(train), "--test", str(test), "--new", str(new),
        "--model", str(model_path), "--out", str(tmp_path / "out"),
    ])
    _check(code, 0, "paired compare exit code after a model load failure")
    _check(seen["model"] is None, True, "paired compare is called without a model")
    warnings = seen["warnings"]
    assert isinstance(warnings, list) and warnings, "load warning was not passed to paired compare."


def test_log_input_features_names_each_table_and_the_model(caplog: pytest.LogCaptureFixture) -> None:
    """Each dataset's columns and the model's feature names are logged."""
    estimator, frame = _fitted_classifier()
    with caplog.at_level(logging.INFO, logger="population_drift"):
        log_input_features({"train": frame, "new": frame[["x"]]}, estimator)
    _check("train columns (2): ['x', 'y']" in caplog.text, True, "train columns logged")
    _check("new columns (1): ['x']" in caplog.text, True, "new columns logged")
    _check("Model expects (1): ['x']" in caplog.text, True, "model features logged")


def test_log_input_features_says_when_there_is_no_model(caplog: pytest.LogCaptureFixture) -> None:
    """A feature-only run still lists table columns."""
    frame = pd.DataFrame({"x": [0.0]})
    with caplog.at_level(logging.INFO, logger="population_drift"):
        log_input_features({"train": frame}, None)
    _check("Model expects: (no model)" in caplog.text, True, "no-model line")


def test_log_input_features_rejects_an_empty_table_map() -> None:
    """At least one dataset must be provided."""
    with pytest.raises(ValueError, match="tables is empty"):
        log_input_features({}, None)


def test_prepare_model_input_lists_expected_and_available_columns() -> None:
    """A missing model column names both the expected names and the table columns."""
    estimator, frame = _fitted_classifier()
    with pytest.raises(ValueError, match="Model feature names: \\['x'\\]. Data columns: \\['other'\\]"):
        prepare_model_input(pd.DataFrame({"other": [0.0]}), estimator, ["x"])
