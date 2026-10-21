"""Unit tests for loading a pickled LightGBM estimator from a ``.pkl`` file."""

from __future__ import annotations

import pickle
from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd
import pytest

from population_drift.analysis import (
    _copy_warnings,
    load_model,
    load_model_or_skip,
    main,
    predict_contributions,
    predict_scores,
    prepare_model_input,
    print_model_predictors,
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


def test_print_model_predictors_lists_booster_names(capsys: pytest.CaptureFixture[str]) -> None:
    """Stdout shows the model class and the booster predictor names."""
    estimator, _frame = _fitted_classifier()
    print_model_predictors(estimator, ["x"])
    out = capsys.readouterr().out
    _check("LGBMClassifier" in out, True, "printed model type")
    _check("x" in out, True, "printed booster predictor name")
    _check("Comparison feature list" in out, True, "printed comparison feature list")


def test_print_model_predictors_lists_nested_wrapper_names(capsys: pytest.CaptureFixture[str]) -> None:
    """A Model wrapper still prints the inner LightGBM predictor names."""
    estimator, _frame = _fitted_classifier()
    print_model_predictors(Model(estimator), ["z"])
    out = capsys.readouterr().out
    _check("test_model_io.Model" in out, True, "printed wrapper type")
    _check("x" in out, True, "printed inner booster predictor name")
    _check("z" in out, True, "printed comparison feature list name")


def test_prepare_model_input_does_not_require_inner_booster_names() -> None:
    """A wrapper without booster_ keeps the comparison feature list for column alignment."""
    estimator, frame = _fitted_classifier()
    table = pd.DataFrame({"z": frame["x"].to_numpy()})
    prepared = prepare_model_input(table, Model(estimator), ["z"])
    _check(list(prepared.columns), ["z"], "wrapper alignment uses the feature list")


def test_predict_contributions_uses_the_nested_lightgbm_estimator() -> None:
    """TreeSHAP is taken from the inner LightGBM estimator, not wrapper.predict()."""
    estimator, frame = _fitted_classifier()
    contrib = predict_contributions(Model(estimator), frame[["x"]])
    _check(contrib.shape, (len(frame), 1), "TreeSHAP shape from a Model wrapper")


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


def test_compare_cli_prints_model_predictors(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """The compare CLI prints booster predictor names before the analysis runs."""
    monkeypatch.setattr(
        "population_drift.analysis.run_comparison",
        lambda *args, **kwargs: {"verdict_level": "mild"},
    )
    estimator, frame = _fitted_classifier()
    train = tmp_path / "train.csv"
    new = tmp_path / "new.csv"
    frame.to_csv(train, index=False)
    frame[["x"]].to_csv(new, index=False)
    model_path = tmp_path / "model.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(estimator, handle)
    code = main([
        "compare", "--train", str(train), "--new", str(new), "--model", str(model_path),
        "--out", str(tmp_path / "out"),
    ])
    _check(code, 0, "compare exit code when printing predictors")
    out = capsys.readouterr().out
    _check("Model predictors" in out, True, "CLI printed model predictors")
    _check("x" in out, True, "CLI printed predictor name x")


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
