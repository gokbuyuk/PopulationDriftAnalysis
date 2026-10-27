"""Tests for scoring a table with a fitted model outside the drift comparison."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from population_drift.analysis import prepare_model_input, predict_scores
from population_drift.batch_score import BatchScoreRequest, BatchScorer, main, run_batch_score


def _classifier() -> tuple[lgb.LGBMClassifier, pd.DataFrame]:
    """A small binary model and the frame it was fitted on.

    Returns:
        Fitted classifier and a frame with an id column, two features, and the label.
    """
    frame = pd.DataFrame({
        "id": [f"r{i}" for i in range(60)],
        "x": [0.0, 1.0, 0.2, 0.8] * 15,
        "z": [0.1, 0.2, 0.9, 0.4] * 15,
        "y": [0, 1, 0, 1] * 15,
    })
    model = lgb.LGBMClassifier(n_estimators=8, num_leaves=4, verbose=-1, random_state=0)
    model.fit(frame[["x", "z"]], frame["y"])
    return model, frame


def _regressor() -> tuple[lgb.LGBMRegressor, pd.DataFrame]:
    """A small regression model and the frame it was fitted on.

    Returns:
        Fitted regressor and a frame with one feature and the target.
    """
    frame = pd.DataFrame({"x": np.linspace(0.0, 1.0, 40), "y": np.linspace(0.0, 2.0, 40)})
    model = lgb.LGBMRegressor(n_estimators=6, num_leaves=4, verbose=-1, random_state=0)
    model.fit(frame[["x"]], frame["y"])
    return model, frame


def _dump(model: object, path: Path) -> Path:
    """Pickle ``model`` to ``path``.

    Args:
        model: Fitted estimator.
        path: Destination ``.pkl`` path.

    Returns:
        ``path``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(model, handle)
    return path


def _request(tmp_path: Path, model: object, data: pd.DataFrame, **overrides: object) -> BatchScoreRequest:
    """Write ``model`` and ``data`` and return a scoring request.

    Args:
        tmp_path: Test directory.
        model: Fitted estimator.
        data: Rows to score.
        **overrides: Request fields to replace.

    Returns:
        Request pointing at the written files.
    """
    data_path = tmp_path / "input.csv"
    data.to_csv(data_path, index=False)
    values: dict[str, object] = {
        "model_path": str(_dump(model, tmp_path / "model.pkl")),
        "data_path": str(data_path),
        "scores_path": str(tmp_path / "scores.csv"),
        "output_dir": str(tmp_path / "summary"),
    }
    values.update(overrides)
    return BatchScoreRequest(**values)  # type: ignore[arg-type]


def _check(actual: object, expected: object, what: str) -> None:
    """Fail with the comparison that did not hold.

    Args:
        actual: Observed value.
        expected: Required value.
        what: Short description of the comparison.
    """
    if actual != expected:
        pytest.fail(f"{what}: got {actual!r}, expected {expected!r}.")


def test_score_returns_one_score_per_input_row() -> None:
    """Each input row is scored in that same order."""
    model, frame = _classifier()
    data = frame.iloc[::-1][["id", "x", "z"]].reset_index(drop=True)
    result = BatchScorer(model).score(data, id_columns=["id"])
    expected = predict_scores(model, prepare_model_input(data, model, ["x", "z"]), "binary")
    _check(list(result.scores.columns), ["row_index", "id", "score"], "score columns")
    _check(result.scores["row_index"].tolist(), list(range(len(data))), "row positions")
    _check(result.scores["id"].tolist(), data["id"].tolist(), "id values")
    if not np.allclose(result.scores["score"].to_numpy(), expected):
        pytest.fail("scores diverged from predict_scores.")


def test_score_writes_distribution_importance_and_plot(tmp_path: Path) -> None:
    """A path run saves row scores, the distribution, and SHAP importance."""
    model, frame = _classifier()
    data = frame[["id", "x", "z"]]
    request = _request(tmp_path, model, data, id_columns=("id",))
    result = run_batch_score(request)
    saved = pd.read_csv(request.scores_path)
    distribution = pd.read_csv(tmp_path / "summary" / "score_distribution.csv")
    if not np.allclose(saved["score"].to_numpy(), result.scores["score"].to_numpy(), rtol=0, atol=1e-15):
        pytest.fail("saved scores diverged from the returned scores.")
    histogram = pd.read_csv(tmp_path / "summary" / "score_histogram.csv")
    importance = pd.read_csv(tmp_path / "summary" / "feature_importance.csv")
    metadata = json.loads((tmp_path / "summary" / "scoring.json").read_text(encoding="utf-8"))
    plot = tmp_path / "summary" / "score_distribution.png"
    _check(int(distribution.loc[0, "n"]), len(data), "distribution row count")
    _check(float(distribution.loc[0, "positive_rate"]), float((result.scores["score"] >= 0.5).mean()), "positive rate")
    _check(abs(float(histogram["share"].sum()) - 1.0) < 1e-9, True, "histogram shares sum to 1")
    _check(int(histogram["count"].sum()), len(data), "histogram counts")
    _check(abs(float(importance["shap_share"].sum()) - 1.0) < 1e-9, True, "SHAP shares sum to 1")
    _check(int(importance.loc[importance["rank"].idxmin(), "rank"]), 1, "top rank is 1")
    _check(int(importance["n_explained"].iloc[0]), len(data), "every row is explained")
    _check(set(importance["feature"]), {"x", "z"}, "importance features")
    _check(metadata["n_rows"], len(data), "metadata row count")
    _check(metadata["problem_type"], "binary", "metadata problem type")
    _check(plot.is_file() and plot.read_bytes().startswith(b"\x89PNG"), True, "PNG plot")


def test_shap_importance_is_capped_at_max_shap_rows(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Feature importance explains at most ``max_shap_rows`` rows; scores stay complete."""
    model, frame = _classifier()
    seen: list[int] = []

    def fake_contributions(fitted: object, features: pd.DataFrame) -> np.ndarray:
        """Record the explained row count and return a fixed contribution matrix.

        Args:
            fitted: Unused model.
            features: Rows passed to TreeSHAP.

        Returns:
            One column of ones per feature.
        """
        if not isinstance(features, pd.DataFrame):
            raise TypeError("features must be a DataFrame.")
        seen.append(len(features))
        return np.ones((len(features), features.shape[1]))

    monkeypatch.setattr("population_drift.analysis.predict_contributions", fake_contributions)
    result = run_batch_score(_request(tmp_path, model, frame[["id", "x", "z"]], max_shap_rows=7))
    _check(seen, [7], "explained row count")
    _check(len(result.scores), len(frame), "every row is still scored")
    _check(int(result.importance["n_explained"].iloc[0]), 7, "importance records the cap")


def test_missing_sentinels_score_like_missing_values(tmp_path: Path) -> None:
    """A sentinel code is scored as missing, matching an explicit NaN."""
    model, frame = _classifier()
    raw = frame[["x", "z"]].copy()
    raw.loc[0, "x"] = -999
    request = _request(tmp_path, model, raw, missing_sentinels=(-999.0,))
    scored = run_batch_score(request)
    with_nan = raw.copy()
    with_nan.loc[0, "x"] = np.nan
    expected = BatchScorer(model).score(with_nan).scores["score"].to_numpy()
    if not np.allclose(scored.scores["score"].to_numpy(), expected):
        pytest.fail("sentinel scores diverged from NaN scores.")


def test_regression_distribution_has_no_positive_rate(tmp_path: Path) -> None:
    """Regression output is the prediction, without a class rate."""
    model, frame = _regressor()
    result = run_batch_score(_request(
        tmp_path, model, frame[["x"]], problem_type="regression", scores_path=str(tmp_path / "scores.parquet"),
    ))
    saved = pd.read_parquet(tmp_path / "scores.parquet")
    _check("positive_rate" in result.distribution.columns, False, "no positive rate column")
    _check(len(saved), len(frame), "parquet row count")
    if not np.allclose(saved["score"].to_numpy(), result.scores["score"].to_numpy()):
        pytest.fail("parquet scores diverged from the returned scores.")


def test_constant_scores_use_one_histogram_bin() -> None:
    """Identical scores are one bin covering every row."""
    model, frame = _classifier()
    data = frame.iloc[[0, 0]][["x", "z"]].reset_index(drop=True)
    histogram = BatchScorer(model, n_bins=10).score(data).histogram
    _check(len(histogram), 1, "one bin")
    _check(int(histogram.loc[0, "count"]), 2, "both rows")
    _check(float(histogram.loc[0, "share"]), 1.0, "full share")


def test_declared_features_must_cover_the_model() -> None:
    """An explicit feature list that drops a model column is rejected."""
    model, frame = _classifier()
    with pytest.raises(ValueError, match="Feature list is missing model columns"):
        BatchScorer(model).score(frame[["x", "z"]], feature_columns=["x"])


def test_empty_table_is_rejected() -> None:
    """Scoring needs at least one row."""
    model, _frame = _classifier()
    empty = pd.DataFrame({"x": pd.Series(dtype=float), "z": pd.Series(dtype=float)})
    with pytest.raises(ValueError, match="at least one row"):
        BatchScorer(model).score(empty)


def test_missing_model_column_is_rejected() -> None:
    """A table without a model feature fails before scoring."""
    model, _frame = _classifier()
    with pytest.raises(ValueError, match="missing from the data"):
        BatchScorer(model).score(pd.DataFrame({"x": [0.1, 0.2]}))


def test_id_column_cannot_reuse_the_score_name() -> None:
    """``score`` and ``row_index`` stay reserved for the output."""
    model, frame = _classifier()
    renamed = frame.rename(columns={"id": "score"})
    with pytest.raises(ValueError, match="clashes"):
        BatchScorer(model).score(renamed, id_columns=["score"])


def test_request_rejects_an_unknown_problem_type(tmp_path: Path) -> None:
    """Only binary and regression models are scored."""
    model, frame = _classifier()
    with pytest.raises(ValueError, match="problem_type"):
        _request(tmp_path, model, frame[["x", "z"]], problem_type="multiclass")


def test_request_rejects_an_unsupported_scores_suffix(tmp_path: Path) -> None:
    """Row scores are written as CSV, Parquet, or Feather."""
    model, frame = _classifier()
    with pytest.raises(ValueError, match="Unsupported table type"):
        _request(tmp_path, model, frame[["x", "z"]], scores_path=str(tmp_path / "scores.txt"))


def test_cli_scores_a_csv(tmp_path: Path) -> None:
    """The module CLI writes scores and the summary directory."""
    model, frame = _classifier()
    model_path = _dump(model, tmp_path / "model.pkl")
    data_path = tmp_path / "input.csv"
    frame[["id", "x", "z"]].to_csv(data_path, index=False)
    scores_path = tmp_path / "scores.csv"
    code = main([
        "--model", str(model_path),
        "--data", str(data_path),
        "--scores", str(scores_path),
        "--out", str(tmp_path / "summary"),
        "--id-cols", "id",
    ])
    _check(code, 0, "exit code")
    saved = pd.read_csv(scores_path)
    _check(list(saved.columns), ["row_index", "id", "score"], "cli score columns")
    _check((tmp_path / "summary" / "feature_importance.csv").is_file(), True, "cli importance file")


def test_cli_returns_2_when_the_model_file_is_missing(tmp_path: Path) -> None:
    """A missing model path is an input error, not an uncaught exception."""
    data_path = tmp_path / "input.csv"
    pd.DataFrame({"x": [0.0]}).to_csv(data_path, index=False)
    code = main([
        "--model", str(tmp_path / "missing.pkl"),
        "--data", str(data_path),
        "--scores", str(tmp_path / "scores.csv"),
        "--out", str(tmp_path / "summary"),
    ])
    _check(code, 2, "exit code for a missing model")


def test_cli_requires_a_scores_path(tmp_path: Path) -> None:
    """``--scores`` is required."""
    with pytest.raises(SystemExit):
        main(["--model", "model.pkl", "--data", "input.csv", "--out", str(tmp_path)])
