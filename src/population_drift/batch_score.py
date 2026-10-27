"""Score a new table with a fitted LightGBM model.

This script is separate from the drift comparison. It writes one score per input
row, plus the score distribution and TreeSHAP feature importance.

Edit ``examples/batch_score.json`` and run:

    python -m population_drift.batch_score --config examples/batch_score.json

Flags on the command line replace the same field from the config file.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from population_drift.analysis import (  # noqa: E402
    Thresholds,
    _parse_sentinels,
    _split_list,
    load_model,
    load_table,
    log_input_features,
    model_feature_names,
    prepare_model_input,
    predict_scores,
    shap_feature_importance,
    write_json,
)

logger = logging.getLogger("population_drift.batch_score")

_RESERVED_COLUMNS = frozenset({"row_index", "score"})
_TABLE_SUFFIXES = frozenset({".csv", ".parquet", ".pq", ".feather"})
_PROBLEM_TYPES = frozenset({"binary", "regression"})
_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR"})
_CONFIG_FIELDS = frozenset({
    "model_path", "data_path", "scores_path", "output_dir", "problem_type", "threshold",
    "missing_sentinels", "id_columns", "feature_columns", "features_file",
    "max_shap_rows", "seed", "n_bins", "log_level",
})
_QUANTILES = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
_QUANTILE_NAMES = ("p01", "p05", "p25", "p50", "p75", "p95", "p99")


def _require_path(value: object, name: str) -> str:
    """Return a non-empty path string.

    Args:
        value: Candidate path.
        name: Field name used in the error.

    Returns:
        The path as a string.

    Raises:
        ValueError: If ``value`` is blank or not a path.
    """
    if isinstance(value, Path):
        value = str(value)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty path.")
    return value


def _require_int(value: object, name: str, minimum: int) -> int:
    """Return an integer at least ``minimum``.

    Args:
        value: Candidate integer.
        name: Field name used in the error.
        minimum: Inclusive lower bound.

    Returns:
        The integer.

    Raises:
        ValueError: If ``value`` is not an integer or is below ``minimum``.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def _require_seed(value: object) -> int:
    """Return an integer seed.

    Args:
        value: Candidate seed.

    Returns:
        The seed.

    Raises:
        ValueError: If ``value`` is not an integer.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("seed must be an integer.")
    return int(value)


def _require_finite_number(value: object, name: str) -> float:
    """Return a finite float.

    Args:
        value: Candidate number.
        name: Field name used in the error.

    Returns:
        The value as a float.

    Raises:
        ValueError: If ``value`` is not a finite number.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite number.")
    return float(value)


def _require_table_suffix(path: str) -> None:
    """Reject a score path that is not CSV, Parquet, or Feather.

    Args:
        path: Score file path.

    Raises:
        ValueError: If the suffix is unsupported.
    """
    suffix = Path(path).suffix.lower()
    if suffix not in _TABLE_SUFFIXES:
        raise ValueError(f"Unsupported table type '{suffix}'. Use .csv, .parquet or .feather.")


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    """Raise if ``frame`` lacks any of ``columns``.

    Args:
        frame: Table to check.
        columns: Required column names.
        name: Table name used in the error.

    Raises:
        ValueError: If a required column is absent.
    """
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}.")


def _as_sentinels(value: object) -> tuple[Any, ...]:
    """Copy missing-value codes.

    Args:
        value: Sequence of sentinel codes.

    Returns:
        The codes as a tuple.

    Raises:
        TypeError: If ``value`` is a string or not a sequence.
    """
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("missing_sentinels must be a sequence of codes.")
    return tuple(value)


def _as_name_tuple(value: Sequence[str], name: str) -> tuple[str, ...]:
    """Copy a sequence of non-empty column names.

    Args:
        value: Column names.
        name: Field name used in the error.

    Returns:
        The names as a tuple.

    Raises:
        TypeError: If ``value`` is not a sequence of strings.
        ValueError: If a name is blank or repeated.
    """
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence of column names.")
    names = tuple(value)
    if any(not isinstance(item, str) or not item.strip() for item in names):
        raise ValueError(f"{name} must contain non-empty strings.")
    duplicates = sorted({item for item in names if names.count(item) > 1})
    if duplicates:
        raise ValueError(f"Duplicated {name}: {duplicates}.")
    return names


def _optional_names(value: Sequence[str] | None, name: str) -> tuple[str, ...] | None:
    """Copy an optional column list. An empty list is rejected.

    Args:
        value: Column names, or None when the caller did not set the list.
        name: Field name used in the error.

    Returns:
        The names, or None.

    Raises:
        TypeError: If ``value`` is not a sequence of strings.
        ValueError: If the list is empty, blank, or contains duplicates.
    """
    if value is None:
        return None
    names = _as_name_tuple(value, name)
    if not names:
        raise ValueError(f"{name} is empty.")
    return names


def _reject_reserved_ids(id_columns: Sequence[str]) -> None:
    """Reject id columns that collide with score-file columns.

    Args:
        id_columns: Columns to copy onto the score file.

    Raises:
        ValueError: If a name is ``row_index`` or ``score``.
    """
    reserved = [column for column in id_columns if column in _RESERVED_COLUMNS]
    if reserved:
        raise ValueError(f"id column {reserved[0]!r} clashes with an output column.")


def _require_finite_scores(scores: np.ndarray, n_rows: int) -> np.ndarray:
    """Return one finite score per input row.

    Args:
        scores: Model output.
        n_rows: Expected length.

    Returns:
        A 1-D float array of length ``n_rows``.

    Raises:
        ValueError: If the shape is wrong or any score is non-finite.
    """
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or len(values) != n_rows:
        raise ValueError("Model scores must be one finite value per input row.")
    if not np.isfinite(values).all():
        raise ValueError("Model scores contain non-finite values.")
    return values


@dataclass
class BatchScoreRequest:
    """Paths and settings for scoring one file.

    Args:
        model_path: Fitted LightGBM model.
        data_path: Table to score.
        scores_path: Destination for one score per row.
        output_dir: Directory for the distribution, importance, plot, and metadata.
        problem_type: ``binary`` or ``regression``.
        threshold: Positive-class cutoff. Used when ``problem_type`` is binary.
        missing_sentinels: Codes treated as missing while loading ``data_path``.
        id_columns: Columns copied onto the score file.
        feature_columns: Explicit feature list. Omit to use the model's stored names.
        max_shap_rows: Maximum rows used for TreeSHAP importance.
        seed: Seed for the SHAP subsample.
        n_bins: Histogram bins when the scores are not constant.
    """

    model_path: str
    data_path: str
    scores_path: str
    output_dir: str
    problem_type: str = "binary"
    threshold: float = 0.5
    missing_sentinels: tuple[Any, ...] = ()
    id_columns: tuple[str, ...] = ()
    feature_columns: tuple[str, ...] | None = None
    max_shap_rows: int = 5000
    seed: int = 42
    n_bins: int = 20

    def __post_init__(self) -> None:
        """Reject a request the scorer cannot run.

        Raises:
            TypeError: If a column list or sentinel list has the wrong type.
            ValueError: If a path, problem type, or numeric setting is invalid.
        """
        self.model_path = _require_path(self.model_path, "model_path")
        self.data_path = _require_path(self.data_path, "data_path")
        self.scores_path = _require_path(self.scores_path, "scores_path")
        self.output_dir = _require_path(self.output_dir, "output_dir")
        _require_table_suffix(self.scores_path)
        if self.problem_type not in _PROBLEM_TYPES:
            raise ValueError("problem_type must be 'binary' or 'regression'.")
        self.threshold = _require_finite_number(self.threshold, "threshold")
        self._check_threshold()
        self.max_shap_rows = _require_int(self.max_shap_rows, "max_shap_rows", 1)
        self.seed = _require_seed(self.seed)
        self.n_bins = _require_int(self.n_bins, "n_bins", 2)
        self.missing_sentinels = _as_sentinels(self.missing_sentinels)
        self.id_columns = _as_name_tuple(self.id_columns, "id_columns")
        self.feature_columns = _optional_names(self.feature_columns, "feature_columns")
        _reject_reserved_ids(self.id_columns)

    def _check_threshold(self) -> None:
        """Require a binary cutoff on the probability scale.

        Raises:
            ValueError: If a binary threshold is outside ``[0, 1]``.
        """
        if self.problem_type == "binary" and not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1 for a binary model.")


@dataclass
class BatchScoreResult:
    """Tables produced by one scoring run.

    Args:
        scores: One row per input row, with ``row_index`` and ``score``.
        distribution: One-row score summary.
        histogram: Score bins.
        importance: TreeSHAP importance, largest mean absolute SHAP first.
    """

    scores: pd.DataFrame
    distribution: pd.DataFrame
    histogram: pd.DataFrame
    importance: pd.DataFrame

    def __post_init__(self) -> None:
        """Reject tables that do not match the scoring contract.

        Raises:
            TypeError: If a field is not a DataFrame.
            ValueError: If a required column is missing.
        """
        _check_result_frame(self.scores, "scores", ("row_index", "score"))
        _check_result_frame(self.distribution, "distribution", ("n", "mean", "std", "min", "max"))
        _check_result_frame(self.histogram, "histogram", ("bin_left", "bin_right", "count", "share"))
        _check_result_frame(
            self.importance, "importance",
            ("feature", "mean_abs_shap", "shap_share", "rank", "n_explained"),
        )


def _check_result_frame(frame: object, name: str, columns: Sequence[str]) -> None:
    """Check one result table.

    Args:
        frame: Candidate table.
        name: Field name used in the error.
        columns: Required columns.

    Raises:
        TypeError: If ``frame`` is not a DataFrame.
        ValueError: If a required column is missing.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame.")
    _require_columns(frame, columns, name)


class BatchScorer:
    """Score every row of a table and summarize the scores with TreeSHAP."""

    def __init__(
        self,
        model: Any,
        *,
        problem_type: str = "binary",
        threshold: float = 0.5,
        max_shap_rows: int = 5000,
        seed: int = 42,
        n_bins: int = 20,
    ) -> None:
        """Store a fitted model and the scoring settings.

        Args:
            model: LightGBM estimator, Booster, or scoring wrapper.
            problem_type: ``binary`` or ``regression``.
            threshold: Positive-class cutoff for binary scores.
            max_shap_rows: Maximum rows used for TreeSHAP importance.
            seed: Seed for the SHAP subsample.
            n_bins: Histogram bins when scores vary.

        Raises:
            TypeError: If ``model`` is missing.
            ValueError: If a setting is invalid.
        """
        if model is None:
            raise TypeError("model is required.")
        if problem_type not in _PROBLEM_TYPES:
            raise ValueError("problem_type must be 'binary' or 'regression'.")
        self.model = model
        self.problem_type = problem_type
        self.threshold = _require_finite_number(threshold, "threshold")
        self._check_threshold()
        self.max_shap_rows = _require_int(max_shap_rows, "max_shap_rows", 1)
        self.seed = _require_seed(seed)
        self.n_bins = _require_int(n_bins, "n_bins", 2)
        self._near_band = float(Thresholds().near_threshold_band)

    def score(
        self,
        data: pd.DataFrame,
        *,
        id_columns: Sequence[str] = (),
        feature_columns: Sequence[str] | None = None,
    ) -> BatchScoreResult:
        """Score ``data`` and build the distribution and importance tables.

        Args:
            data: Table containing the model features.
            id_columns: Columns copied onto the score table.
            feature_columns: Explicit feature list. Omit to use names stored on the model.

        Returns:
            Per-row scores, distribution, histogram, and SHAP importance.

        Raises:
            TypeError: If ``data`` is not a DataFrame or a column list is invalid.
            ValueError: If the table is empty, columns are missing, or scores are non-finite.
        """
        ids = _as_name_tuple(id_columns, "id_columns")
        features = _optional_names(feature_columns, "feature_columns")
        self._validate_frame(data, ids, features)
        matrix = prepare_model_input(data, self.model, self._fallback_features(data, ids, features))
        values = _require_finite_scores(predict_scores(self.model, matrix, self.problem_type), len(data))
        return BatchScoreResult(
            scores=_score_table(data, values, ids),
            distribution=_score_distribution(values, self.problem_type, self.threshold, self._near_band),
            histogram=_score_histogram(values, self.n_bins),
            importance=_ranked_importance(self.model, matrix, self.max_shap_rows, self.seed),
        )

    def _check_threshold(self) -> None:
        """Require a binary cutoff on the probability scale.

        Raises:
            ValueError: If a binary threshold is outside ``[0, 1]``.
        """
        if self.problem_type == "binary" and not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1 for a binary model.")

    def _validate_frame(
        self,
        data: pd.DataFrame,
        id_columns: tuple[str, ...],
        feature_columns: tuple[str, ...] | None,
    ) -> None:
        """Reject a table that cannot be scored.

        Args:
            data: Candidate rows.
            id_columns: Columns that must exist on ``data``.
            feature_columns: Explicit feature list, if the caller set one.

        Raises:
            TypeError: If ``data`` is not a DataFrame.
            ValueError: If the table is empty or columns are missing.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"data must be a pandas DataFrame, got {type(data).__name__}.")
        if len(data) == 0:
            raise ValueError("data must contain at least one row.")
        if data.columns.duplicated().any():
            raise ValueError("data has duplicated column names.")
        _reject_reserved_ids(id_columns)
        missing_ids = [column for column in id_columns if column not in data.columns]
        if missing_ids:
            raise ValueError(f"id columns missing from the data: {missing_ids}.")
        self._check_declared_features(feature_columns)

    def _check_declared_features(self, feature_columns: tuple[str, ...] | None) -> None:
        """Require an explicit feature list to cover named model inputs.

        Args:
            feature_columns: Caller-supplied names, or None to use the model names.

        Raises:
            ValueError: If the list omits a feature the model named.
        """
        if feature_columns is None:
            return
        expected = model_feature_names(self.model)
        if not expected or all(name.startswith("Column_") for name in expected):
            return
        missing = [name for name in expected if name not in feature_columns]
        if missing:
            raise ValueError(f"Feature list is missing model columns: {missing}.")

    def _fallback_features(
        self,
        data: pd.DataFrame,
        id_columns: tuple[str, ...],
        feature_columns: tuple[str, ...] | None,
    ) -> list[str]:
        """Feature order used when the model did not store names.

        Args:
            data: Table being scored.
            id_columns: Columns left out of the fallback list.
            feature_columns: Explicit list, when the caller provided one.

        Returns:
            Feature names in the order ``prepare_model_input`` should try.
        """
        if feature_columns is not None:
            return list(feature_columns)
        skipped = set(id_columns)
        return [column for column in data.columns if column not in skipped]


def _score_table(data: pd.DataFrame, scores: np.ndarray, id_columns: Sequence[str]) -> pd.DataFrame:
    """Row index, requested ids, and the score, in input order.

    Args:
        data: Scored table.
        scores: One score per row of ``data``.
        id_columns: Columns copied through from ``data``.

    Returns:
        Frame with ``row_index``, the id columns, and ``score``.
    """
    table = pd.DataFrame({"row_index": np.arange(len(data), dtype=int)})
    for column in id_columns:
        table[column] = data[column].to_numpy()
    table["score"] = scores
    return table


def _binary_rate_columns(scores: np.ndarray, threshold: float, near_band: float) -> dict[str, float]:
    """Positive rate and near-threshold share for binary scores.

    Args:
        scores: Finite predicted probabilities.
        threshold: Operating cutoff.
        near_band: Half-width of the near-threshold band.

    Returns:
        ``threshold``, ``positive_rate``, and ``near_threshold_share``.
    """
    return {
        "threshold": float(threshold),
        "positive_rate": float(np.mean(scores >= threshold)),
        "near_threshold_share": float(np.mean(np.abs(scores - threshold) <= near_band)),
    }


def _score_distribution(
    scores: np.ndarray, problem_type: str, threshold: float, near_band: float,
) -> pd.DataFrame:
    """One-row summary of a score vector.

    ``std`` is the population standard deviation of this batch (divisor ``n``).

    Args:
        scores: Finite scores.
        problem_type: ``binary`` or ``regression``.
        threshold: Operating cutoff. Used for the binary positive rate.
        near_band: Absolute band around ``threshold`` counted as near-threshold.

    Returns:
        Quantiles of the scores. Binary rows also include the positive rate.
    """
    quantiles = np.quantile(scores, _QUANTILES)
    row: dict[str, float | int] = {
        "n": int(len(scores)),
        "mean": float(scores.mean()),
        "std": float(scores.std(ddof=0)),
        "min": float(scores.min()),
        "max": float(scores.max()),
    }
    row.update({name: float(value) for name, value in zip(_QUANTILE_NAMES, quantiles, strict=True)})
    if problem_type == "binary":
        row.update(_binary_rate_columns(scores, threshold, near_band))
    return pd.DataFrame([row])


def _score_histogram(scores: np.ndarray, n_bins: int) -> pd.DataFrame:
    """Equal-width score bins. A constant score is one bin.

    Args:
        scores: Finite scores, one per row.
        n_bins: Bin count used when the scores are not constant.

    Returns:
        Columns ``bin_left``, ``bin_right``, ``count``, and ``share``.
    """
    low, high = float(scores.min()), float(scores.max())
    if low == high:
        return pd.DataFrame([{
            "bin_left": low, "bin_right": high, "count": int(len(scores)), "share": 1.0,
        }])
    edges = np.linspace(low, high, n_bins + 1)
    counts, edges = np.histogram(scores, bins=edges)
    return pd.DataFrame({
        "bin_left": edges[:-1],
        "bin_right": edges[1:],
        "count": counts.astype(int),
        "share": counts / len(scores),
    })


def _ranked_importance(model: Any, features: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    """SHAP importance with rank 1 on the largest mean absolute contribution.

    Args:
        model: Fitted model.
        features: Prepared model matrix.
        max_rows: TreeSHAP row cap.
        seed: Subsample seed.

    Returns:
        Importance table plus ``rank`` and ``n_explained``.
    """
    table = shap_feature_importance(model, features, max_rows=max_rows, seed=seed).copy()
    table["rank"] = table["mean_abs_shap"].rank(ascending=False, method="min").astype(int)
    table["n_explained"] = min(len(features), max_rows)
    if len(features) > max_rows:
        logger.info(
            "TreeSHAP importance uses %d of %d rows (max_shap_rows=%d, seed=%d).",
            max_rows, len(features), max_rows, seed,
        )
    return table


class ScoreWriter:
    """Write row scores and the distribution, histogram, importance, and plot."""

    def __init__(self, scores_path: str | Path, output_dir: str | Path) -> None:
        """Store the output locations.

        Args:
            scores_path: Per-row score file. The suffix selects CSV, Parquet, or Feather.
            output_dir: Directory for the distribution, importance, plot, and metadata.

        Raises:
            ValueError: If a path is blank or the score suffix is unsupported.
        """
        self.scores_path = Path(_require_path(scores_path, "scores_path"))
        self.output_dir = Path(_require_path(output_dir, "output_dir"))
        _require_table_suffix(str(self.scores_path))

    def write(self, result: BatchScoreResult, metadata: Mapping[str, Any]) -> None:
        """Write every scoring artifact.

        Args:
            result: Tables from ``BatchScorer.score``.
            metadata: JSON-serialisable run description stored as ``scoring.json``.

        Raises:
            TypeError: If ``result`` or ``metadata`` has the wrong type.
            ValueError: If the score path suffix is unsupported.
        """
        if not isinstance(result, BatchScoreResult):
            raise TypeError("result must be a BatchScoreResult.")
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping.")
        _write_table(result.scores, self.scores_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        _write_table(result.distribution, self.output_dir / "score_distribution.csv")
        _write_table(result.histogram, self.output_dir / "score_histogram.csv")
        _write_table(result.importance, self.output_dir / "feature_importance.csv")
        _save_score_plot(result.histogram, self.output_dir / "score_distribution.png", metadata)
        write_json(dict(metadata), self.output_dir / "scoring.json")


def _write_table(frame: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame as CSV, Parquet, or Feather.

    Args:
        frame: Table to write.
        path: Destination. The parent directory is created.

    Raises:
        ValueError: If the suffix is not supported.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        # 17 significant digits round-trips float64, so reloaded scores match the run.
        frame.to_csv(path, index=False, float_format="%.17g")
        return
    if suffix in {".parquet", ".pq"}:
        frame.to_parquet(path, index=False)
        return
    if suffix == ".feather":
        frame.to_feather(path)
        return
    raise ValueError(f"Unsupported table type '{suffix}'. Use .csv, .parquet or .feather.")


def _save_score_plot(histogram: pd.DataFrame, path: Path, metadata: Mapping[str, Any]) -> None:
    """Save a histogram of score counts.

    Args:
        histogram: Bins from the scoring run.
        path: PNG destination.
        metadata: May include ``problem_type`` and ``threshold`` for a cutoff line.

    Raises:
        ValueError: If the histogram has no rows.
    """
    if histogram.empty:
        raise ValueError("histogram is empty.")
    figure, axis = plt.subplots(figsize=(6, 4))
    try:
        _draw_histogram(axis, histogram)
        _draw_threshold(axis, metadata)
        axis.set_xlabel("score")
        axis.set_ylabel("rows")
        axis.set_title("Score distribution")
        figure.tight_layout()
        figure.savefig(path, dpi=120)
    finally:
        plt.close(figure)


def _draw_histogram(axis: plt.Axes, histogram: pd.DataFrame) -> None:
    """Draw histogram bars.

    Args:
        axis: Target axes.
        histogram: Bins with ``bin_left``, ``bin_right``, and ``count``.
    """
    left = histogram["bin_left"].to_numpy(dtype=float)
    right = histogram["bin_right"].to_numpy(dtype=float)
    width = right - left
    if np.all(width == 0):
        axis.bar(left, histogram["count"], width=0.1, align="center", color="#334155")
        return
    axis.bar(left, histogram["count"], width=width, align="edge", color="#334155")


def _draw_threshold(axis: plt.Axes, metadata: Mapping[str, Any]) -> None:
    """Draw the binary operating cutoff when the metadata includes one.

    Args:
        axis: Target axes.
        metadata: Run description.
    """
    if metadata.get("problem_type") != "binary":
        return
    threshold = metadata.get("threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not np.isfinite(threshold):
        return
    axis.axvline(float(threshold), color="#b45309", linestyle="--", label=f"threshold {float(threshold):g}")
    axis.legend()


def _scoring_metadata(request: BatchScoreRequest, result: BatchScoreResult) -> dict[str, Any]:
    """JSON description of one scoring run.

    Args:
        request: Paths and settings.
        result: Tables just scored.

    Returns:
        Metadata written to ``scoring.json``.
    """
    distribution = result.distribution.iloc[0]
    metadata: dict[str, Any] = {
        "model_path": request.model_path,
        "data_path": request.data_path,
        "scores_path": request.scores_path,
        "output_dir": request.output_dir,
        "n_rows": int(distribution["n"]),
        "n_features": int(len(result.importance)),
        "n_explained": int(result.importance["n_explained"].iloc[0]),
        "problem_type": request.problem_type,
        "threshold": request.threshold,
        "seed": request.seed,
        "max_shap_rows": request.max_shap_rows,
        "n_bins": request.n_bins,
        "mean_score": float(distribution["mean"]),
        "id_columns": list(request.id_columns),
        "feature_columns": None if request.feature_columns is None else list(request.feature_columns),
    }
    if request.problem_type == "binary":
        metadata["positive_rate"] = float(distribution["positive_rate"])
    return metadata


def run_batch_score(request: BatchScoreRequest) -> BatchScoreResult:
    """Load a model and a table, score every row, and write the artifacts.

    Args:
        request: Model path, data path, and output paths.

    Returns:
        The in-memory scores and summaries. The same tables are written to disk.

    Raises:
        TypeError: If ``request`` is not a ``BatchScoreRequest``.
        FileNotFoundError: If the model or data file does not exist.
        ValueError: If the table cannot be scored.
    """
    if not isinstance(request, BatchScoreRequest):
        raise TypeError("request must be a BatchScoreRequest.")
    model = load_model(request.model_path)
    data = load_table(request.data_path, request.missing_sentinels or None)
    log_input_features({"data": data}, model)
    scorer = BatchScorer(
        model,
        problem_type=request.problem_type,
        threshold=request.threshold,
        max_shap_rows=request.max_shap_rows,
        seed=request.seed,
        n_bins=request.n_bins,
    )
    result = scorer.score(data, id_columns=request.id_columns, feature_columns=request.feature_columns)
    ScoreWriter(request.scores_path, request.output_dir).write(result, _scoring_metadata(request, result))
    logger.info("Scored %d rows to %s", len(result.scores), request.scores_path)
    logger.info("Wrote score distribution and SHAP importance to %s", request.output_dir)
    return result


def load_score_config(path: str | Path) -> dict[str, Any]:
    """Read a JSON file of scoring paths and settings.

    Args:
        path: Config file. Paths inside it are used as written, relative to the
            working directory, the same way as the command-line flags.

    Returns:
        The JSON object. Unknown fields are rejected.

    Raises:
        TypeError: If ``path`` is empty.
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a JSON object or contains an unknown field.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    config_path = Path(_require_path(path, "config"))
    if not config_path.is_file():
        raise FileNotFoundError(f"Score config not found: {config_path}")
    loaded = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("Score config must be a JSON object.")
    _reject_unknown_config_fields(loaded)
    return loaded


def request_from_config(path: str | Path) -> BatchScoreRequest:
    """Build a scoring request from a JSON config file.

    Args:
        path: Config file. See :func:`load_score_config`.

    Returns:
        Request for :func:`run_batch_score`. ``log_level`` is checked and not stored.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If a field is unknown or a value is invalid.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    values = _values_from_config(load_score_config(path))
    _require_log_level(values.pop("log_level"))
    return BatchScoreRequest(**values)


def _reject_unknown_config_fields(config: Mapping[str, Any]) -> None:
    """Reject config keys the scorer does not read.

    Args:
        config: Parsed JSON object.

    Raises:
        TypeError: If ``config`` is not a mapping.
        ValueError: If a key is not a scoring setting.
    """
    if not isinstance(config, Mapping):
        raise TypeError("Score config must be a JSON object.")
    unknown = sorted(set(config) - _CONFIG_FIELDS)
    if unknown:
        raise ValueError(f"Unknown score config fields: {unknown}.")


def _require_log_level(value: object) -> str:
    """Return a logging level name.

    Args:
        value: Candidate level.

    Returns:
        ``DEBUG``, ``INFO``, ``WARNING``, or ``ERROR``.

    Raises:
        ValueError: If ``value`` is not one of those names.
    """
    if not isinstance(value, str) or value not in _LOG_LEVELS:
        raise ValueError("log_level must be DEBUG, INFO, WARNING, or ERROR.")
    return value


def _coerce_sentinel(value: object) -> Any:
    """Return one missing-value code from a config list.

    Args:
        value: A number or string code.

    Returns:
        A float for numeric codes, otherwise the string.

    Raises:
        ValueError: If ``value`` is null, a boolean, or a non-finite number.
        TypeError: If ``value`` is not a number or string.
    """
    if isinstance(value, bool) or value is None:
        raise ValueError(f"missing_sentinels contains {value!r}.")
    if isinstance(value, (int, float)):
        if not np.isfinite(value):
            raise ValueError("missing_sentinels must be finite numbers or strings.")
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    raise TypeError(f"missing_sentinels contains {type(value).__name__}.")


def _sentinels_from_config(value: object) -> tuple[Any, ...]:
    """Copy missing-value codes from a config field.

    Args:
        value: A list of codes, a comma-separated string, or null.

    Returns:
        Codes passed to :func:`population_drift.analysis.load_table`.

    Raises:
        TypeError: If ``value`` is not a list or string.
        ValueError: If a code is null or non-finite.
    """
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(_parse_sentinels(value))
    if isinstance(value, Sequence):
        return tuple(_coerce_sentinel(item) for item in value)
    raise TypeError("missing_sentinels must be a list or a comma-separated string.")


def _id_columns_from_config(value: object) -> tuple[str, ...]:
    """Copy id columns from a config field.

    Args:
        value: A list of names, a comma-separated string, or null.

    Returns:
        Column names. Null means no id columns.

    Raises:
        TypeError: If ``value`` is not a list or string.
        ValueError: If a name is blank or repeated.
    """
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(_split_list(value))
    return _as_name_tuple(value, "id_columns")


def _feature_columns_from_config(value: object) -> tuple[str, ...] | None:
    """Copy an explicit feature list from a config field.

    Args:
        value: A list of names, a comma-separated string, or null.

    Returns:
        Feature names, or None when the model names should be used.

    Raises:
        TypeError: If ``value`` is not a list or string.
        ValueError: If the list is empty or a name is blank.
    """
    if value is None:
        return None
    if isinstance(value, str):
        names = tuple(_split_list(value))
        if not names:
            raise ValueError("feature_columns is empty.")
        return names
    return _optional_names(value, "feature_columns")


def _features_from_config(config: Mapping[str, Any]) -> tuple[str, ...] | None:
    """Resolve ``feature_columns`` or ``features_file`` from a config object.

    Args:
        config: Parsed score config.

    Returns:
        The feature list, or None when neither field is set.

    Raises:
        ValueError: If both fields are set, or the feature file is empty.
        FileNotFoundError: If ``features_file`` does not exist.
    """
    columns = _feature_columns_from_config(config.get("feature_columns"))
    file_value = config.get("features_file")
    if file_value is None:
        return columns
    file_path = _require_path(file_value, "features_file")
    if columns is not None:
        raise ValueError("Pass only one of feature_columns and features_file.")
    return _features_from_file(file_path)


def _values_from_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Map a config object onto request fields plus ``log_level``.

    Args:
        config: Parsed score config. Missing fields keep the script defaults.

    Returns:
        Arguments for :class:`BatchScoreRequest` and a ``log_level`` entry.

    Raises:
        ValueError: If a field is unknown or a feature setting is invalid.
    """
    _reject_unknown_config_fields(config)
    return {
        "model_path": config.get("model_path", ""),
        "data_path": config.get("data_path", ""),
        "scores_path": config.get("scores_path", ""),
        "output_dir": config.get("output_dir", ""),
        "problem_type": config.get("problem_type", "binary"),
        "threshold": config.get("threshold", 0.5),
        "missing_sentinels": _sentinels_from_config(config.get("missing_sentinels", ())),
        "id_columns": _id_columns_from_config(config.get("id_columns", ())),
        "feature_columns": _features_from_config(config),
        "max_shap_rows": config.get("max_shap_rows", 5000),
        "seed": config.get("seed", 42),
        "n_bins": config.get("n_bins", 20),
        "log_level": config.get("log_level", "INFO"),
    }


def _apply_cli_overrides(values: dict[str, Any], args: argparse.Namespace) -> None:
    """Replace config values with flags that were actually passed.

    Args:
        values: Request fields from the config, including defaults.
        args: Parsed CLI arguments. Omitted flags are absent or None.

    Raises:
        ValueError: If both feature flags are set.
    """
    for flag, field in (
        ("model", "model_path"), ("data", "data_path"), ("scores", "scores_path"),
        ("out", "output_dir"), ("problem_type", "problem_type"), ("threshold", "threshold"),
        ("max_shap_rows", "max_shap_rows"), ("seed", "seed"), ("n_bins", "n_bins"),
        ("log_level", "log_level"),
    ):
        chosen = getattr(args, flag, None)
        if chosen is not None:
            values[field] = chosen
    if getattr(args, "missing_sentinels", None) is not None:
        values["missing_sentinels"] = tuple(_parse_sentinels(args.missing_sentinels))
    if getattr(args, "id_cols", None) is not None:
        values["id_columns"] = tuple(_split_list(args.id_cols))
    if getattr(args, "features", None) or getattr(args, "features_file", None):
        values["feature_columns"] = _cli_features(args)


def build_parser() -> argparse.ArgumentParser:
    """Argument parser for ``python -m population_drift.batch_score``.

    Omitted flags are left unset so a config file can supply them. Defaults are
    applied in :func:`_values_from_config`.

    Returns:
        Parser with the config path, model, data, and output arguments.
    """
    parser = argparse.ArgumentParser(
        prog="python -m population_drift.batch_score",
        description="Score each row of a table and save the score distribution and SHAP importance.",
    )
    parser.add_argument("--config", help="JSON file of paths and scoring settings. Flags override the file.")
    parser.add_argument("--model", help="Fitted LightGBM model (.pkl/.pickle/.joblib/.txt).")
    parser.add_argument("--data", help="Table to score (.csv/.parquet/.feather).")
    parser.add_argument("--scores", help="Where to write one score per row.")
    parser.add_argument("--out", help="Directory for the distribution, importance, and plot.")
    parser.add_argument("--features", help="Comma-separated feature columns. Used when the model has no names.")
    parser.add_argument("--features-file", help="Text file with one feature name per line.")
    parser.add_argument("--id-cols", help="Comma-separated columns to copy onto the score file.")
    parser.add_argument("--problem-type", choices=["binary", "regression"])
    parser.add_argument("--threshold", type=float, help="Binary positive-class cutoff. Default: 0.5.")
    parser.add_argument("--missing-sentinels", help="Comma-separated codes to treat as missing, e.g. -999.")
    parser.add_argument("--max-shap-rows", type=int, help="Maximum rows used for TreeSHAP importance. Default: 5000.")
    parser.add_argument("--n-bins", type=int, help="Histogram bins when scores vary. Default: 20.")
    parser.add_argument("--seed", type=int, help="Seed for the SHAP subsample. Default: 42.")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def _cli_features(args: argparse.Namespace) -> tuple[str, ...] | None:
    """Feature list from ``--features`` or ``--features-file``.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Feature names, or None when both flags are unset.

    Raises:
        ValueError: If the feature file is empty or both flags are set.
    """
    if args.features and args.features_file:
        raise ValueError("Pass only one of --features and --features-file.")
    if args.features_file:
        return _features_from_file(args.features_file)
    if args.features:
        return tuple(_split_list(args.features))
    return None


def _features_from_file(path: str) -> tuple[str, ...]:
    """Read one feature name per line.

    Args:
        path: Text file. Blank lines and ``#`` comments are ignored.

    Returns:
        Feature names in file order.

    Raises:
        ValueError: If the file has no feature names.
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    names = [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]
    if not names:
        raise ValueError(f"Feature file {path} has no feature names.")
    return tuple(names)


def _request_from_args(args: argparse.Namespace) -> tuple[BatchScoreRequest, str]:
    """Build a request from the config file and any overriding flags.

    Args:
        args: Parsed ``build_parser`` arguments.

    Returns:
        The scoring request and the logging level name.

    Raises:
        FileNotFoundError: If ``--config`` or a feature file does not exist.
        ValueError: If a path or setting is invalid.
        json.JSONDecodeError: If the config file is not valid JSON.
    """
    config = load_score_config(args.config) if args.config else {}
    values = _values_from_config(config)
    _apply_cli_overrides(values, args)
    log_level = _require_log_level(values.pop("log_level"))
    return BatchScoreRequest(**values), log_level


def _configure_logging(level: str) -> None:
    """Send logs to stderr at ``level``.

    Args:
        level: ``DEBUG``, ``INFO``, ``WARNING``, or ``ERROR``.
    """
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Arguments, excluding the program name. Defaults to ``sys.argv``.

    Returns:
        0 on success, 2 when the inputs cannot be scored.
    """
    args = build_parser().parse_args(argv)
    try:
        request, log_level = _request_from_args(args)
    except (TypeError, ValueError, FileNotFoundError, OSError, json.JSONDecodeError) as exc:
        _configure_logging("ERROR")
        logger.error("%s", exc)
        return 2
    _configure_logging(log_level)
    try:
        run_batch_score(request)
    except (TypeError, ValueError, FileNotFoundError, OSError) as exc:
        logger.error("%s", exc)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
