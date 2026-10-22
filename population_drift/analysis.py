"""Population drift analysis: training (source) population vs a new (target) population.

The module provides:

* one function per comparison task (univariate, joint, support, model-based and
  label-free performance estimation), each returning a :class:`TaskResult`;
* :func:`population_drift_analyzer`, the orchestrator that runs every task on two datasets;
* :func:`build_baseline`, which runs the analyzer on random pairs of subsets of the
  training population to build the "natural variability" (null) reference;
* :func:`run_comparison`, which runs one labelled population vs the new population;
* :func:`run_paired_comparison`, which uses a fit set and an evaluation set,
  assigning each test to the pair that statistic can support.

Command line (see README.md):

    python -m population_drift compare --train labeled.csv --new new.csv \\
        --features-file features.txt --out runs/comparison --report runs/report.html
    python -m population_drift compare --train fit.csv --test evaluation.csv --new new.csv \\
        --features-file features.txt --out runs/comparison --report runs/report.html
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import pickle
import platform
import sys
import time
import traceback
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy
import shap
import sklearn
from scipy import stats
from scipy.spatial.distance import cdist, jensenshannon
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.weightstats import ttost_ind

logger = logging.getLogger("population_drift")

GLOBAL_FEATURE = "__global__"
CONTINUOUS = "continuous"
CATEGORICAL = "categorical"
MIN_GROUP_ROWS = 30
LEVEL_ORDER = {"info": -1, "mild": 0, "moderate": 1, "severe": 2}
VERDICT_TEXT = {
    "severe": "Different: do not use the model as is",
    "moderate": "Use with caution: mitigation and verification needed",
    "mild": "Similar: the model can be used (with monitoring)",
}
_DISPLAY_METRICS = {
    "auc": "area under the curve (AUC)",
    "brier": "Brier score",
    "log_loss": "log loss",
    "rmse": "root mean squared error (RMSE)",
    "mae": "mean absolute error (MAE)",
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
}
ANALYSIS_SCOPES = ("all", "features", "model")
FEATURE_TASKS = (
    "continuous", "equivalence", "categorical", "missingness", "missingness_patterns",
    "domain_classifier", "mmd", "energy", "correlation", "range_coverage", "joint_outliers",
)
MODEL_TASKS = (
    "prediction_drift", "shap_drift", "calibration", "weighted_performance", "cbpe", "label_shift",
)
_SIGNAL_TASKS = (
    ("Feature drift", "continuous"), ("Missing-value rates", "missingness"),
    ("Unseen category levels", "categorical"), ("classifier two-sample", "domain_classifier"),
    ("kernel/energy", "mmd"), ("Dependence structure", "correlation"),
    ("Out-of-support", "joint_outliers"), ("Values outside", "range_coverage"),
    ("Model score distribution", "prediction_drift"), ("Predicted positive rate", "prediction_drift"),
    ("Model attributions", "shap_drift"), ("effective sample size", "weighted_performance"),
    ("importance weighting", "weighted_performance"), ("CBPE", "cbpe"),
    ("BBSE", "label_shift"), ("calibration", "calibration"),
)


# --------------------------------------------------------------------------------------
# Configuration and result containers
# --------------------------------------------------------------------------------------
@dataclass
class Thresholds:
    """Rule-of-thumb thresholds used for interpretation. All are conventions, not tests."""

    psi_moderate: float = 0.10
    psi_major: float = 0.25
    ks_moderate: float = 0.10
    ks_major: float = 0.20
    smd_small: float = 0.10
    smd_large: float = 0.50
    cramers_v_moderate: float = 0.10
    cramers_v_major: float = 0.30
    cohens_h_small: float = 0.20
    cohens_h_large: float = 0.50
    unseen_level_share: float = 0.01
    domain_auc_mild: float = 0.55
    domain_auc_strong: float = 0.80
    ood_share_caution: float = 0.02
    ood_share_material: float = 0.05
    range_outside_ratio: float = 2.0
    ess_ratio_good: float = 0.50
    ess_ratio_poor: float = 0.10
    correlation_pair_delta: float = 0.20
    importance_rank_rho: float = 0.80
    ece_max: float = 0.05
    near_threshold_band: float = 0.05
    baseline_percentile: float = 95.0
    important_feature_cumulative_share: float = 0.80


@dataclass
class DriftConfig:
    """Settings for every task. ``for_baseline`` returns a cheaper copy for baseline rounds."""

    alpha: float = 0.05
    correction_method: str = "fdr_bh"
    n_bootstrap: int = 200
    n_permutations: int = 100
    n_classifier_permutations: int = 20
    cv_folds: int = 5
    psi_bins: int = 10
    discrete_max_unique: int = 10
    min_expected_count: float = 5.0
    equivalence_margin_smd: float = 0.10
    range_quantiles: tuple[float, float] = (0.005, 0.995)
    ood_quantile: float = 0.01
    max_kernel_rows: int = 2000
    max_correlation_rows: int = 20_000
    max_classifier_rows: int = 200_000
    max_shap_rows: int = 5000
    max_iforest_rows: int = 100_000
    weight_clip_quantile: float = 0.99
    decision_threshold: float = 0.5
    problem_type: str = "binary"
    performance_tolerance: float = 0.02
    seed: int = 42
    n_jobs: int = -1
    thresholds: Thresholds = field(default_factory=Thresholds)

    def __post_init__(self) -> None:
        if not 0 < self.alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha}.")
        if self.correction_method not in {"fdr_bh", "holm", "bonferroni"}:
            raise ValueError(f"Unsupported correction_method '{self.correction_method}'.")
        if self.problem_type not in {"binary", "regression"}:
            raise ValueError(f"problem_type must be 'binary' or 'regression', got '{self.problem_type}'.")
        if self.cv_folds < 2:
            raise ValueError(f"cv_folds must be >= 2, got {self.cv_folds}.")
        for name in ("n_bootstrap", "n_permutations", "n_classifier_permutations"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0.")
        low, high = self.range_quantiles
        if not 0 < low < high < 1:
            raise ValueError(f"range_quantiles must satisfy 0 < low < high < 1, got {self.range_quantiles}.")
        if not 0 < self.ood_quantile < 0.5:
            raise ValueError(f"ood_quantile must be in (0, 0.5), got {self.ood_quantile}.")
        if not 0.5 < self.weight_clip_quantile <= 1:
            raise ValueError(f"weight_clip_quantile must be in (0.5, 1], got {self.weight_clip_quantile}.")
        if self.performance_tolerance < 0:
            raise ValueError("performance_tolerance must be >= 0.")

    def for_baseline(self) -> "DriftConfig":
        """Copy without bootstrap/permutation inference; baseline rounds only need the statistics."""
        return dataclasses.replace(self, n_bootstrap=0, n_permutations=0, n_classifier_permutations=0)


@dataclass
class TaskResult:
    """Output of a single comparison task.

    Attributes:
        name: Task name.
        tables: Named result tables; ``"main"`` is the primary table.
        plot_data: JSON-serialisable data used by the report for plots.
        arrays: In-memory arrays reused by later tasks (not persisted).
        notes: Human-readable caveats produced by the task.
    """

    name: str
    tables: dict[str, pd.DataFrame]
    plot_data: dict[str, Any] = field(default_factory=dict)
    arrays: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


@dataclass
class DriftAnalysisResult:
    """All task results of one source-vs-target analysis plus run metadata."""

    tasks: dict[str, TaskResult]
    metadata: dict[str, Any]

    def table(self, task: str, name: str = "main") -> pd.DataFrame:
        """Return a task table, or an empty frame if the task did not run."""
        task_result = self.tasks.get(task)
        if task_result is None:
            return pd.DataFrame()
        return task_result.tables.get(name, pd.DataFrame())


@dataclass
class BaselineResult:
    """Long table of baseline statistics, its per-metric summary, and metadata."""

    statistics: pd.DataFrame
    summary: pd.DataFrame
    metadata: dict[str, Any]


# Metrics compared against the baseline null. All are oriented so that larger = more drift.
# Structure: task -> table -> (key column, metric columns).
COMPARABLE_METRICS: dict[str, dict[str, tuple[str, tuple[str, ...]]]] = {
    "continuous": {"main": ("feature", ("ks_d", "ad_stat", "cvm_stat", "wasserstein_norm", "psi", "smd_abs",
                                        "log_var_ratio_abs"))},
    "categorical": {"main": ("feature", ("cramers_v", "js_distance", "psi", "unseen_share"))},
    "missingness": {"main": ("feature", ("cohens_h_abs",))},
    "missingness_patterns": {"main": ("feature", ("js_distance",))},
    "domain_classifier": {"main": ("feature", ("auc",))},
    "mmd": {"main": ("feature", ("mmd2",))},
    "energy": {"main": ("feature", ("energy_distance",))},
    "correlation": {"main": ("feature", ("frobenius",)), "pairs": ("feature", ("abs_delta",))},
    "range_coverage": {"main": ("feature", ("outside_share",))},
    "joint_outliers": {"main": ("feature", ("ood_share",))},
    "prediction_drift": {"main": ("feature", ("ks_d", "psi", "positive_rate_cohens_h_abs"))},
    "shap_drift": {"main": ("feature", ("shap_ks_d", "mean_abs_shap_rel_change_abs")),
                   "summary": ("feature", ("rank_rho_deficit",))},
    "weighted_performance": {"main": ("metric", ("degradation",)), "summary": ("feature", ("ess_deficit",))},
    "cbpe": {"main": ("metric", ("degradation",))},
}


# --------------------------------------------------------------------------------------
# Generic helpers
# --------------------------------------------------------------------------------------
def _to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy/pandas objects to JSON-compatible Python objects."""
    if isinstance(obj, dict):
        return {str(key): _to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return _to_jsonable(obj.tolist())
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, pd.DataFrame):
        return _to_jsonable(obj.to_dict(orient="records"))
    return obj


def write_json(obj: Any, path: Path) -> None:
    """Write ``obj`` as indented JSON, converting numpy types."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_jsonable(obj), indent=2))


def _numeric_values(series: pd.Series) -> np.ndarray:
    """Finite numeric values of a series (missing and non-numeric values dropped)."""
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return values[np.isfinite(values)]


def _category_labels(series: pd.Series) -> pd.Series:
    """String labels with missing values preserved; numeric codes are normalised so 1 and 1.0 match."""
    if pd.api.types.is_numeric_dtype(series) and not isinstance(series.dtype, pd.CategoricalDtype):
        return series.astype(float).astype(str).where(series.notna())
    return series.astype(str).where(series.notna())


def _subsample(df: pd.DataFrame, max_rows: int, rng: np.random.Generator) -> pd.DataFrame:
    """Random subsample without replacement, keeping the original row order."""
    if len(df) <= max_rows:
        return df
    return df.iloc[np.sort(rng.choice(len(df), size=max_rows, replace=False))]


def _cohens_h(p_target: float, p_source: float) -> float:
    """Cohen's h effect size for two proportions."""
    return float(2 * np.arcsin(np.sqrt(p_target)) - 2 * np.arcsin(np.sqrt(p_source)))


def _two_proportion_p(count_target: int, n_target: int, count_source: int, n_source: int) -> float:
    """Two-sided two-proportion z-test p-value; 1.0 when both proportions are degenerate."""
    pooled = (count_target + count_source) / (n_target + n_source)
    if pooled in (0.0, 1.0):
        return 1.0
    _, p_value = proportions_ztest([count_target, count_source], [n_target, n_source])
    return float(p_value)


def adjust_pvalues(table: pd.DataFrame, p_columns: Sequence[str], config: DriftConfig) -> pd.DataFrame:
    """Add ``<col>_adj`` columns corrected for multiple testing across the rows of the table.

    Args:
        table: Table with one row per feature (or pair).
        p_columns: Raw p-value columns to correct.
        config: Uses ``alpha`` and ``correction_method``.

    Returns:
        Copy of ``table`` with adjusted p-value columns.
    """
    table = table.copy()
    for column in p_columns:
        if column not in table.columns:
            continue
        adjusted = np.full(len(table), np.nan)
        mask = table[column].notna().to_numpy()
        if mask.any():
            adjusted[mask] = multipletests(table.loc[mask, column].to_numpy(dtype=float), alpha=config.alpha,
                                           method=config.correction_method)[1]
        table[f"{column}_adj"] = adjusted
    return table


def _psi_from_frequencies(p: np.ndarray, q: np.ndarray, eps: float = 1e-4) -> float:
    """PSI between two aligned frequency vectors, with empty bins floored at ``eps``."""
    p = np.clip(np.asarray(p, dtype=float), eps, None)
    q = np.clip(np.asarray(q, dtype=float), eps, None)
    p, q = p / p.sum(), q / q.sum()
    return float(np.sum((q - p) * np.log(q / p)))


def population_stability_index(source_values: np.ndarray, target_values: np.ndarray, n_bins: int = 10) -> float:
    """PSI with bins at source quantiles.

    Args:
        source_values: Finite source values.
        target_values: Finite target values.
        n_bins: Number of source-quantile bins (ties collapse bins).

    Returns:
        PSI value (0 = identical binned distributions).

    Raises:
        ValueError: If either input is empty.
    """
    source_values = np.asarray(source_values, dtype=float)
    target_values = np.asarray(target_values, dtype=float)
    if len(source_values) == 0 or len(target_values) == 0:
        raise ValueError("PSI requires non-empty inputs.")
    inner_edges = np.unique(np.quantile(source_values, np.linspace(0, 1, n_bins + 1)[1:-1]))
    n_cells = len(inner_edges) + 1
    p = np.bincount(np.searchsorted(inner_edges, source_values, side="right"), minlength=n_cells) / len(source_values)
    q = np.bincount(np.searchsorted(inner_edges, target_values, side="right"), minlength=n_cells) / len(target_values)
    return _psi_from_frequencies(p, q)


def _bootstrap_two_sample(statistics: dict[str, Callable[[np.ndarray, np.ndarray], float]], source: np.ndarray,
                          target: np.ndarray, n_bootstrap: int,
                          rng: np.random.Generator) -> dict[str, tuple[float, float]]:
    """Percentile 95% bootstrap CIs of two-sample statistics (both samples resampled)."""
    draws = {name: np.empty(n_bootstrap) for name in statistics}
    for b in range(n_bootstrap):
        source_b = source[rng.integers(0, len(source), len(source))]
        target_b = target[rng.integers(0, len(target), len(target))]
        for name, statistic in statistics.items():
            draws[name][b] = statistic(source_b, target_b)
    return {name: tuple(np.nanpercentile(values, [2.5, 97.5])) for name, values in draws.items()}


def _anderson_darling(source: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """k-sample Anderson-Darling statistic and p-value (SciPy caps p to [0.001, 0.25])."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = stats.anderson_ksamp([source, target])
            p_value = getattr(result, "pvalue", None)
            if p_value is None:
                p_value = result.significance_level
        return float(result.statistic), float(p_value)
    except ValueError:
        return np.nan, np.nan


def _cramer_von_mises(source: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """Two-sample Cramér-von Mises statistic and p-value."""
    try:
        result = stats.cramervonmises_2samp(source, target)
        return float(result.statistic), float(result.pvalue)
    except ValueError:
        return np.nan, np.nan


def _merge_sparse_levels(counts: np.ndarray, min_expected: float) -> np.ndarray:
    """Merge columns of a 2 x k contingency table whose expected counts fall below ``min_expected``."""
    counts = counts[:, counts.sum(axis=0) > 0]
    total = counts.sum()
    smallest_row = counts.sum(axis=1).min()
    sparse = counts.sum(axis=0) * smallest_row / total < min_expected
    if not sparse.any():
        return counts
    merged = np.column_stack([counts[:, ~sparse], counts[:, sparse].sum(axis=1)])
    if merged.shape[1] > 2 and merged[:, -1].sum() * smallest_row / total < min_expected:
        smallest = int(np.argmin(merged[:, :-1].sum(axis=0)))
        merged[:, smallest] += merged[:, -1]
        merged = merged[:, :-1]
    return merged


def _chi_square_homogeneity(source_counts: np.ndarray, target_counts: np.ndarray,
                            min_expected: float) -> tuple[float, float, int, float, bool]:
    """Chi-square test of homogeneity with sparse-level merging.

    Returns:
        (chi2 statistic, p-value, degrees of freedom, Cramér's V, whether levels were merged).
    """
    table = np.vstack([source_counts, target_counts]).astype(float)
    merged = _merge_sparse_levels(table, min_expected)
    if merged.shape[1] < 2:
        return 0.0, 1.0, 0, 0.0, merged.shape[1] != table.shape[1]
    chi2, p_value, dof, _ = stats.chi2_contingency(merged, correction=False)
    cramers_v = float(np.sqrt(chi2 / merged.sum()))
    return float(chi2), float(p_value), int(dof), cramers_v, merged.shape[1] != table.shape[1]


# --------------------------------------------------------------------------------------
# Data loading and validation
# --------------------------------------------------------------------------------------
def load_table(path: str | Path, missing_sentinels: Sequence[Any] | None = None) -> pd.DataFrame:
    """Load a CSV / Parquet / Feather file and optionally map sentinel codes to NaN.

    Args:
        path: File path.
        missing_sentinels: Values to treat as missing (e.g. ``[-999]``).

    Returns:
        Loaded DataFrame.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the extension is not supported.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path, low_memory=False)
    elif suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    elif suffix == ".feather":
        df = pd.read_feather(path)
    else:
        raise ValueError(f"Unsupported file type '{suffix}'. Use .csv, .parquet or .feather.")
    if missing_sentinels:
        df = df.replace(list(missing_sentinels), np.nan)
    logger.info("Loaded %s: %d rows x %d columns", path, len(df), df.shape[1])
    return df


def validate_inputs(source: pd.DataFrame, target: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    """Check that both populations contain the features and report schema differences.

    Args:
        source: Training population.
        target: New population.
        feature_cols: Feature columns to compare.

    Returns:
        Schema table with one row per feature.

    Raises:
        ValueError: If inputs are empty, features are duplicated, or columns are missing.
    """
    if source.empty or target.empty:
        raise ValueError(
            f"Source row count is {len(source)}; target row count is {len(target)}. Both populations must be non-empty."
        )
    feature_cols = list(feature_cols)
    if not feature_cols:
        raise ValueError("At least one feature column is required.")
    duplicates = sorted({col for col in feature_cols if feature_cols.count(col) > 1})
    if duplicates:
        raise ValueError(f"Duplicated feature columns: {duplicates}")
    missing_source = [col for col in feature_cols if col not in source.columns]
    missing_target = [col for col in feature_cols if col not in target.columns]
    if missing_source or missing_target:
        raise ValueError(f"Missing columns - source: {missing_source}, target: {missing_target}")

    rows = []
    for col in feature_cols:
        numeric_source = pd.api.types.is_numeric_dtype(source[col])
        numeric_target = pd.api.types.is_numeric_dtype(target[col])
        rows.append({
            "feature": col,
            "dtype_source": str(source[col].dtype),
            "dtype_target": str(target[col].dtype),
            "dtype_compatible": numeric_source == numeric_target,
            "n_unique_source": int(source[col].nunique(dropna=True)),
            "n_unique_target": int(target[col].nunique(dropna=True)),
            "missing_rate_source": float(source[col].isna().mean()),
            "missing_rate_target": float(target[col].isna().mean()),
            "constant_source": source[col].nunique(dropna=True) <= 1,
        })
    schema = pd.DataFrame(rows)
    for row in schema.itertuples():
        if not row.dtype_compatible:
            logger.warning("Feature %s has incompatible dtypes (%s vs %s)", row.feature, row.dtype_source,
                           row.dtype_target)
        if row.missing_rate_source == 1.0 or row.missing_rate_target == 1.0:
            logger.warning("Feature %s is entirely missing in one population", row.feature)
    return schema


def infer_feature_types(source: pd.DataFrame, feature_cols: Sequence[str],
                        categorical_cols: Sequence[str] | None = None,
                        discrete_max_unique: int = 10) -> dict[str, str]:
    """Classify features as continuous or categorical.

    Declared categoricals, non-numeric and boolean columns, and numeric columns with at most
    ``discrete_max_unique`` distinct values are categorical (KS-type tests are invalid with heavy ties).

    Raises:
        ValueError: If a declared categorical column is not a feature.
    """
    declared = set(categorical_cols or [])
    unknown = declared - set(feature_cols)
    if unknown:
        raise ValueError(f"Categorical columns not in feature list: {sorted(unknown)}")
    types = {}
    for col in feature_cols:
        series = source[col]
        is_numeric = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)
        if col in declared or not is_numeric or series.nunique(dropna=True) <= discrete_max_unique:
            types[col] = CATEGORICAL
        else:
            types[col] = CONTINUOUS
    return types


# --------------------------------------------------------------------------------------
# Univariate tasks
# --------------------------------------------------------------------------------------
def compare_continuous_distributions(source: pd.DataFrame, target: pd.DataFrame, columns: Sequence[str],
                                     config: DriftConfig) -> TaskResult:
    """Two-sample tests and effect sizes for continuous features.

    Tests: Kolmogorov-Smirnov, Anderson-Darling, Cramér-von Mises (multiplicity-adjusted).
    Effect sizes: SMD (+ analytic 95% CI), variance ratio, Wasserstein distance scaled by the
    source SD, PSI (with bootstrap CIs when ``n_bootstrap > 0``).
    """
    rng = np.random.default_rng(config.seed)
    probs = np.linspace(0.0, 1.0, 101)
    rows, quantiles = [], {}
    for col in columns:
        s, t = _numeric_values(source[col]), _numeric_values(target[col])
        if len(s) < 3 or len(t) < 3:
            rows.append({"feature": col, "n_source": len(s), "n_target": len(t),
                         "note": "fewer than 3 non-missing values"})
            continue
        var_s, var_t = s.var(ddof=1), t.var(ddof=1)
        pooled_sd = np.sqrt((var_s + var_t) / 2)
        smd = (t.mean() - s.mean()) / pooled_sd if pooled_sd > 0 else 0.0
        smd_se = np.sqrt((len(s) + len(t)) / (len(s) * len(t)) + smd ** 2 / (2 * (len(s) + len(t))))
        var_ratio = var_t / var_s if var_s > 0 else np.nan
        scale = np.sqrt(var_s) if var_s > 0 else 1.0
        ks = stats.ks_2samp(s, t)
        ad_stat, ad_p = _anderson_darling(s, t)
        cvm_stat, cvm_p = _cramer_von_mises(s, t)

        def wasserstein_norm(a: np.ndarray, b: np.ndarray) -> float:
            return stats.wasserstein_distance(a, b) / scale

        def psi(a: np.ndarray, b: np.ndarray) -> float:
            return population_stability_index(a, b, config.psi_bins)

        row = {
            "feature": col, "n_source": len(s), "n_target": len(t),
            "mean_source": s.mean(), "mean_target": t.mean(),
            "median_source": np.median(s), "median_target": np.median(t),
            "sd_source": np.sqrt(var_s), "sd_target": np.sqrt(var_t),
            "smd": smd, "smd_abs": abs(smd), "smd_ci_low": smd - 1.96 * smd_se, "smd_ci_high": smd + 1.96 * smd_se,
            "var_ratio": var_ratio,
            "log_var_ratio_abs": abs(np.log(var_ratio)) if np.isfinite(var_ratio) and var_ratio > 0 else np.nan,
            "ks_d": ks.statistic, "ks_p": ks.pvalue, "ad_stat": ad_stat, "ad_p": ad_p,
            "cvm_stat": cvm_stat, "cvm_p": cvm_p,
            "wasserstein_norm": wasserstein_norm(s, t), "psi": psi(s, t),
        }
        if config.n_bootstrap > 0:
            cis = _bootstrap_two_sample({"wasserstein_norm": wasserstein_norm, "psi": psi}, s, t,
                                        config.n_bootstrap, rng)
            row.update({"wasserstein_norm_ci_low": cis["wasserstein_norm"][0],
                        "wasserstein_norm_ci_high": cis["wasserstein_norm"][1],
                        "psi_ci_low": cis["psi"][0], "psi_ci_high": cis["psi"][1]})
        rows.append(row)
        quantiles[col] = {"probs": probs, "source": np.quantile(s, probs), "target": np.quantile(t, probs)}
    table = adjust_pvalues(pd.DataFrame(rows), ["ks_p", "ad_p", "cvm_p"], config)
    return TaskResult("continuous", {"main": table}, plot_data={"quantiles": quantiles})


def run_equivalence_tests(source: pd.DataFrame, target: pd.DataFrame, columns: Sequence[str],
                          config: DriftConfig) -> TaskResult:
    """TOST equivalence test of means with margin ``equivalence_margin_smd`` x pooled SD.

    ``equivalent=True`` is positive evidence of similarity (unlike a non-significant difference test).
    """
    rows = []
    for col in columns:
        s, t = _numeric_values(source[col]), _numeric_values(target[col])
        if len(s) < 3 or len(t) < 3:
            continue
        pooled_sd = np.sqrt((s.var(ddof=1) + t.var(ddof=1)) / 2)
        margin = config.equivalence_margin_smd * pooled_sd
        if margin == 0:
            p_value = 0.0 if t.mean() == s.mean() else 1.0
        else:
            try:
                p_value, _, _ = ttost_ind(t, s, -margin, margin, usevar="unequal")
            except (ValueError, ZeroDivisionError, FloatingPointError):
                p_value = np.nan
        rows.append({"feature": col, "margin_smd": config.equivalence_margin_smd, "margin_raw": margin,
                     "mean_difference": t.mean() - s.mean(), "tost_p": float(p_value),
                     "equivalent": bool(np.isfinite(p_value) and p_value < config.alpha)})
    return TaskResult("equivalence", {"main": pd.DataFrame(rows)})


def compare_categorical_distributions(source: pd.DataFrame, target: pd.DataFrame, columns: Sequence[str],
                                      config: DriftConfig, max_plot_levels: int = 12) -> TaskResult:
    """Chi-square homogeneity test, Cramér's V, Jensen-Shannon distance, PSI and unseen-level share."""
    rows, frequencies = [], {}
    for col in columns:
        s, t = _category_labels(source[col]).dropna(), _category_labels(target[col]).dropna()
        if len(s) == 0 or len(t) == 0:
            rows.append({"feature": col, "n_source": len(s), "n_target": len(t), "note": "no non-missing values"})
            continue
        levels = sorted(set(s.unique()) | set(t.unique()))
        source_counts = s.value_counts().reindex(levels, fill_value=0).to_numpy()
        target_counts = t.value_counts().reindex(levels, fill_value=0).to_numpy()
        p, q = source_counts / source_counts.sum(), target_counts / target_counts.sum()
        chi2, chi2_p, dof, cramers_v, merged = _chi_square_homogeneity(source_counts, target_counts,
                                                                       config.min_expected_count)
        js = jensenshannon(p, q, base=2)
        unseen_levels = [lvl for lvl, cnt in zip(levels, source_counts) if cnt == 0]
        rows.append({
            "feature": col, "n_source": len(s), "n_target": len(t),
            "n_levels_source": int((source_counts > 0).sum()), "n_levels_target": int((target_counts > 0).sum()),
            "n_unseen_levels": len(unseen_levels), "unseen_share": float(t.isin(unseen_levels).mean()),
            "chi2_stat": chi2, "chi2_dof": dof, "chi2_p": chi2_p, "sparse_levels_merged": merged,
            "cramers_v": cramers_v, "js_distance": float(js) if np.isfinite(js) else 0.0,
            "psi": _psi_from_frequencies(p, q),
            "top_level_source": levels[int(np.argmax(p))], "top_level_target": levels[int(np.argmax(q))],
        })
        order = np.argsort(-np.maximum(p, q))
        top = order[:max_plot_levels]
        rest = order[max_plot_levels:]
        plot_levels = [levels[i] for i in top] + (["(other)"] if len(rest) else [])
        frequencies[col] = {"levels": plot_levels,
                            "source": list(p[top]) + ([p[rest].sum()] if len(rest) else []),
                            "target": list(q[top]) + ([q[rest].sum()] if len(rest) else [])}
    table = adjust_pvalues(pd.DataFrame(rows), ["chi2_p"], config)
    return TaskResult("categorical", {"main": table}, plot_data={"frequencies": frequencies})


def compare_missingness(source: pd.DataFrame, target: pd.DataFrame, columns: Sequence[str],
                        config: DriftConfig) -> TaskResult:
    """Per-feature missing-rate comparison: two-proportion z-test and Cohen's h."""
    rows = []
    for col in columns:
        k_s, k_t = int(source[col].isna().sum()), int(target[col].isna().sum())
        n_s, n_t = len(source), len(target)
        rate_s, rate_t = k_s / n_s, k_t / n_t
        h = _cohens_h(rate_t, rate_s)
        rows.append({"feature": col, "missing_rate_source": rate_s, "missing_rate_target": rate_t,
                     "difference": rate_t - rate_s, "cohens_h": h, "cohens_h_abs": abs(h),
                     "z_p": _two_proportion_p(k_t, n_t, k_s, n_s)})
    table = adjust_pvalues(pd.DataFrame(rows), ["z_p"], config)
    return TaskResult("missingness", {"main": table})


def compare_missingness_patterns(source: pd.DataFrame, target: pd.DataFrame, columns: Sequence[str],
                                 config: DriftConfig, max_patterns: int = 20) -> TaskResult:
    """Compare joint missing-value patterns (which features are missing together)."""
    columns = list(columns)

    def pattern_codes(df: pd.DataFrame) -> pd.Series:
        packed = np.packbits(df[columns].isna().to_numpy(), axis=1)
        return pd.Series([row.tobytes() for row in packed])

    def pattern_label(code: bytes) -> str:
        bits = np.unpackbits(np.frombuffer(code, dtype=np.uint8))[:len(columns)].astype(bool)
        missing = [col for col, flag in zip(columns, bits) if flag]
        label = "complete" if not missing else "missing: " + ", ".join(missing)
        return label if len(label) <= 90 else label[:87] + "..."

    codes_s, codes_t = pattern_codes(source), pattern_codes(target)
    patterns = sorted(set(codes_s) | set(codes_t))
    counts_s = codes_s.value_counts().reindex(patterns, fill_value=0).to_numpy()
    counts_t = codes_t.value_counts().reindex(patterns, fill_value=0).to_numpy()
    chi2, chi2_p, dof, cramers_v, _ = _chi_square_homogeneity(counts_s, counts_t, config.min_expected_count)
    p, q = counts_s / counts_s.sum(), counts_t / counts_t.sum()
    js = jensenshannon(p, q, base=2) if len(patterns) > 1 else 0.0
    complete = bytes(np.packbits(np.zeros((1, len(columns)), dtype=bool), axis=1)[0])
    complete_idx = patterns.index(complete) if complete in patterns else None
    main = pd.DataFrame([{
        "feature": GLOBAL_FEATURE, "n_patterns_source": int((counts_s > 0).sum()),
        "n_patterns_target": int((counts_t > 0).sum()),
        "complete_share_source": float(p[complete_idx]) if complete_idx is not None else 0.0,
        "complete_share_target": float(q[complete_idx]) if complete_idx is not None else 0.0,
        "chi2_stat": chi2, "chi2_p": chi2_p, "cramers_v": cramers_v,
        "js_distance": float(js) if np.isfinite(js) else 0.0,
    }])
    order = np.argsort(-np.maximum(p, q))[:max_patterns]
    detail = pd.DataFrame({"pattern": [pattern_label(patterns[i]) for i in order],
                           "share_source": p[order], "share_target": q[order]})
    return TaskResult("missingness_patterns", {"main": main, "patterns": detail})


# --------------------------------------------------------------------------------------
# Multivariate tasks
# --------------------------------------------------------------------------------------
def _prepare_model_frames(source: pd.DataFrame, target: pd.DataFrame, columns: Sequence[str],
                          feature_types: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Model-ready frames: numeric continuous columns, shared-category pandas categoricals."""
    xs, xt = {}, {}
    for col in columns:
        if feature_types[col] == CATEGORICAL:
            s_labels, t_labels = _category_labels(source[col]), _category_labels(target[col])
            categories = sorted(set(s_labels.dropna()) | set(t_labels.dropna()))
            xs[col] = pd.Categorical(s_labels, categories=categories)
            xt[col] = pd.Categorical(t_labels, categories=categories)
        else:
            xs[col] = pd.to_numeric(source[col], errors="coerce").to_numpy(dtype=float)
            xt[col] = pd.to_numeric(target[col], errors="coerce").to_numpy(dtype=float)
    return pd.DataFrame(xs), pd.DataFrame(xt)


def _domain_classifier(config: DriftConfig, seed: int) -> lgb.LGBMClassifier:
    """LightGBM classifier used to separate the two populations."""
    return lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, min_child_samples=20,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                              random_state=seed, n_jobs=config.n_jobs, verbose=-1)


def _oof_domain_scores(features: pd.DataFrame, domain: np.ndarray, config: DriftConfig, seed: int) -> np.ndarray:
    """Out-of-fold P(target | x) with stratified K-fold."""
    scores = np.zeros(len(domain))
    folds = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=seed)
    for train_idx, test_idx in folds.split(features, domain):
        model = _domain_classifier(config, seed)
        model.fit(features.iloc[train_idx], domain[train_idx])
        scores[test_idx] = model.predict_proba(features.iloc[test_idx])[:, 1]
    return scores


def classifier_two_sample_test(source: pd.DataFrame, target: pd.DataFrame, columns: Sequence[str],
                               feature_types: dict[str, str], config: DriftConfig) -> TaskResult:
    """Classifier two-sample test (adversarial validation).

    A LightGBM classifier predicts population membership; H0 (same joint distribution) implies an
    out-of-fold AUC of 0.5. Reports AUC with stratified bootstrap CI, Mann-Whitney p-value on the
    out-of-fold scores, a permutation p-value (``n_classifier_permutations`` refits) and the features
    driving the separation (gain importance).

    Raises:
        ValueError: If either population has too few rows for cross-validation.
    """
    rng = np.random.default_rng(config.seed)
    half = config.max_classifier_rows // 2
    src_pos = np.arange(len(source)) if len(source) <= half else np.sort(rng.choice(len(source), half, replace=False))
    tgt_pos = np.arange(len(target)) if len(target) <= half else np.sort(rng.choice(len(target), half, replace=False))
    xs, xt = _prepare_model_frames(source.iloc[src_pos], target.iloc[tgt_pos], columns, feature_types)
    if min(len(xs), len(xt)) < 2 * config.cv_folds:
        raise ValueError("Too few rows for the classifier two-sample test.")
    features = pd.concat([xs, xt], ignore_index=True)
    domain = np.r_[np.zeros(len(xs), dtype=int), np.ones(len(xt), dtype=int)]
    scores = _oof_domain_scores(features, domain, config, config.seed)
    auc = roc_auc_score(domain, scores)
    mw_p = stats.mannwhitneyu(scores[domain == 1], scores[domain == 0], alternative="greater").pvalue

    null_auc = np.array([])
    permutation_p = np.nan
    if config.n_classifier_permutations > 0:
        null_auc = np.array([
            roc_auc_score(perm, _oof_domain_scores(features, perm, config, config.seed + i + 1))
            for i, perm in enumerate(rng.permutation(domain) for _ in range(config.n_classifier_permutations))
        ])
        permutation_p = (1 + np.sum(null_auc >= auc)) / (1 + len(null_auc))

    ci_low = ci_high = np.nan
    if config.n_bootstrap > 0:
        idx_s, idx_t = np.flatnonzero(domain == 0), np.flatnonzero(domain == 1)
        boot = []
        for _ in range(config.n_bootstrap):
            sample = np.r_[rng.choice(idx_s, len(idx_s)), rng.choice(idx_t, len(idx_t))]
            boot.append(roc_auc_score(domain[sample], scores[sample]))
        ci_low, ci_high = np.percentile(boot, [2.5, 97.5])

    full_model = _domain_classifier(config, config.seed).fit(features, domain)
    gain = full_model.booster_.feature_importance(importance_type="gain")
    importance = pd.DataFrame({"feature": list(features.columns), "gain": gain})
    importance["gain_share"] = importance["gain"] / importance["gain"].sum() if importance["gain"].sum() > 0 else 0.0
    importance = importance.sort_values("gain", ascending=False, ignore_index=True)

    main = pd.DataFrame([{"feature": GLOBAL_FEATURE, "auc": auc, "auc_ci_low": ci_low, "auc_ci_high": ci_high,
                          "mannwhitney_p": mw_p, "permutation_p": permutation_p,
                          "n_permutations": len(null_auc), "n_source_rows": len(xs), "n_target_rows": len(xt)}])
    bins = np.linspace(0, 1, 41)
    plot = {"bins": bins, "source_hist": np.histogram(scores[domain == 0], bins=bins, density=True)[0],
            "target_hist": np.histogram(scores[domain == 1], bins=bins, density=True)[0],
            "permutation_null_auc": null_auc}
    arrays = {"source_positions": src_pos, "source_domain_scores": scores[domain == 0],
              "n_source": len(xs), "n_target": len(xt)}
    return TaskResult("domain_classifier", {"main": main, "importance": importance}, plot_data=plot, arrays=arrays)


def _encode_for_distance(source: pd.DataFrame, target: pd.DataFrame, columns: Sequence[str],
                         feature_types: dict[str, str], max_levels: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Numeric matrices for distance-based methods.

    Continuous features: source-median imputation and source standardisation. Categorical features:
    one-hot encoding of the ``max_levels`` most frequent source levels (missing is its own level).
    """
    blocks_s, blocks_t = [], []
    for col in columns:
        if feature_types[col] == CONTINUOUS:
            s = pd.to_numeric(source[col], errors="coerce").to_numpy(dtype=float)
            t = pd.to_numeric(target[col], errors="coerce").to_numpy(dtype=float)
            finite = s[np.isfinite(s)]
            if len(finite) == 0:
                continue
            median, mean, sd = np.median(finite), finite.mean(), finite.std()
            sd = sd if sd > 0 else 1.0
            s = np.where(np.isfinite(s), s, median)
            t = np.where(np.isfinite(t), t, median)
            blocks_s.append(((s - mean) / sd)[:, None])
            blocks_t.append(((t - mean) / sd)[:, None])
        else:
            s_labels = _category_labels(source[col]).fillna("__missing__")
            t_labels = _category_labels(target[col]).fillna("__missing__")
            for level in s_labels.value_counts().index[:max_levels]:
                blocks_s.append((s_labels == level).to_numpy(dtype=float)[:, None])
                blocks_t.append((t_labels == level).to_numpy(dtype=float)[:, None])
    if not blocks_s:
        raise ValueError("No usable features for distance-based tests.")
    return np.hstack(blocks_s), np.hstack(blocks_t)


def _quadratic_two_sample_test(matrix: np.ndarray, n_source: int,
                               combine: Callable[[float, float, float], float], n_permutations: int,
                               rng: np.random.Generator) -> tuple[float, float, np.ndarray]:
    """Permutation test for statistics built from within/between block means of a symmetric matrix.

    ``combine(mean_xx, mean_yy, mean_xy)`` receives off-diagonal within-group means and the
    between-group mean. Each permutation costs one matrix-vector product.
    """
    total = matrix.sum()
    diag = np.diag(matrix).copy()
    diag_total = diag.sum()

    def statistic(in_source: np.ndarray) -> float:
        z = in_source.astype(float)
        m = z.sum()
        n = len(z) - m
        mz = matrix @ z
        s_xx = float(z @ mz)
        s_xy = float(mz.sum() - s_xx)
        s_yy = total - s_xx - 2 * s_xy
        d_x = float(diag @ z)
        return combine((s_xx - d_x) / (m * (m - 1)), (s_yy - (diag_total - d_x)) / (n * (n - 1)), s_xy / (m * n))

    labels = np.zeros(len(matrix), dtype=bool)
    labels[:n_source] = True
    observed = statistic(labels)
    null = np.array([statistic(rng.permutation(labels)) for _ in range(n_permutations)])
    p_value = (1 + np.sum(null >= observed)) / (1 + n_permutations) if n_permutations else np.nan
    return observed, p_value, null


def mmd_test(source: pd.DataFrame, target: pd.DataFrame, columns: Sequence[str], feature_types: dict[str, str],
             config: DriftConfig) -> TaskResult:
    """Maximum Mean Discrepancy (RBF kernel, median-heuristic bandwidth) with permutation p-value.

    Uses up to ``max_kernel_rows`` rows per population (O(n^2) memory).
    """
    rng = np.random.default_rng(config.seed)
    zs, zt = _encode_for_distance(_subsample(source, config.max_kernel_rows, rng),
                                  _subsample(target, config.max_kernel_rows, rng), columns, feature_types)
    pooled = np.vstack([zs, zt])
    sq_dist = cdist(pooled, pooled, "sqeuclidean")
    probe = rng.choice(len(pooled), size=min(len(pooled), 1000), replace=False)
    probe_dist = np.sqrt(sq_dist[np.ix_(probe, probe)][np.triu_indices(len(probe), 1)])
    bandwidth = float(np.median(probe_dist[probe_dist > 0])) if np.any(probe_dist > 0) else 1.0
    kernel = np.exp(-sq_dist / (2 * bandwidth ** 2))
    observed, p_value, null = _quadratic_two_sample_test(kernel, len(zs), lambda xx, yy, xy: xx + yy - 2 * xy,
                                                         config.n_permutations, rng)
    main = pd.DataFrame([{"feature": GLOBAL_FEATURE, "mmd2": observed, "permutation_p": p_value,
                          "bandwidth": bandwidth, "n_source_rows": len(zs), "n_target_rows": len(zt)}])
    return TaskResult("mmd", {"main": main}, plot_data={"permutation_null": null})


def energy_distance_test(source: pd.DataFrame, target: pd.DataFrame, columns: Sequence[str],
                         feature_types: dict[str, str], config: DriftConfig) -> TaskResult:
    """Energy distance (Székely & Rizzo) with permutation p-value on standardised features."""
    rng = np.random.default_rng(config.seed + 1)
    zs, zt = _encode_for_distance(_subsample(source, config.max_kernel_rows, rng),
                                  _subsample(target, config.max_kernel_rows, rng), columns, feature_types)
    pooled = np.vstack([zs, zt])
    distances = cdist(pooled, pooled, "euclidean")
    observed, p_value, null = _quadratic_two_sample_test(distances, len(zs), lambda xx, yy, xy: 2 * xy - xx - yy,
                                                         config.n_permutations, rng)
    m, n = len(zs), len(zt)
    main = pd.DataFrame([{"feature": GLOBAL_FEATURE, "energy_distance": observed,
                          "energy_statistic": observed * m * n / (m + n), "permutation_p": p_value,
                          "n_source_rows": m, "n_target_rows": n}])
    return TaskResult("energy", {"main": main}, plot_data={"permutation_null": null})


def compare_correlation_structure(source: pd.DataFrame, target: pd.DataFrame, columns: Sequence[str],
                                  config: DriftConfig) -> TaskResult:
    """Spearman correlation structure: global Frobenius-norm permutation test and per-pair Fisher z-tests.

    Per-pair z-tests use the Fieller et al. (1957) variance 1.06 / (n - 3) for Spearman's rho.
    """
    columns = list(columns)
    if len(columns) < 2:
        return TaskResult("correlation", {"main": pd.DataFrame(), "pairs": pd.DataFrame()},
                          notes=["Fewer than two continuous features; correlation structure not compared."])
    rng = np.random.default_rng(config.seed)
    s = _subsample(source[columns], config.max_correlation_rows, rng).apply(pd.to_numeric, errors="coerce")
    t = _subsample(target[columns], config.max_correlation_rows, rng).apply(pd.to_numeric, errors="coerce")
    corr_s, corr_t = s.corr(method="spearman"), t.corr(method="spearman")

    def frobenius(a: pd.DataFrame, b: pd.DataFrame) -> float:
        diff = (b.corr(method="spearman") - a.corr(method="spearman")).fillna(0.0).to_numpy()
        return float(np.linalg.norm(np.triu(diff, 1)))

    observed = float(np.linalg.norm(np.triu((corr_t - corr_s).fillna(0.0).to_numpy(), 1)))
    null = np.array([])
    p_value = np.nan
    if config.n_permutations > 0:
        pooled = pd.concat([s, t], ignore_index=True)
        labels = np.r_[np.ones(len(s), dtype=bool), np.zeros(len(t), dtype=bool)]
        null = np.empty(config.n_permutations)
        for i in range(config.n_permutations):
            perm = rng.permutation(labels)
            null[i] = frobenius(pooled[perm], pooled[~perm])
        p_value = (1 + np.sum(null >= observed)) / (1 + len(null))

    pairs = []
    for i, a in enumerate(columns):
        for b in columns[i + 1:]:
            n_s = int((s[a].notna() & s[b].notna()).sum())
            n_t = int((t[a].notna() & t[b].notna()).sum())
            rho_s, rho_t = corr_s.loc[a, b], corr_t.loc[a, b]
            p_pair = np.nan
            if n_s > 3 and n_t > 3 and np.isfinite(rho_s) and np.isfinite(rho_t):
                z = (np.arctanh(np.clip(rho_t, -0.999999, 0.999999)) - np.arctanh(np.clip(rho_s, -0.999999, 0.999999)))
                z /= np.sqrt(1.06 / (n_t - 3) + 1.06 / (n_s - 3))
                p_pair = float(2 * stats.norm.sf(abs(z)))
            pairs.append({"feature": f"{a}|{b}", "feature_a": a, "feature_b": b, "rho_source": rho_s,
                          "rho_target": rho_t, "delta": rho_t - rho_s, "abs_delta": abs(rho_t - rho_s),
                          "fisher_z_p": p_pair})
    pairs_table = adjust_pvalues(pd.DataFrame(pairs), ["fisher_z_p"], config)
    pairs_table = pairs_table.sort_values("abs_delta", ascending=False, ignore_index=True)
    main = pd.DataFrame([{"feature": GLOBAL_FEATURE, "frobenius": observed, "permutation_p": p_value,
                          "n_pairs": len(pairs_table),
                          "max_abs_delta": float(pairs_table["abs_delta"].max()) if len(pairs_table) else np.nan}])
    plot = {"columns": columns, "source_corr": corr_s.to_numpy(), "target_corr": corr_t.to_numpy(),
            "permutation_null": null}
    return TaskResult("correlation", {"main": main, "pairs": pairs_table}, plot_data=plot)


# --------------------------------------------------------------------------------------
# Support / out-of-distribution tasks
# --------------------------------------------------------------------------------------
def check_range_coverage(source: pd.DataFrame, target: pd.DataFrame, columns: Sequence[str],
                         config: DriftConfig) -> TaskResult:
    """Share of target values outside the source [low, high] quantile range per continuous feature.

    Under H0 the expected share is ``low + (1 - high)``; a one-sided binomial test checks for excess.
    Tree models do not extrapolate, so values outside the training range get edge-of-range predictions.
    """
    low_q, high_q = config.range_quantiles
    expected = low_q + (1 - high_q)
    rows = []
    for col in columns:
        s, t = _numeric_values(source[col]), _numeric_values(target[col])
        if len(s) < 3 or len(t) == 0:
            continue
        low, high = np.quantile(s, [low_q, high_q])
        below, above = int((t < low).sum()), int((t > high).sum())
        outside = below + above
        rows.append({"feature": col, "source_low": low, "source_high": high, "source_min": s.min(),
                     "source_max": s.max(), "target_min": t.min(), "target_max": t.max(),
                     "share_below": below / len(t), "share_above": above / len(t), "outside_share": outside / len(t),
                     "expected_share": expected, "ratio_to_expected": (outside / len(t)) / expected,
                     "beyond_source_max_share": float(((t < s.min()) | (t > s.max())).mean()),
                     "binomial_p": stats.binomtest(outside, len(t), expected, alternative="greater").pvalue})
    table = adjust_pvalues(pd.DataFrame(rows), ["binomial_p"], config)
    return TaskResult("range_coverage", {"main": table})


def detect_joint_outliers(source: pd.DataFrame, target: pd.DataFrame, columns: Sequence[str],
                          feature_types: dict[str, str], config: DriftConfig) -> TaskResult:
    """Joint novelty detection with an Isolation Forest fitted on the source.

    The threshold is the ``ood_quantile`` of scores on a held-out source split, so the expected
    target share under H0 is ``ood_quantile``; a one-sided binomial test checks for excess.
    """
    rng = np.random.default_rng(config.seed)
    src = _subsample(source, config.max_iforest_rows, rng)
    tgt = _subsample(target, config.max_iforest_rows, rng)
    zs, zt = _encode_for_distance(src, tgt, columns, feature_types)
    order = rng.permutation(len(zs))
    n_fit = int(0.7 * len(zs))
    fit_rows, calib_rows = zs[order[:n_fit]], zs[order[n_fit:]]
    if len(calib_rows) < MIN_GROUP_ROWS:
        raise ValueError("Too few source rows to calibrate the Isolation Forest threshold.")
    forest = IsolationForest(n_estimators=200, random_state=config.seed, n_jobs=config.n_jobs).fit(fit_rows)
    calib_scores, target_scores = forest.score_samples(calib_rows), forest.score_samples(zt)
    threshold = float(np.quantile(calib_scores, config.ood_quantile))
    n_ood = int((target_scores < threshold).sum())
    main = pd.DataFrame([{"feature": GLOBAL_FEATURE, "ood_share": n_ood / len(zt),
                          "expected_share": config.ood_quantile, "n_ood": n_ood, "n_target_rows": len(zt),
                          "score_threshold": threshold,
                          "binomial_p": stats.binomtest(n_ood, len(zt), config.ood_quantile,
                                                        alternative="greater").pvalue}])
    bins = np.linspace(min(calib_scores.min(), target_scores.min()), max(calib_scores.max(), target_scores.max()), 41)
    plot = {"bins": bins, "source_hist": np.histogram(calib_scores, bins=bins, density=True)[0],
            "target_hist": np.histogram(target_scores, bins=bins, density=True)[0], "threshold": threshold}
    return TaskResult("joint_outliers", {"main": main}, plot_data=plot)


# --------------------------------------------------------------------------------------
# Model helpers and model-based tasks
# --------------------------------------------------------------------------------------
def load_model(path: str | Path) -> Any:
    """Load a pickled LightGBM estimator (``.pkl``) or a native Booster text file.

    Args:
        path: Model file. ``.pkl`` / ``.pickle`` are pickle or joblib dumps;
            ``.joblib`` is joblib; ``.txt`` / ``.model`` / ``.lgb`` are Booster files.

    Returns:
        The unpickled estimator or a ``lightgbm.Booster``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the suffix is not a supported model format, or a pickle cannot be read.
        TypeError: If ``path`` is empty.
    """
    if path is None or str(path).strip() == "":
        raise TypeError("Model path is empty.")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    suffix = path.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        return _load_pickle(path)
    if suffix == ".joblib":
        return joblib.load(path)
    if suffix in {".txt", ".model", ".lgb"}:
        return lgb.Booster(model_file=str(path))
    raise ValueError(f"Unsupported model file '{path.suffix}'. Use .pkl/.pickle/.joblib or .txt/.model.")


def load_model_or_skip(path: str | Path | None) -> tuple[Any | None, str | None]:
    """Load a model if a path is set. A load failure skips the model instead of aborting.

    Args:
        path: Model file, or blank/None to run without a model.

    Returns:
        ``(model, None)`` on success, ``(None, None)`` when no path is set,
        ``(None, error)`` when ``path`` is set but the file cannot be loaded.
    """
    if path is None or not str(path).strip():
        return None, None
    try:
        return load_model(path), None
    except Exception as exc:  # noqa: BLE001 - a bad model must not abort feature tests
        message = (
            f"Model file {path} could not be loaded ({type(exc).__name__}: {exc}). "
            "Feature-distribution tests will still run; model-based tests are skipped."
        )
        logger.error("%s", message)
        return None, message


def _load_pickle(path: Path) -> Any:
    """Unpickle a LightGBM estimator from ``.pkl``.

    Tries ``pickle.load`` first, then ``joblib.load`` for joblib dumps saved with a ``.pkl`` suffix.

    Args:
        path: Existing pickle path.

    Returns:
        The unpickled object.

    Raises:
        ValueError: If neither pickle nor joblib can read the file.
    """
    try:
        with path.open("rb") as handle:
            return pickle.load(handle)
    except Exception as pickle_error:
        try:
            return joblib.load(path)
        except Exception as joblib_error:
            raise ValueError(
                f"Could not load pickled model {path}: pickle: {pickle_error}; joblib: {joblib_error}."
            ) from joblib_error


def _lightgbm_booster(model: Any) -> lgb.Booster | None:
    """Return the LightGBM Booster used for feature names.

    Accepts a native Booster, a sklearn LightGBM estimator (``booster_``), or a
    scoring wrapper that stores one of those as ``.model``.

    Args:
        model: Fitted model or wrapper.

    Returns:
        The Booster, or None if none is found.
    """
    current = model
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, lgb.Booster):
            return current
        booster = getattr(current, "booster_", None)
        if isinstance(booster, lgb.Booster):
            return booster
        current = getattr(current, "model", None)
    return None


def _lightgbm_tree_model(model: Any) -> Any:
    """Return a LightGBM estimator or Booster for ``shap.TreeExplainer``.

    Prefers the sklearn estimator so TreeExplainer sees the problem type.
    Walks ``.model`` the same way as :func:`_lightgbm_booster`.

    Args:
        model: Fitted model or wrapper.

    Returns:
        ``LGBMModel`` or ``Booster``, or None if none is found.
    """
    current = model
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, (lgb.LGBMModel, lgb.Booster)):
            return current
        current = getattr(current, "model", None)
    booster = _lightgbm_booster(model)
    return booster


def _shap_feature_matrix(values: object, n_rows: int, n_features: int) -> np.ndarray:
    """Normalize ``TreeExplainer.shap_values`` to shape ``(n_rows, n_features)``.

    Args:
        values: Array, class-list, or 3-D class stack from TreeExplainer.
        n_rows: Expected row count.
        n_features: Expected feature count.

    Returns:
        Contribution matrix without a bias column.

    Raises:
        ValueError: If the shape is not binary/regression TreeSHAP.
    """
    if n_rows < 0 or n_features < 0:
        raise ValueError("n_rows and n_features must be >= 0.")
    if isinstance(values, list):
        if not values or len(values) > 2:
            raise ValueError("SHAP contributions are only supported for binary classification and regression.")
        values = values[-1]
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim == 3:
        if matrix.shape[2] > 2:
            raise ValueError("SHAP contributions are only supported for binary classification and regression.")
        matrix = matrix[:, :, -1]
    if matrix.ndim != 2 or matrix.shape != (n_rows, n_features):
        raise ValueError("SHAP contributions are only supported for binary classification and regression.")
    return matrix


def model_feature_names(model: Any) -> list[str] | None:
    """Feature names stored in a LightGBM Booster or sklearn LightGBM estimator, if any."""
    booster = _lightgbm_booster(model)
    return list(booster.feature_name()) if booster is not None else None


def prepare_model_input(df: pd.DataFrame, model: Any, feature_cols: Sequence[str]) -> pd.DataFrame:
    """Select the model's input columns in its training order; object columns become categoricals.

    LightGBM re-maps pandas categoricals to the training categories at prediction time.

    Raises:
        ValueError: If columns required by the model are missing.
    """
    names = model_feature_names(model) or list(feature_cols)
    if all(name.startswith("Column_") for name in names) and not set(names) <= set(df.columns):
        if len(names) != len(feature_cols):
            raise ValueError("Model was trained without feature names and its feature count differs from the "
                             "feature list; cannot align columns.")
        logger.warning("Model has no feature names; using the feature list order positionally.")
        names = list(feature_cols)
    missing = [name for name in names if name not in df.columns]
    if missing:
        raise ValueError(f"Model expects columns missing from the data: {missing}")
    frame = df[names].copy()
    for col in frame.columns:
        if isinstance(frame[col].dtype, pd.CategoricalDtype):
            continue
        if frame[col].dtype == object or pd.api.types.is_string_dtype(frame[col].dtype):
            frame[col] = frame[col].astype("category")
    return frame


def predict_scores(model: Any, features: pd.DataFrame, problem_type: str) -> np.ndarray:
    """1-D model output: positive-class probability (binary) or prediction (regression).

    Raises:
        ValueError: For multiclass outputs (not supported).
    """
    if isinstance(model, lgb.Booster):
        output = model.predict(features)
    elif problem_type == "binary" and hasattr(model, "predict_proba"):
        output = model.predict_proba(features)
    else:
        output = model.predict(features)
    output = np.asarray(output, dtype=float)
    if output.ndim == 2 and output.shape[1] == 2:
        output = output[:, 1]
    if output.ndim != 1:
        raise ValueError("Only binary classification and regression models are supported.")
    return output


def predict_contributions(model: Any, features: pd.DataFrame) -> np.ndarray:
    """TreeSHAP contributions from ``shap.TreeExplainer``, one column per feature.

    Args:
        model: LightGBM Booster, sklearn estimator, or wrapper with ``.model`` / ``.booster_``.
        features: Rows in the model's feature order.

    Returns:
        Array of shape ``(n_rows, n_features)``.

    Raises:
        TypeError: If ``model`` is missing, ``features`` is not a DataFrame, or no tree model is found.
        ValueError: If the contribution matrix is not binary/regression TreeSHAP.
    """
    if model is None:
        raise TypeError("model is required.")
    if not isinstance(features, pd.DataFrame):
        raise TypeError(f"features must be a pandas DataFrame, got {type(features).__name__}.")
    tree_model = _lightgbm_tree_model(model)
    if tree_model is None:
        raise TypeError(
            "TreeSHAP requires a LightGBM Booster or sklearn estimator with booster_. "
            f"Got {type(model).__name__}."
        )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="LightGBM binary classifier with TreeExplainer shap values output has changed to a list of ndarray",
        )
        values = shap.TreeExplainer(tree_model).shap_values(features)
    return _shap_feature_matrix(values, len(features), features.shape[1])


def _labels_and_scores(scores: np.ndarray, labels: np.ndarray, problem_type: str) -> tuple[np.ndarray, np.ndarray]:
    """Drop rows with missing labels; validate binary labels are 0/1."""
    labels = pd.to_numeric(pd.Series(labels), errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(labels) & np.isfinite(scores)
    labels, scores = labels[mask], scores[mask]
    if problem_type == "binary":
        unique = set(np.unique(labels))
        if not unique <= {0.0, 1.0} or len(unique) < 2:
            raise ValueError(f"Binary labels must contain both 0 and 1; found {sorted(unique)}.")
        labels = labels.astype(int)
    return scores, labels


def compare_prediction_distributions(source_scores: np.ndarray, target_scores: np.ndarray,
                                     config: DriftConfig) -> TaskResult:
    """Compare model output distributions: KS, PSI and, for binary models, the predicted-positive rate
    and the share of borderline predictions near the decision threshold."""
    s, t = np.asarray(source_scores, float), np.asarray(target_scores, float)
    ks = stats.ks_2samp(s, t)
    row = {"feature": GLOBAL_FEATURE, "mean_source": s.mean(), "mean_target": t.mean(), "ks_d": ks.statistic,
           "ks_p": ks.pvalue, "psi": population_stability_index(s, t, config.psi_bins)}
    if config.problem_type == "binary":
        thr, band = config.decision_threshold, config.thresholds.near_threshold_band
        pos_s, pos_t = int((s >= thr).sum()), int((t >= thr).sum())
        near_s, near_t = int((np.abs(s - thr) <= band).sum()), int((np.abs(t - thr) <= band).sum())
        h = _cohens_h(pos_t / len(t), pos_s / len(s))
        row.update({"threshold": thr, "positive_rate_source": pos_s / len(s), "positive_rate_target": pos_t / len(t),
                    "positive_rate_cohens_h": h, "positive_rate_cohens_h_abs": abs(h),
                    "positive_rate_p": _two_proportion_p(pos_t, len(t), pos_s, len(s)),
                    "near_threshold_share_source": near_s / len(s), "near_threshold_share_target": near_t / len(t),
                    "near_threshold_p": _two_proportion_p(near_t, len(t), near_s, len(s))})
    bins = np.linspace(min(s.min(), t.min()), max(s.max(), t.max()), 41)
    plot = {"bins": bins, "source_hist": np.histogram(s, bins=bins, density=True)[0],
            "target_hist": np.histogram(t, bins=bins, density=True)[0],
            "threshold": config.decision_threshold if config.problem_type == "binary" else None}
    return TaskResult("prediction_drift", {"main": pd.DataFrame([row])}, plot_data=plot)


def compare_shap_distributions(model: Any, source_features: pd.DataFrame, target_features: pd.DataFrame,
                               config: DriftConfig) -> TaskResult:
    """Compare TreeSHAP attributions: per-feature KS on contributions, mean |SHAP| change, and the rank
    correlation of importance. Also defines the model-important features (top features reaching
    ``important_feature_cumulative_share`` of total mean |SHAP| on the source)."""
    rng = np.random.default_rng(config.seed)
    xs = _subsample(source_features, config.max_shap_rows, rng)
    xt = _subsample(target_features, config.max_shap_rows, rng)
    cs, ct = predict_contributions(model, xs), predict_contributions(model, xt)
    names = list(source_features.columns)
    mean_abs_s, mean_abs_t = np.abs(cs).mean(axis=0), np.abs(ct).mean(axis=0)
    total = mean_abs_s.sum()
    share = mean_abs_s / total if total > 0 else np.zeros_like(mean_abs_s)
    table = pd.DataFrame({"feature": names, "mean_abs_shap_source": mean_abs_s, "mean_abs_shap_target": mean_abs_t,
                          "importance_share_source": share})
    table["importance_rank_source"] = table["mean_abs_shap_source"].rank(ascending=False, method="min").astype(int)
    table["importance_rank_target"] = table["mean_abs_shap_target"].rank(ascending=False, method="min").astype(int)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(mean_abs_s > 0, (mean_abs_t - mean_abs_s) / mean_abs_s, np.nan)
    table["mean_abs_shap_rel_change"] = rel
    table["mean_abs_shap_rel_change_abs"] = np.abs(rel)
    ks = [stats.ks_2samp(cs[:, j], ct[:, j]) for j in range(len(names))]
    table["shap_ks_d"] = [r.statistic for r in ks]
    table["shap_ks_p"] = [r.pvalue for r in ks]
    table = adjust_pvalues(table, ["shap_ks_p"], config)
    table = table.sort_values("mean_abs_shap_source", ascending=False, ignore_index=True)
    cumulative_before = table["importance_share_source"].cumsum() - table["importance_share_source"]
    table["important"] = (cumulative_before < config.thresholds.important_feature_cumulative_share) & (
        table["importance_share_source"] > 0)
    rho = stats.spearmanr(mean_abs_s, mean_abs_t).statistic if len(names) > 2 else np.nan
    summary = pd.DataFrame([{"feature": GLOBAL_FEATURE, "importance_rank_rho": rho,
                             "rank_rho_deficit": 1 - rho if np.isfinite(rho) else np.nan,
                             "n_important": int(table["important"].sum()), "n_rows_source": len(xs),
                             "n_rows_target": len(xt)}])
    return TaskResult("shap_drift", {"main": table, "summary": summary})


def assess_calibration(source_scores: np.ndarray, source_labels: np.ndarray, config: DriftConfig,
                       n_bins: int = 10) -> TaskResult:
    """Calibration of the model on the source: Brier score, ECE, calibration slope and intercept.

    Good calibration is a prerequisite for confidence-based performance estimation (CBPE).
    """
    scores, labels = _labels_and_scores(np.asarray(source_scores, float), source_labels, "binary")
    p = np.clip(scores, 1e-6, 1 - 1e-6)
    bin_idx = np.minimum((p * n_bins).astype(int), n_bins - 1)
    ece = sum(abs(labels[bin_idx == b].mean() - p[bin_idx == b].mean()) * (bin_idx == b).mean()
              for b in range(n_bins) if (bin_idx == b).any())
    logit = np.log(p / (1 - p))[:, None]
    recalibration = LogisticRegression(C=1e6, max_iter=1000).fit(logit, labels)
    quantile_bins = pd.qcut(p, q=n_bins, labels=False, duplicates="drop")
    reliability = pd.DataFrame({"p": p, "y": labels, "bin": quantile_bins}).groupby("bin").agg(
        mean_predicted=("p", "mean"), observed_rate=("y", "mean"), n=("y", "size")).reset_index(drop=True)
    main = pd.DataFrame([{"feature": GLOBAL_FEATURE, "brier": brier_score_loss(labels, p), "ece": float(ece),
                          "calibration_slope": float(recalibration.coef_[0, 0]),
                          "calibration_intercept": float(recalibration.intercept_[0]), "n": len(p)}])
    return TaskResult("calibration", {"main": main, "reliability": reliability},
                      plot_data={"reliability": reliability.to_dict(orient="list")})


def compute_density_ratio_weights(domain_scores_source: np.ndarray, n_source: int, n_target: int,
                                  clip_quantile: float) -> tuple[np.ndarray, dict[str, float]]:
    """Importance weights w(x) = p_target(x) / p_source(x) for source rows from domain-classifier scores.

    w = P(target|x) / P(source|x) * n_source / n_target, clipped at ``clip_quantile``.

    Returns:
        (weights, info) where info holds the Kish effective sample size and clipping details.
    """
    p = np.clip(np.asarray(domain_scores_source, float), 1e-6, 1 - 1e-6)
    weights = p / (1 - p) * n_source / n_target
    cap = float(np.quantile(weights, clip_quantile))
    n_clipped = int((weights > cap).sum())
    weights = np.minimum(weights, cap)
    ess = float(weights.sum() ** 2 / np.square(weights).sum())
    return weights, {"ess": ess, "ess_ratio": ess / len(weights), "ess_deficit": 1 - ess / len(weights),
                     "weight_clip_value": cap, "n_clipped": n_clipped, "n_rows": len(weights)}


def _metric_specs(problem_type: str) -> list[tuple[str, Callable[..., float], bool]]:
    """(name, metric(y, score, sample_weight=None), higher_is_better) for a problem type."""
    if problem_type == "binary":
        return [
            ("auc", lambda y, p, sample_weight=None: roc_auc_score(y, p, sample_weight=sample_weight), True),
            ("brier", lambda y, p, sample_weight=None: brier_score_loss(y, p, sample_weight=sample_weight), False),
            ("log_loss", lambda y, p, sample_weight=None: log_loss(y, np.clip(p, 1e-6, 1 - 1e-6),
                                                                   sample_weight=sample_weight, labels=[0, 1]), False),
        ]
    return [
        ("rmse", lambda y, p, sample_weight=None: float(np.sqrt(mean_squared_error(y, p, sample_weight=sample_weight))),
         False),
        ("mae", lambda y, p, sample_weight=None: mean_absolute_error(y, p, sample_weight=sample_weight), False),
    ]


def estimate_importance_weighted_performance(source_scores: np.ndarray, source_labels: np.ndarray,
                                             domain_scores_source: np.ndarray, n_source_domain: int,
                                             n_target_domain: int, config: DriftConfig) -> TaskResult:
    """Estimate target performance by re-weighting labelled source rows with density-ratio weights.

    Assumes covariate shift only (P(y|x) unchanged) and overlap. Reliability is summarised by the
    Kish effective sample size ratio. Bootstrap CIs resample source rows with their weights.
    """
    weights, info = compute_density_ratio_weights(domain_scores_source, n_source_domain, n_target_domain,
                                                  config.weight_clip_quantile)
    scores = np.asarray(source_scores, float)
    labels_numeric = pd.to_numeric(pd.Series(source_labels), errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(labels_numeric) & np.isfinite(scores)
    scores, labels, weights = scores[mask], labels_numeric[mask], weights[mask]
    if config.problem_type == "binary":
        labels = labels.astype(int)
    rng = np.random.default_rng(config.seed)
    rows = []
    for name, metric, higher_is_better in _metric_specs(config.problem_type):
        reference = metric(labels, scores)
        estimate = metric(labels, scores, sample_weight=weights)
        ci_low = ci_high = np.nan
        if config.n_bootstrap > 0:
            boot = []
            for _ in range(config.n_bootstrap):
                idx = rng.integers(0, len(scores), len(scores))
                try:
                    boot.append(metric(labels[idx], scores[idx], sample_weight=weights[idx]))
                except ValueError:
                    continue
            if boot:
                ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
        degradation = reference - estimate if higher_is_better else estimate - reference
        rows.append({"metric": name, "higher_is_better": higher_is_better, "reference_value": reference,
                     "estimate": estimate, "ci_low": ci_low, "ci_high": ci_high, "degradation": degradation})
    summary = pd.DataFrame([{"feature": GLOBAL_FEATURE, **info}])
    weight_bins = np.quantile(weights, np.linspace(0, 1, 41))
    plot = {"weights_hist_bins": np.unique(weight_bins),
            "weights_hist": np.histogram(weights, bins=np.unique(weight_bins))[0]}
    return TaskResult("weighted_performance", {"main": pd.DataFrame(rows), "summary": summary}, plot_data=plot)


def _expected_auc(probabilities: np.ndarray) -> float:
    """Expected ROC AUC if labels ~ Bernoulli(probabilities) (ratio of expectations, self-pairs removed)."""
    p = np.asarray(probabilities, float)
    _, inverse = np.unique(p, return_inverse=True)
    pos = np.bincount(inverse, weights=p)
    neg = np.bincount(inverse, weights=1 - p)
    neg_below = np.cumsum(neg) - neg
    self_pairs = np.sum(p * (1 - p))
    numerator = np.sum(pos * (neg_below + 0.5 * neg)) - 0.5 * self_pairs
    denominator = p.sum() * (1 - p).sum() - self_pairs
    return float(numerator / denominator) if denominator > 0 else np.nan


def _expected_binary_metrics(probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    """CBPE expected metrics from calibrated probabilities."""
    p = np.clip(np.asarray(probabilities, float), 0, 1)
    predicted = p >= threshold
    tp, fp = p[predicted].sum(), (1 - p[predicted]).sum()
    fn, tn = p[~predicted].sum(), (1 - p[~predicted]).sum()
    return {"auc": _expected_auc(p), "accuracy": (tp + tn) / len(p),
            "precision": tp / (tp + fp) if tp + fp > 0 else np.nan,
            "recall": tp / (tp + fn) if tp + fn > 0 else np.nan}


def _realized_binary_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float]:
    """Observed metrics on labelled data."""
    predicted = scores >= threshold
    tp = int((predicted & (labels == 1)).sum())
    fp = int((predicted & (labels == 0)).sum())
    fn = int((~predicted & (labels == 1)).sum())
    return {"auc": roc_auc_score(labels, scores), "accuracy": float((predicted == (labels == 1)).mean()),
            "precision": tp / (tp + fp) if tp + fp > 0 else np.nan,
            "recall": tp / (tp + fn) if tp + fn > 0 else np.nan}


def estimate_performance_cbpe(source_scores: np.ndarray, source_labels: np.ndarray | None,
                              target_scores: np.ndarray, config: DriftConfig) -> TaskResult:
    """Confidence-Based Performance Estimation: expected metrics on the target from predicted probabilities.

    Assumes the model is calibrated and there is no concept shift. Degradation is measured like-for-like
    (CBPE on source vs CBPE on target) so a shared calibration bias cancels; the gap between the realised
    and estimated source metric is the sanity check of the calibration assumption.

    Raises:
        ValueError: For non-binary problems.
    """
    if config.problem_type != "binary":
        raise ValueError("CBPE is implemented for binary classification only.")
    thr = config.decision_threshold
    source = np.asarray(source_scores, float)
    target = np.asarray(target_scores, float)
    source_estimated = _expected_binary_metrics(source, thr)
    target_estimated = _expected_binary_metrics(target, thr)
    realized = {name: np.nan for name in target_estimated}
    if source_labels is not None:
        scores, labels = _labels_and_scores(source, source_labels, "binary")
        realized = _realized_binary_metrics(scores, labels, thr)
    rng = np.random.default_rng(config.seed)
    boot = {name: [] for name in target_estimated}
    for _ in range(config.n_bootstrap):
        sample = _expected_binary_metrics(target[rng.integers(0, len(target), len(target))], thr)
        for name, value in sample.items():
            boot[name].append(value)
    rows = []
    for name, estimate in target_estimated.items():
        reference = source_estimated[name]
        ci_low, ci_high = (np.nanpercentile(boot[name], [2.5, 97.5]) if boot[name] else (np.nan, np.nan))
        rows.append({"metric": name, "higher_is_better": True, "source_realized": realized[name],
                     "source_estimated": source_estimated[name], "reference_value": reference, "estimate": estimate,
                     "ci_low": ci_low, "ci_high": ci_high, "degradation": reference - estimate,
                     "source_calibration_gap": realized[name] - source_estimated[name]})
    return TaskResult("cbpe", {"main": pd.DataFrame(rows)})


def _bbse_target_prevalence(source_pred: np.ndarray, source_labels: np.ndarray,
                            target_pred: np.ndarray) -> tuple[float, float]:
    """BBSE estimate of the target positive rate and the condition number of the confusion matrix."""
    confusion = np.array([[np.mean((source_pred == i) & (source_labels == j)) for j in (0, 1)] for i in (0, 1)])
    target_rates = np.array([np.mean(target_pred == 0), np.mean(target_pred == 1)])
    prior_source = np.array([np.mean(source_labels == 0), np.mean(source_labels == 1)])
    try:
        class_weights = np.linalg.solve(confusion, target_rates)
    except np.linalg.LinAlgError:
        return np.nan, np.inf
    prior_target = np.clip(class_weights * prior_source, 0, None)
    if prior_target.sum() == 0:
        return np.nan, float(np.linalg.cond(confusion))
    return float(prior_target[1] / prior_target.sum()), float(np.linalg.cond(confusion))


def estimate_label_shift_bbse(source_scores: np.ndarray, source_labels: np.ndarray, target_scores: np.ndarray,
                              config: DriftConfig) -> TaskResult:
    """Black Box Shift Estimation (Lipton et al., 2018) of the target class prior.

    Assumes P(x|y) is unchanged. The bootstrap CI resamples source and target rows; prior shift is
    flagged when the CI excludes the source prevalence.
    """
    scores, labels = _labels_and_scores(np.asarray(source_scores, float), source_labels, "binary")
    thr = config.decision_threshold
    source_pred = (scores >= thr).astype(int)
    target_pred = (np.asarray(target_scores, float) >= thr).astype(int)
    estimate, condition = _bbse_target_prevalence(source_pred, labels, target_pred)
    rng = np.random.default_rng(config.seed)
    boot = []
    for _ in range(config.n_bootstrap):
        idx_s = rng.integers(0, len(labels), len(labels))
        idx_t = rng.integers(0, len(target_pred), len(target_pred))
        boot.append(_bbse_target_prevalence(source_pred[idx_s], labels[idx_s], target_pred[idx_t])[0])
    ci_low, ci_high = np.nanpercentile(boot, [2.5, 97.5]) if boot else (np.nan, np.nan)
    source_prevalence = float(labels.mean())
    shifted = bool(np.isfinite(ci_low) and not ci_low <= source_prevalence <= ci_high)
    main = pd.DataFrame([{"feature": GLOBAL_FEATURE, "source_prevalence": source_prevalence,
                          "target_prevalence_estimate": estimate, "ci_low": ci_low, "ci_high": ci_high,
                          "prior_shift_detected": shifted, "confusion_condition_number": condition}])
    return TaskResult("label_shift", {"main": main})


# --------------------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------------------
def _safe_task(failed: dict[str, str], name: str, fn: Callable[..., TaskResult], *args: Any) -> TaskResult | None:
    """Run one task; a failure is logged and recorded instead of aborting the analysis."""
    start = time.perf_counter()
    try:
        result = fn(*args)
    except Exception as exc:  # noqa: BLE001 - one failing task must not abort the whole analysis
        logger.error("Task '%s' failed: %s", name, exc)
        logger.debug(traceback.format_exc())
        failed[name] = f"{type(exc).__name__}: {exc}"
        return None
    logger.debug("Task '%s' finished in %.2fs", name, time.perf_counter() - start)
    return result


def task_pair_map(mode: str) -> dict[str, str]:
    """Name the populations compared by each task.

    Args:
        mode: ``single`` uses one labelled table. ``paired`` uses a fit set and an evaluation set.

    Returns:
        Task name to the pair label stored on the run and printed in the report.

    Raises:
        ValueError: If ``mode`` is not ``single`` or ``paired``.
    """
    if mode not in ("single", "paired"):
        raise ValueError(f"mode must be 'single' or 'paired', got {mode!r}.")
    feature = "labelled set vs new" if mode == "single" else "fit set vs new"
    scored = "labelled set vs new" if mode == "single" else "evaluation set vs new"
    labelled_only = "labelled set" if mode == "single" else "evaluation set"
    pairs = {task: feature for task in FEATURE_TASKS}
    pairs.update({task: scored for task in MODEL_TASKS})
    pairs["calibration"] = labelled_only
    return pairs


def population_drift_analyzer(source: pd.DataFrame, target: pd.DataFrame, feature_cols: Sequence[str],
                              categorical_cols: Sequence[str] | None = None, model: Any | None = None,
                              label_col: str | None = None, config: DriftConfig | None = None, *,
                              scope: str = "all") -> DriftAnalysisResult:
    """Run comparison tasks on two populations.

    ``scope="features"`` runs margins, joint structure, and support. ``scope="model"`` runs
    scores, SHAP, calibration, and label-free performance, and stores the weighting
    classifier as ``weight_domain`` so it does not replace the fit-set separability test.

    Args:
        source: Training (reference) population.
        target: New population (or a pseudo-new subset when building the baseline).
        feature_cols: Features to compare (the model's input features).
        categorical_cols: Features to treat as categorical (others are inferred).
        model: Optional trained LightGBM model (Booster or sklearn estimator) enabling model-based tasks.
        label_col: Optional label column in ``source`` enabling calibration, weighting and BBSE.
        config: Analysis settings.
        scope: Which tasks to run. See the summary above.

    Returns:
        :class:`DriftAnalysisResult` with one :class:`TaskResult` per task that ran.

    Raises:
        ValueError: If ``scope`` is unknown, or the inputs fail validation.
    """
    if scope not in ANALYSIS_SCOPES:
        raise ValueError(f"scope must be one of {ANALYSIS_SCOPES}, got {scope!r}.")
    config = config or DriftConfig()
    start = time.perf_counter()
    feature_cols = list(feature_cols)
    schema = validate_inputs(source, target, feature_cols)
    if label_col is not None and label_col not in source.columns:
        raise ValueError(f"Label column '{label_col}' not found in the source population.")
    if label_col is not None and label_col in feature_cols:
        raise ValueError("The label column must not be one of the features.")
    feature_types = infer_feature_types(source, feature_cols, categorical_cols, config.discrete_max_unique)
    continuous = [col for col in feature_cols if feature_types[col] == CONTINUOUS]
    categorical = [col for col in feature_cols if feature_types[col] == CATEGORICAL]

    failed: dict[str, str] = {}
    tasks: dict[str, TaskResult] = {"schema": TaskResult("schema", {"main": schema})}

    def run(name: str, fn: Callable[..., TaskResult], *args: Any) -> TaskResult | None:
        result = _safe_task(failed, name, fn, *args)
        if result is not None:
            tasks[name] = result
        return result

    domain = None
    if scope in ("all", "features"):
        run("continuous", compare_continuous_distributions, source, target, continuous, config)
        run("equivalence", run_equivalence_tests, source, target, continuous, config)
        run("categorical", compare_categorical_distributions, source, target, categorical, config)
        run("missingness", compare_missingness, source, target, feature_cols, config)
        run("missingness_patterns", compare_missingness_patterns, source, target, feature_cols, config)
        domain = run("domain_classifier", classifier_two_sample_test, source, target, feature_cols, feature_types, config)
        run("mmd", mmd_test, source, target, feature_cols, feature_types, config)
        run("energy", energy_distance_test, source, target, feature_cols, feature_types, config)
        run("correlation", compare_correlation_structure, source, target, continuous, config)
        run("range_coverage", check_range_coverage, source, target, continuous, config)
        run("joint_outliers", detect_joint_outliers, source, target, feature_cols, feature_types, config)
    if scope == "model":
        domain = run("domain_classifier", classifier_two_sample_test, source, target, feature_cols, feature_types, config)
    if model is not None and scope in ("all", "model"):
        _run_model_tasks(run, failed, source, target, feature_cols, model, label_col, domain, config)
        if scope == "model":
            _detach_weight_domain(tasks)

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_source": len(source), "n_target": len(target), "feature_cols": feature_cols,
        "feature_types": feature_types, "label_col": label_col, "has_model": model is not None,
        "problem_type": config.problem_type, "config": dataclasses.asdict(config), "failed_tasks": failed,
        "notes": {name: result.notes for name, result in tasks.items() if result.notes},
        "elapsed_seconds": round(time.perf_counter() - start, 2),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "scipy": scipy.__version__, "scikit-learn": sklearn.__version__, "lightgbm": lgb.__version__,
                     "shap": shap.__version__},
    }
    return DriftAnalysisResult(tasks, metadata)


def _detach_weight_domain(tasks: dict[str, TaskResult]) -> None:
    """Move the weighting classifier off the fit-set separability task name."""
    if "domain_classifier" not in tasks:
        return
    weight = tasks.pop("domain_classifier")
    weight.name = "weight_domain"
    tasks["weight_domain"] = weight


def _run_model_tasks(run: Callable[..., TaskResult | None], failed: dict[str, str], source: pd.DataFrame,
                     target: pd.DataFrame, feature_cols: list[str], model: Any, label_col: str | None,
                     domain: TaskResult | None, config: DriftConfig) -> None:
    """Model-based tasks: prediction drift, SHAP drift, calibration and label-free performance estimation."""
    try:
        x_source = prepare_model_input(source, model, feature_cols)
        x_target = prepare_model_input(target, model, feature_cols)
        source_scores = predict_scores(model, x_source, config.problem_type)
        target_scores = predict_scores(model, x_target, config.problem_type)
    except Exception as exc:  # noqa: BLE001 - model problems disable model tasks only
        logger.error("Model prediction failed; model-based tasks skipped: %s", exc)
        failed["model_prediction"] = f"{type(exc).__name__}: {exc}"
        return
    binary = config.problem_type == "binary"
    run("prediction_drift", compare_prediction_distributions, source_scores, target_scores, config)
    run("shap_drift", compare_shap_distributions, model, x_source, x_target, config)
    labels = source[label_col].to_numpy() if label_col is not None else None
    if labels is not None:
        if binary:
            run("calibration", assess_calibration, source_scores, labels, config)
            run("label_shift", estimate_label_shift_bbse, source_scores, labels, target_scores, config)
        if domain is not None:
            positions = domain.arrays["source_positions"]
            run("weighted_performance", estimate_importance_weighted_performance, source_scores[positions],
                labels[positions], domain.arrays["source_domain_scores"], domain.arrays["n_source"],
                domain.arrays["n_target"], config)
    if binary:
        run("cbpe", estimate_performance_cbpe, source_scores, labels, target_scores, config)


# --------------------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------------------
def collect_comparable_metrics(result: DriftAnalysisResult) -> pd.DataFrame:
    """Long table (task, table, feature, metric, value) of the statistics compared with the baseline."""
    rows = []
    for task, table_specs in COMPARABLE_METRICS.items():
        for table_name, (key_col, metrics) in table_specs.items():
            table = result.table(task, table_name)
            if table.empty or key_col not in table.columns:
                continue
            for metric in metrics:
                if metric not in table.columns:
                    continue
                for key, value in zip(table[key_col], table[metric]):
                    rows.append({"task": task, "table": table_name, "feature": str(key), "metric": metric,
                                 "value": float(value) if pd.notna(value) else np.nan})
    return pd.DataFrame(rows, columns=["task", "table", "feature", "metric", "value"])


def save_analysis(result: DriftAnalysisResult, output_dir: str | Path) -> Path:
    """Write every task table to ``tables/<task>__<table>.csv`` plus plot data and metadata JSON."""
    output_dir = Path(output_dir)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    for task, task_result in result.tasks.items():
        for table_name, table in task_result.tables.items():
            table.to_csv(tables_dir / f"{task}__{table_name}.csv", index=False)
    write_json({task: tr.plot_data for task, tr in result.tasks.items() if tr.plot_data}, output_dir / "plot_data.json")
    write_json(result.metadata, output_dir / "metadata.json")
    logger.info("Saved analysis to %s", output_dir)
    return output_dir


# --------------------------------------------------------------------------------------
# Baseline pipeline
# --------------------------------------------------------------------------------------
def summarize_baseline(statistics: pd.DataFrame) -> pd.DataFrame:
    """Per-metric summary (mean, sd, percentiles) of the baseline null distributions."""
    grouped = statistics.groupby(["task", "table", "feature", "metric"], sort=False)["value"]
    return grouped.agg(
        n_rounds="count", mean="mean", std="std", p50=lambda v: v.quantile(0.50), p90=lambda v: v.quantile(0.90),
        p95=lambda v: v.quantile(0.95), p99=lambda v: v.quantile(0.99), max="max").reset_index()


def build_baseline(source: pd.DataFrame, feature_cols: Sequence[str], categorical_cols: Sequence[str] | None = None,
                   model: Any | None = None, label_col: str | None = None, n_rounds: int = 30,
                   subset_size: int | None = None, subset_fraction: float = 0.3, config: DriftConfig | None = None,
                   output_dir: str | Path | None = None, *, scope: str = "all") -> BaselineResult:
    """Build the source-vs-source null: run the analyzer on ``n_rounds`` random splits of the training data.

    In each round a random subset of ``subset_size`` rows plays the "new population" and the remaining
    rows play the training population. Set ``subset_size`` to the expected size of the new population;
    many statistics are biased upward in small samples, so sizes should match.

    Raises:
        ValueError: If ``n_rounds`` < 2 or the subset size leaves too few rows on either side.
    """
    config = config or DriftConfig()
    if n_rounds < 2:
        raise ValueError("n_rounds must be at least 2.")
    n = len(source)
    size = int(subset_size) if subset_size else int(round(subset_fraction * n))
    if not MIN_GROUP_ROWS <= size <= n - MIN_GROUP_ROWS:
        raise ValueError(f"Subset size {size} must be between {MIN_GROUP_ROWS} and {n - MIN_GROUP_ROWS} "
                         f"for a training population of {n} rows.")
    round_config = config.for_baseline()
    rng = np.random.default_rng(config.seed)
    frames, failed_counts = [], {}
    feature_types: dict[str, str] = {}
    for r in range(n_rounds):
        start = time.perf_counter()
        order = rng.permutation(n)
        pseudo_new, reference = source.iloc[order[:size]], source.iloc[order[size:]]
        result = population_drift_analyzer(reference, pseudo_new, feature_cols, categorical_cols, model, label_col,
                                           dataclasses.replace(round_config, seed=config.seed + r + 1), scope=scope)
        feature_types = result.metadata["feature_types"]
        long = collect_comparable_metrics(result)
        long.insert(0, "round", r)
        frames.append(long)
        for task in result.metadata["failed_tasks"]:
            failed_counts[task] = failed_counts.get(task, 0) + 1
        logger.info("Baseline round %d/%d finished in %.1fs", r + 1, n_rounds, time.perf_counter() - start)
    statistics = pd.concat(frames, ignore_index=True)
    summary = summarize_baseline(statistics)
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "n_rounds": n_rounds,
        "subset_size": size, "n_source": n, "feature_cols": list(feature_cols), "feature_types": feature_types,
        "categorical_cols": list(categorical_cols or []), "label_col": label_col, "has_model": model is not None,
        "scope": scope,
        "config": dataclasses.asdict(config), "failed_task_counts": failed_counts,
    }
    baseline = BaselineResult(statistics, summary, metadata)
    if output_dir is not None:
        save_baseline(baseline, output_dir)
    return baseline


def save_baseline(baseline: BaselineResult, output_dir: str | Path) -> Path:
    """Write baseline statistics, summary and metadata."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline.statistics.to_csv(output_dir / "baseline_statistics.csv", index=False)
    baseline.summary.to_csv(output_dir / "baseline_summary.csv", index=False)
    write_json(baseline.metadata, output_dir / "baseline_metadata.json")
    logger.info("Saved baseline (%d rounds) to %s", baseline.metadata["n_rounds"], output_dir)
    return output_dir


def load_baseline(baseline_dir: str | Path) -> BaselineResult:
    """Load a baseline written by :func:`save_baseline`."""
    baseline_dir = Path(baseline_dir)
    required = ["baseline_statistics.csv", "baseline_summary.csv", "baseline_metadata.json"]
    missing = [name for name in required if not (baseline_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Baseline directory {baseline_dir} is missing {missing}.")
    key_types = {"task": str, "table": str, "feature": str, "metric": str}
    statistics = pd.read_csv(baseline_dir / "baseline_statistics.csv", dtype=key_types)
    summary = pd.read_csv(baseline_dir / "baseline_summary.csv", dtype=key_types)
    metadata = json.loads((baseline_dir / "baseline_metadata.json").read_text())
    return BaselineResult(statistics, summary, metadata)


def compare_to_baseline(observed: pd.DataFrame, baseline_statistics: pd.DataFrame,
                        percentile: float = 95.0) -> pd.DataFrame:
    """Position of each observed statistic within its baseline null distribution.

    Returns:
        Table with the baseline threshold (``percentile``), the empirical p-value
        ``(1 + #null >= observed) / (1 + n_null)`` and ``exceeds_baseline``.
    """
    keys = ["task", "table", "feature", "metric"]
    null_groups = {key: group["value"].dropna().to_numpy() for key, group in baseline_statistics.groupby(keys)}
    rows = []
    for record in observed.itertuples(index=False):
        null = null_groups.get((record.task, record.table, record.feature, record.metric), np.array([]))
        row = {"task": record.task, "table": record.table, "feature": record.feature, "metric": record.metric,
               "observed": record.value, "n_baseline_rounds": len(null), "baseline_median": np.nan,
               "baseline_threshold": np.nan, "baseline_max": np.nan, "empirical_p": np.nan, "exceeds_baseline": None}
        if len(null) and np.isfinite(record.value):
            threshold = float(np.percentile(null, percentile))
            row.update({"baseline_median": float(np.median(null)), "baseline_threshold": threshold,
                        "baseline_max": float(null.max()),
                        "empirical_p": float((1 + np.sum(null >= record.value)) / (1 + len(null))),
                        "exceeds_baseline": bool(record.value > threshold)})
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Interpretation
# --------------------------------------------------------------------------------------
def _baseline_lookup(baseline_comparison: pd.DataFrame | None) -> dict[tuple[str, str, str], Any]:
    """(task, feature, metric) -> exceeds_baseline (True / False / None)."""
    if baseline_comparison is None or baseline_comparison.empty:
        return {}
    return {(row.task, str(row.feature), row.metric): row.exceeds_baseline
            for row in baseline_comparison.itertuples(index=False)}


def _beyond_baseline(lookup: dict, task: str, feature: str, metrics: Sequence[str]) -> bool:
    """True if any metric exceeds its baseline threshold; True when no baseline exists (no evidence to clear)."""
    known = [lookup.get((task, feature, metric)) for metric in metrics]
    known = [value for value in known if value is not None and not (isinstance(value, float) and np.isnan(value))]
    return any(bool(value) for value in known) if known else True


def _worst(levels: Sequence[str]) -> str:
    """Most severe of a set of traffic-light levels."""
    levels = [level for level in levels if level in LEVEL_ORDER and level != "info"]
    return max(levels, key=LEVEL_ORDER.get) if levels else "mild"


def _value(table: pd.DataFrame, column: str, default: float = np.nan) -> float:
    """First value of a single-row table column, or ``default``."""
    if table.empty or column not in table.columns:
        return default
    value = table[column].iloc[0]
    return float(value) if pd.notna(value) else default


def assess_features(result: DriftAnalysisResult, lookup: dict, config: DriftConfig) -> pd.DataFrame:
    """Per-feature severity combining significance, effect size, baseline position and model importance.

    * severe: significant (adjusted p < alpha), beyond the baseline, and a large effect.
    * moderate: significant, beyond the baseline, and at least a moderate effect.
    * mild: otherwise (including statistically detectable but negligible differences).
    Without a model every feature is treated as important (conservative).
    """
    th, alpha = config.thresholds, config.alpha
    feature_types = result.metadata["feature_types"]
    continuous = result.table("continuous").set_index("feature") if not result.table("continuous").empty else None
    categorical = result.table("categorical").set_index("feature") if not result.table("categorical").empty else None
    equivalence = result.table("equivalence").set_index("feature") if not result.table("equivalence").empty else None
    shap = result.table("shap_drift").set_index("feature") if not result.table("shap_drift").empty else None
    missing = result.table("missingness").set_index("feature") if not result.table("missingness").empty else None
    rows = []
    for feature in result.metadata["feature_cols"]:
        ftype = feature_types[feature]
        row = {"feature": feature, "type": ftype, "severity": "mild", "important": True,
               "model_importance_share": np.nan, "significant": False, "beyond_baseline": False,
               "psi": np.nan, "ks_d": np.nan, "smd": np.nan, "cramers_v": np.nan, "equivalent_mean": None,
               "reasons": ""}
        if shap is not None and feature in shap.index:
            row["important"] = bool(shap.loc[feature, "important"])
            row["model_importance_share"] = float(shap.loc[feature, "importance_share_source"])
        reasons = []
        if ftype == CONTINUOUS and continuous is not None and feature in continuous.index \
                and pd.notna(continuous.loc[feature].get("ks_d", np.nan)):
            stat = continuous.loc[feature]
            psi, ks_d, smd_abs = stat["psi"], stat["ks_d"], stat["smd_abs"]
            significant = stat.get("ks_p_adj", 1.0) < alpha or stat.get("ad_p_adj", 1.0) < alpha
            beyond = _beyond_baseline(lookup, "continuous", feature, ["ks_d", "psi", "wasserstein_norm"])
            large = psi >= th.psi_major or ks_d >= th.ks_major or smd_abs >= th.smd_large
            moderate = psi >= th.psi_moderate or ks_d >= th.ks_moderate or smd_abs >= th.smd_small
            row.update({"psi": psi, "ks_d": ks_d, "smd": stat["smd"], "significant": bool(significant),
                        "beyond_baseline": bool(beyond)})
            if equivalence is not None and feature in equivalence.index:
                row["equivalent_mean"] = bool(equivalence.loc[feature, "equivalent"])
            reasons += [f"Population Stability Index (PSI)={psi:.3f}",
                        f"Kolmogorov–Smirnov statistic (KS D)={ks_d:.3f}",
                        f"standardized mean difference (SMD)={stat['smd']:+.2f}"]
            if np.isfinite(stat.get("var_ratio", np.nan)):
                reasons.append(f"var ratio={stat['var_ratio']:.2f}")
        elif ftype == CATEGORICAL and categorical is not None and feature in categorical.index \
                and pd.notna(categorical.loc[feature].get("cramers_v", np.nan)):
            stat = categorical.loc[feature]
            psi, v, unseen = stat["psi"], stat["cramers_v"], stat["unseen_share"]
            significant = stat.get("chi2_p_adj", 1.0) < alpha
            beyond = _beyond_baseline(lookup, "categorical", feature, ["cramers_v", "psi", "js_distance"])
            large = psi >= th.psi_major or v >= th.cramers_v_major
            moderate = psi >= th.psi_moderate or v >= th.cramers_v_moderate or unseen > th.unseen_level_share
            row.update({"psi": psi, "cramers_v": v, "significant": bool(significant), "beyond_baseline": bool(beyond)})
            reasons += [f"Population Stability Index (PSI)={psi:.3f}", f"Cramér's V={v:.3f}"]
            if unseen > 0:
                reasons.append(f"unseen levels={unseen:.1%}")
        else:
            significant = beyond = large = moderate = False
            reasons.append("distribution not tested (insufficient data)")
        if significant and beyond and large:
            row["severity"] = "severe"
        elif significant and beyond and moderate:
            row["severity"] = "moderate"
        if reasons and not reasons[-1].startswith("distribution not tested"):
            reasons.append("adj. p<α" if significant else "not significant")
            reasons.append("beyond baseline" if beyond else "within baseline")
        if row["equivalent_mean"]:
            reasons.append("means equivalent (two one-sided tests, TOST)")
        if missing is not None and feature in missing.index:
            miss = missing.loc[feature]
            if (miss["cohens_h_abs"] >= th.cohens_h_small and miss["z_p_adj"] < alpha
                    and _beyond_baseline(lookup, "missingness", feature, ["cohens_h_abs"])):
                miss_level = "severe" if miss["cohens_h_abs"] >= th.cohens_h_large else "moderate"
                row["severity"] = _worst([row["severity"], miss_level])
                row["significant"] = row["beyond_baseline"] = True
                reasons.append(f"missing rate {miss['missing_rate_source']:.1%}→{miss['missing_rate_target']:.1%} "
                               f"(h={miss['cohens_h']:+.2f})")
        row["reasons"] = "; ".join(reasons)
        rows.append(row)
    table = pd.DataFrame(rows)
    order = table["severity"].map(LEVEL_ORDER)
    return table.assign(_order=order).sort_values(["_order", "psi"], ascending=[False, False]).drop(
        columns="_order").reset_index(drop=True)


def _signal(name: str, level: str, value: str, detail: str, important: bool = False) -> dict[str, Any]:
    return {"signal": name, "level": level, "value": value, "detail": detail, "important": important}


def _performance_signal(name: str, table: pd.DataFrame, metric: str, config: DriftConfig) -> dict[str, Any] | None:
    """Traffic light for an estimated metric against ``performance_tolerance`` (absolute for AUC,
    relative for regression error metrics).

    severe: the whole 95% CI is beyond tolerance; moderate: the point estimate is beyond tolerance, or the CI
    crosses it while the point degradation exceeds half the tolerance; mild otherwise. A wide CI alone
    is reported as imprecision, not as degradation.
    """
    if table.empty or metric not in set(table["metric"]):
        return None
    row = table.set_index("metric").loc[metric]
    reference, estimate, low, high = row["reference_value"], row["estimate"], row["ci_low"], row["ci_high"]
    relative = config.problem_type == "regression"
    tolerance = config.performance_tolerance * abs(reference) if relative else config.performance_tolerance
    higher_is_better = bool(row["higher_is_better"])
    bound = reference - tolerance if higher_is_better else reference + tolerance
    worse = (lambda x: x < bound) if higher_is_better else (lambda x: x > bound)
    has_ci = np.isfinite(low) and np.isfinite(high)
    ci_worst, ci_best = (low, high) if higher_is_better else (high, low)
    degradation = reference - estimate if higher_is_better else estimate - reference
    crosses = has_ci and worse(ci_worst)
    if has_ci and worse(ci_best):
        level = "severe"
    elif worse(estimate) or (crosses and degradation > tolerance / 2):
        level = "moderate"
    else:
        level = "mild"
    ci_text = f" [95% confidence interval (CI) {low:.3f}–{high:.3f}]" if has_ci else ""
    detail = f"Tolerance {tolerance:.3f}; severe only if the whole confidence interval (CI) is beyond tolerance."
    if crosses and level == "mild":
        detail += " The confidence interval (CI) is wide relative to the tolerance (imprecise estimate)."
    shown = _DISPLAY_METRICS.get(str(metric), str(metric))
    return _signal(name, level, f"{shown} {estimate:.3f}{ci_text} vs reference {reference:.3f}", detail)


def interpret_results(result: DriftAnalysisResult, baseline_comparison: pd.DataFrame | None,
                      config: DriftConfig) -> dict[str, Any]:
    """Traffic-light signals and an overall verdict following the decision framework.

    Verdict: severe if any severe signal; mild if no severe, at most two moderate signals and none involving
    model-important features; moderate otherwise. ``info`` signals are reported but not counted.
    """
    th, alpha = config.thresholds, config.alpha
    lookup = _baseline_lookup(baseline_comparison)
    features = assess_features(result, lookup, config)
    has_model = result.metadata["has_model"]
    signals = []

    important = features[features["important"]]
    other = features[~features["important"]]
    label = "model-important features" if has_model else "all features (no model supplied)"
    flagged = important[important["severity"] != "mild"]
    signals.append(_signal(f"Feature drift: {label}", _worst(important["severity"]),
                           f"{(important.severity == 'severe').sum()} severe, {(important.severity == 'moderate').sum()} moderate "
                           f"of {len(important)}",
                           ", ".join(f"{r.feature} ({r.severity})" for r in flagged.itertuples()) or "none flagged",
                           important=True))
    if has_model and len(other):
        level = "moderate" if (other["severity"] != "mild").any() else "mild"
        flagged = other[other["severity"] != "mild"]
        signals.append(_signal("Feature drift: low-importance features", level,
                               f"{len(flagged)} of {len(other)} flagged",
                               (", ".join(flagged["feature"]) or "none flagged") + " (capped at moderate: low impact)"))

    missing = result.table("missingness")
    if not missing.empty:
        flagged = missing[(missing["cohens_h_abs"] >= th.cohens_h_small) & (missing["z_p_adj"] < alpha)]
        flagged = flagged[np.array([_beyond_baseline(lookup, "missingness", f, ["cohens_h_abs"])
                                    for f in flagged["feature"]], dtype=bool)]
        important_set = set(important["feature"])
        level = "mild"
        if len(flagged):
            level = "moderate"
            if ((flagged["cohens_h_abs"] >= th.cohens_h_large) & flagged["feature"].isin(important_set)).any():
                level = "severe"
        signals.append(_signal("Missing-value rates", level, f"{len(flagged)} feature(s) with |h| ≥ {th.cohens_h_small}",
                               ", ".join(f"{r.feature} ({r.missing_rate_source:.1%}→{r.missing_rate_target:.1%})"
                                         for r in flagged.itertuples()) or "no material change",
                               important=bool(flagged["feature"].isin(important_set).any())))

    categorical = result.table("categorical")
    if not categorical.empty and "unseen_share" in categorical:
        flagged = categorical[categorical["unseen_share"] > th.unseen_level_share]
        signals.append(_signal("Unseen category levels", "moderate" if len(flagged) else "mild",
                               f"{len(flagged)} feature(s) > {th.unseen_level_share:.0%} of rows",
                               ", ".join(f"{r.feature} ({r.unseen_share:.1%})" for r in flagged.itertuples())
                               or "none"))

    domain = result.table("domain_classifier")
    if not domain.empty:
        auc = _value(domain, "auc")
        within = not _beyond_baseline(lookup, "domain_classifier", GLOBAL_FEATURE, ["auc"])
        if within or auc <= th.domain_auc_mild:
            level = "mild"
        elif auc <= th.domain_auc_strong:
            level = "moderate"
        else:
            level = "severe"
        p_text = _value(domain, "permutation_p")
        p_text = f"permutation p={p_text:.3f}" if np.isfinite(p_text) else f"Mann-Whitney p={_value(domain, 'mannwhitney_p'):.2g}"
        drivers = result.table("domain_classifier", "importance").head(3)["feature"].tolist()
        signals.append(_signal("Joint distribution: classifier two-sample test", level,
                               f"area under the curve (AUC) {auc:.3f} ({p_text})",
                               ("within baseline" if within else "beyond baseline") + f"; top drivers: {', '.join(drivers)}"))

    kernel_tests = [(task, metric, label_text) for task, metric, label_text in
                    (("mmd", "mmd2", "squared maximum mean discrepancy (MMD²)"),
                     ("energy", "energy_distance", "energy")) if not result.table(task).empty]
    kernel_flags, kernel_values = [], []
    for task, metric, label_text in kernel_tests:
        table = result.table(task)
        p_value = _value(table, "permutation_p")
        beyond = _beyond_baseline(lookup, task, GLOBAL_FEATURE, [metric])
        significant = p_value < alpha / len(kernel_tests) if np.isfinite(p_value) else True
        kernel_flags.append(beyond and significant)
        kernel_values.append(f"{label_text} {_value(table, metric):.4f} (p={p_value:.3f})")
    if kernel_values:
        signals.append(_signal("Joint distribution: kernel/energy tests", "moderate" if any(kernel_flags) else "mild",
                               "; ".join(kernel_values),
                               f"Permutation tests (Bonferroni over {len(kernel_tests)}); moderate when significant "
                               "and beyond baseline"))

    correlation, pairs = result.table("correlation"), result.table("correlation", "pairs")
    if not correlation.empty:
        p_value = _value(correlation, "permutation_p")
        global_flag = _beyond_baseline(lookup, "correlation", GLOBAL_FEATURE, ["frobenius"]) and (
            p_value < alpha if np.isfinite(p_value) else True)
        important_set = set(important["feature"])
        pair_flags = pairs[(pairs["abs_delta"] > th.correlation_pair_delta) & (pairs["fisher_z_p_adj"] < alpha)
                           & pairs["feature_a"].isin(important_set) & pairs["feature_b"].isin(important_set)]
        signals.append(_signal("Dependence structure (Spearman)",
                               "moderate" if global_flag or len(pair_flags) else "mild",
                               f"Frobenius {_value(correlation, 'frobenius'):.3f} (p={p_value:.3f})",
                               ", ".join(f"{r.feature_a}–{r.feature_b} Δρ={r.delta:+.2f}" for r in pair_flags.head(5).itertuples())
                               or "no material pairwise change among important features"))

    outliers = result.table("joint_outliers")
    if not outliers.empty:
        share = _value(outliers, "ood_share")
        within = not _beyond_baseline(lookup, "joint_outliers", GLOBAL_FEATURE, ["ood_share"])
        if within or share < th.ood_share_caution:
            level = "mild"
        else:
            level = "moderate" if share <= th.ood_share_material else "severe"
        signals.append(_signal("Out-of-support rows (Isolation Forest)", level,
                               f"{share:.1%} of target rows (expected {config.ood_quantile:.0%})",
                               f"binomial p={_value(outliers, 'binomial_p'):.3g}; "
                               + ("within baseline" if within else "beyond baseline")))

    coverage = result.table("range_coverage")
    if not coverage.empty:
        flagged = coverage[(coverage["ratio_to_expected"] > th.range_outside_ratio) & (coverage["binomial_p_adj"] < alpha)]
        flagged = flagged[np.array([_beyond_baseline(lookup, "range_coverage", f, ["outside_share"])
                                    for f in flagged["feature"]], dtype=bool)]
        important_flags = flagged[flagged["feature"].isin(set(important["feature"]))]
        signals.append(_signal("Values outside the training range", "moderate" if len(important_flags) else "mild",
                               f"{len(flagged)} feature(s) > {th.range_outside_ratio:g}× expected",
                               ", ".join(f"{r.feature} ({r.outside_share:.1%})" for r in flagged.itertuples()) or "none",
                               important=bool(len(important_flags))))

    prediction = result.table("prediction_drift")
    if not prediction.empty:
        psi = _value(prediction, "psi")
        within = not _beyond_baseline(lookup, "prediction_drift", GLOBAL_FEATURE, ["psi", "ks_d"])
        level = "mild" if within or psi < th.psi_moderate else "moderate" if psi <= th.psi_major else "severe"
        signals.append(_signal("Model score distribution", level,
                               f"score Population Stability Index (PSI) {psi:.3f}, "
                               f"Kolmogorov–Smirnov statistic (KS D) {_value(prediction, 'ks_d'):.3f}",
                               "within baseline" if within else "beyond baseline", important=True))
        if config.problem_type == "binary":
            h, p_value = _value(prediction, "positive_rate_cohens_h_abs"), _value(prediction, "positive_rate_p")
            signals.append(_signal("Predicted positive rate", "moderate" if h >= th.cohens_h_small and p_value < alpha else "mild",
                                   f"{_value(prediction, 'positive_rate_source'):.1%} → {_value(prediction, 'positive_rate_target'):.1%}",
                                   f"Cohen's h={h:.2f}, p={p_value:.3g}"))

    shap_summary, shap_table = result.table("shap_drift", "summary"), result.table("shap_drift")
    if not shap_summary.empty:
        rho = _value(shap_summary, "importance_rank_rho")
        flagged = shap_table[shap_table["important"] & (shap_table["shap_ks_p_adj"] < alpha)
                             & (shap_table["shap_ks_d"] >= th.ks_moderate)]
        flagged = flagged[np.array([_beyond_baseline(lookup, "shap_drift", f, ["shap_ks_d"])
                                    for f in flagged["feature"]], dtype=bool)]
        level = "moderate" if (np.isfinite(rho) and rho < th.importance_rank_rho) or len(flagged) else "mild"
        signals.append(_signal("Model attributions (Shapley additive explanations, SHAP)", level,
                               f"importance rank ρ={rho:.2f}; {len(flagged)} important feature(s) with shifted "
                               "Shapley additive explanations (SHAP)",
                               ", ".join(flagged["feature"]) or "no material attribution shift on important features",
                               important=bool(len(flagged))))

    weighted_summary = result.table("weighted_performance", "summary")
    if not weighted_summary.empty:
        ratio = _value(weighted_summary, "ess_ratio")
        level = "mild" if ratio > th.ess_ratio_good else "moderate" if ratio >= th.ess_ratio_poor else "severe"
        signals.append(_signal("Population overlap (effective sample size)", level,
                               f"effective sample size (ESS) ratio {ratio:.2f}",
                               f"effective sample size (ESS) {_value(weighted_summary, 'ess'):.0f} of "
                               f"{_value(weighted_summary, 'n_rows'):.0f} source rows"))
    primary = "auc" if config.problem_type == "binary" else "rmse"
    weighted_signal = _performance_signal("Estimated performance: importance weighting",
                                          result.table("weighted_performance"), primary, config)
    if weighted_signal:
        signals.append(weighted_signal)
    cbpe_signal = _performance_signal(
        "Estimated performance: confidence-based performance estimation (CBPE)", result.table("cbpe"), "auc", config)
    if cbpe_signal:
        ece = _value(result.table("calibration"), "ece")
        if np.isfinite(ece) and ece > th.ece_max:
            cbpe_signal["level"] = "info"
            cbpe_signal["detail"] = (
                f"Unreliable: model miscalibrated on source (expected calibration error (ECE) {ece:.3f} > {th.ece_max}).")
        signals.append(cbpe_signal)
    label_shift = result.table("label_shift")
    if not label_shift.empty:
        shifted = bool(label_shift["prior_shift_detected"].iloc[0])
        signals.append(_signal("Class prior (black-box shift estimation, BBSE)", "moderate" if shifted else "mild",
                               f"{_value(label_shift, 'source_prevalence'):.1%} → "
                               f"{_value(label_shift, 'target_prevalence_estimate'):.1%} "
                               f"[{_value(label_shift, 'ci_low'):.1%}–{_value(label_shift, 'ci_high'):.1%}]",
                               ("Recalibrate intercept/threshold if shifted. " if shifted else
                                "CI includes source prevalence. ")
                               + "Valid only if P(x|y) is unchanged; under covariate shift prefer the "
                                 "confidence-based performance estimation (CBPE) and importance-weighted estimates."))
    calibration = result.table("calibration")
    if not calibration.empty:
        signals.append(_signal(
            "Source calibration (prerequisite for confidence-based performance estimation, CBPE)", "info",
            f"expected calibration error (ECE) {_value(calibration, 'ece'):.3f}, "
            f"slope {_value(calibration, 'calibration_slope'):.2f}",
                               "Computed on the training population; optimistic if these rows were used for training."))

    counted = [s for s in signals if s["level"] != "info"]
    n_severe = sum(s["level"] == "severe" for s in counted)
    n_moderate = sum(s["level"] == "moderate" for s in counted)
    moderate_important = any(s["level"] == "moderate" and s["important"] for s in counted)
    if n_severe:
        verdict_level = "severe"
    elif n_moderate <= 2 and not moderate_important:
        verdict_level = "mild"
    else:
        verdict_level = "moderate"
    verdict = VERDICT_TEXT[verdict_level]
    rule = ("severe if any severe signal; mild if no severe, ≤2 moderate and none on model-important features; moderate otherwise. "
            "Whether a given feature vector still maps to the same label cannot be assessed without labels.")
    _attach_pairs(signals, result.metadata.get("task_pairs", {}))
    return {"verdict": verdict, "verdict_level": verdict_level, "verdict_rule": rule, "n_severe": n_severe,
            "n_moderate": n_moderate, "signals": signals, "feature_assessment": features}


def _attach_pairs(signals: list[dict[str, Any]], task_pairs: dict[str, str]) -> None:
    """Record the population pair on each signal."""
    if not task_pairs:
        return
    for signal in signals:
        for needle, task in _SIGNAL_TASKS:
            if needle in signal["signal"]:
                signal["pair"] = task_pairs[task]
                break


# --------------------------------------------------------------------------------------
# Comparison pipeline and console output
# --------------------------------------------------------------------------------------
def _copy_warnings(extra_warnings: Sequence[str] | None) -> list[str]:
    """Copy caller-supplied warnings, or return an empty list.

    Args:
        extra_warnings: Messages to include on the decision, or None.

    Returns:
        A new list of warning strings.

    Raises:
        TypeError: If ``extra_warnings`` is a string or not a sequence.
    """
    if extra_warnings is None:
        return []
    if isinstance(extra_warnings, str) or not isinstance(extra_warnings, Sequence):
        raise TypeError("extra_warnings must be a sequence of strings.")
    return [str(item) for item in extra_warnings]


def run_comparison(source: pd.DataFrame, target: pd.DataFrame, feature_cols: Sequence[str],
                   categorical_cols: Sequence[str] | None = None, model: Any | None = None,
                   label_col: str | None = None, config: DriftConfig | None = None,
                   baseline_dir: str | Path | None = None, output_dir: str | Path = "drift_runs/comparison",
                   n_baseline_rounds: int = 30, extra_warnings: Sequence[str] | None = None) -> dict[str, Any]:
    """Full analysis of training vs new population, positioned against the baseline null.

    If ``baseline_dir`` is None a size-matched baseline is built and saved to ``output_dir/baseline``.
    ``extra_warnings`` is copied onto the decision; the CLI uses this when a model file cannot be loaded.

    Returns:
        Decision dictionary (also written to ``decision.json``).

    Raises:
        ValueError: If the baseline was built for a different feature set.
    """
    config = config or DriftConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_warnings = _copy_warnings(extra_warnings)
    if baseline_dir is not None:
        baseline = load_baseline(baseline_dir)
        baseline_features = set(baseline.metadata["feature_cols"])
        if baseline_features != set(feature_cols):
            raise ValueError(
                "The baseline feature set does not match this comparison. "
                f"Only in the baseline: {sorted(baseline_features - set(feature_cols)) or 'none'}. "
                f"Only in this comparison: {sorted(set(feature_cols) - baseline_features) or 'none'}."
            )
        ratio = len(target) / baseline.metadata["subset_size"]
        if not 0.5 <= ratio <= 2.0:
            run_warnings.append(
                f"New population has {len(target)} rows but the baseline used subsets of "
                f"{baseline.metadata['subset_size']} rows (ratio {ratio:.2f}). Size-dependent statistics "
                "(Population Stability Index (PSI), Kolmogorov–Smirnov statistic (KS D), "
                "classifier area under the curve (AUC)) are not directly comparable; rebuild the baseline with --subset-size "
                f"{len(target)}.")
    else:
        size = min(len(target), len(source) // 2)
        if size < len(target):
            run_warnings.append(f"Training population too small to match the new population size; baseline "
                                f"subsets use {size} rows.")
        logger.info("No baseline supplied: building a size-matched baseline (%d rounds of %d rows)",
                    n_baseline_rounds, size)
        baseline_dir = output_dir / "baseline"
        baseline = build_baseline(source, feature_cols, categorical_cols, model, label_col, n_baseline_rounds,
                                  subset_size=size, config=config, output_dir=baseline_dir)
    if baseline.metadata.get("has_model") != (model is not None):
        run_warnings.append("Baseline and comparison differ in whether a model was supplied; model-based "
                            "metrics have no baseline reference.")

    logger.info("Running full analysis: labelled set (%d rows) vs new population (%d rows)", len(source), len(target))
    result = population_drift_analyzer(source, target, feature_cols, categorical_cols, model, label_col, config)
    _stamp_reference(result, "single")
    return _emit_decision(result, baseline, baseline_dir, output_dir, config, run_warnings)


def _stamp_reference(result: DriftAnalysisResult, mode: str) -> None:
    """Record which populations each task compared."""
    result.metadata["reference_mode"] = mode
    result.metadata["task_pairs"] = task_pair_map(mode)


def _emit_decision(result: DriftAnalysisResult, baseline: BaselineResult, baseline_dir: str | Path,
                   output_dir: Path, config: DriftConfig, run_warnings: list[str]) -> dict[str, Any]:
    """Save the tables, score them against the baseline, and write ``decision.json``."""
    save_analysis(result, output_dir)
    comparison = compare_to_baseline(collect_comparable_metrics(result), baseline.statistics,
                                     config.thresholds.baseline_percentile)
    comparison.to_csv(output_dir / "baseline_comparison.csv", index=False)
    decision = interpret_results(result, comparison, config)
    decision["feature_assessment"].to_csv(output_dir / "feature_assessment.csv", index=False)
    if result.metadata["failed_tasks"]:
        run_warnings.append(f"Tasks that failed: {result.metadata['failed_tasks']}")
    payload = {key: value for key, value in decision.items() if key != "feature_assessment"}
    payload.update({"warnings": run_warnings, "baseline_dir": str(baseline_dir),
                    "baseline_rounds": baseline.metadata["n_rounds"],
                    "baseline_subset_size": baseline.metadata["subset_size"],
                    "reference_mode": result.metadata.get("reference_mode", "single"),
                    "task_pairs": result.metadata.get("task_pairs", {})})
    write_json(payload, output_dir / "decision.json")
    print(format_console_summary(decision, run_warnings))
    return decision


def run_paired_comparison(fit_set: pd.DataFrame, evaluation: pd.DataFrame, new: pd.DataFrame,
                          feature_cols: Sequence[str], categorical_cols: Sequence[str] | None = None,
                          model: Any | None = None, label_col: str | None = None,
                          config: DriftConfig | None = None, output_dir: str | Path = "drift_runs/comparison",
                          n_baseline_rounds: int = 30, extra_warnings: Sequence[str] | None = None) -> dict[str, Any]:
    """Compare the new population with a fit set and an evaluation set.

    Feature, joint, and support tests use the fit set. Score, attribution, and
    label-free performance tests use the evaluation set. Calibration uses the
    evaluation labels. Each reference gets its own size-matched baseline.

    Args:
        fit_set: Rows the model was fitted on.
        evaluation: Rows the model was scored on and was not fitted on.
        new: Unlabelled population.
        feature_cols: Model inputs. The label is rejected.
        categorical_cols: Columns treated as categorical.
        model: Fitted model, or None for feature-only checks.
        label_col: Label on the evaluation set.
        config: Run configuration.
        output_dir: Directory for the merged comparison and the baseline.
        n_baseline_rounds: Rounds in each reference's baseline.
        extra_warnings: Messages copied onto the decision (for example a model load error).

    Returns:
        Decision dictionary. Each signal has a ``pair`` field.

    Raises:
        ValueError: If columns are missing or a reference has fewer than ``len(new) + 30`` rows.
    """
    config = config or DriftConfig()
    output_dir = Path(output_dir)
    _validate_paired(fit_set, evaluation, new, feature_cols, label_col)
    baseline_dir = output_dir / "baseline"
    baseline = _paired_baseline(fit_set, evaluation, new, feature_cols, categorical_cols, model, label_col,
                                config, n_baseline_rounds, baseline_dir)
    result = _paired_analysis(fit_set, evaluation, new, feature_cols, categorical_cols, model, label_col, config)
    return _emit_decision(result, baseline, baseline_dir, output_dir, config, _copy_warnings(extra_warnings))


def _validate_paired(fit_set: pd.DataFrame, evaluation: pd.DataFrame, new: pd.DataFrame,
                     feature_cols: Sequence[str], label_col: str | None) -> None:
    """Reject inputs that cannot support the two-reference assignment."""
    if not feature_cols:
        raise ValueError("feature_cols is empty.")
    if label_col and label_col in feature_cols:
        raise ValueError(f"Leave the label column '{label_col}' out of the feature list.")
    _require_columns(fit_set, feature_cols, "Fit set")
    _require_columns(evaluation, feature_cols, "Evaluation set")
    _require_columns(new, feature_cols, "New population")
    if label_col and label_col not in evaluation.columns:
        raise ValueError(f"Evaluation set has no label column '{label_col}'.")
    _require_reference_size("Fit set", len(fit_set), len(new))
    _require_reference_size("Evaluation set", len(evaluation), len(new))


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    """Raise if ``frame`` does not contain every name in ``columns``."""
    missing = [col for col in columns if col not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}.")


def _require_reference_size(name: str, n_reference: int, n_new: int) -> None:
    """Require a reference large enough for a baseline matched to ``n_new``."""
    if n_new < MIN_GROUP_ROWS:
        raise ValueError(f"The new population has {n_new} rows; at least {MIN_GROUP_ROWS} are required.")
    minimum = n_new + MIN_GROUP_ROWS
    if n_reference < minimum:
        raise ValueError(f"{name} has {n_reference} rows. A baseline matched to {n_new} new rows needs at least {minimum}.")


def _paired_baseline(fit_set: pd.DataFrame, evaluation: pd.DataFrame, new: pd.DataFrame, feature_cols: Sequence[str],
                     categorical_cols: Sequence[str] | None, model: Any | None, label_col: str | None,
                     config: DriftConfig, n_rounds: int, output_dir: Path) -> BaselineResult:
    """Size-matched baselines for the fit-set tests and the evaluation-set tests."""
    _warn_if_new_exceeds_half("Fit set", len(fit_set), len(new))
    _warn_if_new_exceeds_half("Evaluation set", len(evaluation), len(new))
    fit_base = build_baseline(fit_set, feature_cols, categorical_cols, None, None, n_rounds,
                              subset_size=len(new), config=config, scope="features")
    eval_base = build_baseline(evaluation, feature_cols, categorical_cols, model, label_col, n_rounds,
                               subset_size=len(new), config=config, scope="model")
    statistics = pd.concat([fit_base.statistics, eval_base.statistics], ignore_index=True)
    metadata = dict(eval_base.metadata)
    metadata.update({"n_source": len(fit_set), "n_source_fit": len(fit_set), "n_source_evaluation": len(evaluation),
                     "has_model": model is not None, "scope": "paired", "feature_cols": list(feature_cols)})
    baseline = BaselineResult(statistics, summarize_baseline(statistics), metadata)
    save_baseline(baseline, output_dir)
    return baseline


def _warn_if_new_exceeds_half(name: str, n_reference: int, n_new: int) -> None:
    """Warn when a baseline round would hold out more rows than it keeps."""
    if n_new > n_reference / 2:
        logger.warning("%s has %d rows and the new population has %d. Baseline rounds then hold out more rows than they keep.",
                       name, n_reference, n_new)


def _paired_analysis(fit_set: pd.DataFrame, evaluation: pd.DataFrame, new: pd.DataFrame, feature_cols: Sequence[str],
                     categorical_cols: Sequence[str] | None, model: Any | None, label_col: str | None,
                     config: DriftConfig) -> DriftAnalysisResult:
    """Run feature tests on the fit set and score tests on the evaluation set."""
    logger.info("Feature tests use the fit set (%d rows). Score and performance tests use the evaluation set (%d rows).",
                len(fit_set), len(evaluation))
    feature = population_drift_analyzer(fit_set, new, feature_cols, categorical_cols, None, None, config, scope="features")
    scored = population_drift_analyzer(evaluation, new, feature_cols, categorical_cols, model, label_col, config,
                                       scope="model")
    tasks = {name: task for name, task in feature.tasks.items()}
    tasks.update({name: task for name, task in scored.tasks.items() if name != "schema"})
    metadata = dict(feature.metadata)
    metadata["failed_tasks"] = {**feature.metadata["failed_tasks"], **scored.metadata["failed_tasks"]}
    metadata.update({"has_model": model is not None, "label_col": label_col, "n_fit": len(fit_set),
                     "n_evaluation": len(evaluation), "n_target": len(new),
                     "elapsed_seconds": round(feature.metadata["elapsed_seconds"] + scored.metadata["elapsed_seconds"], 2)})
    result = DriftAnalysisResult(tasks, metadata)
    _stamp_reference(result, "paired")
    return result


def format_console_summary(decision: dict[str, Any], run_warnings: Sequence[str] = ()) -> str:
    """Plain-text summary of the verdict, signals and flagged features."""
    symbols = {"mild": "[mild]", "moderate": "[moderate]", "severe": "[severe]", "info": "[info]"}
    lines = ["", "=" * 100, f"VERDICT: {decision['verdict']}  ({decision['verdict_level'].upper()})",
             f"Rule: {decision['verdict_rule']}", "=" * 100]
    for signal in decision["signals"]:
        pair = f" [{signal['pair']}]" if signal.get("pair") else ""
        lines.append(f"{symbols[signal['level']]} {signal['signal']}{pair}: {signal['value']}")
        lines.append(f"       {signal['detail']}")
    features = decision["feature_assessment"]
    flagged = features[features["severity"] != "mild"]
    lines.append("-" * 100)
    lines.append(f"Flagged features: {len(flagged)} of {len(features)}")
    for row in flagged.itertuples():
        prefix = f"{row.reference} " if "reference" in features.columns else ""
        lines.append(f"  {row.severity.upper():5s} {prefix}{row.feature} ({row.type}"
                     f"{', model-important' if row.important else ''}): {row.reasons}")
    for message in run_warnings:
        lines.append(f"WARNING: {message}")
    lines.append("=" * 100)
    return "\n".join(lines)


def format_baseline_summary(baseline: BaselineResult) -> str:
    """Plain-text summary of the natural variability of the global metrics."""
    summary = baseline.summary
    global_rows = summary[summary["feature"] == GLOBAL_FEATURE]
    lines = ["", "=" * 80,
             f"BASELINE: {baseline.metadata['n_rounds']} rounds, subsets of {baseline.metadata['subset_size']} "
             f"vs remaining {baseline.metadata['n_source'] - baseline.metadata['subset_size']} rows",
             "Natural variability of global statistics (median / 95th percentile):", "-" * 80]
    for row in global_rows.itertuples():
        lines.append(f"  {row.task + '.' + row.metric:45s} {row.p50:10.4f} / {row.p95:10.4f}")
    per_feature = summary[(summary["metric"] == "psi") & (summary["task"].isin(["continuous", "categorical"]))]
    if not per_feature.empty:
        lines.append(f"  Feature Population Stability Index (PSI) 95th percentile: median {per_feature['p95'].median():.4f}, "
                     f"max {per_feature['p95'].max():.4f} ({per_feature.loc[per_feature['p95'].idxmax(), 'feature']})")
    lines.append("=" * 80)
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Command line interface
# --------------------------------------------------------------------------------------
def _split_list(value: str | None) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()] if value else []


def _parse_sentinels(value: str | None) -> list[Any]:
    sentinels = []
    for item in _split_list(value):
        try:
            sentinels.append(float(item))
        except ValueError:
            sentinels.append(item)
    return sentinels


def resolve_features(args: argparse.Namespace, source: pd.DataFrame, target: pd.DataFrame | None) -> list[str]:
    """Feature list from --features / --features-file, else all shared columns minus label and excludes."""
    if args.features_file:
        lines = Path(args.features_file).read_text().splitlines()
        features = [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]
    elif args.features:
        features = _split_list(args.features)
    else:
        excluded = set(_split_list(args.exclude)) | ({args.label_col} if args.label_col else set())
        shared = [col for col in source.columns if target is None or col in target.columns]
        features = [col for col in shared if col not in excluded]
        logger.info("No feature list given; using %d shared columns", len(features))
    return features


def build_parser() -> argparse.ArgumentParser:
    """Argument parser with ``baseline`` and ``compare`` sub-commands."""
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--train", required=True,
                        help="Labelled table (csv/parquet/feather). Used as given. Pass --test for a separate evaluation set.")
    common.add_argument("--features", help="Comma-separated feature columns.")
    common.add_argument("--features-file", help="Text file with one feature name per line.")
    common.add_argument("--exclude", help="Columns to exclude when no feature list is given (IDs, dates).")
    common.add_argument("--categorical", help="Comma-separated categorical feature columns.")
    common.add_argument("--label-col", help="Label column in the training population (optional).")
    common.add_argument(
        "--model",
        help="Pickled LightGBM estimator (.pkl/.pickle/.joblib) or Booster .txt. "
             "If the file cannot be loaded, feature-distribution tests still run.",
    )
    common.add_argument("--problem-type", choices=["binary", "regression"], default="binary")
    common.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for binary models.")
    common.add_argument("--missing-sentinels", help="Comma-separated codes to treat as missing, e.g. -999.")
    common.add_argument("--n-rounds", type=int, default=30, help="Number of baseline rounds.")
    common.add_argument("--seed", type=int, default=42)
    common.add_argument("--n-jobs", type=int, default=-1)
    common.add_argument("--report", help="Write the HTML report to this path after the run.")
    common.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    parser = argparse.ArgumentParser(
        prog="python -m population_drift",
        description="Population drift analysis for model transferability.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    baseline = sub.add_parser("baseline", parents=[common], help="Build the source-vs-source baseline.")
    baseline.add_argument("--out", required=True, help="Output directory for the baseline.")
    baseline.add_argument("--subset-size", type=int, help="Rows per pseudo-new subset (set to expected new size).")
    baseline.add_argument("--subset-fraction", type=float, default=0.3, help="Used when --subset-size is absent.")
    compare = sub.add_parser("compare", parents=[common],
                             help="Compare the new population with the fit set and the evaluation set.")
    compare.add_argument("--new", required=True, help="New unlabelled population (csv/parquet/feather).")
    compare.add_argument("--out", required=True, help="Output directory.")
    compare.add_argument("--test", help="Evaluation set the model was scored on. If omitted, every test uses --train.")
    compare.add_argument("--baseline", help="Baseline directory for a single labelled set. Not used with --test.")
    compare.add_argument("--n-bootstrap", type=int, default=200)
    compare.add_argument("--n-permutations", type=int, default=100)
    compare.add_argument("--n-classifier-permutations", type=int, default=20)
    compare.add_argument("--performance-tolerance", type=float, default=0.02,
                         help="Acceptable AUC drop (binary) or relative error increase (regression).")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sentinels = _parse_sentinels(args.missing_sentinels)
    source = load_table(args.train, sentinels)
    target = load_table(args.new, sentinels) if args.command == "compare" else None
    features = resolve_features(args, source, target)
    categorical = _split_list(args.categorical) or None
    model, model_error = load_model_or_skip(args.model)
    config_kwargs = {"problem_type": args.problem_type, "decision_threshold": args.threshold, "seed": args.seed,
                     "n_jobs": args.n_jobs}
    if args.command == "compare":
        config_kwargs.update({"n_bootstrap": args.n_bootstrap, "n_permutations": args.n_permutations,
                              "n_classifier_permutations": args.n_classifier_permutations,
                              "performance_tolerance": args.performance_tolerance})
    config = DriftConfig(**config_kwargs)

    if args.command == "baseline":
        baseline = build_baseline(source, features, categorical, model, args.label_col, args.n_rounds,
                                  args.subset_size, args.subset_fraction, config, args.out)
        print(format_baseline_summary(baseline))
        if args.report:
            from population_drift.report import build_report
            build_report(args.report, baseline_dir=args.out)
    else:
        return _run_compare_command(args, source, target, features, categorical, model, config, sentinels,
                                   model_error)
    return 0


def _run_compare_command(args: argparse.Namespace, source: pd.DataFrame, target: pd.DataFrame, features: list[str],
                         categorical: list[str] | None, model: Any | None, config: DriftConfig,
                         sentinels: list[Any], model_error: str | None = None) -> int:
    """Compare the new population with the labelled set, or with the fit and evaluation sets."""
    run_warnings = [model_error] if model_error else []
    if args.test:
        if args.baseline:
            logger.error("A supplied evaluation set builds one baseline per reference. Omit --baseline.")
            return 2
        evaluation = load_table(args.test, sentinels)
        run_paired_comparison(source, evaluation, target, features, categorical, model, args.label_col, config,
                              args.out, args.n_rounds, extra_warnings=run_warnings)
    else:
        run_comparison(source, target, features, categorical, model, args.label_col, config, args.baseline,
                       args.out, args.n_rounds, extra_warnings=run_warnings)
    if args.report:
        from population_drift.report import build_report
        build_report(args.report, comparison_dir=args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
