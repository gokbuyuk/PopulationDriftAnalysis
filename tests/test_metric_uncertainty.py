"""Tests for Wilks intervals on confusion-matrix metrics (Urlus et al. 2023)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2

from population_drift.analysis import (
    DriftAnalysisResult,
    DriftConfig,
    TaskResult,
    assess_calibration,
    compare_prediction_distributions,
    estimate_performance_cbpe,
    interpret_results,
)
from population_drift.expectation_report import _validity_diagnostics
from population_drift.metric_uncertainty import BinomialProportion, ConfusionMatrix
from population_drift.report import ComparisonRun, section_performance, signal_baseline


def _check(actual: object, expected: object, what: str) -> None:
    """Fail with the comparison that did not hold.

    Args:
        actual: Observed value.
        expected: Required value.
        what: Label for the failure.
    """
    if actual != expected:
        pytest.fail(f"{what}: got {actual!r}, expected {expected!r}.")


def _matrix() -> ConfusionMatrix:
    """Counts whose precision-recall point is interior."""
    return ConfusionMatrix(50, 10, 30, 20)


def _interval(matrix: ConfusionMatrix, metric: str):
    """One stored interval."""
    return next(item for item in matrix.intervals() if item.metric == metric)


def test_profile_at_the_mle_is_zero_and_matches_equation_6() -> None:
    """The unconstrained fit is the observed precision and recall."""
    matrix = _matrix()
    statistic = matrix.precision_recall_statistic(matrix.precision(), matrix.recall())
    _check(statistic, 0.0, "profile statistic at the MLE")
    precision, recall = matrix.precision(), matrix.recall()
    gamma = 1.0 / precision + 1.0 / recall - 1.0
    p_tp = (50 + 10 + 20) / (gamma * 110)
    assert p_tp == pytest.approx(50 / 110)


def test_normal_moments_match_the_paper() -> None:
    """Equations 13–15, and the np(1-p) >= 10 validity rule."""
    matrix = _matrix()
    precision = _interval(matrix, "precision")
    recall = _interval(matrix, "recall")
    assert precision.normal_se == pytest.approx(math.sqrt(50 * 10 / 60**3))
    assert recall.normal_se == pytest.approx(math.sqrt(50 * 20 / 70**3))
    _check(precision.normal_valid, False, "precision normal validity at np(1-p) = 8.3")
    _check(recall.normal_valid, True, "recall normal validity at np(1-p) = 14.3")


def test_recall_and_fpr_match_the_class_binomial_intervals() -> None:
    """Appendix G: recall and false-positive rate factor into binomial profiles."""
    matrix = _matrix()
    recall = BinomialProportion(50, 70).interval()
    false_positive = BinomialProportion(10, 40).interval()
    assert _interval(matrix, "recall").ci_low == pytest.approx(recall[0])
    assert _interval(matrix, "recall").ci_high == pytest.approx(recall[1])
    assert _interval(matrix, "fpr").ci_low == pytest.approx(false_positive[0])
    assert _interval(matrix, "fpr").ci_high == pytest.approx(false_positive[1])


def test_precision_bounds_sit_on_the_chi_square_critical_value() -> None:
    """Each finite interior endpoint has profile statistic equal to chi-square 0.95, df 1."""
    matrix = _matrix()
    precision = _interval(matrix, "precision")
    critical = float(chi2.ppf(0.95, 1))
    assert matrix._precision_statistic(precision.ci_low) == pytest.approx(critical, rel=1e-4)
    assert matrix._precision_statistic(precision.ci_high) == pytest.approx(critical, rel=1e-4)
    assert precision.ci_low < precision.estimate < precision.ci_high


def test_large_sample_wilks_width_matches_the_normal_approximation() -> None:
    """Where the normal approximation is valid, the two 95% widths agree."""
    matrix = ConfusionMatrix(5000, 1000, 3000, 2000)
    for metric in ("precision", "recall", "fpr", "f1"):
        item = _interval(matrix, metric)
        half_width = 0.5 * (item.ci_high - item.ci_low)
        assert half_width == pytest.approx(1.96 * item.normal_se, rel=0.01)
        _check(item.normal_valid, True, f"{metric} normal validity")


def test_boundary_cells_keep_the_estimate_inside_the_unit_interval() -> None:
    """A zero cell puts the estimate on the boundary and the interval stays in [0, 1]."""
    no_false_positive = {item.metric: item for item in ConfusionMatrix(20, 0, 30, 5).intervals()}
    _check(no_false_positive["precision"].estimate, 1.0, "precision with no false positives")
    _check(no_false_positive["precision"].ci_high, 1.0, "upper precision bound")
    assert no_false_positive["precision"].ci_low < 1.0
    no_true_positive = {item.metric: item for item in ConfusionMatrix(0, 4, 30, 6).intervals()}
    _check(no_true_positive["precision"].ci_low, 0.0, "lower precision bound")
    _check(no_true_positive["f1"].estimate, 0.0, "F1 with no true positives")
    undefined = _interval(ConfusionMatrix(0, 0, 10, 4), "precision")
    _check(undefined.method, "undefined", "precision with no predicted positives")
    assert math.isnan(undefined.estimate)


def test_joint_wilks_region_covers_the_true_precision_and_recall() -> None:
    """About 95% of multinomial samples keep the true point inside the chi-square(2) region.

    Seed 0, 300 samples, n = 400, probabilities (0.30, 0.10, 0.40, 0.20). The binomial
    standard error of a 0.95 rate on 300 trials is about 0.013, so the acceptance band
    is wider than a 3-standard-error interval.
    """
    rng = np.random.default_rng(0)
    probabilities = np.array([0.30, 0.10, 0.40, 0.20])
    true_precision, true_recall = 0.30 / 0.40, 0.30 / 0.50
    critical = float(chi2.ppf(0.95, 2))
    covered = 0
    for _ in range(300):
        counts = [int(count) for count in rng.multinomial(400, probabilities)]
        score = ConfusionMatrix(*counts).precision_recall_statistic(true_precision, true_recall)
        covered += score <= critical
    assert 0.90 <= covered / 300 <= 0.99


def test_inputs_are_rejected() -> None:
    """Public constructors and the confidence level fail before any interval is built."""
    with pytest.raises(ValueError, match="binary"):
        ConfusionMatrix.from_predictions([0, 1, 2], [0.1, 0.2, 0.3], 0.5)
    with pytest.raises(ValueError, match="same length"):
        ConfusionMatrix.from_predictions([0, 1], [0.1], 0.5)
    with pytest.raises(ValueError, match="positive"):
        BinomialProportion(1, 0)
    with pytest.raises(TypeError, match="integer"):
        ConfusionMatrix(True, 0, 0, 1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="confidence"):
        _matrix().intervals(confidence=1.0)


def test_from_predictions_uses_the_threshold() -> None:
    """A score equal to the threshold counts as a positive prediction."""
    matrix = ConfusionMatrix.from_predictions([1, 0, 1, 0], [0.9, 0.5, 0.2, 0.1], 0.5)
    _check((matrix.true_positive, matrix.false_positive, matrix.true_negative, matrix.false_negative),
           (1, 1, 1, 1), "confusion counts")


def test_calibration_and_positive_rate_store_wilks_intervals() -> None:
    """The labelled operating point and both predicted-positive rates carry intervals."""
    rng = np.random.default_rng(1)
    scores = rng.uniform(0.0, 1.0, 80)
    labels = rng.binomial(1, 0.4, size=80)
    calibration = assess_calibration(scores, labels, DriftConfig(decision_threshold=0.5))
    table = calibration.tables["operating_point"].set_index("metric")
    _check(set(table.index), {"precision", "recall", "fpr", "f1"}, "operating-point metrics")
    assert table.loc["precision", "ci_low"] <= table.loc["precision", "estimate"] <= table.loc["precision", "ci_high"]
    prediction = compare_prediction_distributions(scores, rng.uniform(0.0, 1.0, 40), DriftConfig())
    row = prediction.tables["main"].iloc[0]
    assert row["positive_rate_source_ci_low"] <= row["positive_rate_source"] <= row["positive_rate_source_ci_high"]
    assert row["positive_rate_target_ci_low"] <= row["positive_rate_target"] <= row["positive_rate_target_ci_high"]


def test_cbpe_realized_precision_uses_the_wilks_interval() -> None:
    """The labelled precision interval is separate from the bootstrap interval on the new scores."""
    scores = np.array([0.2, 0.8, 0.7, 0.1, 0.9, 0.4])
    labels = np.array([0, 1, 1, 0, 1, 0])
    result = estimate_performance_cbpe(scores, labels, np.array([0.3, 0.6, 0.8]),
                                      DriftConfig(n_bootstrap=0, decision_threshold=0.5))
    table = result.tables["main"].set_index("metric")
    assert table.loc["precision", "source_realized_ci_low"] <= table.loc["precision", "source_realized"]
    assert table.loc["precision", "source_realized"] <= table.loc["precision", "source_realized_ci_high"]
    assert math.isnan(table.loc["auc", "source_realized_ci_low"])


def test_operating_point_signal_is_info_and_does_not_change_the_verdict() -> None:
    """The new interval is reported on the calibration pair and is not counted."""
    frame = _matrix().interval_frame(0.5)
    continuous = pd.DataFrame([{
        "feature": "x", "psi": 0.01, "ks_d": 0.01, "smd": 0.0, "smd_abs": 0.0,
        "ks_p_adj": 1.0, "ad_p_adj": 1.0, "var_ratio": 1.0,
    }])
    calibration = {
        "main": pd.DataFrame([{"ece": 0.01, "calibration_slope": 1.0}]),
        "operating_point": frame,
    }
    result = DriftAnalysisResult(
        {"continuous": TaskResult("continuous", {"main": continuous}),
         "calibration": TaskResult("calibration", calibration)},
        {"has_model": False, "feature_cols": ["x"], "feature_types": {"x": "continuous"},
         "task_pairs": {"continuous": "fit set vs new", "calibration": "evaluation set"}},
    )
    baseline = pd.DataFrame([{"task": "continuous", "feature": "x", "metric": "psi", "exceeds_baseline": False}])
    decision = interpret_results(result, baseline, DriftConfig())
    info = [signal for signal in decision["signals"] if signal["level"] == "info"]
    _check(decision["verdict_level"], "mild", "verdict")
    _check(len(info), 2, "info signals")
    operating = next(signal for signal in info if signal["signal"].startswith("Operating-point"))
    calibration_signal = next(signal for signal in info if signal["signal"].startswith("Source calibration"))
    _check(operating["pair"], "evaluation set", "operating-point pair")
    _check(calibration_signal["pair"], "evaluation set", "calibration pair")
    assert "precision" in operating["value"]


def _comparison(tables: dict[str, pd.DataFrame]) -> ComparisonRun:
    """Minimal comparison run for report text."""
    return ComparisonRun(
        Path("."), tables, {},
        {"n_source": 80, "n_target": 40, "feature_types": {}, "has_model": True, "label_col": "target",
         "problem_type": "binary", "config": {"performance_tolerance": 0.02}, "failed_tasks": {}, "task_pairs": {}},
        {"verdict": "Similar", "verdict_level": "mild", "verdict_rule": "No severe signal.", "signals": [],
         "warnings": []},
        pd.DataFrame(), pd.DataFrame(),
    )


def test_reports_quote_the_wilks_interval_as_labelled_sampling_uncertainty() -> None:
    """The detailed section and the expectation validity line name the labelled interval."""
    frame = _matrix().interval_frame(0.5)
    run = _comparison({"calibration__operating_point": frame, "calibration__main": pd.DataFrame([{"ece": 0.02}])})
    body = section_performance(run).body
    _check("Operating-point sampling uncertainty" in body, True, "performance section")
    _check("not performance on the new population" in body, True, "not a new-population estimate")
    lines = _validity_diagnostics(run)
    _check(any("Labelled operating-point precision" in line for line in lines), True, "expectation validity line")
    value, note = signal_baseline({"signal": "Operating-point sampling uncertainty"}, run)
    _check(value, "Wilks 95% profile interval", "baseline label")
    _check("Not a percentile" in note, True, "baseline is not a percentile")
