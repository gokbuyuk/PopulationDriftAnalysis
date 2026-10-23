"""Tests for the performance-expectation headline and its report."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pytest

from population_drift.analysis import DriftConfig
from population_drift.expectation_report import (
    expectation_report_path,
    interval_position,
    judge_expectation,
    labelled_sample_size,
    wilson_interval,
)
from population_drift.report import ComparisonRun, build_report


def _check(actual: object, expected: object, what: str) -> None:
    """Fail with the comparison that did not hold.

    Args:
        actual: Observed value.
        expected: Required value.
        what: Label for the failure.
    """
    if actual != expected:
        pytest.fail(f"{what}: got {actual!r}, expected {expected!r}.")


def _metric(reference: float, estimate: float, low: float, high: float, metric: str = "auc") -> pd.DataFrame:
    """One performance row.

    Args:
        reference: Reference metric.
        estimate: New-population estimate.
        low: Interval lower end.
        high: Interval upper end.
        metric: Metric name.

    Returns:
        A one-row table.
    """
    return pd.DataFrame([{
        "metric": metric,
        "higher_is_better": metric != "rmse",
        "reference_value": reference,
        "estimate": estimate,
        "ci_low": low,
        "ci_high": high,
        "source_realized": reference,
    }])


def _run(tables: dict[str, pd.DataFrame] | None = None, *, mode: str = "paired",
         ess: float = 0.80, ece: float = 0.02, rate_tolerance: float | None = None,
         positive: tuple[float, float] = (0.40, 0.42), n_target: int = 2000,
         features: pd.DataFrame | None = None, problem: str = "binary") -> ComparisonRun:
    """Comparison with the tables the headline reads.

    Args:
        tables: Extra task tables. Performance tables are added when absent.
        mode: ``paired`` or ``single``.
        ess: Kish ESS ratio.
        ece: Reference expected calibration error.
        rate_tolerance: Operating-rate band, or None.
        positive: Reference and new positive rates.
        n_target: New-row count.
        features: Feature assessment table.
        problem: ``binary`` or ``regression``.

    Returns:
        A comparison run.
    """
    stored = dict(tables or {})
    metric = "rmse" if problem == "regression" else "auc"
    stored.setdefault("weighted_performance__main", _metric(0.80, 0.795, 0.785, 0.805, metric))
    stored.setdefault("weighted_performance__summary", pd.DataFrame([{
        "ess_ratio": ess, "ess": ess * 100, "n_rows": 100,
    }]))
    stored.setdefault("cbpe__main", _metric(0.80, 0.795, 0.785, 0.805))
    stored.setdefault("calibration__main", pd.DataFrame([{"ece": ece, "calibration_slope": 1.0}]))
    stored.setdefault("prediction_drift__main", pd.DataFrame([{
        "positive_rate_source": positive[0], "positive_rate_target": positive[1],
        "near_threshold_share_source": 0.10, "near_threshold_share_target": 0.11,
        "psi": 0.04, "ks_d": 0.03, "threshold": 0.35,
        "positive_rate_cohens_h_abs": 0.04,
    }]))
    stored.setdefault("label_shift__main", pd.DataFrame([{
        "source_prevalence": 0.30, "target_prevalence_estimate": 0.31,
        "ci_low": 0.28, "ci_high": 0.34,
    }]))
    return ComparisonRun(
        Path("."), stored, {},
        {
            "n_source": 4000, "n_target": n_target, "n_evaluation": 1000,
            "feature_types": {"f01": "continuous"}, "has_model": True, "label_col": "target",
            "reference_mode": mode, "created_at": "2026-10-22T00:00:00Z",
            "config": {
                "problem_type": problem, "performance_tolerance": 0.02,
                "rate_tolerance": rate_tolerance,
                "thresholds": {"ess_ratio_poor": 0.10, "ess_ratio_good": 0.50, "ece_max": 0.05},
            },
            "failed_tasks": {}, "task_pairs": {"prediction_drift": "evaluation set vs new"},
        },
        {"verdict_level": "mild", "warnings": [], "baseline_rounds": 30},
        features if features is not None else pd.DataFrame(),
        pd.DataFrame(),
    )


def test_interval_position_names_the_three_locations() -> None:
    """A bound of 0.78 separates an acceptable interval, a worse interval, and a crossing one."""
    _check(interval_position(0.79, 0.785, 0.80, 0.78, True), "inside", "auc inside")
    _check(interval_position(0.70, 0.68, 0.72, 0.78, True), "outside", "auc outside")
    _check(interval_position(0.78, 0.77, 0.79, 0.78, True), "crosses", "auc crosses")
    _check(interval_position(1.02, 1.00, 1.03, 1.04, False), "inside", "rmse inside")


def test_within_when_both_estimates_clear_the_bound() -> None:
    """Paired mode with both intervals above reference − 0.02 is within expectation."""
    judged = judge_expectation(_run())
    _check(judged.headline, "within", "headline")
    _check(any(item.label == "Expected precision at the threshold" for item in judged.metrics), False, "no precision row")


def test_disagreement_is_not_identified() -> None:
    """One estimate inside the band and one entirely worse is not averaged."""
    run = _run(tables={
        "weighted_performance__main": _metric(0.80, 0.70, 0.68, 0.72),
        "cbpe__main": _metric(0.80, 0.795, 0.785, 0.805),
    })
    _check(judge_expectation(run).headline, "not_identified", "headline")


def test_thin_overlap_and_poor_calibration_drop_both_estimates() -> None:
    """ESS below 0.10 and ECE above 0.05 leave no estimate the headline can use."""
    run = _run(ess=0.05, ece=0.09)
    judged = judge_expectation(run)
    _check(judged.headline, "not_identified", "headline")
    weighted = next(item for item in judged.metrics if item.label == "Weighted AUC")
    _check(weighted.usable, False, "weighted dropped")
    _check("overlap too thin" in weighted.note, True, "ess note")


def test_moderate_ess_that_barely_clears_is_imprecise() -> None:
    """An interval that clears the bound by less than half the tolerance stays imprecise below ESS 0.50."""
    run = _run(
        ess=0.20, ece=0.09,
        tables={"weighted_performance__main": _metric(0.80, 0.790, 0.781, 0.800)},
    )
    _check(judge_expectation(run).headline, "imprecise", "headline")


def test_in_sample_reference_cannot_be_within() -> None:
    """The same acceptable intervals on the fitting rows are imprecise."""
    _check(judge_expectation(_run(mode="single")).headline, "imprecise", "headline")


def test_rate_outside_its_band_sets_the_headline() -> None:
    """A pre-set rate tolerance can make the headline outside while AUC stays acceptable."""
    run = _run(rate_tolerance=0.02, positive=(0.40, 0.70), n_target=5000)
    judged = judge_expectation(run)
    _check(judged.headline, "outside", "headline")
    _check(judged.performance_headline, "within", "auc still inside")


def test_unset_rate_tolerance_does_not_change_a_within_headline() -> None:
    """Without a rate band, a large operating-rate change stays off the headline."""
    run = _run(positive=(0.40, 0.70), n_target=5000)
    _check(judge_expectation(run).headline, "within", "headline")


def test_cause_rows_keep_only_important_features_that_moved() -> None:
    """Mild and low-importance features do not appear as causes."""
    features = pd.DataFrame([
        {"feature": "f01", "type": "continuous", "severity": "severe", "important": True,
         "model_importance_share": 0.70, "psi": 0.40, "smd": 0.55},
        {"feature": "f02", "type": "continuous", "severity": "mild", "important": True,
         "model_importance_share": 0.20, "psi": 0.01, "smd": 0.01},
        {"feature": "f03", "type": "continuous", "severity": "moderate", "important": False,
         "model_importance_share": 0.01, "psi": 0.30, "smd": 0.20},
    ])
    judged = judge_expectation(_run(features=features))
    _check([item.feature for item in judged.causes], ["f01"], "causes")
    _check(judged.low_importance, 1, "low importance count")
    _check("higher" in judged.causes[0].direction, True, "direction")


def test_labelled_sample_size_shrinks_as_the_tolerance_grows() -> None:
    """A wider AUC interval needs fewer labelled rows, and the search is finite."""
    tight = labelled_sample_size(0.80, 0.30, 0.02)
    wide = labelled_sample_size(0.80, 0.30, 0.05)
    _check(tight is not None and wide is not None and tight > wide, True, "ordering")
    _check(labelled_sample_size(0.80, 0.01, 0.001), None, "unresolvable")


def test_wilson_interval_covers_the_known_zero_success_case() -> None:
    """Zero successes in ten trials has a Wilson upper bound near 0.28."""
    low, high = wilson_interval(0, 10)
    _check(math.isclose(low, 0.0, abs_tol=1e-12), True, "lower")
    _check(abs(high - 0.2775) < 0.01, True, "upper")


def test_negative_rate_tolerance_is_rejected() -> None:
    """A negative operating-rate band is not a tolerance."""
    with pytest.raises(ValueError, match="rate_tolerance"):
        DriftConfig(rate_tolerance=-0.01)


def test_expectation_report_is_written_beside_the_detailed_report(tmp_path: Path) -> None:
    """A comparison write produces the expectation page, and an empty run is not identified."""
    comparison = tmp_path / "comparison"
    comparison.mkdir()
    (comparison / "tables").mkdir()
    pd.DataFrame([{
        "feature": "x1", "type": "continuous", "severity": "mild", "important": True,
        "model_importance_share": 1.0, "psi": 0.01, "ks_d": 0.01, "smd": 0.01,
        "cramers_v": float("nan"), "reasons": "within baseline",
    }]).to_csv(comparison / "feature_assessment.csv", index=False)
    (comparison / "decision.json").write_text(json.dumps({
        "verdict": "Similar", "verdict_level": "mild", "verdict_rule": "rule",
        "signals": [], "warnings": [], "baseline_rounds": 30, "baseline_subset_size": 100,
    }), encoding="utf-8")
    (comparison / "metadata.json").write_text(json.dumps({
        "n_source": 500, "n_target": 100, "feature_types": {"x1": "continuous"},
        "has_model": True, "label_col": "target", "problem_type": "binary",
        "created_at": "2026-01-01T00:00:00Z", "reference_mode": "paired",
        "config": {
            "problem_type": "binary", "performance_tolerance": 0.02, "alpha": 0.05,
            "equivalence_margin_smd": 0.1, "cv_folds": 5, "n_bootstrap": 200,
            "n_permutations": 100, "n_classifier_permutations": 20, "correction_method": "fdr_bh",
            "range_quantiles": [0.005, 0.995], "ood_quantile": 0.01,
            "thresholds": {
                "psi_moderate": 0.1, "psi_major": 0.25, "ks_moderate": 0.1, "ks_major": 0.2,
                "smd_small": 0.1, "smd_large": 0.5, "cramers_v_moderate": 0.1, "cramers_v_major": 0.3,
                "unseen_level_share": 0.01, "cohens_h_small": 0.2, "cohens_h_large": 0.5,
                "domain_auc_mild": 0.55, "domain_auc_strong": 0.8, "correlation_pair_delta": 0.2,
                "ood_share_caution": 0.02, "ood_share_material": 0.05, "importance_rank_rho": 0.8,
                "important_feature_cumulative_share": 0.8, "ess_ratio_poor": 0.1, "ess_ratio_good": 0.5,
                "baseline_percentile": 95, "ece_max": 0.05,
            },
        },
        "failed_tasks": {}, "task_pairs": {},
    }), encoding="utf-8")
    out = tmp_path / "report.html"
    build_report(out, comparison_dir=comparison, top_k=1)
    path = expectation_report_path(out)
    text = path.read_text(encoding="utf-8")
    _check(path.is_file(), True, "html exists")
    _check(path.with_suffix(".md").is_file(), True, "markdown exists")
    _check("Not identified" in text, True, "headline")
    _check("concept" in text.lower() or "P(y|x)" in text, True, "identifiability")
    _check("report.html" in text, True, "link to the detailed report")
