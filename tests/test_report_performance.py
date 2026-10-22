"""Unit tests for label-free reference vs estimate copy in the drift report."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from population_drift.report import (
    ComparisonRun,
    section_performance,
    section_short_model_evidence,
)


def _check(actual: object, expected: object, what: str) -> None:
    """Fail with the comparison that did not hold."""
    if actual != expected:
        pytest.fail(f"{what}: got {actual!r}, expected {expected!r}.")


def _run(tables: dict[str, pd.DataFrame] | None = None) -> ComparisonRun:
    """Minimal comparison with optional performance tables.

    Args:
        tables: Task tables keyed as ``{task}__{name}``.

    Returns:
        A comparison run the performance sections can render.
    """
    return ComparisonRun(
        Path("."),
        tables or {},
        {},
        {
            "n_source": 100,
            "n_target": 40,
            "feature_types": {"x1": "continuous"},
            "has_model": True,
            "label_col": "target",
            "problem_type": "binary",
            "config": {"performance_tolerance": 0.02},
            "failed_tasks": {},
            "task_pairs": {
                "weighted_performance": "evaluation set vs new",
                "cbpe": "evaluation set vs new",
            },
        },
        {
            "verdict": "Use with caution",
            "verdict_level": "moderate",
            "verdict_rule": "Moderate signals without a severe signal.",
            "signals": [],
            "warnings": [],
        },
        pd.DataFrame(),
        pd.DataFrame(),
    )


def _metric_table(reference: float, estimate: float) -> pd.DataFrame:
    """One AUC row in the shape ``plot_performance`` reads.

    Args:
        reference: Stored reference metric.
        estimate: Stored label-free estimate.

    Returns:
        A one-row performance table.
    """
    return pd.DataFrame([{
        "metric": "auc",
        "higher_is_better": True,
        "reference_value": reference,
        "estimate": estimate,
        "ci_low": estimate - 0.02,
        "ci_high": estimate + 0.02,
        "degradation": reference - estimate,
    }])


def _performance_run() -> ComparisonRun:
    """Comparison that has both label-free estimators."""
    return _run({
        "weighted_performance__main": _metric_table(0.80, 0.74),
        "weighted_performance__summary": pd.DataFrame([{"feature": "__global__", "ess_ratio": 0.62}]),
        "cbpe__main": _metric_table(0.79, 0.73),
        "calibration__main": pd.DataFrame([{"feature": "__global__", "ece": 0.03}]),
    })


def test_detailed_performance_section_explains_how_points_are_found() -> None:
    """The detailed report says how the diamond and circle are computed without new labels."""
    body = section_performance(_performance_run()).body
    _check("How reference and estimate are found" in body, True, "callout heading")
    _check("labels on the new population" in body, True, "no new labels")
    _check("unweighted metric on labelled" in body, True, "weighting reference")
    _check("density ratio" in body, True, "weighting uses domain-classifier weights")
    _check("expected metric" in body, True, "CBPE is an expectation over scores")
    _check("new predicted probabilities" in body, True, "CBPE estimate uses new scores")
    _check("If the methods disagree" in body, True, "disagreement rule")


def test_short_model_section_explains_how_points_are_found() -> None:
    """The data-scientist brief repeats the same construction, next to the plot."""
    body = section_short_model_evidence(_performance_run(), None).body
    _check("How reference and estimate are found" in body, True, "callout heading")
    _check("labels on the new population" in body, True, "no new labels")
    _check("unweighted metric on labelled" in body, True, "weighting reference")
    _check("new predicted probabilities" in body, True, "CBPE estimate uses new scores")


def test_reference_estimate_copy_omits_methods_that_did_not_run() -> None:
    """Only the estimators that produced a table are explained."""
    weighting_only = _run({"weighted_performance__main": _metric_table(0.80, 0.74)})
    body = section_performance(weighting_only).body
    _check("How reference and estimate are found" in body, True, "callout still present")
    _check("unweighted metric on labelled" in body, True, "weighting explained")
    _check("new predicted probabilities" in body, False, "CBPE omitted")
    _check("If the methods disagree" in body, False, "one method is not a disagreement")
