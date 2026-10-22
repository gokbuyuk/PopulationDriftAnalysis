"""Unit tests for the short decision report written beside the detailed report."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from population_drift.report import (
    ComparisonRun,
    build_report,
    section_short_decision,
    section_short_signals,
    short_report_path,
)


def _check(actual: object, expected: object, what: str) -> None:
    """Fail with the comparison that did not hold."""
    if actual != expected:
        pytest.fail(f"{what}: got {actual!r}, expected {expected!r}.")


def _comparison_fixture(directory: Path, *, level: str = "severe") -> Path:
    """Write the minimum comparison files ``build_report`` needs.

    Args:
        directory: Output directory for the fake comparison.
        level: Verdict level stored on ``decision.json``.

    Returns:
        The comparison directory.
    """
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "tables").mkdir()
    features = pd.DataFrame([{
        "feature": "x1", "type": "continuous", "severity": "severe", "important": True,
        "model_importance_share": 0.9, "psi": 0.40, "ks_d": 0.25, "smd": 0.6,
        "cramers_v": float("nan"), "reasons": "PSI 0.40",
    }])
    features.to_csv(directory / "feature_assessment.csv", index=False)
    decision = {
        "verdict": "Different: do not use the model as is",
        "verdict_level": level,
        "verdict_rule": "Any counted signal is severe.",
        "signals": [{
            "level": "severe",
            "signal": "Feature drift: model-important features",
            "pair": "labelled set vs new",
            "value": "1 severe",
            "detail": "x1 (severe)",
            "important": True,
        }],
        "warnings": [],
        "baseline_rounds": 10,
        "baseline_subset_size": 100,
        "n_severe": 1,
        "n_moderate": 0,
    }
    (directory / "decision.json").write_text(json.dumps(decision), encoding="utf-8")
    metadata = {
        "n_source": 500,
        "n_target": 100,
        "feature_types": {"x1": "continuous"},
        "has_model": True,
        "label_col": "target",
        "problem_type": "binary",
        "created_at": "2026-01-01T00:00:00Z",
        "config": {
            "alpha": 0.05,
            "performance_tolerance": 0.02,
            "equivalence_margin_smd": 0.1,
            "thresholds": {
                "psi_moderate": 0.1, "psi_major": 0.25, "ks_moderate": 0.1, "ks_major": 0.2,
                "smd_small": 0.1, "smd_large": 0.5, "cramers_v_moderate": 0.1, "cramers_v_major": 0.3,
                "unseen_level_share": 0.01, "cohens_h_small": 0.2, "cohens_h_large": 0.5,
                "domain_auc_mild": 0.55, "domain_auc_strong": 0.8, "correlation_pair_delta": 0.2,
                "ood_share_caution": 0.02, "ood_share_material": 0.05, "importance_rank_rho": 0.8,
                "important_feature_cumulative_share": 0.8, "ess_ratio_poor": 0.1, "ess_ratio_good": 0.5,
                "baseline_percentile": 95,
            },
            "cv_folds": 5,
            "n_bootstrap": 200,
            "n_permutations": 200,
            "n_classifier_permutations": 20,
            "correction_method": "bh",
            "range_quantiles": [0.005, 0.995],
            "ood_quantile": 0.95,
        },
        "task_pairs": {"continuous": "labelled set vs new"},
    }
    (directory / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return directory


def test_short_report_path_inserts_short_before_suffix() -> None:
    """The short sibling keeps the stem and adds ``_short``."""
    _check(short_report_path("runs/report.html"), Path("runs/report_short.html"), "html sibling")
    _check(short_report_path(Path("out/drift.md")), Path("out/drift_short.md"), "markdown sibling")


def test_short_report_path_rejects_blank() -> None:
    """A blank destination is invalid."""
    with pytest.raises(ValueError, match="output_path"):
        short_report_path("   ")


def test_section_short_decision_states_action_and_limits() -> None:
    """The decision section carries the verdict, action, and concept-shift limit."""
    run = ComparisonRun(
        Path("."),
        {},
        {},
        {
            "n_source": 10, "n_target": 5, "feature_types": {"a": "continuous"},
            "has_model": False, "label_col": None,
        },
        {
            "verdict": "Different: do not use the model as is",
            "verdict_level": "severe",
            "verdict_rule": "Any counted signal is severe.",
            "signals": [],
            "warnings": ["demo warning"],
            "baseline_rounds": 30,
            "baseline_subset_size": 5,
        },
        pd.DataFrame(),
        pd.DataFrame(),
    )
    section = section_short_decision(run, "report.html")
    _check(section.anchor, "decision", "anchor")
    _check("Do not score the new population" in section.body, True, "severe action")
    _check("<h3>Limits</h3>" in section.body, True, "limits heading")
    _check("concept shift" in section.body, True, "concept-shift caveat")
    _check("report.html" in section.body, True, "pointer to detailed report")
    _check("demo warning" in section.body, True, "run warning surfaced")


def test_section_short_signals_omits_long_detail_column() -> None:
    """The short signals table keeps observed and baseline, not the long detail cell."""
    run = ComparisonRun(
        Path("."),
        {},
        {},
        {"n_source": 1, "n_target": 1, "feature_types": {}, "has_model": False, "label_col": None},
        {
            "verdict": "Use with caution",
            "verdict_level": "moderate",
            "verdict_rule": "Moderate signals without a severe signal.",
            "signals": [{
                "level": "moderate",
                "signal": "Score distribution",
                "pair": "evaluation set vs new",
                "value": "PSI 0.15",
                "detail": "long explanation that belongs only in the detailed report",
            }],
            "warnings": [],
            "baseline_rounds": 10,
            "baseline_subset_size": 1,
        },
        pd.DataFrame(),
        pd.DataFrame(),
    )
    body = section_short_signals(run).body
    _check("Score distribution" in body, True, "signal name")
    _check("PSI 0.15" in body, True, "observed value")
    _check("long explanation that belongs only in the detailed report" in body, False, "detail omitted")


def test_build_report_writes_detailed_and_short_outputs(tmp_path: Path) -> None:
    """A comparison destination also emits the short HTML and markdown decision brief."""
    comparison = _comparison_fixture(tmp_path / "comparison")
    out = tmp_path / "report.html"
    written = build_report(out, comparison_dir=comparison, top_k=2)
    short_html = short_report_path(out)
    short_md = short_html.with_suffix(".md")
    _check(written, out.resolve(), "detailed path returned")
    _check(out.is_file(), True, "detailed HTML")
    _check(out.with_suffix(".md").is_file(), True, "detailed markdown")
    _check(short_html.is_file(), True, "short HTML")
    _check(short_md.is_file(), True, "short markdown")
    short_text = short_md.read_text(encoding="utf-8")
    detailed_text = out.with_suffix(".md").read_text(encoding="utf-8")
    _check("# Population drift: decision brief" in short_text, True, "short title")
    _check("## Decision" in short_text, True, "decision heading")
    _check("## Signals that drive the verdict" in short_text, True, "signals heading")
    _check("Do not score the new population" in short_text, True, "action in short report")
    _check("Methods and thresholds" in short_text, False, "methods stay out of short report")
    _check("Methods and thresholds" in detailed_text, True, "detailed report still has methods")
