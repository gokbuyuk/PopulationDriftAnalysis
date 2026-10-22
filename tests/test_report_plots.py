"""Unit tests for kernel-test and domain-classifier report plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from population_drift.report import (
    BaselineRun,
    ComparisonRun,
    _domain_classifier_figure,
    _draw_domain_drivers,
    _kernel_interpretation,
    _kernel_panel_reading,
    _kernel_results,
    _include_zero,
    _null_panel,
    _origin_span,
    _style_axes,
    null_histogram_bins,
    section_joint,
    section_short_model_evidence,
)


def _check(actual: object, expected: object, what: str) -> None:
    """Fail with the comparison that did not hold."""
    if actual != expected:
        pytest.fail(f"{what}: got {actual!r}, expected {expected!r}.")


def _domain_run() -> tuple[ComparisonRun, BaselineRun]:
    """Comparison and baseline with domain-classifier and kernel-test outputs."""
    bins = np.linspace(0.0, 1.0, 41)
    tables = {
        "domain_classifier__main": pd.DataFrame([{
            "feature": "__global__", "auc": 0.82, "auc_ci_low": 0.80, "auc_ci_high": 0.84,
            "permutation_p": 0.01,
        }]),
        "domain_classifier__importance": pd.DataFrame([
            {"feature": "x1", "mean_abs_shap": 0.6, "shap_share": 0.6},
            {"feature": "x2", "mean_abs_shap": 0.4, "shap_share": 0.4},
        ]),
        "mmd__main": pd.DataFrame([{
            "mmd2": 0.05, "permutation_p": 0.01, "n_source_rows": 100, "n_target_rows": 80,
        }]),
        "energy__main": pd.DataFrame([{
            "energy_distance": 0.02, "permutation_p": 0.40, "n_source_rows": 100, "n_target_rows": 80,
        }]),
    }
    run = ComparisonRun(
        Path("."),
        tables,
        {
            "domain_classifier": {
                "bins": bins.tolist(),
                "source_hist": np.ones(40).tolist(),
                "target_hist": np.linspace(0.0, 2.0, 40).tolist(),
            },
        },
        {
            "n_source": 100, "n_target": 80, "feature_types": {"x1": "continuous"},
            "has_model": True, "label_col": "y", "config": {"alpha": 0.05, "thresholds": {}},
        },
        {"signals": [], "warnings": [], "baseline_rounds": 30, "baseline_subset_size": 80},
        pd.DataFrame(),
        pd.DataFrame(),
    )
    baseline = BaselineRun(
        Path("."),
        pd.DataFrame([
            *[{"task": "domain_classifier", "metric": "auc", "feature": "__global__", "value": value}
              for value in np.linspace(0.48, 0.58, 30)],
            *[{"task": "mmd", "metric": "mmd2", "feature": "__global__", "value": value}
              for value in np.linspace(0.001, 0.01, 30)],
            *[{"task": "energy", "metric": "energy_distance", "feature": "__global__", "value": value}
              for value in np.linspace(0.01, 0.03, 30)],
        ]),
        pd.DataFrame(),
        {},
    )
    return run, baseline


def test_origin_span_caps_unit_positive_values_at_one() -> None:
    """AUC-like values use [0, 1], not a zoomed-in window around 0.9."""
    _check(_origin_span(0.72, 0.91), (0.0, 1.0), "unit positive cap")


def test_origin_span_uses_ten_times_a_small_positive_max() -> None:
    """Brier-like values get 10× headroom, still capped at 1."""
    _check(_origin_span(0.003, 0.004), (0.0, 0.04), "10× small positive")


def test_origin_span_caps_unit_negative_values_at_minus_one() -> None:
    """Negative unit-scale values use [-1, 0]."""
    _check(_origin_span(-0.8, -0.3), (-1.0, 0.0), "unit negative cap")


def test_origin_span_uses_ten_times_a_small_negative_min() -> None:
    """Small negative values get 10× headroom, still capped at -1."""
    _check(_origin_span(-0.004, -0.003), (-0.04, 0.0), "10× small negative")


def test_origin_span_does_not_clip_values_above_one() -> None:
    """A statistic larger than 1 still fits on the axis."""
    _check(_origin_span(0.5, 2.5), (0.0, 2.5), "above-one data visible")


def test_origin_span_mixed_unit_values_use_plus_minus_one() -> None:
    """When signs mix inside [-1, 1], the axis is [-1, 1]."""
    _check(_origin_span(-0.2, 0.5), (-1.0, 1.0), "mixed unit")


def test_origin_span_rejects_non_finite_bounds() -> None:
    """Non-finite data cannot define an origin span."""
    with pytest.raises(ValueError, match="finite"):
        _origin_span(0.0, float("nan"))


def test_style_axes_sets_unit_x_limits_and_leaves_category_y() -> None:
    """Forest-plot x goes 0→1; labelled method rows keep their y window."""
    fig, ax = plt.subplots()
    try:
        ax.plot([0.72, 0.91], [0, 1])
        ax.set_yticks([0, 1], ["weighting", "CBPE"])
        ax.set_ylim(-0.6, 1.4)
        _style_axes(ax)
        _check(ax.get_xlim(), (0.0, 1.0), "x is 0 to 1")
        y0, y1 = ax.get_ylim()
        _check(round(y0, 6), -0.6, "categorical y min")
        _check(round(y1, 6), 1.4, "categorical y max")
    finally:
        plt.close(fig)


def test_include_zero_rejects_unknown_axis() -> None:
    """Only x and y can be expanded to the origin."""
    fig, ax = plt.subplots()
    try:
        with pytest.raises(ValueError, match="which"):
            _include_zero(ax, "z")
    finally:
        plt.close(fig)


def test_null_panel_x_axis_uses_zero_to_one_for_auc() -> None:
    """An AUC baseline histogram is drawn on [0, 1], not [0.48, 0.82]."""
    fig, ax = plt.subplots()
    try:
        _null_panel(ax, np.linspace(0.48, 0.58, 30), 0.82, "AUC")
        x0, x1 = ax.get_xlim()
        _check(round(x0, 6), 0.0, "AUC null starts at 0")
        _check(round(x1, 6), 1.0, "AUC null ends at 1")
    finally:
        plt.close(fig)


def test_null_histogram_bins_is_at_least_40() -> None:
    """A 30-round baseline uses 40 bins, not the old n/2 cap of 15."""
    _check(null_histogram_bins(30), 40, "30 baseline values")
    _check(null_histogram_bins(10), 40, "fewer values still use the floor")
    _check(null_histogram_bins(80), 60, "large baselines are capped")
    _check(null_histogram_bins(0), 40, "empty null uses the floor")


def test_null_histogram_bins_rejects_invalid_counts() -> None:
    """Bin count requires a non-negative integer."""
    with pytest.raises(ValueError, match="n_values"):
        null_histogram_bins(-1)
    with pytest.raises(ValueError, match="n_values"):
        null_histogram_bins(True)


def test_null_panel_draws_at_least_40_bars() -> None:
    """Kernel-test histograms are granular enough that bars are not wide blocks."""
    fig, ax = plt.subplots()
    try:
        _null_panel(ax, np.linspace(0.0, 1.0, 30), 0.8, "MMD²")
        _check(len(ax.patches) >= 40, True, f"bar count {len(ax.patches)}")
    finally:
        plt.close(fig)


def test_kernel_panel_reading_states_what_each_distance_measures() -> None:
    """MMD² is a local kernel distance; energy distance is Euclidean geometry."""
    mmd = _kernel_panel_reading("MMD²")
    energy = _kernel_panel_reading("energy")
    _check("local neighborhoods" in mmd, True, "MMD² neighborhood reading")
    _check("saturates beyond the bandwidth" in mmd, True, "RBF saturation")
    _check("Euclidean" in energy, True, "energy geometry")
    _check("same group" in energy, True, "within-group vs between-group distances")
    with pytest.raises(ValueError, match="title"):
        _kernel_panel_reading("  ")


def test_kernel_interpretation_has_one_paragraph_per_plot() -> None:
    """Shared joint reading, then one histogram paragraph per kernel panel."""
    run, baseline = _domain_run()
    entries, _rows = _kernel_results(run, baseline)
    text = _kernel_interpretation(entries)
    _check(text.count("<p>"), 3, "shared reading plus one paragraph per kernel plot")
    _check("joint feature space" in text, True, "what both distances measure")
    _check("MMD²" in text, True, "MMD² panel")
    _check("energy" in text.lower(), True, "energy panel")
    _check("local neighborhoods" in text, True, "MMD² reading")
    _check("Euclidean" in text, True, "energy reading")
    _check("grey" in text.lower() or "gray" in text.lower(), True, "baseline histogram colour")
    _check("95th" in text, True, "baseline percentile")
    _check("exceeds sampling noise" in text, True, "MMD² is beyond the percentile")
    _check("consistent with sampling noise" in text, True, "energy is inside the percentile")


def test_kernel_interpretation_rejects_non_mapping_entries() -> None:
    """Entries must be mappings with title, null, and observed."""
    with pytest.raises(ValueError, match="entries"):
        _kernel_interpretation("not a sequence of dicts")  # type: ignore[arg-type]


def test_domain_classifier_drivers_use_shap_share() -> None:
    """Driver bars are mean-|SHAP| shares, even when a gain column is also present."""
    fig, ax = plt.subplots()
    try:
        _draw_domain_drivers(ax, pd.DataFrame({
            "feature": ["x1", "x2"],
            "shap_share": [0.7, 0.3],
            "gain_share": [0.1, 0.9],
        }))
        widths = [patch.get_width() for patch in ax.patches]
        _check(widths, [0.7, 0.3], "bar widths follow shap_share, not gain_share")
        _check(ax.get_xlabel(), "SHAP importance", "axis label")
    finally:
        plt.close(fig)


def test_domain_classifier_drivers_require_shap_share() -> None:
    """A gain-only importance table is rejected instead of being plotted as SHAP."""
    fig, ax = plt.subplots()
    try:
        with pytest.raises(ValueError, match="shap_share"):
            _draw_domain_drivers(ax, pd.DataFrame({"feature": ["x1"], "gain_share": [1.0]}))
    finally:
        plt.close(fig)


def test_domain_classifier_scores_and_auc_share_a_wide_top_row() -> None:
    """Scores by population and AUC vs baseline each take about half the figure width."""
    run, baseline = _domain_run()
    fig = _domain_classifier_figure(run, baseline)
    if fig is None:
        pytest.fail("domain-classifier figure was not built")
    try:
        fig.canvas.draw()
        scores, auc = fig.axes[0].get_position(), fig.axes[1].get_position()
        _check(abs(scores.y0 - auc.y0) < 0.08, True, "scores and AUC on the same row")
        _check(scores.width > 0.35, True, f"scores width {scores.width:.3f}")
        _check(auc.width > 0.35, True, f"AUC width {auc.width:.3f}")
        drivers = fig.axes[2].get_position()
        _check(drivers.width > 0.70, True, f"drivers width {drivers.width:.3f}")
        _check(drivers.y0 < scores.y0, True, "drivers sit on the row below")
    finally:
        plt.close(fig)


def test_joint_and_short_sections_include_kernel_interpretation() -> None:
    """Detailed and short reports both explain the kernel-test panels."""
    run, baseline = _domain_run()
    joint = section_joint(run, baseline)
    short = section_short_model_evidence(run, baseline)
    if joint is None or short is None:
        pytest.fail("expected joint and model-evidence sections")
    _check("exceeds sampling noise" in joint.body, True, "detailed joint interpretation")
    _check("exceeds sampling noise" in short.body, True, "short model-evidence interpretation")
    _check("local neighborhoods" in joint.body, True, "MMD² reading in detailed report")
    _check("Euclidean" in short.body, True, "energy reading in short report")
    _check("Separated masses" in joint.body or "score plot overlays" in joint.body, True,
           "scores-by-population reading")
    _check("AUC panel" in joint.body, True, "AUC vs baseline reading")
