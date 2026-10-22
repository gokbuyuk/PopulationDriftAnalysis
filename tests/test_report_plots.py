"""Unit tests for kernel-test, domain-classifier, and feature-evidence report plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from population_drift.report import (
    BaselineRun,
    ComparisonRun,
    _axis_limits,
    _domain_classifier_figure,
    _draw_domain_drivers,
    _draw_domain_scores,
    _half_or_unit_axis_limit,
    _kernel_interpretation,
    _kernel_panel_reading,
    _kernel_results,
    _null_panel,
    null_histogram_bins,
    plot_calibration,
    plot_categories,
    plot_feature_overview,
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


def _feature_run(rows: list[dict[str, object]]) -> ComparisonRun:
    """Comparison whose feature-evidence plots use the given assessment rows.

    Args:
        rows: Feature-assessment records, including PSI and SHAP share.

    Returns:
        A comparison with an empty baseline table.
    """
    return ComparisonRun(
        Path("."),
        {},
        {},
        {
            "n_source": 10, "n_target": 5, "feature_types": {},
            "has_model": True, "label_col": None,
        },
        {"signals": [], "warnings": [], "baseline_rounds": 10, "baseline_subset_size": 5},
        pd.DataFrame(rows),
        pd.DataFrame(),
    )


def _overview_snapshot(monkeypatch: pytest.MonkeyPatch, run: ComparisonRun) -> dict[str, object]:
    """Axis limits and annotation texts from ``plot_feature_overview``.

    Args:
        monkeypatch: Pytest patcher used to intercept ``figure_html``.
        run: Comparison to plot.

    Returns:
        Titles, limits, and annotation strings for each panel.
    """
    snapshot: dict[str, object] = {}

    def _capture(fig: plt.Figure, alt: str, terms: str = "") -> str:
        snapshot["titles"] = [ax.get_title() for ax in fig.axes]
        snapshot["xlims"] = [ax.get_xlim() for ax in fig.axes]
        snapshot["ylims"] = [ax.get_ylim() for ax in fig.axes]
        snapshot["texts"] = [[text.get_text() for text in ax.texts] for ax in fig.axes]
        plt.close(fig)
        return ""

    monkeypatch.setattr("population_drift.report.figure_html", _capture)
    plot_feature_overview(run)
    if not snapshot:
        pytest.fail("plot_feature_overview did not build a figure")
    return snapshot


def test_half_or_unit_axis_limit_stays_at_half_when_values_are_at_most_half() -> None:
    """The default upper bound is 0.5, including a value of exactly 0.5."""
    _check(_half_or_unit_axis_limit([0.0, 0.25, 0.5]), 0.5, "all values <= 0.5")
    _check(_half_or_unit_axis_limit([]), 0.5, "empty values")
    _check(_half_or_unit_axis_limit([np.nan]), 0.5, "non-finite values")


def test_half_or_unit_axis_limit_opens_to_one_when_any_value_exceeds_half() -> None:
    """Any finite value above 0.5 raises the bound to 1."""
    _check(_half_or_unit_axis_limit([0.2, 0.5001]), 1.0, "just above 0.5")
    _check(_half_or_unit_axis_limit([0.9, np.nan]), 1.0, "NaN is ignored")


def test_half_or_unit_axis_limit_rejects_non_numeric_input() -> None:
    """The bound requires a numeric sequence."""
    with pytest.raises(TypeError, match="values"):
        _half_or_unit_axis_limit(None)
    with pytest.raises(TypeError, match="values"):
        _half_or_unit_axis_limit("psi")


def test_axis_limits_start_at_zero_for_probabilities() -> None:
    """Probability axes start at 0 and pad the maximum by 0.1."""
    _check(_axis_limits([0.0, 1.0], probability=True), (0.0, 1.1), "full unit range")
    _check(_axis_limits([0.2, 0.8], probability=True), (0.0, 0.9), "min above 0 still starts at 0")


def test_axis_limits_pad_max_and_keep_zero_when_min_is_zero() -> None:
    """Non-probability axes use max + 0.1 and start at 0 when the minimum is 0."""
    _check(_axis_limits([0.0, 0.4]), (0.0, 0.5), "min 0")
    _check(_axis_limits([0.2, 0.5]), (0.1, 0.6), "positive min is padded")
    _check(_axis_limits([0.05, 0.5]), (0.0, 0.6), "padding below a small min does not go negative")


def test_axis_limits_pad_below_zero_when_values_are_negative() -> None:
    """Negative data keeps a negative side down to min - 0.1."""
    _check(_axis_limits([-0.3, 0.5]), (-0.4, 0.6), "signed range")


def test_axis_limits_rejects_invalid_input() -> None:
    """The bound requires finite numeric values."""
    with pytest.raises(TypeError, match="values"):
        _axis_limits(None)
    with pytest.raises(TypeError, match="values"):
        _axis_limits("auc")
    with pytest.raises(ValueError, match="finite"):
        _axis_limits([np.nan])


def test_domain_scores_x_axis_starts_at_zero() -> None:
    """Scores by population is a probability axis and must not show negatives."""
    fig, ax = plt.subplots()
    try:
        _draw_domain_scores(ax, {
            "bins": np.linspace(0.0, 1.0, 41).tolist(),
            "source_hist": np.ones(40).tolist(),
            "target_hist": np.linspace(0.0, 2.0, 40).tolist(),
        })
        _check(ax.get_xlim(), (0.0, 1.1), "probability x-axis")
        _check(ax.get_ylim(), (0.0, 2.1), "count y-axis")
    finally:
        plt.close(fig)


def test_domain_drivers_x_axis_starts_at_zero() -> None:
    """SHAP-share bars start at 0 and pad the largest share by 0.1."""
    fig, ax = plt.subplots()
    try:
        _draw_domain_drivers(ax, pd.DataFrame({
            "feature": ["x1", "x2"],
            "shap_share": [0.7, 0.3],
        }))
        _check(ax.get_xlim(), (0.0, 0.8), "share x-axis")
    finally:
        plt.close(fig)


def test_null_panel_pads_positive_and_signed_ranges() -> None:
    """Baseline histograms start at 0 when min is 0, and show min - 0.1 when signed."""
    fig, ax = plt.subplots()
    try:
        _null_panel(ax, np.linspace(0.0, 1.0, 30), 0.8, "MMD²")
        _check(ax.get_xlim(), (0.0, 1.1), "non-negative statistic")
    finally:
        plt.close(fig)
    fig, ax = plt.subplots()
    try:
        _null_panel(ax, np.linspace(-0.3, 0.5, 30), 0.4, "Δ")
        _check(ax.get_xlim(), (-0.4, 0.6), "signed statistic")
    finally:
        plt.close(fig)


def _capture_plot(monkeypatch: pytest.MonkeyPatch, fn: object, *args: object) -> dict[str, object]:
    """Axis limits from a plot function that embeds a figure.

    Args:
        monkeypatch: Pytest patcher used to intercept ``figure_html``.
        fn: Plot function.
        *args: Arguments forwarded to ``fn``.

    Returns:
        X and Y limits for each panel.
    """
    snapshot: dict[str, object] = {}

    def _capture(fig: plt.Figure, alt: str, terms: str = "") -> str:
        snapshot["xlims"] = [ax.get_xlim() for ax in fig.axes if ax.axison]
        snapshot["ylims"] = [ax.get_ylim() for ax in fig.axes if ax.axison]
        plt.close(fig)
        return ""

    monkeypatch.setattr("population_drift.report.figure_html", _capture)
    fn(*args)
    if not snapshot:
        pytest.fail("plot did not build a figure")
    return snapshot


def test_category_share_axis_starts_at_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """Category frequency bars are a probability axis starting at 0."""
    run = ComparisonRun(
        Path("."),
        {"categorical__main": pd.DataFrame([{"feature": "c1", "cramers_v": 0.2, "psi": 0.1}])},
        {"categorical": {"frequencies": {
            "c1": {"levels": ["a", "b"], "source": [0.6, 0.4], "target": [0.5, 0.5]},
        }}},
        {"n_source": 10, "n_target": 5, "feature_types": {}, "has_model": False, "label_col": None},
        {"signals": [], "warnings": [], "baseline_rounds": 10, "baseline_subset_size": 5},
        pd.DataFrame(),
        pd.DataFrame(),
    )
    snapshot = _capture_plot(monkeypatch, plot_categories, run, ["c1"])
    _check(snapshot["ylims"][0], (0.0, 0.7), "share y-axis")


def test_calibration_axes_start_at_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """Calibration predicted probability and positive rate start at 0."""
    run = ComparisonRun(
        Path("."),
        {"calibration__reliability": pd.DataFrame({
            "mean_predicted": [0.2, 0.8], "observed_rate": [0.25, 0.75],
        })},
        {},
        {"n_source": 10, "n_target": 5, "feature_types": {}, "has_model": True, "label_col": "y"},
        {"signals": [], "warnings": [], "baseline_rounds": 10, "baseline_subset_size": 5},
        pd.DataFrame(),
        pd.DataFrame(),
    )
    snapshot = _capture_plot(monkeypatch, plot_calibration, run)
    _check(snapshot["xlims"][0], (0.0, 1.1), "predicted probability")
    _check(snapshot["ylims"][0], (0.0, 1.1), "positive rate")


def test_psi_by_feature_axis_is_half_when_all_psi_are_at_most_half(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PSI-by-feature uses [0, 0.5] when no bar exceeds 0.5."""
    run = _feature_run([
        {"feature": "a", "severity": "mild", "psi": 0.20, "model_importance_share": 0.1,
         "beyond_baseline": False},
        {"feature": "b", "severity": "moderate", "psi": 0.50, "model_importance_share": 0.2,
         "beyond_baseline": True},
    ])
    snapshot = _overview_snapshot(monkeypatch, run)
    _check(snapshot["titles"][0], "PSI by feature", "left panel title")
    _check(snapshot["xlims"][0], (0.0, 0.5), "PSI-by-feature x-axis")


def test_psi_by_feature_axis_is_one_when_any_psi_exceeds_half(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PSI-by-feature uses [0, 1] when any bar exceeds 0.5."""
    run = _feature_run([
        {"feature": "a", "severity": "mild", "psi": 0.20, "model_importance_share": 0.1,
         "beyond_baseline": False},
        {"feature": "b", "severity": "severe", "psi": 0.51, "model_importance_share": 0.2,
         "beyond_baseline": True},
    ])
    snapshot = _overview_snapshot(monkeypatch, run)
    _check(snapshot["xlims"][0], (0.0, 1.0), "PSI-by-feature x-axis")


def test_psi_vs_shap_axes_use_half_or_unit_limits_independently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each scatter axis is 0.5 unless that axis has a value above 0.5."""
    run = _feature_run([
        {"feature": "a", "severity": "mild", "psi": 0.10, "model_importance_share": 0.20,
         "beyond_baseline": False},
        {"feature": "b", "severity": "moderate", "psi": 0.40, "model_importance_share": 0.60,
         "beyond_baseline": True},
    ])
    snapshot = _overview_snapshot(monkeypatch, run)
    _check(snapshot["titles"][1], "PSI vs SHAP importance", "scatter title")
    _check(snapshot["xlims"][1], (0.0, 1.0), "SHAP importance x-axis")
    _check(snapshot["ylims"][1], (0.0, 0.5), "PSI y-axis")


def test_psi_vs_shap_ylim_is_one_when_psi_exceeds_half(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The scatter PSI axis opens to 1 when any PSI exceeds 0.5."""
    run = _feature_run([
        {"feature": "a", "severity": "mild", "psi": 0.10, "model_importance_share": 0.10,
         "beyond_baseline": False},
        {"feature": "b", "severity": "severe", "psi": 0.70, "model_importance_share": 0.20,
         "beyond_baseline": True},
    ])
    snapshot = _overview_snapshot(monkeypatch, run)
    _check(snapshot["xlims"][1], (0.0, 0.5), "SHAP importance stays at 0.5")
    _check(snapshot["ylims"][1], (0.0, 1.0), "PSI y-axis opens to 1")


def test_psi_vs_shap_labels_only_features_beyond_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scatter annotations are only for features above the baseline."""
    run = _feature_run([
        {"feature": "drifted", "severity": "mild", "psi": 0.08, "model_importance_share": 0.10,
         "beyond_baseline": True},
        {"feature": "important", "severity": "moderate", "psi": 0.12, "model_importance_share": 0.80,
         "beyond_baseline": False},
        {"feature": "quiet", "severity": "mild", "psi": 0.02, "model_importance_share": 0.90,
         "beyond_baseline": False},
    ])
    snapshot = _overview_snapshot(monkeypatch, run)
    labels = snapshot["texts"][1]
    _check(labels, ["drifted"], "only the beyond-baseline feature is labelled")


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
