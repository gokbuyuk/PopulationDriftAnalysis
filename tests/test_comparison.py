"""Unit tests for the comparison decisions: pairs, baseline, feature severity, and the verdict."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from population_drift.analysis import (
    DriftAnalysisResult,
    DriftConfig,
    TaskResult,
    _paired_analysis,
    _performance_signal,
    assess_features,
    compare_to_baseline,
    interpret_results,
    run_comparison,
    run_paired_comparison,
    task_pair_map,
    validate_inputs,
)


def _continuous(feature: str, psi: float, p_value: float) -> dict[str, object]:
    """One continuous-feature row. KS D and SMD stay below the moderate cutoffs."""
    return {
        "feature": feature, "psi": psi, "ks_d": 0.01, "smd": 0.0, "smd_abs": 0.0,
        "ks_p_adj": p_value, "ad_p_adj": 1.0, "var_ratio": 1.0,
    }


def _analysis(rows: list[dict[str, object]], extra: dict[str, dict[str, pd.DataFrame]] | None = None, *,
              has_model: bool = False, pairs: dict[str, str] | None = None) -> DriftAnalysisResult:
    """A result whose only required task is the continuous table."""
    tasks: dict[str, TaskResult] = {"continuous": TaskResult("continuous", {"main": pd.DataFrame(rows)})}
    for name, tables in (extra or {}).items():
        tasks[name] = TaskResult(name, tables)
    features = [str(row["feature"]) for row in rows]
    return DriftAnalysisResult(tasks, {
        "feature_cols": features,
        "feature_types": {feature: "continuous" for feature in features},
        "has_model": has_model,
        "task_pairs": pairs or {},
    })


def _check(actual: object, expected: object, what: str) -> None:
    """Fail with the comparison that did not hold."""
    if actual != expected:
        pytest.fail(f"{what}: got {actual!r}, expected {expected!r}.")


def test_task_pair_map_uses_the_labelled_file_for_every_test_when_no_evaluation_set_is_given() -> None:
    """A single labelled table is the reference for margins and for scores."""
    pairs = task_pair_map("single")
    _check(pairs["continuous"], "labelled set vs new", "continuous pair without an evaluation set")
    _check(pairs["prediction_drift"], "labelled set vs new", "score pair without an evaluation set")
    _check(pairs["calibration"], "labelled set", "calibration pair without an evaluation set")


def test_task_pair_map_assigns_scores_to_the_evaluation_set() -> None:
    """With two labelled files, margins use the fit set and scores use the evaluation set."""
    pairs = task_pair_map("paired")
    _check(pairs["continuous"], "fit set vs new", "continuous pair with an evaluation set")
    _check(pairs["prediction_drift"], "evaluation set vs new", "score pair with an evaluation set")
    _check(pairs["calibration"], "evaluation set", "calibration pair with an evaluation set")


def test_task_pair_map_rejects_an_unknown_mode() -> None:
    """The mode is named in the error."""
    with pytest.raises(ValueError) as caught:
        task_pair_map("random")
    _check(str(caught.value), "mode must be 'single' or 'paired', got 'random'.", "task_pair_map error")


def test_validate_inputs_names_which_population_is_empty() -> None:
    """An empty frame says which side has zero rows."""
    source = pd.DataFrame({"x": [1.0]})
    with pytest.raises(ValueError) as caught:
        validate_inputs(source, pd.DataFrame({"x": []}), ["x"])
    _check(str(caught.value), "Source row count is 1; target row count is 0. Both populations must be non-empty.",
           "validate_inputs empty-target error")


def test_validate_inputs_names_the_missing_columns_on_each_side() -> None:
    """A missing feature says whether it is absent from the source, the target, or both."""
    source = pd.DataFrame({"x": [1.0]})
    target = pd.DataFrame({"y": [1.0]})
    with pytest.raises(ValueError) as caught:
        validate_inputs(source, target, ["x", "y"])
    _check(str(caught.value), "Missing columns - source: ['y'], target: ['x']", "validate_inputs missing-column error")


def test_compare_to_baseline_uses_the_add_one_empirical_p_value() -> None:
    """p = (1 + count of nulls at least as large) / (1 + rounds), and the 95th percentile is the threshold."""
    observed = pd.DataFrame([
        {"task": "continuous", "table": "main", "feature": "x", "metric": "psi", "value": 0.40},
    ])
    null = pd.DataFrame([
        {"task": "continuous", "table": "main", "feature": "x", "metric": "psi", "value": value}
        for value in (0.10, 0.20, 0.30)
    ])
    row = compare_to_baseline(observed, null, percentile=95).iloc[0]
    _check(round(float(row["empirical_p"]), 6), 0.25, "empirical p-value for an observed value above every null")
    _check(bool(row["exceeds_baseline"]), True, "exceeds_baseline when the observed value is above the 95th percentile")


def test_compare_to_baseline_does_not_call_a_statistic_extreme_when_it_has_no_null() -> None:
    """A metric absent from the baseline stays uncompared instead of being treated as drift."""
    observed = pd.DataFrame([
        {"task": "continuous", "table": "main", "feature": "x", "metric": "psi", "value": 0.40},
    ])
    row = compare_to_baseline(observed, pd.DataFrame(columns=["task", "table", "feature", "metric", "value"])).iloc[0]
    _check(row["exceeds_baseline"], None, "exceeds_baseline with no baseline rounds")
    _check(int(row["n_baseline_rounds"]), 0, "baseline round count with no null")


def test_assess_features_is_severe_only_when_significance_baseline_and_effect_all_hold() -> None:
    """A large PSI is not severe when the test is not significant or the value is inside the baseline."""
    rows = [
        _continuous("large_and_beyond", 0.30, 0.001),
        _continuous("large_but_inside_baseline", 0.30, 0.001),
        _continuous("large_but_not_significant", 0.30, 0.50),
        _continuous("moderate_and_beyond", 0.12, 0.001),
    ]
    lookup = {
        ("continuous", "large_and_beyond", "psi"): True,
        ("continuous", "large_but_inside_baseline", "psi"): False,
        ("continuous", "large_but_not_significant", "psi"): True,
        ("continuous", "moderate_and_beyond", "psi"): True,
    }
    severity = assess_features(_analysis(rows), lookup, DriftConfig()).set_index("feature")["severity"]
    _check(severity["large_and_beyond"], "severe", "PSI 0.30, adjusted p < 0.05, beyond the baseline")
    _check(severity["large_but_inside_baseline"], "mild", "PSI 0.30 that does not exceed the baseline")
    _check(severity["large_but_not_significant"], "mild", "PSI 0.30 whose adjusted p is 0.50")
    _check(severity["moderate_and_beyond"], "moderate", "PSI 0.12, adjusted p < 0.05, beyond the baseline")


def test_interpret_results_is_severe_when_any_counted_signal_is_severe() -> None:
    """One severe feature signal sets the verdict, and the pair from the run is copied onto that signal."""
    rows = [_continuous("x", 0.30, 0.001)]
    pairs = task_pair_map("single")
    baseline = pd.DataFrame([{"task": "continuous", "feature": "x", "metric": "psi", "exceeds_baseline": True}])
    decision = interpret_results(_analysis(rows, pairs=pairs), baseline, DriftConfig())
    _check(decision["verdict_level"], "severe", "verdict with one severe feature")
    _check(decision["n_severe"], 1, "severe-signal count")
    feature_signal = decision["signals"][0]
    _check(feature_signal["pair"], "labelled set vs new", "pair recorded on the feature-drift signal")


def test_interpret_results_stays_mild_for_two_moderate_signals_that_are_not_model_important() -> None:
    """Two moderate signals are allowed when neither is marked important."""
    rows = [_continuous("x", 0.01, 1.0)]
    extra = {
        "categorical": {"main": pd.DataFrame([{"feature": "c", "unseen_share": 0.02}])},
        "domain_classifier": {
            "main": pd.DataFrame([{"feature": "__global__", "auc": 0.70, "permutation_p": 0.01}]),
            "importance": pd.DataFrame([{"feature": "x", "shap_share": 1.0}]),
        },
    }
    baseline = pd.DataFrame([
        {"task": "continuous", "feature": "x", "metric": "psi", "exceeds_baseline": False},
        {"task": "domain_classifier", "feature": "__global__", "metric": "auc", "exceeds_baseline": True},
    ])
    decision = interpret_results(_analysis(rows, extra), baseline, DriftConfig())
    _check(decision["verdict_level"], "mild", "verdict with unseen levels and domain AUC 0.70, neither important")
    _check(decision["n_moderate"], 2, "moderate-signal count for unseen levels plus domain AUC")
    _check(decision["n_severe"], 0, "severe-signal count when no signal is severe")


def test_interpret_results_is_moderate_when_a_third_unimportant_moderate_signal_is_added() -> None:
    """Three moderate signals are more than the mild rule allows, even when none is model-important."""
    rows = [_continuous("x", 0.01, 1.0)]
    extra = {
        "categorical": {"main": pd.DataFrame([{"feature": "c", "unseen_share": 0.02}])},
        "domain_classifier": {
            "main": pd.DataFrame([{"feature": "__global__", "auc": 0.70, "permutation_p": 0.01}]),
            "importance": pd.DataFrame([{"feature": "x", "shap_share": 1.0}]),
        },
        "mmd": {"main": pd.DataFrame([{"feature": "__global__", "mmd2": 0.01, "permutation_p": 0.01}])},
    }
    baseline = pd.DataFrame([
        {"task": "continuous", "feature": "x", "metric": "psi", "exceeds_baseline": False},
        {"task": "domain_classifier", "feature": "__global__", "metric": "auc", "exceeds_baseline": True},
        {"task": "mmd", "feature": "__global__", "metric": "mmd2", "exceeds_baseline": True},
    ])
    decision = interpret_results(_analysis(rows, extra), baseline, DriftConfig())
    _check(decision["verdict_level"], "moderate", "verdict with three moderate signals")
    _check(decision["n_moderate"], 3, "moderate-signal count after adding the kernel test")


def test_interpret_results_does_not_let_an_info_signal_change_the_verdict() -> None:
    """Calibration is reported and is not counted."""
    rows = [_continuous("x", 0.01, 1.0)]
    extra = {"calibration": {"main": pd.DataFrame([{"ece": 0.90, "calibration_slope": 0.5}])}}
    baseline = pd.DataFrame([{"task": "continuous", "feature": "x", "metric": "psi", "exceeds_baseline": False}])
    decision = interpret_results(_analysis(rows, extra), baseline, DriftConfig())
    info = [signal for signal in decision["signals"] if signal["level"] == "info"]
    _check(decision["verdict_level"], "mild", "verdict when the only extra signal is calibration")
    _check(len(info), 1, "number of info signals")
    _check(decision["n_moderate"], 0, "moderate count when calibration is the extra signal")


def test_performance_signal_is_severe_only_when_the_whole_interval_is_beyond_tolerance() -> None:
    """A point estimate beyond tolerance is moderate. The whole interval beyond tolerance is severe."""
    config = DriftConfig()

    def signal(estimate: float, low: float, high: float) -> str:
        table = pd.DataFrame([{
            "metric": "auc", "higher_is_better": True, "reference_value": 0.80,
            "estimate": estimate, "ci_low": low, "ci_high": high,
        }])
        result = _performance_signal("Estimated performance", table, "auc", config)
        assert result is not None, "AUC row was not scored."
        return str(result["level"])

    _check(signal(0.70, 0.68, 0.72), "severe", "AUC interval 0.68–0.72 against reference 0.80 and tolerance 0.02")
    _check(signal(0.77, 0.76, 0.79), "moderate", "AUC point 0.77 beyond tolerance while the interval reaches 0.79")
    _check(signal(0.795, 0.785, 0.805), "mild", "AUC interval inside the 0.02 tolerance")


def test_paired_comparison_names_the_reference_that_is_too_small() -> None:
    """The error names the short reference, its row count, and the required count."""
    fit_set = pd.DataFrame({"x": range(200), "target": [0, 1] * 100})
    evaluation = fit_set.iloc[:40].copy()
    new = pd.DataFrame({"x": range(50)})
    with pytest.raises(ValueError) as caught:
        run_paired_comparison(fit_set, evaluation, new, ["x"], label_col="target", output_dir="unused")
    _check(str(caught.value), "Evaluation set has 40 rows. A baseline matched to 50 new rows needs at least 80.",
           "undersized evaluation error")


def test_paired_comparison_names_a_missing_column_and_which_table_lacks_it() -> None:
    """The error names the table and the column."""
    fit_set = pd.DataFrame({"x": range(100)})
    evaluation = pd.DataFrame({"y": range(100)})
    new = pd.DataFrame({"x": range(40)})
    with pytest.raises(ValueError) as caught:
        run_paired_comparison(fit_set, evaluation, new, ["x"], output_dir="unused")
    _check(str(caught.value), "Evaluation set is missing columns: ['x'].", "missing-column error")


def test_paired_comparison_rejects_the_label_as_a_feature() -> None:
    """The error names the label column that was left in the feature list."""
    frame = pd.DataFrame({"x": range(100), "target": [0, 1] * 50})
    new = pd.DataFrame({"x": range(40), "target": [0, 1] * 20})
    with pytest.raises(ValueError) as caught:
        run_paired_comparison(frame, frame.iloc[:80], new, ["x", "target"], label_col="target", output_dir="unused")
    _check(str(caught.value), "Leave the label column 'target' out of the feature list.", "label-in-features error")


def test_paired_analysis_uses_the_fit_set_for_feature_tests_and_the_evaluation_set_for_scores(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Feature scope receives the fit rows. Model scope receives the evaluation rows and the label."""
    seen: list[tuple[str, int, object, object]] = []

    def fake(*args: object, scope: str = "all", **kwargs: object) -> DriftAnalysisResult:
        source = args[0]
        seen.append((scope, int(source["x"].iloc[0]), args[4], args[5]))  # type: ignore[index]
        return DriftAnalysisResult({}, {"failed_tasks": {}, "elapsed_seconds": 0.0, "feature_cols": ["x"],
                                        "feature_types": {"x": "continuous"}, "has_model": args[4] is not None})

    monkeypatch.setattr("population_drift.analysis.population_drift_analyzer", fake)
    fit_set = pd.DataFrame({"x": [1]})
    evaluation = pd.DataFrame({"x": [2], "target": [0]})
    _paired_analysis(fit_set, evaluation, pd.DataFrame({"x": [3]}), ["x"], None, "model-object", "target", DriftConfig())
    _check(seen[0], ("features", 1, None, None), "first analyzer call")
    _check(seen[1], ("model", 2, "model-object", "target"), "second analyzer call")


def test_run_comparison_names_features_that_differ_from_the_baseline(tmp_path: Path) -> None:
    """A baseline built on a different feature list fails before the analysis and names both sides."""
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    pd.DataFrame(columns=["task", "table", "feature", "metric", "value"]).to_csv(
        baseline / "baseline_statistics.csv", index=False)
    pd.DataFrame(columns=["task", "table", "feature", "metric"]).to_csv(baseline / "baseline_summary.csv", index=False)
    (baseline / "baseline_metadata.json").write_text(json.dumps({
        "feature_cols": ["a"], "subset_size": 10, "has_model": False, "n_rounds": 1,
    }))
    source, target = pd.DataFrame({"b": [1.0]}), pd.DataFrame({"b": [2.0]})
    with pytest.raises(ValueError) as caught:
        run_comparison(source, target, ["b"], baseline_dir=baseline, output_dir=tmp_path / "out", n_baseline_rounds=1)
    _check(str(caught.value),
           "The baseline feature set does not match this comparison. Only in the baseline: ['a']. "
           "Only in this comparison: ['b'].",
           "baseline feature-set error")
