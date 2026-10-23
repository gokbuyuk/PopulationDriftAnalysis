"""Stakeholder report: will performance on the new population stay inside the pre-set band?

The headline uses label-free estimates that remain valid. Feature rows explain that
headline. Marginal tests that do not move it stay in the detailed report.
"""

from __future__ import annotations

import html
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from population_drift.report import (
    SOURCE_COLOR,
    TARGET_COLOR,
    ComparisonRun,
    Section,
    _axis_limits,
    _collapse_shared_text,
    _style_axes,
    figure_html,
    plot_calibration,
    plot_categories,
    plot_ecdfs,
    write_report_files,
)

_Z = 1.959963984540054
_HEADLINE_TEXT = {
    "within": "Within expectation",
    "imprecise": "Imprecise",
    "outside": "Outside expectation",
    "not_identified": "Not identified",
}
_ACTIONS = {
    "within": (
        "The valid estimates sit inside the tolerance. This is not a measurement of accuracy "
        "on new labels. Score the new population only with a scheduled rerun."
    ),
    "imprecise": (
        "Do not treat the tolerance as confirmed. Label the sample below before relying on the transfer."
    ),
    "outside": (
        "Do not expect the reference performance on this population. Hold unchanged scoring, "
        "label the sample below, and retrain, recalibrate, or restrict the rows you score."
    ),
    "not_identified": (
        "Do not quote a performance number for the new population. An estimate is missing, "
        "or the valid estimates disagree."
    ),
}


@dataclass(frozen=True)
class PerformanceView:
    """One estimated metric, and whether the headline may use it.

    Attributes:
        label: Row name.
        usable: Whether the estimate's assumption holds.
        position: ``inside``, ``outside``, ``crosses``, ``point_only``, or ``unused``.
        counts: Whether the headline aggregates this row.
        reference: Reference value of the metric.
        estimate: Estimate on the new population.
        ci_low: Lower end of the stored 95% interval.
        ci_high: Upper end of the stored 95% interval.
        assumption: Assumption the number requires.
        note: Why the row was dropped or discounted.
    """

    label: str
    usable: bool
    position: str
    counts: bool
    reference: float
    estimate: float
    ci_low: float
    ci_high: float
    assumption: str
    note: str


@dataclass(frozen=True)
class RateView:
    """Observed operating rate and, when a band was fixed, where its interval sits.

    Attributes:
        source: Reference positive rate.
        target: New positive rate.
        ci_low: Wilson lower end for the new rate.
        ci_high: Wilson upper end for the new rate.
        tolerance: Absolute band around the reference rate, or None when unset.
        position: ``inside``, ``outside``, ``crosses``, or ``shown``.
        near_source: Reference share of scores within 0.05 of the threshold.
        near_target: New share of scores within 0.05 of the threshold.
        psi: Score population stability index.
        ks: Score Kolmogorov–Smirnov statistic.
        threshold: Production threshold stored on the run.
    """

    source: float
    target: float
    ci_low: float
    ci_high: float
    tolerance: float | None
    position: str
    near_source: float
    near_target: float
    psi: float
    ks: float
    threshold: float


@dataclass(frozen=True)
class Cause:
    """One model-important feature that cleared the drift gates.

    Attributes:
        feature: Feature name.
        kind: ``continuous`` or ``categorical``.
        direction: Plain-language change.
        effect: One effect size.
        share: Share of mean absolute TreeSHAP.
        severity: ``moderate`` or ``severe``.
    """

    feature: str
    kind: str
    direction: str
    effect: str
    share: str
    severity: str


@dataclass(frozen=True)
class ExpectationJudgement:
    """Headline and the rows the expectation report renders.

    Attributes:
        headline: ``within``, ``imprecise``, ``outside``, or ``not_identified``.
        performance_headline: Headline before the operating-rate and in-sample rules.
        in_sample: Whether score tests used the fitting rows.
        paragraphs: Sentences under the headline.
        action: What the headline authorizes.
        anchor: Label and value pairs fixed for the run.
        metrics: Performance rows, including rows that were dropped.
        rate: Operating rate, or None when scores are absent.
        causes: At most five important features that moved.
        extra_causes: Important features omitted past the five-row cap.
        low_importance: Count of non-important features that moved.
        support: Tail-share clauses for important features.
        shap_clause: Rank-correlation clause, or an empty string.
        validity: Lines on overlap, calibration, and failures.
        confirm: Labelled-sample and archive lines.
        bbse: Alternative prior sentence, or an empty string.
    """

    headline: str
    performance_headline: str
    in_sample: bool
    paragraphs: tuple[str, ...]
    action: str
    anchor: tuple[tuple[str, str], ...]
    metrics: tuple[PerformanceView, ...]
    rate: RateView | None
    causes: tuple[Cause, ...]
    extra_causes: int
    low_importance: int
    support: tuple[str, ...]
    shap_clause: str
    validity: tuple[str, ...]
    confirm: tuple[str, ...]
    bbse: str


def expectation_report_path(output_path: str | Path) -> Path:
    """Sibling path for the expectation report (``report.html`` → ``report_expectation.html``).

    Args:
        output_path: Destination of the detailed HTML report.

    Returns:
        Path with ``_expectation`` inserted before the suffix.

    Raises:
        ValueError: If ``output_path`` is blank.
    """
    if not str(output_path).strip():
        raise ValueError("output_path is empty.")
    path = Path(output_path)
    suffix = path.suffix or ".html"
    return path.with_name(f"{path.stem}_expectation{suffix}")


def judge_expectation(run: ComparisonRun) -> ExpectationJudgement:
    """Decide whether estimated performance stays inside the pre-set tolerance.

    Args:
        run: Loaded comparison outputs.

    Returns:
        The headline and the rows that support it.

    Raises:
        TypeError: If ``run`` is not a comparison.
    """
    if not isinstance(run, ComparisonRun):
        raise TypeError(f"run must be a ComparisonRun, got {type(run).__name__}.")
    metrics = tuple(_metric_views(run))
    rate = _rate_view(run)
    in_sample = _in_sample(run)
    performance = _combine([item.position for item in metrics if item.counts and item.usable])
    return _judgement(run, metrics, rate, in_sample, performance)


def _judgement(run: ComparisonRun, metrics: tuple[PerformanceView, ...], rate: RateView | None,
               in_sample: bool, performance: str) -> ExpectationJudgement:
    """Assemble the report model from pieces that are already judged.

    Args:
        run: Loaded comparison.
        metrics: Performance rows.
        rate: Operating rate, or None.
        in_sample: Whether the reference scores were fit on these rows.
        performance: Headline from the valid estimates alone.

    Returns:
        The report model.
    """
    headline = _apply_context(performance, rate, in_sample)
    causes, extra = _causes(run)
    return ExpectationJudgement(
        headline=headline,
        performance_headline=performance,
        in_sample=in_sample,
        paragraphs=_paragraphs(headline, performance, rate, in_sample),
        action=_ACTIONS[headline],
        anchor=_anchor(run, rate),
        metrics=metrics,
        rate=rate,
        causes=tuple(causes),
        extra_causes=extra,
        low_importance=_low_importance(run),
        support=tuple(_support_lines(run)),
        shap_clause=_shap_clause(run),
        validity=tuple(_validity_lines(run, metrics, in_sample)),
        confirm=tuple(_confirm_lines(run, metrics)),
        bbse=_bbse_line(run, bool(causes)),
    )


def write_expectation_report(detailed_output_path: str | Path, run: ComparisonRun, detailed_name: str) -> Path:
    """Write the expectation report next to the detailed report.

    Args:
        detailed_output_path: Destination of the detailed HTML report.
        run: Loaded comparison outputs.
        detailed_name: Filename of that detailed report, linked from this page.

    Returns:
        Resolved path of the written HTML report.

    Raises:
        TypeError: If ``run`` is not a comparison.
        ValueError: If ``detailed_name`` is blank.
    """
    if not isinstance(detailed_name, str) or not detailed_name.strip():
        raise ValueError("detailed_name is empty.")
    judgement = judge_expectation(run)
    sections = _sections(judgement, run, detailed_name.strip())
    return write_report_files(
        expectation_report_path(detailed_output_path),
        "Population drift: performance on the new data",
        _subtitle(run),
        sections,
    )


def interval_position(estimate: float, low: float, high: float, bound: float, higher_is_better: bool) -> str:
    """Place an estimate relative to a one-sided performance bound.

    Args:
        estimate: Point estimate.
        low: Lower end of the 95% interval.
        high: Upper end of the 95% interval.
        bound: Worst acceptable value.
        higher_is_better: Whether larger values are better.

    Returns:
        ``inside``, ``outside``, ``crosses``, ``point_only``, or ``unused``.

    Raises:
        ValueError: If ``bound`` is not finite.
    """
    if not math.isfinite(bound):
        raise ValueError("bound must be finite.")
    if not math.isfinite(estimate):
        return "unused"
    if not (math.isfinite(low) and math.isfinite(high)):
        return "point_only"
    if higher_is_better:
        return _sided_position(low >= bound, high < bound)
    return _sided_position(high <= bound, low > bound)


def labelled_sample_size(auc: float, prevalence: float, tolerance: float, max_n: int = 2_000_000) -> int | None:
    """Smallest n whose Hanley–McNeil 95% AUC interval has half-width at most ``tolerance``.

    The standard error is the one in Hanley and McNeil (1982) for a single AUC.
    It sizes a confidence interval, not a test of a difference between two AUCs.

    Args:
        auc: Anticipated AUC, in (0, 1).
        prevalence: Anticipated positive rate, in (0, 1).
        tolerance: Largest acceptable half-width.
        max_n: Largest n to search.

    Returns:
        The smallest such n, or None when ``max_n`` is still too small.

    Raises:
        ValueError: If an argument is outside its range.
    """
    _validate_sample_size(auc, prevalence, tolerance, max_n)
    if _auc_half_width(auc, prevalence, max_n) > tolerance:
        return None
    low, high = 2, max_n
    while low < high:
        mid = (low + high) // 2
        if _auc_half_width(auc, prevalence, mid) <= tolerance:
            high = mid
        else:
            low = mid + 1
    return low


def wilson_interval(successes: float, n: int) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Args:
        successes: Number of successes. Need not be an integer when it was recovered from a rate.
        n: Number of trials.

    Returns:
        Lower and upper 95% bounds.

    Raises:
        ValueError: If ``n`` is not positive or ``successes`` is outside ``[0, n]``.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}.")
    if not 0 <= successes <= n:
        raise ValueError(f"successes must be in [0, n], got {successes}.")
    p = successes / n
    den = 1 + _Z ** 2 / n
    center = (p + _Z ** 2 / (2 * n)) / den
    margin = _Z * math.sqrt(p * (1 - p) / n + _Z ** 2 / (4 * n * n)) / den
    return center - margin, center + margin


def _sided_position(entirely_acceptable: bool, entirely_worse: bool) -> str:
    """Map the two ends of an interval to a position word.

    Args:
        entirely_acceptable: Whether every point in the interval is acceptable.
        entirely_worse: Whether every point in the interval is worse than the bound.

    Returns:
        ``inside``, ``outside``, or ``crosses``.
    """
    if entirely_acceptable:
        return "inside"
    if entirely_worse:
        return "outside"
    return "crosses"


def _validate_sample_size(auc: float, prevalence: float, tolerance: float, max_n: int) -> None:
    """Reject sample-size arguments outside their ranges.

    Args:
        auc: Anticipated AUC.
        prevalence: Anticipated positive rate.
        tolerance: Largest acceptable half-width.
        max_n: Largest n to search.

    Raises:
        ValueError: If an argument is outside its range.
    """
    if not 0 < auc < 1:
        raise ValueError(f"auc must be in (0, 1), got {auc}.")
    if not 0 < prevalence < 1:
        raise ValueError(f"prevalence must be in (0, 1), got {prevalence}.")
    if tolerance <= 0:
        raise ValueError(f"tolerance must be positive, got {tolerance}.")
    if max_n < 2:
        raise ValueError(f"max_n must be at least 2, got {max_n}.")


def _auc_half_width(auc: float, prevalence: float, n: int) -> float:
    """Hanley–McNeil 95% half-width for one AUC at sample size ``n``.

    Args:
        auc: Anticipated AUC.
        prevalence: Anticipated positive rate.
        n: Total labelled rows.

    Returns:
        Approximate half-width of the 95% interval.
    """
    n_pos = max(prevalence * n, 1.0)
    n_neg = max((1 - prevalence) * n, 1.0)
    q1 = auc / (2 - auc)
    q2 = 2 * auc * auc / (1 + auc)
    var = auc * (1 - auc) + (n_pos - 1) * (q1 - auc ** 2) + (n_neg - 1) * (q2 - auc ** 2)
    var = max(var, 0.0) / (n_pos * n_neg)
    return _Z * math.sqrt(var)


def _combine(positions: Sequence[str]) -> str:
    """Aggregate valid metric positions into a performance headline.

    Args:
        positions: ``inside``, ``outside``, ``crosses``, or ``point_only``.

    Returns:
        ``within``, ``imprecise``, ``outside``, or ``not_identified``.
    """
    if not positions:
        return "not_identified"
    if "outside" in positions and any(item != "outside" for item in positions):
        return "not_identified"
    if all(item == "outside" for item in positions):
        return "outside"
    if any(item != "inside" for item in positions):
        return "imprecise"
    return "within"


def _apply_context(performance: str, rate: RateView | None, in_sample: bool) -> str:
    """Apply the in-sample cap and the operating-rate band.

    Args:
        performance: Headline from the valid performance estimates.
        rate: Operating rate, or None.
        in_sample: Whether the reference scores were fit on these rows.

    Returns:
        The headline shown on the page.
    """
    headline = "imprecise" if in_sample and performance == "within" else performance
    if rate is None or rate.position == "shown":
        return headline
    if rate.position == "outside":
        return "outside"
    if rate.position == "crosses" and headline == "within":
        return "imprecise"
    return headline


def _paragraphs(headline: str, performance: str, rate: RateView | None, in_sample: bool) -> tuple[str, ...]:
    """Sentences that state the headline in the terms that produced it.

    Args:
        headline: Headline after context rules.
        performance: Headline from the estimates alone.
        rate: Operating rate, or None.
        in_sample: Whether the reference scores were fit on these rows.

    Returns:
        Paragraphs for the decision section.
    """
    lines = [
        "New labels are absent, so a change in P(y|x) is unidentified. "
        "Every performance number below is an estimate under the assumption in its row."
    ]
    lines.append(_performance_sentence(performance))
    lines.extend(_rate_sentences(rate))
    if in_sample and headline == "outside":
        lines.append("The reference scores are in-sample, so the gap versus production performance may be larger.")
    elif in_sample:
        lines.append("The reference scores are in-sample, so this run cannot conclude that performance is within expectation.")
    if performance == "not_identified" and headline == "outside":
        lines.append("Discrimination on the new population remains not identified.")
    return tuple(lines)


def _rate_sentences(rate: RateView | None) -> list[str]:
    """Sentences for an operating-rate interval that misses or crosses its band.

    Args:
        rate: Operating rate, or None.

    Returns:
        Zero or one sentence.
    """
    if rate is None or rate.tolerance is None:
        return []
    if rate.position == "outside":
        return [f"The new operating-rate interval lies outside ±{_metric_num(rate.tolerance)} of the reference rate."]
    if rate.position == "crosses":
        return ["The new operating-rate interval crosses its band."]
    return []


def _performance_sentence(performance: str) -> str:
    """One sentence for the estimate-only headline.

    Args:
        performance: ``within``, ``imprecise``, ``outside``, or ``not_identified``.

    Returns:
        The sentence.
    """
    if performance == "within":
        return "Every valid performance estimate lies on the acceptable side of the tolerance."
    if performance == "imprecise":
        return "A valid estimate is consistent with the tolerance and too wide to confirm it."
    if performance == "outside":
        return "A valid performance estimate lies entirely on the worse side of the tolerance."
    return "No valid performance estimate identifies whether the new population stays inside the tolerance."


def _metric_views(run: ComparisonRun) -> list[PerformanceView]:
    """Weighted metric, and CBPE when the problem is binary.

    Args:
        run: Loaded comparison.

    Returns:
        Rows in display order. Dropped estimators are included with ``usable`` false.
    """
    views = [_weighted_view(run)]
    if _problem(run) != "binary":
        return views
    cbpe = _estimate_view(
        run, "cbpe", "auc", "CBPE AUC",
        "Calibrated scores and an unchanged map from score to label.",
        _cbpe_block(run), True,
    )
    views.append(cbpe)
    precision = _precision_view(run, cbpe.usable)
    if precision is not None:
        views.append(precision)
    return views


def _weighted_view(run: ComparisonRun) -> PerformanceView:
    """Importance-weighted AUC or RMSE, dropped when the effective sample is too small.

    Args:
        run: Loaded comparison.

    Returns:
        The weighted-performance row.
    """
    name = "rmse" if _problem(run) == "regression" else "auc"
    label = "Weighted RMSE" if name == "rmse" else "Weighted AUC"
    view = _estimate_view(
        run, "weighted_performance", name, label,
        "Covariate shift: P(y|x) unchanged. Weights are out-of-fold density ratios.",
        "", True,
    )
    return _discount_ess(run, view)


def _precision_view(run: ComparisonRun, cbpe_usable: bool) -> PerformanceView | None:
    """Expected precision at the threshold. It does not set the headline.

    Args:
        run: Loaded comparison.
        cbpe_usable: Whether CBPE AUC was accepted.

    Returns:
        The precision row, or None when CBPE is not usable.
    """
    if not cbpe_usable:
        return None
    row = _metric_series(run.table("cbpe"), "precision")
    if row is None:
        return None
    return PerformanceView(
        "Expected precision at the threshold", True, "unused", False,
        _finite(row.get("reference_value")), _finite(row.get("estimate")),
        _finite(row.get("ci_low")), _finite(row.get("ci_high")),
        "Same assumption as CBPE. Shown for the operating point. The tolerance applies to AUC.",
        "",
    )


def _estimate_view(run: ComparisonRun, task: str, metric: str, label: str, assumption: str,
                   block: str, counts: bool) -> PerformanceView:
    """Build one performance row from a task table.

    Args:
        run: Loaded comparison.
        task: Task name.
        metric: Metric name stored in the table.
        label: Row label.
        assumption: Assumption text.
        block: Reason the row is unusable, or an empty string when it can be judged.
        counts: Whether the headline uses the row.

    Returns:
        The row. An unusable row has position ``unused``.
    """
    failed = _failed(run, task)
    series = _metric_series(run.table(task), metric)
    if failed or series is None or block:
        note = failed or block or "not computed"
        return PerformanceView(label, False, "unused", False, math.nan, math.nan, math.nan, math.nan, assumption, note)
    higher = _as_bool(series.get("higher_is_better", True))
    reference, estimate = _finite(series.get("reference_value")), _finite(series.get("estimate"))
    low, high = _finite(series.get("ci_low")), _finite(series.get("ci_high"))
    amount = _tolerance_amount(run, reference)
    bound = reference - amount if higher else reference + amount
    position = interval_position(estimate, low, high, bound, higher) if math.isfinite(amount) else "unused"
    return PerformanceView(label, True, position, counts, reference, estimate, low, high, assumption, "")


def _discount_ess(run: ComparisonRun, view: PerformanceView) -> PerformanceView:
    """Drop or widen the weighted row when the Kish effective sample is thin.

    Args:
        run: Loaded comparison.
        view: Weighted row before the effective-sample rule.

    Returns:
        The row the headline should see.
    """
    if not view.usable:
        return view
    ess = _scalar_of(run.table("weighted_performance", "summary"), "ess_ratio")
    poor, good = _threshold(run, "ess_ratio_poor", 0.10), _threshold(run, "ess_ratio_good", 0.50)
    if not math.isfinite(ess):
        return _unusable(view, "overlap was not computed")
    if ess < poor:
        return _unusable(view, "overlap too thin to estimate")
    note = "The interval resamples rows and holds the weights fixed."
    if ess < good and view.position == "inside" and _barely_inside(view, _tolerance_amount(run, view.reference)):
        note = "ESS is below 0.50 and the interval clears the bound by less than half the tolerance. " + note
        return _replace(view, position="crosses", note=note)
    return _replace(view, note=note)


def _cbpe_block(run: ComparisonRun) -> str:
    """Reason to drop CBPE, or an empty string when ECE is acceptable.

    Args:
        run: Loaded comparison.

    Returns:
        Drop reason. Empty when the calibration gate passes.
    """
    ece = _scalar_of(run.table("calibration"), "ece")
    if not math.isfinite(ece):
        return "calibration was not computed"
    if ece > _threshold(run, "ece_max", 0.05):
        return "scores are not calibrated on the reference"
    return ""


def _unusable(view: PerformanceView, note: str) -> PerformanceView:
    """Copy a row as dropped.

    Args:
        view: Original row.
        note: Why it was dropped.

    Returns:
        A row the headline ignores.
    """
    return _replace(view, usable=False, position="unused", counts=False, note=note)


def _replace(view: PerformanceView, **changes: object) -> PerformanceView:
    """Copy a performance row with fields replaced.

    Args:
        view: Original row.
        **changes: Dataclass fields to replace.

    Returns:
        The copied row.
    """
    data = view.__dict__.copy()
    data.update(changes)
    return PerformanceView(**data)


def _barely_inside(view: PerformanceView, amount: float) -> bool:
    """Whether an acceptable interval clears the bound by less than half the tolerance.

    Args:
        view: Row whose position is ``inside``.
        amount: Tolerance in metric units.

    Returns:
        True when the clearance is smaller than half the tolerance.
    """
    if not math.isfinite(amount) or amount <= 0:
        return False
    higher = "RMSE" not in view.label
    gap = (view.ci_low - (view.reference - amount)) if higher else ((view.reference + amount) - view.ci_high)
    return gap < amount / 2


def _rate_view(run: ComparisonRun) -> RateView | None:
    """Operating rate for a binary score, with a Wilson interval on the new rate.

    Args:
        run: Loaded comparison.

    Returns:
        The rate, or None when the problem is not binary or scores are absent.
    """
    if _problem(run) != "binary":
        return None
    table = run.table("prediction_drift")
    if table.empty or "positive_rate_target" not in table.columns:
        return None
    source = _scalar_of(table, "positive_rate_source")
    target = _scalar_of(table, "positive_rate_target")
    n_new = int(run.metadata.get("n_target") or 0)
    low, high = _rate_interval(target, n_new)
    tolerance = _optional_tolerance(run)
    return RateView(
        source, target, low, high, tolerance,
        _rate_position(source, low, high, tolerance),
        _scalar_of(table, "near_threshold_share_source"),
        _scalar_of(table, "near_threshold_share_target"),
        _scalar_of(table, "psi"), _scalar_of(table, "ks_d"),
        _scalar_of(table, "threshold"),
    )


def _rate_interval(target: float, n_new: int) -> tuple[float, float]:
    """Wilson interval for the new operating rate.

    Args:
        target: New positive rate.
        n_new: New-row count.

    Returns:
        Lower and upper bounds, or a pair of NaNs when the count is missing.
    """
    if n_new <= 0 or not math.isfinite(target):
        return math.nan, math.nan
    successes = min(max(target * n_new, 0.0), float(n_new))
    return wilson_interval(successes, n_new)


def _rate_position(source: float, low: float, high: float, tolerance: float | None) -> str:
    """Place the new-rate interval against the reference rate ± tolerance.

    Args:
        source: Reference positive rate.
        low: Lower end of the new-rate interval.
        high: Upper end of the new-rate interval.
        tolerance: Absolute band, or None when it was not fixed.

    Returns:
        ``shown``, ``inside``, ``outside``, or ``crosses``.
    """
    if tolerance is None or not math.isfinite(source) or not (math.isfinite(low) and math.isfinite(high)):
        return "shown"
    band_low, band_high = source - tolerance, source + tolerance
    if high < band_low or low > band_high:
        return "outside"
    if low >= band_low and high <= band_high:
        return "inside"
    return "crosses"


def _causes(run: ComparisonRun) -> tuple[list[Cause], int]:
    """Model-important features that are moderate or severe, capped at five.

    Args:
        run: Loaded comparison.

    Returns:
        The displayed causes and the count omitted past the cap.
    """
    features = run.features
    if features.empty or not {"feature", "severity", "important"} <= set(features.columns):
        return [], 0
    flagged = features[features["severity"].isin(["moderate", "severe"]) & features["important"].map(_as_bool)].copy()
    if flagged.empty:
        return [], 0
    share = pd.to_numeric(flagged.get("model_importance_share"), errors="coerce")
    flagged = flagged.assign(_share=share).sort_values("_share", ascending=False)
    causes = [_cause(row) for row in flagged.head(5).to_dict(orient="records")]
    return causes, max(len(flagged) - 5, 0)


def _cause(row: dict[str, object]) -> Cause:
    """One cause row.

    Args:
        row: Feature-assessment record.

    Returns:
        Display fields for that feature.
    """
    kind = str(row.get("type") or "")
    smd, cramers = _finite(row.get("smd")), _finite(row.get("cramers_v"))
    if kind == "continuous" and math.isfinite(smd):
        direction = f"new mean is {'higher' if smd > 0 else 'lower'} (SMD {smd:+.2f})"
    elif math.isfinite(cramers):
        direction = f"category mix changed (V {cramers:.2f})"
    else:
        direction = "distribution changed"
    missing = re.search(r"missing rate [^;]+", str(row.get("reasons") or ""))
    if missing:
        direction = f"{direction}; {missing.group(0)}"
    share = _finite(row.get("model_importance_share"))
    return Cause(
        str(row["feature"]), kind, direction, f"PSI {_metric_num(_finite(row.get('psi')))}",
        "–" if not math.isfinite(share) else f"{share:.0%}", str(row.get("severity") or ""),
    )


def _support_lines(run: ComparisonRun) -> list[str]:
    """Important features with more than 2% of new values outside the training 0.5–99.5% range.

    Args:
        run: Loaded comparison.

    Returns:
        One clause per such feature.
    """
    coverage, features = run.table("range_coverage"), run.features
    if coverage.empty or features.empty or "outside_share" not in coverage.columns:
        return []
    important = set(features.loc[features["important"].map(_as_bool), "feature"].astype(str))
    lines = []
    for _, row in coverage.iterrows():
        share = _finite(row.get("outside_share"))
        if str(row.get("feature")) in important and math.isfinite(share) and share > 0.02:
            lines.append(f"{row['feature']}: {_pct(share)} of new values are outside the training 0.5–99.5% range.")
    return lines


def _shap_clause(run: ComparisonRun) -> str:
    """Clause when the ranking of mean absolute SHAP changed.

    Args:
        run: Loaded comparison.

    Returns:
        The clause, or an empty string when rank correlation stays at or above 0.80.
    """
    rho = _scalar_of(run.table("shap_drift", "summary"), "importance_rank_rho")
    cutoff = _threshold(run, "importance_rank_rho", 0.80)
    if not math.isfinite(rho) or rho >= cutoff:
        return ""
    return f"Rank correlation of mean |SHAP| is {rho:.2f}, below {cutoff:.2f}."


def _bbse_line(run: ComparisonRun, causes: bool) -> str:
    """Prior estimate when the important features held still and the operating rate moved.

    Args:
        run: Loaded comparison.
        causes: Whether any important feature earned a cause row.

    Returns:
        The sentence, or an empty string when BBSE does not belong on the page.
    """
    if causes or _problem(run) != "binary":
        return ""
    shift, pred = run.table("label_shift"), run.table("prediction_drift")
    h = _scalar_of(pred, "positive_rate_cohens_h_abs")
    if shift.empty or not math.isfinite(h) or h < _threshold(run, "cohens_h_small", 0.20):
        return ""
    source, estimate = _scalar_of(shift, "source_prevalence"), _scalar_of(shift, "target_prevalence_estimate")
    low, high = _scalar_of(shift, "ci_low"), _scalar_of(shift, "ci_high")
    return (
        f"Class prior under P(x|y) unchanged: {_pct(source)} → {_pct(estimate)} "
        f"[{_pct(low)}–{_pct(high)}]. Use this only as a reading of the rate change. "
        "The feature margins did not earn a cause row."
    )


def _validity_lines(run: ComparisonRun, metrics: Sequence[PerformanceView], in_sample: bool) -> list[str]:
    """Overlap, calibration, and task failures.

    Args:
        run: Loaded comparison.
        metrics: Performance rows. Present so the section stays tied to the judged rows.
        in_sample: Whether score tests used the fitting rows.

    Returns:
        Lines for the validity section.
    """
    usable = sum(item.counts and item.usable for item in metrics)
    verb = "enters" if usable == 1 else "enter"
    lines = [f"{usable} performance estimate{'s' if usable != 1 else ''} {verb} the headline."]
    if in_sample:
        lines.append("Score, calibration, and performance tests used one labelled file. Those references are in-sample.")
    lines.extend(_validity_diagnostics(run))
    failed = run.metadata.get("failed_tasks") or {}
    if isinstance(failed, dict) and failed:
        lines.extend(f"{name} failed: {message}" for name, message in failed.items())
    for warning in run.decision.get("warnings") or []:
        lines.append(str(warning))
    return lines


def _validity_diagnostics(run: ComparisonRun) -> list[str]:
    """ESS, calibration, domain AUC, and the realized-versus-CBPE note.

    Args:
        run: Loaded comparison.

    Returns:
        Diagnostic lines that were computable.
    """
    lines = []
    summary = run.table("weighted_performance", "summary")
    ess, ess_n = _scalar_of(summary, "ess_ratio"), _scalar_of(summary, "ess")
    if math.isfinite(ess):
        rows = f"{ess_n:.0f}" if math.isfinite(ess_n) else "–"
        lines.append(f"Kish ESS ratio {ess:.2f} ({rows} effective rows). Below 0.10 the weighted estimate is dropped.")
    ece = _scalar_of(run.table("calibration"), "ece")
    slope = _scalar_of(run.table("calibration"), "calibration_slope")
    if math.isfinite(ece):
        lines.append(f"Reference ECE {_metric_num(ece)}, calibration slope {_metric_num(slope)}. CBPE is dropped above ECE 0.05.")
    auc = _scalar_of(run.table("domain_classifier"), "auc")
    if math.isfinite(auc):
        lines.append(f"Domain-classifier AUC {auc:.3f}. This is the overlap summary, not a second performance number.")
    realized = _metric_series(run.table("cbpe"), "auc")
    if realized is not None and math.isfinite(_finite(realized.get("source_realized"))):
        lines.append(
            f"Realized reference AUC {_metric_num(_finite(realized.get('source_realized')))}. "
            "CBPE compares estimated AUC with estimated AUC."
        )
    return lines


def _confirm_lines(run: ComparisonRun, metrics: Sequence[PerformanceView]) -> list[str]:
    """Labelled sample size and the archive line.

    Args:
        run: Loaded comparison.
        metrics: Performance rows, used for the reference AUC.

    Returns:
        Lines for the confirmation section.
    """
    lines = [_sample_line(run, metrics)]
    lines.append(
        "Archive decision.json, metadata.json, checksums of each input, the model file, "
        "and the code commit with this page."
    )
    lines.append("The model owner decides. A later rerun is the monitoring step. This repository does not schedule one.")
    return lines


def _sample_line(run: ComparisonRun, metrics: Sequence[PerformanceView]) -> str:
    """Sample-size sentence for the tolerance.

    Args:
        run: Loaded comparison.
        metrics: Performance rows.

    Returns:
        The sentence. Regression has no Hanley–McNeil n.
    """
    tolerance = float(_config(run).get("performance_tolerance", 0.02))
    if _problem(run) == "regression":
        return (
            f"Label enough new rows that the interval for RMSE can resolve a relative change of {tolerance:.3f}. "
            "The metric that overturns the headline is RMSE on the worse side of that tolerance."
        )
    auc = _reference_auc(metrics)
    prevalence = _scalar_of(run.table("label_shift"), "source_prevalence")
    if not (math.isfinite(auc) and math.isfinite(prevalence) and 0 < auc < 1 and 0 < prevalence < 1):
        return (
            "Label a sample of the new population. The metric that overturns the headline is AUC "
            f"on the worse side of the tolerance ({tolerance:.3f}). Prevalence was not available, so n is not computed."
        )
    return _auc_sample_sentence(auc, prevalence, tolerance)


def _auc_sample_sentence(auc: float, prevalence: float, tolerance: float) -> str:
    """Sample-size sentence once AUC and prevalence are known.

    Args:
        auc: Reference AUC.
        prevalence: Reference positive rate.
        tolerance: Absolute AUC tolerance.

    Returns:
        The sentence, including the case where two million rows are not enough.
    """
    n = labelled_sample_size(auc, prevalence, tolerance)
    if n is None:
        return (
            f"At prevalence {_pct(prevalence)} and AUC {_metric_num(auc)}, two million labelled rows "
            f"still leave a 95% AUC interval wider than ±{tolerance:.3f}."
        )
    return (
        f"About {n:,} labelled new rows (prevalence {_pct(prevalence)}, AUC {_metric_num(auc)}) "
        f"make a 95% AUC interval no wider than ±{tolerance:.3f} (Hanley and McNeil 1982). "
        "That resolves one AUC. It does not size a test of the difference from the reference. "
        "The result that overturns the headline is an AUC on the worse side of the tolerance."
    )


def _reference_auc(metrics: Sequence[PerformanceView]) -> float:
    """First finite AUC reference on the page.

    Args:
        metrics: Performance rows.

    Returns:
        The reference, or NaN.
    """
    for item in metrics:
        if item.label == "Weighted AUC" and math.isfinite(item.reference):
            return item.reference
    for item in metrics:
        if item.label == "CBPE AUC" and math.isfinite(item.reference):
            return item.reference
    return math.nan


def _anchor(run: ComparisonRun, rate: RateView | None) -> tuple[tuple[str, str], ...]:
    """Facts that define what 'expected' means for this run.

    Args:
        run: Loaded comparison.
        rate: Operating rate, or None.

    Returns:
        Label and value pairs.
    """
    meta = run.metadata
    tolerance = float(_config(run).get("performance_tolerance", 0.02))
    kind = "relative RMSE" if _problem(run) == "regression" else "absolute AUC"
    rate_text = "not set" if _optional_tolerance(run) is None else f"±{_metric_num(_optional_tolerance(run) or 0)}"
    rows = [
        ("Reference rows", f"{int(meta.get('n_source') or 0):,}"),
        ("New rows", f"{int(meta.get('n_target') or 0):,}"),
        ("Performance tolerance", f"{tolerance:.3f} ({kind}, worse than reference)"),
        ("Operating-rate tolerance", rate_text),
        ("Reference", "evaluation set" if not _in_sample(run) else "labelled file, in-sample"),
        ("Baseline rounds", str(run.decision.get("baseline_rounds") or "–")),
    ]
    if meta.get("n_evaluation"):
        rows.insert(2, ("Evaluation rows", f"{int(meta['n_evaluation']):,}"))
    if rate is not None and math.isfinite(rate.threshold):
        rows.insert(3, ("Threshold", _metric_num(rate.threshold)))
    return tuple(rows)


def _sections(judgement: ExpectationJudgement, run: ComparisonRun, detailed_name: str) -> list[Section]:
    """Render the seven sections of the expectation report.

    Args:
        judgement: Headline and rows.
        run: Loaded comparison, used for plots.
        detailed_name: Detailed-report filename.

    Returns:
        Sections in reading order.
    """
    return [
        Section("expectation", "Expectation", _expectation_html(judgement)),
        Section("observed", "Observed score", _observed_html(judgement, run)),
        Section("estimates", "Estimated performance", _estimates_html(judgement)),
        Section("causes", "Why it moved", _causes_html(judgement, run)),
        Section("validity", "Whether the estimate can be used", _validity_html(judgement, run)),
        Section("confirm", "What would confirm it", _list_html(judgement.confirm)),
        Section("scope", "What this page leaves out", _scope_html(detailed_name, judgement)),
    ]


def _expectation_html(judgement: ExpectationJudgement) -> str:
    """Headline, anchor, and action.

    Args:
        judgement: Report model.

    Returns:
        HTML for the first section.
    """
    level = judgement.headline
    paragraphs = "".join(f"<p>{html.escape(line)}</p>" for line in judgement.paragraphs)
    anchor = _table(("Item", "Value"), judgement.anchor)
    return (
        f'<div class="verdict {level}"><div class="verdict-label">Expectation</div>'
        f'<div class="verdict-text">{html.escape(_HEADLINE_TEXT[level])}</div>'
        f'<div class="verdict-rule">{html.escape(judgement.action)}</div></div>'
        f"{paragraphs}{anchor}"
    )


def _observed_html(judgement: ExpectationJudgement, run: ComparisonRun) -> str:
    """Score histogram and the operating rate.

    Args:
        judgement: Report model.
        run: Loaded comparison.

    Returns:
        HTML for the observed section.
    """
    rate = judgement.rate
    if rate is None:
        failed = _failed(run, "prediction_drift")
        reason = failed or "Scores were not computed."
        return f"<p>{html.escape(reason)}</p>"
    band = "A rate tolerance was not fixed, so the rate does not set the headline."
    if rate.tolerance is not None:
        band = f"Rate band position: {rate.position}."
    text = (
        f"<p>Positive rate {_pct(rate.source)} → <b>{_pct(rate.target)}</b> "
        f"(95% interval for the new rate {_pct(rate.ci_low)}–{_pct(rate.ci_high)}). {html.escape(band)}</p>"
        f"<p>Score PSI {_metric_num(rate.psi)}, KS D {_metric_num(rate.ks)}. "
        f"Share within 0.05 of the threshold: {_pct(rate.near_source)} → {_pct(rate.near_target)}.</p>"
    )
    return _pair(run, "prediction_drift") + _score_plot(run) + text


def _estimates_html(judgement: ExpectationJudgement) -> str:
    """Performance table and the optional prior sentence.

    Args:
        judgement: Report model.

    Returns:
        HTML for the estimates section.
    """
    rows = [_metric_cells(item) for item in judgement.metrics]
    table = _table(("Estimate", "Reference", "New", "95% interval", "Versus bound", "Assumption"), rows)
    extra = f"<p>{html.escape(judgement.bbse)}</p>" if judgement.bbse else ""
    rule = (
        "<p>Acceptable means on the better side of reference − tolerance for AUC, "
        "or reference + tolerance × |reference| for RMSE. "
        "Outside means the whole interval is on the worse side. "
        "A crossing interval is imprecise, not a failure.</p>"
    )
    return rule + table + extra


def _causes_html(judgement: ExpectationJudgement, run: ComparisonRun) -> str:
    """Cause table, support clauses, and at most three distribution plots.

    Args:
        judgement: Report model.
        run: Loaded comparison.

    Returns:
        HTML for the cause section.
    """
    if not judgement.causes:
        lead = "<p>No model-important feature moved beyond the significance, baseline, and effect-size gates.</p>"
    else:
        rows = [(item.feature, item.direction, item.effect, item.share, item.severity) for item in judgement.causes]
        lead = _table(("Feature", "Direction", "Effect", "Share of mean |SHAP|", "Severity"), rows)
        if judgement.extra_causes:
            lead += f"<p>{judgement.extra_causes} other important features also cleared those gates.</p>"
    if judgement.low_importance:
        lead += f"<p>{judgement.low_importance} low-importance features moved. They are outside the 80% SHAP set.</p>"
    if judgement.support:
        lead += "<ul>" + "".join(f"<li>{html.escape(line)}</li>" for line in judgement.support) + "</ul>"
    if judgement.shap_clause:
        lead += f"<p>{html.escape(judgement.shap_clause)}</p>"
    names = [item.feature for item in judgement.causes[:3]]
    kinds = {item.feature: item.kind for item in judgement.causes}
    continuous = [name for name in names if kinds.get(name) == "continuous"]
    categorical = [name for name in names if name not in continuous]
    return _pair(run, "continuous", "shap_drift") + lead + plot_ecdfs(run, continuous) + plot_categories(run, categorical)


def _scope_html(detailed_name: str, judgement: ExpectationJudgement) -> str:
    """What remains in the detailed report, and which baseline rounds were used.

    Args:
        detailed_name: Detailed-report filename.
        judgement: Report model. Used for the baseline line already stored on the anchor.

    Returns:
        HTML for the closing section.
    """
    del judgement
    return (
        "<p>Marginal tests, kernel tests, dependence tests, support tests, and every mild signal are in "
        f"<code>{html.escape(detailed_name)}</code>. They explain columns that did not earn a cause row. "
        "They do not change the headline on this page.</p>"
        "<p>The headline compares the valid performance estimates, and the operating rate when its "
        "tolerance was fixed, with those tolerances. The size-matched baseline decides which feature "
        "movements are sampling noise. Cause rows are the model-important features that also clear "
        "adjusted significance and a moderate effect size.</p>"
    )


def _validity_html(judgement: ExpectationJudgement, run: ComparisonRun) -> str:
    """Validity lines, plus the reliability curve when calibration fails the CBPE gate.

    Args:
        judgement: Report model.
        run: Loaded comparison.

    Returns:
        HTML for the validity section.
    """
    body = _list_html(judgement.validity)
    ece = _scalar_of(run.table("calibration"), "ece")
    if math.isfinite(ece) and ece > _threshold(run, "ece_max", 0.05):
        body += plot_calibration(run)
    return body


def _list_html(lines: Sequence[str]) -> str:
    """A paragraph list.

    Args:
        lines: Plain sentences.

    Returns:
        HTML. An empty sequence becomes one sentence.
    """
    if not lines:
        return "<p>Nothing further was recorded.</p>"
    return "<ul>" + "".join(f"<li>{html.escape(line)}</li>" for line in lines) + "</ul>"


def _metric_cells(item: PerformanceView) -> tuple[str, str, str, str, str, str]:
    """Table cells for one performance row.

    Args:
        item: Performance row.

    Returns:
        Six plain-text cells.
    """
    versus = {
        "inside": "inside",
        "outside": "worse than the bound",
        "crosses": "crosses the bound",
        "point_only": "no interval",
        "unused": item.note or "not used",
    }[item.position]
    assumption = item.assumption if not item.note or item.position == "unused" else f"{item.assumption} {item.note}"
    return (
        item.label, _metric_num(item.reference), _metric_num(item.estimate),
        _interval(item.ci_low, item.ci_high), versus, assumption,
    )


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """HTML table with escaped cells.

    Args:
        headers: Column titles.
        rows: Body rows of plain text.

    Returns:
        The table markup.
    """
    head = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>" for row in rows
    )
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _score_plot(run: ComparisonRun) -> str:
    """Overlaid score histogram with the threshold marked.

    Args:
        run: Loaded comparison.

    Returns:
        An image tag, or an empty string when plot data was not stored.
    """
    prediction = run.plot_data.get("prediction_drift")
    if not prediction or "bins" not in prediction:
        return ""
    bins = np.asarray(prediction["bins"], float)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    centers, width = (bins[:-1] + bins[1:]) / 2, bins[1] - bins[0]
    ax.bar(centers, prediction["source_hist"], width, color=SOURCE_COLOR, alpha=0.55, label="reference")
    ax.bar(centers, prediction["target_hist"], width, color=TARGET_COLOR, alpha=0.55, label="new")
    if prediction.get("threshold") is not None:
        ax.axvline(prediction["threshold"], color="black", ls="--", lw=1, label="threshold")
    ax.set_xlabel("score")
    ax.set_ylabel("density")
    ax.set_title("Score distribution")
    ax.legend(frameon=False, fontsize=8)
    _style_axes(ax)
    scores = list(bins) + ([prediction["threshold"]] if prediction.get("threshold") is not None else [])
    ax.set_xlim(*_axis_limits(scores))
    ax.set_ylim(*_axis_limits(list(prediction["source_hist"]) + list(prediction["target_hist"])))
    fig.tight_layout()
    return figure_html(fig, "Score distribution, reference and new, with the production threshold")


def _pair(run: ComparisonRun, *tasks: str) -> str:
    """One line naming the populations a section used.

    Args:
        run: Loaded comparison.
        *tasks: Task names whose pair labels should be shown.

    Returns:
        A paragraph, or an empty string.
    """
    pairs = run.metadata.get("task_pairs") or {}
    labels = []
    for task in tasks:
        label = pairs.get(task)
        if label and label not in labels:
            labels.append(str(label))
    if not labels:
        return ""
    return f"<p class='pair'>Pair: {html.escape(' · '.join(labels))}.</p>"


def _subtitle(run: ComparisonRun) -> str:
    """One-line context under the report title.

    Args:
        run: Loaded comparison.

    Returns:
        Reference size, new size, and the run timestamp when present.
    """
    meta = run.metadata
    created = str(meta.get("created_at") or "").strip()
    text = f"Reference {int(meta.get('n_source') or 0):,} rows · new {int(meta.get('n_target') or 0):,} rows"
    return f"{text} · {created}" if created else text


def _low_importance(run: ComparisonRun) -> int:
    """Count of non-important features with moderate or severe severity.

    Args:
        run: Loaded comparison.

    Returns:
        The count. Zero when the assessment table is absent.
    """
    features = run.features
    if features.empty or "important" not in features.columns:
        return 0
    mask = ~features["important"].map(_as_bool) & features["severity"].isin(["moderate", "severe"])
    return int(mask.sum())


def _in_sample(run: ComparisonRun) -> bool:
    """Whether the scored reference is the fitting data.

    Args:
        run: Loaded comparison.

    Returns:
        True for single-file mode, and when the mode was not recorded.
    """
    mode = run.metadata.get("reference_mode") or run.decision.get("reference_mode")
    if mode == "paired":
        return False
    if mode == "single":
        return True
    pair = str((run.metadata.get("task_pairs") or {}).get("prediction_drift") or "")
    return "evaluation" not in pair


def _config(run: ComparisonRun) -> dict[str, object]:
    """Config dict stored on the comparison, or an empty dict.

    Args:
        run: Loaded comparison.

    Returns:
        The config mapping.
    """
    config = run.metadata.get("config") or {}
    return config if isinstance(config, dict) else {}


def _threshold(run: ComparisonRun, name: str, default: float) -> float:
    """One threshold from the saved config.

    Args:
        run: Loaded comparison.
        name: Threshold field.
        default: Value when the field is absent.

    Returns:
        The threshold.
    """
    thresholds = _config(run).get("thresholds") or {}
    if not isinstance(thresholds, dict) or thresholds.get(name) is None:
        return default
    return float(thresholds[name])


def _optional_tolerance(run: ComparisonRun) -> float | None:
    """Operating-rate tolerance, or None when it was not fixed.

    Args:
        run: Loaded comparison.

    Returns:
        The tolerance.
    """
    value = _config(run).get("rate_tolerance")
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return None
    return float(value)


def _tolerance_amount(run: ComparisonRun, reference: float) -> float:
    """Tolerance in metric units. Regression uses a relative tolerance.

    Args:
        run: Loaded comparison.
        reference: Reference metric.

    Returns:
        The absolute amount, or NaN when the reference is missing.
    """
    tolerance = float(_config(run).get("performance_tolerance", 0.02))
    if not math.isfinite(reference):
        return math.nan
    if _problem(run) == "regression":
        return tolerance * abs(reference)
    return tolerance


def _problem(run: ComparisonRun) -> str:
    """Problem type, defaulting to binary.

    Args:
        run: Loaded comparison.

    Returns:
        ``binary`` or ``regression``.
    """
    value = _config(run).get("problem_type") or run.metadata.get("problem_type") or "binary"
    return str(value)


def _failed(run: ComparisonRun, task: str) -> str:
    """Failure message for one task.

    Args:
        run: Loaded comparison.
        task: Task name.

    Returns:
        The message, or an empty string when the task did not fail.
    """
    failed = run.metadata.get("failed_tasks") or {}
    if not isinstance(failed, dict) or not failed.get(task):
        return ""
    return str(failed[task])


def _metric_series(table: pd.DataFrame, metric: str) -> pd.Series | None:
    """The row of a performance table for one metric.

    Args:
        table: Task table.
        metric: Metric name.

    Returns:
        The row, or None when it is absent.
    """
    if table.empty or "metric" not in table.columns:
        return None
    matched = table[table["metric"].astype(str) == metric]
    if matched.empty:
        return None
    return matched.iloc[0]


def _scalar_of(table: pd.DataFrame, column: str) -> float:
    """First finite value of a column, or NaN.

    Args:
        table: Task table.
        column: Column name.

    Returns:
        The value.
    """
    if table is None or table.empty or column not in table.columns:
        return math.nan
    value = table[column].iloc[0]
    return _finite(value)


def _finite(value: object) -> float:
    """Float value, or NaN when it is missing.

    Args:
        value: Cell value.

    Returns:
        A float.
    """
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def _as_bool(value: object) -> bool:
    """Interpret a CSV or in-memory boolean.

    Args:
        value: Cell value.

    Returns:
        The boolean.
    """
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def _metric_num(value: float) -> str:
    """Three-decimal metric, or a dash when missing.

    Args:
        value: Number to format.

    Returns:
        The text.
    """
    return "–" if not math.isfinite(value) else f"{value:.3f}"


def _interval(low: float, high: float) -> str:
    """Closed interval text.

    Args:
        low: Lower end.
        high: Upper end.

    Returns:
        The text, or a dash when either end is missing.
    """
    if not (math.isfinite(low) and math.isfinite(high)):
        return "–"
    return f"{low:.3f}–{high:.3f}"


def _pct(value: float) -> str:
    """Percent with one decimal, or a dash.

    Args:
        value: Proportion.

    Returns:
        The text.
    """
    return "–" if not math.isfinite(value) else f"{value:.1%}"
