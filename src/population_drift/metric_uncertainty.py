"""Sampling uncertainty for binary metrics from a confusion matrix.

The labelled operating point is the setting in Urlus et al. (2023): one trained
classifier, one threshold, and a finite test set. The four cells are a
multinomial sample. A 95% interval for one metric is the Wilks profile
likelihood-ratio interval, chi-square with one degree of freedom. The
bivariate-normal standard error is the linear approximation in their equations
13–15 and 44–45; it is stored and marked invalid when ``n p (1-p) < 10`` or the
estimate is 0 or 1, which is where that approximation fails.

The quoted interval is the Wilks interval. These intervals describe sampling
uncertainty on the labelled rows that produced the confusion matrix. They are
not an estimate of performance on an unlabelled population.

Urlus, R. E. Q., Baak, M. A., Collot, S., and Fridman Rojas, I. (2023).
Pointwise sampling uncertainties on the Precision-Recall curve. Proceedings of
AISTATS, PMLR 206. Reference implementation:
https://github.com/RUrlus/ModelMetricUncertainty
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import chi2

logger = logging.getLogger("population_drift.metric_uncertainty")

WILKS_METHOD = "wilks_profile_chi2_1"
_NORMAL_COUNT = 10.0
_EDGE = 1e-12
_PROFILE_TOLERANCE = 1e-4


def _as_count(name: str, value: int) -> int:
    """Return ``value`` as an int.

    Args:
        name: Argument name used in the error.
        value: Count to check.

    Returns:
        The count as a built-in int.

    Raises:
        TypeError: If ``value`` is a bool or not an integer.
        ValueError: If ``value`` is negative.
    """
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be a non-negative integer.")
    if int(value) < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value}.")
    return int(value)


def _require_confidence(confidence: float) -> float:
    """Return a confidence level inside (0, 1).

    Args:
        confidence: Requested confidence level.

    Returns:
        ``confidence`` as a float.

    Raises:
        ValueError: If ``confidence`` is outside (0, 1).
    """
    if not (0.0 < float(confidence) < 1.0):
        raise ValueError(f"confidence must be in (0, 1), got {confidence}.")
    return float(confidence)


def _require_unit_interval(name: str, value: float) -> float:
    """Return a finite value in [0, 1].

    Args:
        name: Argument name used in the error.
        value: Candidate probability.

    Returns:
        ``value`` as a float.

    Raises:
        ValueError: If ``value`` is not a finite number in [0, 1].
    """
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1], got {value}.")
    return number


def _binomial_log_likelihood(successes: int, trials: int, probability: float) -> float:
    """Binomial log-likelihood, up to the combinatorial constant.

    Args:
        successes: Observed successes.
        trials: Number of trials.
        probability: Success probability in [0, 1].

    Returns:
        The log-likelihood, or ``-inf`` when a positive count has probability 0.
    """
    if successes > 0 and probability <= 0.0:
        return -math.inf
    failures = trials - successes
    if failures > 0 and probability >= 1.0:
        return -math.inf
    total = 0.0
    if successes:
        total += successes * math.log(probability)
    if failures:
        total += failures * math.log(1.0 - probability)
    return total


def _min_over(function: Callable[[float], float], low: float, high: float) -> float:
    """Minimum of a univariate function on a closed interval.

    Args:
        function: Function of one float. May return ``inf``.
        low: Left endpoint.
        high: Right endpoint.

    Returns:
        The smallest value found. ``inf`` when no finite value exists.
    """
    if high <= low:
        return math.inf
    grid = np.linspace(low, high, 48)
    values = [float(function(float(point))) for point in grid]
    best_index = int(np.argmin(values))
    best = values[best_index]
    if math.isinf(best):
        return math.inf
    span = float(grid[1] - grid[0])
    left = max(low, float(grid[best_index]) - span)
    right = min(high, float(grid[best_index]) + span)
    refined = minimize_scalar(lambda point: float(function(float(point))), bounds=(left, right),
                              method="bounded", options={"xatol": 1e-12})
    candidate = float(refined.fun) if refined.success else math.inf
    return min(best, candidate)


def _profile_bound(statistic: Callable[[float], float], estimate: float, critical: float, *, lower: bool) -> float:
    """One endpoint of a profile interval on [0, 1].

    Args:
        statistic: Profile log-likelihood ratio. Zero at ``estimate``.
        estimate: Point estimate in [0, 1].
        critical: Chi-square critical value.
        lower: Whether to return the lower endpoint.

    Returns:
        The endpoint. A boundary of 0 or 1 means that side stays inside the interval.
    """
    if lower and estimate <= 0.0:
        return 0.0
    if not lower and estimate >= 1.0:
        return 1.0
    edge = _EDGE if lower else 1.0 - _EDGE
    if statistic(edge) <= critical:
        return 0.0 if lower else 1.0
    left, right = (edge, estimate) if lower else (estimate, edge)
    for _ in range(60):
        mid = 0.5 * (left + right)
        outside = statistic(mid) > critical
        if lower:
            left, right = (mid, right) if outside else (left, mid)
        else:
            left, right = (left, mid) if outside else (mid, right)
    return 0.5 * (left + right)


def _statistic_interval(estimate: float, statistic: Callable[[float], float], confidence: float,
                       metric: str) -> tuple[float, float]:
    """Profile-likelihood interval for a statistic that is zero at the estimate.

    Args:
        estimate: Point estimate in [0, 1], or NaN when the metric is undefined.
        statistic: Profile log-likelihood ratio.
        confidence: Confidence level in (0, 1).
        metric: Metric name used in the error log.

    Returns:
        Lower and upper bounds. Both are NaN when ``estimate`` is NaN or the
        profile at the estimate is not zero.
    """
    if not math.isfinite(estimate):
        return math.nan, math.nan
    fitted = statistic(float(estimate))
    if fitted > _PROFILE_TOLERANCE:
        logger.error("Wilks profile at the %s estimate is %.3g; interval omitted.", metric, fitted)
        return math.nan, math.nan
    critical = float(chi2.ppf(confidence, 1))
    low = _profile_bound(statistic, estimate, critical, lower=True)
    high = _profile_bound(statistic, estimate, critical, lower=False)
    return low, high


class BinomialProportion:
    """Wilks interval for a binomial proportion.

    This is the one-parameter case of the multinomial profile in Urlus et al.
    (2023). Recall, given the positive rows, and the false-positive rate, given
    the negative rows, reduce to this interval. The predicted-positive rate does
    too, because that rate has no class label.

    Args:
        successes: Number of observed successes.
        trials: Number of trials.

    Raises:
        TypeError: If a count is not an integer.
        ValueError: If a count is negative or ``successes`` exceeds ``trials``.
    """

    def __init__(self, successes: int, trials: int) -> None:
        self.successes = _as_count("successes", successes)
        self.trials = _as_count("trials", trials)
        if self.trials == 0:
            raise ValueError("trials must be positive.")
        if self.successes > self.trials:
            raise ValueError(f"successes ({self.successes}) exceeds trials ({self.trials}).")

    def estimate(self) -> float:
        """Return the maximum-likelihood proportion.

        Returns:
            Successes divided by trials.
        """
        return self.successes / self.trials

    def profile_statistic(self, probability: float) -> float:
        """Return the profile log-likelihood ratio at ``probability``.

        Args:
            probability: Hypothesised success probability in [0, 1].

        Returns:
            ``-2`` times the log likelihood ratio against the maximum-likelihood
            estimate. Zero at the estimate.

        Raises:
            ValueError: If ``probability`` is not a finite number in [0, 1].
        """
        probability = _require_unit_interval("probability", probability)
        maximum = _binomial_log_likelihood(self.successes, self.trials, self.estimate())
        constrained = _binomial_log_likelihood(self.successes, self.trials, probability)
        return max(0.0, -2.0 * (constrained - maximum))

    def interval(self, confidence: float = 0.95) -> tuple[float, float]:
        """Return the Wilks confidence interval.

        Args:
            confidence: Confidence level in (0, 1).

        Returns:
            Lower and upper bounds in [0, 1].

        Raises:
            ValueError: If ``confidence`` is outside (0, 1).
        """
        confidence = _require_confidence(confidence)
        return _statistic_interval(self.estimate(), self.profile_statistic, confidence, "binomial")

    def normal_standard_error(self) -> tuple[float, bool]:
        """Return the binomial standard error and whether it is usable.

        Returns:
            Standard error, and whether ``n p (1-p) >= 10`` with ``p`` away from 0 and 1.
        """
        probability = self.estimate()
        variance = probability * (1.0 - probability) / self.trials
        count = self.trials * probability * (1.0 - probability)
        return math.sqrt(variance), _normal_flag(count, probability)


@dataclass(frozen=True)
class MetricInterval:
    """One operating-point metric and its sampling interval.

    Args:
        metric: ``precision``, ``recall``, ``fpr``, or ``f1``.
        estimate: Point estimate. NaN when the denominator is zero.
        ci_low: Lower Wilks bound. NaN when the interval was not computed.
        ci_high: Upper Wilks bound. NaN when the interval was not computed.
        confidence: Confidence level used for the interval.
        method: ``wilks_profile_chi2_1``, or ``undefined`` when no interval exists.
        normal_se: Bivariate-normal or delta-method standard error. NaN when undefined.
        normal_valid: Whether that standard error meets the paper's count rule.
    """

    metric: str
    estimate: float
    ci_low: float
    ci_high: float
    confidence: float
    method: str
    normal_se: float
    normal_valid: bool


def _prediction_arrays(labels: np.ndarray, scores: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Validate labels, scores, and the decision threshold.

    Args:
        labels: Candidate label array.
        scores: Candidate score array.
        threshold: Decision threshold.

    Returns:
        One-dimensional labels, finite scores, and the threshold.

    Raises:
        ValueError: If the arrays are empty, misaligned, non-binary, or non-finite.
    """
    label_values = np.asarray(labels)
    score_values = np.asarray(scores, dtype=float)
    if label_values.ndim != 1 or score_values.ndim != 1 or label_values.shape != score_values.shape:
        raise ValueError("labels and scores must be one-dimensional and the same length.")
    if label_values.size == 0:
        raise ValueError("labels and scores must be non-empty.")
    if not np.isfinite(score_values).all() or not math.isfinite(float(threshold)):
        raise ValueError("scores and threshold must be finite.")
    observed = set(np.unique(label_values).tolist())
    if not observed <= {0, 1, 0.0, 1.0}:
        raise ValueError(f"labels must be binary 0/1, found {sorted(observed)}.")
    return label_values, score_values, float(threshold)


def _rate(numerator: int, denominator: int) -> float:
    """Return a ratio, or NaN when the denominator is zero.

    Args:
        numerator: Numerator count.
        denominator: Denominator count.

    Returns:
        The ratio, or NaN.
    """
    if denominator == 0:
        return math.nan
    return numerator / denominator


def _likelihood_ratio(counts: tuple[int, int, int, int], probs: tuple[float, float, float, float],
                      maximum_ll: float) -> float:
    """Profile log-likelihood ratio of one constrained probability vector.

    Args:
        counts: Observed confusion-matrix counts.
        probs: Constrained probabilities in the same order.
        maximum_ll: Log-likelihood at the unconstrained maximum.

    Returns:
        The non-negative ratio, or ``inf`` when the constraint is impossible.
    """
    constrained = _cell_log_likelihood(counts, probs)
    if math.isinf(constrained):
        return math.inf
    gap = constrained - maximum_ll
    if gap > 1e-6:
        logger.error("Constrained likelihood exceeded the MLE by %.3g.", gap)
        return math.inf
    return max(0.0, -2.0 * gap)


def _cell_log_likelihood(counts: tuple[int, int, int, int], probs: tuple[float, float, float, float]) -> float:
    """Multinomial log-likelihood, up to the combinatorial constant.

    Args:
        counts: True positive, false positive, true negative, false negative.
        probs: Probabilities in the same order.

    Returns:
        The log-likelihood, or ``-inf`` when a positive count has probability <= 0.
    """
    total = 0.0
    for count, prob in zip(counts, probs):
        if count == 0:
            continue
        if prob <= 0.0:
            return -math.inf
        total += count * math.log(prob)
    return total


def _pr_probabilities(precision: float, recall: float, true_negative: int, n: int,
                      active: int) -> tuple[float, ...] | None:
    """Constrained probabilities at a fixed precision and recall.

    Equation 6 of Urlus et al. (2023) gives the profiled true-positive
    probability. The true-negative probability at that maximum is the observed
    true-negative share.

    Args:
        precision: Hypothesised precision in (0, 1).
        recall: Hypothesised recall in (0, 1).
        true_negative: Observed true-negative count.
        n: Total count.
        active: Count that is not a true negative.

    Returns:
        Probabilities ``(tp, fp, tn, fn)``, or None when the point is not interior.
    """
    if not (0.0 < precision < 1.0 and 0.0 < recall < 1.0):
        return None
    gamma = 1.0 / precision + 1.0 / recall - 1.0
    p_tp = active / (gamma * n)
    p_fp = ((1.0 - precision) / precision) * p_tp
    p_fn = ((1.0 - recall) / recall) * p_tp
    return (p_tp, p_fp, true_negative / n, p_fn)


def _precision_for_f1(f1: float, recall: float) -> float | None:
    """Precision implied by an F1 score and a recall.

    Args:
        f1: Hypothesised F1 score in (0, 1).
        recall: Hypothesised recall in (0, 1).

    Returns:
        The matching precision, or None when it falls outside (0, 1).
    """
    denominator = 2.0 * recall - f1
    if denominator <= 0.0:
        return None
    precision = f1 * recall / denominator
    if not 0.0 < precision < 1.0:
        return None
    return precision


def _normal_flag(variance_count: float, estimate: float) -> bool:
    """Return whether the normal approximation is in its stated range.

    Args:
        variance_count: ``n p (1-p)`` for the binomial margin of this metric.
        estimate: Point estimate.

    Returns:
        True when the count is at least 10 and the estimate is not 0 or 1.
    """
    return math.isfinite(variance_count) and variance_count >= _NORMAL_COUNT and 0.0 < estimate < 1.0


class ConfusionMatrix:
    """Confusion-matrix metrics and their Wilks intervals.

    Args:
        true_positive: True positive count.
        false_positive: False positive count.
        true_negative: True negative count.
        false_negative: False negative count.

    Raises:
        TypeError: If a count is not an integer.
        ValueError: If a count is negative or the matrix is empty.
    """

    def __init__(self, true_positive: int, false_positive: int, true_negative: int, false_negative: int) -> None:
        self.true_positive = _as_count("true_positive", true_positive)
        self.false_positive = _as_count("false_positive", false_positive)
        self.true_negative = _as_count("true_negative", true_negative)
        self.false_negative = _as_count("false_negative", false_negative)
        self.counts = (self.true_positive, self.false_positive, self.true_negative, self.false_negative)
        self.n = sum(self.counts)
        if self.n == 0:
            raise ValueError("confusion matrix must contain at least one observation.")
        self._mle_ll = _cell_log_likelihood(self.counts, tuple(count / self.n for count in self.counts))

    @classmethod
    def from_predictions(cls, labels: np.ndarray, scores: np.ndarray, threshold: float) -> ConfusionMatrix:
        """Build the matrix at ``scores >= threshold``.

        Args:
            labels: Binary labels, 0 or 1, one per row.
            scores: Finite scores, one per row, aligned with ``labels``.
            threshold: Decision threshold. A score at the threshold is positive.

        Returns:
            The confusion matrix of those rows.

        Raises:
            ValueError: If the arrays are empty, misaligned, non-binary, or non-finite.
        """
        label_values, score_values, threshold = _prediction_arrays(labels, scores, threshold)
        positive = label_values.astype(float) == 1.0
        predicted = score_values >= threshold
        true_positive = int(np.sum(predicted & positive))
        false_positive = int(np.sum(predicted & ~positive))
        false_negative = int(np.sum(~predicted & positive))
        true_negative = int(np.sum(~predicted & ~positive))
        return cls(true_positive, false_positive, true_negative, false_negative)

    def precision(self) -> float:
        """Return precision, or NaN when nothing was predicted positive.

        Returns:
            True positives over predicted positives.
        """
        return _rate(self.true_positive, self.true_positive + self.false_positive)

    def recall(self) -> float:
        """Return recall, or NaN when there are no positive labels.

        Returns:
            True positives over positive labels.
        """
        return _rate(self.true_positive, self.true_positive + self.false_negative)

    def false_positive_rate(self) -> float:
        """Return the false-positive rate, or NaN when there are no negative labels.

        Returns:
            False positives over negative labels.
        """
        return _rate(self.false_positive, self.false_positive + self.true_negative)

    def f1(self) -> float:
        """Return the positive F1 score, or NaN when precision and recall are undefined.

        Returns:
            The harmonic mean of precision and recall. Zero when both are zero.
        """
        precision, recall = self.precision(), self.recall()
        if not math.isfinite(precision) or not math.isfinite(recall):
            return math.nan
        if precision + recall == 0.0:
            return 0.0
        return 2.0 * precision * recall / (precision + recall)

    def precision_recall_statistic(self, precision: float, recall: float) -> float:
        """Return the joint profile log-likelihood ratio at one precision-recall point.

        The critical value for a joint 95% region is the chi-square 0.95 quantile
        with two degrees of freedom (5.991), as in Urlus et al. (2023), section 5.

        Args:
            precision: Hypothesised precision in [0, 1].
            recall: Hypothesised recall in [0, 1].

        Returns:
            The profile log-likelihood ratio. ``inf`` when the point cannot
            generate a positive observed cell.

        Raises:
            ValueError: If either argument is not a finite number in [0, 1].
        """
        precision = _require_unit_interval("precision", precision)
        recall = _require_unit_interval("recall", recall)
        active = self.n - self.true_negative
        probs = _pr_probabilities(precision, recall, self.true_negative, self.n, active)
        if probs is None:
            return math.inf
        return _likelihood_ratio(self.counts, probs, self._mle_ll)

    def _precision_statistic(self, precision: float) -> float:
        """Profile log-likelihood ratio for precision, with recall profiled out."""
        if precision <= 0.0:
            return 0.0 if self.true_positive == 0 else math.inf
        if precision >= 1.0:
            return 0.0 if self.false_positive == 0 and self.true_positive > 0 else math.inf
        return _min_over(lambda recall: self.precision_recall_statistic(precision, recall), _EDGE, 1.0 - _EDGE)

    def _f1_statistic(self, f1: float) -> float:
        """Profile log-likelihood ratio for F1, with recall profiled out."""
        estimate = self.f1()
        if f1 <= 0.0:
            return 0.0 if estimate == 0.0 else math.inf
        if f1 >= 1.0:
            return 0.0 if estimate == 1.0 else math.inf
        lower = f1 / (2.0 - f1)
        if lower >= 1.0 - _EDGE:
            return math.inf

        def score(recall: float) -> float:
            precision = _precision_for_f1(f1, recall)
            if precision is None:
                return math.inf
            return self.precision_recall_statistic(precision, recall)

        return _min_over(score, lower + _EDGE, 1.0 - _EDGE)

    def _pr_variances(self) -> tuple[float, float, float]:
        """Return precision variance, recall variance, and their covariance.

        Returns:
            The three moments from equations 13–15. NaN when a denominator is zero.
        """
        tp, fp, fn = self.true_positive, self.false_positive, self.false_negative
        predicted, actual = tp + fp, tp + fn
        var_p = tp * fp / predicted**3 if predicted else math.nan
        var_r = tp * fn / actual**3 if actual else math.nan
        cov = tp * fp * fn / (predicted**2 * actual**2) if predicted and actual else math.nan
        return var_p, var_r, cov

    def _precision_normal(self) -> tuple[float, bool]:
        """Bivariate-normal standard error of precision and its validity flag."""
        variance, _, _ = self._pr_variances()
        predicted = self.true_positive + self.false_positive
        count = self.true_positive * self.false_positive / predicted if predicted else math.nan
        standard_error = math.sqrt(variance) if math.isfinite(variance) else math.nan
        return standard_error, _normal_flag(count, self.precision())

    def _recall_normal(self) -> tuple[float, bool]:
        """Bivariate-normal standard error of recall and its validity flag."""
        _, variance, _ = self._pr_variances()
        actual = self.true_positive + self.false_negative
        count = self.true_positive * self.false_negative / actual if actual else math.nan
        standard_error = math.sqrt(variance) if math.isfinite(variance) else math.nan
        return standard_error, _normal_flag(count, self.recall())

    def _fpr_normal(self) -> tuple[float, bool]:
        """Binomial standard error of the false-positive rate and its validity flag."""
        negatives = self.false_positive + self.true_negative
        if negatives == 0:
            return math.nan, False
        proportion = BinomialProportion(self.false_positive, negatives)
        return proportion.normal_standard_error()

    def _f1_normal(self) -> tuple[float, bool]:
        """Delta-method standard error of F1 from the precision-recall covariance."""
        precision, recall = self.precision(), self.recall()
        var_p, var_r, cov = self._pr_variances()
        _, precision_ok = self._precision_normal()
        _, recall_ok = self._recall_normal()
        if not (precision_ok and recall_ok) or precision + recall == 0.0:
            return math.nan, False
        total = precision + recall
        d_precision = 2.0 * recall**2 / total**2
        d_recall = 2.0 * precision**2 / total**2
        variance = d_precision**2 * var_p + d_recall**2 * var_r + 2.0 * d_precision * d_recall * cov
        return math.sqrt(max(variance, 0.0)), True

    def _recall_interval(self, confidence: float) -> tuple[float, float]:
        """Wilks interval for recall. Empty positive class returns NaN bounds."""
        actual = self.true_positive + self.false_negative
        if actual == 0:
            return math.nan, math.nan
        return BinomialProportion(self.true_positive, actual).interval(confidence)

    def _fpr_interval(self, confidence: float) -> tuple[float, float]:
        """Wilks interval for the false-positive rate."""
        negatives = self.false_positive + self.true_negative
        if negatives == 0:
            return math.nan, math.nan
        return BinomialProportion(self.false_positive, negatives).interval(confidence)

    def _one_interval(self, metric: str, estimate: float, bounds: tuple[float, float],
                      normal: tuple[float, bool], confidence: float) -> MetricInterval:
        """Assemble one interval row."""
        defined = math.isfinite(estimate) and math.isfinite(bounds[0]) and math.isfinite(bounds[1])
        return MetricInterval(
            metric, estimate, bounds[0], bounds[1], confidence,
            WILKS_METHOD if defined else "undefined", normal[0], bool(normal[1]) if defined else False,
        )

    def intervals(self, confidence: float = 0.95) -> tuple[MetricInterval, ...]:
        """Return Wilks intervals for precision, recall, false-positive rate, and F1.

        Recall and the false-positive rate are binomial proportions of the actual
        class (Appendix G). Precision and F1 profile the remaining multinomial
        nuisance parameter.

        Args:
            confidence: Confidence level in (0, 1).

        Returns:
            One interval per metric, in that order.

        Raises:
            ValueError: If ``confidence`` is outside (0, 1).
        """
        confidence = _require_confidence(confidence)
        precision_bounds = _statistic_interval(self.precision(), self._precision_statistic, confidence, "precision")
        f1_bounds = _statistic_interval(self.f1(), self._f1_statistic, confidence, "f1")
        specs = (
            ("precision", self.precision(), precision_bounds, self._precision_normal()),
            ("recall", self.recall(), self._recall_interval(confidence), self._recall_normal()),
            ("fpr", self.false_positive_rate(), self._fpr_interval(confidence), self._fpr_normal()),
            ("f1", self.f1(), f1_bounds, self._f1_normal()),
        )
        return tuple(
            self._one_interval(metric, estimate, bounds, normal, confidence)
            for metric, estimate, bounds, normal in specs
        )

    def interval_frame(self, threshold: float, confidence: float = 0.95) -> pd.DataFrame:
        """Return the intervals as a table with the counts that produced them.

        Args:
            threshold: Decision threshold stored on every row.
            confidence: Confidence level in (0, 1).

        Returns:
            One row per metric.

        Raises:
            ValueError: If ``threshold`` is not finite or ``confidence`` is outside (0, 1).
        """
        if not math.isfinite(float(threshold)):
            raise ValueError(f"threshold must be finite, got {threshold}.")
        rows = []
        for item in self.intervals(confidence):
            rows.append({
                "metric": item.metric, "estimate": item.estimate, "ci_low": item.ci_low,
                "ci_high": item.ci_high, "confidence": item.confidence, "method": item.method,
                "normal_se": item.normal_se, "normal_valid": item.normal_valid, "threshold": float(threshold),
                "tp": self.true_positive, "fp": self.false_positive, "tn": self.true_negative,
                "fn": self.false_negative, "n": self.n,
            })
        return pd.DataFrame(rows)
