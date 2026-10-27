"""Tests for LeDell et al. (2015) influence-curve CV-AUC confidence intervals."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from population_drift.analysis import (
    DriftConfig,
    _fold_influence,
    classifier_two_sample_test,
    cv_auc_with_ci,
)


def _check(actual: object, expected: object, what: str) -> None:
    """Fail with the comparison that did not hold."""
    if actual != expected:
        pytest.fail(f"{what}: got {actual!r}, expected {expected!r}.")


def _fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two folds of four rows with one score tie in fold 0."""
    labels = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    scores = np.array([0.9, 0.5, 0.5, 0.1, 0.8, 0.7, 0.3, 0.2])
    fold_ids = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    return labels, scores, fold_ids


def test_cv_auc_with_ci_matches_hand_computation() -> None:
    """Known scores yield the hand-computed mean fold AUC, SE and clipped interval."""
    labels, scores, fold_ids = _fixture()
    result = cv_auc_with_ci(labels, scores, fold_ids)
    _check(result["cv_auc"], 0.9375, "cv_auc")
    _check(result["cv_auc_se"], 0.0625, "cv_auc_se")
    assert result["cv_auc_ci_low"] == pytest.approx(0.8150022509662467)
    _check(result["cv_auc_ci_high"], 1.0, "cv_auc_ci_high clipped at 1")


def test_fold_influence_is_centered_by_class() -> None:
    """Influence values average to zero within each class in every fold."""
    labels, scores, fold_ids = _fixture()
    pi1 = float(np.mean(labels == 1))
    pi0 = 1.0 - pi1
    for fold in np.unique(fold_ids):
        mask = fold_ids == fold
        _, influence = _fold_influence(labels[mask], scores[mask], pi1, pi0)
        fold_labels = labels[mask]
        assert influence[fold_labels == 1].mean() == pytest.approx(0.0, abs=1e-12)
        assert influence[fold_labels == 0].mean() == pytest.approx(0.0, abs=1e-12)


def test_perfect_separation_clips_interval_at_one() -> None:
    """Perfect ranking gives cv_auc = 1 and an upper bound clipped at 1."""
    labels = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    scores = np.array([0.9, 0.1, 0.8, 0.2, 0.95, 0.05, 0.7, 0.3])
    fold_ids = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    result = cv_auc_with_ci(labels, scores, fold_ids)
    _check(result["cv_auc"], 1.0, "cv_auc")
    _check(result["cv_auc_ci_high"], 1.0, "upper clip")
    assert result["cv_auc_ci_low"] <= result["cv_auc"] <= result["cv_auc_ci_high"]


def test_cv_auc_with_ci_rejects_invalid_inputs() -> None:
    """Validation raises ValueError for bad shapes, labels, folds and confidence."""
    labels, scores, fold_ids = _fixture()
    with pytest.raises(ValueError, match="same length"):
        cv_auc_with_ci(labels[:-1], scores, fold_ids)
    with pytest.raises(ValueError, match="binary"):
        cv_auc_with_ci(np.array([1, 2, 0, 0, 1, 1, 0, 0]), scores, fold_ids)
    with pytest.raises(ValueError, match="both classes"):
        cv_auc_with_ci(np.array([1, 1, 1, 1, 0, 0, 0, 0]), scores, np.array([0, 0, 0, 0, 1, 1, 1, 1]))
    with pytest.raises(ValueError, match="confidence"):
        cv_auc_with_ci(labels, scores, fold_ids, confidence=1.0)


def test_classifier_two_sample_test_reports_cv_auc_columns() -> None:
    """Domain classifier stores cv_auc* and no longer stores auc_ci_*."""
    rng = np.random.default_rng(0)
    n = 40
    source = pd.DataFrame({"x": rng.normal(0, 1, n), "y": rng.normal(0, 1, n)})
    target = pd.DataFrame({"x": rng.normal(1, 1, n), "y": rng.normal(1, 1, n)})
    config = DriftConfig(cv_folds=2, n_classifier_permutations=0, n_bootstrap=0, seed=0)
    result = classifier_two_sample_test(source, target, ["x", "y"],
                                        {"x": "continuous", "y": "continuous"}, config)
    row = result.tables["main"].iloc[0]
    for column in ("cv_auc", "cv_auc_se", "cv_auc_ci_low", "cv_auc_ci_high"):
        assert column in result.tables["main"].columns, column
        assert np.isfinite(row[column]), column
    assert "auc_ci_low" not in result.tables["main"].columns
    assert "auc_ci_high" not in result.tables["main"].columns
    assert row["cv_auc_ci_low"] <= row["cv_auc"] <= row["cv_auc_ci_high"]
