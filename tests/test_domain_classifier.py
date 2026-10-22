"""Unit tests for domain-classifier TreeSHAP importance."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Callable

import numpy as np
import pandas as pd
import pytest

import population_drift.analysis as analysis
from population_drift.analysis import DriftConfig, classifier_two_sample_test


def _check(actual: object, expected: object, what: str) -> None:
    """Fail with the comparison that did not hold."""
    if actual != expected:
        pytest.fail(f"{what}: got {actual!r}, expected {expected!r}.")


def _populations(n: int = 40) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, str]]:
    """Two populations that differ on ``driver`` and match on ``noise``."""
    if n < 4:
        raise ValueError(f"n must be >= 4, got {n}.")
    rng = np.random.default_rng(0)
    source = pd.DataFrame({"driver": rng.normal(0.0, 1.0, n), "noise": rng.normal(0.0, 1.0, n)})
    target = pd.DataFrame({"driver": rng.normal(3.0, 1.0, n), "noise": rng.normal(0.0, 1.0, n)})
    columns = ["driver", "noise"]
    return source, target, columns, {name: "continuous" for name in columns}


def _fast_config(**overrides: object) -> DriftConfig:
    """Classifier config without bootstrap or permutation refits."""
    values = {"n_bootstrap": 0, "n_permutations": 0, "n_classifier_permutations": 0, "cv_folds": 2}
    values.update(overrides)
    return DriftConfig(**values)  # type: ignore[arg-type]


class _FakeTreeExplainer:
    """Test double for ``shap.TreeExplainer`` that records the explained model."""

    models: list[object] = []
    values_for: Callable[[pd.DataFrame], object] | None = None

    def __init__(self, model: object) -> None:
        """Record ``model``; ``values_for`` must already be set on the class."""
        if model is None:
            raise TypeError("model is required.")
        if _FakeTreeExplainer.values_for is None:
            raise TypeError("values_for is required.")
        _FakeTreeExplainer.models.append(model)

    def shap_values(self, features: pd.DataFrame) -> object:
        """Return the scripted contribution matrix.

        Args:
            features: Rows passed to the explainer.

        Returns:
            Scripted SHAP values.

        Raises:
            TypeError: If ``features`` is not a DataFrame.
        """
        if not isinstance(features, pd.DataFrame):
            raise TypeError("features must be a DataFrame.")
        values_for = _FakeTreeExplainer.values_for
        if values_for is None:
            raise TypeError("values_for is required.")
        return values_for(features)


def _install_tree_explainer(
    monkeypatch: pytest.MonkeyPatch,
    values_for: Callable[[pd.DataFrame], object],
) -> list[object]:
    """Replace ``analysis.shap.TreeExplainer`` with ``_FakeTreeExplainer``."""
    if values_for is None:
        raise TypeError("values_for is required.")
    _FakeTreeExplainer.models = []
    _FakeTreeExplainer.values_for = values_for
    monkeypatch.setattr(
        analysis, "shap", SimpleNamespace(TreeExplainer=_FakeTreeExplainer), raising=False)
    return _FakeTreeExplainer.models


def test_domain_classifier_importance_uses_tree_explainer(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Separation drivers come from ``shap.TreeExplainer``, not split gain."""
    source, target, columns, types = _populations()
    seen = _install_tree_explainer(
        monkeypatch,
        lambda data: np.column_stack([np.full(len(data), 2.0), np.full(len(data), 1.0)]),
    )
    result = classifier_two_sample_test(source, target, columns, types, _fast_config())
    _check(len(seen) > 0, True, "TreeExplainer was constructed")
    table = result.tables["importance"]
    _check("shap_share" in table.columns, True, "shap_share column")
    _check("mean_abs_shap" in table.columns, True, "mean_abs_shap column")
    _check("gain_share" in table.columns, False, "gain_share is not kept")
    _check(table.iloc[0]["feature"], "driver", "larger |SHAP| feature is first")
    _check(round(float(table.iloc[0]["shap_share"]), 6), round(2.0 / 3.0, 6), "driver shap share")
    _check(round(float(table.iloc[1]["shap_share"]), 6), round(1.0 / 3.0, 6), "noise shap share")


def test_domain_classifier_shap_respects_max_shap_rows(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """TreeSHAP on the domain classifier is capped at ``max_shap_rows``."""
    source, target, columns, types = _populations(30)
    seen: list[int] = []

    def fake_contributions(model: object, features: pd.DataFrame) -> np.ndarray:
        if not isinstance(features, pd.DataFrame):
            raise TypeError("features must be a DataFrame.")
        seen.append(len(features))
        return np.ones((len(features), features.shape[1]))

    monkeypatch.setattr(analysis, "predict_contributions", fake_contributions)
    classifier_two_sample_test(
        source, target, columns, types, _fast_config(max_shap_rows=12))
    _check(seen, [12], "explained row count")


def test_domain_classifier_ranks_the_shifted_feature_first() -> None:
    """A live domain classifier puts the shifted feature first in mean |SHAP|."""
    source, target, columns, types = _populations(60)
    result = classifier_two_sample_test(source, target, columns, types, _fast_config())
    table = result.tables["importance"]
    _check(table.iloc[0]["feature"], "driver", "shifted feature is the top SHAP driver")
    total = float(table["shap_share"].sum())
    if abs(total - 1.0) > 1e-9:
        pytest.fail(f"shap_share sum: got {total}, expected 1.0.")


def test_domain_classifier_shap_importance_rejects_a_non_frame() -> None:
    """The explainer helper names the required frame type."""
    from population_drift.analysis import _domain_classifier_shap_importance

    with pytest.raises(TypeError, match="features must be a pandas DataFrame"):
        _domain_classifier_shap_importance(object(), [1.0], _fast_config())


def test_domain_classifier_shap_importance_rejects_a_missing_model() -> None:
    """A missing classifier is rejected before TreeExplainer runs."""
    from population_drift.analysis import _domain_classifier_shap_importance

    with pytest.raises(TypeError, match="model is required"):
        _domain_classifier_shap_importance(None, pd.DataFrame({"x": [1.0]}), _fast_config())
