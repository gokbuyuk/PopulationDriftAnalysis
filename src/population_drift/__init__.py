"""Population drift toolkit.

Import the public API from this package (``DriftConfig``,
``population_drift_analyzer``, ``build_baseline``, ``run_comparison``,
``run_paired_comparison``) or from
``population_drift.analysis``. Run the command line with
``python -m population_drift``. The path runner is
``python -m population_drift.run_drift``.
"""

from __future__ import annotations

from typing import Any

from population_drift import analysis as _analysis


def __getattr__(name: str) -> Any:
    """Resolve public names from ``population_drift.analysis`` on demand."""
    if name.startswith("_"):
        raise AttributeError(f"module 'population_drift' has no attribute {name!r}")
    try:
        return getattr(_analysis, name)
    except AttributeError as exc:
        raise AttributeError(f"module 'population_drift' has no attribute {name!r}") from exc


def __dir__() -> list[str]:
    """Include the analysis module's public names in tab completion."""
    names = set(globals())
    names.update(name for name in dir(_analysis) if not name.startswith("_"))
    return sorted(names)
