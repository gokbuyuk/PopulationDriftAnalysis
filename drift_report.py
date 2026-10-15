"""Build a standalone HTML report from the outputs of ``population_drift.py``.

Usage:
    python drift_report.py --comparison runs/comparison --out runs/report.html   # full analysis report
    python drift_report.py --baseline runs/baseline --out runs/baseline.html     # baseline-only report

The report embeds all plots (PNG, base64) and styles, so it is a single self-contained file.
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

logger = logging.getLogger("drift_report")

GLOBAL_FEATURE = "__global__"
SOURCE_COLOR = "#2563eb"
TARGET_COLOR = "#ea580c"
LEVEL_COLORS = {"green": "#16a34a", "amber": "#d97706", "red": "#dc2626", "info": "#64748b"}
LEVEL_LABELS = {"green": "OK", "amber": "Caution", "red": "Material", "info": "Info"}
KEY_DTYPES = {"task": str, "table": str, "feature": str, "metric": str}
GLOBAL_METRIC_LABELS = {
    ("domain_classifier", "auc"): "Domain classifier AUC",
    ("mmd", "mmd2"): "MMD²",
    ("energy", "energy_distance"): "Energy distance",
    ("correlation", "frobenius"): "Correlation Δ (Frobenius)",
    ("joint_outliers", "ood_share"): "Out-of-support share",
    ("missingness_patterns", "js_distance"): "Missingness pattern JS",
    ("prediction_drift", "psi"): "Score PSI",
    ("prediction_drift", "ks_d"): "Score KS D",
    ("prediction_drift", "positive_rate_cohens_h_abs"): "Positive rate |h|",
    ("shap_drift", "rank_rho_deficit"): "1 − SHAP rank ρ",
    ("weighted_performance", "ess_deficit"): "1 − ESS ratio",
}


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------
def _read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV; empty files give an empty frame. Key columns are read as strings."""
    try:
        return pd.read_csv(path, dtype={k: v for k, v in KEY_DTYPES.items()})
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


@dataclass
class BaselineRun:
    """Baseline outputs: long statistics, summary and metadata."""

    directory: Path
    statistics: pd.DataFrame
    summary: pd.DataFrame
    metadata: dict[str, Any]

    def null(self, task: str, metric: str, feature: str = GLOBAL_FEATURE) -> np.ndarray:
        """Baseline values of one statistic across rounds."""
        stats = self.statistics
        mask = (stats["task"] == task) & (stats["metric"] == metric) & (stats["feature"] == feature)
        return stats.loc[mask, "value"].dropna().to_numpy()


@dataclass
class ComparisonRun:
    """Comparison outputs: task tables, plot data, metadata, decision and baseline positions."""

    directory: Path
    tables: dict[str, pd.DataFrame]
    plot_data: dict[str, Any]
    metadata: dict[str, Any]
    decision: dict[str, Any]
    features: pd.DataFrame
    baseline_comparison: pd.DataFrame

    def table(self, task: str, name: str = "main") -> pd.DataFrame:
        return self.tables.get(f"{task}__{name}", pd.DataFrame())

    def baseline_threshold(self, task: str, metric: str) -> dict[str, float]:
        """feature -> baseline threshold for one statistic."""
        bc = self.baseline_comparison
        if bc.empty:
            return {}
        rows = bc[(bc["task"] == task) & (bc["metric"] == metric)]
        return dict(zip(rows["feature"], rows["baseline_threshold"]))


def load_baseline_run(directory: str | Path) -> BaselineRun:
    """Load a baseline directory written by ``population_drift.py baseline``."""
    directory = Path(directory)
    for name in ("baseline_statistics.csv", "baseline_summary.csv", "baseline_metadata.json"):
        if not (directory / name).exists():
            raise FileNotFoundError(f"{directory} is not a baseline directory (missing {name}).")
    return BaselineRun(directory, _read_csv(directory / "baseline_statistics.csv"),
                       _read_csv(directory / "baseline_summary.csv"),
                       json.loads((directory / "baseline_metadata.json").read_text()))


def load_comparison_run(directory: str | Path) -> ComparisonRun:
    """Load a comparison directory written by ``population_drift.py compare``."""
    directory = Path(directory)
    for name in ("decision.json", "metadata.json", "feature_assessment.csv", "tables"):
        if not (directory / name).exists():
            raise FileNotFoundError(f"{directory} is not a comparison directory (missing {name}).")
    tables = {path.stem: _read_csv(path) for path in sorted((directory / "tables").glob("*.csv"))}
    plot_path = directory / "plot_data.json"
    bc_path = directory / "baseline_comparison.csv"
    return ComparisonRun(directory, tables, json.loads(plot_path.read_text()) if plot_path.exists() else {},
                         json.loads((directory / "metadata.json").read_text()),
                         json.loads((directory / "decision.json").read_text()),
                         _read_csv(directory / "feature_assessment.csv"),
                         _read_csv(bc_path) if bc_path.exists() else pd.DataFrame())


# --------------------------------------------------------------------------------------
# Formatting helpers
# --------------------------------------------------------------------------------------
def _fmt(value: Any) -> str:
    """Compact, readable cell formatting."""
    if value is None:
        return "–"
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return "–"
        if value == 0:
            return "0"
        if abs(value) < 1e-3:
            return f"{value:.1e}"
        if abs(value) >= 100:
            return f"{value:,.0f}" if float(value).is_integer() else f"{value:,.1f}"
        return f"{value:.3g}"
    if isinstance(value, str) and value in ("True", "False"):
        return "yes" if value == "True" else "no"
    return html.escape(str(value))


def _pct(value: float) -> str:
    return "–" if value is None or not np.isfinite(value) else f"{value:.1%}"


def _scalar(table: pd.DataFrame, column: str) -> float:
    if table.empty or column not in table.columns:
        return np.nan
    value = table[column].iloc[0]
    return float(value) if pd.notna(value) else np.nan


def _level_badge(level: str) -> str:
    level = level if level in LEVEL_COLORS else "info"
    return f'<span class="badge {level}">{LEVEL_LABELS[level]}</span>'


def table_html(df: pd.DataFrame, columns: Sequence[str] | None = None, rename: dict[str, str] | None = None,
               level_col: str | None = None, max_rows: int | None = None,
               wide_cols: Sequence[str] = ("reasons", "pattern", "Methods")) -> str:
    """Render a DataFrame as an HTML table; ``level_col`` values are rendered as coloured badges."""
    if df is None or df.empty:
        return '<p class="muted">No data.</p>'
    columns = [c for c in (columns or list(df.columns)) if c in df.columns]
    rename = rename or {}
    view = df[columns].head(max_rows) if max_rows else df[columns]
    header = "".join(f"<th>{html.escape(rename.get(c, c))}</th>" for c in columns)
    body = []
    for _, row in view.iterrows():
        cells = []
        for col in columns:
            value = row[col]
            css = ' class="wide"' if col in wide_cols else ""
            cells.append(f"<td>{_level_badge(str(value))}</td>" if col == level_col else f"<td{css}>{_fmt(value)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    note = f'<p class="muted">Showing {max_rows} of {len(df)} rows.</p>' if max_rows and len(df) > max_rows else ""
    return f'<div class="table-wrap"><table><thead><tr>{header}</tr></thead><tbody>{"".join(body)}</tbody></table></div>{note}'


def details(summary: str, content: str) -> str:
    return f"<details><summary>{html.escape(summary)}</summary>{content}</details>"


def figure_html(fig: plt.Figure, alt: str) -> str:
    """Embed a Matplotlib figure as a base64 PNG."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f'<img class="plot" alt="{html.escape(alt)}" src="data:image/png;base64,{encoded}">'


def _style_axes(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="both", alpha=0.25)


# --------------------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------------------
def plot_feature_overview(run: ComparisonRun) -> str:
    """PSI per feature (coloured by severity, baseline p95 ticks) and drift vs model importance."""
    features = run.features.dropna(subset=["psi"]).sort_values("psi")
    if features.empty:
        return ""
    p95 = {**run.baseline_threshold("continuous", "psi"), **run.baseline_threshold("categorical", "psi")}
    has_importance = features["model_importance_share"].notna().any()
    height = max(3.5, 0.3 * len(features) + 1.2)
    fig, axes = plt.subplots(1, 2 if has_importance else 1, figsize=(13 if has_importance else 7.5, height))
    axes = np.atleast_1d(axes)
    ax = axes[0]
    colors = [LEVEL_COLORS.get(level, "#999") for level in features["severity"]]
    ax.barh(features["feature"], features["psi"], color=colors)
    for i, feature in enumerate(features["feature"]):
        if feature in p95 and np.isfinite(p95[feature]):
            ax.plot([p95[feature]] * 2, [i - 0.4, i + 0.4], color="black", lw=2)
    ax.axvline(0.10, ls="--", color="#999", lw=1)
    ax.axvline(0.25, ls="--", color="#444", lw=1)
    ax.set_xlabel("PSI  (dashed: 0.10 / 0.25 rules of thumb; black tick: baseline 95th pct)")
    ax.set_title("Drift per feature (colour = severity)")
    _style_axes(ax)
    if has_importance:
        ax = axes[1]
        ax.scatter(features["model_importance_share"], features["psi"], c=colors, s=60, edgecolor="white")
        for row in features.itertuples():
            if row.severity != "green" or row.model_importance_share >= features["model_importance_share"].quantile(0.75):
                ax.annotate(row.feature, (row.model_importance_share, row.psi), xytext=(4, 3),
                            textcoords="offset points", fontsize=8)
        ax.axhline(0.10, ls="--", color="#999", lw=1)
        ax.axhline(0.25, ls="--", color="#444", lw=1)
        ax.set_xlabel("Model importance (share of mean |SHAP| on training population)")
        ax.set_ylabel("PSI")
        ax.set_title("Drift vs model importance (top-right = critical)")
        _style_axes(ax)
    fig.tight_layout()
    return figure_html(fig, "Feature drift overview")


def plot_ecdfs(run: ComparisonRun, features: Sequence[str]) -> str:
    """ECDF overlays (from stored quantiles) for selected continuous features."""
    quantiles = run.plot_data.get("continuous", {}).get("quantiles", {})
    features = [f for f in features if f in quantiles]
    if not features:
        return ""
    stats = run.table("continuous").set_index("feature")
    ncols = min(3, len(features))
    nrows = math.ceil(len(features) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.1 * nrows), squeeze=False)
    for ax, feature in zip(axes.flat, features):
        q = quantiles[feature]
        ax.step(q["source"], q["probs"], where="post", color=SOURCE_COLOR, label="training", lw=1.6)
        ax.step(q["target"], q["probs"], where="post", color=TARGET_COLOR, label="new", lw=1.6)
        row = stats.loc[feature]
        ax.set_title(f"{feature}: KS D={row['ks_d']:.3f}, PSI={row['psi']:.3f}", fontsize=10)
        ax.set_ylabel("ECDF")
        _style_axes(ax)
    for ax in list(axes.flat)[len(features):]:
        ax.axis("off")
    axes.flat[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return figure_html(fig, "ECDF comparison")


def plot_categories(run: ComparisonRun, features: Sequence[str]) -> str:
    """Grouped level-frequency bars for selected categorical features."""
    frequencies = run.plot_data.get("categorical", {}).get("frequencies", {})
    features = [f for f in features if f in frequencies]
    if not features:
        return ""
    ncols = min(3, len(features))
    nrows = math.ceil(len(features) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.1 * nrows), squeeze=False)
    stats = run.table("categorical").set_index("feature")
    for ax, feature in zip(axes.flat, features):
        data = frequencies[feature]
        x = np.arange(len(data["levels"]))
        ax.bar(x - 0.2, data["source"], 0.4, color=SOURCE_COLOR, label="training")
        ax.bar(x + 0.2, data["target"], 0.4, color=TARGET_COLOR, label="new")
        ax.set_xticks(x, [str(level)[:12] for level in data["levels"]], rotation=45 if len(x) > 5 else 0, fontsize=8)
        ax.set_title(f"{feature}: V={stats.loc[feature, 'cramers_v']:.3f}, PSI={stats.loc[feature, 'psi']:.3f}",
                     fontsize=10)
        ax.set_ylabel("share")
        _style_axes(ax)
    for ax in list(axes.flat)[len(features):]:
        ax.axis("off")
    axes.flat[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return figure_html(fig, "Category frequencies")


def plot_missingness(run: ComparisonRun) -> str:
    """Missing rates per feature for features with any missing values."""
    table = run.table("missingness")
    if table.empty:
        return ""
    table = table[(table["missing_rate_source"] > 0) | (table["missing_rate_target"] > 0)]
    if table.empty:
        return ""
    y = np.arange(len(table))
    fig, ax = plt.subplots(figsize=(7.5, 0.5 * len(table) + 1.2))
    ax.barh(y + 0.18, table["missing_rate_source"], 0.36, color=SOURCE_COLOR, label="training")
    ax.barh(y - 0.18, table["missing_rate_target"], 0.36, color=TARGET_COLOR, label="new")
    ax.set_yticks(y, table["feature"])
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("missing rate")
    ax.legend(frameon=False)
    _style_axes(ax)
    fig.tight_layout()
    return figure_html(fig, "Missing rates")


def _null_panel(ax: plt.Axes, null: np.ndarray, observed: float, title: str, percentile: float = 95) -> None:
    """Histogram of baseline values with the observed value and the baseline threshold."""
    if len(null):
        ax.hist(null, bins=min(20, max(5, len(null) // 2)), color="#94a3b8", edgecolor="white")
        threshold = np.percentile(null, percentile)
        ax.axvline(threshold, color="black", ls="--", lw=1, label=f"baseline p{percentile:.0f}")
    if np.isfinite(observed):
        ax.axvline(observed, color=TARGET_COLOR, lw=2.2, label="observed")
    ax.set_title(title, fontsize=10)
    ax.set_yticks([])
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
    ax.ticklabel_format(axis="x", style="sci", scilimits=(-3, 4))
    ax.tick_params(axis="x", labelsize=8)
    _style_axes(ax)


def plot_null_grid(entries: Sequence[dict[str, Any]], title: str) -> str:
    """Grid of baseline-null histograms with observed values."""
    entries = [e for e in entries if len(e["null"]) or np.isfinite(e["observed"])]
    if not entries:
        return ""
    ncols = min(4, len(entries))
    nrows = math.ceil(len(entries) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.5 * nrows + 0.6), squeeze=False)
    for ax, entry in zip(axes.flat, entries):
        _null_panel(ax, entry["null"], entry["observed"], entry["title"])
    for ax in list(axes.flat)[len(entries):]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False, fontsize=8, ncol=2)
    fig.suptitle(title, fontsize=11, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=1.6)
    return figure_html(fig, title)


def plot_domain_classifier(run: ComparisonRun, baseline: BaselineRun | None) -> str:
    """Domain-classifier score distributions, AUC vs baseline null, and separation drivers."""
    data = run.plot_data.get("domain_classifier")
    main = run.table("domain_classifier")
    if not data or main.empty:
        return ""
    importance = run.table("domain_classifier", "importance").head(10).iloc[::-1]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    centers = (np.array(data["bins"][:-1]) + np.array(data["bins"][1:])) / 2
    width = data["bins"][1] - data["bins"][0]
    axes[0].bar(centers, data["source_hist"], width, color=SOURCE_COLOR, alpha=0.55, label="training")
    axes[0].bar(centers, data["target_hist"], width, color=TARGET_COLOR, alpha=0.55, label="new")
    axes[0].set_xlabel("out-of-fold P(new population | x)")
    axes[0].set_title("Classifier scores by population", fontsize=10)
    axes[0].legend(frameon=False, fontsize=8)
    _style_axes(axes[0])
    null = baseline.null("domain_classifier", "auc") if baseline else np.array([])
    _null_panel(axes[1], null, _scalar(main, "auc"), f"AUC {_scalar(main, 'auc'):.3f} vs baseline null")
    axes[1].legend(frameon=False, fontsize=7)
    axes[2].barh(importance["feature"], importance["gain_share"], color="#475569")
    axes[2].set_xlabel("share of split gain")
    axes[2].set_title("Features driving the separation", fontsize=10)
    _style_axes(axes[2])
    fig.tight_layout()
    return figure_html(fig, "Classifier two-sample test")


def plot_correlation_delta(run: ComparisonRun) -> str:
    """Heatmap of Spearman correlation change (new − training)."""
    data = run.plot_data.get("correlation")
    if not data or not data.get("columns"):
        return ""
    columns = data["columns"]
    delta = np.array(data["target_corr"], dtype=float) - np.array(data["source_corr"], dtype=float)
    size = max(4.5, 0.42 * len(columns) + 2)
    fig, ax = plt.subplots(figsize=(size + 1, size))
    image = ax.imshow(np.nan_to_num(delta), cmap="RdBu_r", vmin=-0.6, vmax=0.6)
    ax.set_xticks(range(len(columns)), columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(columns)), columns, fontsize=8)
    if len(columns) <= 16:
        for i in range(len(columns)):
            for j in range(len(columns)):
                if i != j and abs(delta[i, j]) >= 0.1:
                    ax.text(j, i, f"{delta[i, j]:+.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, shrink=0.8, label="Δ Spearman ρ (new − training)")
    ax.set_title("Change in correlation structure", fontsize=10)
    fig.tight_layout()
    return figure_html(fig, "Correlation change")


def plot_support(run: ComparisonRun) -> str:
    """Isolation-Forest score distributions and per-feature out-of-range shares."""
    outliers = run.plot_data.get("joint_outliers")
    coverage = run.table("range_coverage")
    panels = int(bool(outliers)) + int(not coverage.empty)
    if not panels:
        return ""
    fig, axes = plt.subplots(1, panels, figsize=(6.5 * panels, 3.6), squeeze=False)
    axes = list(axes.flat)
    if outliers:
        ax = axes.pop(0)
        bins = np.array(outliers["bins"])
        centers, width = (bins[:-1] + bins[1:]) / 2, bins[1] - bins[0]
        ax.bar(centers, outliers["source_hist"], width, color=SOURCE_COLOR, alpha=0.55, label="training (held-out)")
        ax.bar(centers, outliers["target_hist"], width, color=TARGET_COLOR, alpha=0.55, label="new")
        ax.axvline(outliers["threshold"], color="black", ls="--", lw=1, label="OOD threshold")
        ax.set_xlabel("Isolation Forest score (lower = more unusual)")
        ax.set_title("Joint novelty", fontsize=10)
        ax.legend(frameon=False, fontsize=8)
        _style_axes(ax)
    if not coverage.empty:
        ax = axes.pop(0)
        cov = coverage.sort_values("outside_share")
        ax.barh(cov["feature"], cov["outside_share"], color=np.where(cov["ratio_to_expected"] > 2, TARGET_COLOR, "#94a3b8"))
        ax.axvline(cov["expected_share"].iloc[0], color="black", ls="--", lw=1, label="expected under no shift")
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_xlabel("share of new rows outside training [0.5%, 99.5%] range")
        ax.set_title("Values outside the training range", fontsize=10)
        ax.legend(frameon=False, fontsize=8)
        _style_axes(ax)
    fig.tight_layout()
    return figure_html(fig, "Support checks")


def plot_model_outputs(run: ComparisonRun) -> str:
    """Model score distributions and mean |SHAP| by population."""
    prediction = run.plot_data.get("prediction_drift")
    shap = run.table("shap_drift")
    panels = int(bool(prediction)) + int(not shap.empty)
    if not panels:
        return ""
    fig, axes = plt.subplots(1, panels, figsize=(6.5 * panels, max(3.6, 0.28 * min(len(shap), 15) + 1)), squeeze=False)
    axes = list(axes.flat)
    if prediction:
        ax = axes.pop(0)
        bins = np.array(prediction["bins"])
        centers, width = (bins[:-1] + bins[1:]) / 2, bins[1] - bins[0]
        ax.bar(centers, prediction["source_hist"], width, color=SOURCE_COLOR, alpha=0.55, label="training")
        ax.bar(centers, prediction["target_hist"], width, color=TARGET_COLOR, alpha=0.55, label="new")
        if prediction.get("threshold") is not None:
            ax.axvline(prediction["threshold"], color="black", ls="--", lw=1, label="decision threshold")
        ax.set_xlabel("model score")
        ax.set_title("Model score distribution", fontsize=10)
        ax.legend(frameon=False, fontsize=8)
        _style_axes(ax)
    if not shap.empty:
        ax = axes.pop(0)
        top = shap.head(15).iloc[::-1]
        y = np.arange(len(top))
        ax.barh(y - 0.2, top["mean_abs_shap_source"], 0.4, color=SOURCE_COLOR, label="training")
        ax.barh(y + 0.2, top["mean_abs_shap_target"], 0.4, color=TARGET_COLOR, label="new")
        ax.set_yticks(y, [f"{f} *" if imp else f for f, imp in zip(top["feature"], top["important"])], fontsize=8)
        ax.set_xlabel("mean |SHAP|  (* = model-important)")
        ax.set_title("Feature attributions", fontsize=10)
        ax.legend(frameon=False, fontsize=8)
        _style_axes(ax)
    fig.tight_layout()
    return figure_html(fig, "Model outputs")


def plot_performance(run: ComparisonRun) -> str:
    """Reference vs estimated metric with 95% CI for each estimation method."""
    rows = []
    for task, label in (("weighted_performance", "Importance weighting"), ("cbpe", "CBPE")):
        table = run.table(task)
        for row in table.itertuples():
            rows.append((row.metric, label, row.reference_value, row.estimate, row.ci_low, row.ci_high))
    if not rows:
        return ""
    df = pd.DataFrame(rows, columns=["metric", "method", "reference", "estimate", "low", "high"])
    metrics = list(dict.fromkeys(df["metric"]))
    ncols = min(4, len(metrics))
    nrows = math.ceil(len(metrics) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.4 * nrows), squeeze=False)
    for ax, metric in zip(axes.flat, metrics):
        sub = df[df["metric"] == metric].reset_index(drop=True)
        for i, row in sub.iterrows():
            ax.plot(row["reference"], i, marker="D", color="#64748b", ms=7, label="reference" if i == 0 else None)
            if np.isfinite(row["low"]):
                ax.plot([row["low"], row["high"]], [i, i], color=TARGET_COLOR, lw=2)
            ax.plot(row["estimate"], i, marker="o", color=TARGET_COLOR, ms=7, label="new (estimate)" if i == 0 else None)
        ax.set_yticks(range(len(sub)), sub["method"], fontsize=8)
        ax.set_ylim(-0.6, len(sub) - 0.4)
        ax.set_title(metric, fontsize=10)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        ax.tick_params(axis="x", labelsize=8)
        _style_axes(ax)
    for ax in list(axes.flat)[len(metrics):]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False, fontsize=8, ncol=2)
    fig.suptitle("Reference vs estimate on the new population (95% CI)", fontsize=11, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return figure_html(fig, "Estimated performance")


def plot_calibration(run: ComparisonRun) -> str:
    """Reliability diagram on the training population."""
    reliability = run.table("calibration", "reliability")
    if reliability.empty:
        return ""
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    ax.plot([0, 1], [0, 1], ls="--", color="#999", lw=1)
    ax.plot(reliability["mean_predicted"], reliability["observed_rate"], marker="o", color=SOURCE_COLOR)
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed positive rate")
    ax.set_title("Calibration on training population", fontsize=10)
    _style_axes(ax)
    fig.tight_layout()
    return figure_html(fig, "Calibration")


def plot_baseline_feature_variability(baseline: BaselineRun) -> str:
    """95th percentile of per-feature PSI and KS D across baseline rounds."""
    summary = baseline.summary
    psi = summary[(summary["metric"] == "psi") & summary["task"].isin(["continuous", "categorical"])]
    if psi.empty:
        return ""
    psi = psi.sort_values("p95")
    fig, ax = plt.subplots(figsize=(7.5, max(3.5, 0.3 * len(psi) + 1)))
    ax.barh(psi["feature"], psi["p95"], color="#94a3b8", label="p95")
    ax.scatter(psi["p50"], psi["feature"], color="black", s=14, zorder=3, label="median")
    ax.axvline(0.10, ls="--", color="#999", lw=1)
    ax.set_xlabel("PSI between random training subsets (natural variability)")
    ax.legend(frameon=False, fontsize=8)
    _style_axes(ax)
    fig.tight_layout()
    return figure_html(fig, "Baseline per-feature PSI")


def global_null_entries(baseline: BaselineRun | None, run: ComparisonRun | None) -> list[dict[str, Any]]:
    """Entries for the observed-vs-baseline grid of global statistics."""
    entries = []
    observed = {}
    if run is not None and not run.baseline_comparison.empty:
        bc = run.baseline_comparison[run.baseline_comparison["feature"] == GLOBAL_FEATURE]
        observed = {(r.task, r.metric): r.observed for r in bc.itertuples()}
    for (task, metric), label in GLOBAL_METRIC_LABELS.items():
        null = baseline.null(task, metric) if baseline else np.array([])
        value = observed.get((task, metric), np.nan)
        if len(null) or np.isfinite(value):
            entries.append({"title": label, "null": null, "observed": float(value) if pd.notna(value) else np.nan})
    return entries


# --------------------------------------------------------------------------------------
# Section builders
# --------------------------------------------------------------------------------------
@dataclass
class Section:
    anchor: str
    title: str
    body: str


def _feature_counts(features: pd.DataFrame) -> dict[str, int]:
    return {level: int((features["severity"] == level).sum()) for level in ("red", "amber", "green")}


def section_summary(run: ComparisonRun, baseline: BaselineRun | None) -> Section:
    """Verdict, signal table, key findings and caveats."""
    decision, meta = run.decision, run.metadata
    types = meta["feature_types"]
    n_cont = sum(t == "continuous" for t in types.values())
    level = decision["verdict_level"]
    context = (f"Training population: <b>{meta['n_source']:,}</b> rows · New population: <b>{meta['n_target']:,}</b> rows · "
               f"{len(types)} features ({n_cont} continuous, {len(types) - n_cont} categorical) · "
               f"Model: <b>{'yes' if meta['has_model'] else 'no'}</b> · "
               f"Training labels: <b>{html.escape(str(meta['label_col'])) if meta['label_col'] else 'no'}</b> · "
               f"Baseline: <b>{_fmt(decision.get('baseline_rounds'))}</b> rounds of "
               f"<b>{_fmt(decision.get('baseline_subset_size'))}</b>-row training subsets")
    signal_rows = "".join(
        f"<tr><td>{_level_badge(s['level'])}</td><td>{html.escape(s['signal'])}</td>"
        f"<td>{html.escape(s['value'])}</td><td class='muted'>{html.escape(s['detail'])}</td></tr>"
        for s in sorted(decision["signals"], key=lambda s: -{"red": 2, "amber": 1, "green": 0, "info": -1}[s["level"]]))
    signals = (f'<div class="table-wrap"><table><thead><tr><th>Level</th><th>Signal</th><th>Observed</th>'
               f'<th>Detail</th></tr></thead><tbody>{signal_rows}</tbody></table></div>')

    findings = []
    features = run.features
    flagged = features[features["severity"] != "green"]
    counts = _feature_counts(features)
    findings.append(f"<b>{counts['red']}</b> feature(s) with material drift and <b>{counts['amber']}</b> with moderate "
                    f"drift out of {len(features)}; <b>{counts['green']}</b> are within natural variability or negligible.")
    for row in flagged.head(6).itertuples():
        importance = " (model-important)" if meta["has_model"] and row.important else ""
        findings.append(f"{_level_badge(row.severity)} <b>{html.escape(row.feature)}</b>{importance}: "
                        f"{html.escape(str(row.reasons))}")
    domain = run.table("domain_classifier")
    if not domain.empty:
        auc = _scalar(domain, "auc")
        strength = ("indistinguishable" if auc <= 0.55 else "mildly separable" if auc <= 0.65
                    else "moderately separable" if auc <= 0.80 else "strongly separable")
        findings.append(f"Jointly, the populations are <b>{strength}</b> (classifier AUC {auc:.3f}).")
    outliers = run.table("joint_outliers")
    if not outliers.empty:
        findings.append(f"<b>{_pct(_scalar(outliers, 'ood_share'))}</b> of new rows fall outside the training support "
                        f"(expected {_pct(_scalar(outliers, 'expected_share'))}); predictions for these rows are "
                        "extrapolations.")
    for task, label in (("weighted_performance", "importance weighting"), ("cbpe", "CBPE")):
        perf = run.table(task)
        if not perf.empty:
            row = perf.iloc[0]
            findings.append(f"Estimated {row['metric'].upper()} on the new population ({label}): <b>{row['estimate']:.3f}</b> "
                            f"[{_fmt(row['ci_low'])}–{_fmt(row['ci_high'])}] vs reference {row['reference_value']:.3f}.")
    label_shift = run.table("label_shift")
    if not label_shift.empty:
        findings.append(f"Estimated positive rate: {_pct(_scalar(label_shift, 'source_prevalence'))} → "
                        f"<b>{_pct(_scalar(label_shift, 'target_prevalence_estimate'))}</b> "
                        f"[{_pct(_scalar(label_shift, 'ci_low'))}–{_pct(_scalar(label_shift, 'ci_high'))}] (BBSE).")

    caveats = ["Without labels for the new population, <b>concept shift</b> (a change in P(y|x)) cannot be detected; "
               "all performance estimates assume it is absent. Label a small random sample to verify."]
    if meta["has_model"] and meta["label_col"]:
        caveats.append("Label-based checks use the training-population rows supplied. If these rows were used to fit "
                       "the model, calibration and performance references are optimistic; prefer a labelled holdout.")
    caveats += [html.escape(w) for w in decision.get("warnings", [])]
    body = (f'<div class="verdict {level}"><div class="verdict-label">Verdict</div>'
            f'<div class="verdict-text">{html.escape(decision["verdict"])}</div>'
            f'<div class="verdict-rule">{html.escape(decision["verdict_rule"])}</div></div>'
            f'<p class="context">{context}</p><h3>Signals</h3>{signals}'
            f'<h3>Key findings</h3><ul>{"".join(f"<li>{f}</li>" for f in findings)}</ul>'
            f'<div class="callout warn"><b>Caveats</b><ul>{"".join(f"<li>{c}</li>" for c in caveats)}</ul></div>'
            '<div class="callout"><b>How to read this report.</b> Every statistic is compared with the <i>baseline</i>: '
            'the distribution of the same statistic between random subsets of the training population (pure sampling '
            'noise). A difference is flagged only if it is statistically significant after multiple-testing correction, '
            'larger than the baseline 95th percentile, and at least moderate by effect-size rules of thumb. '
            f'{_level_badge("green")} within noise or negligible · {_level_badge("amber")} real, moderate · '
            f'{_level_badge("red")} real, large.</div>')
    return Section("summary", "Executive summary", body)


def section_features(run: ComparisonRun) -> Section:
    """Overview of per-feature drift with the feature assessment table."""
    features = run.features
    counts = _feature_counts(features)
    has_model = run.metadata["has_model"]
    critical = features[(features["severity"] == "red") & features["important"]]
    text = (f"<p>{counts['red']} red, {counts['amber']} amber and {counts['green']} green features. "
            + (f"Critical (large drift on a model-important feature): <b>{', '.join(map(html.escape, critical['feature']))}</b>. "
               if len(critical) else "No large drift on model-important features. ")
            + ("Model importance is the share of mean |SHAP| on the training population; drift on low-importance "
               "features rarely affects predictions." if has_model else
               "No model was supplied, so every feature is treated as important.") + "</p>")
    columns = ["feature", "type", "severity", "important", "model_importance_share", "psi", "ks_d", "smd",
               "cramers_v", "equivalent_mean", "significant", "beyond_baseline", "reasons"]
    rename = {"model_importance_share": "importance share", "equivalent_mean": "equivalent mean (TOST)",
              "beyond_baseline": "beyond baseline", "ks_d": "KS D", "smd": "SMD", "cramers_v": "Cramér's V",
              "psi": "PSI"}
    body = plot_feature_overview(run) + text + table_html(features, columns, rename, level_col="severity")
    return Section("features", "Feature-level drift", body)


def section_continuous(run: ComparisonRun, top_k: int) -> Section | None:
    """ECDFs of the most drifted continuous features and the full test table."""
    table = run.table("continuous")
    if table.empty or "ks_d" not in table:
        return None
    severity = run.features.set_index("feature")["severity"]
    equivalence = run.table("equivalence")
    merged = table.merge(equivalence[["feature", "equivalent", "tost_p"]], on="feature", how="left") \
        if not equivalence.empty else table
    merged["severity"] = merged["feature"].map(severity)
    merged["baseline_p95_psi"] = merged["feature"].map(run.baseline_threshold("continuous", "psi"))
    merged["baseline_p95_ks"] = merged["feature"].map(run.baseline_threshold("continuous", "ks_d"))
    order = merged.assign(_o=merged["severity"].map({"red": 2, "amber": 1, "green": 0})).sort_values(
        ["_o", "ks_d"], ascending=False)
    top = order["feature"].head(top_k).tolist()
    n_sig = int((merged["ks_p_adj"] < run.metadata["config"]["alpha"]).sum())
    n_equiv = int(merged.get("equivalent", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
    text = (f"<p>{n_sig} of {len(merged)} continuous features differ significantly (KS, adjusted p &lt; α); "
            f"{n_equiv} have statistically equivalent means (TOST, |SMD| &lt; "
            f"{run.metadata['config']['equivalence_margin_smd']}). The ECDFs show the {len(top)} most drifted features: "
            "horizontal gaps are location shifts, different slopes are spread changes, divergence at the ends are "
            "tail changes. Significance alone is not decisive with large samples; use SMD, PSI and the baseline "
            "columns for magnitude.</p>")
    columns = ["feature", "severity", "mean_source", "mean_target", "smd", "smd_ci_low", "smd_ci_high", "var_ratio",
               "ks_d", "baseline_p95_ks", "ks_p_adj", "ad_p_adj", "cvm_p_adj", "wasserstein_norm", "psi",
               "psi_ci_low", "psi_ci_high", "baseline_p95_psi", "equivalent", "tost_p"]
    rename = {"smd_ci_low": "SMD CI low", "smd_ci_high": "SMD CI high", "ks_p_adj": "KS p adj",
              "ad_p_adj": "AD p adj", "cvm_p_adj": "CvM p adj", "wasserstein_norm": "W/SD",
              "baseline_p95_psi": "baseline p95 PSI", "baseline_p95_ks": "baseline p95 KS"}
    body = plot_ecdfs(run, top) + text + details("Full continuous-feature table",
                                                  table_html(order, columns, rename, level_col="severity"))
    return Section("continuous", "Continuous features", body)


def section_categorical(run: ComparisonRun, top_k: int) -> Section | None:
    """Level frequencies of the most drifted categorical features and the full table."""
    table = run.table("categorical")
    if table.empty or "cramers_v" not in table:
        return None
    table = table.copy()
    table["severity"] = table["feature"].map(run.features.set_index("feature")["severity"])
    table["baseline_p95_psi"] = table["feature"].map(run.baseline_threshold("categorical", "psi"))
    order = table.assign(_o=table["severity"].map({"red": 2, "amber": 1, "green": 0})).sort_values(
        ["_o", "psi"], ascending=False)
    unseen = table[table["unseen_share"] > 0]
    text = (f"<p>{int((table['chi2_p_adj'] < run.metadata['config']['alpha']).sum())} of {len(table)} categorical "
            "features differ significantly (chi-square, adjusted). Cramér's V: 0.1 small, 0.3 medium, 0.5 large. "
            + (f"Levels never seen in training appear in: {', '.join(f'{html.escape(r.feature)} ({r.unseen_share:.1%})' for r in unseen.itertuples())}; "
               "LightGBM routes unseen levels down a default branch." if len(unseen) else "No unseen levels.") + "</p>")
    columns = ["feature", "severity", "n_levels_source", "n_levels_target", "unseen_share", "chi2_stat", "chi2_dof",
               "chi2_p_adj", "cramers_v", "js_distance", "psi", "baseline_p95_psi", "top_level_source",
               "top_level_target"]
    body = plot_categories(run, order["feature"].head(top_k).tolist()) + text + details(
        "Full categorical-feature table", table_html(order, columns, {"chi2_p_adj": "chi2 p adj"}, level_col="severity"))
    return Section("categorical", "Categorical features", body)


def section_missingness(run: ComparisonRun) -> Section | None:
    """Missing-rate comparison and missingness patterns."""
    table = run.table("missingness")
    if table.empty:
        return None
    alpha = run.metadata["config"]["alpha"]
    flagged = table[(table["cohens_h_abs"] >= 0.2) & (table["z_p_adj"] < alpha)]
    patterns = run.table("missingness_patterns", "patterns")
    text = ("<p>" + (f"Material missing-rate changes (|Cohen's h| ≥ 0.2, adjusted p &lt; α): "
                     f"{', '.join(f'{html.escape(r.feature)} {r.missing_rate_source:.1%} → {r.missing_rate_target:.1%}' for r in flagged.itertuples())}. "
                     if len(flagged) else "No material change in missing rates. ")
            + "LightGBM learns a default direction for missing values, so a change in missingness changes which branch "
              "rows follow; it often signals a data-collection difference.</p>")
    body = (plot_missingness(run) + text
            + details("Missing-rate table", table_html(table, ["feature", "missing_rate_source", "missing_rate_target",
                                                               "difference", "cohens_h", "z_p_adj"]))
            + details("Most frequent missingness patterns", table_html(patterns)))
    return Section("missingness", "Missing values", body)


def section_joint(run: ComparisonRun, baseline: BaselineRun | None) -> Section | None:
    """Classifier two-sample test, MMD and energy distance."""
    domain = run.table("domain_classifier")
    if domain.empty:
        return None
    auc = _scalar(domain, "auc")
    null = baseline.null("domain_classifier", "auc") if baseline else np.array([])
    p95 = np.percentile(null, 95) if len(null) else np.nan
    perm_p = _scalar(domain, "permutation_p")
    drivers = run.table("domain_classifier", "importance").head(3)["feature"].tolist()
    text = (f"<p>A LightGBM classifier trained to tell the populations apart reaches an out-of-fold AUC of "
            f"<b>{auc:.3f}</b> [{_fmt(_scalar(domain, 'auc_ci_low'))}–{_fmt(_scalar(domain, 'auc_ci_high'))}]"
            + (f", permutation p = {perm_p:.3f}" if np.isfinite(perm_p) else "")
            + (f"; the baseline 95th percentile is {p95:.3f}" if np.isfinite(p95) else "")
            + ". AUC ≈ 0.5 means indistinguishable; 0.55–0.65 mild, 0.65–0.80 moderate, &gt; 0.80 strong shift. "
            f"The separation is driven mostly by <b>{', '.join(map(html.escape, drivers))}</b>. A low AUC can still "
            "hide a moderate shift in a single feature, so read it together with the feature table.</p>")
    kernel_entries = []
    kernel_rows = []
    for task, metric, label in (("mmd", "mmd2", "MMD² (RBF kernel)"), ("energy", "energy_distance", "Energy distance")):
        table = run.table(task)
        if table.empty:
            continue
        kernel_entries.append({"title": label, "null": baseline.null(task, metric) if baseline else np.array([]),
                               "observed": _scalar(table, metric)})
        kernel_rows.append({"test": label, "statistic": _scalar(table, metric),
                            "permutation p": _scalar(table, "permutation_p"),
                            "rows used (training/new)": f"{int(_scalar(table, 'n_source_rows')):,} / "
                                                        f"{int(_scalar(table, 'n_target_rows')):,}"})
    kernel_text = ("<p>Kernel (MMD) and energy-distance permutation tests confirm the joint result with different "
                   "sensitivity; p &lt; α together with a value beyond the baseline indicates a real joint shift.</p>")
    body = (plot_domain_classifier(run, baseline) + text
            + plot_null_grid(kernel_entries, "Kernel tests: observed vs baseline") + kernel_text
            + table_html(pd.DataFrame(kernel_rows))
            + details("Classifier test details", table_html(domain))
            + details("All separation drivers", table_html(run.table("domain_classifier", "importance"))))
    return Section("joint", "Joint distribution", body)


def section_dependence(run: ComparisonRun) -> Section | None:
    """Correlation structure change."""
    main, pairs = run.table("correlation"), run.table("correlation", "pairs")
    if main.empty:
        return None
    alpha = run.metadata["config"]["alpha"]
    material = pairs[(pairs["abs_delta"] > 0.2) & (pairs["fisher_z_p_adj"] < alpha)]
    text = (f"<p>Global change in the Spearman correlation matrix: Frobenius norm {_scalar(main, 'frobenius'):.3f} "
            f"(permutation p = {_fmt(_scalar(main, 'permutation_p'))}). "
            + (f"{len(material)} pair(s) changed by more than 0.2 (Fisher z, adjusted): "
               f"{', '.join(f'{html.escape(r.feature_a)}–{html.escape(r.feature_b)} ({r.rho_source:+.2f} → {r.rho_target:+.2f})' for r in material.head(5).itertuples())}. "
               if len(material) else "No pair changed by more than 0.2. ")
            + "Changed dependencies matter because tree models exploit feature interactions.</p>")
    body = plot_correlation_delta(run) + text + details(
        "Top correlation changes", table_html(pairs, ["feature_a", "feature_b", "rho_source", "rho_target", "delta",
                                                      "fisher_z_p_adj"], max_rows=30))
    return Section("dependence", "Dependence structure", body)


def section_support(run: ComparisonRun) -> Section | None:
    """Out-of-support checks."""
    outliers, coverage = run.table("joint_outliers"), run.table("range_coverage")
    if outliers.empty and coverage.empty:
        return None
    parts = []
    if not outliers.empty:
        parts.append(f"An Isolation Forest fitted on the training population flags <b>{_pct(_scalar(outliers, 'ood_share'))}</b> "
                     f"of new rows as unusual (expected {_pct(_scalar(outliers, 'expected_share'))} by construction; "
                     f"binomial p = {_fmt(_scalar(outliers, 'binomial_p'))}). Rule of thumb: &lt; 2% fine, 2–5% caution, "
                     "&gt; 5% material.")
    if not coverage.empty:
        flagged = coverage[coverage["ratio_to_expected"] > 2]
        parts.append((f"Features with more than twice the expected share outside the training range: "
                      f"{', '.join(f'{html.escape(r.feature)} ({r.outside_share:.1%})' for r in flagged.itertuples())}. "
                      if len(flagged) else "No feature has an excess of out-of-range values. ")
                     + "Tree models do not extrapolate: beyond the training range predictions stay at the edge value.")
    body = (plot_support(run) + f"<p>{' '.join(parts)}</p>"
            + details("Range coverage table", table_html(coverage, ["feature", "source_low", "source_high", "target_min",
                                                                    "target_max", "share_below", "share_above",
                                                                    "outside_share", "ratio_to_expected",
                                                                    "binomial_p_adj"]))
            + details("Isolation Forest details", table_html(outliers)))
    return Section("support", "Support and out-of-distribution rows", body)


def section_model(run: ComparisonRun) -> Section | None:
    """Prediction and attribution drift."""
    prediction, shap_summary = run.table("prediction_drift"), run.table("shap_drift", "summary")
    if prediction.empty and shap_summary.empty:
        return None
    parts = []
    if not prediction.empty:
        parts.append(f"Model score PSI {_scalar(prediction, 'psi'):.3f} (KS D {_scalar(prediction, 'ks_d'):.3f}); "
                     "PSI &lt; 0.10 stable, 0.10–0.25 monitor, &gt; 0.25 act.")
        if "positive_rate_source" in prediction:
            parts.append(f"Predicted positive rate at threshold {_scalar(prediction, 'threshold'):g}: "
                         f"{_pct(_scalar(prediction, 'positive_rate_source'))} → "
                         f"<b>{_pct(_scalar(prediction, 'positive_rate_target'))}</b> (Cohen's h "
                         f"{_scalar(prediction, 'positive_rate_cohens_h'):+.2f}); borderline predictions "
                         f"{_pct(_scalar(prediction, 'near_threshold_share_source'))} → "
                         f"{_pct(_scalar(prediction, 'near_threshold_share_target'))}.")
    if not shap_summary.empty:
        parts.append(f"Importance ranking is preserved with Spearman ρ = {_scalar(shap_summary, 'importance_rank_rho'):.2f} "
                     "(&lt; 0.8 suggests the model relies on features differently). A shifted score distribution is not "
                     "necessarily worse performance: it may reflect a different case mix.")
    shap = run.table("shap_drift")
    body = (plot_model_outputs(run) + f"<p>{' '.join(parts)}</p>"
            + details("Prediction drift table", table_html(prediction))
            + details("SHAP drift table", table_html(shap, ["feature", "important", "importance_share_source",
                                                            "importance_rank_source", "importance_rank_target",
                                                            "mean_abs_shap_source", "mean_abs_shap_target",
                                                            "mean_abs_shap_rel_change", "shap_ks_d", "shap_ks_p_adj"])))
    return Section("model", "Model behaviour", body)


def section_performance(run: ComparisonRun) -> Section | None:
    """Label-free performance estimates, calibration and prior shift."""
    weighted, cbpe = run.table("weighted_performance"), run.table("cbpe")
    calibration, label_shift = run.table("calibration"), run.table("label_shift")
    if weighted.empty and cbpe.empty and label_shift.empty:
        return None
    items = []
    summary = run.table("weighted_performance", "summary")
    if not weighted.empty:
        items.append(f"<b>Importance weighting</b> (assumes covariate shift only): labelled training rows are re-weighted "
                     f"to resemble the new population. Effective sample size ratio {_scalar(summary, 'ess_ratio'):.2f} "
                     "(&gt; 0.5 reliable, 0.1–0.5 caution, &lt; 0.1 unreliable).")
    if not cbpe.empty:
        ece = _scalar(calibration, "ece")
        items.append("<b>CBPE</b> (assumes a calibrated model and no concept shift): expected metrics from predicted "
                     "probabilities, compared like-for-like with CBPE on the training population. "
                     + (f"Calibration ECE {ece:.3f}: " + ("acceptable." if ece <= 0.05 else
                                                         "<b>too high, so CBPE estimates are unreliable</b>.")
                        if np.isfinite(ece) else ""))
    if not label_shift.empty:
        shifted = str(label_shift["prior_shift_detected"].iloc[0]) == "True"
        items.append(f"<b>BBSE</b> (assumes P(x|y) unchanged): estimated positive rate "
                     f"{_pct(_scalar(label_shift, 'target_prevalence_estimate'))} vs "
                     f"{_pct(_scalar(label_shift, 'source_prevalence'))} in training"
                     + (" — prior shift detected; recalibrate the intercept or threshold." if shifted else
                        " — no significant prior shift."))
    tolerance = run.metadata["config"]["performance_tolerance"]
    text = (f"<ul>{''.join(f'<li>{i}</li>' for i in items)}</ul><p>Tolerance for the primary metric: {tolerance:g} "
            f"({'relative' if run.metadata['problem_type'] == 'regression' else 'absolute AUC'}). A signal is red only "
            "when the whole 95% CI is beyond tolerance. If the methods disagree, treat the result as inconclusive.</p>")
    plots = plot_performance(run) + plot_calibration(run)
    tables = (details("Importance-weighted estimates", table_html(weighted) + table_html(summary))
              + details("CBPE estimates", table_html(cbpe))
              + details("Calibration and prior shift", table_html(calibration) + table_html(label_shift)))
    return Section("performance", "Estimated performance (no labels)", plots + text + tables)


def section_baseline(run: ComparisonRun | None, baseline: BaselineRun | None) -> Section | None:
    """Observed vs baseline for every global statistic, plus the full comparison table."""
    if baseline is None:
        return None
    meta = baseline.metadata
    entries = global_null_entries(baseline, run)
    observed_text = ("the orange line is the new population. Values right of the dashed 95th-percentile line exceed "
                     "sampling noise. The empirical p-value is (1 + #baseline ≥ observed) / (1 + rounds), so its "
                     "resolution is limited by the number of rounds." if run is not None else
                     "the dashed line is the 95th percentile used as the threshold for the new population.")
    text = (f"<p>The baseline repeats the analysis {meta['n_rounds']} times on random splits of the training population "
            f"({meta['subset_size']:,} rows vs the remaining {meta['n_source'] - meta['subset_size']:,}). Grey histograms "
            f"show the natural variability of each statistic; {observed_text}</p>")
    body = plot_null_grid(entries, "Global statistics: observed vs baseline") + text
    if run is not None and not run.baseline_comparison.empty:
        bc = run.baseline_comparison.sort_values(["exceeds_baseline", "empirical_p"], ascending=[False, True])
        body += details("Full baseline comparison (all statistics)",
                        table_html(bc, ["task", "feature", "metric", "observed", "baseline_median", "baseline_threshold",
                                        "baseline_max", "empirical_p", "exceeds_baseline"], max_rows=400))
    else:
        body += plot_baseline_feature_variability(baseline)
        body += details("Baseline summary (all statistics)", table_html(baseline.summary, max_rows=400))
    return Section("baseline", "Baseline reference", body)


def section_methods(config: dict[str, Any]) -> Section:
    """Compact methods and thresholds appendix."""
    th = config["thresholds"]
    rows = [
        ("Continuous features", "KS, Anderson–Darling, Cramér–von Mises; SMD, variance ratio, Wasserstein/SD, PSI",
         f"PSI {th['psi_moderate']}/{th['psi_major']}; KS D {th['ks_moderate']}/{th['ks_major']}; |SMD| {th['smd_small']}/{th['smd_large']}"),
        ("Equivalence", f"TOST on means, margin {config['equivalence_margin_smd']} SD", "p < α ⇒ equivalent"),
        ("Categorical features", "Chi-square homogeneity (sparse levels merged); Cramér's V, Jensen–Shannon, PSI",
         f"V {th['cramers_v_moderate']}/{th['cramers_v_major']}; unseen levels > {th['unseen_level_share']:.0%}"),
        ("Missingness", "Two-proportion z-test, Cohen's h; pattern chi-square / JS", f"|h| {th['cohens_h_small']}/{th['cohens_h_large']}"),
        ("Joint distribution", f"LightGBM classifier two-sample test ({config['cv_folds']}-fold OOF AUC, permutation p); "
         "MMD (RBF) and energy distance with permutation p", f"AUC {th['domain_auc_mild']}/{th['domain_auc_strong']}"),
        ("Dependence", "Spearman matrices: Frobenius-norm permutation test; per-pair Fisher z (var 1.06/(n−3))",
         f"|Δρ| > {th['correlation_pair_delta']}"),
        ("Support", f"Range [{config['range_quantiles'][0]}, {config['range_quantiles'][1]}] quantiles + binomial test; "
         f"Isolation Forest at {config['ood_quantile']:.0%} held-out quantile + binomial test",
         f"OOD share {th['ood_share_caution']:.0%}/{th['ood_share_material']:.0%}"),
        ("Model behaviour", "Score KS/PSI; positive-rate z-test; TreeSHAP KS and rank correlation",
         f"rank ρ < {th['importance_rank_rho']}; important = top {th['important_feature_cumulative_share']:.0%} of |SHAP|"),
        ("Performance (no labels)", "Importance weighting (domain-classifier density ratio, Kish ESS); CBPE; BBSE",
         f"ESS ratio {th['ess_ratio_poor']}/{th['ess_ratio_good']}; tolerance {config['performance_tolerance']}"),
        ("Inference", f"{config['correction_method']} correction at α = {config['alpha']}; {config['n_bootstrap']} bootstrap "
         f"resamples; {config['n_permutations']} permutations ({config['n_classifier_permutations']} for the classifier)",
         f"baseline threshold = p{th['baseline_percentile']:.0f}"),
    ]
    table = pd.DataFrame(rows, columns=["Area", "Methods", "Thresholds (amber / red)"])
    note = ("<p class='muted'>Thresholds are conventions (Siddiqi 2006 for PSI, Cohen 1988 for effect sizes, Austin 2009 "
            "for SMD); the baseline comparison is the statistically grounded reference.</p>")
    return Section("methods", "Methods and thresholds", table_html(table) + note)


def baseline_only_sections(baseline: BaselineRun) -> list[Section]:
    """Sections for a report built from a baseline run alone."""
    meta = baseline.metadata
    failed = meta.get("failed_task_counts") or {}
    intro = (f"<div class='verdict info'><div class='verdict-label'>Baseline only</div><div class='verdict-text'>"
             f"Natural variability of the training population</div><div class='verdict-rule'>{meta['n_rounds']} rounds · "
             f"{meta['subset_size']:,}-row subsets vs remaining {meta['n_source'] - meta['subset_size']:,} rows · "
             f"{len(meta['feature_cols'])} features · model: {'yes' if meta['has_model'] else 'no'}</div></div>"
             "<p>This report shows how much each drift statistic varies between random subsets of the <i>same</i> "
             "population. These distributions are the reference for the new population: once it is available, run "
             "<code>population_drift.py compare --baseline &lt;this directory&gt;</code>. Build the baseline with "
             "<code>--subset-size</code> close to the expected new-population size, because many statistics shrink as "
             "sample size grows.</p>"
             + (f"<div class='callout warn'>Tasks that failed in some rounds: {html.escape(str(failed))}</div>" if failed else ""))
    sections = [Section("summary", "Overview", intro)]
    baseline_section = section_baseline(None, baseline)
    if baseline_section:
        sections.append(baseline_section)
    sections.append(section_methods(meta["config"]))
    return sections


# --------------------------------------------------------------------------------------
# Page assembly
# --------------------------------------------------------------------------------------
CSS = """
:root{--bg:#f6f7f9;--card:#fff;--text:#1f2933;--muted:#5b6b7b;--border:#dde3ea;--accent:#1d4ed8}
*{box-sizing:border-box}body{margin:0;font-family:-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
background:var(--bg);color:var(--text);line-height:1.55;font-size:15px}
header{background:linear-gradient(135deg,#1e3a8a,#2563eb);color:#fff;padding:1.8rem 2rem}
header h1{margin:0 0 .3rem;font-size:1.6rem}header p{margin:0;opacity:.9}
.layout{display:grid;grid-template-columns:230px minmax(0,1fr);gap:1.4rem;max-width:1400px;margin:0 auto;padding:1.4rem}
nav{position:sticky;top:1rem;align-self:start;background:var(--card);border:1px solid var(--border);
border-radius:10px;padding:1rem;font-size:.9rem}nav a{display:block;color:var(--muted);text-decoration:none;padding:.22rem 0}
nav a:hover{color:var(--accent)}section{min-width:0;background:var(--card);border:1px solid var(--border);border-radius:10px;
padding:1.3rem 1.6rem;margin-bottom:1.4rem}h2{margin-top:0;color:#1e3a8a;border-bottom:2px solid var(--border);
padding-bottom:.35rem}h3{color:#1e40af;margin:1.2rem 0 .5rem}
.plot{max-width:100%;height:auto;display:block;margin:.6rem 0}
.table-wrap{overflow-x:auto;margin:.6rem 0}table{border-collapse:collapse;font-size:.82rem;width:100%}
th,td{border:1px solid var(--border);padding:.35rem .5rem;text-align:left;vertical-align:top}th{background:#eef2f7;
position:sticky;top:0}tbody tr:nth-child(even){background:#fafbfc}td.wide{min-width:280px}
.badge{display:inline-block;padding:.05rem .5rem;border-radius:999px;font-size:.75rem;font-weight:600;color:#fff}
.badge.green{background:#16a34a}.badge.amber{background:#d97706}.badge.red{background:#dc2626}.badge.info{background:#64748b}
.verdict{border-radius:10px;padding:1rem 1.3rem;color:#fff;margin-bottom:1rem}.verdict.green{background:#15803d}
.verdict.amber{background:#b45309}.verdict.red{background:#b91c1c}.verdict.info{background:#475569}
.verdict-label{text-transform:uppercase;font-size:.75rem;letter-spacing:.08em;opacity:.85}
.verdict-text{font-size:1.4rem;font-weight:700}.verdict-rule{font-size:.85rem;opacity:.9;margin-top:.3rem}
.context{color:var(--muted);font-size:.9rem}.muted{color:var(--muted)}
.callout{border-left:4px solid var(--accent);background:#eef4ff;padding:.7rem 1rem;border-radius:6px;margin:1rem 0;font-size:.9rem}
.callout.warn{border-left-color:#d97706;background:#fff6e8}.callout ul{margin:.3rem 0 0 1rem;padding:0}
details{margin:.6rem 0}summary{cursor:pointer;color:var(--accent);font-weight:600}
code{background:#eef2f7;padding:.05rem .3rem;border-radius:4px}
footer{text-align:center;color:var(--muted);font-size:.8rem;padding:1rem 0 2rem}
@media(max-width:900px){.layout{grid-template-columns:minmax(0,1fr)}nav{position:static}}
@media print{nav{display:none}.layout{grid-template-columns:minmax(0,1fr)}details{display:block}}
"""


def render_page(title: str, subtitle: str, sections: Sequence[Section]) -> str:
    """Assemble the standalone HTML page."""
    nav = "".join(f'<a href="#{s.anchor}">{html.escape(s.title)}</a>' for s in sections)
    body = "".join(f'<section id="{s.anchor}"><h2>{html.escape(s.title)}</h2>{s.body}</section>' for s in sections)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return (f'<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1"><title>{html.escape(title)}</title>'
            f'<style>{CSS}</style></head><body><header><h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p></header>'
            f'<div class="layout"><nav><strong>Contents</strong>{nav}</nav><main>{body}</main></div>'
            f'<footer>Generated {generated} by drift_report.py · standalone file, no external resources</footer>'
            '</body></html>')


def build_report(output_path: str | Path, comparison_dir: str | Path | None = None,
                 baseline_dir: str | Path | None = None, title: str | None = None, top_k: int = 6) -> Path:
    """Build the HTML report.

    Args:
        output_path: Destination HTML file.
        comparison_dir: Output directory of ``population_drift.py compare`` (full report).
        baseline_dir: Baseline directory; defaults to the one recorded in the comparison's decision.json.
        title: Optional report title.
        top_k: Number of features shown in distribution plots.

    Returns:
        Path of the written report.

    Raises:
        ValueError: If neither a comparison nor a baseline directory is provided.
    """
    if comparison_dir is None and baseline_dir is None:
        raise ValueError("Provide a comparison directory, a baseline directory, or both.")
    run = load_comparison_run(comparison_dir) if comparison_dir else None
    if baseline_dir is None and run is not None and run.decision.get("baseline_dir"):
        candidate = Path(run.decision["baseline_dir"])
        baseline_dir = candidate if candidate.exists() else None
    baseline = load_baseline_run(baseline_dir) if baseline_dir else None

    if run is not None:
        sections = [section_summary(run, baseline), section_features(run)]
        for builder in (lambda: section_continuous(run, top_k), lambda: section_categorical(run, top_k),
                        lambda: section_missingness(run), lambda: section_joint(run, baseline),
                        lambda: section_dependence(run), lambda: section_support(run), lambda: section_model(run),
                        lambda: section_performance(run), lambda: section_baseline(run, baseline)):
            section = builder()
            if section is not None:
                sections.append(section)
        sections.append(section_methods(run.metadata["config"]))
        subtitle = (f"Training population ({run.metadata['n_source']:,} rows) vs new population "
                    f"({run.metadata['n_target']:,} rows) · analysis run {run.metadata['created_at']}")
        title = title or "Population drift report: is the model transferable?"
    else:
        sections = baseline_only_sections(baseline)
        subtitle = f"Training population baseline · {baseline.metadata['created_at']}"
        title = title or "Population drift baseline report"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_page(title, subtitle, sections), encoding="utf-8")
    logger.info("Report written to %s", output_path)
    return output_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the HTML drift report.")
    parser.add_argument("--comparison", help="Comparison output directory.")
    parser.add_argument("--baseline", help="Baseline directory (optional with --comparison).")
    parser.add_argument("--out", required=True, help="Output HTML path.")
    parser.add_argument("--title")
    parser.add_argument("--top-k", type=int, default=6, help="Features shown in distribution plots.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    build_report(args.out, args.comparison, args.baseline, args.title, args.top_k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
