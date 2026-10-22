"""Build a standalone HTML report from the outputs of ``python -m population_drift``.

Usage:
    python -m population_drift.report --comparison runs/comparison --out runs/report.html
    python -m population_drift.report --baseline runs/baseline --out runs/baseline.html

The report embeds all plots (PNG, base64) and styles, so the HTML file is self-contained.
A markdown copy is written next to it, with the same plots saved as PNG files.

A comparison also writes a short decision report beside the detailed one
(``report_short.html`` / ``report_short.md``) with the verdict, counted signals,
flagged features, and the plots a data scientist needs to decide whether to score
new data.
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import logging
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

logger = logging.getLogger("drift_report")

GLOBAL_FEATURE = "__global__"
SOURCE_COLOR = "#2563eb"
TARGET_COLOR = "#ea580c"
LEVEL_COLORS = {"mild": "#16a34a", "moderate": "#d97706", "severe": "#dc2626", "info": "#64748b"}
LEVEL_LABELS = {"mild": "mild", "moderate": "moderate", "severe": "severe", "info": "info"}
KEY_DTYPES = {"task": str, "table": str, "feature": str, "metric": str}
METRIC_DISPLAY = {
    "auc": "area under the curve (AUC)",
    "brier": "Brier score",
    "log_loss": "log loss",
    "rmse": "root mean squared error (RMSE)",
    "mae": "mean absolute error (MAE)",
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
}
PLOT_METRIC = {
    "auc": "AUC", "brier": "Brier", "log_loss": "log loss", "rmse": "RMSE", "mae": "MAE",
    "accuracy": "accuracy", "precision": "precision", "recall": "recall",
}
PLOT_GLOBAL_LABELS = {
    ("domain_classifier", "auc"): "AUC",
    ("mmd", "mmd2"): "MMD²",
    ("energy", "energy_distance"): "energy",
    ("correlation", "frobenius"): "correlation Δ",
    ("joint_outliers", "ood_share"): "OOD share",
    ("missingness_patterns", "js_distance"): "missingness JS",
    ("prediction_drift", "psi"): "score PSI",
    ("prediction_drift", "ks_d"): "score KS",
    ("prediction_drift", "positive_rate_cohens_h_abs"): "positive rate |h|",
    ("shap_drift", "rank_rho_deficit"): "1 − SHAP rank ρ",
    ("weighted_performance", "ess_deficit"): "1 − ESS",
}
GLOBAL_METRIC_LABELS = {
    ("domain_classifier", "auc"): "Domain classifier area under the curve (AUC)",
    ("mmd", "mmd2"): "squared maximum mean discrepancy (MMD²)",
    ("energy", "energy_distance"): "Energy distance",
    ("correlation", "frobenius"): "Correlation Δ (Frobenius)",
    ("joint_outliers", "ood_share"): "Out-of-support share",
    ("missingness_patterns", "js_distance"): "Missingness pattern Jensen–Shannon (JS)",
    ("prediction_drift", "psi"): "Score Population Stability Index (PSI)",
    ("prediction_drift", "ks_d"): "Score Kolmogorov–Smirnov statistic (KS D)",
    ("prediction_drift", "positive_rate_cohens_h_abs"): "Positive rate |h|",
    ("shap_drift", "rank_rho_deficit"): "1 − Shapley additive explanations (SHAP) rank ρ",
    ("weighted_performance", "ess_deficit"): "1 − effective sample size (ESS) ratio",
}

@dataclass(frozen=True)
class _Acronym:
    """Short name, glossary group, and how to read the number.

    Args:
        name: Words used the first time the short name appears in prose.
        category: Glossary heading.
        reading: What a larger or smaller value means in this report.
    """

    name: str
    category: str
    reading: str


# Insertion order is the order inside each glossary group.
ACRONYMS: dict[str, _Acronym] = {
    "KS": _Acronym("Kolmogorov–Smirnov", "Tests", "Two-sample test of one continuous distribution. Adjusted p below α: the distributions differ. Size is KS D."),
    "AD": _Acronym("Anderson–Darling", "Tests", "Two-sample test with more weight in the tails than KS. Adjusted p below α: the tails differ. Not a severity size."),
    "CvM": _Acronym("Cramér–von Mises", "Tests", "Two-sample test of the full ECDF gap, not only its maximum. Adjusted p below α: the distributions differ. Not a severity size."),
    "TOST": _Acronym("two one-sided tests", "Tests", "Equivalence test for two means, margin 0.10 SD. p below α: means agree. Not a test of variance or shape."),
    "BH": _Acronym("Benjamini–Hochberg", "Tests", "False-discovery adjustment inside one p-value family. Significant means adjusted p below α."),
    "KS D": _Acronym(
        "Kolmogorov–Smirnov statistic", "Magnitudes",
        "Largest vertical gap between the two ECDFs, from 0 to 1. 0.20 means the cumulative shares differ by 20 percentage points at that gap. "
        "Distance only: no sign, and it does not separate a mean shift from a spread or tail change. "
        "≥ 0.10 moderate, ≥ 0.20 large, and only with adjusted p below α and a value above the baseline 95th percentile. "
        "The same reading applies to score KS D and SHAP KS D."),
    "PSI": _Acronym("Population Stability Index", "Magnitudes", "Change in bin shares. Below 0.10 stable, 0.10–0.25 monitor, above 0.25 large."),
    "SMD": _Acronym("standardized mean difference", "Magnitudes", "Mean gap divided by pooled SD. Absolute SMD ≥ 0.10 small, ≥ 0.50 large."),
    "JS": _Acronym("Jensen–Shannon", "Magnitudes", "Distance between category shares. 0 means the same frequencies; larger means less overlap."),
    "MMD": _Acronym("maximum mean discrepancy", "Magnitudes", "Kernel distance between the two samples. Larger means the joint distributions differ. Cannot be severe."),
    "MMD²": _Acronym("squared maximum mean discrepancy", "Magnitudes", "Square of MMD, the stored number. Same reading as MMD."),
    "RBF": _Acronym("radial basis function", "Magnitudes", "Kernel used for MMD. Bandwidth is the median distance between rows."),
    "ECE": _Acronym("expected calibration error", "Magnitudes", "Mean gap between predicted probability and observed rate. Above 0.05, probabilities are off and CBPE is info."),
    "ESS": _Acronym("effective sample size", "Magnitudes", "Kish ratio after weighting. Above 0.50 usable, 0.10–0.50 caution, below 0.10 the weights have collapsed."),
    "OOD": _Acronym("out-of-distribution", "Magnitudes", "Share of new rows the isolation forest calls unusual. Below 2% ordinary, 2–5% caution, above 5% material."),
    "SD": _Acronym("standard deviation", "Magnitudes", "Spread of one column. SMD and Wasserstein/SD are in these units."),
    "ECDF": _Acronym("empirical cumulative distribution function", "Magnitudes", "Share of rows at or below a value. A sideways gap is a location shift; a steeper curve is a tighter spread."),
    "AUC": _Acronym("area under the curve", "Scores", "Area under the ROC curve. 0.5 means no separation. Domain classifier: ≤ 0.55 mild, ≤ 0.80 moderate, above 0.80 severe."),
    "ROC": _Acronym("receiver operating characteristic", "Scores", "True-positive rate against false-positive rate across cutoffs. AUC is its area."),
    "SHAP": _Acronym("Shapley additive explanations", "Scores", "A feature's additive contribution to one score. Mean absolute SHAP rank correlation below 0.80: the ranking of reasons changed."),
    "TreeSHAP": _Acronym("Shapley additive explanations from trees", "Scores", "SHAP for a tree model, from shap.TreeExplainer."),
    "OOF": _Acronym("out-of-fold", "Scores", "Score from a model that did not train on that row."),
    "CBPE": _Acronym("confidence-based performance estimation", "Estimates", "Expected metric if labels were drawn from the predicted probabilities. Requires calibrated scores."),
    "BBSE": _Acronym("black-box shift estimation", "Estimates", "Estimated new class prior. Usable only if P(x given y) is unchanged."),
    "CI": _Acronym("confidence interval", "Estimates", "Range for an estimate. Performance is severe only when the whole interval lies past the tolerance."),
    "RMSE": _Acronym("root mean squared error", "Estimates", "Square root of mean squared error. Larger is worse."),
    "MAE": _Acronym("mean absolute error", "Estimates", "Average absolute error. Larger is worse."),
}
_ACRONYM_CATEGORIES = ("Tests", "Magnitudes", "Scores", "Estimates")
# Hypothesis tests that have no short name. KS, AD, CvM, TOST, and BH stay in the acronym table.
_STATISTICAL_TESTS: tuple[tuple[str, str], ...] = (
    ("Chi-square", "Test that two samples share one discrete distribution. Levels with expected count below 5 are merged. "
     "Adjusted p below α: the category shares differ. Also used on the joint missingness pattern. Size is Cramér's V."),
    ("Two-proportion z-test", "Test that two samples share a rate. Used for missingness, the predicted positive rate, "
     "and the share of scores within 0.05 of the threshold. Adjusted p below α: the rates differ. Size is Cohen's h."),
    ("Permutation test", "Shuffle the population label and recompute the statistic. "
     "p = (1 + shuffles at least as large) / (1 + shuffles). "
     "Used for domain-classifier AUC, MMD², energy distance, and the Spearman-matrix Frobenius norm."),
    ("Mann–Whitney", "Test that domain-classifier scores are higher for new rows than for training rows. "
     "For fixed scores this is the same comparison as AUC. The verdict uses the permutation p, because the scores were fitted on these rows."),
    ("Fisher z", "Test that one Spearman correlation differs between the samples. "
     "The z variance is 1.06/(n−3) in each sample (Fieller, Hartley, and Pearson 1957). Adjusted p below α: that pair changed."),
    ("Binomial test", "One-sided test that the new share outside the training 0.5–99.5% range, or the isolation-forest outlier share, "
     "exceeds the share expected from the training rule. Adjusted p below α: more new rows fall outside support."),
    ("Baseline empirical p", "Share of random reference splits at least as large as the new population: "
     "(1 + such rounds) / (1 + rounds). Beyond baseline means the statistic exceeds that reference's 95th percentile."),
)
_SAME_CONCEPT = {"TreeSHAP": "SHAP", "SHAP": "TreeSHAP"}
_PHRASES: tuple[tuple[str, str], ...] = (
    ("empirical cumulative distribution functions (empirical cumulative distribution function (ECDF)s)", "ECDFs"),
    ("empirical cumulative distribution functions", "ECDFs"),
    ("empirical cumulative distribution function (ECDF)", "ECDF"),
    ("empirical cumulative distribution function", "ECDF"),
    ("squared maximum mean discrepancy (MMD²)", "MMD²"),
    ("squared maximum mean discrepancy", "MMD²"),
    ("maximum mean discrepancy (MMD)", "MMD"),
    ("maximum mean discrepancy", "MMD"),
    ("Shapley additive explanations (SHAP)", "SHAP"),
    ("Shapley additive explanations, SHAP", "SHAP"),
    ("Shapley additive explanations", "SHAP"),
    ("TreeSHAP (Shapley additive explanations)", "TreeSHAP"),
    ("confidence-based performance estimation (CBPE)", "CBPE"),
    ("confidence-based performance estimation, CBPE", "CBPE"),
    ("confidence-based performance estimation", "CBPE"),
    ("black-box shift estimation (BBSE)", "BBSE"),
    ("black-box shift estimation, BBSE", "BBSE"),
    ("black-box shift estimation", "BBSE"),
    ("Kolmogorov–Smirnov statistic (KS D)", "KS D"),
    ("Kolmogorov-Smirnov statistic (KS D)", "KS D"),
    ("Kolmogorov–Smirnov statistic", "KS D"),
    ("Kolmogorov-Smirnov statistic", "KS D"),
    ("standardized mean difference (SMD)", "SMD"),
    ("standardized mean difference", "SMD"),
    ("Population Stability Index (PSI)", "PSI"),
    ("Population Stability Index", "PSI"),
    ("expected calibration error (ECE)", "ECE"),
    ("expected calibration error", "ECE"),
    ("effective sample size (ESS)", "ESS"),
    ("effective sample size", "ESS"),
    ("receiver operating characteristic (ROC)", "ROC"),
    ("radial basis function (RBF)", "RBF"),
    ("radial basis function", "RBF"),
    ("out-of-distribution (OOD)", "OOD"),
    ("out-of-distribution", "OOD"),
    ("Benjamini–Hochberg (BH)", "BH"),
    ("Benjamini-Hochberg (BH)", "BH"),
    ("Benjamini–Hochberg", "BH"),
    ("Benjamini-Hochberg", "BH"),
    ("Cramér–von Mises (CvM)", "CvM"),
    ("Cramér-von Mises (CvM)", "CvM"),
    ("Cramér–von Mises", "CvM"),
    ("Cramér-von Mises", "CvM"),
    ("Anderson–Darling (AD)", "AD"),
    ("Anderson-Darling (AD)", "AD"),
    ("Anderson–Darling", "AD"),
    ("Anderson-Darling", "AD"),
    ("Jensen–Shannon (JS)", "JS"),
    ("Jensen-Shannon (JS)", "JS"),
    ("Jensen–Shannon", "JS"),
    ("Jensen-Shannon", "JS"),
    ("Kolmogorov–Smirnov (KS)", "KS"),
    ("Kolmogorov-Smirnov (KS)", "KS"),
    ("Kolmogorov–Smirnov", "KS"),
    ("Kolmogorov-Smirnov", "KS"),
    ("two one-sided tests (TOST)", "TOST"),
    ("two one-sided tests", "TOST"),
    ("confidence interval (CI)", "CI"),
    ("confidence interval", "CI"),
    ("standard deviation (SD)", "SD"),
    ("standard deviation", "SD"),
    ("area under the curve (AUC)", "AUC"),
    ("area under the curve", "AUC"),
    ("out-of-fold (OOF)", "OOF"),
    ("out-of-fold", "OOF"),
    ("root mean squared error (RMSE)", "RMSE"),
    ("root mean squared error", "RMSE"),
    ("mean absolute error (MAE)", "MAE"),
    ("mean absolute error", "MAE"),
    ("Wasserstein distance / standard deviation (W/SD)", "Wasserstein/SD"),
    ("Wasserstein distance / standard deviation (SD)", "Wasserstein/SD"),
)
_PHRASES = tuple(sorted(_PHRASES, key=lambda item: len(item[0]), reverse=True))
_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9])(" + "|".join(re.escape(token) for token in sorted([*ACRONYMS, "ECDFs"], key=len, reverse=True))
    + r")(?![A-Za-z0-9²])")
_HTML_BLOCK_RE = re.compile(r"(<style\b.*?</style>|<table\b.*?</table>|<img\b[^>]*>)", re.I | re.S)
_SRC_RE = re.compile(r'src="data:image/[^"]*"')
_TAG_RE = re.compile(r"<[^>]+>")
_CODE_RE = re.compile(r"<code>.*?</code>", re.S)
_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
_MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
_TABLE_LINE_RE = re.compile(r"^\|.+\|[ \t]*$")


def _canon(raw: str) -> str:
    """Map a matched token onto a dictionary key.

    Args:
        raw: Token found in the document, including the plural ``ECDFs``.

    Returns:
        Dictionary key.
    """
    return "ECDF" if raw == "ECDFs" else raw


def _phrase(raw: str) -> str:
    """Long name plus short name for a first prose mention.

    Args:
        raw: Token found in the document.

    Returns:
        ``meaning (short name)``.
    """
    if raw == "ECDFs":
        return "empirical cumulative distribution functions (ECDFs)"
    return f"{ACRONYMS[_canon(raw)].name} ({raw})"


def _mark(introduced: set[str], used: set[str], raw: str) -> None:
    """Record a short name as used, and as already introduced.

    Args:
        introduced: Short names whose long form must not be repeated.
        used: Short names that appear in this document.
        raw: Token found in the document.
    """
    token = _canon(raw)
    used.add(token)
    introduced.add(token)
    other = _SAME_CONCEPT.get(token)
    if other:
        introduced.add(other)


def _shorten(text: str) -> str:
    """Replace long names with their short names.

    Args:
        text: Document fragment.

    Returns:
        The fragment with each known long name replaced.
    """
    for phrase, short in _PHRASES:
        text = re.sub(re.escape(phrase), short, text, flags=re.IGNORECASE)
    return text


def _hold_code(text: str) -> tuple[str, list[str]]:
    """Replace inline and HTML code with placeholders.

    Args:
        text: Fragment that may contain code.

    Returns:
        The fragment with code removed, and the code pieces in order.
    """
    held: list[str] = []

    def stash(match: re.Match[str]) -> str:
        held.append(match.group(0))
        return f"\x00C{len(held) - 1}\x00"

    return _CODE_RE.sub(stash, _INLINE_CODE_RE.sub(stash, text)), held


def _hold_images(text: str) -> tuple[str, list[str]]:
    """Replace markdown images with placeholders so acronym expansion skips them.

    Args:
        text: Markdown fragment.

    Returns:
        The fragment with images removed, and the image markup in order.
    """
    held: list[str] = []

    def stash(match: re.Match[str]) -> str:
        held.append(match.group(0))
        return f"\x00I{len(held) - 1}\x00"

    return _MD_IMAGE_RE.sub(stash, text), held


def _restore_images(text: str, held: list[str]) -> str:
    """Put markdown images back.

    Args:
        text: Fragment with image placeholders.
        held: Image markup from ``_hold_images``.

    Returns:
        The fragment with images restored.
    """
    for index, piece in enumerate(held):
        text = text.replace(f"\x00I{index}\x00", piece)
    return text


def _restore_code(text: str, held: list[str]) -> str:
    """Put code pieces back.

    Args:
        text: Fragment with code placeholders.
        held: Code pieces from ``_hold_code``.

    Returns:
        The fragment with code restored.
    """
    for index, piece in enumerate(held):
        text = text.replace(f"\x00C{index}\x00", piece)
    return text


def _expand_prose(text: str, introduced: set[str], used: set[str]) -> str:
    """Expand the first prose mention of each short name.

    Args:
        text: Prose, already shortened.
        introduced: Short names already seen in the document.
        used: Short names that appear in this document.

    Returns:
        Prose with the first mention of each new short name expanded.
    """
    masked, held = _hold_code(text)
    masked, images = _hold_images(masked)
    for image in images:
        _scan(image, introduced, used)

    def repl(match: re.Match[str]) -> str:
        raw = match.group(1)
        if _canon(raw) in introduced:
            used.add(_canon(raw))
            return raw
        _mark(introduced, used, raw)
        return _phrase(raw)

    return _restore_images(_restore_code(_TOKEN_RE.sub(repl, masked), held), images)


def _scan(text: str, introduced: set[str], used: set[str]) -> None:
    """Record short names in a table or plot without expanding them.

    Args:
        text: Visible table or plot text.
        introduced: Short names already seen in the document.
        used: Short names that appear in this document.
    """
    for match in _TOKEN_RE.finditer(text):
        _mark(introduced, used, match.group(1))


def _img_terms(tag: str) -> str:
    """Alt text and plot terms from an image tag.

    Args:
        tag: One ``<img>`` tag.

    Returns:
        The text that counts as a plot mention.
    """
    return " ".join(re.findall(r'(?:alt|data-terms)="([^"]*)"', tag))


def _glossary_groups(used: set[str]) -> list[tuple[str, list[str]]]:
    """Used short names grouped in glossary order.

    Args:
        used: Short names found in the document.

    Returns:
        ``(category, tokens)`` pairs. Empty categories are omitted.
    """
    groups = []
    for category in _ACRONYM_CATEGORIES:
        tokens = [token for token, entry in ACRONYMS.items() if entry.category == category and token in used]
        if tokens:
            groups.append((category, tokens))
    return groups


def _acronym_table_html(tokens: list[str]) -> str:
    """One glossary table.

    Args:
        tokens: Short names in display order.

    Returns:
        An HTML table with name and reading.
    """
    rows = "".join(
        f"<tr><td>{html.escape(token)}</td><td>{html.escape(ACRONYMS[token].name)}</td>"
        f"<td>{html.escape(ACRONYMS[token].reading)}</td></tr>"
        for token in tokens)
    return ('<div class="table-wrap"><table><thead><tr><th>Acronym</th><th>Meaning</th><th>Reading</th></tr>'
            f"</thead><tbody>{rows}</tbody></table></div>")


def _tests_table_html() -> str:
    """HTML table of hypothesis tests that have no short name.

    Returns:
        A two-column table.
    """
    rows = "".join(f"<tr><td>{html.escape(name)}</td><td>{html.escape(reading)}</td></tr>"
                   for name, reading in _STATISTICAL_TESTS)
    return ('<div class="table-wrap"><table><thead><tr><th>Test</th><th>Reading</th></tr></thead>'
            f"<tbody>{rows}</tbody></table></div>")


def _appendix_blocks_html(used: set[str]) -> str:
    """Grouped acronym tables, with the tests that have no short name under Tests.

    Args:
        used: Short names found in the report.

    Returns:
        HTML headings and tables.
    """
    blocks: list[str] = []
    saw_tests = False
    for category, tokens in _glossary_groups(used):
        table = _acronym_table_html(tokens)
        if category == "Tests":
            table += _tests_table_html()
            saw_tests = True
        blocks.append(f"<h3>{html.escape(category)}</h3>{table}")
    if not saw_tests:
        blocks.insert(0, f"<h3>Tests</h3>{_tests_table_html()}")
    return "".join(blocks)


def _insert_html_glossary(page: str, used: set[str]) -> str:
    """Append the appendix and the contents link.

    Args:
        page: Report HTML without the appendix.
        used: Short names found in the report.

    Returns:
        Report HTML with the appendix before the footer.
    """
    section = f'<section id="appendix"><h2>Appendix</h2>{_appendix_blocks_html(used)}</section>'
    page = page.replace("</nav>", '<a href="#appendix">Appendix</a></nav>', 1)
    return page.replace("</main>", section + "</main>", 1)


def _hide_image_sources(page: str) -> tuple[str, list[str]]:
    """Replace embedded PNG payloads with placeholders.

    Args:
        page: Report HTML.

    Returns:
        The page with ``src`` payloads removed, and those payloads in order.
    """
    sources: list[str] = []

    def hold_src(match: re.Match[str]) -> str:
        sources.append(match.group(0))
        return f'src="__SRC_{len(sources) - 1}__"'

    return _SRC_RE.sub(hold_src, page), sources


def _restore_image_sources(page: str, sources: list[str]) -> str:
    """Put embedded PNG payloads back.

    Args:
        page: HTML whose image payloads were replaced.
        sources: Payloads from ``_hide_image_sources``.

    Returns:
        HTML with the original ``src`` attributes.
    """
    for index, source in enumerate(sources):
        page = page.replace(f'src="__SRC_{index}__"', source, 1)
    return page


def _note_html_block(block: str, introduced: set[str], used: set[str]) -> None:
    """Record short names in one table or plot.

    Args:
        block: A ``<table>`` or ``<img>`` element.
        introduced: Short names already seen.
        used: Short names found in this page.
    """
    if block.lower().startswith("<table"):
        _scan(_TAG_RE.sub(" ", block), introduced, used)
    elif block.lower().startswith("<img"):
        _scan(_img_terms(block), introduced, used)


def _walk_html(page: str) -> tuple[str, set[str]]:
    """Expand first prose mentions and record table and plot short names.

    Args:
        page: Shortened report HTML, with image payloads restored.

    Returns:
        The walked HTML and the short names it uses.
    """
    introduced: set[str] = set()
    used: set[str] = set()
    parts: list[str] = []
    position = 0
    for match in _HTML_BLOCK_RE.finditer(page):
        parts.append(_expand_prose(page[position:match.start()], introduced, used))
        _note_html_block(match.group(0), introduced, used)
        parts.append(match.group(0))
        position = match.end()
    parts.append(_expand_prose(page[position:], introduced, used))
    return "".join(parts), used


def apply_acronyms_html(page: str) -> str:
    """Shorten names, expand the first prose mention, and append the acronym dictionary.

    Args:
        page: Standalone report HTML.

    Returns:
        The page with short names and a final acronym section.

    Raises:
        TypeError: If ``page`` is not a string.
    """
    if not isinstance(page, str):
        raise TypeError(f"page must be str, got {type(page).__name__}.")
    page = re.sub(r'<section id="(?:acronyms|appendix)">.*?</section>', "", page, count=1, flags=re.S)
    page = page.replace('<a href="#acronyms">Acronyms</a>', "", 1)
    page = page.replace('<a href="#appendix">Appendix</a>', "", 1)
    hidden, sources = _hide_image_sources(page)
    walked, used = _walk_html(_restore_image_sources(_shorten(hidden), sources))
    return walked if not used else _insert_html_glossary(walked, used)


def _split_markdown_tables(text: str) -> list[tuple[str, str]]:
    """Split a markdown fragment into prose and pipe tables.

    Args:
        text: Markdown with no fenced code.

    Returns:
        ``(kind, chunk)`` pairs in document order. ``kind`` is ``prose`` or ``table``.
    """
    segments: list[tuple[str, str]] = []
    buffer: list[str] = []
    kind = "prose"
    for line in text.splitlines(keepends=True):
        line_kind = "table" if _TABLE_LINE_RE.match(line.rstrip("\n")) else "prose"
        if buffer and line_kind != kind:
            segments.append((kind, "".join(buffer)))
            buffer = []
        kind = line_kind
        buffer.append(line)
    if buffer:
        segments.append((kind, "".join(buffer)))
    return segments


def _read_fence(lines: list[str], index: int) -> tuple[str, int]:
    """Read one fenced code block.

    Args:
        lines: Markdown lines, with their line endings.
        index: Index of the opening fence.

    Returns:
        The fence text, including both fences, and the index after it.
    """
    code = [lines[index]]
    index += 1
    while index < len(lines) and not lines[index].startswith("```"):
        code.append(lines[index])
        index += 1
    if index < len(lines):
        code.append(lines[index])
        index += 1
    return "".join(code), index


def _markdown_segments(text: str) -> list[tuple[str, str]]:
    """Split markdown into code, tables, and prose.

    Args:
        text: A whole markdown document.

    Returns:
        ``(kind, chunk)`` pairs. ``kind`` is ``code``, ``table``, or ``prose``.
    """
    lines = text.splitlines(keepends=True)
    segments: list[tuple[str, str]] = []
    buffer: list[str] = []
    index = 0
    while index < len(lines):
        if not lines[index].startswith("```"):
            buffer.append(lines[index])
            index += 1
            continue
        if buffer:
            segments.extend(_split_markdown_tables("".join(buffer)))
            buffer = []
        chunk, index = _read_fence(lines, index)
        segments.append(("code", chunk))
    if buffer:
        segments.extend(_split_markdown_tables("".join(buffer)))
    return segments


def _markdown_tests_table() -> str:
    """Markdown table of hypothesis tests that have no short name.

    Returns:
        A two-column table.
    """
    body = "\n".join(f"| {name} | {reading} |" for name, reading in _STATISTICAL_TESTS)
    return f"\n| Test | Reading |\n|---|---|\n{body}\n"


def _markdown_glossary(used: set[str]) -> str:
    """Markdown appendix.

    Args:
        used: Short names found in the document.

    Returns:
        A final ``## Appendix`` section, or an empty string when nothing was used.
    """
    groups = _glossary_groups(used)
    if not groups:
        return ""
    parts = ["\n## Appendix\n"]
    saw_tests = False
    for category, tokens in groups:
        body = "\n".join(
            f"| {token} | {ACRONYMS[token].name} | {ACRONYMS[token].reading} |" for token in tokens)
        extra = ""
        if category == "Tests":
            extra = _markdown_tests_table()
            saw_tests = True
        parts.append(f"\n### {category}\n\n| Acronym | Meaning | Reading |\n|---|---|---|\n{body}\n{extra}")
    if not saw_tests:
        parts.insert(1, f"\n### Tests\n{_markdown_tests_table()}")
    return "".join(parts)


def _rewrite_markdown(text: str) -> tuple[str, set[str]]:
    """Shorten a markdown document and expand first prose mentions.

    Args:
        text: Markdown with any previous acronym section removed.

    Returns:
        The rewritten body and the short names it uses.
    """
    introduced: set[str] = set()
    used: set[str] = set()
    parts: list[str] = []
    for kind, chunk in _markdown_segments(text):
        if kind == "code":
            parts.append(chunk)
            continue
        shortened = _shorten(chunk)
        if kind == "table":
            _scan(shortened, introduced, used)
            parts.append(shortened)
            continue
        parts.append(_expand_prose(shortened, introduced, used))
    return "".join(parts).rstrip() + "\n", used


def apply_acronyms_markdown(text: str) -> str:
    """Use short names in markdown and append every acronym the document uses.

    Args:
        text: Markdown document.

    Returns:
        The document with short names and a final acronym section.

    Raises:
        TypeError: If ``text`` is not a string.
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}.")
    body, used = _rewrite_markdown(re.sub(r"\n## (?:Acronyms|Appendix)\n.*\Z", "\n", text, flags=re.S))
    glossary = _markdown_glossary(used)
    return body if not glossary else body + glossary


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
    """Load a baseline directory written by ``python -m population_drift baseline``."""
    directory = Path(directory)
    for name in ("baseline_statistics.csv", "baseline_summary.csv", "baseline_metadata.json"):
        if not (directory / name).exists():
            raise FileNotFoundError(f"{directory} is not a baseline directory (missing {name}).")
    return BaselineRun(directory, _read_csv(directory / "baseline_statistics.csv"),
                       _read_csv(directory / "baseline_summary.csv"),
                       json.loads((directory / "baseline_metadata.json").read_text()))


def load_comparison_run(directory: str | Path) -> ComparisonRun:
    """Load a comparison directory written by ``python -m population_drift compare``."""
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
    """Coloured label. The text is the level name: severe, moderate, mild, or info."""
    level = level if level in LEVEL_COLORS else "info"
    return f'<span class="badge {level}">{LEVEL_LABELS[level]}</span>'


def _task_errors(run: ComparisonRun, *tasks: str) -> str:
    """Warning listing comparison steps that failed and were skipped.

    Args:
        run: Comparison whose metadata stores ``failed_tasks``.
        *tasks: Task names that belong in this section.

    Returns:
        A warning callout, or an empty string when none of these tasks failed.
    """
    failed = run.metadata.get("failed_tasks") or {}
    if not isinstance(failed, dict):
        return ""
    items = [f"<li><b>{html.escape(task)}</b>: {html.escape(str(failed[task]))}</li>"
             for task in tasks if failed.get(task)]
    if not items:
        return ""
    return ("<div class='callout warn'><b>Skipped.</b> This step failed and the rest of the comparison continued."
            f"<ul>{''.join(items)}</ul></div>")


def _section_with_errors(run: ComparisonRun, anchor: str, title: str, body: str, *tasks: str) -> Section | None:
    """Return a section, keeping it when its tasks failed and it would otherwise be empty.

    Args:
        run: Comparison that records failed tasks.
        anchor: Section id.
        title: Section heading.
        body: Section HTML. Empty when the task produced no table.
        *tasks: Task names whose errors belong here.

    Returns:
        The section, or None when there is neither content nor a failure.
    """
    error = _task_errors(run, *tasks)
    if not body and not error:
        return None
    return Section(anchor, title, error + body)


def _pair_line(run: ComparisonRun, *tasks: str) -> str:
    """One line naming the populations compared by these tasks."""
    pairs = run.metadata.get("task_pairs") or {}
    labels: list[str] = []
    for task in tasks:
        label = pairs.get(task)
        if label and label not in labels:
            labels.append(label)
    if not labels:
        return ""
    return f"<p class='pair'>Pair: {html.escape(' · '.join(labels))}.</p>"


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


def figure_html(fig: plt.Figure, alt: str, terms: str = "") -> str:
    """Embed a Matplotlib figure as a base64 PNG.

    Args:
        fig: Figure to encode. Closed after saving.
        alt: Short description stored on the image.
        terms: Short names drawn on the figure. They are not visible HTML, and the acronym
            dictionary still lists them.

    Returns:
        An ``<img>`` tag with the PNG embedded.
    """
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    listed = f' data-terms="{html.escape(terms)}"' if terms else ""
    return f'<img class="plot" alt="{html.escape(alt)}"{listed} src="data:image/png;base64,{encoded}">'


def _style_axes(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="both", alpha=0.25)


def _as_grid(axes: Any) -> np.ndarray:
    """Return subplot axes as a two-dimensional grid.

    Args:
        axes: Axes, or an array of axes from ``plt.subplots``.

    Returns:
        Array of shape ``(nrows, ncols)``.
    """
    grid = np.atleast_1d(axes)
    if grid.ndim == 1:
        return grid.reshape(1, -1)
    return grid


def _collapse_shared_text(fig: plt.Figure, axes: Any) -> str | None:
    """Promote a title or axis label that every visible panel repeats.

    Args:
        fig: Figure that owns the panels.
        axes: Subplot axes. Turned-off panels are ignored.

    Returns:
        The shared title, or None when panel titles differ.
    """
    grid = _as_grid(axes)
    panels = [ax for ax in grid.flat if ax.axison]
    if len(panels) < 2:
        return None
    titles = [ax.get_title() for ax in panels]
    shared = titles[0] if titles[0] and all(title == titles[0] for title in titles) else None
    if shared:
        fig.suptitle(shared, fontsize=11)
        for ax in panels:
            ax.set_title("")
    _clear_shared_axis_label(panels, "get_ylabel", "set_ylabel", grid[:, 1:])
    if grid.shape[0] > 1:
        _clear_shared_axis_label(panels, "get_xlabel", "set_xlabel", grid[:-1, :])
    return shared


def _clear_shared_axis_label(panels: list[plt.Axes], getter: str, setter: str, targets: np.ndarray) -> None:
    """Clear one axis label on ``targets`` when every visible panel uses the same text.

    Args:
        panels: Visible panels.
        getter: ``get_xlabel`` or ``get_ylabel``.
        setter: ``set_xlabel`` or ``set_ylabel``.
        targets: Panels that should lose the repeated label.
    """
    labels = [getattr(ax, getter)() for ax in panels]
    if not labels[0] or any(label != labels[0] for label in labels):
        return
    for ax in targets.flat:
        if ax.axison:
            getattr(ax, setter)("")


def shap_importance(table: pd.DataFrame) -> pd.DataFrame:
    """SHAP importance as each feature's share of mean absolute SHAP.

    Args:
        table: ``shap_drift`` table with mean absolute SHAP on each population.

    Returns:
        Copy with ``shap_importance_source`` and ``shap_importance_target``. The two columns each sum to 1
        when that population has positive total attribution.

    Raises:
        ValueError: If a required column is missing.
    """
    required = {"feature", "mean_abs_shap_source", "mean_abs_shap_target"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"SHAP table is missing columns: {missing}.")
    frame = table.copy()
    frame["shap_importance_source"] = _attribution_share(frame["mean_abs_shap_source"])
    frame["shap_importance_target"] = _attribution_share(frame["mean_abs_shap_target"])
    return frame


def _attribution_share(values: pd.Series) -> pd.Series:
    """Normalize a non-negative attribution vector to shares.

    Args:
        values: Mean absolute SHAP by feature.

    Returns:
        Shares summing to 1, or zeros when the total is 0.
    """
    total = float(values.sum())
    if total <= 0:
        return pd.Series(np.zeros(len(values)), index=values.index)
    return values / total


def _plot_shap_importance(ax: plt.Axes, shap: pd.DataFrame) -> None:
    """Compare training and new-population SHAP importance.

    Args:
        ax: Panel to draw on.
        shap: ``shap_drift`` table, most important training features first.
    """
    ranked = shap_importance(shap).head(15).iloc[::-1]
    y = np.arange(len(ranked))
    ax.barh(y - 0.2, ranked["shap_importance_source"], 0.4, color=SOURCE_COLOR, label="training")
    ax.barh(y + 0.2, ranked["shap_importance_target"], 0.4, color=TARGET_COLOR, label="new")
    flags = ranked["important"] if "important" in ranked.columns else np.zeros(len(ranked), dtype=bool)
    labels = [f"{feature} *" if flag else str(feature) for feature, flag in zip(ranked["feature"], flags)]
    ax.set_yticks(y, labels, fontsize=8)
    ax.set_xlabel("SHAP importance")
    ax.set_title("SHAP importance")
    ax.legend(frameon=False, fontsize=8)
    _style_axes(ax)


# --------------------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------------------
def plot_feature_overview(run: ComparisonRun) -> str:
    """Population Stability Index (PSI) per feature (coloured by severity, baseline p95 ticks) and drift vs model importance."""
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
    ax.set_xlabel("PSI  (dashed 0.10 / 0.25; tick = baseline p95)")
    ax.set_title("PSI by feature")
    _style_axes(ax)
    if has_importance:
        ax = axes[1]
        ax.scatter(features["model_importance_share"], features["psi"], c=colors, s=60, edgecolor="white")
        for row in features.itertuples():
            if row.severity != "mild" or row.model_importance_share >= features["model_importance_share"].quantile(0.75):
                ax.annotate(row.feature, (row.model_importance_share, row.psi), xytext=(4, 3),
                            textcoords="offset points", fontsize=8)
        ax.axhline(0.10, ls="--", color="#999", lw=1)
        ax.axhline(0.25, ls="--", color="#444", lw=1)
        ax.set_xlabel("SHAP importance")
        ax.set_ylabel("PSI")
        ax.set_title("PSI vs SHAP importance")
        _style_axes(ax)
    _collapse_shared_text(fig, axes)
    fig.tight_layout()
    return figure_html(fig, "Feature drift overview", "PSI SHAP" if has_importance else "PSI")


def plot_ecdfs(run: ComparisonRun, features: Sequence[str]) -> str:
    """empirical cumulative distribution function (ECDF) overlays (from stored quantiles) for selected continuous features."""
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
        ax.set_title(f"{feature}: KS {row['ks_d']:.3f}, PSI {row['psi']:.3f}", fontsize=10)
        ax.set_ylabel("ECDF")
        _style_axes(ax)
    for ax in list(axes.flat)[len(features):]:
        ax.axis("off")
    axes.flat[0].legend(frameon=False, fontsize=8)
    _collapse_shared_text(fig, axes)
    fig.tight_layout()
    return figure_html(fig, "ECDF comparison", "KS PSI ECDF")


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
        ax.set_title(f"{feature}: Cramér's V {stats.loc[feature, 'cramers_v']:.3f}, PSI {stats.loc[feature, 'psi']:.3f}", fontsize=10)
        ax.set_ylabel("share")
        _style_axes(ax)
    for ax in list(axes.flat)[len(features):]:
        ax.axis("off")
    axes.flat[0].legend(frameon=False, fontsize=8)
    _collapse_shared_text(fig, axes)
    fig.tight_layout()
    return figure_html(fig, "Category frequencies", "PSI")


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
    _collapse_shared_text(fig, axes)
    fig.suptitle(title, fontsize=11, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=1.6)
    terms = " ".join([title, *[str(entry["title"]) for entry in entries]])
    return figure_html(fig, title, terms)


def plot_domain_classifier(run: ComparisonRun, baseline: BaselineRun | None) -> str:
    """Domain-classifier score distributions, area under the curve (AUC) vs baseline null, and separation drivers."""
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
    axes[0].set_xlabel("OOF P(new | x)")
    axes[0].set_title("Scores by population", fontsize=10)
    axes[0].legend(frameon=False, fontsize=8)
    _style_axes(axes[0])
    null = baseline.null("domain_classifier", "auc") if baseline else np.array([])
    _null_panel(axes[1], null, _scalar(main, "auc"), f"AUC {_scalar(main, 'auc'):.3f} vs baseline")
    axes[1].legend(frameon=False, fontsize=7)
    axes[2].barh(importance["feature"], importance["gain_share"], color="#475569")
    axes[2].set_xlabel("gain share")
    axes[2].set_title("Separation drivers", fontsize=10)
    _style_axes(axes[2])
    _collapse_shared_text(fig, axes)
    fig.tight_layout()
    return figure_html(fig, "Classifier two-sample test", "AUC OOF")


def _lower_triangle(delta: np.ndarray) -> np.ma.MaskedArray:
    """Hide the diagonal and the upper half of a symmetric matrix.

    Args:
        delta: Square correlation-change matrix.

    Returns:
        The strict lower triangle. The other cells are masked.

    Raises:
        ValueError: If ``delta`` is not square.
    """
    if delta.ndim != 2 or delta.shape[0] != delta.shape[1]:
        raise ValueError(f"Correlation change must be square, got shape {delta.shape}.")
    mask = np.triu(np.ones(delta.shape, dtype=bool), k=0)
    return np.ma.masked_where(mask, delta)


def plot_correlation_delta(run: ComparisonRun) -> str:
    """Lower triangle of the Spearman correlation change (new − training)."""
    data = run.plot_data.get("correlation")
    if not data or not data.get("columns"):
        return ""
    columns = data["columns"]
    delta = np.array(data["target_corr"], dtype=float) - np.array(data["source_corr"], dtype=float)
    shown = _lower_triangle(delta)
    size = max(4.5, 0.42 * len(columns) + 2)
    fig, ax = plt.subplots(figsize=(size + 1, size))
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("white")
    image = ax.imshow(shown, cmap=cmap, vmin=-0.6, vmax=0.6)
    ax.set_xticks(range(len(columns)), columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(columns)), columns, fontsize=8)
    if len(columns) <= 16:
        for i, j in zip(*np.tril_indices(len(columns), k=-1)):
            if abs(delta[i, j]) >= 0.1:
                ax.text(j, i, f"{delta[i, j]:+.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, shrink=0.8, label="Δρ (new − training)")
    ax.set_title("Δ correlation", fontsize=10)
    fig.tight_layout()
    return figure_html(fig, "Correlation change")


def plot_support(run: ComparisonRun) -> str:
    """Isolation-Forest score distributions and per-feature out-of-range shares."""
    outliers = run.plot_data.get("joint_outliers")
    coverage = run.table("range_coverage")
    panels = int(bool(outliers)) + int(not coverage.empty)
    if not panels:
        return ""
    fig, grid = plt.subplots(1, panels, figsize=(6.5 * panels, 3.6), squeeze=False)
    axes = list(grid.flat)
    if outliers:
        ax = axes.pop(0)
        bins = np.array(outliers["bins"])
        centers, width = (bins[:-1] + bins[1:]) / 2, bins[1] - bins[0]
        ax.bar(centers, outliers["source_hist"], width, color=SOURCE_COLOR, alpha=0.55, label="training (held-out)")
        ax.bar(centers, outliers["target_hist"], width, color=TARGET_COLOR, alpha=0.55, label="new")
        ax.axvline(outliers["threshold"], color="black", ls="--", lw=1, label="OOD threshold")
        ax.set_xlabel("IF score (lower = more unusual)")
        ax.set_title("Joint novelty", fontsize=10)
        ax.legend(frameon=False, fontsize=8)
        _style_axes(ax)
    if not coverage.empty:
        ax = axes.pop(0)
        cov = coverage.sort_values("outside_share")
        ax.barh(cov["feature"], cov["outside_share"], color=np.where(cov["ratio_to_expected"] > 2, TARGET_COLOR, "#94a3b8"))
        ax.axvline(cov["expected_share"].iloc[0], color="black", ls="--", lw=1, label="expected under no shift")
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_xlabel("share outside training 0.5–99.5%")
        ax.set_title("Outside training range", fontsize=10)
        ax.legend(frameon=False, fontsize=8)
        _style_axes(ax)
    _collapse_shared_text(fig, grid)
    fig.tight_layout()
    return figure_html(fig, "Support checks", "OOD")


def plot_model_outputs(run: ComparisonRun) -> str:
    """Score distributions and SHAP importance for the training and new populations."""
    prediction = run.plot_data.get("prediction_drift")
    shap = run.table("shap_drift")
    panels = int(bool(prediction)) + int(not shap.empty)
    if not panels:
        return ""
    fig, grid = plt.subplots(1, panels, figsize=(6.5 * panels, max(3.6, 0.28 * min(len(shap), 15) + 1)), squeeze=False)
    axes = list(grid.flat)
    if prediction:
        ax = axes.pop(0)
        bins = np.array(prediction["bins"])
        centers, width = (bins[:-1] + bins[1:]) / 2, bins[1] - bins[0]
        ax.bar(centers, prediction["source_hist"], width, color=SOURCE_COLOR, alpha=0.55, label="training")
        ax.bar(centers, prediction["target_hist"], width, color=TARGET_COLOR, alpha=0.55, label="new")
        if prediction.get("threshold") is not None:
            ax.axvline(prediction["threshold"], color="black", ls="--", lw=1, label="threshold")
        ax.set_xlabel("score")
        ax.set_title("Score distribution", fontsize=10)
        ax.legend(frameon=False, fontsize=8)
        _style_axes(ax)
    if not shap.empty:
        _plot_shap_importance(axes.pop(0), shap)
    _collapse_shared_text(fig, grid)
    fig.tight_layout()
    return figure_html(fig, "Model outputs", "SHAP")


def plot_performance(run: ComparisonRun) -> str:
    """Reference vs estimated metric with 95% confidence interval (CI) for each estimation method."""
    rows = []
    for task, label in (("weighted_performance", "weighting"), ("cbpe", "CBPE")):
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
        ax.set_title(PLOT_METRIC.get(str(metric), str(metric)), fontsize=10)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        ax.tick_params(axis="x", labelsize=8)
        _style_axes(ax)
    for ax in list(axes.flat)[len(metrics):]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False, fontsize=8, ncol=2)
    _collapse_shared_text(fig, axes)
    fig.suptitle("Reference vs estimate (95% CI)", fontsize=11, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    labels = " ".join(PLOT_METRIC.get(str(metric), str(metric)) for metric in metrics)
    return figure_html(fig, "Estimated performance", f"CI CBPE {labels}")


def plot_calibration(run: ComparisonRun) -> str:
    """Reliability diagram on the training population."""
    reliability = run.table("calibration", "reliability")
    if reliability.empty:
        return ""
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    ax.plot([0, 1], [0, 1], ls="--", color="#999", lw=1)
    ax.plot(reliability["mean_predicted"], reliability["observed_rate"], marker="o", color=SOURCE_COLOR)
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("positive rate")
    ax.set_title("Calibration", fontsize=10)
    _style_axes(ax)
    fig.tight_layout()
    return figure_html(fig, "Calibration")


def plot_baseline_feature_variability(baseline: BaselineRun) -> str:
    """95th percentile of per-feature Population Stability Index (PSI) and Kolmogorov–Smirnov statistic (KS D) across baseline rounds."""
    summary = baseline.summary
    psi = summary[(summary["metric"] == "psi") & summary["task"].isin(["continuous", "categorical"])]
    if psi.empty:
        return ""
    psi = psi.sort_values("p95")
    fig, ax = plt.subplots(figsize=(7.5, max(3.5, 0.3 * len(psi) + 1)))
    ax.barh(psi["feature"], psi["p95"], color="#94a3b8", label="p95")
    ax.scatter(psi["p50"], psi["feature"], color="black", s=14, zorder=3, label="median")
    ax.axvline(0.10, ls="--", color="#999", lw=1)
    ax.set_xlabel("PSI on random training splits")
    ax.legend(frameon=False, fontsize=8)
    _style_axes(ax)
    fig.tight_layout()
    return figure_html(fig, "Baseline per-feature PSI", "PSI KS")


def global_null_entries(baseline: BaselineRun | None, run: ComparisonRun | None) -> list[dict[str, Any]]:
    """Entries for the observed-vs-baseline grid of global statistics."""
    entries = []
    observed = {}
    if run is not None and not run.baseline_comparison.empty:
        bc = run.baseline_comparison[run.baseline_comparison["feature"] == GLOBAL_FEATURE]
        observed = {(r.task, r.metric): r.observed for r in bc.itertuples()}
    for (task, metric), label in PLOT_GLOBAL_LABELS.items():
        null = baseline.null(task, metric) if baseline else np.array([])
        value = observed.get((task, metric), np.nan)
        if len(null) or np.isfinite(value):
            entries.append({"title": label, "null": null, "observed": float(value) if pd.notna(value) else np.nan})
    return entries


_CONTINUOUS_BASELINE = (("psi", "PSI"), ("ks_d", "KS D"), ("wasserstein_norm", "Wasserstein/SD"))
_CATEGORICAL_BASELINE = (("psi", "PSI"), ("cramers_v", "Cramér's V"), ("js_distance", "JS"))


def _reference_name(signal: dict[str, Any]) -> str:
    """Name the population whose random splits produced this signal's baseline.

    Args:
        signal: One decision signal, possibly with a ``pair`` field.

    Returns:
        ``fit set``, ``evaluation set``, ``labelled set``, or ``reference``.
    """
    pair = str(signal.get("pair") or "")
    if pair.startswith("evaluation"):
        return "evaluation set"
    if pair.startswith("fit"):
        return "fit set"
    if pair.startswith("labelled"):
        return "labelled set"
    return "reference"


def _sampling_rationale(run: ComparisonRun, signal: dict[str, Any], statistic: str) -> str:
    """Explain a 95th percentile taken from random splits of the reference.

    Args:
        run: Comparison whose decision records the baseline size.
        signal: Signal whose pair names the reference.
        statistic: Statistic the percentile belongs to.

    Returns:
        One sentence stating the split design and that the percentile is sampling noise.
    """
    rounds = run.decision.get("baseline_rounds")
    size = run.decision.get("baseline_subset_size")
    split = f"{int(rounds)} random splits" if isinstance(rounds, (int, float)) and not isinstance(rounds, bool) else "random splits"
    held = f", each holding out {int(size)} rows" if isinstance(size, (int, float)) and not isinstance(size, bool) else ""
    return (f"95th percentile of {statistic} on {split} of the {_reference_name(signal)}{held}. "
            "That percentile is sampling noise at this comparison size, under exchangeability inside the reference.")


def _threshold(run: ComparisonRun, name: str, default: float) -> float:
    """Read one interpretation cutoff from the run config.

    Args:
        run: Comparison whose metadata stores ``config.thresholds``.
        name: Threshold field name.
        default: Value used when the run did not record the field.

    Returns:
        The cutoff as a float.
    """
    thresholds = (run.metadata.get("config") or {}).get("thresholds") or {}
    try:
        return float(thresholds.get(name, default))
    except (TypeError, ValueError):
        return default


def _baseline_frame_row(run: ComparisonRun, task: str, metric: str, feature: str) -> pd.Series | None:
    """One stored baseline row, or None when the percentile is missing.

    Args:
        run: Comparison that holds ``baseline_comparison``.
        task: Task name in that table.
        metric: Metric name in that table.
        feature: Feature name, or ``__global__``.

    Returns:
        The matching row, or None.
    """
    bc = run.baseline_comparison
    if bc.empty or "baseline_threshold" not in bc.columns:
        return None
    rows = bc[(bc["task"] == task) & (bc["metric"] == metric) & (bc["feature"].astype(str) == str(feature))]
    if rows.empty or not np.isfinite(rows.iloc[0]["baseline_threshold"]):
        return None
    return rows.iloc[0]


def _baseline_number(metric: str, value: float) -> str:
    """Format a baseline statistic. Shares are percents; other metrics use the report formatter.

    Args:
        metric: Metric name from ``baseline_comparison``.
        value: Numeric baseline value.

    Returns:
        Display text.
    """
    if metric in {"ood_share", "outside_share"}:
        return _pct(value)
    return _fmt(value)


def _percentile_text(label: str, row: pd.Series, metric: str = "") -> str:
    """Format one 95th percentile with its median and round count.

    Args:
        label: Display name of the statistic.
        row: One ``baseline_comparison`` row.
        metric: Metric name, used to choose percent formatting for shares.

    Returns:
        A single phrase, for example ``AUC 95th percentile 0.532, median 0.513, 10 rounds``.
    """
    rounds = row["n_baseline_rounds"] if "n_baseline_rounds" in row.index else np.nan
    median = row["baseline_median"] if "baseline_median" in row.index else np.nan
    median_text = (f", median {_baseline_number(metric, float(median))}"
                   if pd.notna(median) and np.isfinite(median) else "")
    rounds_text = f", {int(rounds)} rounds" if pd.notna(rounds) else ""
    threshold = _baseline_number(metric, float(row["baseline_threshold"]))
    return f"{label} 95th percentile {threshold}{median_text}{rounds_text}"


def _global_baseline(run: ComparisonRun, signal: dict[str, Any], specs: Sequence[tuple[str, str, str]],
                     statistic: str) -> tuple[str, str]:
    """Percentiles for global statistics, plus the sampling-noise rationale.

    Args:
        run: Comparison holding the baseline table.
        signal: Signal whose pair names the reference.
        specs: ``(task, metric, label)`` triples.
        statistic: Words used in the rationale for those statistics.

    Returns:
        Value text and rationale text.
    """
    parts = []
    for task, metric, label in specs:
        row = _baseline_frame_row(run, task, metric, GLOBAL_FEATURE)
        if row is not None:
            parts.append(_percentile_text(label, row, metric))
    if not parts:
        return "No sampling baseline was stored for this signal.", _sampling_rationale(run, signal, statistic)
    return ". ".join(parts), _sampling_rationale(run, signal, statistic)


def _feature_percentiles(run: ComparisonRun, task: str, metric: str, features: Sequence[str]) -> str:
    """95th percentiles for a list of features.

    Args:
        run: Comparison holding the baseline table.
        task: Task name.
        metric: Metric name.
        features: Feature names to look up.

    Returns:
        ``feature percentile`` phrases joined by commas, or an empty string.
    """
    parts = []
    for feature in features:
        row = _baseline_frame_row(run, task, metric, feature)
        if row is not None:
            parts.append(f"{feature} {_baseline_number(metric, float(row['baseline_threshold']))}")
    return ", ".join(parts)


def _names_in_detail(detail: str) -> list[str]:
    """Feature names written at the start of a signal detail.

    Args:
        detail: Signal detail text.

    Returns:
        Names before the first space of each comma-separated piece. Empty when nothing was flagged.
    """
    if not detail or detail.startswith("no ") or detail == "none":
        return []
    names = []
    for part in detail.split(","):
        token = part.strip().split(" ")[0]
        if token and token not in {"none", "within", "beyond"}:
            names.append(token)
    return names


def _baseline_feature_drift(signal: dict[str, Any], run: ComparisonRun, *, low_importance: bool) -> tuple[str, str]:
    """Per-feature percentiles used by a feature-drift signal.

    Args:
        signal: Feature-drift signal.
        run: Comparison holding the feature assessment and the baseline table.
        low_importance: Whether the signal is the low-importance group.

    Returns:
        Percentiles of the flagged features, and why those percentiles are the baseline.
    """
    features = run.features
    if features.empty or "severity" not in features.columns:
        return "–", "No feature assessment was stored."
    chosen = features[features["severity"] != "mild"]
    if "important" in features.columns:
        chosen = chosen[~features.loc[chosen.index, "important"] if low_importance else features.loc[chosen.index, "important"]]
    lines = []
    for row in chosen.itertuples():
        task, metrics = ("categorical", _CATEGORICAL_BASELINE) if row.type == "categorical" else ("continuous", _CONTINUOUS_BASELINE)
        bits = [f"{label} {_fmt(float(found['baseline_threshold']))}"
                for metric, label in metrics
                if (found := _baseline_frame_row(run, task, metric, row.feature)) is not None]
        if bits:
            lines.append(f"{row.feature}: {', '.join(bits)}")
    value = "; ".join(lines) if lines else "No flagged feature has a stored percentile."
    rationale = _sampling_rationale(
        run, signal, "PSI, KS D, and Wasserstein distance / SD, or PSI, Cramér's V, and JS distance")
    return value, rationale + " A feature is beyond baseline when any of those statistics exceeds its own percentile."


def _baseline_low_importance(signal: dict[str, Any], run: ComparisonRun) -> tuple[str, str]:
    """Baseline for the low-importance feature-drift signal.

    Args:
        signal: The low-importance signal.
        run: Comparison holding the feature assessment.

    Returns:
        Percentiles and rationale for the non-important flagged features.
    """
    return _baseline_feature_drift(signal, run, low_importance=True)


def _baseline_important_features(signal: dict[str, Any], run: ComparisonRun) -> tuple[str, str]:
    """Baseline for the model-important feature-drift signal.

    Args:
        signal: The model-important, or no-model, feature-drift signal.
        run: Comparison holding the feature assessment.

    Returns:
        Percentiles and rationale for the important flagged features.
    """
    return _baseline_feature_drift(signal, run, low_importance=False)


def _baseline_from_detail(signal: dict[str, Any], run: ComparisonRun, task: str, metric: str, label: str,
                          statistic: str) -> tuple[str, str]:
    """Percentiles for the features named in a signal detail.

    Args:
        signal: Signal whose detail names features.
        run: Comparison holding the baseline table.
        task: Baseline task.
        metric: Baseline metric.
        label: Display name of the metric.
        statistic: Words used in the sampling rationale.

    Returns:
        Percentiles for the named features, and the sampling rationale.
    """
    listed = _feature_percentiles(run, task, metric, _names_in_detail(str(signal.get("detail") or "")))
    value = f"{label} 95th percentile: {listed}" if listed else "No flagged feature has a stored percentile."
    return value, _sampling_rationale(run, signal, statistic)


def _baseline_missing(signal: dict[str, Any], run: ComparisonRun) -> tuple[str, str]:
    """Baseline for the missingness signal.

    Args:
        signal: Missing-value signal.
        run: Comparison holding the baseline table.

    Returns:
        |Cohen's h| percentiles and the sampling rationale, including the 0.20 convention.
    """
    value, rationale = _baseline_from_detail(
        signal, run, "missingness", "cohens_h_abs", "|Cohen's h|", "absolute Cohen's h")
    cutoff = _threshold(run, "cohens_h_small", 0.20)
    return value, rationale + f" The level also requires |h| ≥ {_fmt(cutoff)} (Cohen 1988)."


def _baseline_range(signal: dict[str, Any], run: ComparisonRun) -> tuple[str, str]:
    """Baseline for values outside the training quantiles.

    Args:
        signal: Range-coverage signal.
        run: Comparison holding the baseline table.

    Returns:
        Outside-share percentiles and why they are compared with twice the expected tail.
    """
    value, rationale = _baseline_from_detail(
        signal, run, "range_coverage", "outside_share", "outside-share",
        "the share of values outside the reference 0.5%–99.5% quantiles")
    ratio = _threshold(run, "range_outside_ratio", 2.0)
    return value, rationale + f" The level also requires more than {_fmt(ratio)} times the expected tail share."


def _baseline_unseen(signal: dict[str, Any], run: ComparisonRun) -> tuple[str, str]:
    """Cutoff for unseen category levels.

    Args:
        signal: Unseen-level signal. Unused except to keep the dispatcher uniform.
        run: Comparison whose config stores the cutoff.

    Returns:
        The share cutoff and why it is not a sampling percentile.
    """
    del signal
    share = _threshold(run, "unseen_level_share", 0.01)
    return (f"Unseen-level share {_pct(share)}",
            "A feature is flagged above this share of new rows. LightGBM sends an unseen level down a default path. "
            "The cutoff is fixed (Thresholds.unseen_level_share). It is not a sampling percentile.")


def _baseline_domain(signal: dict[str, Any], run: ComparisonRun) -> tuple[str, str]:
    """Baseline for the domain-classifier AUC.

    Args:
        signal: Domain-classifier signal.
        run: Comparison holding the baseline table.

    Returns:
        AUC percentile and the mild/moderate/severe bands applied after it.
    """
    value, rationale = _global_baseline(run, signal, [("domain_classifier", "auc", "AUC")], "domain-classifier AUC")
    mild, strong = _threshold(run, "domain_auc_mild", 0.55), _threshold(run, "domain_auc_strong", 0.80)
    return value, rationale + (f" Within this percentile the level is mild. Beyond it, AUC ≤ {_fmt(mild)} is mild, "
                               f"through {_fmt(strong)} is moderate, and above that is severe.")


def _baseline_kernel(signal: dict[str, Any], run: ComparisonRun) -> tuple[str, str]:
    """Baseline for the kernel and energy tests.

    Args:
        signal: Kernel/energy signal.
        run: Comparison holding the baseline table.

    Returns:
        MMD and energy percentiles. The rationale states this signal cannot be severe.
    """
    specs = (("mmd", "mmd2", "MMD²"), ("energy", "energy_distance", "energy distance"))
    value, rationale = _global_baseline(run, signal, specs, "squared MMD and energy distance")
    return value, rationale + " Moderate only when the permutation p-value is also below α/2. This signal cannot be severe."


def _baseline_dependence(signal: dict[str, Any], run: ComparisonRun) -> tuple[str, str]:
    """Baseline for the Spearman dependence signal.

    Args:
        signal: Dependence signal.
        run: Comparison holding the baseline table.

    Returns:
        Frobenius percentile, plus the separate fixed cutoff for one pair.
    """
    value, rationale = _global_baseline(run, signal, [("correlation", "frobenius", "Frobenius")],
                                        "the Spearman upper-triangle Frobenius norm")
    delta = _threshold(run, "correlation_pair_delta", 0.20)
    return value, rationale + (f" An important pair with |Δρ| > {_fmt(delta)} is a separate fixed cutoff, "
                               "not this percentile.")


def _baseline_ood(signal: dict[str, Any], run: ComparisonRun) -> tuple[str, str]:
    """Baseline for the out-of-support share.

    Args:
        signal: Isolation-forest signal.
        run: Comparison holding the baseline table.

    Returns:
        Share percentile. The rationale separates it from the 1% score quantile and the 5% band.
    """
    value, rationale = _global_baseline(run, signal, [("joint_outliers", "ood_share", "out-of-support share")],
                                        "the isolation-forest out-of-support share")
    material = _threshold(run, "ood_share_material", 0.05)
    return value, rationale + (f" The expected share in the observed column is the forest's score quantile, not this percentile. "
                               f"Severe when the share also exceeds {_pct(material)}.")


def _baseline_score(signal: dict[str, Any], run: ComparisonRun) -> tuple[str, str]:
    """Baseline for the model-score distribution.

    Args:
        signal: Score-distribution signal.
        run: Comparison holding the baseline table.

    Returns:
        Score PSI and KS percentiles, plus the fixed PSI bands.
    """
    specs = (("prediction_drift", "psi", "score PSI"), ("prediction_drift", "ks_d", "score KS D"))
    value, rationale = _global_baseline(run, signal, specs, "score PSI and score KS D")
    moderate, major = _threshold(run, "psi_moderate", 0.10), _threshold(run, "psi_major", 0.25)
    return value, rationale + (f" Mild also when PSI < {_fmt(moderate)}; severe when PSI > {_fmt(major)} (Siddiqi 2006).")


def _baseline_positive_rate(signal: dict[str, Any], run: ComparisonRun) -> tuple[str, str]:
    """Cutoff for the predicted-positive-rate signal.

    Args:
        signal: Positive-rate signal.
        run: Comparison holding the baseline table.

    Returns:
        Cohen's h cutoff. The stored sampling percentile is reported as unused by this level.
    """
    cutoff = _threshold(run, "cohens_h_small", 0.20)
    row = _baseline_frame_row(run, "prediction_drift", "positive_rate_cohens_h_abs", GLOBAL_FEATURE)
    stored = (f" Stored sampling baseline: {_percentile_text('|h|', row)} on the {_reference_name(signal)}."
              if row is not None else "")
    return (f"|Cohen's h| {_fmt(cutoff)}",
            "The level is moderate at or above this value when the two-proportion p-value is below α (Cohen 1988). "
            "It does not use the sampling percentile." + stored)


def _baseline_shap(signal: dict[str, Any], run: ComparisonRun) -> tuple[str, str]:
    """Baseline for Shapley-attribution drift.

    Args:
        signal: SHAP signal.
        run: Comparison holding the baseline table.

    Returns:
        Rank-deficit percentile and per-feature KS percentiles, with both gates named.
    """
    value, rationale = _global_baseline(run, signal, [("shap_drift", "rank_rho_deficit", "1 − rank ρ")],
                                        "1 minus the rank correlation of mean absolute SHAP")
    listed = _feature_percentiles(run, "shap_drift", "shap_ks_d", _names_in_detail(str(signal.get("detail") or "")))
    if listed:
        value = f"{value}. SHAP KS D 95th percentile: {listed}"
    rho, ks = _threshold(run, "importance_rank_rho", 0.80), _threshold(run, "ks_moderate", 0.10)
    return value, rationale + (f" Also moderate when rank ρ < {_fmt(rho)}, or an important feature has "
                               f"KS D ≥ {_fmt(ks)}, adjusted p < α, and KS D above its own percentile.")


def _baseline_ess(signal: dict[str, Any], run: ComparisonRun) -> tuple[str, str]:
    """Cutoffs for the Kish effective-sample-size ratio.

    Args:
        signal: Overlap signal.
        run: Comparison holding the baseline table.

    Returns:
        The 0.50 and 0.10 bands. The stored deficit percentile is reported as unused.
    """
    good, poor = _threshold(run, "ess_ratio_good", 0.50), _threshold(run, "ess_ratio_poor", 0.10)
    row = _baseline_frame_row(run, "weighted_performance", "ess_deficit", GLOBAL_FEATURE)
    stored = (f" Stored sampling baseline: {_percentile_text('1 − ESS ratio', row)} on the {_reference_name(signal)}."
              if row is not None else "")
    return (f"ESS ratio {_fmt(good)} and {_fmt(poor)}",
            f"Mild above {_fmt(good)}, moderate down to {_fmt(poor)}, severe below {_fmt(poor)}. "
            "These are fixed Kish bands, not sampling percentiles." + stored)


def _baseline_performance(signal: dict[str, Any], run: ComparisonRun) -> tuple[str, str]:
    """Tolerance bound for a label-free performance estimate.

    Args:
        signal: Weighting or CBPE signal.
        run: Comparison whose performance table holds the reference metric.

    Returns:
        Reference metric and tolerance, and why the tolerance is not a sampling percentile.
    """
    task = "cbpe" if "CBPE" in signal["signal"] else "weighted_performance"
    table = run.table(task)
    if table.empty or "reference_value" not in table.columns:
        return "–", "No performance table was stored, so the tolerance bound cannot be shown."
    metric_name = "auc"
    if "metric" in table.columns and (table["metric"] == "auc").any():
        primary = table.set_index("metric").loc["auc"]
    else:
        primary = table.iloc[0]
        metric_name = str(primary["metric"]) if "metric" in primary.index else metric_name
    reference = float(primary["reference_value"])
    tolerance = float((run.metadata.get("config") or {}).get("performance_tolerance", 0.02))
    shown = METRIC_DISPLAY.get(metric_name, metric_name)
    relative = (run.metadata.get("config") or {}).get("problem_type") == "regression"
    kind = f"{_fmt(tolerance)} times the reference error" if relative else f"{_fmt(tolerance)} absolute {shown}"
    return (f"Reference {shown} {_fmt(reference)}; tolerance {_fmt(tolerance)}",
            f"The material change is {kind}. Severe only when the whole 95% confidence interval is beyond it. "
            "performance_tolerance is set for the deployment. It is not a percentile of the baseline rounds.")


def _baseline_bbse(signal: dict[str, Any], run: ComparisonRun) -> tuple[str, str]:
    """Comparison value for the class-prior estimate.

    Args:
        signal: BBSE signal. Unused except to keep the dispatcher uniform.
        run: Comparison whose label-shift table holds the source prevalence.

    Returns:
        Source prevalence, and why an interval that excludes it is a shift.
    """
    del signal
    prevalence = _scalar(run.table("label_shift"), "source_prevalence")
    shown = _pct(prevalence) if np.isfinite(prevalence) else "–"
    return (f"Source prevalence {shown}",
            "Moderate when the 95% interval for the new prevalence excludes this value. "
            "The estimator assumes P(x|y) is unchanged. No sampling baseline is computed for it.")


def _baseline_calibration(signal: dict[str, Any], run: ComparisonRun) -> tuple[str, str]:
    """ECE cutoff that downgrades confidence-based performance estimation.

    Args:
        signal: Calibration signal. Unused except to keep the dispatcher uniform.
        run: Comparison whose config stores the ECE cutoff.

    Returns:
        The ECE cutoff and why exceeding it changes the CBPE signal to info.
    """
    del signal
    cutoff = _threshold(run, "ece_max", 0.05)
    return (f"ECE {_fmt(cutoff)}",
            "Expected calibration error above this downgrades confidence-based performance estimation to info, "
            "because that estimator needs calibrated probabilities. The cutoff is fixed. It is not a sampling percentile.")


_SIGNAL_BASELINES = (
    ("low-importance", _baseline_low_importance),
    ("Feature drift", _baseline_important_features),
    ("Missing-value rates", _baseline_missing),
    ("Unseen category levels", _baseline_unseen),
    ("classifier two-sample", _baseline_domain),
    ("kernel/energy", _baseline_kernel),
    ("Dependence structure", _baseline_dependence),
    ("Out-of-support", _baseline_ood),
    ("Values outside", _baseline_range),
    ("Model score distribution", _baseline_score),
    ("Predicted positive rate", _baseline_positive_rate),
    ("Model attributions", _baseline_shap),
    ("effective sample size", _baseline_ess),
    ("importance weighting", _baseline_performance),
    ("calibration", _baseline_calibration),
    ("CBPE", _baseline_performance),
    ("BBSE", _baseline_bbse),
)


def signal_baseline(signal: dict[str, Any], run: ComparisonRun) -> tuple[str, str]:
    """Baseline number for one signal, and why that number is the comparison.

    Args:
        signal: One entry of ``decision["signals"]``.
        run: Comparison that holds the baseline table, config, and task tables.

    Returns:
        Value text and rationale text. A signal with no mapped comparison says so.
    """
    name = str(signal.get("signal") or "")
    for needle, builder in _SIGNAL_BASELINES:
        if needle in name:
            return builder(signal, run)
    return "–", "This signal has no baseline comparison."


def _concise_baseline(value: str) -> str:
    """Reduce a baseline value to the comparison bar.

    Args:
        value: Full text from :func:`signal_baseline`.

    Returns:
        A short cell string. Per-feature lists stay out of the summary table.
    """
    if "95th percentile" not in value:
        return "p95 per feature" if re.match(r"^[A-Za-z0-9_]+: ", value) else value
    parts = []
    for piece in value.split(". "):
        piece = re.sub(r", median [^,]*", "", piece)
        piece = re.sub(r", \d+ rounds", "", piece)
        if "95th percentile:" in piece:
            label = piece.split("95th percentile:", 1)[0].strip()
            parts.append(f"{label} p95 per feature" if label else "p95 per feature")
            continue
        match = re.match(r"(.+?) 95th percentile (\S+)", piece)
        parts.append(f"{match.group(1)} p95 {match.group(2)}" if match else piece)
    return " · ".join(parts)


def _section_value(value: str) -> str:
    """Keep a per-feature bar, and drop median and round counts from a global bar.

    Args:
        value: Full baseline text.

    Returns:
        Text placed in a section note.
    """
    if "95th percentile" not in value:
        return value
    parts = []
    for piece in value.split(". "):
        detailed = "95th percentile:" in piece or bool(re.match(r"^[A-Za-z0-9_]+: ", piece))
        parts.append(piece if detailed else _concise_baseline(piece))
    return ". ".join(parts)


def _baseline_cell(signal: dict[str, Any], run: ComparisonRun) -> str:
    """HTML cell with only the comparison bar.

    Args:
        signal: One decision signal.
        run: Comparison passed to :func:`signal_baseline`.

    Returns:
        A table cell. The reason for the bar is written in the matching section.
    """
    value, _rationale = signal_baseline(signal, run)
    return f"<td>{html.escape(_concise_baseline(value))}</td>"


def _section_baseline(run: ComparisonRun, *needles: str, numbers: bool = True) -> str:
    """Baseline bar and rule for the signals that belong in one section.

    Args:
        run: Comparison whose decision lists the signals.
        *needles: Substrings of signal names.
        numbers: Include the stored values. Off when the section table already shows them.

    Returns:
        A muted paragraph, or an empty string when no signal matches.
    """
    seen: list[str] = []
    for signal in run.decision.get("signals") or []:
        name = str(signal.get("signal") or "")
        if not any(needle in name for needle in needles):
            continue
        value, rationale = signal_baseline(signal, run)
        repeated = " That percentile is sampling noise at this comparison size, under exchangeability inside the reference."
        why = rationale.replace(repeated, "").strip()
        text = f"{_section_value(value)}. {why}" if numbers else why
        if text and text not in seen:
            seen.append(text)
    if not seen:
        return ""
    body = " ".join(html.escape(part) for part in seen)
    return f"<p class='muted'><b>Baseline.</b> {body}</p>"


# --------------------------------------------------------------------------------------
# Section builders
# --------------------------------------------------------------------------------------
@dataclass
class Section:
    anchor: str
    title: str
    body: str


def _feature_counts(features: pd.DataFrame) -> dict[str, int]:
    return {level: int((features["severity"] == level).sum()) for level in ("severe", "moderate", "mild")}


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
        f"<td>{html.escape(s.get('pair') or '')}</td>"
        f"<td>{html.escape(s['value'])}</td>{_baseline_cell(s, run)}"
        f"<td class='muted'>{html.escape(s['detail'])}</td></tr>"
        for s in sorted(decision["signals"], key=lambda s: -{"severe": 2, "moderate": 1, "mild": 0, "info": -1}[s["level"]]))
    signals = (f'<div class="table-wrap"><table><thead><tr><th>Level</th><th>Signal</th><th>Pair</th><th>Observed</th>'
               f'<th>Baseline</th><th>Detail</th></tr></thead><tbody>{signal_rows}</tbody></table></div>')

    findings = []
    features = run.features
    flagged = features[features["severity"] != "mild"]
    counts = _feature_counts(features)
    findings.append(f"<b>{counts['severe']}</b> feature(s) with material drift and <b>{counts['moderate']}</b> with moderate "
                    f"drift out of {len(features)}; <b>{counts['mild']}</b> are within natural variability or negligible.")
    for row in flagged.head(6).itertuples():
        importance = " (model-important)" if meta["has_model"] and row.important else ""
        findings.append(f"{_level_badge(row.severity)} <b>{html.escape(row.feature)}</b>{importance}: "
                        f"{html.escape(str(row.reasons))}")
    domain = run.table("domain_classifier")
    if not domain.empty:
        auc = _scalar(domain, "auc")
        strength = ("indistinguishable" if auc <= 0.55 else "mildly separable" if auc <= 0.65
                    else "moderately separable" if auc <= 0.80 else "strongly separable")
        findings.append(f"Jointly, the populations are <b>{strength}</b> (classifier area under the curve (AUC) {auc:.3f}).")
    outliers = run.table("joint_outliers")
    if not outliers.empty:
        findings.append(f"<b>{_pct(_scalar(outliers, 'ood_share'))}</b> of new rows fall outside the training support "
                        f"(expected {_pct(_scalar(outliers, 'expected_share'))}); predictions for these rows are "
                        "extrapolations.")
    for task, label in (("weighted_performance", "importance weighting"), ("cbpe", "confidence-based performance estimation (CBPE)")):
        perf = run.table(task)
        if not perf.empty:
            row = perf.iloc[0]
            metric_name = METRIC_DISPLAY.get(str(row["metric"]), str(row["metric"]))
            findings.append(f"Estimated {metric_name} on the new population ({label}): <b>{row['estimate']:.3f}</b> "
                            f"[{_fmt(row['ci_low'])}–{_fmt(row['ci_high'])}] vs reference {row['reference_value']:.3f}.")
    label_shift = run.table("label_shift")
    if not label_shift.empty:
        findings.append(f"Estimated positive rate: {_pct(_scalar(label_shift, 'source_prevalence'))} → "
                        f"<b>{_pct(_scalar(label_shift, 'target_prevalence_estimate'))}</b> "
                        f"[{_pct(_scalar(label_shift, 'ci_low'))}–{_pct(_scalar(label_shift, 'ci_high'))}] "
                        "(black-box shift estimation (BBSE)).")

    caveats = ["Without labels for the new population, <b>concept shift</b> (a change in P(y|x)) cannot be detected; "
               "all performance estimates assume it is absent. Label a small random sample to verify."]
    if meta["has_model"] and meta["label_col"]:
        caveats.append("Label-based checks use the training-population rows supplied. If these rows were used to fit "
                       "the model, calibration and performance references are optimistic; prefer a labelled holdout.")
    caveats += [html.escape(w) for w in decision.get("warnings", [])]
    body = (f'<div class="verdict {level}"><div class="verdict-label">Verdict</div>'
            f'<div class="verdict-text">{html.escape(decision["verdict"])}</div>'
            f'<div class="verdict-rule">{html.escape(decision["verdict_rule"])}</div></div>'
            f'<p class="context">{context}</p>'
            '<div class="callout"><b>How to read this report.</b> Every statistic is compared with the <i>baseline</i>: '
            'the distribution of the same statistic between random subsets of the training population (pure sampling '
            'noise). That noise is computed under exchangeability inside the reference: any row can stand in for any '
            'other, so the two sides of a random split are not systematically distinguishable. The 95th percentile is '
            'the distinguishability left by the split alone. The new population exceeds it when the new population is '
            'more distinguishable from the reference than those two pieces are from each other. A difference is flagged '
            'only if it is statistically significant after multiple-testing correction, larger than the baseline 95th '
            'percentile, and at least moderate by effect-size rules of thumb. '
            f'{_level_badge("mild")} within noise or negligible · {_level_badge("moderate")} real, moderate · '
            f'{_level_badge("severe")} real, large.</div>'
            f'<h3>Signals</h3>{signals}'
            f'<h3>Key findings</h3><ul>{"".join(f"<li>{f}</li>" for f in findings)}</ul>'
            f'<div class="callout warn"><b>Caveats</b><ul>{"".join(f"<li>{c}</li>" for c in caveats)}</ul></div>')
    return Section("summary", "Executive summary", body)


def section_features(run: ComparisonRun) -> Section:
    """Overview of per-feature drift with the feature assessment table."""
    features = run.features.copy()
    p95 = {**run.baseline_threshold("continuous", "psi"), **run.baseline_threshold("categorical", "psi")}
    features["psi_p95"] = features["feature"].map(p95)
    counts = _feature_counts(features)
    has_model = run.metadata["has_model"]
    critical = features[(features["severity"] == "severe") & features["important"]]
    text = (f"<p>{counts['severe']} severe, {counts['moderate']} moderate and {counts['mild']} mild features. "
            + (f"Critical (large drift on a model-important feature): <b>{', '.join(map(html.escape, critical['feature']))}</b>. "
               if len(critical) else "No large drift on model-important features. ")
            + ("Model importance is the share of mean |Shapley additive explanations (SHAP)| on the training population; drift on low-importance "
               "features rarely affects predictions." if has_model else
               "No model was supplied, so every feature is treated as important.") + "</p>")
    columns = ["feature", "type", "severity", "important", "model_importance_share", "psi", "psi_p95", "ks_d", "smd",
               "cramers_v", "equivalent_mean", "significant", "beyond_baseline", "reasons"]
    rename = {"model_importance_share": "importance share", "equivalent_mean": "equivalent mean, TOST",
              "beyond_baseline": "beyond baseline", "ks_d": "KS D", "smd": "SMD", "cramers_v": "Cramér's V",
              "psi": "PSI", "psi_p95": "PSI p95"}
    note = _section_baseline(run, "Feature drift", numbers=False)
    body = plot_feature_overview(run) + text + note + table_html(features, columns, rename, level_col="severity")
    return Section("features", "Feature-level drift", _pair_line(run, "continuous", "categorical", "missingness") + body)


def section_continuous(run: ComparisonRun, top_k: int) -> Section | None:
    """ECDFs of the most drifted continuous features and the full test table."""
    table = run.table("continuous")
    if table.empty or "ks_d" not in table:
        return _section_with_errors(run, "continuous", "Continuous features", "", "continuous", "equivalence")
    severity = run.features.set_index("feature")["severity"]
    equivalence = run.table("equivalence")
    merged = table.merge(equivalence[["feature", "equivalent", "tost_p"]], on="feature", how="left") \
        if not equivalence.empty else table
    merged["severity"] = merged["feature"].map(severity)
    merged["baseline_p95_psi"] = merged["feature"].map(run.baseline_threshold("continuous", "psi"))
    merged["baseline_p95_ks"] = merged["feature"].map(run.baseline_threshold("continuous", "ks_d"))
    order = merged.assign(_o=merged["severity"].map({"severe": 2, "moderate": 1, "mild": 0})).sort_values(
        ["_o", "ks_d"], ascending=False)
    top = order["feature"].head(top_k).tolist()
    n_sig = int((merged["ks_p_adj"] < run.metadata["config"]["alpha"]).sum())
    n_equiv = int(merged.get("equivalent", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
    text = (f"<p>{n_sig} of {len(merged)} continuous features differ significantly (Kolmogorov–Smirnov (KS), adjusted p &lt; α); "
            f"{n_equiv} have statistically equivalent means (two one-sided tests (TOST), |standardized mean difference (SMD)| &lt; "
            f"{run.metadata['config']['equivalence_margin_smd']}). The ECDFs show the {len(top)} most drifted features: "
            "horizontal gaps are location shifts, different slopes are spread changes, divergence at the ends are "
            "tail changes. Significance alone is not decisive with large samples; use standardized mean difference (SMD), Population Stability Index (PSI) and the baseline "
            "columns for magnitude.</p>")
    columns = ["feature", "severity", "mean_source", "mean_target", "smd", "smd_ci_low", "smd_ci_high", "var_ratio",
               "ks_d", "baseline_p95_ks", "ks_p_adj", "ad_p_adj", "cvm_p_adj", "wasserstein_norm", "psi",
               "psi_ci_low", "psi_ci_high", "baseline_p95_psi", "equivalent", "tost_p"]
    rename = {"smd_ci_low": "standardized mean difference (SMD) confidence interval (CI) low", "smd_ci_high": "standardized mean difference (SMD) confidence interval (CI) high", "ks_p_adj": "Kolmogorov–Smirnov (KS) p adj",
              "ad_p_adj": "Anderson–Darling (AD) p adj", "cvm_p_adj": "Cramér–von Mises (CvM) p adj",
              "wasserstein_norm": "Wasserstein distance / standard deviation (W/SD)",
              "baseline_p95_psi": "PSI p95", "baseline_p95_ks": "KS p95"}
    note = _section_baseline(run, "Feature drift", numbers=False)
    body = plot_ecdfs(run, top) + text + note + details("Full continuous-feature table",
                                                         table_html(order, columns, rename, level_col="severity"))
    return _section_with_errors(run, "continuous", "Continuous features", _pair_line(run, "continuous") + body,
                                "continuous", "equivalence")


def section_categorical(run: ComparisonRun, top_k: int) -> Section | None:
    """Level frequencies of the most drifted categorical features and the full table."""
    table = run.table("categorical")
    if table.empty or "cramers_v" not in table:
        return _section_with_errors(run, "categorical", "Categorical features", "", "categorical")
    table = table.copy()
    table["severity"] = table["feature"].map(run.features.set_index("feature")["severity"])
    table["baseline_p95_psi"] = table["feature"].map(run.baseline_threshold("categorical", "psi"))
    order = table.assign(_o=table["severity"].map({"severe": 2, "moderate": 1, "mild": 0})).sort_values(
        ["_o", "psi"], ascending=False)
    unseen = table[table["unseen_share"] > 0]
    text = (f"<p>{int((table['chi2_p_adj'] < run.metadata['config']['alpha']).sum())} of {len(table)} categorical "
            "features differ significantly (chi-square, adjusted). Cramér's V: 0.1 small, 0.3 medium, 0.5 large. "
            + (f"Levels never seen in training appear in: {', '.join(f'{html.escape(r.feature)} ({r.unseen_share:.1%})' for r in unseen.itertuples())}; "
               "LightGBM routes unseen levels down a default branch." if len(unseen) else "No unseen levels.") + "</p>")
    columns = ["feature", "severity", "n_levels_source", "n_levels_target", "unseen_share", "chi2_stat", "chi2_dof",
               "chi2_p_adj", "cramers_v", "js_distance", "psi", "baseline_p95_psi", "top_level_source",
               "top_level_target"]
    note = _section_baseline(run, "Feature drift", numbers=False) + _section_baseline(run, "Unseen category")
    body = plot_categories(run, order["feature"].head(top_k).tolist()) + text + note + details(
        "Full categorical-feature table", table_html(order, columns, {"chi2_p_adj": "chi2 p adj", "cramers_v": "Cramér's V", "baseline_p95_psi": "PSI p95"}, level_col="severity"))
    return _section_with_errors(run, "categorical", "Categorical features", _pair_line(run, "categorical") + body,
                                "categorical")


def section_missingness(run: ComparisonRun) -> Section | None:
    """Missing-rate comparison and missingness patterns."""
    table = run.table("missingness")
    if table.empty:
        return _section_with_errors(run, "missingness", "Missing values", "", "missingness", "missingness_patterns")
    alpha = run.metadata["config"]["alpha"]
    flagged = table[(table["cohens_h_abs"] >= 0.2) & (table["z_p_adj"] < alpha)]
    patterns = run.table("missingness_patterns", "patterns")
    text = ("<p>" + (f"Material missing-rate changes (|Cohen's h| ≥ 0.2, adjusted p &lt; α): "
                     f"{', '.join(f'{html.escape(r.feature)} {r.missing_rate_source:.1%} → {r.missing_rate_target:.1%}' for r in flagged.itertuples())}. "
                     if len(flagged) else "No material change in missing rates. ")
            + "LightGBM learns a default direction for missing values, so a change in missingness changes which branch "
              "rows follow; it often signals a data-collection difference.</p>")
    note = _section_baseline(run, "Missing-value")
    body = (plot_missingness(run) + text + note
            + details("Missing-rate table", table_html(table, ["feature", "missing_rate_source", "missing_rate_target",
                                                               "difference", "cohens_h", "z_p_adj"]))
            + details("Most frequent missingness patterns", table_html(patterns)))
    return _section_with_errors(run, "missingness", "Missing values",
                                _pair_line(run, "missingness", "missingness_patterns") + body,
                                "missingness", "missingness_patterns")


def _domain_paragraph(run: ComparisonRun, baseline: BaselineRun | None, domain: pd.DataFrame) -> str:
    """Domain-classifier result, or an empty string when that task produced no table.

    Args:
        run: Comparison, used for the driver table.
        baseline: Baseline run, used for the AUC percentile.
        domain: Domain-classifier main table.

    Returns:
        One paragraph of HTML.
    """
    if domain.empty:
        return ""
    auc = _scalar(domain, "auc")
    null = baseline.null("domain_classifier", "auc") if baseline else np.array([])
    p95 = np.percentile(null, 95) if len(null) else np.nan
    perm_p = _scalar(domain, "permutation_p")
    drivers = run.table("domain_classifier", "importance").head(3)["feature"].tolist()
    return (f"<p>A LightGBM classifier trained to tell the populations apart reaches an out-of-fold (OOF) area under the curve (AUC) of "
            f"<b>{auc:.3f}</b> [{_fmt(_scalar(domain, 'auc_ci_low'))}–{_fmt(_scalar(domain, 'auc_ci_high'))}]"
            + (f", permutation p = {perm_p:.3f}" if np.isfinite(perm_p) else "")
            + (f"; the baseline 95th percentile is {p95:.3f}" if np.isfinite(p95) else "")
            + ". An area under the curve (AUC) near 0.5 means indistinguishable; 0.55–0.65 mild, 0.65–0.80 moderate, &gt; 0.80 strong shift. "
            f"The separation is driven mostly by <b>{', '.join(map(html.escape, drivers))}</b>. A low area under the curve (AUC) can still "
            "hide a moderate shift in a single feature, so read it together with the feature table.</p>")


def _kernel_results(run: ComparisonRun, baseline: BaselineRun | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """MMD and energy rows for the joint-distribution section.

    Args:
        run: Comparison.
        baseline: Baseline run, or None.

    Returns:
        Plot entries and table rows. Empty tables are omitted.
    """
    entries: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    specs = (("mmd", "mmd2", "squared maximum mean discrepancy (MMD²) (radial basis function (RBF) kernel)", "MMD²"),
             ("energy", "energy_distance", "Energy distance", "energy"))
    for task, metric, label, plot_label in specs:
        table = run.table(task)
        if table.empty:
            continue
        entries.append({"title": plot_label, "null": baseline.null(task, metric) if baseline else np.array([]),
                        "observed": _scalar(table, metric)})
        rows.append({"test": label, "statistic": _scalar(table, metric), "permutation p": _scalar(table, "permutation_p"),
                     "rows used (training/new)": f"{int(_scalar(table, 'n_source_rows')):,} / "
                                                 f"{int(_scalar(table, 'n_target_rows')):,}"})
    return entries, rows


def section_joint(run: ComparisonRun, baseline: BaselineRun | None) -> Section | None:
    """Classifier two-sample test, maximum mean discrepancy (MMD) and energy distance."""
    domain = run.table("domain_classifier")
    tasks = ("domain_classifier", "mmd", "energy")
    if domain.empty and run.table("mmd").empty and run.table("energy").empty:
        return _section_with_errors(run, "joint", "Joint distribution", "", *tasks)
    entries, rows = _kernel_results(run, baseline)
    note = _section_baseline(run, "classifier two-sample", "kernel/energy")
    kernel_text = ("<p>Kernel (MMD) and energy-distance permutation tests confirm the joint result with different "
                   "sensitivity; p &lt; α together with a value beyond the baseline indicates a real joint shift.</p>")
    body = (plot_domain_classifier(run, baseline) + _domain_paragraph(run, baseline, domain) + note
            + plot_null_grid(entries, "Kernel tests: observed vs baseline") + kernel_text
            + table_html(pd.DataFrame(rows))
            + details("Classifier test details", table_html(domain))
            + details("All separation drivers", table_html(run.table("domain_classifier", "importance"))))
    return _section_with_errors(run, "joint", "Joint distribution", _pair_line(run, *tasks) + body, *tasks)


def section_dependence(run: ComparisonRun) -> Section | None:
    """Correlation structure change."""
    main, pairs = run.table("correlation"), run.table("correlation", "pairs")
    if main.empty:
        return _section_with_errors(run, "dependence", "Dependence structure", "", "correlation")
    alpha = run.metadata["config"]["alpha"]
    material = pairs[(pairs["abs_delta"] > 0.2) & (pairs["fisher_z_p_adj"] < alpha)]
    text = (f"<p>Global change in the Spearman correlation matrix: Frobenius norm {_scalar(main, 'frobenius'):.3f} "
            f"(permutation p = {_fmt(_scalar(main, 'permutation_p'))}). "
            + (f"{len(material)} pair(s) changed by more than 0.2 (Fisher z, adjusted): "
               f"{', '.join(f'{html.escape(r.feature_a)}–{html.escape(r.feature_b)} ({r.rho_source:+.2f} → {r.rho_target:+.2f})' for r in material.head(5).itertuples())}. "
               if len(material) else "No pair changed by more than 0.2. ")
            + "Changed dependencies matter because tree models exploit feature interactions.</p>")
    note = _section_baseline(run, "Dependence")
    body = plot_correlation_delta(run) + text + note + details(
        "Top correlation changes", table_html(pairs, ["feature_a", "feature_b", "rho_source", "rho_target", "delta",
                                                      "fisher_z_p_adj"], max_rows=30))
    return _section_with_errors(run, "dependence", "Dependence structure", _pair_line(run, "correlation") + body,
                                "correlation")


def section_support(run: ComparisonRun) -> Section | None:
    """Out-of-support checks."""
    outliers, coverage = run.table("joint_outliers"), run.table("range_coverage")
    if outliers.empty and coverage.empty:
        return _section_with_errors(run, "support", "Support and out-of-distribution (OOD) rows", "",
                                    "range_coverage", "joint_outliers")
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
    note = _section_baseline(run, "Out-of-support", "Values outside")
    body = (plot_support(run) + f"<p>{' '.join(parts)}</p>" + note
            + details("Range coverage table", table_html(coverage, ["feature", "source_low", "source_high", "target_min",
                                                                    "target_max", "share_below", "share_above",
                                                                    "outside_share", "ratio_to_expected",
                                                                    "binomial_p_adj"]))
            + details("Isolation Forest details", table_html(outliers)))
    return _section_with_errors(run, "support", "Support and out-of-distribution (OOD) rows",
                                _pair_line(run, "range_coverage", "joint_outliers") + body,
                                "range_coverage", "joint_outliers")


def section_model(run: ComparisonRun) -> Section | None:
    """Prediction and attribution drift."""
    prediction, shap_summary = run.table("prediction_drift"), run.table("shap_drift", "summary")
    if prediction.empty and shap_summary.empty:
        return _section_with_errors(run, "model", "Model behaviour", "",
                                    "model_prediction", "prediction_drift", "shap_drift")
    parts = []
    if not prediction.empty:
        parts.append(f"Model score Population Stability Index (PSI) {_scalar(prediction, 'psi'):.3f} (Kolmogorov–Smirnov statistic (KS D) {_scalar(prediction, 'ks_d'):.3f}); "
                     "Population Stability Index (PSI) &lt; 0.10 stable, 0.10–0.25 monitor, &gt; 0.25 act.")
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
    note = _section_baseline(run, "Model score", "Predicted positive", "Model attributions")
    body = (plot_model_outputs(run) + f"<p>{' '.join(parts)}</p>" + note
            + details("Prediction drift table", table_html(prediction))
            + details("Shapley additive explanations (SHAP) drift table", table_html(shap, ["feature", "important", "importance_share_source",
                                                            "importance_rank_source", "importance_rank_target",
                                                            "mean_abs_shap_source", "mean_abs_shap_target",
                                                            "mean_abs_shap_rel_change", "shap_ks_d", "shap_ks_p_adj"])))
    return _section_with_errors(run, "model", "Model behaviour",
                                _pair_line(run, "prediction_drift", "shap_drift") + body,
                                "model_prediction", "prediction_drift", "shap_drift")


def section_performance(run: ComparisonRun) -> Section | None:
    """Label-free performance estimates, calibration and prior shift."""
    weighted, cbpe = run.table("weighted_performance"), run.table("cbpe")
    calibration, label_shift = run.table("calibration"), run.table("label_shift")
    if weighted.empty and cbpe.empty and label_shift.empty and calibration.empty:
        return _section_with_errors(
            run, "performance", "Estimated performance (no labels)", "",
            "model_prediction", "calibration", "weighted_performance", "cbpe", "label_shift")
    items = []
    summary = run.table("weighted_performance", "summary")
    if not weighted.empty:
        items.append(f"<b>Importance weighting</b> (assumes covariate shift only): labelled training rows are re-weighted "
                     f"to resemble the new population. Effective sample size (ESS) ratio {_scalar(summary, 'ess_ratio'):.2f} "
                     "(&gt; 0.5 reliable, 0.1–0.5 caution, &lt; 0.1 unreliable).")
    if not cbpe.empty:
        ece = _scalar(calibration, "ece")
        items.append("<b>Confidence-based performance estimation (CBPE)</b> (assumes a calibrated model and an unchanged "
                     "map from features to labels): expected metrics from predicted "
                     "probabilities, compared like-for-like with confidence-based performance estimation (CBPE) on the training population. "
                     + (f"Expected calibration error (ECE) {ece:.3f}: " + ("acceptable." if ece <= 0.05 else
                                                         "<b>too high, so confidence-based performance estimation (CBPE) estimates are unreliable</b>.")
                        if np.isfinite(ece) else ""))
    if not label_shift.empty:
        shifted = str(label_shift["prior_shift_detected"].iloc[0]) == "True"
        items.append(f"<b>Black-box shift estimation (BBSE)</b> (assumes P(x|y) unchanged): estimated positive rate "
                     f"{_pct(_scalar(label_shift, 'target_prevalence_estimate'))} vs "
                     f"{_pct(_scalar(label_shift, 'source_prevalence'))} in training"
                     + (" — prior shift detected; recalibrate the intercept or threshold." if shifted else
                        " — no significant prior shift."))
    tolerance = run.metadata["config"]["performance_tolerance"]
    text = (f"<ul>{''.join(f'<li>{i}</li>' for i in items)}</ul><p>Tolerance for the primary metric: {tolerance:g} "
            f"({'relative' if run.metadata['problem_type'] == 'regression' else 'absolute area under the curve (AUC)'}). A signal is severe only "
            "when the whole 95% confidence interval (CI) is beyond tolerance. If the methods disagree, treat the result as inconclusive.</p>")
    note = _section_baseline(run, "effective sample size", "importance weighting", "CBPE", "BBSE", "calibration")
    plots = plot_performance(run) + plot_calibration(run)
    tables = (details("Importance-weighted estimates", table_html(weighted) + table_html(summary))
              + details("Confidence-based performance estimation (CBPE) estimates", table_html(cbpe))
              + details("Calibration and prior shift", table_html(calibration) + table_html(label_shift)))
    return _section_with_errors(
        run, "performance", "Estimated performance (no labels)",
        _pair_line(run, "calibration", "weighted_performance", "cbpe", "label_shift") + plots + text + note + tables,
        "model_prediction", "calibration", "weighted_performance", "cbpe", "label_shift")


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


def section_methods(config: dict[str, Any], task_pairs: dict[str, str] | None = None) -> Section:
    """Compact methods and thresholds appendix."""
    th = config["thresholds"]
    rows = [
        ("Continuous features", "Kolmogorov–Smirnov (KS), Anderson–Darling (AD), Cramér–von Mises (CvM); standardized mean difference (SMD), variance ratio, Wasserstein distance / standard deviation (SD), Population Stability Index (PSI)",
         f"Population Stability Index (PSI) {th['psi_moderate']}/{th['psi_major']}; Kolmogorov–Smirnov statistic (KS D) {th['ks_moderate']}/{th['ks_major']}; |standardized mean difference (SMD)| {th['smd_small']}/{th['smd_large']}"),
        ("Equivalence", f"Two one-sided tests (TOST) on means, margin {config['equivalence_margin_smd']} standard deviation (SD)", "p < α ⇒ equivalent"),
        ("Categorical features", "Chi-square homogeneity (sparse levels merged); Cramér's V, Jensen–Shannon (JS), Population Stability Index (PSI)",
         f"Cramér's V {th['cramers_v_moderate']}/{th['cramers_v_major']}; unseen levels > {th['unseen_level_share']:.0%}"),
        ("Missingness", "Two-proportion z-test, Cohen's h; pattern chi-square / Jensen–Shannon (JS)", f"|h| {th['cohens_h_small']}/{th['cohens_h_large']}"),
        ("Joint distribution", f"LightGBM classifier two-sample test ({config['cv_folds']}-fold out-of-fold (OOF) area under the curve (AUC), permutation p); "
         "maximum mean discrepancy (MMD) (RBF) and energy distance with permutation p", f"area under the curve (AUC) {th['domain_auc_mild']}/{th['domain_auc_strong']}"),
        ("Dependence", "Spearman matrices: Frobenius-norm permutation test; per-pair Fisher z (var 1.06/(n−3))",
         f"|Δρ| > {th['correlation_pair_delta']}"),
        ("Support", f"Range [{config['range_quantiles'][0]}, {config['range_quantiles'][1]}] quantiles + binomial test; "
         f"Isolation Forest at {config['ood_quantile']:.0%} held-out quantile + binomial test",
         f"out-of-distribution (OOD) share {th['ood_share_caution']:.0%}/{th['ood_share_material']:.0%}"),
        ("Model behaviour", "Score Kolmogorov–Smirnov (KS)/Population Stability Index (PSI); positive-rate z-test; TreeSHAP (Shapley additive explanations) Kolmogorov–Smirnov (KS) and rank correlation",
         f"rank ρ < {th['importance_rank_rho']}; important = top {th['important_feature_cumulative_share']:.0%} of |Shapley additive explanations (SHAP)|"),
        ("Performance (no labels)", "Importance weighting (domain-classifier density ratio, Kish effective sample size (ESS)); confidence-based performance estimation (CBPE); black-box shift estimation (BBSE)",
         f"effective sample size (ESS) ratio {th['ess_ratio_poor']}/{th['ess_ratio_good']}; tolerance {config['performance_tolerance']}"),
        ("Inference", f"{config['correction_method']} correction at α = {config['alpha']}; {config['n_bootstrap']} bootstrap "
         f"resamples; {config['n_permutations']} permutations ({config['n_classifier_permutations']} for the classifier)",
         f"baseline threshold = p{th['baseline_percentile']:.0f}"),
    ]
    pairs = {
        "Continuous features": "continuous", "Equivalence": "equivalence", "Categorical features": "categorical",
        "Missingness": "missingness", "Joint distribution": "domain_classifier", "Dependence": "correlation",
        "Support": "range_coverage", "Model behaviour": "prediction_drift",
        "Performance (no labels)": "weighted_performance",
    }
    recorded = task_pairs or {}
    rows = [(*row, recorded.get(pairs.get(row[0], ""), "")) for row in rows]
    table = pd.DataFrame(rows, columns=["Area", "Methods", "Thresholds (moderate / severe)", "Pair"])
    note = ("<p class='muted'>Thresholds are conventions (Siddiqi 2006 for Population Stability Index (PSI), Cohen 1988 for effect sizes, Austin 2009 "
            "for standardized mean difference (SMD)); the baseline comparison is the statistically grounded reference.</p>")
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
             "<code>python -m population_drift compare --baseline &lt;this directory&gt;</code>. Build the baseline with "
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
.badge.mild{background:#16a34a}.badge.moderate{background:#d97706}.badge.severe{background:#dc2626}.badge.info{background:#64748b}
.verdict{border-radius:10px;padding:1rem 1.3rem;color:#fff;margin-bottom:1rem}.verdict.mild{background:#15803d}
.verdict.moderate{background:#b45309}.verdict.severe{background:#b91c1c}.verdict.info{background:#475569}
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
    page = (f'<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1"><title>{html.escape(title)}</title>'
            f'<style>{CSS}</style></head><body><header><h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p></header>'
            f'<div class="layout"><nav><strong>Contents</strong>{nav}</nav><main>{body}</main></div>'
            f'<footer>Generated {generated} by population_drift.report · standalone file, no external resources</footer>'
            '</body></html>')
    return apply_acronyms_html(page)


def _render_section(anchor: str, title: str, builder: Callable[[], Section | None]) -> Section | None:
    """Render one section. A failure becomes an error section so the report still finishes.

    Args:
        anchor: Section id used when rendering fails.
        title: Heading used when rendering fails.
        builder: Section function.

    Returns:
        The section, an error section, or None when the builder omits it.
    """
    try:
        return builder()
    except Exception as exc:
        logger.exception("Report section %s failed", anchor)
        message = html.escape(f"{type(exc).__name__}: {exc}")
        note = ("<div class='callout warn'><b>Skipped.</b> This section could not be rendered."
                f"<ul><li>{message}</li></ul></div>")
        return Section(anchor, title, note)


def _comparison_sections(run: ComparisonRun, baseline: BaselineRun | None, top_k: int) -> list[Section]:
    """Report sections for one comparison. A failed section stays as an error note.

    Args:
        run: Comparison outputs.
        baseline: Baseline outputs, or None.
        top_k: Features shown in distribution plots.

    Returns:
        Sections that have content or a failure message.
    """
    builders = (
        ("summary", "Executive summary", lambda: section_summary(run, baseline)),
        ("features", "Feature-level drift", lambda: section_features(run)),
        ("continuous", "Continuous features", lambda: section_continuous(run, top_k)),
        ("categorical", "Categorical features", lambda: section_categorical(run, top_k)),
        ("missingness", "Missing values", lambda: section_missingness(run)),
        ("joint", "Joint distribution", lambda: section_joint(run, baseline)),
        ("dependence", "Dependence structure", lambda: section_dependence(run)),
        ("support", "Support and out-of-distribution (OOD) rows", lambda: section_support(run)),
        ("model", "Model behaviour", lambda: section_model(run)),
        ("performance", "Estimated performance (no labels)", lambda: section_performance(run)),
        ("baseline", "Baseline reference", lambda: section_baseline(run, baseline)),
    )
    sections = [section for anchor, title, builder in builders
                if (section := _render_section(anchor, title, builder)) is not None]
    sections.append(section_methods(run.metadata["config"], run.metadata.get("task_pairs")))
    return sections


_DECISION_ACTIONS = {
    "severe": ("Do not score the new population with this model as-is. Investigate the severe "
               "signals, then retrain, recalibrate, or restrict scoring to in-support rows."),
    "moderate": ("Scoring is not ruled out, but review every moderate signal before production use. "
                 "Prefer a labelled sample check and continued monitoring."),
    "mild": ("No severe barrier to scoring under the toolkit rule. This is not a measurement of "
             "accuracy on the new population; keep monitoring."),
}


def short_report_path(output_path: str | Path) -> Path:
    """Sibling path for the short decision report (``report.html`` → ``report_short.html``).

    Args:
        output_path: Destination of the detailed HTML report.

    Returns:
        Path with ``_short`` inserted before the suffix.

    Raises:
        ValueError: If ``output_path`` is blank.
    """
    if not str(output_path).strip():
        raise ValueError("output_path is empty.")
    path = Path(output_path)
    suffix = path.suffix or ".html"
    return path.with_name(f"{path.stem}_short{suffix}")


def _decision_action(level: str) -> str:
    """One-sentence scoring recommendation for a verdict level.

    Args:
        level: ``severe``, ``moderate``, ``mild``, or another label.

    Returns:
        Recommendation text. Unknown levels get a review-first sentence.
    """
    return _DECISION_ACTIONS.get(
        level, "Review the signals below before deciding whether to score new data.")


def _short_context(run: ComparisonRun) -> str:
    """One-line sample and baseline sizes for the short report.

    Args:
        run: Comparison outputs.

    Returns:
        HTML paragraph of run sizes.
    """
    meta, decision = run.metadata, run.decision
    types = meta["feature_types"]
    n_cont = sum(kind == "continuous" for kind in types.values())
    return (f"<p class='context'>Training: <b>{meta['n_source']:,}</b> · New: <b>{meta['n_target']:,}</b> · "
            f"{len(types)} features ({n_cont} continuous) · "
            f"Model: <b>{'yes' if meta['has_model'] else 'no'}</b> · "
            f"Baseline: <b>{_fmt(decision.get('baseline_rounds'))}</b> × "
            f"<b>{_fmt(decision.get('baseline_subset_size'))}</b>-row subsets</p>")


def section_short_decision(run: ComparisonRun, detailed_name: str) -> Section:
    """Verdict, action, and the label-free limitation.

    Args:
        run: Comparison outputs.
        detailed_name: Filename of the detailed report sibling.

    Returns:
        Decision section for the short report.
    """
    decision = run.decision
    level = decision["verdict_level"]
    caveats = [
        "Without labels on the new population, <b>concept shift</b> (a change in P(y|x)) cannot be "
        "detected. Performance numbers below are estimates under stated assumptions.",
        f"Full tables, methods, and baseline diagnostics: <code>{html.escape(detailed_name)}</code>.",
    ]
    caveats += [html.escape(warning) for warning in decision.get("warnings", [])]
    body = (
        f'<div class="verdict {level}"><div class="verdict-label">Verdict</div>'
        f'<div class="verdict-text">{html.escape(decision["verdict"])}</div>'
        f'<div class="verdict-rule">{html.escape(decision["verdict_rule"])}</div></div>'
        f'{_short_context(run)}'
        f'<div class="callout"><b>Action.</b> {html.escape(_decision_action(level))}</div>'
        f'<h3>Limits</h3><ul>{"".join(f"<li>{item}</li>" for item in caveats)}</ul>'
    )
    return Section("decision", "Decision", body)


def section_short_signals(run: ComparisonRun) -> Section:
    """Counted signals with observed value and baseline comparison.

    Args:
        run: Comparison outputs.

    Returns:
        Signals section for the short report.
    """
    order = {"severe": 2, "moderate": 1, "mild": 0, "info": -1}
    rows = "".join(
        f"<tr><td>{_level_badge(signal['level'])}</td>"
        f"<td>{html.escape(signal['signal'])}</td>"
        f"<td>{html.escape(signal.get('pair') or '')}</td>"
        f"<td>{html.escape(signal['value'])}</td>"
        f"{_baseline_cell(signal, run)}</tr>"
        for signal in sorted(run.decision["signals"], key=lambda item: -order[item["level"]])
    )
    table = (
        '<div class="table-wrap"><table><thead><tr>'
        "<th>Level</th><th>Signal</th><th>Pair</th><th>Observed</th><th>Baseline</th>"
        f"</tr></thead><tbody>{rows}</tbody></table></div>"
    )
    note = ("<p>A signal enters the verdict only when it is significant after multiple-testing "
            "correction, beyond the baseline 95th percentile, and at least moderate by effect size. "
            f"{_level_badge('mild')} noise or negligible · {_level_badge('moderate')} real, moderate · "
            f"{_level_badge('severe')} real, large. <code>info</code> rows are reported, not counted.</p>")
    return Section("signals", "Signals that drive the verdict", note + table)


def _flagged_feature_frame(run: ComparisonRun) -> pd.DataFrame:
    """Non-mild features ordered by severity then Population Stability Index (PSI).

    Args:
        run: Comparison outputs.

    Returns:
        Feature assessment rows that are not mild. May be empty.
    """
    features = run.features
    flagged = features[features["severity"] != "mild"].copy()
    if flagged.empty:
        return flagged
    flagged["_rank"] = flagged["severity"].map({"severe": 2, "moderate": 1}).fillna(0)
    return flagged.sort_values(["_rank", "psi"], ascending=[False, False]).drop(columns="_rank")


def section_short_features(run: ComparisonRun, top_k: int) -> Section | None:
    """Flagged features, overview plot, and ECDFs or category bars for the worst cases.

    Args:
        run: Comparison outputs.
        top_k: How many drifted features to plot.

    Returns:
        Features section, or None when the assessment table is empty.
    """
    if run.features.empty:
        return None
    if top_k < 1:
        raise ValueError("top_k must be >= 1.")
    counts = _feature_counts(run.features)
    flagged = _flagged_feature_frame(run)
    text = (f"<p><b>{counts['severe']}</b> severe, <b>{counts['moderate']}</b> moderate, "
            f"<b>{counts['mild']}</b> mild of {len(run.features)} features. "
            "Model-important features are the smallest TreeSHAP set reaching 80% of mean |SHAP|.</p>")
    columns = ["feature", "type", "severity", "important", "psi", "ks_d", "smd", "cramers_v", "reasons"]
    rename = {"ks_d": "KS D", "smd": "SMD", "cramers_v": "Cramér's V", "psi": "PSI"}
    table = (table_html(flagged, columns, rename, level_col="severity")
             if not flagged.empty else "<p>No feature exceeded mild.</p>")
    names = flagged["feature"].head(top_k).tolist() if not flagged.empty else []
    plots = plot_feature_overview(run)
    if names:
        types = run.features.set_index("feature")["type"]
        continuous = [name for name in names if types.get(name) == "continuous"]
        categorical = [name for name in names if name not in continuous]
        plots += plot_ecdfs(run, continuous) + plot_categories(run, categorical)
    return Section("features", "Feature evidence",
                   _pair_line(run, "continuous", "categorical") + plots + text + table)


def section_short_model_evidence(run: ComparisonRun, baseline: BaselineRun | None) -> Section | None:
    """Joint separation, score drift, and label-free performance plots with key numbers.

    Args:
        run: Comparison outputs.
        baseline: Baseline outputs, or None.

    Returns:
        Model-evidence section when any of those tasks produced output, else None.
    """
    parts: list[str] = []
    domain = run.table("domain_classifier")
    if not domain.empty or not run.table("mmd").empty or not run.table("energy").empty:
        parts.append(plot_domain_classifier(run, baseline))
        parts.append(_domain_paragraph(run, baseline, domain))
        entries, rows = _kernel_results(run, baseline)
        parts.append(plot_null_grid(entries, "Kernel tests: observed vs baseline"))
        if rows:
            parts.append(table_html(pd.DataFrame(rows)))
    prediction = run.table("prediction_drift")
    shap_summary = run.table("shap_drift", "summary")
    if not prediction.empty or not shap_summary.empty:
        parts.append(plot_model_outputs(run))
        if not prediction.empty:
            parts.append(
                f"<p>Score PSI {_scalar(prediction, 'psi'):.3f}; KS D {_scalar(prediction, 'ks_d'):.3f}"
                + (f"; positive rate {_pct(_scalar(prediction, 'positive_rate_source'))} → "
                   f"<b>{_pct(_scalar(prediction, 'positive_rate_target'))}</b>"
                   if "positive_rate_source" in prediction else "")
                + ".</p>")
        if not shap_summary.empty:
            parts.append(
                f"<p>SHAP importance rank Spearman ρ = {_scalar(shap_summary, 'importance_rank_rho'):.2f} "
                "(below 0.80 means the ranking of reasons changed).</p>")
    weighted, cbpe = run.table("weighted_performance"), run.table("cbpe")
    if not weighted.empty or not cbpe.empty or not run.table("label_shift").empty:
        parts.append(plot_performance(run) + plot_calibration(run))
        for task, label in (("weighted_performance", "importance weighting"),
                            ("cbpe", "confidence-based performance estimation (CBPE)")):
            table = run.table(task)
            if table.empty:
                continue
            row = table.iloc[0]
            metric = METRIC_DISPLAY.get(str(row["metric"]), str(row["metric"]))
            parts.append(
                f"<p>{label}: estimated {metric} <b>{row['estimate']:.3f}</b> "
                f"[{_fmt(row['ci_low'])}–{_fmt(row['ci_high'])}] vs reference {row['reference_value']:.3f}.</p>")
    if not parts:
        return None
    return Section(
        "model", "Model and performance evidence",
        _pair_line(run, "domain_classifier", "prediction_drift", "weighted_performance", "cbpe") + "".join(parts),
    )


def _short_comparison_sections(
        run: ComparisonRun, baseline: BaselineRun | None, top_k: int, detailed_name: str) -> list[Section]:
    """Essential sections for a data-scientist decision report.

    Args:
        run: Comparison outputs.
        baseline: Baseline outputs, or None.
        top_k: Features shown in distribution plots.
        detailed_name: Filename of the detailed report sibling.

    Returns:
        Short-report sections that have content.
    """
    builders = (
        ("decision", "Decision", lambda: section_short_decision(run, detailed_name)),
        ("signals", "Signals that drive the verdict", lambda: section_short_signals(run)),
        ("features", "Feature evidence", lambda: section_short_features(run, top_k)),
        ("model", "Model and performance evidence", lambda: section_short_model_evidence(run, baseline)),
    )
    return [section for anchor, title, builder in builders
            if (section := _render_section(anchor, title, builder)) is not None]


_IMG_TAG_RE = re.compile(r"<img\b[^>]*>", re.I | re.S)
_TABLE_RE = re.compile(r"<table\b.*?</table>", re.I | re.S)
_ATTR_RE = re.compile(r'([a-zA-Z_:][-a-zA-Z0-9:._]*)="([^"]*)"')
_VERDICT_RE = re.compile(
    r'<div class="verdict (\w+)">\s*<div class="verdict-label">.*?</div>\s*'
    r'<div class="verdict-text">(.*?)</div>\s*<div class="verdict-rule">(.*?)</div>\s*</div>',
    re.I | re.S,
)
_CALLOUT_RE = re.compile(r'<div class="callout[^"]*">(.*?)</div>', re.I | re.S)
_DETAILS_RE = re.compile(r"<details>\s*<summary>(.*?)</summary>(.*?)</details>", re.I | re.S)


def html_fragment_to_markdown(fragment: str, plot_dir: Path, plot_rel: str) -> str:
    """Turn a report HTML fragment into markdown and write embedded PNGs.

    Args:
        fragment: HTML from one report section.
        plot_dir: Directory for extracted PNG files.
        plot_rel: Relative directory name used in markdown image links.

    Returns:
        Markdown for the fragment.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If ``plot_rel`` is blank.
    """
    if not isinstance(fragment, str):
        raise TypeError("fragment must be a string.")
    if not isinstance(plot_dir, Path):
        raise TypeError("plot_dir must be a Path.")
    if not str(plot_rel).strip():
        raise ValueError("plot_rel is empty.")
    text = _replace_images(fragment, plot_dir, plot_rel.strip())
    return _collapse_blank(_inline_markdown(_replace_blocks(text)))


def write_report_files(
        output_path: str | Path, title: str, subtitle: str, sections: Sequence[Section]) -> Path:
    """Write the HTML report and a markdown copy with PNG plots.

    Args:
        output_path: Destination path. ``.html`` and ``.md`` siblings are both written.
        title: Report title.
        subtitle: One-line context under the title.
        sections: Rendered report sections.

    Returns:
        The resolved destination path.

    Raises:
        ValueError: If ``output_path`` or ``title`` is blank.
        TypeError: If ``subtitle`` is not a string.
    """
    if not str(output_path).strip():
        raise ValueError("output_path is empty.")
    if not isinstance(title, str) or not title.strip():
        raise ValueError("title is empty.")
    if not isinstance(subtitle, str):
        raise TypeError("subtitle must be a string.")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    html_path, md_path = path.with_suffix(".html"), path.with_suffix(".md")
    html_path.write_text(render_page(title, subtitle, sections), encoding="utf-8")
    md_path.write_text(_render_markdown(title, subtitle, sections, md_path), encoding="utf-8")
    logger.info("Report written to %s and %s", html_path, md_path)
    return path.resolve()


def _render_markdown(title: str, subtitle: str, sections: Sequence[Section], md_path: Path) -> str:
    """Assemble markdown from section HTML, extracting plots next to ``md_path``."""
    plot_dir = md_path.parent / f"{md_path.stem}_plots"
    parts = [f"# {title}\n\n{subtitle}\n"]
    for section in sections:
        parts.append(f"\n## {section.title}\n\n")
        parts.append(html_fragment_to_markdown(section.body, plot_dir, plot_dir.name))
        parts.append("\n")
    return apply_acronyms_markdown("".join(parts))


def _replace_images(html_text: str, plot_dir: Path, plot_rel: str) -> str:
    """Write base64 PNGs to ``plot_dir`` and replace ``<img>`` tags with markdown links."""
    plot_dir.mkdir(parents=True, exist_ok=True)
    index = _next_plot_index(plot_dir)

    def repl(match: re.Match[str]) -> str:
        nonlocal index
        attrs = dict(_ATTR_RE.findall(match.group(0)))
        src = attrs.get("src", "")
        if not src.startswith("data:image/png;base64,"):
            return ""
        name = f"plot_{index:02d}.png"
        (plot_dir / name).write_bytes(base64.b64decode(src.split(",", 1)[1]))
        index += 1
        return f"\n\n![{attrs.get('alt', 'plot')}]({plot_rel}/{name})\n\n"

    return _IMG_TAG_RE.sub(repl, html_text)


def _next_plot_index(plot_dir: Path) -> int:
    """Return the next 1-based plot index that does not overwrite an existing PNG."""
    found = []
    for path in plot_dir.glob("plot_*.png"):
        suffix = path.stem.split("_")[-1]
        if suffix.isdigit():
            found.append(int(suffix))
    return max(found, default=0) + 1


def _replace_blocks(text: str) -> str:
    """Convert tables, verdicts, headings, lists, and callouts to markdown."""
    text = _TABLE_RE.sub(lambda match: f"\n{_table_to_markdown(match.group(0))}\n", text)
    text = _VERDICT_RE.sub(_verdict_md, text)
    text = re.sub(r"<h3[^>]*>(.*?)</h3>", lambda m: f"\n### {_inline_markdown(m.group(1))}\n", text, flags=re.I | re.S)
    text = re.sub(r"<li[^>]*>(.*?)</li>", lambda m: f"- {_inline_markdown(m.group(1))}\n", text, flags=re.I | re.S)
    text = _DETAILS_RE.sub(_details_md, text)
    text = _CALLOUT_RE.sub(_callout_md, text)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    return re.sub(r"</p>", "\n\n", text, flags=re.I)


def _table_to_markdown(table_html: str) -> str:
    """Convert one HTML table to a pipe table."""
    rows = []
    for row_html in re.findall(r"<tr\b[^>]*>(.*?)</tr>", table_html, flags=re.I | re.S):
        cells = re.findall(r"<t[hd]\b[^>]*>(.*?)</t[hd]>", row_html, flags=re.I | re.S)
        rows.append([_cell_text(cell) for cell in cells])
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    padded = [row + [""] * (width - len(row)) for row in rows]
    header, body = padded[0], padded[1:]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines) + "\n"


def _cell_text(fragment: str) -> str:
    """Plain table-cell text with pipes escaped."""
    return " ".join(_inline_markdown(fragment).split()).replace("|", "\\|")


def _verdict_md(match: re.Match[str]) -> str:
    """Turn the HTML verdict banner into a blockquote."""
    verdict = _inline_markdown(match.group(2)).strip()
    rule = _inline_markdown(match.group(3)).strip()
    return f"\n> **Verdict ({match.group(1)}):** {verdict}\n>\n> {rule}\n"


def _details_md(match: re.Match[str]) -> str:
    """Unfold a details block into bold summary plus body."""
    summary = _inline_markdown(match.group(1)).strip()
    body = _inline_markdown(match.group(2)).strip()
    return f"\n**{summary}**\n\n{body}\n"


def _callout_md(match: re.Match[str]) -> str:
    """Turn a callout div into a markdown blockquote."""
    inner = _inline_markdown(match.group(1)).strip()
    quoted = "\n".join(f"> {line}" if line else ">" for line in inner.splitlines())
    return f"\n{quoted}\n"


def _inline_markdown(fragment: str) -> str:
    """Convert inline HTML emphasis to markdown and drop remaining tags."""
    fragment = re.sub(r'<span class="badge[^"]*">([^<]*)</span>', r"**\1**", fragment, flags=re.I)
    fragment = re.sub(r"</?(?:b|strong)>", "**", fragment, flags=re.I)
    fragment = re.sub(r"</?(?:i|em)>", "*", fragment, flags=re.I)
    fragment = re.sub(r"<code>(.*?)</code>", r"`\1`", fragment, flags=re.I | re.S)
    return html.unescape(_TAG_RE.sub("", fragment))


def _collapse_blank(text: str) -> str:
    """Trim trailing spaces and collapse runs of blank lines."""
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def build_report(output_path: str | Path, comparison_dir: str | Path | None = None,
                 baseline_dir: str | Path | None = None, title: str | None = None, top_k: int = 6) -> Path:
    """Build the detailed HTML report and, for a comparison, a short decision report.

    Both the detailed and short reports also write a markdown copy with PNG plots.

    Args:
        output_path: Destination HTML file for the detailed report.
        comparison_dir: Output directory of ``python -m population_drift compare`` (full report).
        baseline_dir: Baseline directory; defaults to the one recorded in the comparison's decision.json.
        title: Optional report title.
        top_k: Number of features shown in distribution plots.

    Returns:
        Path of the written detailed report.

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
        sections = _comparison_sections(run, baseline, top_k)
        subtitle = (f"Training population ({run.metadata['n_source']:,} rows) vs new population "
                    f"({run.metadata['n_target']:,} rows) · analysis run {run.metadata['created_at']}")
        title = title or "Population drift report: is the model transferable?"
        detailed = write_report_files(output_path, title, subtitle, sections)
        short_top_k = min(top_k, 3)
        short_sections = _short_comparison_sections(run, baseline, short_top_k, Path(output_path).name)
        write_report_files(
            short_report_path(output_path),
            "Population drift: decision brief",
            subtitle,
            short_sections,
        )
        return detailed

    sections = baseline_only_sections(baseline)
    subtitle = f"Training population baseline · {baseline.metadata['created_at']}"
    title = title or "Population drift baseline report"
    return write_report_files(output_path, title, subtitle, sections)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m population_drift.report",
        description="Build the HTML drift report.",
    )
    parser.add_argument("--comparison", help="Comparison output directory.")
    parser.add_argument("--baseline", help="Baseline directory (optional with --comparison).")
    parser.add_argument(
        "--out", required=True,
        help=("Output HTML path for the detailed report. A markdown copy with PNG plots is written "
              "beside it. A comparison also writes ``*_short.html`` / ``*_short.md``."),
    )
    parser.add_argument("--title")
    parser.add_argument("--top-k", type=int, default=6, help="Features shown in distribution plots.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    build_report(args.out, args.comparison, args.baseline, args.title, args.top_k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
