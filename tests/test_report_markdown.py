"""Unit tests for the markdown report that sits next to the HTML report."""

from __future__ import annotations

from pathlib import Path

import pytest

from population_drift.report import (
    ACRONYMS,
    Section,
    apply_acronyms_markdown,
    html_fragment_to_markdown,
    write_report_files,
    _markdown_two_column_table,
    _two_column_table_html,
)

# 1x1 PNG. Source: W3C PNG test suite, 1x1 red pixel.
_PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQ"
    "AAAABJRU5ErkJggg=="
)


def _check(actual: object, expected: object, what: str) -> None:
    """Fail with the comparison that did not hold."""
    if actual != expected:
        pytest.fail(f"{what}: got {actual!r}, expected {expected!r}.")


def test_html_fragment_to_markdown_writes_pngs_and_a_pipe_table(tmp_path: Path) -> None:
    """Plots become files plus image links. Tables become pipe tables."""
    fragment = (
        f"<h3>Signals</h3>"
        f'<img class="plot" alt="Score PSI" src="data:image/png;base64,{_PNG}">'
        "<table><thead><tr><th>Level</th><th>Signal</th></tr></thead>"
        '<tbody><tr><td><span class="badge severe">severe</span></td>'
        "<td>Feature drift</td></tr></tbody></table>"
    )
    plots = tmp_path / "report_plots"
    text = html_fragment_to_markdown(fragment, plots, "report_plots")
    _check("### Signals" in text, True, "h3 becomes a markdown heading")
    _check("![Score PSI](report_plots/plot_01.png)" in text, True, "plot becomes a relative image link")
    _check("| Level | Signal |" in text, True, "table header")
    _check("**severe**" in text, True, "severity badge")
    png = plots / "plot_01.png"
    _check(png.is_file(), True, "PNG written")
    _check(png.stat().st_size > 0, True, "PNG is non-empty")


def test_write_report_files_emits_html_and_markdown(tmp_path: Path) -> None:
    """An HTML destination also writes a markdown copy with a plots directory."""
    sections = [Section(
        "summary",
        "Executive summary",
        f'<div class="verdict severe"><div class="verdict-label">Verdict</div>'
        f'<div class="verdict-text">Different: do not use the model as is</div>'
        f'<div class="verdict-rule">Any counted signal is severe.</div></div>'
        f'<img class="plot" alt="ECDF" src="data:image/png;base64,{_PNG}">',
    )]
    html_path = write_report_files(tmp_path / "report.html", "Title", "Subtitle", sections)
    md_path = html_path.with_suffix(".md")
    _check(html_path.is_file(), True, "HTML exists")
    _check(md_path.is_file(), True, "markdown exists")
    markdown = md_path.read_text(encoding="utf-8")
    _check("# Title" in markdown, True, "markdown title")
    _check("**Verdict (severe):**" in markdown, True, "verdict block")
    _check("![ECDF](report_plots/plot_01.png)" in markdown, True, "plot link in markdown")
    _check("<img" in html_path.read_text(encoding="utf-8"), True, "HTML still embeds the plot")
    _check((tmp_path / "report_plots" / "plot_01.png").is_file(), True, "plot file next to markdown")


def test_write_report_files_rejects_an_empty_path() -> None:
    """A blank destination is invalid."""
    with pytest.raises(ValueError, match="output_path"):
        write_report_files("", "Title", "Subtitle", [])


def test_markdown_glossary_includes_mmd_and_energy_readings() -> None:
    """The appendix explains MMD² and energy distance when those magnitudes appear."""
    text = apply_acronyms_markdown("The squared maximum mean discrepancy (MMD²) is 0.01.\n")
    _check("Energy distance" in text, True, "named magnitude in appendix")
    _check("Euclidean distance" in text, True, "energy reading")
    _check("Local:" in ACRONYMS["MMD"].reading, True, "MMD neighborhood reading")
    _check("Unbiased square" in ACRONYMS["MMD²"].reading, True, "MMD² stored number")


def test_two_column_glossary_tables_reject_empty_input() -> None:
    """Name/reading tables need two headings and at least one row."""
    with pytest.raises(ValueError, match="rows"):
        _two_column_table_html(("Statistic", "Reading"), ())
    with pytest.raises(ValueError, match="headers"):
        _markdown_two_column_table(("Statistic",), (("Energy distance", "reading"),))
