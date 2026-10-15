"""Tests for the concise decision proposal document."""

from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path


def test_proposal_v2_contains_the_required_decision_summary() -> None:
    """The concise proposal exists, parses as HTML, and states the required claims."""
    path = Path("docs/proposal_v2.html")
    assert path.exists(), "Expected docs/proposal_v2.html to exist."

    text = path.read_text(encoding="utf-8")
    HTMLParser().feed(text)

    required = [
        "severe",
        "moderate",
        "mild",
        "30",
        "42",
        "0.02",
        "concept shift",
        "decision.json",
    ]
    assert all(value in text for value in required)
    assert len(text.splitlines()) <= 220
