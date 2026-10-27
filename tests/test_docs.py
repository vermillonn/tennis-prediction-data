"""Portfolio documentation: relative links resolve, headline figures match the published reports, screenshots are
present and small, and the stated operational status matches the configuration."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

from tennis_data.config import PROJECT_ROOT
from tennis_data.dashboard import data as D

DOCS = [PROJECT_ROOT / "README.md", *sorted((PROJECT_ROOT / "docs").glob("*.md")),
        PROJECT_ROOT / "dashboard" / "data" / "README.md"]  # fmt: skip
LINK = re.compile(r"\]\(([^)\s]+)\)")
SCREENSHOTS = PROJECT_ROOT / "docs" / "assets" / "screenshots"


def _text(name: str) -> str:
    return (PROJECT_ROOT / name).read_text()


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: str(p.relative_to(PROJECT_ROOT)))
def test_relative_links_resolve(doc: Path):
    broken = []
    for target in LINK.findall(doc.read_text()):
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        path = target.split("#", 1)[0]
        if path and not (doc.parent / path).exists():
            broken.append(target)
    assert not broken, f"{doc.name}: broken relative links {broken}"


def test_headline_figures_match_published_reports():
    rows = {r["phase"]: r for r in D.headline(D.published())}
    assert set(rows) == {"development", "exploratory"}
    for name in ("README.md", "docs/research_summary.md"):
        text = _text(name)
        for r in rows.values():
            for value in (r["m4a"]["log_loss"], r["elo"]["log_loss"], r["d_log_loss"], *r["ll_ci95"]):
                shown = f"{value:.5f}".replace("-", "−")
                assert shown in text, f"{name} lacks the published figure {shown} ({r['phase']})"
            assert f"{r['matches']:,}" in text


def test_screenshots_exist_and_are_small():
    shots = sorted(SCREENSHOTS.glob("*.png"))
    assert len(shots) >= 5, "one screenshot per dashboard section"
    for p in shots:
        assert p.stat().st_size < 300_000, f"{p.name} is too large for the repository"
    readme = _text("README.md")
    for section in ("1-overview", "2-match-explorer", "3-player-comparison", "4-model-evaluation", "5-data-status"):
        assert (SCREENSHOTS / f"{section}.png").exists()
        assert f"docs/assets/screenshots/{section}.png" in readme


def test_stated_status_matches_configuration():
    status = tomllib.loads(_text("config/prospective.toml"))["status"]
    assert (status["backfill"], status["prospective_predictions"]) == ("INCOMPLETE", "DISABLED")
    readme = _text("README.md")
    assert "Prospective predictions are **disabled**" in readme and "backfill is **incomplete**" in readme
    assert "**not deployed**" in readme
    assert "streamlit.app" not in readme, "no deployed application exists; do not link one"
    for name in ("README.md", "docs/research_summary.md", "docs/portfolio_summary.md"):
        text = _text(name)
        assert "untouched holdout" in text or "not an independent test" in text, name
