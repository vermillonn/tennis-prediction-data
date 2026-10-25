"""The published Phase 3B reports agree with the configuration and the pre-specified selection policy.

These tests read only committed files (no data), so they run in CI. They guard against a silent change to
the frozen M4A specification or to the selection policy after publication.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tennis_data.modeling.config import load_modeling_config
from tennis_data.modeling.experiments import SIMPLICITY_ORDER, choose_with_tolerance
from tennis_data.modeling.pipeline import code_hash

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports" / "modeling"


@pytest.fixture(scope="module")
def summary():
    return json.loads((REPORTS / "modeling_summary.json").read_text())


def test_published_configuration_and_code_match_the_repository(summary):
    cfg = load_modeling_config()
    assert summary["configuration_hash"] == cfg.configuration_hash()
    assert summary["code_hash"] == code_hash(cfg), "modeling code changed after the reports were generated"
    assert summary["model_version"] == cfg.full_version


def test_selection_follows_the_prespecified_rule(summary):
    cfg = load_modeling_config()
    sel = summary["selection"]
    scores = sel["candidate_dev_log_loss"]
    assert sorted(scores) == sorted(cfg.candidates)
    by_simplicity = sorted(cfg.candidates, key=lambda e: (len(cfg.experiment(e).inputs), cfg.candidates.index(e)))
    assert sel["lowest"] == min(scores, key=scores.get) == "M4B"
    assert choose_with_tolerance(scores, by_simplicity, cfg.simplicity_tolerance)[0] == sel["selected"] == "M4A"
    # M4A is within the tolerance of M4B and has fewer inputs
    assert 0 < scores["M4A"] - scores["M4B"] <= cfg.simplicity_tolerance
    assert len(cfg.experiment("M4A").inputs) < len(cfg.experiment("M4B").inputs)
    cal = sel["calibration_scores"]
    assert choose_with_tolerance(cal, SIMPLICITY_ORDER, cfg.simplicity_tolerance)[0] == sel["calibration"] == "raw"
    assert sel["C"] == 10.0 and sel["C"] in cfg.c_grid


def test_frozen_model_is_referenced_consistently(summary):
    cfg = load_modeling_config()
    assert summary["selected_model"] == "M4A+raw"
    assert set(summary["production"]["coef_original_units"]) == set(cfg.experiment("M4A").inputs)
    short = summary["frozen_hash"][:16]
    for name in ("development_report.md", "exploratory_report.md", "model_card.md"):
        assert short in (REPORTS / name).read_text(), name
    text = (REPORTS / "development_report.md").read_text()
    assert "Selected after the simplicity rule: **M4A** (C = 10.0)" in text and "**raw**" in text


def test_evidence_tiers_are_stated():
    dev = (REPORTS / "development_report.md").read_text()
    exp = (REPORTS / "exploratory_report.md").read_text()
    assert "This is not an independent test" in dev
    assert "*not* an untouched holdout" in exp and "Prospective evaluation protocol" in exp
    docs = (ROOT / "docs" / "modeling.md").read_text()
    for heading in ("Evidence tiers", "M4A versus M4B", "Open research questions"):
        assert heading in docs, heading
