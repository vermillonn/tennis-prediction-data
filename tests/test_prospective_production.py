"""End-to-end fixture test of the frozen 2026 production M4A inference path (no retraining, no row-level data).

``tests/fixtures/m4a_production`` holds byte-exact copies of the Phase 3B ``frozen_model.json`` and
``production_model.json`` with their published checksums, and hand-made feature states whose expected probabilities
were computed outside the package from the published coefficients.
"""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from tennis_data.prospective.config import load_prospective_config
from tennis_data.prospective.freshness import assess
from tennis_data.prospective.inference import (
    ArtifactIntegrityError,
    apply_model,
    derive,
    load_frozen,
    load_production,
    predict_state,
)
from tennis_data.prospective.pipeline import PredictionBlocked, check_enabled, check_freshness, check_timing
from tennis_data.prospective.state import UpcomingMatch

FIX = Path(__file__).parent / "fixtures" / "m4a_production"
GOLDEN = json.loads((FIX / "golden.json").read_text())
M4A_INPUTS = ["elo_logit", "spw_shrunk_diff", "rpw_shrunk_diff", "spw_decay_diff", "rpw_decay_diff",
              "decay_missing_diff"]  # fmt: skip


def _cfg(modeling_dir: Path):
    cfg = load_prospective_config()  # the real repository configuration (frozen hash bc4e5c870c6e21b9)
    return replace(cfg, modeling=replace(cfg.modeling, modeling_dir=modeling_dir))


@pytest.fixture
def cfg(tmp_path):
    d = tmp_path / "modeling"
    shutil.copytree(FIX, d)
    return _cfg(d)


def _state(rows=None, swap=False) -> pl.DataFrame:
    out = []
    for r in rows or GOLDEN["rows"]:
        a = {"spw": r["asd"], "rpw": r["ard"]}
        b = {"spw": r["bsd"], "rpw": r["brd"]}
        if swap:
            a, b = b, a
        sgn = -1.0 if swap else 1.0
        out.append({
            "match_id": f"pending:{r['match_id']}", "player_a_id": 2 if swap else 1, "player_b_id": 1 if swap else 2,
            "surface": "Hard", "prediction_batch": date(2026, 10, 31),
            "predicted_probability_a": 1 - r["p"] if swap else r["p"],
            "spw_diff": None if r["spw"] is None else sgn * r["spw"], "rpw_diff": None if r["rpw"] is None else sgn * r["rpw"],
            "a_spw_decay52w": a["spw"], "b_spw_decay52w": b["spw"], "a_rpw_decay52w": a["rpw"], "b_rpw_decay52w": b["rpw"],
            "a_has_stats_history": True, "b_has_stats_history": True, "a_weeks_since_last_stats": 1,
            "b_weeks_since_last_stats": 1, "a_service_points_observed": 5000, "b_service_points_observed": 5000,
            "a_service_ess_decay52w": 900.0, "b_service_ess_decay52w": 900.0, "adjustment_fit_batch": date(2026, 10, 24),
            "global_match_count_a": 300, "global_match_count_b": 300,
        })  # fmt: skip
    return pl.DataFrame(out, infer_schema_length=None)


def test_artifacts_are_the_published_frozen_model(cfg):
    frozen = load_frozen(cfg)
    assert frozen["frozen_hash"].startswith("bc4e5c870c6e21b9")
    assert frozen["configuration_hash"] == "5e72010100065889" == cfg.modeling.configuration_hash()
    art = load_production(cfg)
    assert art.model_id == "M4A+raw" and art.calibrator["method"] == "raw"
    assert list(art.inputs) == M4A_INPUTS  # feature order of the frozen specification
    assert art.training_cutoff == "2026-06-06" and art.source == "production"
    man = json.loads((FIX / "manifest.json").read_text())
    assert art.artifact_sha256 == man["artifacts"]["models/production_model.json"]["sha256"]


def test_golden_probabilities_through_the_full_inference_path(cfg):
    art = load_production(cfg)
    out = predict_state(_state(), art, cfg, prediction_mode="exploratory_retrospective", snapshot_id="fixture",
                        components={"fixture": "m4a_production"}, generated_at=datetime(2026, 10, 26, tzinfo=UTC))  # fmt: skip
    for r in out.iter_rows(named=True):
        exp = GOLDEN["expected"][r["match_id"]]["probability_a"]
        assert r["probability_a"] == pytest.approx(exp, abs=1e-12), r["match_id"]
        assert abs(r["probability_a"] + r["probability_b"] - 1) <= 1e-12
        assert r["frozen_hash"].startswith("bc4e5c870c6e21b9") and r["prediction_mode"] == "exploratory_retrospective"
    assert out.filter(pl.col("match_id") == "coin_flip")["probability_a"].item() == 0.5


def test_preprocessing_and_frozen_fallback(cfg):
    art = load_production(cfg)
    x, m = derive(_state(), art, cfg)
    for i, r in enumerate(GOLDEN["rows"]):
        exp = GOLDEN["expected"][r["match_id"]]["inputs"]
        assert x[i].tolist() == pytest.approx([exp[k] for k in M4A_INPUTS], abs=1e-15)
    filled = {r["match_id"].removeprefix("pending:"): [n for n in M4A_INPUTS if r[f"filled_{n}"]]
              for r in m.iter_rows(named=True)}  # fmt: skip
    assert filled["a_no_recent_form"] == ["spw_decay_diff", "rpw_decay_diff"]
    assert filled["both_no_recent_form"] == ["spw_shrunk_diff", "rpw_shrunk_diff", "spw_decay_diff", "rpw_decay_diff"]
    assert filled["established"] == []
    # Scaling uses the frozen training-fold RMS values; nothing is refitted on the inference rows.
    prod = json.loads((FIX / "models" / "production_model.json").read_text())
    xs = x / np.array([prod["fit"]["scale"][k] for k in M4A_INPUTS])
    z = xs @ np.array([prod["fit"]["coef_scaled"][k] for k in M4A_INPUTS])
    pa, _ = apply_model(x, art)
    np.testing.assert_allclose(pa, 1 / (1 + np.exp(-z)), atol=1e-15)


def test_column_order_of_the_state_does_not_matter(cfg):
    art = load_production(cfg)
    st = _state()
    shuffled = st.select(list(reversed(st.columns)))
    a = predict_state(st, art, cfg, prediction_mode="historical_reconstruction", snapshot_id="s", components={},
                      generated_at=datetime(2026, 10, 26, tzinfo=UTC))  # fmt: skip
    b = predict_state(shuffled, art, cfg, prediction_mode="historical_reconstruction", snapshot_id="s", components={},
                      generated_at=datetime(2026, 10, 26, tzinfo=UTC))  # fmt: skip
    assert a.equals(b)


def test_player_order_symmetry(cfg):
    art = load_production(cfg)
    kw = dict(prediction_mode="exploratory_retrospective", snapshot_id="s", components={},
              generated_at=datetime(2026, 10, 26, tzinfo=UTC))  # fmt: skip
    ab = predict_state(_state(), art, cfg, **kw)
    ba = predict_state(_state(swap=True), art, cfg, **kw)
    np.testing.assert_allclose(ab["probability_a"].to_numpy(), ba["probability_b"].to_numpy(), atol=1e-12)
    np.testing.assert_allclose(ab["baseline_elo_probability_a"].to_numpy(), 1 - ba["baseline_elo_probability_a"].to_numpy(),
                               atol=1e-12)  # fmt: skip


@pytest.mark.parametrize("target", ["models/production_model.json", "models/frozen_model.json", "manifest.json"])
def test_any_artifact_change_stops_inference(cfg, target):
    path = cfg.modeling.modeling_dir / target
    d = json.loads(path.read_text())
    if target == "manifest.json":
        d["frozen_hash"] = "0" * 64
    elif target.endswith("production_model.json"):
        d["specification"]["inputs"] = list(reversed(d["specification"]["inputs"]))  # feature order tampered
    else:
        d["specification"]["C"] = 1.0
    path.write_text(json.dumps(d))
    with pytest.raises(ArtifactIntegrityError):
        load_production(cfg)


def test_wrong_expected_hash_is_refused(cfg):
    with pytest.raises(ArtifactIntegrityError, match="expected"):
        load_production(replace(cfg, expected_frozen_hash="a" * 16))


# ----------------------------------------------------------------------------------- stale-state rejection


def _probe_row(batch=date(2026, 10, 31)):
    return _state()[:1].with_columns(pl.lit(batch).alias("prediction_batch"),
                                     pl.lit(date(2026, 5, 23)).alias("adjustment_fit_batch"),
                                     pl.lit(23).alias("a_weeks_since_last_stats"))  # fmt: skip


def _archive_cover():
    """The real archive's coverage at its end (see reports/prospective/backfill_coverage.md)."""
    weeks = {date(2026, 5, 23) - timedelta(weeks=k) for k in range(20)}
    return {"last_update_release_batch": date(2026, 5, 30), "last_stats_release_batch": date(2026, 5, 16),
            "result_weeks": weeks}  # fmt: skip


def test_stale_state_is_rejected_for_a_2026_prospective_match():
    cfg = load_prospective_config()
    fresh = assess(_probe_row(), _archive_cover(), cfg, now=datetime(2026, 10, 30, tzinfo=UTC))["established"]
    assert fresh["status"] == "fail"
    assert fresh["checks"]["missing_backfill_weeks"]["value"] == 22
    assert fresh["checks"]["last_eligible_result"]["status"] == "fail"
    assert fresh["checks"]["opponent_fit_age_weeks"]["status"] == "fail"  # the probe row's fit is 23 weeks old
    with pytest.raises(PredictionBlocked, match="freshness"):
        check_freshness({"m": fresh})


def test_prospective_predictions_are_disabled_by_the_status_gate():
    cfg = load_prospective_config()
    assert (cfg.backfill_status, cfg.prospective_predictions) == ("INCOMPLETE", "DISABLED")
    with pytest.raises(PredictionBlocked, match="DISABLED"):
        check_enabled(cfg)


def test_status_gate_cannot_be_enabled_while_backfill_is_incomplete(tmp_path):
    from tennis_data.config import ConfigError

    text = load_prospective_config().config_path.read_text()
    p = tmp_path / "p.toml"
    p.write_text(text.replace('prospective_predictions = "DISABLED"', 'prospective_predictions = "ENABLED"'))
    with pytest.raises(ConfigError, match="INCOMPLETE"):
        load_prospective_config(p)


def test_timing_guards_for_prospective_requests():
    cfg = load_prospective_config()
    hist = pl.DataFrame({"tournament_date": [date(2026, 5, 25)]})
    m = UpcomingMatch(
        "x", 1, 2, "t", date(2026, 11, 2), "M", "Hard", "R32", scheduled_start_at="2026-11-03T12:00:00+00:00"
    )
    check_timing(cfg, [m], hist, datetime(2026, 11, 3, 11, tzinfo=UTC))  # before the verified start: allowed
    with pytest.raises(PredictionBlocked, match="after the match start"):
        check_timing(cfg, [m], hist, datetime(2026, 11, 3, 12, tzinfo=UTC))
    early = UpcomingMatch("y", 1, 2, "t", date(2026, 10, 19), "A", "Hard", "R32")
    with pytest.raises(PredictionBlocked, match="before the prospective period"):
        check_timing(cfg, [early], hist, datetime(2026, 10, 1, tzinfo=UTC))
    with pytest.raises(PredictionBlocked, match="contamination"):
        check_timing(
            cfg, [m], pl.DataFrame({"tournament_date": [date(2026, 11, 9)]}), datetime(2026, 11, 1, tzinfo=UTC)
        )


def test_math_guard():
    assert math.isfinite(GOLDEN["expected"]["extreme_elo"]["probability_a"])
