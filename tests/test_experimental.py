"""Experimental single-match predictions on the synthetic world: player-order symmetry, cutoff enforcement,
unavailable or unverifiable state, stale-data labelling, name resolution, and that the gate and ledger are untouched."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta

import polars as pl
import pytest

from conftest import FakeArchive
from prospective_helpers import build_world, prospective_config
from tennis_data.cli import main
from tennis_data.prospective.backfill import load_state_snapshot
from tennis_data.prospective.config import load_prospective_config
from tennis_data.prospective.experimental import ExperimentalRefused, predict_match, render
from tennis_data.prospective.inference import load_production
from tennis_data.prospective.state import make_snapshot


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("exp")
    (tmp / "archive").mkdir()
    cfg, _ = build_world(tmp, FakeArchive(tmp / "archive"))
    snap = load_state_snapshot(cfg)
    cutoff = date.fromisoformat(load_production(cfg).training_cutoff)
    return {"cfg": cfg, "snap": snap, "cutoff": cutoff, "tmp": tmp}


@pytest.fixture(scope="module")
def first_valid(world):
    """The first date whose weekly batch is not before the production fit's training cutoff."""
    return world["cutoff"] + timedelta(days=1)


@pytest.fixture(scope="module")
def pair(world, first_valid):
    cfg, snap = world["cfg"], world["snap"]
    now = datetime(2020, 6, 1, tzinfo=UTC)
    ab = predict_match(cfg, "Alice Archer", "Hal Hunt", "Hard", first_valid, now=now, snapshot=snap)
    ba = predict_match(cfg, "Hal Hunt", "Alice Archer", "Hard", first_valid, now=now, snapshot=snap)
    return ab, ba


def test_player_order_symmetry(pair):
    ab, ba = pair
    assert ab["player_a"]["probability"] + ab["player_b"]["probability"] == pytest.approx(1, abs=1e-12)
    assert ab["player_a"]["probability"] == pytest.approx(ba["player_b"]["probability"], abs=1e-12)
    assert ab["player_a"]["surface_elo_probability"] == pytest.approx(ba["player_b"]["surface_elo_probability"],
                                                                      abs=1e-12)  # fmt: skip
    assert ab["player_a"]["probability"] != pytest.approx(0.5)  # the players differ in skill


def test_report_identifies_model_state_and_cutoffs(world, pair):
    ab, _ = pair
    assert ab["model"]["frozen_hash"] == load_production(world["cfg"]).frozen_hash[:16]
    assert ab["model"]["artifact"] == "production"
    assert ab["state"]["state_snapshot_id"] == world["snap"].snapshot_id
    assert date.fromisoformat(ab["prediction_cutoff_batch_start"]) >= world["cutoff"]
    assert date.fromisoformat(ab["data_cutoff"]["last_results_entered_state"]) <= date.fromisoformat(
        ab["prediction_cutoff_batch_start"]
    )
    assert ab["prediction_mode"] == "experimental" and "EXPERIMENTAL" in ab["labels"]
    assert "not an independently verified prospective forecast" in ab["notice"]


def test_cutoff_before_training_cutoff_is_refused(world):
    with pytest.raises(ExperimentalRefused, match="training cutoff"):
        predict_match(world["cfg"], "Alice Archer", "Hal Hunt", "Hard", world["cutoff"] - timedelta(days=14),
                      snapshot=world["snap"])  # fmt: skip


def test_state_with_results_released_after_the_cutoff_is_refused(world, first_valid):
    snap = world["snap"]
    m = snap.matches
    late = m.head(1).with_columns(pl.lit("late-result").alias("match_id"),
                                  pl.lit(first_valid + timedelta(days=7)).alias("tournament_date"))  # fmt: skip
    contaminated = make_snapshot(pl.concat([m, late]), snap.quarantine_ids, {"test": "late"})
    with pytest.raises(ExperimentalRefused, match="released at or after the cutoff"):
        predict_match(world["cfg"], "Alice Archer", "Hal Hunt", "Hard", first_valid, snapshot=contaminated)


def test_unverifiable_state_is_refused(world, first_valid, tmp_path):
    cfg = load_prospective_config(prospective_config(tmp_path, world["cfg"].modeling.config_path,
                                                     world["cfg"].expected_frozen_hash))  # fmt: skip
    bf = cfg.prospective_dir / "backfill"
    bf.mkdir(parents=True)
    world["snap"].matches.head(0).write_parquet(bf / "matches.parquet")
    (bf / "backfill_summary.json").write_text(json.dumps({"raw_snapshot_id": "x", "state_snapshot_id": "0" * 16}))
    with pytest.raises(ExperimentalRefused, match="no verified state"):
        predict_match(cfg, "Alice Archer", "Hal Hunt", "Hard", first_valid)


def test_missing_state_is_refused(world, first_valid, tmp_path):
    """A clone without the local research data (no validated archive) has no state to predict from."""
    cfg = world["cfg"]
    validated = cfg.data.paths.validated_dir
    hidden = validated.with_name(validated.name + "_hidden")
    validated.rename(hidden)
    try:
        with pytest.raises(ExperimentalRefused, match="no verified state"):
            predict_match(cfg, "Alice Archer", "Hal Hunt", "Hard", first_valid)
    finally:
        hidden.rename(validated)


def test_stale_state_is_labelled(world):
    late = predict_match(world["cfg"], "Alice Archer", "Hal Hunt", "Hard", world["cutoff"] + timedelta(days=365),
                         snapshot=world["snap"])  # fmt: skip
    assert late["freshness"]["status"] == "fail"
    assert late["labels"] == ["EXPERIMENTAL", "STALE DATA"]
    text = render(late)
    assert text.splitlines()[0] == "[EXPERIMENTAL / STALE DATA]" and "STALE DATA:" in text


def test_fresh_state_is_labelled_experimental_only(pair):
    ab, _ = pair
    assert ab["freshness"]["status"] != "fail"  # one week after the last synthetic results
    assert ab["labels"] == ["EXPERIMENTAL"] and ab["stale_notice"] is None
    assert render(ab).splitlines()[0] == "[EXPERIMENTAL]"


@pytest.mark.parametrize(
    ("query", "expected"), [("Alice Archer", 100), ("archer alice", 100), ("Archer", 100), ("800", 800)]
)
def test_player_resolution(world, first_valid, query, expected):
    r = predict_match(world["cfg"], query, "Bob Baker", "Hard", first_valid, snapshot=world["snap"])
    assert r["player_a"]["player_id"] == expected


@pytest.mark.parametrize("query", ["Nobody Known", "Ian Ives", "Unknown", "123456"])
def test_unknown_or_stateless_players_are_refused(world, first_valid, query):
    with pytest.raises(ExperimentalRefused):
        predict_match(world["cfg"], query, "Bob Baker", "Hard", first_valid, snapshot=world["snap"])


def test_invalid_requests_are_refused(world, first_valid):
    cfg, snap = world["cfg"], world["snap"]
    with pytest.raises(ExperimentalRefused, match="same player"):
        predict_match(cfg, "Alice Archer", "100", "Hard", first_valid, snapshot=snap)
    with pytest.raises(ExperimentalRefused, match="surface"):
        predict_match(cfg, "Alice Archer", "Hal Hunt", "Sand", first_valid, snapshot=snap)
    with pytest.raises(ExperimentalRefused, match="event start"):
        predict_match(cfg, "Alice Archer", "Hal Hunt", "Hard", first_valid,
                      event_start=first_valid + timedelta(days=1), snapshot=snap)  # fmt: skip


def test_gate_and_ledger_untouched(world, first_valid, capsys):
    cfg = world["cfg"]
    before = sorted(p for p in world["tmp"].rglob("*") if p.is_file())
    rc = main(["predict-experimental", "--player-a", "Alice Archer", "--player-b", "Hal Hunt", "--surface", "Hard",
               "--date", str(first_valid), "--prospective-config", str(cfg.config_path)])  # fmt: skip
    out = capsys.readouterr().out
    assert rc == 0 and out.startswith("[EXPERIMENTAL")
    assert "prospective predictions DISABLED" in out and "nothing written to the ledger" in out
    assert sorted(p for p in world["tmp"].rglob("*") if p.is_file()) == before  # no file written anywhere
    assert load_prospective_config(cfg.config_path).prospective_predictions == "DISABLED"
