"""Experimental single-match predictions on the synthetic world: player-order symmetry, cutoff enforcement,
unavailable or unverifiable state, stale-data labelling, name resolution, and that the gate and ledger are untouched."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta

import polars as pl
import pytest

from conftest import FakeArchive
from prospective_helpers import build_world, prospective_config
from tennis_data.cli import main
from tennis_data.prospective.backfill import load_state_snapshot
from tennis_data.prospective.config import load_prospective_config
from tennis_data.prospective.experimental import ExperimentalRefused, predict_match, render, snapshot_store
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


def test_results_after_the_cutoff_are_removed_not_used(world, first_valid, pair):
    """A state holding a later result is cut back to the cutoff; the prediction equals the clean state's."""
    snap = world["snap"]
    m = snap.matches
    late = m.head(1).with_columns(pl.lit("late-result").alias("match_id"),
                                  pl.lit(first_valid + timedelta(days=7)).alias("tournament_date"))  # fmt: skip
    contaminated = make_snapshot(pl.concat([m, late]), snap.quarantine_ids, {"test": "late"})
    r = predict_match(world["cfg"], "Alice Archer", "Hal Hunt", "Hard", first_valid, snapshot=contaminated,
                      now=datetime(2020, 6, 1, tzinfo=UTC))  # fmt: skip
    assert r["state"]["results_removed_for_cutoff"] == 1 and "POINT-IN-TIME RECONSTRUCTION" in r["labels"]
    assert r["state"]["components"]["matches"] == snap.components["matches"]  # same rows as the clean state
    assert r["state"]["components"]["derived_from"] == contaminated.snapshot_id
    assert r["player_a"]["probability"] == pytest.approx(pair[0]["player_a"]["probability"], abs=1e-15)


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


# ------------------------------------------------------------------------------- registered snapshot selection


BF_WEEKS = (date(2020, 6, 15), date(2020, 6, 22), date(2020, 7, 6))


def _write_state_dir(src, archive_matches, extra: pl.DataFrame | None, cfg) -> str:
    """A Phase 4A directory (optionally with accepted backfill rows); returns its state ID."""
    from tennis_data.prospective.backfill import load_archive

    src.mkdir(parents=True)
    (src / "phase4a_manifest.json").write_text("{}")
    if extra is None:
        return load_state_snapshot(replace(cfg, prospective_dir=src)).snapshot_id
    bf = src / "backfill"
    bf.mkdir()
    extra.write_parquet(bf / "matches.parquet")
    hist, qids = load_archive(cfg)
    comp = {"archive_commit": cfg.data.source.commit, "backfill_raw_snapshot": "synthetic"}
    sid = make_snapshot(pl.concat([hist.select(extra.columns), extra]), qids, comp).snapshot_id
    (bf / "backfill_summary.json").write_text(json.dumps({"raw_snapshot_id": "synthetic", "state_snapshot_id": sid}))
    return sid


@pytest.fixture()
def store(world, tmp_path):
    """A content-addressed store with a version-0 baseline and an inactive results-only version 2."""
    from tennis_data.operations.controlled_backfill import snapshot_dir

    cfg = replace(world["cfg"], prospective_dir=tmp_path / "prospective")
    st = snapshot_store(cfg)
    m = world["snap"].matches
    hunt_wins = m.filter((pl.col("player_a_id").is_in([100, 800])) & (pl.col("player_b_id").is_in([100, 800])))
    rows = pl.concat([m.tail(6), hunt_wins.tail(2)])
    extra = pl.concat([
        rows.with_columns(pl.format("tour:bf-{}-{}", pl.lit(i), pl.col("match_id")).alias("match_id"),
                          pl.lit(f"bf-{i}").alias("tournament_id"), pl.lit(d).alias("tournament_date"),
                          pl.lit(d.year, dtype=pl.Int32).alias("season"), pl.lit("none").alias("stats_status"),
                          *[pl.lit(None, dtype=rows.schema[c]).alias(c) for c in rows.columns if "_svpt" in c
                            or "_first_" in c or "_second_" in c])
        for i, d in enumerate(BF_WEEKS)])  # fmt: skip
    v0_id = _write_state_dir(tmp_path / "src0", m, None, cfg)
    v2_id = _write_state_dir(tmp_path / "src2", m, extra, cfg)
    base = "prospective-baseline-" + v0_id
    man0 = snapshot_dir(tmp_path / "src0", st, base, {"state_snapshot_id": v0_id,
                        "state_components": {"archive_commit": cfg.data.source.commit}})  # fmt: skip
    man2 = snapshot_dir(tmp_path / "src2", st, "historical-v2-" + v2_id,
                        {"state_snapshot_id": v2_id, "parent": base, "role": "results-only test backfill"})  # fmt: skip
    reg = {"format": "snapshot-registry-1", "snapshots": [
        {"version": 0, "name": base, "state_snapshot_id": v0_id, "tree_hash": man0["tree_hash"],
         "active_phase4a_directory": True},
        {"version": 2, "name": man2["name"], "state_snapshot_id": v2_id, "tree_hash": man2["tree_hash"],
         "parent": base, "active_phase4a_directory": False}]}  # fmt: skip
    (st / "registry.json").write_text(json.dumps(reg))
    return {"cfg": cfg, "store": st, "v0": v0_id, "v2": v2_id, "man2": man2}


def _tree(path):
    return {p.relative_to(path).as_posix(): p.read_bytes() for p in sorted(path.rglob("*")) if p.is_file()}


LATER = date(2020, 8, 3)  # after every synthetic backfill week


def test_v2_is_verified_restored_read_only_and_never_activated(store):
    before = _tree(store["store"])
    r = predict_match(store["cfg"], "Alice Archer", "Hal Hunt", "Hard", LATER, state_version="v2")
    assert r["state"]["version"] == 2 and r["state"]["active"] is False
    assert r["state"]["selected_state_snapshot_id"] == store["v2"] == r["state"]["state_snapshot_id"]
    assert r["state"]["parent_state_snapshot_id"] == store["v0"]
    assert "INACTIVE STATE v2" in r["labels"] and r["labels"][0] == "EXPERIMENTAL"
    assert _tree(store["store"]) == before  # store, manifests and registry untouched
    assert not store["cfg"].prospective_dir.exists()  # nothing written to a Phase 4A directory
    reg = json.loads((store["store"] / "registry.json").read_text())
    assert [s["active_phase4a_directory"] for s in reg["snapshots"]] == [True, False]


def test_v2_mixed_freshness_and_difference_from_v0(store):
    v0 = predict_match(store["cfg"], "Alice Archer", "Hal Hunt", "Hard", LATER, state_version="v0")
    v2 = predict_match(store["cfg"], "Alice Archer", "Hal Hunt", "Hard", LATER, state_version="v2")
    assert v2["elo_freshness"]["status"] != "fail" and v2["statistics_freshness"]["status"] == "fail"
    assert "MIXED FRESHNESS" in v2["labels"] and any(w.startswith("mixed freshness") for w in v2["warnings"])
    assert v0["elo_freshness"]["status"] == "fail" and "MIXED FRESHNESS" not in v0["labels"]
    assert v2["player_a"]["global_elo_matches"] > v0["player_a"]["global_elo_matches"]  # backfilled results used
    assert v2["player_a"]["probability"] != pytest.approx(v0["player_a"]["probability"], abs=1e-9)
    text = render(v2)
    assert "MIXED FRESHNESS" in text.splitlines()[0] and "INACTIVE, not activated" in text
    assert v2["player_a"]["results_after_archive_end"] > 0 and v0["player_a"]["results_after_archive_end"] == 0
    assert any(w.startswith("backfill after the archive end") for w in v2["warnings"])
    assert not any(w.startswith("backfill after the archive end") for w in v0["warnings"])


def test_v2_symmetry(store):
    ab = predict_match(store["cfg"], "Alice Archer", "Hal Hunt", "Clay", LATER, state_version="v2")
    ba = predict_match(store["cfg"], "Hal Hunt", "Alice Archer", "Clay", LATER, state_version="v2")
    assert ab["player_a"]["probability"] == pytest.approx(ba["player_b"]["probability"], abs=1e-12)
    assert ab["player_a"]["surface_elo_probability"] == pytest.approx(ba["player_b"]["surface_elo_probability"],
                                                                      abs=1e-12)  # fmt: skip


def test_v2_before_its_backfill_is_cut_back_to_the_cutoff(store, first_valid):
    """A cutoff before the backfilled weeks: v2 is reduced to an earlier point-in-time state equal to v0."""
    v0 = predict_match(store["cfg"], "Alice Archer", "Hal Hunt", "Hard", first_valid, state_version="v0")
    v2 = predict_match(store["cfg"], "Alice Archer", "Hal Hunt", "Hard", first_valid, state_version="v2")
    assert v2["state"]["results_removed_for_cutoff"] == 3 * 8
    assert v2["state"]["components"]["matches"] == v0["state"]["components"]["matches"]  # same rows as v0
    assert v2["state"]["components"]["derived_from"] == store["v2"] == v2["state"]["selected_state_snapshot_id"]
    assert "POINT-IN-TIME RECONSTRUCTION" in v2["labels"]
    assert v2["player_a"]["probability"] == pytest.approx(v0["player_a"]["probability"], abs=1e-15)
    mid = predict_match(store["cfg"], "Alice Archer", "Hal Hunt", "Hard", date(2020, 6, 29), state_version="v2")
    assert mid["state"]["results_removed_for_cutoff"] == 8  # only the 2020-07-06 week is after this cutoff
    assert date.fromisoformat(mid["data_cutoff"]["latest_event_start_in_state"]) < date(2020, 6, 27)


def test_tampered_snapshot_object_is_refused(store):
    f = next(iter(store["man2"]["files"].values()))
    obj = store["store"] / "objects" / f["sha256"][:2] / f["sha256"]
    obj.write_bytes(obj.read_bytes() + b"x")
    with pytest.raises(ExperimentalRefused, match="failed verification"):
        predict_match(store["cfg"], "Alice Archer", "Hal Hunt", "Hard", LATER, state_version="v2")


def test_registry_and_provenance_mismatches_are_refused(store):
    path = store["store"] / "registry.json"
    reg = json.loads(path.read_text())
    reg["snapshots"][1]["state_snapshot_id"] = "f" * 16
    path.write_text(json.dumps(reg))
    with pytest.raises(ExperimentalRefused, match="disagree"):
        predict_match(store["cfg"], "Alice Archer", "Hal Hunt", "Hard", LATER, state_version="v2")
    reg["snapshots"][1]["state_snapshot_id"] = store["v2"]
    reg["snapshots"][1]["parent"] = "someone-else"
    path.write_text(json.dumps(reg))
    with pytest.raises(ExperimentalRefused, match="parent"):
        predict_match(store["cfg"], "Alice Archer", "Hal Hunt", "Hard", LATER, state_version="v2")


def test_missing_snapshot_is_refused(store, world):
    with pytest.raises(ExperimentalRefused, match="not registered"):
        predict_match(store["cfg"], "Alice Archer", "Hal Hunt", "Hard", LATER, state_version="v1")
    (store["store"] / "manifests" / f"{store['man2']['name']}.json").unlink()
    with pytest.raises(ExperimentalRefused, match="not available locally"):
        predict_match(store["cfg"], "Alice Archer", "Hal Hunt", "Hard", LATER, state_version="v2")
    with pytest.raises(ExperimentalRefused, match="no snapshot registry"):  # the synthetic world has no store
        predict_match(world["cfg"], "Alice Archer", "Hal Hunt", "Hard", LATER, state_version="v2")
