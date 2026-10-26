"""Phase 5A final safety review: availability safeguard, licence records, identity checks, event release flags."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from types import SimpleNamespace

import polars as pl
import pytest

from tennis_data.operations import authorization_review as ar
from tennis_data.operations import availability as av
from tennis_data.operations import exports, licence

POL = av.AvailabilityPolicy()


def _ts(*a):
    return datetime(*a, tzinfo=UTC)


def _rows(match_dates, ingested):
    n = len(match_dates)
    return pl.DataFrame({"state_eligible_at": [_ts(2026, 6, 13)] * n, "release_batch": [date(2026, 6, 6)] * n,
                         "match_date": match_dates, "ingested_at": ingested},
                        schema={"state_eligible_at": av.UTC_TS, "release_batch": pl.Date, "match_date": pl.Date,
                                "ingested_at": av.UTC_TS})  # fmt: skip


def test_availability_never_earlier_than_frozen_and_waits_for_match_day():
    df = _rows([date(2026, 6, 10), date(2026, 6, 14), None],
               [_ts(2026, 6, 11), _ts(2026, 6, 14, 20), _ts(2026, 6, 15)])  # fmt: skip
    a = av.annotate(df, POL)
    assert a["availability_status"].to_list() == ["ok", "ok", "ok"]
    # weekday match: the frozen release (Saturday) is later than the buffer; Sunday final: two days after it;
    # undated: release batch + 21 days
    assert a["available_at"].to_list() == [_ts(2026, 6, 13), _ts(2026, 6, 16), _ts(2026, 6, 27)]
    r = av.evaluate(df, POL, [_ts(2026, 6, 14), _ts(2026, 6, 16)])
    assert r["never_earlier_than_frozen"] and r["frozen_played_on_or_after_release"] == 1
    assert r["safeguard_played_on_or_after_available"] == 0
    assert r["usable_at"] == {"2026-06-14T00:00:00+00:00": 1, "2026-06-16T00:00:00+00:00": 2}


def test_availability_fails_closed_without_evidence_or_with_impossible_ingestion():
    a = av.annotate(_rows([date(2026, 6, 14), date(2026, 6, 14)], [None, _ts(2026, 6, 12)]), POL)
    assert a["availability_status"].to_list() == ["no_availability_evidence", "ingested_before_match_date"]
    assert a["available_at"].null_count() == 2
    assert a.select(av.usable(a, _ts(2030, 1, 1))).to_series().sum() == 0


def test_availability_policy_is_versioned_and_validated(tmp_path):
    p = av.load_policy()
    assert p.version == "state_availability_v1" and p.enabled is False
    assert p.policy_hash() == av.AvailabilityPolicy().policy_hash()
    assert av.AvailabilityPolicy(match_day_buffer_days=3).policy_hash() != p.policy_hash()
    bad = tmp_path / "a.toml"
    bad.write_text('[policy]\nversion = "state_availability_v1"\nmatch_day_buffer_days = 0\n')
    with pytest.raises(ValueError):
        av.load_policy(bad)


def _export(tmp_path):
    cfg = SimpleNamespace(workspace_dir=tmp_path)
    files = []
    for name, role, body in (("2026.csv", "tml_matches", b"a"), ("ATP_Database.csv", "tml_players", b"b"),
                             ("x.csv", "crosscheck_vbt", b"c")):  # fmt: skip
        p = tmp_path / name
        p.write_bytes(body)
        files.append((p, role))
    exports.register(cfg, "L", files, provider="tml_export", origin="test", acquired_on="2026-10-26")
    return cfg


def test_licence_record_keeps_hashes_and_only_named_roles(tmp_path):
    cfg = _export(tmp_path)
    before = json.loads((tmp_path / "exports" / "L" / "manifest.json").read_text())
    man = licence.record(cfg, "L", ["tml_matches", "tml_players"], licence.TENNISMYLIFE, "2026-10-26", "s")
    assert [(f["stored_as"], f["sha256"]) for f in man["files"]] == [
        (f["stored_as"], f["sha256"]) for f in before["files"]
    ]
    st = {f["role"]: f["licence"]["status"] for f in man["files"]}
    assert st == {"tml_matches": "research_use_subject_to_terms", "tml_players": "research_use_subject_to_terms",
                  "crosscheck_vbt": "not verified"}  # fmt: skip
    tml = next(f for f in man["files"] if f["role"] == "tml_matches")["licence"]
    assert tml["commercial_use"].startswith("unverified") and tml["raw_redistribution"].startswith("prohibited")
    again = licence.record(cfg, "L", ["tml_matches", "tml_players"], licence.TENNISMYLIFE, "2026-10-26", "s")
    assert "licence_history" not in again  # idempotent


def test_licence_record_refuses_unknown_roles_and_changed_files(tmp_path):
    cfg = _export(tmp_path)
    with pytest.raises(ValueError, match="no file with role"):
        licence.record(cfg, "L", ["unknown"], licence.TENNISMYLIFE, "2026-10-26", "s")
    (tmp_path / "exports" / "L" / "files" / "crosscheck_vbt.csv").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="do not match"):
        licence.record(cfg, "L", ["tml_matches"], licence.TENNISMYLIFE, "2026-10-26", "s")
    man = json.loads((tmp_path / "exports" / "L" / "manifest.json").read_text())
    assert all("licence" not in f for f in man["files"])  # nothing written


def _arch():
    return {1: {"name": "Ann Example", "birth_date": date(2000, 1, 1), "country": "SLO"},
            2: {"name": "Ann Example", "birth_date": date(2000, 1, 1), "country": "SLO"},
            3: {"name": "Bo Test", "birth_date": date(1960, 1, 1), "country": "USA"},
            4: {"name": "Cy Undated", "birth_date": None, "country": "HKG"},
            5: {"name": "Bo Test", "birth_date": date(1999, 5, 5), "country": "USA"}}  # fmt: skip


def _prop(pid, cid, confirm=0):
    return {"provider": "tml_export", "provider_player_id": pid, "canonical_player_id": cid, "method": "name_dob",
            "overlap_matches_confirming": confirm, "overlap_matches_contradicting": 0}  # fmt: skip


def test_check_mappings_flags():
    hist = {1: {"archive_matches": 10, "first": date(2015, 1, 1), "last": date(2026, 1, 1)},
            3: {"archive_matches": 2, "first": date(1975, 1, 1), "last": date(1985, 1, 1)}}  # fmt: skip
    prov = {"A": {"country": "SVN", "birth_date": date(2000, 1, 1)}, "B": {"country": "USA", "birth_date": date(1960, 1, 1)},
            "C": {"country": "HKG", "birth_date": None}, "D": {"country": "USA", "birth_date": date(1960, 1, 1)}}  # fmt: skip
    gap = {k: date(2026, 7, 1) for k in prov}
    rows = ar.check_mappings([_prop("A", 1, 3), _prop("B", 3), _prop("C", 4), _prop("D", 3)], _arch(), hist, prov, gap,
                             existing={("tml_export", "Z"): 1}, ambiguous_candidates={4})  # fmt: skip
    by = {r["provider_player_id"]: r for r in rows}
    # A: ISO SVN = IOC SLO; another archive record with the same name and birth date; canonical held by an override
    assert "country_differs" not in by["A"]["flags"]
    assert set(by["A"]["blocking"]) == {"same_name_same_birth_date_record", "conflicts_existing_override"}
    assert by["A"]["evidence_class"] == "overlap_corroborated"
    # B and D map to the same canonical player; 66 years old, inactive since 1985, namesake with another birth date
    assert "not_one_to_one" in by["B"]["blocking"] and "not_one_to_one" in by["D"]["blocking"]
    assert {"unusual_age_at_gap_match", "archive_inactive_over_3_years", "same_name_other_birth_date"} <= set(
        by["B"]["flags"]
    )
    # C: no birth date on either side, canonical also an ambiguous candidate, no history
    assert {"birth_date_conflict", "canonical_is_ambiguous_candidate"} <= set(by["C"]["blocking"])
    assert "no_archive_history" in by["C"]["flags"] and by["C"]["review_decision"] == "PENDING"


def _entry(cands, name="Ann Example", dob="2000-01-01", ctry="SVN", cat="ambiguous"):
    return {"provider_player_id": "P", "name": name, "birth_date": dob, "country": ctry, "method": "m", "category": cat,
            "candidates": cands}  # fmt: skip


def _c(i, name="Ann Example", dob="2000-01-01", ctry="SLO"):
    return {"id": i, "name": name, "birth_date": dob, "country": ctry}


def test_propose_ambiguous_only_when_defensible():
    h = {1: {"archive_matches": 5, "last": date(2026, 1, 1)}, 2: {"archive_matches": 0, "last": None},
         3: {"archive_matches": 7, "last": date(2025, 1, 1)}}  # fmt: skip
    assert ar.propose_ambiguous(_entry([_c(1)]), h, 3)["proposed_canonical_id"] == 1
    dup = ar.propose_ambiguous(_entry([_c(1), _c(2)]), h, 3)  # duplicate archive IDs, one with matches
    assert dup["proposed_canonical_id"] == 1 and "duplicate archive IDs" in dup["reason"]
    both = ar.propose_ambiguous(_entry([_c(1), _c(3)]), h, 3)
    assert both["proposed_canonical_id"] is None and both["grade"] == "UNRESOLVED"
    spelled = ar.propose_ambiguous(_entry([_c(1, name="Anne Exampel")]), h, 3)
    assert spelled["proposed_canonical_id"] is None and "different spelling" in spelled["reason"]
    joined = ar.propose_ambiguous(_entry([_c(1, name="An N Example")], name="Ann Example"), h, 3)
    assert joined["proposed_canonical_id"] == 1  # same letters, split differently
    assert ar.propose_ambiguous(_entry([_c(1)], dob=None), h, 3)["proposed_canonical_id"] is None
    undated = ar.propose_ambiguous(_entry([_c(1, dob=None)]), h, 3)
    assert undated["proposed_canonical_id"] is None and "no birth date" in undated["reason"]
    new = ar.propose_ambiguous(_entry([], cat="unmatched"), h, 1)
    assert new["proposed_canonical_id"] is None and new["affected_matches"] == 1


def test_review_events_flags():
    base = {"level": "A", "draw_size": 28, "weeks": 1, "batch": "2026-06-06", "release_batch": "2026-06-06",
            "matches": 27}  # fmt: skip
    ok, late, us, masters = ar.review_events([
        {**base, "tournament": "Ok", "start": "2026-06-08", "last_match": "2026-06-12", "enters_state": "2026-06-13",
         "played_on_or_after": 0},
        {**base, "tournament": "Davis Cup X", "start": "2026-06-12", "last_match": "2026-06-14",
         "enters_state": "2026-06-13", "played_on_or_after": 2},
        {**base, "tournament": "US", "level": "G", "weeks": 2, "start": "2026-08-30", "last_match": "2026-09-13",
         "enters_state": "2026-09-12", "played_on_or_after": 1},
        {**base, "tournament": "Paris", "level": "M", "draw_size": 56, "start": "2026-10-31",
         "last_match": "2026-11-08", "enters_state": "2026-11-07", "played_on_or_after": 2}])  # fmt: skip
    assert ok["flags"] == [] and ok["overrun_days"] == 0
    assert set(late["flags"]) == {"results_released_on_or_before_match_day", "starts_mid_week", "team_event"}
    assert late["overrun_days"] == 2
    assert "event_longer_than_frozen_weeks" in us["flags"] and us["span_days"] == 15
    assert {"masters_below_96_draw_one_week_rule", "event_longer_than_frozen_weeks"} <= set(masters["flags"])
