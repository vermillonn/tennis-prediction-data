"""Phase 5A: provider access classification, TML conversion, compatibility study, gap audit, readiness, closed gate,
and the operations pipeline end to end on the synthetic Phase 4A world (no network, no real provider data)."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import tomllib
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import pytest

from conftest import FakeArchive
from prospective_helpers import build_world
from tennis_data.config import PROJECT_ROOT
from tennis_data.normalization.schema import SOURCE_MATCH_COLUMNS
from tennis_data.operations import access, compatibility, coverage, readiness, tml
from tennis_data.operations.config import Provider, load_operations_config
from tennis_data.prospective.providers.fixture import FixtureProvider

# ------------------------------------------------------------------------------------------------- access


def _prov(**kw):
    base = dict(name="p", kind="api", probe="https://api.example.test/x", credential_env="EX_KEY", auth="header:x-key",
                verify=None, offers={}, terms="t")  # fmt: skip
    return Provider(**{**base, **kw})


def _sackmann_csv(rows=1) -> bytes:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(SOURCE_MATCH_COLUMNS))
    w.writeheader()
    for i in range(rows):
        w.writerow({c: "" for c in SOURCE_MATCH_COLUMNS} | {"tourney_date": f"2026010{i + 1}", "w_svpt": "60"})
    return buf.getvalue().encode()


def test_access_outcomes_are_distinct():
    def blocked(url, headers, timeout):
        raise access.NetworkFailure("proxy", "egress proxy refused the connection (403 Forbidden)")

    def dns(url, headers, timeout):
        raise access.NetworkFailure("dns", "host name did not resolve")

    def status(code, body=b""):
        return lambda url, headers, timeout: (code, body)

    p = _prov()
    assert access.probe(p, blocked, {})["status"] == "NETWORK_BLOCKED"
    assert access.probe(p, dns, {})["status"] == "DNS_FAILURE"
    absent = access.probe(p, status(401), {})
    assert absent["status"] == "CREDENTIALS_ABSENT" and absent["credentials"] == "absent"
    assert access.probe(p, status(403), {"EX_KEY": "k"})["status"] == "AUTH_REJECTED"  # present but refused
    assert access.probe(p, status(429), {})["status"] == "RATE_LIMITED"
    assert access.probe(p, status(404), {})["status"] == "NOT_FOUND"
    api_200 = access.probe(p, status(200, b"{}"), {})
    assert api_200["status"] == "REACHABLE" and "requires EX_KEY" in api_200["detail"]  # not verified data
    repo = _prov(kind="repo", credential_env=None, auth=None, verify="sackmann_csv")
    ok = access.probe(repo, status(200, _sackmann_csv(2)), {})
    assert ok["status"] == "DATA_VERIFIED" and ok["data"]["rows"] == 2 and ok["data"]["rows_with_serve_stats"] == 2
    assert access.probe(repo, status(200, b"a,b\n1,2\n"), {})["status"] == "SCHEMA_MISMATCH"
    assert access.probe(_prov(probe=""), status(200), {})["status"] == "NOT_PROBED"


def test_credentials_are_attached_only_to_the_provider_and_never_reported():
    seen = []

    def t(url, headers, timeout):
        seen.append((url, headers))
        return 401, b""

    secret = "s3cr3t-value"
    out = access.audit([_prov(), _prov(name="q", auth="query:APIkey")], t, {"EX_KEY": secret})
    assert seen[0][1] == {"x-key": secret} and "APIkey=" + secret in seen[1][0]
    assert secret not in json.dumps(out)
    assert {r["status"] for r in out["results"]} == {"AUTH_REJECTED"} and out["verified_sources"] == []


# ------------------------------------------------------------------------------------------------- TML conversion


def _tml_files(rows, players, player_encoding="cp1252"):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(SOURCE_MATCH_COLUMNS))
    w.writeheader()
    for r in rows:
        w.writerow({c: "" for c in SOURCE_MATCH_COLUMNS} | r)
    pbuf = io.StringIO()
    pw = csv.writer(pbuf, quoting=csv.QUOTE_ALL)
    pw.writerow(["id", "player", "atpname", "birthdate", "weight", "height", "turnedpro", "birthplace", "coaches", "hand",
                 "backhand", "ioc"])  # fmt: skip
    for p in players:
        pw.writerow([p[0], p[1], p[1], p[2], "", "", "", "", "", "R", "2H", p[3]])
    return {"2026.csv": buf.getvalue().encode(), tml.PLAYER_FILE: pbuf.getvalue().encode(player_encoding)}


def _row(tid, num, rnd, w, lo, level="250", date_="20260105", score="6-4 6-4", **kw):
    base = {"tourney_id": tid, "tourney_name": "Synthetic", "surface": "Hard", "draw_size": "32", "tourney_level": level,
            "tourney_date": date_, "match_num": num, "winner_id": w, "loser_id": lo, "winner_name": w,
            "loser_name": lo, "score": score, "best_of": "3", "round": rnd, "w_svpt": "60", "w_1stIn": "38",
            "w_1stWon": "30", "w_2ndWon": "12", "l_svpt": "65", "l_1stIn": "40", "l_1stWon": "25", "l_2ndWon": "10"}  # fmt: skip
    return base | kw


def test_tml_conversion_is_deterministic_and_never_guesses():
    rows = [_row("t1", "1", "R32", "A1", "B2"), _row("t1", "", "R16", "A1", "C3", level="500"),
            _row("t1", "3", "R16", "D4", "", score="6-3 6-2"), _row("t2", "1", "R32", "A1", "B2", level="X")]  # fmt: skip
    players = [("A1", "Jose Ramón Núñez", "19990101", "ESP"), ("B2", "Li Wei", "20000202", "CHN"),
               ("C3", "Ann Bee", "", "USA"), ("D4", "Don Dee", "19980303", "USA")]  # fmt: skip
    files = _tml_files(rows, players)
    doc, log = tml.convert(files, {"250": "A", "500": "A"}, "2026-01-10T00:00:00+00:00")
    again, _ = tml.convert(files, {"250": "A", "500": "A"}, "2026-01-10T00:00:00+00:00")
    assert doc == again and log["player_file_encoding"] == "cp1252"
    assert log["missing_match_num"] == 1 and len(log["rejected_missing_player_id"]) == 1
    assert log["unmapped_levels"] == {"X": 1}  # passed through for Phase 4A to quarantine
    ids = [m["id"] for m in doc["matches"]]
    assert "t1:R16:A1-C3" in ids and len(ids) == 3
    a1 = next(p for p in doc["players"] if p["id"] == "A1")
    assert a1["first_name"] == "Jose" and a1["last_name"] == "Ramón Núñez" and a1["birth_date"] == "1999-01-01"
    assert "atp_id" not in a1  # no claimed archive id: the identity resolver decides
    assert all(m["stats_definition"] == tml.STATS_DEFINITION for m in doc["matches"])
    players_df, matches, fixtures, errors = FixtureProvider(None).parse(doc, "sha", datetime.now(UTC))
    assert matches.height == 3 and fixtures.height == 0 and not errors
    assert set(matches["tournament_level"]) == {"A", "X"}


# ------------------------------------------------------------------------------------------------- compatibility


def _cfg_ops(**over):
    return replace(load_operations_config(), **over)


def _hist(rows):
    cols = {"match_id": pl.String, "tournament_id": pl.String, "season": pl.Int32, "tournament_date": pl.Date, "tournament_name": pl.String,
            "tournament_level": pl.String, "round": pl.String, "player_a_id": pl.Int64, "player_b_id": pl.Int64,
            "winner_id": pl.Int64, "score": pl.String, "match_status": pl.String,
            **{f"player_{s}_{f}": pl.Int64 for s in "ab" for f in compatibility.FIELDS.values()}}  # fmt: skip
    return pl.DataFrame(rows, schema=cols, orient="row") if rows else pl.DataFrame(schema=cols)


def _cand(rows):
    from tennis_data.prospective.schema import MATCH_SCHEMA, frame

    return frame(rows, MATCH_SCHEMA)


def _c(mid, w, lo, rnd="R32", d=date(2025, 3, 3), name="Ev", **stats):
    base = {"provider": "tml_github", "provider_match_id": mid, "source_version": 1, "tournament_name": name,
            "tournament_level": "A", "event_start_date": d, "round": rnd, "status": "completed", "score": "6-4 6-4",
            "winner_provider_id": w, "loser_provider_id": lo, "stats_definition": "tml_unverified"}  # fmt: skip
    for k in compatibility.FIELDS:
        base[f"w_{k}"], base[f"l_{k}"] = 50, 40
    return base | stats


def _h(mid, a, b, winner, rnd="R32", d=date(2025, 3, 3), **stats):
    r = {"match_id": mid, "tournament_id": f"t-{d.isoformat()}", "season": d.year, "tournament_date": d, "tournament_name": "Ev", "tournament_level": "A",
         "round": rnd, "player_a_id": a, "player_b_id": b, "winner_id": winner, "score": "6-4 6-4",
         "match_status": "completed"}  # fmt: skip
    for f in compatibility.FIELDS.values():
        r[f"player_a_{f}"] = 50 if a == winner else 40
        r[f"player_b_{f}"] = 40 if a == winner else 50
    return r | stats


def test_compatibility_measures_agreement_and_swaps():
    dec = pl.DataFrame({"provider_player_id": ["W", "L", "X", "Y", "Z", "Q"], "category": ["high_confidence"] * 4 +
                        ["ambiguous", "high_confidence"], "canonical_player_id": [1, 2, 3, 4, None, 5]})  # fmt: skip
    cand = _cand([_c("m1", "W", "L"), _c("m2", "X", "Y", **{"w_svpt": 51}), _c("m3", "Z", "W"),  # Z ambiguous
                  _c("m4", "X", "W", rnd="QF"), _c("m5", "Q", "W", rnd="SF")])  # fmt: skip
    hist = _hist([_h("h1", 1, 2, 1), _h("h2", 3, 4, 3), _h("h4a", 3, 1, 3, rnd="QF"), _h("h4b", 1, 3, 3, rnd="QF"),
                  _h("h5", 5, 1, 1, rnd="SF")])  # h5: archive says 1 won  # fmt: skip
    m = compatibility.match_pairs(cand, dec, hist, 7)
    assert m["accounting"] == {"candidate_records": 5, "unmapped_player": 1, "no_archive_match": 0,
                               "several_archive_matches": 1, "matched": 3}  # fmt: skip
    res = compatibility.compare(m["pairs"], min_matched=1, min_exact=0.98)
    assert res["fields"]["svpt"]["comparable"] == 6 and res["fields"]["svpt"]["exact"] == 3  # m2 loser + m1 both
    assert res["winner_agreement"] == 2 and res["winner_disagreements"][0]["round"] == "SF"
    assert not res["statistics_compatible"] and res["mismatches_explained_by_swapped_sides"] == 1  # h5: sides swap
    big = compatibility.compare(m["pairs"][:1], min_matched=1, min_exact=0.98)
    assert big["statistics_compatible"]  # identical values on enough observations
    assert not compatibility.compare(m["pairs"][:1], min_matched=500, min_exact=0.98)["statistics_compatible"]


# ------------------------------------------------------------------------------------------------- gap audit


def test_gap_weeks_and_statuses():
    assert coverage.weeks(date(2026, 6, 1), date(2026, 6, 14)) == [date(2026, 6, 1)]  # week of 06-08 still running
    assert coverage.weeks(date(2026, 6, 1), date(2026, 6, 15)) == [date(2026, 6, 1), date(2026, 6, 8)]
    cfg = _cfg_ops(first_week=date(2026, 6, 1), calendar_reference_season=2025)
    hist = _hist([_h("r1", 1, 2, 1, d=date(2025, 6, 2)), _h("r2", 3, 4, 3, d=date(2025, 6, 9))])
    hist = hist.with_columns(pl.col("tournament_name").replace({"Ev": "Ref"}))
    cand = _cand([_c("c1", "W", "L", d=date(2026, 6, 1), name="Ref")])
    dec = pl.DataFrame({"provider_player_id": ["W", "L"], "usable": [True, False]})
    acc = pl.DataFrame(schema={"tournament_date": pl.Date, "stats_status": pl.String})
    q = pl.DataFrame({"provider": ["tml_github"], "provider_match_id": ["c1"], "reason": ["identity_unresolved"]})
    out = coverage.audit(cfg, date(2026, 6, 21), hist, cand, acc, q, dec)
    w1, w2 = out["weeks"]
    assert (
        w1["status"] == "PARTIAL"
        and w1["retrieved"] == 1
        and w1["quarantined"] == 1
        and w1["unresolved_identities"] == 1
    )
    assert w2["status"] == "NO_DATA" and w2["expected_events_inferred"] == ["Ref"]
    assert all(w["status"] != "COMPLETE" for w in out["weeks"]) and "inferred" in out["calendar_basis"]


# ------------------------------------------------------------------------------------------------- readiness


def _all_pass_evidence():
    return {"as_of": "2030-01-01", "pools": ["A"],
            "coverage": {"weeks_total": 2, "weeks_validated": 2, "expected_events_inferred_total": 3,
                         "observed_expected_events": 3, "validated_levels": ["A"]},
            "compatibility": {"statistics_compatible": True, "summary": "ok"}, "no_imputation": True,
            "identity": {"unresolved": 0, "records_usable": True},
            "freshness": {"result_age_days": 2, "stats_age_days": 2, "fail_result_age_days": 35},
            "schedule": {"verified_sources": ["p"], "fixtures": 3, "fixtures_resolved": True,
                         "actual_start_sources": ["p"]},
            "model": {"hashes_match": True, "replay_passed": True}, "ledger_config_valid": True,
            "timestamp_authority": {"enabled": True, "trusted_cas": 1},
            "gate": {"backfill": "INCOMPLETE", "prospective_predictions": "DISABLED"}}  # fmt: skip


def test_readiness_requires_every_check_and_never_applies_a_change():
    req = load_operations_config().required_checks
    ev = _all_pass_evidence()
    before = json.dumps(ev, sort_keys=True)
    ok = readiness.evaluate(ev, req)
    assert ok["verdict"] == "READY_TO_PROPOSE_ACTIVATION" and ok["failing_required"] == []
    assert ok["proposed_configuration_change"]["applied"] is False and not ok["gate_changed_by_evaluator"]
    assert json.dumps(ev, sort_keys=True) == before and readiness.evaluate(ev, req) == ok  # pure, deterministic
    for key, bad in (("schedule", {"verified_sources": []}), ("timestamp_authority", {"enabled": False}),
                     ("compatibility", {"statistics_compatible": False}),
                     ("identity", {"unresolved": 1, "records_usable": True})):  # fmt: skip
        r = readiness.evaluate({**ev, key: bad}, req)
        assert r["verdict"] == "NOT_READY" and r["proposed_configuration_change"] is None
    opened = readiness.evaluate({**ev, "schedule": {}, "gate": {"backfill": "COMPLETE",
                                                                 "prospective_predictions": "ENABLED"}}, req)  # fmt: skip
    assert {c["id"]: c["status"] for c in opened["checks"]}["status_gate_consistent"] == "FAIL"


def test_gate_stays_closed_and_has_no_bypass():
    with (PROJECT_ROOT / "config" / "prospective.toml").open("rb") as fh:
        assert tomllib.load(fh)["status"] == {"backfill": "INCOMPLETE", "prospective_predictions": "DISABLED"}
    text = (PROJECT_ROOT / "config" / "operations.toml").read_text().lower()
    assert "enable" not in text.replace("enabled", "") and "bypass" not in text
    cfg = load_operations_config()
    assert cfg.expected_frozen_hash == cfg.prospective.expected_frozen_hash == "bc4e5c870c6e21b9"
    assert cfg.expected_configuration_hash == cfg.prospective.modeling.configuration_hash() == "5e72010100065889"
    src = Path(readiness.__file__).read_text()
    assert "prospective.toml" in src and "write_text" not in src  # the evaluator never writes configuration


# ------------------------------------------------------------------------------------------------- end to end


@pytest.fixture(scope="module")
def ops_world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("ops")
    (tmp / "archive").mkdir()
    pcfg, res = build_world(tmp, FakeArchive(tmp / "archive"))
    return tmp, pcfg


def _ops_cfg(tmp, pcfg, name):
    ws = tmp / name
    return replace(load_operations_config(prospective=pcfg), workspace_dir=ws, reports_dir=ws / "reports",
                   first_week=date(2020, 5, 25), archive_last_week=date(2020, 5, 18), calendar_reference_season=2019,
                   overlap_seasons=(2020,), min_matched_observations=1,
                   tml={**load_operations_config().tml, "commit": "test", "files": ["2026.csv", tml.PLAYER_FILE]})  # fmt: skip


def _store_tml(cfg, files):
    d = cfg.workspace_dir / "tml" / "test"
    d.mkdir(parents=True, exist_ok=True)
    man = {"repository": "synthetic", "commit": "test", "files": {}}
    for n, b in files.items():
        (d / n).write_bytes(b)
        man["files"][n] = {"sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b), "url": "synthetic",
                           "retrieved_at": "2020-07-01T00:00:00.000000Z"}  # fmt: skip
    (d / "manifest.json").write_text(json.dumps(man))


GAP_ROWS = [_row("2020-x1", "1", "R32", "TA", "TB", date_="20200601"), _row("2020-x1", "2", "R32", "TC", "TD",
            date_="20200601"), _row("2020-x1", "3", "QF", "TA", "TD", date_="20200601", score="6-3 2-1 RET")]  # fmt: skip
TML_PLAYERS = [("TA", "Alice Archer", "", "USA"), ("TB", "Bob Baker", "", "USA"), ("TC", "Carl Cole", "", "USA"),
               ("TD", "Dan Dorn", "", "USA")]  # fmt: skip


def test_operations_pipeline_end_to_end(ops_world):
    from tennis_data.operations import pipeline, report

    tmp, pcfg = ops_world
    cfg = _ops_cfg(tmp, pcfg, "e2e")
    _store_tml(cfg, _tml_files(GAP_ROWS, TML_PLAYERS))
    # without reviewed identities nothing is accepted
    r0 = pipeline.run(cfg, date(2020, 7, 1), probe=False, fetch=False)["results"]
    assert r0["backfill"]["accepted_records"] == 0 and r0["state"]["match_state_unchanged"] is not False
    assert r0["readiness"]["verdict"] == "NOT_READY"
    # a reviewer approves the mappings (overrides); the same snapshot now backfills, results only
    pcfg.overrides_path.write_text("".join(
        f'[[mapping]]\nprovider = "tml_github"\nprovider_player_id = "{c}"\ncanonical_player_id = {cid}\n'
        f'reviewer = "test"\nnote = "synthetic"\n\n' for c, cid in (("TA", 100), ("TB", 200), ("TC", 300), ("TD", 400))))  # fmt: skip
    try:
        out = pipeline.run(cfg, date(2020, 7, 1), probe=False, fetch=False)
        again = pipeline.run(cfg, date(2020, 7, 1), probe=False, fetch=False)
    finally:
        pcfg.overrides_path.write_text("")
    res = out["results"]
    assert out["results_sha256"] == again["results_sha256"]  # deterministic for a fixed snapshot and as-of date
    b = res["backfill"]
    assert b["accepted_records"] == 3 and b["phase1_validated"] == 3 and b["state_eligible_updates"] >= 1
    assert b["with_complete_stats"] == 0 and b["stats_eligible"] == 0  # tml_unverified statistics never used
    assert res["state"]["deterministic_rerun"] and not res["state"]["match_state_unchanged"]
    wk = {w["week"]: w for w in res["coverage"]["weeks"]}
    assert wk["2020-06-01"]["validated"] == 3 and wk["2020-06-01"]["status"] in ("PARTIAL", "OBSERVED_UNVERIFIED")
    assert res["readiness"]["verdict"] == "NOT_READY"  # schedules, actual starts, TSA, statistics still missing
    paths = report.write_all(res, out["meta"], cfg)
    assert len(paths) == 6 and all(p.read_text().startswith("# ") for p in paths)


def test_late_tml_correction_is_quarantined_not_silently_applied(ops_world):
    from tennis_data.operations import pipeline
    from tennis_data.prospective.backfill import ingest_file
    from tennis_data.prospective.providers.fixture import FORMAT

    tmp, pcfg = ops_world
    cfg = _ops_cfg(tmp, pcfg, "corr")
    _store_tml(cfg, _tml_files(GAP_ROWS, TML_PLAYERS))
    pipeline.run(cfg, date(2020, 7, 1), probe=False, fetch=False)
    # a later TML file reissues match 1 with another score under the same id and version
    fixed = [dict(GAP_ROWS[0], score="7-5 6-4")] + GAP_ROWS[1:]
    doc, _ = tml.convert(_tml_files(fixed, TML_PLAYERS), cfg.tml["level_map"], "2020-07-02T00:00:00+00:00")
    p = cfg.workspace_dir / "later.json"
    p.write_text(json.dumps(doc))
    ingest_file(cfg.workspace_prospective(), p, FORMAT)
    res = pipeline.run(cfg, date(2020, 7, 1), probe=False, fetch=False)["results"]
    q = {x["reason"]: x["len"] for x in res["backfill"]["reconciliation_quarantine"]}
    assert q.get("version_reissued") == 2  # both contents quarantined; no source silently preferred


# ------------------------------------------------------------------------------------------- owner exports


def test_per_match_dates_become_event_starts_and_gap_filter():
    rows = [_row("t1", "1", "R32", "A1", "B2", date_="20260524"), _row("t1", "2", "F", "A1", "C3", date_="20260607"),
            _row("t2", "1", "R32", "A1", "B2", date_="20260608"), _row("t2", "2", "F", "B2", "C3", date_="20260614")]  # fmt: skip
    players = [
        ("A1", "Ann Ash", "19990101", "USA"),
        ("B2", "Bo Bay", "19990202", "USA"),
        ("C3", "Cy Cox", "19990303", "USA"),
    ]
    files = _tml_files(rows, players, player_encoding="utf-8")
    doc, log = tml.convert(files, {"250": "A"}, "2026-06-20T00:00:00+00:00", provider="owner_x")
    assert doc["provider"] == "owner_x" and sorted(log["events_with_per_match_dates"]) == ["2026.csv:t1", "2026.csv:t2"]
    assert {t["id"]: t["start_date"] for t in doc["tournaments"]} == {"t1": "2026-05-24", "t2": "2026-06-08"}
    gap, glog = tml.convert(files, {"250": "A"}, "2026-06-20T00:00:00+00:00", min_event_start="2026-05-26")
    assert [t["id"] for t in gap["tournaments"]] == ["t2"] and len(gap["matches"]) == 2
    assert glog["excluded_before_min_event_start"] == 2  # an event starting in the archive period is excluded whole


def test_export_registration_records_provenance_and_never_overwrites(tmp_path):
    from tennis_data.operations import exports

    cfg = _cfg_ops(workspace_dir=tmp_path / "ops")
    files = _tml_files(
        [_row("t1", "1", "R32", "A1", "B2")], [("A1", "Ann Ash", "", "USA"), ("B2", "Bo Bay", "", "USA")]
    )
    paths = {}
    for n, b in files.items():
        paths[n] = tmp_path / f"0a1b2c3d-{n}"
        paths[n].write_bytes(b)
    (tmp_path / "ch.csv").write_text("x\n")
    man = exports.register(cfg, "lab", [(paths["2026.csv"], "tml_matches"), (paths[tml.PLAYER_FILE], "tml_players"),
                                         (tmp_path / "ch.csv", "excluded")], provider="owner_x", origin="owner upload",
                           acquired_on="2026-10-26")  # fmt: skip
    assert man["licence_status"] == "not verified" and {f["stored_as"] for f in man["files"]} == {
        "2026.csv",
        tml.PLAYER_FILE,
        "ch.csv",
    }  # upload prefix removed
    ecfg = exports.export_config(cfg, "lab", gap_only=True)
    assert ecfg.tml["files"] == ["2026.csv", tml.PLAYER_FILE] and ecfg.tml["provider"] == "owner_x"
    assert ecfg.tml["min_event_start"] == "2026-05-26" and ecfg.workspace_dir.name == "workspace_gap"
    paths["2026.csv"].write_bytes(files["2026.csv"] + b"\n")
    with pytest.raises(FileExistsError):
        exports.register(cfg, "lab", [(paths["2026.csv"], "tml_matches")], provider="owner_x", origin="o",
                         acquired_on="2026-10-26")  # fmt: skip
    with pytest.raises(ValueError):
        exports.register(cfg, "lab", [(tmp_path / "ch.csv", "import_everything")], provider="x", origin="o",
                         acquired_on="2026-10-26")  # fmt: skip


def test_crosscheck_corroboration_and_completeness():
    from datetime import timedelta

    from tennis_data.operations import crosscheck

    d0 = date(2026, 6, 1)

    def rec(w, lo, d=d0, **kw):
        return {"date": d, "tid": "t", "tournament": "T", "level": "A", "round": "R32", "draw_size": 4,
                "w": frozenset(w.split()), "l": frozenset(lo.split()), "score": "6-4 6-4", "status": ""} | kw  # fmt: skip

    tml_rows = [rec("ann ash", "bo bay"), rec("cy cox", "di dee"), rec("ann ash", "cy cox", round="F"),
                rec("eve eng", "fay fox", d=d0 - timedelta(days=30), tid="early")]  # before the window  # fmt: skip
    other = [rec("ash", "bay", d=d0 + timedelta(days=1)), rec("dee", "cox"), rec("zed zulu", "yan yu")]
    r = crosscheck.corroborate(tml_rows, other, d0)
    assert r["accounting"] == {"tml_records": 3, "matched": 2, "winner_agrees": 1, "winner_differs": 1,
                               "no_counterpart": 1}  # fmt: skip
    assert r["unmatched_source_rows_by_tournament"] == {"T": 1}
    ev = crosscheck.completeness(tml_rows, d0)
    assert ev[0]["expected_matches"] == 3 and ev[0]["matches"] == 3 and ev[0]["status"] == "complete"
    assert crosscheck.completeness(tml_rows[:2], d0)[0]["status"] == "INCOMPLETE"


def test_identity_proposal_file_is_pending_and_excludes_contradicted(tmp_path):
    from tennis_data.operations.backfill_plan import write_proposal

    prop = {"proposed": [
        {"provider": "p", "provider_player_id": "A", "canonical_player_id": 1, "provider_name": "Ann", "archive_name": "Ann",
         "birth_date": "1999-01-01", "evidence_class": "x", "overlap_matches_confirming": 3, "overlap_matches_contradicting": 0},
        {"provider": "p", "provider_player_id": "B", "canonical_player_id": 2, "provider_name": "Bo", "archive_name": "Bob",
         "birth_date": "1999-01-02", "evidence_class": "x", "overlap_matches_confirming": 1, "overlap_matches_contradicting": 1}]}  # fmt: skip
    path = tmp_path / "proposed.toml"
    write_proposal(path, prop)
    d = tomllib.loads(path.read_text())
    assert [m["provider_player_id"] for m in d["mapping"]] == ["A"] and d["mapping"][0]["reviewer"] == "PENDING"
    assert "NOT APPLIED" in path.read_text()


def test_simulation_never_touches_production_state(ops_world):
    from tennis_data.operations import backfill_plan, pipeline

    tmp, pcfg = ops_world
    cfg = _ops_cfg(tmp, pcfg, "sim")
    _store_tml(cfg, _tml_files(GAP_ROWS, TML_PLAYERS))
    dry = pipeline.run(cfg, date(2020, 7, 1), probe=False, fetch=False)["results"]
    assert dry["backfill"]["accepted_records"] == 0  # nothing approved
    before_overrides = pcfg.overrides_path.read_text()
    prod = pcfg.prospective_dir
    before = sorted(p.relative_to(prod).as_posix() for p in prod.rglob("*")) if prod.exists() else []
    proposal = cfg.workspace_dir / "proposed.toml"
    proposal.write_text("".join(f'[[mapping]]\nprovider = "tml_github"\nprovider_player_id = "{c}"\ncanonical_player_id '
                                f'= {cid}\nreviewer = "PENDING"\nnote = "t"\n\n' for c, cid in (("TA", 100), ("TB", 200),
                                                                                                 ("TC", 300), ("TD", 400))))  # fmt: skip
    sim = backfill_plan.simulate(cfg, proposal, cfg.workspace_dir / "sandbox")
    assert sim["gap_validated"] == 3 and sim["deterministic_rerun"] and sim["gap_stats_eligible"] == 0
    assert "SIMULATION" in sim["label"] and pcfg.overrides_path.read_text() == before_overrides
    after = sorted(p.relative_to(prod).as_posix() for p in prod.rglob("*")) if prod.exists() else []
    assert after == before


def test_chronology_audit_match_dates_and_release_lags():
    from datetime import date

    from tennis_data.operations import chronology_audit as ca

    head = "tourney_id,tourney_date,match_num,round,winner_id,loser_id\n"
    files = {"ATP_Database.csv": b"id,player\n",
             "2026.csv": (head + "T1,20260610,1,R32,A,B\nT1,20260614,,F,C,A\nT1,20260614,2,R32,,B\n").encode()}  # fmt: skip
    dates = ca.tml_match_dates(files)
    # keys match the converter's (match_num, else the derived round/pair key); a row without a player is skipped
    assert dates == {"T1:1": date(2026, 6, 10), "T1:F:A-C": date(2026, 6, 14)}
    lags = ca._lags([date(2026, 6, 12), date(2026, 6, 13), date(2026, 6, 14)], [date(2026, 6, 13)] * 3)
    assert lags == {"matches": 3, "played_on_or_after_release": 2, "by_days_after": {"0": 1, "1": 1}}
