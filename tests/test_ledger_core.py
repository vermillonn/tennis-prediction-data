"""Ledger core: canonical hashing, chain, append-only enforcement, tamper detection, checkpoints, export."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta, timezone

import pytest

from ledger_helpers import T0, request, result, writer
from tennis_data.ledger import storage
from tennis_data.ledger.checkpoint import checkpoint_hash, make_checkpoint, verify_checkpoint, write_checkpoint
from tennis_data.ledger.export import export, import_jsonl, read_jsonl
from tennis_data.ledger.hashing import CanonicalizationError, canonical, event_hash, genesis_hash, parse_ts, ts
from tennis_data.ledger.reader import LedgerReader
from tennis_data.ledger.verification import verify_events, verify_ledger


def _fill(w, n=6):
    out = []
    for i in range(n):
        out.append(w.commit_prediction(request(w.cfg, match_id=f"synthetic:m{i}", prob_p1=0.5 + i / 100)))
    out.append(w.record("MATCH_RESULT_RECORDED", "synthetic:m0", result(104925)))
    return out


# ------------------------------------------------------------------------------------------------ hashing


def test_canonical_serialization_rules():
    assert canonical({"b": 1, "a": [1.5, None, True, "é"]}) == '{"a":[1.5,null,true,"é"],"b":1}'.encode()
    assert canonical({"x": -0.0}) == b'{"x":0.0}'
    assert canonical({"x": 0.1 + 0.2}) == b'{"x":0.30000000000000004}'  # shortest round-trip repr
    for bad in ({"x": float("nan")}, {"x": float("inf")}, {1: 2}, {"x": (1, 2)}, {"x": datetime.now(UTC)}):
        with pytest.raises(CanonicalizationError):
            canonical(bad)


def test_timestamps_have_one_form():
    t = datetime(2026, 11, 1, 10, 30, tzinfo=timezone(timedelta(hours=1)))
    assert ts(t) == "2026-11-01T09:30:00.000000Z"
    assert parse_ts(ts(t)) == t
    with pytest.raises(CanonicalizationError):
        ts(datetime(2026, 11, 1))
    with pytest.raises(CanonicalizationError):
        parse_ts("2026-11-01T09:30:00+00:00")


def test_event_hash_is_fixed_by_the_specification():
    body = {"seq": 1, "event_id": "e-1", "payload": {"a": 1}}
    import hashlib

    prev = genesis_hash("L")
    assert prev == hashlib.sha256(b"tennis-ledger/genesis/v1\x00L").hexdigest()

    expected = hashlib.sha256(b"tennis-ledger/event/v1\x00" + canonical(body) + b"\x00" + prev.encode()).hexdigest()
    assert event_hash(body, prev) == expected
    assert event_hash(body, prev) == event_hash(dict(reversed(list(body.items()))), prev)  # key order irrelevant


# --------------------------------------------------------------------------------------------- chain/storage


def test_init_sequential_commits_and_retrieval(tmp_path):
    w = writer(tmp_path)
    rs = _fill(w)
    assert [r.seq for r in rs] == list(range(1, 8))
    assert rs[0].previous_hash == genesis_hash("ledger-test")
    assert all(b.previous_hash == a.event_hash for a, b in zip(rs, rs[1:], strict=False))
    rd = LedgerReader(w.path)
    e = rd.event(rs[2].event_id)
    assert e["event_hash"] == rs[2].event_hash and e["payload"]["prediction"]["probability_a"] in (0.52, 0.48)
    assert rd.head() == (7, rs[-1].event_hash)
    rep = verify_ledger(w.path)
    assert rep.ok and rep.head_hash == rs[-1].event_hash and rep.events == 7
    assert w.init("ledger-test")["ledger_id"] == "ledger-test"  # re-init is a no-op
    with pytest.raises(Exception, match="already initialized"):
        w.init("other")


def test_identical_inputs_give_identical_hashes(tmp_path):
    a, b = writer(tmp_path / "a"), writer(tmp_path / "b")
    ra, rb = _fill(a), _fill(b)
    assert [r.event_hash for r in ra] == [r.event_hash for r in rb]
    assert verify_ledger(a.path).head_hash == verify_ledger(b.path).head_hash


@pytest.mark.parametrize("sql", [
    "UPDATE events SET match_id = 'x' WHERE seq = 1",
    "DELETE FROM events WHERE seq = 2",
    "UPDATE ledger_meta SET ledger_id = 'x'",
    "DELETE FROM ledger_meta",
])  # fmt: skip
def test_update_and_delete_are_blocked(tmp_path, sql):
    w = writer(tmp_path)
    _fill(w, 3)
    with pytest.raises(sqlite3.DatabaseError, match="immutable|append-only"):
        w.con.execute(sql)
    assert verify_ledger(w.path).ok


def test_out_of_chain_inserts_are_blocked(tmp_path):
    w = writer(tmp_path)
    rs = _fill(w, 2)
    row = dict(w.con.execute("SELECT * FROM events WHERE seq = 3").fetchone())
    cols = ", ".join(storage.COLUMNS)
    q = f"INSERT INTO events ({cols}) VALUES ({', '.join('?' * len(storage.COLUMNS))})"
    for change in ({"seq": 9}, {"seq": 3}, {"previous_hash": rs[0].event_hash}):
        r = {**row, **change, "event_id": "x" * 10, "event_hash": "f" * 64}
        if change.get("seq") == 3:
            r["seq"] = 4
            r["previous_hash"] = "0" * 64
        with pytest.raises(sqlite3.DatabaseError, match="sequence|previous_hash"):
            w.con.execute(q, tuple(r[c] for c in storage.COLUMNS))
    bad_prob = {**row, "seq": 4, "previous_hash": rs[-1].event_hash, "event_id": "y" * 10, "event_hash": "e" * 64,
                "probability_a": 1.5, "idempotency_key": "a" * 64}  # fmt: skip
    with pytest.raises(sqlite3.IntegrityError):
        w.con.execute(q, tuple(bad_prob[c] for c in storage.COLUMNS))


# ----------------------------------------------------------------------------------------------- tampering


def _attacker(path):
    """A privileged user who drops the protective triggers (outside the trusted application)."""
    con = sqlite3.connect(path, isolation_level=None)
    for t in ("events_no_update", "events_no_delete", "events_chain"):
        con.execute(f"DROP TRIGGER {t}")
    return con


def _issues(path):
    return {i["code"] for i in verify_ledger(path).issues}


def test_payload_tampering_is_detected(tmp_path):
    w = writer(tmp_path)
    _fill(w)
    con = _attacker(w.path)
    p = json.loads(con.execute("SELECT payload FROM events WHERE seq = 2").fetchone()[0])
    p["prediction"]["probability_a"], p["prediction"]["probability_b"] = 0.9, 0.1
    con.execute("UPDATE events SET payload = ? WHERE seq = 2", (canonical(p).decode(),))
    assert {"event_hash_mismatch", "denormalized_column_mismatch"} <= _issues(w.path)


def test_deletion_and_reordering_are_detected(tmp_path):
    w = writer(tmp_path / "del")
    _fill(w)
    _attacker(w.path).execute("DELETE FROM events WHERE seq = 3")
    assert {"sequence_gap_or_reorder", "previous_hash_mismatch"} <= _issues(w.path)
    w2 = writer(tmp_path / "ord")
    _fill(w2)
    meta = storage.meta(w2.con)
    ev = storage.all_events(w2.con)
    ev[2], ev[3] = ev[3], ev[2]
    codes = {i["code"] for i in verify_events(meta, ev).issues}
    assert {"sequence_gap_or_reorder", "previous_hash_mismatch"} <= codes


def test_broken_predecessor_invalid_hash_and_duplicates(tmp_path):
    w = writer(tmp_path)
    _fill(w)
    meta, ev = storage.meta(w.con), storage.all_events(w.con)
    e1 = [dict(e) for e in ev]
    e1[4]["previous_hash"] = "0" * 64
    assert "previous_hash_mismatch" in {i["code"] for i in verify_events(meta, e1).issues}
    e2 = [dict(e) for e in ev]
    e2[4]["event_hash"] = "a" * 64
    assert "event_hash_mismatch" in {i["code"] for i in verify_events(meta, e2).issues}
    e3 = [dict(e) for e in ev]
    e3[5]["event_id"] = e3[4]["event_id"]
    assert "duplicate_event_id" in {i["code"] for i in verify_events(meta, e3).issues}
    e4 = [dict(e) for e in ev]
    e4[5] = {**e4[4], "seq": 6}
    codes = {i["code"] for i in verify_events(meta, e4).issues}
    assert {"duplicate_event_id", "duplicate_idempotency_key"} <= codes


def test_full_rewrite_is_only_detected_against_a_trusted_head(tmp_path):
    """An attacker who rewrites and re-hashes the whole chain produces a self-consistent ledger: only a previously
    recorded checkpoint exposes it."""
    w = writer(tmp_path / "orig")
    _fill(w)
    cp = make_checkpoint(w.con)
    forged = writer(tmp_path / "forged")  # same ledger id, different history (re-hashed from scratch)
    for i in range(6):
        forged.commit_prediction(request(forged.cfg, match_id=f"synthetic:m{i}", prob_p1=0.4))
    forged.record("MATCH_RESULT_RECORDED", "synthetic:m0", result(104925))
    assert verify_ledger(forged.path).ok  # internally consistent
    v = verify_checkpoint(forged.path, cp)
    assert not v["ok"] and any("history rewritten" in p for p in v["problems"])


# ---------------------------------------------------------------------------------------------- checkpoints


def test_checkpoint_export_and_verification(tmp_path):
    w = writer(tmp_path)
    _fill(w, 3)
    cp1 = make_checkpoint(w.con)
    path = write_checkpoint(cp1, tmp_path / "cp1.json", exported_at=T0)
    assert json.loads(path.read_text()) == cp1 and "exported_at" not in cp1
    assert verify_checkpoint(w.path, cp1)["ok"]
    w.record_checkpoint_event({"checkpoint_hash": cp1["checkpoint_hash"], "covered_seq": cp1["sequence"],
                               "covered_head_hash": cp1["head_hash"], "previous_checkpoint_hash": None})  # fmt: skip
    w.commit_prediction(request(w.cfg, match_id="synthetic:late"))
    cp2 = make_checkpoint(w.con, previous=cp1)
    assert cp2["previous_checkpoint_hash"] == cp1["checkpoint_hash"]
    assert verify_checkpoint(w.path, cp1)["ok"] and verify_checkpoint(w.path, cp2)["ok"]  # history extended, intact
    altered = {**cp1, "head_hash": "0" * 64}
    assert not verify_checkpoint(w.path, altered)["ok"]
    beyond = {**cp2, "sequence": 99}
    beyond["checkpoint_hash"] = checkpoint_hash(beyond)
    assert any("truncated" in p for p in verify_checkpoint(w.path, beyond)["problems"])
    assert make_checkpoint(w.con) == make_checkpoint(w.con)  # deterministic


# ------------------------------------------------------------------------------------------------- export


def test_export_import_preserves_everything(tmp_path):
    w = writer(tmp_path)
    rs = _fill(w)
    w.record("MATCH_CANCELLED", "synthetic:m1", {"reason": "rain", "source": {"provider": "fixture"}})
    w.record("PREDICTION_VOIDED", "synthetic:m1",
             {"reason_code": "match_cancelled", "evidence_event_ids": [LedgerReader(w.path).events_for_match(
                 "synthetic:m1")[-1]["event_id"]], "actor": "test", "note": ""},
             ref_event_id=rs[1].event_id)  # fmt: skip
    info = export(w.path, tmp_path / "exp")
    assert info["events"] == 9
    meta, events = read_jsonl(tmp_path / "exp" / "events.jsonl")
    assert verify_events(meta, events).ok
    assert {e["event_type"] for e in events} >= {"PREDICTION_VOIDED", "MATCH_CANCELLED", "MATCH_RESULT_RECORDED"}
    rep = import_jsonl(tmp_path / "exp" / "events.jsonl", tmp_path / "copy.sqlite")
    assert rep["ok"] and rep["head_hash"] == info["head_hash"]
    # a tampered export does not import
    lines = (tmp_path / "exp" / "events.jsonl").read_text().splitlines()
    lines[3] = lines[3].replace("R32", "R16")  # edit one event's round
    (tmp_path / "bad.jsonl").write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="does not verify"):
        import_jsonl(tmp_path / "bad.jsonl", tmp_path / "copy2.sqlite")
    import duckdb

    con = duckdb.connect()
    n = con.execute(f"SELECT count(*) FROM read_parquet('{tmp_path / 'exp' / 'events.parquet'}')").fetchone()[0]
    assert n == 9
