"""Controlled backfill: content-addressed baseline, approval rules and the import filter."""

from __future__ import annotations

import tomllib
from datetime import date

import pytest

from tennis_data.operations import controlled_backfill as cb


def test_snapshot_restore_verify_and_immutability(tmp_path):
    src = tmp_path / "src"
    (src / "a").mkdir(parents=True)
    (src / "a" / "x.bin").write_bytes(b"one")
    (src / "y.json").write_text("{}")
    store = tmp_path / "store"
    man = cb.snapshot_dir(src, store, "base", {"state_snapshot_id": "s"})
    assert cb.verify_snapshot(store, man) == []
    cb.restore(store, man, tmp_path / "out")
    assert cb.tree(tmp_path / "out") == cb.tree(src)
    with pytest.raises(FileExistsError):  # never restores over existing content
        cb.restore(store, man, tmp_path / "out")
    assert cb.snapshot_dir(src, store, "base", {}) == man  # idempotent for identical content
    (src / "y.json").write_text('{"changed": 1}')
    with pytest.raises(FileExistsError):  # a manifest is never replaced by different content
        cb.snapshot_dir(src, store, "base", {})
    obj = store / "objects" / man["files"]["a/x.bin"]["sha256"][:2] / man["files"]["a/x.bin"]["sha256"]
    obj.write_bytes(b"tampered")
    assert cb.verify_snapshot(store, man) == ["a/x.bin"]


def _m(pid, cid, cls="overlap_corroborated", method="name_dob", blocking=(), confirm=2, contra=0):
    return {"provider_player_id": pid, "canonical_player_id": cid, "evidence_class": cls, "method": method,
            "blocking": list(blocking), "overlap_matches_confirming": confirm, "overlap_matches_contradicting": contra}  # fmt: skip


def _case():
    dob = date(2000, 1, 1)
    archive = {1: {"name": "Ann Example", "birth_date": dob}, 2: {"name": "Bea Sample", "birth_date": dob},
               3: {"name": "Cy Test", "birth_date": dob}, 4: {"name": "Cy Test", "birth_date": None},
               5: {"name": "Di Other", "birth_date": dob}, 6: {"name": "Ed Young", "birth_date": date(2015, 1, 1)},
               7: {"name": "Fay Dup", "birth_date": dob}, 8: {"name": "Gus Name", "birth_date": dob},
               9: {"name": "Hal Amb", "birth_date": dob}, 10: {"name": "Ivy Tam", "birth_date": None}}  # fmt: skip
    provider = {"A": {"name": "Ann Example", "birth_date": dob}, "B": {"name": "Bea Sample", "birth_date": dob},
                "C": {"name": "Cy Test", "birth_date": dob}, "D": {"name": "Di Otherr", "birth_date": dob},
                "E": {"name": "Ed Young", "birth_date": date(2015, 1, 1)}, "F": {"name": "Fay Dup", "birth_date": dob},
                "F2": {"name": "Fay Dup", "birth_date": dob}, "G": {"name": "Gus Name", "birth_date": dob},
                "H": {"name": "Hal Amb", "birth_date": dob}, "T": {"name": "Ivy Tam", "birth_date": None}}  # fmt: skip
    mappings = [_m("A", 1), _m("B", 2, cls="name_and_birth_date_only", confirm=0), _m("C", 3), _m("D", 5),
                _m("E", 6), _m("F", 7), _m("F2", 7), _m("G", 8, contra=1), _m("H", 9),
                _m("T", 10, cls="name_and_birth_date_only", method="name_country", blocking=["birth_date_conflict"])]  # fmt: skip
    decisions = {m["provider_player_id"]: {"category": "high_confidence", "canonical_player_id": m["canonical_player_id"],
                                           "method": m["method"], "candidates": [m["canonical_player_id"]]}
                 for m in mappings}  # fmt: skip
    review = {"mappings": mappings, "ambiguous": [
        {"provider_player_id": "Q", "category": "ambiguous", "proposed_canonical_id": 9,
         "candidates": [{"id": 9}]},
        {"provider_player_id": "N", "category": "unmatched", "proposed_canonical_id": None, "candidates": []}]}  # fmt: skip
    first_gap = {k: date(2026, 7, 1) for k in provider}
    proposed = {m["provider_player_id"]: m["canonical_player_id"] for m in mappings}
    return review, decisions, provider, archive, first_gap, proposed


def test_select_approvals_applies_every_invariant():
    review, decisions, provider, archive, first_gap, proposed = _case()
    sel = cb.select_approvals(review, decisions, provider, archive, {}, first_gap, {("tml_export", "Z"): 2}, proposed)
    assert [e["provider_player_id"] for e in sel["approved"]] == ["A"]
    failed = {e["provider_player_id"]: e["failed"] for e in sel["rejected"]}
    assert failed["B"] == ["one_to_one"]  # canonical 2 already held by another provider id in the overrides
    assert failed["C"] == ["no_indistinguishable_namesake"]  # undated archive record with the same name
    assert failed["D"] == ["names_exact"]
    assert failed["E"] == ["chronology"]  # 11 years old at the gap match
    assert failed["F"] == failed["F2"] == ["one_to_one"]
    assert failed["G"] == ["overlap_corroboration"]
    assert failed["H"] == ["not_ambiguous_candidate"]
    pend = {p["provider_player_id"]: p["class"] for p in sel["pending"]}
    assert pend == {"Q": "ambiguous", "N": "new_player", "T": "name_and_birth_date_only"}  # name_country: pending


def test_select_approvals_requires_provenance_and_phase4a_decision():
    review, decisions, provider, archive, first_gap, proposed = _case()
    proposed["A"] = 99
    decisions["A"]["candidates"] = [1, 3]
    sel = cb.select_approvals(review, decisions, provider, archive, {}, first_gap, {}, proposed)
    a = next(e for e in sel["rejected"] if e["provider_player_id"] == "A")
    assert set(a["failed"]) == {"proposal_provenance", "phase4a_decision"}


def test_write_approvals_records_evidence_and_never_overwrites(tmp_path):
    e = {"provider_player_id": "A", "canonical_player_id": 1, "evidence_class": "overlap_corroborated",
         "provider_name": "Ann Example", "archive_name": "Ann Example", "birth_date": "2000-01-01",
         "overlap_matches_confirming": 2}  # fmt: skip
    p = tmp_path / "a.toml"
    cb.write_approvals(p, [e], "L")
    m = tomllib.loads(p.read_text())["mapping"][0]
    assert m["approval_class"] == "overlap_corroborated" and m["authorization"] == cb.AUTHORIZATION
    assert m["canonical_player_id"] == 1 and "born 2000-01-01" in m["note"]
    cb.write_approvals(p, [e], "L")  # identical: allowed
    with pytest.raises(FileExistsError):
        cb.write_approvals(p, [{**e, "canonical_player_id": 2}], "L")


def test_filter_import_keeps_only_matches_of_approved_players():
    conv = {"schema": "s", "players": [{"id": i} for i in "ABC"],
            "tournaments": [{"id": "t1"}, {"id": "t2"}],
            "matches": [{"id": "m1", "tournament_id": "t1", "round": "F", "player1_id": "A", "player2_id": "B"},
                        {"id": "m2", "tournament_id": "t2", "round": "F", "player1_id": "A", "player2_id": "C"}]}  # fmt: skip
    doc, held = cb.filter_import(conv, {"A", "B"})
    assert [m["id"] for m in doc["matches"]] == ["m1"] and [p["id"] for p in doc["players"]] == ["A", "B"]
    assert [t["id"] for t in doc["tournaments"]] == ["t1"]
    assert held == [{"provider_match_id": "m2", "tournament_id": "t2", "round": "F", "reason": "identity_pending",
                     "pending_players": ["C"]}]  # fmt: skip
