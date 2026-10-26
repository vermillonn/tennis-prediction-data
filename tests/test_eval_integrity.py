"""Phase 4C: snapshot integrity, checkpoint consistency, anchors, receipts and trust levels."""

from __future__ import annotations

import hashlib
import json
from datetime import timedelta
from pathlib import Path

import pytest

from evaluation_helpers import OPENSSL, T0, FakeGitHub, LocalTSA, World, scored_match
from tennis_data.evaluation_prospective import anchoring
from tennis_data.evaluation_prospective.integrity import check_checkpoint, verify_snapshot
from tennis_data.evaluation_prospective.outcomes import empty_outcomes
from tennis_data.evaluation_prospective.pipeline import evaluate
from tennis_data.evaluation_prospective.safeio import UntrustedInputError, resolve_under, strict_json, write_new
from tennis_data.evaluation_prospective.snapshot import load_export, snapshot_from_ledger
from tennis_data.ledger.hashing import canonical

needs_openssl = pytest.mark.skipif(OPENSSL is None, reason="openssl binary not installed")


def _ledger(tmp_path, n=3, ledger_id="ledger-eval-test"):
    wd = World(tmp_path, ledger_id)
    for i in range(n):
        scored_match(wd, f"m-{i}", T0 + timedelta(days=i + 1))
    wd.close()
    return wd, snapshot_from_ledger(wd.cfg.ledger.ledger_path, wd.cfg.snapshot_dir)


def _rewrite(path, fn):
    lines = path.read_bytes().splitlines()
    objs = [json.loads(x) for x in lines]
    objs = fn(objs)
    out = tmp = path.with_name("tampered.jsonl")
    tmp.write_bytes(b"\n".join(canonical(o) for o in objs) + b"\n")
    return out


def test_valid_export_verifies_and_is_content_addressed(tmp_path):
    wd, snap = _ledger(tmp_path)
    rep = verify_snapshot(snap)
    assert rep["ok"] and rep["events"] == 9 and rep["head_hash"] == snap.head_hash
    assert snap.path.parent.name == snap.sha256[:16]
    again = snapshot_from_ledger(wd.cfg.ledger.ledger_path, wd.cfg.snapshot_dir)  # idempotent, same address
    assert again.sha256 == snap.sha256 and again.path == snap.path


def test_tampered_event_fails_and_nothing_is_scored(tmp_path):
    wd, snap = _ledger(tmp_path)

    def bump(objs):
        objs[1]["payload"]["prediction"]["probability_a"] = 0.9
        objs[1]["payload"]["prediction"]["probability_b"] = 0.1
        return objs

    bad = load_export(_rewrite(snap.path, bump))
    rep = verify_snapshot(bad)
    assert not rep["ok"] and {"event_hash_mismatch", "denormalized_column_mismatch"} <= {
        i["code"] for i in rep["issues"]
    }
    res = evaluate(bad, empty_outcomes(), wd.cfg.protocol)
    assert {r["category"] for r in res["classification"]} == {"INVALID_CHAIN"}
    assert res["scores"]["operational"]["n"] == 0 and res["anchoring"]["snapshot_head_evidence"] == "INVALID"


def test_broken_chain_reordered_and_non_canonical_exports(tmp_path):
    _, snap = _ledger(tmp_path)

    def relink(objs):
        objs[3]["previous_hash"] = "0" * 64
        return objs

    assert "previous_hash_mismatch" in {i["code"] for i in verify_snapshot(load_export(_rewrite(snap.path, relink)))[
        "issues"]}  # fmt: skip

    def swap(objs):
        objs[2], objs[3] = objs[3], objs[2]
        return objs

    codes = {i["code"] for i in verify_snapshot(load_export(_rewrite(snap.path, swap)))["issues"]}
    assert "sequence_gap_or_reorder" in codes
    lines = snap.path.read_bytes().splitlines()
    spaced = tmp_path / "spaced.jsonl"
    spaced.write_bytes(b"\n".join([lines[0], json.dumps(json.loads(lines[1]), indent=1).replace("\n", "").encode(),
                                   *lines[2:]]) + b"\n")  # fmt: skip
    with pytest.raises(UntrustedInputError, match="canonical"):
        load_export(spaced)
    dup = tmp_path / "dup.jsonl"
    dup.write_bytes(lines[0][:-1] + b',"format":"x"}\n')
    with pytest.raises(UntrustedInputError, match="duplicate"):
        load_export(dup)


def test_truncated_export_detected_only_against_a_trusted_checkpoint(tmp_path):
    wd, snap = _ledger(tmp_path)
    cp = anchoring.checkpoint_from_snapshot(snap)
    trunc = load_export(_rewrite(snap.path, lambda objs: objs[:-2]))
    assert verify_snapshot(trunc)["ok"]  # a prefix of a valid chain is itself valid
    probs = check_checkpoint(cp, trunc)
    assert any("truncated" in p for p in probs)
    res = evaluate(trunc, empty_outcomes(), wd.cfg.protocol, trusted=[("cp.json", cp)])
    assert not res["integrity"]["evidence_ok"]
    assert {r["category"] for r in res["classification"]} == {"INVALID_EVIDENCE"}


def test_full_rewrite_detected_by_retained_checkpoint(tmp_path):
    wd, snap = _ledger(tmp_path / "a")
    cp = anchoring.checkpoint_from_snapshot(snap)
    # a different history under the same ledger id, internally consistent
    _, forged = _ledger(tmp_path / "b", n=3)
    assert verify_snapshot(forged)["ok"] and forged.ledger_id == snap.ledger_id
    other = World(tmp_path / "c")
    scored_match(other, "m-0", T0 + timedelta(days=1), prob=0.9, elo=0.5)
    other.close()
    rewritten = snapshot_from_ledger(other.cfg.ledger.ledger_path, other.cfg.snapshot_dir)
    assert verify_snapshot(rewritten)["ok"]
    assert any("rewritten" in p or "truncated" in p for p in check_checkpoint(cp, rewritten))
    res = evaluate(rewritten, empty_outcomes(), other.cfg.protocol, trusted=[("cp.json", cp)])
    assert res["anchoring"]["snapshot_head_evidence"] == "INVALID"


def test_fixture_anchor_never_counts_as_external(tmp_path):
    wd, snap = _ledger(tmp_path)
    cp = anchoring.checkpoint_from_snapshot(snap)
    fix = anchoring.LocalFixturePublisher(tmp_path / "fix")
    rec, path = anchoring.publish(cp, fix, tmp_path / "ev", "tester", now=T0)
    rec["receipt"]["trust_level"] = "INDEPENDENTLY_TIMESTAMPED"  # a caller-supplied label is ignored
    v = anchoring.verify_anchor(rec, snap, evidence_dir=tmp_path / "ev", fixture_root=tmp_path / "fix")
    assert v["level"] == "LOCAL_ONLY" and v["receipt_status"] == "fixture_matches"
    _, path2 = anchoring.publish(cp, fix, tmp_path / "ev", "tester", now=T0 + timedelta(hours=1))
    assert path2 != path and path.exists()  # a second publication is a new record beside the first
    (tmp_path / "fix" / rec["receipt"]["path"]).write_bytes(b"{}\n")
    v = anchoring.verify_anchor(rec, snap, evidence_dir=tmp_path / "ev", fixture_root=tmp_path / "fix")
    assert v["level"] == "INVALID"
    v = anchoring.verify_anchor(rec, snap, evidence_dir=tmp_path / "ev", fixture_root=None)
    assert v["level"] == "LOCAL_ONLY" and v["receipt_status"] == "fixture_not_found"
    with pytest.raises(FileExistsError):  # records are never overwritten
        write_new(path, b"x")
    with pytest.raises(anchoring.PublicationError, match="different content"):  # nor fixture publications
        anchoring.publish(cp, fix, tmp_path / "ev", "tester", now=T0 + timedelta(hours=1))


def test_anchor_with_wrong_head_is_invalid(tmp_path):
    wd, snap = _ledger(tmp_path)
    cp = anchoring.checkpoint_from_snapshot(snap)
    body = {**cp, "head_hash": "f" * 64}
    from tennis_data.ledger.checkpoint import checkpoint_hash

    body["checkpoint_hash"] = checkpoint_hash(body)
    rec = anchoring.make_record(body, "local_fixture", "x", {"type": "local_fixture", "path": "a.json"}, T0)
    v = anchoring.verify_anchor(rec, snap, evidence_dir=None)
    assert v["level"] == "INVALID" and any("rewritten" in p for p in v["problems"])
    rec2 = anchoring.make_record(cp, "local_fixture", "x", {}, T0)
    rec2["chain_head_hash"] = "e" * 64
    assert anchoring.verify_anchor(rec2, snap, evidence_dir=None)["level"] == "INVALID"


def test_github_publication_and_verification(tmp_path, monkeypatch):
    wd, snap = _ledger(tmp_path)
    db = wd.cfg.ledger.ledger_path
    before = hashlib.sha256(db.read_bytes()).hexdigest()
    cp = anchoring.checkpoint_from_snapshot(snap)
    gh = FakeGitHub()
    pub = anchoring.GitHubContentsPublisher("owner/anchors", "ledger-checkpoints", "checkpoints", "TEST_GH_TOKEN",
                                            transport=gh.transport)  # fmt: skip
    with pytest.raises(anchoring.PublicationError, match="TEST_GH_TOKEN"):
        anchoring.publish(cp, pub, tmp_path / "ev", "ci", now=T0)
    assert list((tmp_path / "ev" / "failures").iterdir()) and not (tmp_path / "ev" / "records").exists()
    monkeypatch.setenv("TEST_GH_TOKEN", "secret-token-value")
    rec, path = anchoring.publish(cp, pub, tmp_path / "ev", "ci", now=T0)
    assert rec["receipt"]["type"] == "github_contents" and "secret-token-value" not in path.read_text()
    off = anchoring.verify_anchor(rec, snap, evidence_dir=tmp_path / "ev")
    assert off["level"] == "LOCAL_ONLY" and off["receipt_status"] == "not_checked_offline"
    on = anchoring.verify_anchor(rec, snap, evidence_dir=tmp_path / "ev", online=True, transport=gh.transport)
    assert on["level"] == "EXTERNALLY_PUBLISHED" and on["attested_time"] is None  # publication is not a timestamp
    gh.files.pop(("ledger-checkpoints", rec["receipt"]["path"]))  # deleted from the branch later
    on = anchoring.verify_anchor(rec, snap, evidence_dir=tmp_path / "ev", online=True, transport=gh.transport)
    assert "removed_from_branch" in on["receipt_status"]
    gh.serve_other_content = True
    assert anchoring.verify_anchor(rec, snap, evidence_dir=tmp_path / "ev", online=True,
                                   transport=gh.transport)["level"] == "INVALID"  # fmt: skip
    gh.serve_other_content, gh.down = False, True
    v = anchoring.verify_anchor(rec, snap, evidence_dir=tmp_path / "ev", online=True, transport=gh.transport)
    assert v["level"] == "LOCAL_ONLY" and v["receipt_status"].startswith("unreachable")
    assert hashlib.sha256(db.read_bytes()).hexdigest() == before  # the ledger is never touched


def test_github_refuses_to_overwrite_and_fails_closed(tmp_path, monkeypatch):
    _, snap = _ledger(tmp_path)
    monkeypatch.setenv("TEST_GH_TOKEN", "t")
    cp = anchoring.checkpoint_from_snapshot(snap)
    gh = FakeGitHub()
    pub = anchoring.GitHubContentsPublisher("o/r", "b", "c", "TEST_GH_TOKEN", transport=gh.transport)
    gh.files[("b", pub.path_for(cp))] = b"different"
    with pytest.raises(anchoring.PublicationError, match="not overwritten"):
        anchoring.publish(cp, pub, tmp_path / "ev", "ci", now=T0)
    gh2 = FakeGitHub()
    gh2.serve_other_content = True  # read-back differs from what was sent
    pub2 = anchoring.GitHubContentsPublisher("o/r", "b", "c", "TEST_GH_TOKEN", transport=gh2.transport)
    with pytest.raises(anchoring.PublicationError, match="read back"):
        anchoring.publish(cp, pub2, tmp_path / "ev2", "ci", now=T0)
    assert not (tmp_path / "ev2" / "records").exists()
    with pytest.raises(anchoring.PublicationError, match="owner/repo"):
        anchoring.GitHubContentsPublisher("", "b", "c", "X")


@needs_openssl
def test_rfc3161_timestamp_levels(tmp_path):
    wd, snap = _ledger(tmp_path)
    cp = anchoring.checkpoint_from_snapshot(snap)
    tsa, other = LocalTSA(tmp_path / "tsa"), LocalTSA(tmp_path / "tsa2", name="Other Root")
    pub = anchoring.Rfc3161Publisher("https://tsa.test/", [tsa.ca], transport=tsa.transport)
    rec, _ = anchoring.publish(cp, pub, tmp_path / "ev", "ci", now=T0)
    ok = anchoring.verify_anchor(rec, snap, evidence_dir=tmp_path / "ev", tsa_ca_files=[tsa.ca])
    assert ok["level"] == "INDEPENDENTLY_TIMESTAMPED" and ok["attested_time"].endswith("Z")
    untrusted = anchoring.verify_anchor(rec, snap, evidence_dir=tmp_path / "ev", tsa_ca_files=[other.ca])
    assert untrusted["level"] == "LOCAL_ONLY"
    assert anchoring.verify_anchor(rec, snap, evidence_dir=tmp_path / "ev")["level"] == "LOCAL_ONLY"
    tok = tmp_path / "ev" / rec["receipt"]["token_file"]
    good = tok.read_bytes()
    tok.unlink()
    missing = anchoring.verify_anchor(rec, snap, evidence_dir=tmp_path / "ev", tsa_ca_files=[tsa.ca])
    assert missing["level"] == "LOCAL_ONLY" and missing["receipt_status"].startswith("token_missing")
    # a valid token for a different checkpoint does not transfer
    other_cp = {**cp}
    other_cp.update(sequence=cp["sequence"] - 1, head_hash=snap.events[-2]["event_hash"])
    from tennis_data.ledger.checkpoint import checkpoint_hash

    other_cp["checkpoint_hash"] = checkpoint_hash(other_cp)
    tok.write_bytes(good)
    forged = anchoring.make_record(other_cp, "rfc3161", "ci", {**rec["receipt"], "imprint": other_cp["checkpoint_hash"]},
                                   T0)  # fmt: skip
    v = anchoring.verify_anchor(forged, snap, evidence_dir=tmp_path / "ev", tsa_ca_files=[tsa.ca])
    assert v["level"] == "INVALID"
    # a TSA whose token does not verify is never recorded
    bad = anchoring.Rfc3161Publisher("https://tsa.test/", [other.ca], transport=tsa.transport)
    with pytest.raises(anchoring.PublicationError, match="does not verify"):
        anchoring.publish(cp, bad, tmp_path / "ev3", "ci", now=T0)
    assert not (tmp_path / "ev3" / "records").exists()


def test_untrusted_paths_and_json():
    with pytest.raises(UntrustedInputError):
        strict_json(b'{"a": NaN}')
    with pytest.raises(UntrustedInputError):
        strict_json(b'{"a": 1, "a": 2}')
    for rel in ("../x", "/etc/passwd", "a/../../b", "", "c:/x", "a\\b"):
        with pytest.raises(UntrustedInputError):
            resolve_under(Path("/tmp"), rel)


def test_symlinked_receipt_path_rejected(tmp_path):
    (tmp_path / "root").mkdir()
    (tmp_path / "secret.json").write_text("{}")
    (tmp_path / "root" / "link").symlink_to(tmp_path / "secret.json")
    with pytest.raises(UntrustedInputError):
        resolve_under(tmp_path / "root", "link")


def test_malformed_anchor_records_and_checkpoints_fail_closed(tmp_path):
    _, snap = _ledger(tmp_path)
    cp = anchoring.checkpoint_from_snapshot(snap)
    rec = anchoring.make_record(cp, "local_fixture", "x", {"type": "local_fixture", "path": "a.json"}, T0)
    for field, value in (("publisher", "me"), ("checkpoint_sequence", [1]), ("checkpoint", "x"), ("receipt", 5)):
        v = anchoring.verify_anchor({**rec, field: value}, snap, evidence_dir=None)
        assert v["level"] in ("INVALID", "LOCAL_ONLY")
        if field != "receipt":
            assert v["level"] == "INVALID"
    assert anchoring.verify_anchor({"format": "x"}, snap, evidence_dir=None)["level"] == "INVALID"
    fix = anchoring.LocalFixturePublisher(tmp_path / "fix")
    for bad in ({**cp, "head_hash": "0" * 64}, {**cp, "extra": 1}, {"format": "x"}):
        with pytest.raises(anchoring.PublicationError):
            anchoring.publish(bad, fix, tmp_path / "ev", "x", now=T0)
    assert not (tmp_path / "fix").exists()
