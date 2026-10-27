"""Phase 4C regression tests: fail-closed trust gaps, selection invariance, and the repository's evidence state."""

from __future__ import annotations

import tomllib
from datetime import timedelta

import pytest

from evaluation_helpers import OPENSSL, T0, FakeGitHub, LocalTSA, World, outcome_record, scored_match, write_outcomes
from tennis_data.config import PROJECT_ROOT
from tennis_data.evaluation_prospective import anchoring, timestamping
from tennis_data.evaluation_prospective.outcomes import empty_outcomes, load_outcomes
from tennis_data.evaluation_prospective.pipeline import evaluate
from tennis_data.evaluation_prospective.snapshot import snapshot_from_ledger
from tennis_data.ledger.hashing import ts

P1, P2 = 104925, 207989
needs_openssl = pytest.mark.skipif(OPENSSL is None, reason="openssl binary not installed")


def _snap(tmp_path, n=2):
    wd = World(tmp_path)
    for i in range(n):
        scored_match(wd, f"m-{i}", T0 + timedelta(days=i + 1))
    wd.close()
    return wd, snapshot_from_ledger(wd.cfg.ledger.ledger_path, wd.cfg.snapshot_dir)


# ------------------------------------------------------------------------------------------ fail-closed trust
def test_contradictory_anchors_for_one_sequence_invalidate_everything(tmp_path):
    """Two anchors of the same sequence with different checkpoint hashes are fork evidence."""
    wd, snap = _snap(tmp_path)
    cp = anchoring.checkpoint_from_snapshot(snap)
    fork = {**cp, "previous_checkpoint_hash": "a" * 64}  # same head, different (valid) checkpoint body
    from tennis_data.ledger.checkpoint import checkpoint_hash

    fork["checkpoint_hash"] = checkpoint_hash(fork)
    fix = anchoring.LocalFixturePublisher(tmp_path / "fix")
    r1, _ = anchoring.publish(cp, fix, tmp_path / "ev", "x", now=T0)
    r2, _ = anchoring.publish(fork, fix, tmp_path / "ev", "x", now=T0)
    res = evaluate(snap, empty_outcomes(), wd.cfg.protocol, anchor_records=[("a", r1), ("b", r2)],
                   evidence_dir=tmp_path / "ev", fixture_root=tmp_path / "fix")  # fmt: skip
    assert res["anchoring"]["coverage"]["forked_sequences"] == [cp["sequence"]]
    assert res["anchoring"]["snapshot_head_evidence"] == "INVALID"
    assert {r["category"] for r in res["classification"]} == {"INVALID_EVIDENCE"}
    assert res["scores"]["operational"]["n"] == 0


def test_revoked_github_publication_is_not_external_evidence(tmp_path, monkeypatch):
    """The published commit itself is gone (history rewritten and garbage-collected): not EXTERNALLY_PUBLISHED."""
    _, snap = _snap(tmp_path)
    monkeypatch.setenv("TEST_GH_TOKEN", "t")
    cp = anchoring.checkpoint_from_snapshot(snap)
    gh = FakeGitHub()
    pub = anchoring.GitHubContentsPublisher("o/r", "b", "c", "TEST_GH_TOKEN", transport=gh.transport)
    rec, _ = anchoring.publish(cp, pub, tmp_path / "ev", "ci", now=T0)
    gh.files.clear()
    v = anchoring.verify_anchor(rec, snap, evidence_dir=tmp_path / "ev", online=True, transport=gh.transport)
    assert v["level"] == "LOCAL_ONLY" and v["receipt_status"] == "remote_http_404"


@needs_openssl
def test_corrupted_revoked_or_unverifiable_timestamps_fail_closed(tmp_path, monkeypatch):
    _, snap = _snap(tmp_path)
    cp = anchoring.checkpoint_from_snapshot(snap)
    tsa = LocalTSA(tmp_path / "tsa")
    rec, _ = anchoring.publish(cp, anchoring.Rfc3161Publisher("https://tsa.test/", [tsa.ca], tsa.transport),
                               tmp_path / "ev", "ci", now=T0)  # fmt: skip
    ev = tmp_path / "ev"
    assert anchoring.verify_anchor(rec, snap, evidence_dir=ev, tsa_ca_files=[tsa.ca])["level"] == (
        "INDEPENDENTLY_TIMESTAMPED")  # fmt: skip
    # trust withdrawn by the verifier (e.g. the TSA's certificate was revoked): no longer timestamped
    assert anchoring.verify_anchor(rec, snap, evidence_dir=ev, tsa_ca_files=[])["level"] == "LOCAL_ONLY"
    # verifier unavailable: never accepted unverified
    monkeypatch.setattr(timestamping, "openssl_path", lambda: None)
    v = anchoring.verify_anchor(rec, snap, evidence_dir=ev, tsa_ca_files=[tsa.ca])
    assert v["level"] == "LOCAL_ONLY" and v["receipt_status"].startswith("verifier_unavailable")
    monkeypatch.undo()
    # token bytes altered while the receipt still names the original hash
    tok = ev / rec["receipt"]["token_file"]
    raw = tok.read_bytes()
    tok.write_bytes(raw[:-1] + bytes([raw[-1] ^ 0xFF]))  # always changes the bytes (a 0x00 last byte made it a no-op)
    assert tok.read_bytes() != raw
    v = anchoring.verify_anchor(rec, snap, evidence_dir=ev, tsa_ca_files=[tsa.ca])
    assert v["level"] == "INVALID" and v["receipt_status"] == "token_hash_differs"
    # garbage token whose hash the (forged) receipt matches: unreadable, not timestamped
    tok.write_bytes(b"not a token")
    import hashlib

    forged = {**rec, "receipt": {**rec["receipt"], "token_sha256": hashlib.sha256(b"not a token").hexdigest()}}
    v = anchoring.verify_anchor(forged, snap, evidence_dir=ev, tsa_ca_files=[tsa.ca])
    assert v["level"] == "LOCAL_ONLY" and v["receipt_status"].startswith("unreadable_token")


# --------------------------------------------------------------------------------------- selection invariance
def test_later_events_never_change_the_selected_forecast(tmp_path):
    """Postponement, actual start, result, correction, a void of another forecast and outcome-snapshot additions
    all arrive after the forecasts; the selected forecast is the same at every cutoff."""
    wd = World(tmp_path)
    s = T0 + timedelta(days=1)
    f1 = wd.predict("m", s, s - timedelta(hours=10), prob=0.30, sched="v1")  # will lose: A is P1 or P2
    wd.predict("m", s, s - timedelta(hours=5), prob=0.95, sched="v2")
    wd.predict("m", s, s - timedelta(hours=1, minutes=5), prob=0.97, slot="t_minus_1h")
    s2 = s + timedelta(days=4)
    cuts = [ts(s - timedelta(minutes=50))]
    wd.record(s - timedelta(minutes=40), "MATCH_POSTPONED", "m", {
        "new_scheduled_start_at": ts(s + timedelta(days=1)), "reason": "rain", "source": {"provider": "f"}})  # fmt: skip
    cuts.append(ts(s - timedelta(minutes=39)))
    wd.start(s + timedelta(days=1, minutes=10), "m", s + timedelta(days=1, minutes=5))
    first = wd.result(s + timedelta(days=1, hours=3), "m", P2)
    cuts.append(ts(s + timedelta(days=1, hours=4)))
    wd.result(s + timedelta(days=1, hours=5), "m", P1, supersedes=first.event_id, reason="reversed")
    cuts.append(ts(s + timedelta(days=1, hours=6)))
    g1 = wd.predict("m2", s2, s2 - timedelta(hours=10), sched="v1")
    g2 = wd.predict("m2", s2, s2 - timedelta(hours=5), sched="v2")
    cuts.append(ts(s2 - timedelta(hours=4)))
    wd.start(s2 + timedelta(minutes=10), "m2", s2 + timedelta(minutes=5))
    wo = wd.result(s2 + timedelta(hours=1), "m2", P1, status="walkover", score="W/O")
    wd.record(s2 + timedelta(hours=2), "PREDICTION_VOIDED", "m2", {
        "reason_code": "walkover_no_play", "evidence_event_ids": [wo.event_id], "actor": "op", "note": ""},
        ref=g2.event_id)  # fmt: skip
    cuts.append(None)
    wd.close()
    snap = snapshot_from_ledger(wd.cfg.ledger.ledger_path, wd.cfg.snapshot_dir)
    oc_conflict = load_outcomes(write_outcomes(tmp_path / "o.jsonl", [
        outcome_record("m", "feed_b", "completed", P2, (P1, P2))]))  # fmt: skip
    seen = []
    for cut in cuts:
        for oc in (empty_outcomes(), oc_conflict):
            res = evaluate(snap, oc, wd.cfg.protocol, cutoff=cut)
            sel = {r["match_id"]: r for r in res["classification"] if r["selected"]}
            seen.append((sel["m"]["seq"], sel.get("m2", {}).get("seq")))
            assert sel["m"]["scheduled_start_at"] == ts(s)  # the committed schedule is never rewritten
    assert {a for a, _ in seen} == {f1.seq} and {b for _, b in seen} == {None, g1.seq}
    assert [b for _, b in seen][-4:] == [g1.seq] * 4  # from its first cutoff on, m2's selection is fixed
    final = evaluate(snap, empty_outcomes(), wd.cfg.protocol)
    rows = {r["seq"]: r for r in final["classification"]}
    assert rows[f1.seq]["category"] == "SCORABLE" and rows[f1.seq]["outcome_corrected"]
    assert rows[g1.seq]["category"] == "OUTCOME_WALKOVER" and rows[g2.seq]["category"] == "DUPLICATE_SUPERSEDED"
    disputed = evaluate(snap, oc_conflict, wd.cfg.protocol)
    assert {r["seq"]: r for r in disputed["classification"]}[f1.seq]["category"] == "OUTCOME_DISPUTED"


# -------------------------------------------------------------------------------------- repository evidence
def test_committed_reports_and_configuration_claim_no_prospective_evidence():
    rep = PROJECT_ROOT / "reports" / "evaluation"
    live = (rep / "prospective_report.md").read_text()
    assert "**Genuine prospective evaluation.**" in live and "Eligible genuine prospective forecasts: 0." in live
    assert "SYNTHETIC" not in live.split("## Selection")[0].replace("SYNTHETIC/", "")
    for f in sorted((rep / "synthetic_demonstration").glob("*.md")):
        assert "SYNTHETIC DEMONSTRATION: NOT PROSPECTIVE EVIDENCE" in f.read_text(), f.name
    for f in sorted(rep.glob("*_report.md")):  # no live report claims a verified timestamp or anchor
        text = f.read_text()
        assert "0 anchor record(s)" in text and "0 TSA trust anchor(s)" in text, f.name
    with (PROJECT_ROOT / "config" / "evaluation.toml").open("rb") as fh:
        ev = tomllib.load(fh)
    assert ev["publishers"]["github"]["enabled"] is False and ev["publishers"]["rfc3161"]["enabled"] is False
    assert ev["publishers"]["rfc3161"]["trusted_ca_files"] == [] and ev["protocol"]["cohort_mode"] == "PROSPECTIVE"
    with (PROJECT_ROOT / "config" / "prospective.toml").open("rb") as fh:
        st = tomllib.load(fh)["status"]
    assert st == {"backfill": "INCOMPLETE", "prospective_predictions": "DISABLED"}
