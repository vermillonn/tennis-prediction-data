"""Phase 4C: read-only behaviour, reproducibility, independent package verification and the eval-* CLI."""

from __future__ import annotations

import hashlib
import json
from datetime import timedelta
from pathlib import Path

import pytest

from evaluation_helpers import OPENSSL, T0, LocalTSA, World, eval_cfg, outcome_record, scored_match, write_outcomes
from tennis_data.cli import main
from tennis_data.config import PROJECT_ROOT
from tennis_data.evaluation_prospective import anchoring
from tennis_data.evaluation_prospective.demo import build_demo, demo_protocol
from tennis_data.evaluation_prospective.pipeline import build_package
from tennis_data.evaluation_prospective.snapshot import (
    SnapshotChanged,
    assert_unchanged,
    load_export,
    snapshot_from_ledger,
)
from tennis_data.evaluation_prospective.verify import verify_package
from tennis_data.ledger.hashing import canonical, sha256_hex

P1, P2 = 104925, 207989
FROZEN = PROJECT_ROOT / "tests" / "fixtures" / "m4a_production"


def _tree_hash(root: Path) -> dict:
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}  # fmt: skip


def _world(tmp_path, n=3, tsa=None):
    wd = World(tmp_path)
    for i in range(n):
        scored_match(wd, f"m-{i}", T0 + timedelta(days=i + 1), winner=P1 if i % 2 else P2)
    wd.close()
    snap = snapshot_from_ledger(wd.cfg.ledger.ledger_path, wd.cfg.snapshot_dir)
    oc = write_outcomes(tmp_path / "outcomes.jsonl", [
        outcome_record(f"m-{i}", "feed_b", "completed", P1 if i % 2 else P2, (P1, P2)) for i in range(n)])  # fmt: skip
    cp = anchoring.checkpoint_from_snapshot(snap)
    anchoring.publish(cp, anchoring.LocalFixturePublisher(tmp_path / "fix"), wd.cfg.evidence_dir, "test", now=T0)
    if tsa is not None:
        anchoring.publish(cp, anchoring.Rfc3161Publisher("https://tsa.test/", [tsa.ca], tsa.transport),
                          wd.cfg.evidence_dir, "test", now=T0)  # fmt: skip
    return wd, snap, oc


def _package(wd, snap, oc, out, **kw):
    return build_package(out, snapshot_path=snap.path, outcomes_path=oc, protocol=wd.cfg.protocol,
                         evidence_dir=wd.cfg.evidence_dir, fixture_root=wd.tmp / "fix",
                         provenance={"git_commit": "test", "status_gates": {"prospective_predictions": "DISABLED"}},
                         **kw)  # fmt: skip


def test_evaluation_never_mutates_ledger_or_frozen_artifacts(tmp_path):
    wd, snap, oc = _world(tmp_path)
    db = wd.cfg.ledger.ledger_path
    before_db = hashlib.sha256(db.read_bytes()).hexdigest()
    before_frozen = _tree_hash(FROZEN)
    before_ev = _tree_hash(wd.cfg.evidence_dir)
    snap2 = snapshot_from_ledger(db, wd.cfg.snapshot_dir)  # read-only export of the live database
    _package(wd, snap2, oc, tmp_path / "pkg")
    assert hashlib.sha256(db.read_bytes()).hexdigest() == before_db
    assert _tree_hash(FROZEN) == before_frozen and _tree_hash(wd.cfg.evidence_dir) == before_ev
    from tennis_data.ledger.verification import verify_ledger

    assert verify_ledger(db).events == snap.head_seq


def test_snapshot_change_during_evaluation_is_detected(tmp_path):
    _, snap, _ = _world(tmp_path)
    s = load_export(snap.path)
    snap.path.write_bytes(snap.path.read_bytes() + b"\n")  # changed after loading
    with pytest.raises(SnapshotChanged):
        assert_unchanged(s)


def test_reproducible_packages(tmp_path):
    wd, snap, oc = _world(tmp_path)
    a = _package(wd, snap, oc, tmp_path / "a")
    b = _package(wd, snap, oc, tmp_path / "b")
    ha, hb = _tree_hash(tmp_path / "a"), _tree_hash(tmp_path / "b")
    assert ha == hb and a["manifest_sha256"] == b["manifest_sha256"]
    with pytest.raises(FileExistsError):
        _package(wd, snap, oc, tmp_path / "a")


def _rehash(pkg: Path, rel: str):
    """An attacker who edits a file and also updates the manifest consistently."""
    man = json.loads((pkg / "manifest.json").read_bytes())
    man["files"][rel] = sha256_hex((pkg / rel).read_bytes())
    if rel == "results.json":
        man["results_sha256"] = man["files"][rel]
    (pkg / "manifest.json").write_bytes(canonical(man) + b"\n")


def test_independent_verification(tmp_path):
    wd, snap, oc = _world(tmp_path)
    out = _package(wd, snap, oc, tmp_path / "pkg")
    pkg = tmp_path / "pkg"
    v = verify_package(pkg, expected_manifest_sha256=out["manifest_sha256"], protocol=wd.cfg.protocol)
    assert v["ok"], v
    assert v["recomputed"]["operational_n"] == 3 and v["recomputed"]["primary_n"] == 0
    assert not verify_package(pkg, expected_manifest_sha256="0" * 64)["ok"]


@pytest.mark.parametrize("attack", ["corrupt_export", "alter_outcomes", "alter_report", "alter_results",
                                    "unlisted_file", "drop_receipt"])  # fmt: skip
def test_tampered_packages_fail(tmp_path, attack):
    wd, snap, oc = _world(tmp_path)
    _package(wd, snap, oc, tmp_path / "pkg")
    pkg = tmp_path / "pkg"
    if attack == "corrupt_export":
        p = pkg / "inputs/ledger/events.jsonl"
        p.write_bytes(p.read_bytes().replace(b"0.62", b"0.72", 1))
        failed = "manifest"
    elif attack == "alter_outcomes":
        p = pkg / "inputs/outcomes.jsonl"
        p.write_bytes(p.read_bytes().replace(f'"winner_id":{P2}'.encode(), f'"winner_id":{P1}'.encode(), 1))
        _rehash(pkg, "inputs/outcomes.jsonl")
        failed = "recompute"
    elif attack == "alter_report":
        p = pkg / "reports/prospective_report.md"
        p.write_bytes(p.read_bytes().replace(b"**3 matches**", b"**30 matches**"))
        _rehash(pkg, "reports/prospective_report.md")
        failed = "reports"
    elif attack == "alter_results":
        res = json.loads((pkg / "results.json").read_bytes())
        res["scores"]["operational"]["models"]["M4A"]["log_loss"] = 0.1
        (pkg / "results.json").write_bytes(canonical(res) + b"\n")
        _rehash(pkg, "results.json")
        failed = "recompute"
    elif attack == "unlisted_file":
        (pkg / "reports" / "extra.md").write_text("claims")
        failed = "manifest"
    else:
        rec = next((pkg / "inputs/evidence/records").rglob("*.json"))
        rel = rec.relative_to(pkg).as_posix()
        rec.unlink()
        man = json.loads((pkg / "manifest.json").read_bytes())
        del man["files"][rel]
        (pkg / "manifest.json").write_bytes(canonical(man) + b"\n")
        failed = "recompute"
    v = verify_package(pkg)
    assert not v["ok"]
    assert any(c["check"] == failed and not c["ok"] for c in v["checks"]), v["checks"]


@pytest.mark.skipif(OPENSSL is None, reason="openssl binary not installed")
def test_package_with_timestamp_evidence(tmp_path):
    tsa = LocalTSA(tmp_path / "tsa")
    wd, snap, oc = _world(tmp_path, tsa=tsa)
    _package(wd, snap, oc, tmp_path / "pkg", tsa_ca_files=[tsa.ca])
    v = verify_package(tmp_path / "pkg", tsa_ca_files=[tsa.ca])
    assert v["ok"] and v["recomputed"]["primary_n"] == 3, v
    # the same package checked by someone who does not trust that TSA does not reproduce the primary cohort
    v = verify_package(tmp_path / "pkg")
    assert not v["ok"] and any(c["check"] == "trust_anchors" for c in v["checks"])
    # removing the token file (and its manifest entry) is detected by recomputation
    pkg = tmp_path / "pkg"
    tok = next((pkg / "inputs/evidence/tokens").rglob("*.tsr"))
    rel = tok.relative_to(pkg).as_posix()
    tok.unlink()
    man = json.loads((pkg / "manifest.json").read_bytes())
    del man["files"][rel]
    (pkg / "manifest.json").write_bytes(canonical(man) + b"\n")
    v = verify_package(pkg, tsa_ca_files=[tsa.ca])
    assert not v["ok"] and any(c["check"] == "recompute" and not c["ok"] for c in v["checks"])


# --------------------------------------------------------------------------------------------------------- CLI
def _configs(tmp_path: Path) -> Path:
    led = (PROJECT_ROOT / "config/ledger.toml").read_text()
    led = led.replace('prospective_config = "config/prospective.toml"',
                      f'prospective_config = "{PROJECT_ROOT / "config/prospective.toml"}"')  # fmt: skip
    for k in ("ledger_path", "export_dir", "checkpoint_dir", "reports_dir"):
        led = led.replace(f'{k} = "', f'{k} = "{tmp_path}/', 1)
    (tmp_path / "ledger.toml").write_text(led)
    ev = (PROJECT_ROOT / "config/evaluation.toml").read_text()
    ev = ev.replace('ledger_config = "config/ledger.toml"', f'ledger_config = "{tmp_path / "ledger.toml"}"')
    for k in ("snapshot_dir", "package_dir", "reports_dir", "evidence_dir", "trusted_checkpoint_dir", "root"):
        ev = ev.replace(f'{k} = "', f'{k} = "{tmp_path}/', 1)
    (tmp_path / "evaluation.toml").write_text(ev)
    return tmp_path / "evaluation.toml"


def _cli(capsys, *args):
    code = main(list(args))
    out = capsys.readouterr()
    return code, out.out, out.err


def test_cli_end_to_end(tmp_path, capsys):
    conf = str(_configs(tmp_path))
    # no operational ledger: zero eligible prospective forecasts, honestly reported
    code, out, _ = _cli(capsys, "eval-run", "--evaluation-config", conf, "--write-reports")
    assert code == 0 and json.loads(out)["primary_cohort"] == 0
    rep = (tmp_path / "reports" / "evaluation" / "prospective_report.md").read_text()
    assert "Eligible genuine prospective forecasts: 0" in rep and "no operational ledger exists" in rep
    pkg = json.loads(out)["package"]
    code, out, _ = _cli(capsys, "eval-verify", pkg, "--evaluation-config", conf, "--check-protocol")
    assert code == 0 and json.loads(out)["ok"]
    # synthetic demonstration, clearly labelled
    code, out, _ = _cli(capsys, "eval-demo", "--evaluation-config", conf, "--out", str(tmp_path / "demo"), "--weeks",
                        "4", "--per-week", "6", "--write-reports")  # fmt: skip
    assert code == 0
    demo_rep = (tmp_path / "reports/evaluation/synthetic_demonstration/prospective_report.md").read_text()
    assert "SYNTHETIC DEMONSTRATION: NOT PROSPECTIVE EVIDENCE" in demo_rep
    dpkg = str(tmp_path / "demo" / "package")
    code, out, _ = _cli(capsys, "eval-verify", dpkg, "--evaluation-config", conf)
    assert code == 0 and json.loads(out)["ok"]
    code, out, _ = _cli(capsys, "eval-calibration", dpkg, "--evaluation-config", conf, "--cohort", "operational")
    assert code == 0 and out.startswith("bin,bin_lower")
    code, out, _ = _cli(capsys, "eval-exclusions", dpkg, "--evaluation-config", conf)
    assert code == 0 and isinstance(json.loads(out), list)
    code, out, _ = _cli(capsys, "eval-exclusions", dpkg, "--evaluation-config", conf, "--unresolved")
    assert code == 0
    # checkpoint evidence from an export, fixture publication, verification; remote backends stay disabled
    export = next((tmp_path / "demo" / "snapshots").rglob("events.jsonl"))
    cpf = tmp_path / "cp.json"
    code, out, _ = _cli(
        capsys, "eval-checkpoint", "--evaluation-config", conf, "--export", str(export), "--out", str(cpf)
    )
    assert code == 0 and json.loads(out)["sequence"] > 0
    code, out, _ = _cli(capsys, "eval-checkpoint-publish", str(cpf), "--evaluation-config", conf, "--backend",
                        "local_fixture", "--identity", "tester")  # fmt: skip
    assert code == 0 and json.loads(out)["receipt"]["type"] == "local_fixture"
    code, out, _ = _cli(capsys, "eval-checkpoint-verify", "--evaluation-config", conf, "--export", str(export))
    res = json.loads(out)
    assert code == 0 and res["anchors"][0]["level"] == "LOCAL_ONLY"
    code, _, err = _cli(capsys, "eval-checkpoint-publish", str(cpf), "--evaluation-config", conf, "--backend",
                        "github_contents", "--identity", "tester")  # fmt: skip
    assert code == 1 and "not enabled" in err
    # a tampered package refuses table export
    p = Path(dpkg) / "results.json"
    p.write_bytes(p.read_bytes().replace(b'"cohort_mode":"SYNTHETIC"', b'"cohort_mode":"PROSPECTIVE"', 1))
    code, _, err = _cli(capsys, "eval-calibration", dpkg, "--evaluation-config", conf)
    assert code == 2 and "does not verify" in err


def test_live_demo_protocol_differs_and_is_never_prospective(tmp_path):
    cfg = eval_cfg(tmp_path)
    p = demo_protocol(cfg)
    assert p["cohort_mode"] == "SYNTHETIC" and p["version"].endswith("+synthetic-demo")
    assert cfg.protocol["cohort_mode"] == "PROSPECTIVE"
    d = build_demo(cfg, tmp_path / "d", weeks=3, per_week=4, replay_matches=2)
    assert set(d["cases"]) <= {"completed", "pending", "cancelled", "players", "postponed", "late", "no_start",
                               "retired", "walkover", "default", "abandoned", "disputed", "corrected", "duplicate"}  # fmt: skip
