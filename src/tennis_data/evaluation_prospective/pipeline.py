"""Read-only evaluation pipeline and self-contained audit packages.

    verified ledger snapshot → integrity + checkpoint/anchor verification → classification → outcome reconciliation
    → eligibility → paired M4A/Elo metrics → calibration, subgroups, uncertainty → results.json + reports → manifest

``evaluate`` is a pure function of its inputs (snapshot bytes, outcome snapshot, protocol, anchor evidence, the
verifier's TSA trust anchors and the online/offline choice): no wall-clock time, absolute path or host detail enters
``results.json``, so the same inputs give byte-identical results and reports. Operational provenance (git commit,
status gates) is kept in ``manifest.json`` only.

An audit package copies every input it used, so ``verify.verify_package`` can recompute everything without the
original SQLite ledger. Packages are never overwritten.
"""

from __future__ import annotations

import csv
import io
import shutil
import subprocess
from pathlib import Path

from tennis_data.evaluation_prospective import anchoring
from tennis_data.evaluation_prospective.calibration import calibration_tables, subgroup_tables
from tennis_data.evaluation_prospective.config import DEFAULT_LIMITS, EvaluationConfig, Limits, validate_protocol
from tennis_data.evaluation_prospective.coverage import coverage, exclusions, unresolved, voids
from tennis_data.evaluation_prospective.eligibility import classify
from tennis_data.evaluation_prospective.integrity import verify_snapshot
from tennis_data.evaluation_prospective.metrics import score
from tennis_data.evaluation_prospective.outcomes import (
    OutcomeSnapshot,
    assert_outcomes_unchanged,
    empty_outcomes,
    load_outcomes,
)
from tennis_data.evaluation_prospective.safeio import (
    UntrustedInputError,
    read_bounded,
    resolve_under,
    strict_json,
    write_new,
)
from tennis_data.evaluation_prospective.snapshot import LedgerSnapshot, assert_unchanged, load_export
from tennis_data.evaluation_prospective.uncertainty import paired_uncertainty
from tennis_data.ledger.hashing import canonical, sha256_hex

RESULTS_FORMAT = "tennis-prospective-evaluation-1"
PACKAGE_FORMAT = "tennis-evaluation-package-1"
REPORTS = ("prospective_report.md", "coverage_report.md", "integrity_report.md", "calibration_report.md",
           "uncertainty_report.md")  # fmt: skip
NO_LEDGER = "no operational ledger exists"


def git_commit(root: Path | None = None) -> str | None:
    """HEAD of the code that ran, with ``+uncommitted-changes`` when the working tree differs from it."""
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10,  # noqa: S607
                           cwd=root, check=False)  # fmt: skip
        if r.returncode != 0 or not r.stdout.strip():
            return None
        d = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True,  # noqa: S607
                           text=True, timeout=10, cwd=root, check=False)  # fmt: skip
        return r.stdout.strip() + ("+uncommitted-changes" if d.stdout.strip() else "")
    except (OSError, subprocess.TimeoutExpired):
        return None


def evaluate(snap: LedgerSnapshot | None, outcomes: OutcomeSnapshot, protocol: dict, *,
             anchor_records: list[tuple[str, dict]] = (), trusted: list[tuple[str, dict]] = (),
             evidence_dir: Path | None = None, fixture_root: Path | None = None, tsa_ca_files: list[Path] = (),
             online: bool = False, transport=anchoring.urllib_transport, cutoff: str | None = None,
             limits: Limits = DEFAULT_LIMITS, tol: float = 1e-12) -> dict:  # fmt: skip
    validate_protocol(protocol)
    cas = sorted(sha256_hex(Path(c).read_bytes()) for c in tsa_ca_files)
    inputs = {"ledger": snap.identity() if snap else {"status": NO_LEDGER},
              "outcomes": outcomes.identity(), "evaluation_cutoff": cutoff,
              "anchor_records": sorted(name for name, _ in anchor_records),
              "trusted_checkpoints": sorted(name for name, _ in trusted),
              "anchor_verification": "online" if online else "offline", "tsa_trust_anchor_sha256": cas}  # fmt: skip
    if snap is None:
        integ = {"ok": None, "status": NO_LEDGER, "events": 0, "issues": [], "in_chain_checkpoints": []}
        anchors, trusted_cmp = [], []
        cov = anchoring.coverage([], None)
        cls = {"rows": [], "outcomes": {}, "lifecycle_ok": None, "events_considered": 0, "matches": 0}
    else:
        integ = verify_snapshot(snap, tol)
        anchors = []
        for name, rec in anchor_records:
            a = anchoring.verify_anchor(rec, snap, evidence_dir=evidence_dir, fixture_root=fixture_root,
                                        tsa_ca_files=list(tsa_ca_files), online=online, transport=transport,
                                        max_bytes=limits.max_receipt_bytes)  # fmt: skip
            anchors.append({"source": name, **a})
        trusted_cmp = anchoring.compare_trusted(list(trusted), snap)
        cov = anchoring.coverage(anchors, snap)
        evidence_ok = cov["invalid_anchors"] == 0 and not cov["forked_sequences"] and all(t["ok"] for t in trusted_cmp)
        cls = classify(list(snap.events), protocol, outcomes, integrity_ok=integ["ok"], evidence_ok=evidence_ok,
                       anchor_cov=cov, cutoff=cutoff)  # fmt: skip
        integ["lifecycle_ok"] = cls["lifecycle_ok"]
        integ["evidence_ok"] = evidence_ok
    rows = sorted(cls["rows"], key=lambda r: r["seq"])
    cohorts = {"primary": [r for r in rows if r["cohort"] == "primary"],
               "operational": [r for r in rows if r["cohort"] in ("primary", "operational")],
               "retirement_secondary": [r for r in rows if r["cohort"] == "retirement_secondary"]}  # fmt: skip
    head_level = None
    if snap is not None:
        head_level = "INVALID" if not (integ["ok"] and integ.get("evidence_ok")) else anchoring.evidence_for(
            max(snap.head_seq, 1), cov)[0]  # fmt: skip
    mode = protocol["cohort_mode"]
    return {
        "format": RESULTS_FORMAT,
        "cohort_mode": mode,
        "protocol": protocol,
        "protocol_hash": sha256_hex(canonical(protocol)),
        "inputs": inputs,
        "integrity": integ,
        "anchoring": {
            "anchors": anchors,
            "trusted_checkpoints": trusted_cmp,
            "coverage": cov,
            "snapshot_head_evidence": head_level,
        },  # fmt: skip
        "coverage": coverage(rows, protocol)
        | {"events_considered": cls["events_considered"], "matches": cls["matches"]},  # fmt: skip
        "scores": {k: score(v, protocol) for k, v in cohorts.items()},
        "calibration": {
            k: calibration_tables(cohorts[k], protocol["calibration_bins"]) for k in ("primary", "operational")
        },
        "subgroups": {k: subgroup_tables(cohorts[k], protocol) for k in ("primary", "operational")},
        "uncertainty": {k: paired_uncertainty(cohorts[k], protocol) for k in ("primary", "operational")},
        "exclusions": exclusions(rows, mode),
        "voids": voids(rows),
        "unresolved_outcomes": unresolved(cls["outcomes"], rows, mode),
        "outcomes": [cls["outcomes"][m] for m in sorted(cls["outcomes"])],
        "classification": rows,
    }


def results_bytes(results: dict) -> bytes:
    return canonical(results) + b"\n"


# ------------------------------------------------------------------------------------------------- packages
def _copy(src: Path, dst: Path, limit: int) -> None:
    write_new(dst, read_bounded(src, limit))


def _collect_evidence(evidence_dir: Path | None, fixture_root: Path | None, pkg_inputs: Path, limits: Limits,
                      ledger_id: str | None) -> None:  # fmt: skip
    """Copy anchor records of this ledger plus the token and fixture files their receipts reference."""
    for rel, rec in anchoring.load_records(evidence_dir, limits.max_receipt_bytes):
        if "_error" not in rec and rec.get("ledger_id") != ledger_id:
            continue
        _copy(evidence_dir / rel, pkg_inputs / "evidence" / rel, limits.max_receipt_bytes)
        r = rec.get("receipt") if isinstance(rec.get("receipt"), dict) else {}
        try:
            if r.get("type") == "rfc3161" and evidence_dir is not None:
                src = resolve_under(evidence_dir, r.get("token_file", ""))
                dst = resolve_under(pkg_inputs / "evidence", r["token_file"])
                if src.is_file() and not dst.exists():
                    _copy(src, dst, limits.max_receipt_bytes)
            if r.get("type") == "local_fixture" and fixture_root is not None:
                src = resolve_under(fixture_root, r.get("path", ""))
                dst = resolve_under(pkg_inputs / "fixture_publications", r["path"])
                if src.is_file() and not dst.exists():
                    _copy(src, dst, limits.max_receipt_bytes)
        except UntrustedInputError:
            continue  # an unsafe path stays uncopied; verification then reports the evidence as missing


def build_package(out: Path, *, snapshot_path: Path | None, outcomes_path: Path | None, protocol: dict,
                  evidence_dir: Path | None = None, fixture_root: Path | None = None,
                  trusted_dir: Path | None = None, tsa_ca_files: list[Path] = (), online: bool = False,
                  transport=anchoring.urllib_transport, cutoff: str | None = None, limits: Limits = DEFAULT_LIMITS,
                  provenance: dict | None = None) -> dict:  # fmt: skip
    """Copy inputs into a new package directory, evaluate from the copies, write results, reports and manifest."""
    from tennis_data.evaluation_prospective.report import write_reports

    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"package directory {out} is not empty (packages are never overwritten)")
    inp = out / "inputs"
    src_snap = load_export(snapshot_path, limits) if snapshot_path else None
    src_out = load_outcomes(outcomes_path, limits) if outcomes_path else None
    if snapshot_path:
        _copy(snapshot_path, inp / "ledger" / "events.jsonl", limits.max_export_bytes)
    if outcomes_path:
        _copy(outcomes_path, inp / "outcomes.jsonl", limits.max_export_bytes)
    _collect_evidence(evidence_dir, fixture_root, inp, limits, src_snap.ledger_id if src_snap else None)
    for name, _cp in anchoring.load_trusted(trusted_dir, limits.max_receipt_bytes):
        _copy(trusted_dir / name, inp / "trusted_checkpoints" / name, limits.max_receipt_bytes)
    write_new(inp / "protocol.json", canonical(protocol) + b"\n")
    res = evaluate_package_inputs(out, tsa_ca_files=tsa_ca_files, online=online, transport=transport, cutoff=cutoff,
                                  limits=limits)  # fmt: skip
    # the source files must not have changed while the package was being built
    if src_snap is not None:
        assert_unchanged(src_snap)
    if src_out is not None:
        assert_outcomes_unchanged(src_out)
    write_new(out / "results.json", results_bytes(res))
    # normalized exactly as the manifest stores it, so verification re-renders identical reports
    prov = strict_json(canonical({"git_commit": None, "status_gates": None, **(provenance or {})}))
    write_reports(res, prov, out / "reports")
    write_tables(res, out / "tables")
    files = {p.relative_to(out).as_posix(): sha256_hex(p.read_bytes()) for p in sorted(out.rglob("*")) if p.is_file()}
    manifest = {"format": PACKAGE_FORMAT, "files": files, "results_sha256": files["results.json"],
                "identity": package_identity(res), "provenance": prov}  # fmt: skip
    write_new(out / "manifest.json", canonical(manifest) + b"\n")
    return {"package": str(out), "results": res, "manifest": manifest,
            "manifest_sha256": sha256_hex((out / "manifest.json").read_bytes())}  # fmt: skip


def package_identity(res: dict) -> dict:
    led = res["inputs"]["ledger"]
    p = res["protocol"]
    return {"ledger_id": led.get("ledger_id"), "ledger_head_seq": led.get("head_seq"),
            "ledger_head_hash": led.get("head_hash"), "snapshot_sha256": led.get("snapshot_sha256"),
            "outcome_snapshot_sha256": res["inputs"]["outcomes"]["outcome_snapshot_sha256"],
            "evaluation_cutoff": res["inputs"]["evaluation_cutoff"], "protocol_version": p["version"],
            "protocol_hash": res["protocol_hash"], "cohort_mode": res["cohort_mode"],
            "frozen_model_hash": p["approved_frozen_hash"], "configuration_hash": p["approved_configuration_hash"],
            "anchor_verification": res["inputs"]["anchor_verification"],
            "tsa_trust_anchor_sha256": res["inputs"]["tsa_trust_anchor_sha256"]}  # fmt: skip


def evaluate_package_inputs(pkg: Path, *, tsa_ca_files: list[Path] = (), online: bool = False,
                            transport=anchoring.urllib_transport, cutoff: str | None = None,
                            limits: Limits = DEFAULT_LIMITS, protocol: dict | None = None) -> dict:  # fmt: skip
    inp = Path(pkg) / "inputs"
    prot = protocol or strict_json(read_bounded(inp / "protocol.json", limits.max_receipt_bytes))
    snap = load_export(inp / "ledger" / "events.jsonl", limits) if (inp / "ledger" / "events.jsonl").exists() else None
    oc = load_outcomes(inp / "outcomes.jsonl", limits) if (inp / "outcomes.jsonl").exists() else empty_outcomes()
    ev = inp / "evidence"
    res = evaluate(snap, oc, prot, anchor_records=anchoring.load_records(ev, limits.max_receipt_bytes),
                   trusted=anchoring.load_trusted(inp / "trusted_checkpoints", limits.max_receipt_bytes),
                   evidence_dir=ev, fixture_root=inp / "fixture_publications", tsa_ca_files=tsa_ca_files,
                   online=online, transport=transport, cutoff=cutoff, limits=limits)  # fmt: skip
    if snap is not None:
        assert_unchanged(snap)
    return res


def write_tables(res: dict, out: Path) -> None:
    def put(name, rows, cols):
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(cols)
        for r in rows:
            w.writerow(["" if r.get(c) is None else (";".join(r[c]) if isinstance(r.get(c), list) else r[c])
                        for c in cols])  # fmt: skip
        write_new(out / name, buf.getvalue().encode("utf-8"))

    put("classification.csv", res["classification"],
        ["seq", "event_id", "match_id", "mode", "forecast_slot", "recorded_at", "category", "selected", "timing",
         "evidence_level", "attested_time", "outcome", "cohort", "probability_a", "baseline_elo_probability_a", "y",
         "flags"])  # fmt: skip
    put("exclusions.csv", res["exclusions"], ["seq", "event_id", "match_id", "category", "outcome", "timing",
                                              "void_reason", "void_actor", "flags"])  # fmt: skip
    for cohort in ("primary", "operational"):
        for model, rows in res["calibration"][cohort].items():
            put(f"calibration_{cohort}_{model}.csv", rows, ["bin", "bin_lower", "bin_upper", "n", "mean_predicted",
                                                            "observed", "observed_minus_predicted"])  # fmt: skip


def copy_reports(pkg: Path, reports_dir: Path) -> list[Path]:
    """Publish a package's reports into the repository reports directory (replacing earlier generated copies)."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for name in REPORTS:
        shutil.copyfile(Path(pkg) / "reports" / name, reports_dir / name)
        out.append(reports_dir / name)
    return out


def default_provenance(cfg: EvaluationConfig) -> dict:
    p = cfg.ledger.prospective
    return {"git_commit": git_commit(cfg.config_path.parent.parent),
            "status_gates": {"backfill": p.backfill_status, "prospective_predictions": p.prospective_predictions,
                             "real_provider_connectivity": "UNVERIFIED",
                             "external_checkpoint_publication": publication_status(cfg)}}  # fmt: skip


def publication_status(cfg: EvaluationConfig) -> str:
    pubs = cfg.publishers
    on = [k for k in ("github", "rfc3161") if pubs.get(k, {}).get("enabled")]
    return f"CONFIGURED ({', '.join(on)})" if on else "NOT CONFIGURED"
