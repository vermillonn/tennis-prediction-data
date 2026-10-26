"""``tennis-data eval-*`` commands (Phase 4C). Every command is read-only with respect to the ledger.

Publication (``eval-checkpoint-publish``) writes only new evidence files and, for an explicitly enabled remote
backend, a new remote file; it never writes to the ledger.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

COMMANDS = {
    "eval-snapshot": "export the ledger read-only into a content-addressed snapshot",
    "eval-checkpoint": "compute the checkpoint of a snapshot's head (no database access)",
    "eval-checkpoint-publish": "publish a checkpoint with a configured backend and record the anchor",
    "eval-checkpoint-verify": "verify anchor records and retained checkpoints against a snapshot",
    "eval-run": "evaluate a ledger snapshot and write an audit package (and optionally the reports)",
    "eval-verify": "independently verify an audit package by recomputing everything from its inputs",
    "eval-calibration": "print a calibration table of a verified package as CSV",
    "eval-exclusions": "list exclusions or unresolved outcomes of a verified package",
    "eval-demo": "build the SYNTHETIC demonstration ledger and evaluate it (clearly labelled synthetic)",
    "eval-benchmark": "benchmark evaluation stages on synthetic ledgers of increasing size",
}


def register(sub) -> None:
    for name, helptext in COMMANDS.items():
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--evaluation-config", help="config/evaluation.toml by default")
        if name in ("eval-checkpoint", "eval-checkpoint-verify", "eval-run"):
            p.add_argument("--export", help="ledger export events.jsonl (default: snapshot the configured ledger)")
        if name in ("eval-checkpoint-verify", "eval-run", "eval-verify"):
            p.add_argument("--tsa-ca", action="append", default=[], help="TSA CA certificate the verifier trusts")
            p.add_argument("--online", action="store_true", help="fetch remote publications to verify them")
        if name == "eval-checkpoint":
            p.add_argument("--previous", help="previous checkpoint file to chain to")
            p.add_argument("--out", help="write the checkpoint file here (new file only)")
        if name == "eval-checkpoint-publish":
            p.add_argument("checkpoint", help="checkpoint JSON file (from eval-checkpoint or ledger-checkpoint)")
            p.add_argument("--backend", required=True, choices=["local_fixture", "github_contents", "rfc3161"])
            p.add_argument("--identity", required=True, help="publisher identity recorded in the anchor")
        if name == "eval-run":
            p.add_argument("--outcomes", help="outcome snapshot (tennis-outcome-snapshot-1 JSONL)")
            p.add_argument("--cutoff", help="evaluation cutoff (canonical UTC timestamp)")
            p.add_argument("--package", help="new package directory (default: under [outputs].package_dir)")
            p.add_argument("--write-reports", action="store_true", help="also copy the reports to reports_dir")
        if name in ("eval-verify", "eval-calibration", "eval-exclusions"):
            p.add_argument("package")
        if name == "eval-verify":
            p.add_argument("--manifest-sha256", help="independently obtained manifest hash to compare with")
            p.add_argument("--check-protocol", action="store_true", help="require the configured protocol")
        if name == "eval-calibration":
            p.add_argument("--cohort", choices=["primary", "operational"], default="primary")
            p.add_argument("--model", choices=["M4A", "Elo"], default="M4A")
        if name == "eval-exclusions":
            p.add_argument("--unresolved", action="store_true", help="unresolved outcomes instead of exclusions")
        if name == "eval-demo":
            p.add_argument("--out", required=True, help="new directory for the synthetic demo")
            p.add_argument("--write-reports", action="store_true",
                           help="copy the demo reports to reports_dir/synthetic_demonstration")  # fmt: skip
            p.add_argument("--weeks", type=int, default=30)
            p.add_argument("--per-week", type=int, default=14)
        if name == "eval-benchmark":
            p.add_argument("--sizes", default="100,1000,10000")
            p.add_argument("--out", help="work directory (default: temporary)")


def _snapshot(cfg, args):
    from tennis_data.evaluation_prospective.snapshot import load_export, snapshot_from_ledger

    if getattr(args, "export", None):
        return load_export(Path(args.export), cfg.limits)
    if not cfg.ledger.ledger_path.exists():
        return None
    return snapshot_from_ledger(cfg.ledger.ledger_path, cfg.snapshot_dir, cfg.limits)


def _verified(cfg, pkg):
    from tennis_data.evaluation_prospective.safeio import strict_json
    from tennis_data.evaluation_prospective.verify import verify_package

    v = verify_package(Path(pkg), limits=cfg.limits)
    if not v["ok"]:
        print(json.dumps(v, indent=2), file=sys.stderr)
        raise ValueError("package does not verify; refusing to read its tables")
    return strict_json((Path(pkg) / "results.json").read_bytes())


def run(args) -> int:
    from tennis_data.evaluation_prospective import anchoring
    from tennis_data.evaluation_prospective.config import load_evaluation_config

    cfg = load_evaluation_config(args.evaluation_config)
    cmd = args.command
    if cmd == "eval-snapshot":
        snap = _snapshot(cfg, args)
        print(json.dumps(snap.identity() | {"path": str(snap.path)} if snap else {"status": "no ledger"}, indent=2))
        return 0 if snap else 1
    if cmd == "eval-checkpoint":
        from tennis_data.evaluation_prospective.safeio import strict_json, write_new
        from tennis_data.ledger.hashing import canonical

        snap = _snapshot(cfg, args)
        if snap is None:
            print("error: no ledger to checkpoint", file=sys.stderr)
            return 1
        prev = strict_json(Path(args.previous).read_bytes()) if args.previous else None
        cp = anchoring.checkpoint_from_snapshot(snap, prev)
        if args.out:
            write_new(Path(args.out), canonical(cp) + b"\n")
        print(json.dumps(cp, indent=2))
        return 0
    if cmd == "eval-checkpoint-publish":
        from tennis_data.evaluation_prospective.safeio import read_bounded, strict_json

        cp = strict_json(read_bounded(Path(args.checkpoint), cfg.limits.max_receipt_bytes))
        cp = {k: v for k, v in cp.items()}
        try:
            rec, path = anchoring.publish(cp, anchoring.make_publisher(cfg, args.backend), cfg.evidence_dir,
                                          args.identity)  # fmt: skip
        except anchoring.PublicationError as e:
            print(f"publication failed (ledger untouched, no receipt recorded): {e}", file=sys.stderr)
            return 1
        print(json.dumps({"record": str(path), **rec}, indent=2))
        return 0
    if cmd == "eval-checkpoint-verify":
        snap = _snapshot(cfg, args)
        if snap is None:
            print("error: no ledger", file=sys.stderr)
            return 1
        lim = cfg.limits.max_receipt_bytes
        fix = cfg.config_path.parent.parent / cfg.publishers.get("local_fixture", {}).get(
            "root", "data/ledger/fixture_publications")  # fmt: skip
        anchors = [{"source": n, **anchoring.verify_anchor(r, snap, evidence_dir=cfg.evidence_dir, fixture_root=fix,
                                                           tsa_ca_files=[Path(c) for c in args.tsa_ca],
                                                           online=args.online, max_bytes=lim)}
                   for n, r in anchoring.load_records(cfg.evidence_dir, lim)]  # fmt: skip
        trusted = anchoring.compare_trusted(anchoring.load_trusted(cfg.trusted_checkpoint_dir, lim), snap)
        cov = anchoring.coverage(anchors, snap)
        ok = cov["invalid_anchors"] == 0 and not cov["forked_sequences"] and all(t["ok"] for t in trusted)
        print(json.dumps({"ok": ok, "snapshot": snap.identity(), "anchors": anchors, "trusted": trusted,
                          "coverage": cov}, indent=2))  # fmt: skip
        return 0 if ok else 1
    if cmd == "eval-run":
        return _run_eval(cfg, args)
    if cmd == "eval-verify":
        from tennis_data.evaluation_prospective.verify import verify_package

        v = verify_package(Path(args.package), tsa_ca_files=[Path(c) for c in args.tsa_ca], online=args.online,
                           expected_manifest_sha256=args.manifest_sha256,
                           protocol=cfg.protocol if args.check_protocol else None, limits=cfg.limits)  # fmt: skip
        print(json.dumps(v, indent=2))
        return 0 if v["ok"] else 1
    if cmd == "eval-calibration":
        import csv

        res = _verified(cfg, args.package)
        rows = res["calibration"][args.cohort][args.model]
        w = csv.writer(sys.stdout, lineterminator="\n")
        cols = ["bin", "bin_lower", "bin_upper", "n", "mean_predicted", "observed", "observed_minus_predicted"]
        w.writerow(cols)
        w.writerows([[r[c] for c in cols] for r in rows])
        return 0
    if cmd == "eval-exclusions":
        res = _verified(cfg, args.package)
        print(json.dumps(res["unresolved_outcomes"] if args.unresolved else res["exclusions"], indent=2))
        return 0
    if cmd == "eval-demo":
        return _run_demo(cfg, args)
    if cmd == "eval-benchmark":
        from tennis_data.evaluation_prospective.benchmark import run_benchmark

        sizes = [int(x) for x in args.sizes.split(",")]
        print(json.dumps(run_benchmark(cfg, sizes, Path(args.out) if args.out else None), indent=2))
        return 0
    raise ValueError(cmd)


def _run_eval(cfg, args) -> int:
    from tennis_data.evaluation_prospective.pipeline import build_package, copy_reports, default_provenance

    snap = _snapshot(cfg, args)
    fix = cfg.config_path.parent.parent / cfg.publishers.get("local_fixture", {}).get(
        "root", "data/ledger/fixture_publications")  # fmt: skip
    name = f"{snap.sha256[:16] if snap else 'no-ledger'}-{cfg.protocol_hash[:12]}"
    pkg = Path(args.package) if args.package else cfg.package_dir / name
    out = build_package(pkg, snapshot_path=snap.path if snap else None,
                        outcomes_path=Path(args.outcomes) if args.outcomes else None, protocol=cfg.protocol,
                        evidence_dir=cfg.evidence_dir, fixture_root=fix, trusted_dir=cfg.trusted_checkpoint_dir,
                        tsa_ca_files=[Path(c) for c in args.tsa_ca], online=args.online, cutoff=args.cutoff,
                        limits=cfg.limits, provenance=default_provenance(cfg))  # fmt: skip
    res = out["results"]
    if args.write_reports:
        copy_reports(pkg, cfg.reports_dir)
    print(json.dumps({"package": str(pkg), "manifest_sha256": out["manifest_sha256"],
                      "ledger": res["inputs"]["ledger"], "predictions": res["coverage"]["total_predictions"],
                      "primary_cohort": res["scores"]["primary"]["n"],
                      "operational_cohort": res["scores"]["operational"]["n"],
                      "snapshot_head_evidence": res["anchoring"]["snapshot_head_evidence"]}, indent=2))  # fmt: skip
    return 0


def _run_demo(cfg, args) -> int:
    from tennis_data.evaluation_prospective.demo import build_demo, demo_protocol
    from tennis_data.evaluation_prospective.pipeline import build_package, copy_reports, default_provenance
    from tennis_data.evaluation_prospective.snapshot import snapshot_from_ledger

    root = Path(args.out)
    d = build_demo(cfg, root, weeks=args.weeks, per_week=args.per_week)
    snap = snapshot_from_ledger(d["ledger"], root / "snapshots", cfg.limits)
    prov = default_provenance(cfg) | {"data": "SYNTHETIC demonstration (evaluation_prospective.demo)"}
    out = build_package(root / "package", snapshot_path=snap.path, outcomes_path=d["outcomes"],
                        protocol=demo_protocol(cfg), evidence_dir=d["evidence"], fixture_root=d["fixture_root"],
                        trusted_dir=d["trusted"], limits=cfg.limits, provenance=prov)  # fmt: skip
    if args.write_reports:
        copy_reports(root / "package", cfg.reports_dir / "synthetic_demonstration")
    res = out["results"]
    print(json.dumps({"package": str(root / "package"), "manifest_sha256": out["manifest_sha256"],
                      "cases": d["cases"], "operational_cohort": res["scores"]["operational"]["n"],
                      "delta_ll": (res["scores"]["operational"]["paired"] or {}).get("log_loss_diff")}, indent=2))  # fmt: skip
    return 0
