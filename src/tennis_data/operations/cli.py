"""``tennis-data ops-*`` commands (Phase 5A). None of them changes the status gate, the ledger or M4A."""

from __future__ import annotations

import json
from datetime import date

COMMANDS = {
    "ops-access": "probe every catalogued provider once and classify access (no credentials are printed)",
    "ops-run": "access audit, pinned download, conversion, Phase 4A backfill, compatibility, gap audit, readiness",
    "ops-state-check": "run the frozen Elo/feature engines over the workspace state twice and compare",
    "ops-reports": "render reports/operations/*.md from the last ops-run",
    "ops-readiness": "print the readiness verdict of the last ops-run",
    "ops-benchmark": "benchmark conversion, identity, reconciliation, compatibility and schedule normalization",
}


def register(sub) -> None:
    for name, helptext in COMMANDS.items():
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--operations-config", help="config/operations.toml by default")
        if name == "ops-run":
            p.add_argument("--as-of", help="evaluation date (YYYY-MM-DD; default: today, UTC)")
            p.add_argument("--no-probe", action="store_true", help="reuse the last access audit")
            p.add_argument("--no-fetch", action="store_true", help="do not download; use files already stored")
            p.add_argument("--no-reports", action="store_true")


def run(args) -> int:
    from tennis_data.operations.config import load_operations_config

    cfg = load_operations_config(args.operations_config)
    cmd = args.command
    if cmd == "ops-access":
        from tennis_data.operations.access import audit

        a = audit(cfg.providers)
        out = cfg.workspace_dir / "access"
        out.mkdir(parents=True, exist_ok=True)
        (out / "latest.json").write_text(json.dumps(a, indent=2, sort_keys=True) + "\n")
        for r in a["results"]:
            print(f"{r['provider']:20} {r['status']:18} {r['http_status'] or '-':>4}  {r['detail']}")
        return 0
    if cmd == "ops-run":
        from datetime import UTC, datetime

        from tennis_data.operations import pipeline, report

        as_of = date.fromisoformat(args.as_of) if args.as_of else datetime.now(UTC).date()
        r = pipeline.run(cfg, as_of, probe=not args.no_probe, fetch=not args.no_fetch)
        if not args.no_reports:
            sc = cfg.workspace_dir / "state_reconstruction.json"
            report.write_all(r["results"], r["meta"], cfg, json.loads(sc.read_text()) if sc.exists() else None)
        rd = r["results"]["readiness"]
        print(json.dumps({"results_sha256": r["results_sha256"], "verdict": rd["verdict"],
                          "failing_required": rd["failing_required"], "state": r["results"]["state"]}, indent=2))  # fmt: skip
        return 0
    if cmd == "ops-state-check":
        from tennis_data.operations.pipeline import reconstruct_state

        print(json.dumps(reconstruct_state(cfg), indent=2))
        return 0
    if cmd == "ops-reports":
        from tennis_data.operations import pipeline, report

        res, meta = pipeline.load_results(cfg)
        sc = cfg.workspace_dir / "state_reconstruction.json"
        for p in report.write_all(res, meta, cfg, json.loads(sc.read_text()) if sc.exists() else None):
            print(p)
        return 0
    if cmd == "ops-readiness":
        from tennis_data.operations.pipeline import load_results

        r = load_results(cfg)[0]["readiness"]
        print(json.dumps({k: r[k] for k in ("verdict", "failing_required", "proposed_configuration_change")}, indent=2))
        return 0 if r["verdict"] == "READY_TO_PROPOSE_ACTIVATION" else 1
    if cmd == "ops-benchmark":
        from tennis_data.operations.benchmark import run_benchmark

        print(json.dumps(run_benchmark(cfg), indent=2))
        return 0
    raise ValueError(cmd)
