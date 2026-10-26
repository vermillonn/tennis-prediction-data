"""Measured evaluation performance on SYNTHETIC demo ledgers of increasing size (stage timings and peak memory).

Ledgers are written through the real ``LedgerWriter`` (``demo.build_demo``); only the evaluation stages are timed.
"""

from __future__ import annotations

import json
import math
import resource
import tempfile
import time
import tracemalloc
from pathlib import Path

from tennis_data.evaluation_prospective import anchoring, demo
from tennis_data.evaluation_prospective.calibration import calibration_tables, subgroup_tables
from tennis_data.evaluation_prospective.eligibility import classify
from tennis_data.evaluation_prospective.integrity import verify_snapshot
from tennis_data.evaluation_prospective.metrics import score
from tennis_data.evaluation_prospective.outcomes import load_outcomes, reconcile_all
from tennis_data.evaluation_prospective.pipeline import evaluate, results_bytes
from tennis_data.evaluation_prospective.report import render_all
from tennis_data.evaluation_prospective.snapshot import snapshot_from_ledger
from tennis_data.evaluation_prospective.uncertainty import paired_uncertainty
from tennis_data.ledger.hashing import sha256_hex
from tennis_data.ledger.lifecycle import replay


def _timed(fn):
    t = time.perf_counter()
    out = fn()
    return out, round(time.perf_counter() - t, 4)


def run_one(cfg, n: int, root: Path) -> dict:
    weeks = max(4, min(70, n // 10))
    per_week = max(2, math.ceil(n / 1.2 / weeks))
    d, build_s = _timed(lambda: demo.build_demo(cfg, root, weeks=weeks, per_week=per_week,
                                                replay_matches=max(2, n // 25)))  # fmt: skip
    snap = snapshot_from_ledger(d["ledger"], root / "snapshots", cfg.limits)
    oc = load_outcomes(d["outcomes"], cfg.limits)
    prot = demo.demo_protocol(cfg)
    anchors = anchoring.load_records(d["evidence"], cfg.limits.max_receipt_bytes)
    trusted = anchoring.load_trusted(d["trusted"], cfg.limits.max_receipt_bytes)
    integ, t_integ = _timed(lambda: verify_snapshot(snap))
    cov = anchoring.coverage([{"source": nm, **anchoring.verify_anchor(r, snap, evidence_dir=d["evidence"],
                                                                      fixture_root=d["fixture_root"])}
                              for nm, r in anchors], snap)  # fmt: skip
    parsed = [{**e, "payload": json.loads(e["payload"])} for e in snap.events]

    def reconcile():
        states = replay(parsed)
        by: dict = {}
        players = {}
        for e in parsed:
            if e["match_id"] is not None:
                by.setdefault(e["match_id"], []).append(e)
            if e["event_type"] == "PREDICTION_COMMITTED":
                m = e["payload"]["match"]
                players[e["match_id"]] = frozenset((m["player_a_id"], m["player_b_id"]))
        return reconcile_all(states, by, oc, players, None)

    _, t_rec = _timed(reconcile)
    cls, t_cls = _timed(lambda: classify(list(snap.events), prot, oc, integrity_ok=integ["ok"], evidence_ok=True,
                                         anchor_cov=cov))  # fmt: skip
    rows = [r for r in cls["rows"] if r["cohort"] in ("primary", "operational")]

    def metrics():
        return (score(rows, prot), calibration_tables(rows, prot["calibration_bins"]), subgroup_tables(rows, prot),
                paired_uncertainty(rows, prot))  # fmt: skip

    _, t_met = _timed(metrics)

    def run():
        return evaluate(snap, oc, prot, anchor_records=anchors, trusted=trusted, evidence_dir=d["evidence"],
                        fixture_root=d["fixture_root"])  # fmt: skip

    res, t_eval = _timed(run)
    _, t_rep = _timed(lambda: render_all(res, {}))
    tracemalloc.start()  # separate run: tracing slows Python down several-fold, so it is not timed
    res2 = run()
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return {"predictions": res["coverage"]["total_predictions"], "events": len(snap.events),
            "scorable": res["scores"]["operational"]["n"], "ledger_build_seconds": round(build_s, 2),
            "seconds": {"integrity_verification": t_integ, "outcome_reconciliation": t_rec,
                        "eligibility_classification_incl_reconciliation": t_cls, "metrics_calibration_uncertainty": t_met,
                        "report_generation": t_rep, "full_evaluate": t_eval},
            "peak_python_alloc_mb_full_evaluate": round(peak / 2**20, 1),
            "deterministic": sha256_hex(results_bytes(res)) == sha256_hex(results_bytes(res2))}  # fmt: skip


def run_benchmark(cfg, sizes: list[int], out: Path | None = None) -> dict:
    base = Path(out) if out else Path(tempfile.mkdtemp(prefix="eval-bench-"))
    runs = {str(n): run_one(cfg, n, base / f"n{n}") for n in sizes}
    return {"runs": runs, "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
            "note": "SYNTHETIC ledgers; timings exclude ledger construction; bootstrap "
                    f"{cfg.protocol['bootstrap_samples']} samples"}  # fmt: skip
