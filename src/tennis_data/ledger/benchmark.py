"""Measured ledger performance on a synthetic ledger (SYNTHETIC predictions + results), with the system clock."""

from __future__ import annotations

import multiprocessing as mp
import statistics
import tempfile
import time
from dataclasses import replace
from pathlib import Path

from tennis_data.ledger.export import export
from tennis_data.ledger.verification import verify_ledger
from tennis_data.ledger.writer import LedgerWriter, PredictionRequest, SystemClock, orient


def _request(cfg, i: int) -> PredictionRequest:
    mid = f"synthetic:bench-{i:06d}"
    o = orient(mid, 100000 + i % 500, 200000 + i % 700, 0.3 + (i % 400) / 1000, 0.5)
    return PredictionRequest(
        mode="SYNTHETIC", forecast_slot="pre_match", match_id=mid, tournament_id="bench", surface="Hard", round="R32",
        best_of=3, scheduled_start_at="2030-01-01T12:00:00.000000Z", scheduled_start_verified=True,
        schedule_snapshot_id="bench-sched", schedule_retrieved_at="2026-11-01T00:00:00.000000Z", model_id="M4A+raw",
        model_version="m3b-1+engine1", frozen_hash=cfg.prospective.expected_frozen_hash + "0" * 48,
        configuration_hash=cfg.prospective.modeling.configuration_hash(), feature_schema_version="sr-1+engine2",
        state_snapshot_id="bench-state", feature_state_hash=f"fs-{i}", source_snapshots={"bench": "1"},
        prediction_created_at="2026-11-01T00:00:00.000000Z", prediction_cutoff=None,
        feature_availability={"filled_inputs": []}, freshness_status="unknown", inference_engine_version="bench",
        git_commit=None, **o,
    )  # fmt: skip


def _worker(cfg, start, n, barrier, q):
    w = LedgerWriter(cfg, SystemClock())
    barrier.wait()
    t = time.perf_counter()
    for i in range(start, start + n):
        w.commit_prediction(_request(cfg, i))
    q.put(time.perf_counter() - t)
    w.close()


def run_benchmark(cfg, n_events: int = 10000, out: Path | None = None) -> dict:
    tmp = Path(out) if out else Path(tempfile.mkdtemp(prefix="ledger-bench-"))
    tmp.mkdir(parents=True, exist_ok=True)
    path = tmp / "bench.sqlite"
    if path.exists():
        raise FileExistsError(path)
    bcfg = replace(cfg, ledger_path=path)
    w = LedgerWriter(bcfg, SystemClock())
    w.init("ledger-benchmark")
    lat = []
    n_pred = n_events * 4 // 5
    t0 = time.perf_counter()
    for i in range(n_pred):
        t = time.perf_counter()
        w.commit_prediction(_request(bcfg, i))
        lat.append(time.perf_counter() - t)
    for i in range(n_events - n_pred):  # results for some of the matches
        mid = f"synthetic:bench-{i:06d}"
        w.record("MATCH_RESULT_RECORDED", mid, {"status": "completed", "winner_id": 100000 + i % 500, "score": "6-4 6-4",
                                                "source": {"provider": "bench"}, "supersedes_event_id": None,
                                                "correction_reason": None})  # fmt: skip
    append_total = time.perf_counter() - t0
    w.close()
    t = time.perf_counter()
    rep = verify_ledger(path)
    verify_s = time.perf_counter() - t
    t = time.perf_counter()
    export(path, tmp / "export")
    export_s = time.perf_counter() - t
    size = sum(p.stat().st_size for p in tmp.glob("bench.sqlite*"))
    # concurrent writers on a fresh ledger
    cpath = tmp / "concurrent.sqlite"
    ccfg = replace(cfg, ledger_path=cpath)
    LedgerWriter(ccfg, SystemClock()).init("ledger-benchmark-concurrent")
    ctx = mp.get_context("spawn")
    workers, per = 4, 250
    barrier, q = ctx.Barrier(workers), ctx.Queue()
    t = time.perf_counter()
    ps = [ctx.Process(target=_worker, args=(ccfg, k * per, per, barrier, q)) for k in range(workers)]
    for p in ps:
        p.start()
    for p in ps:
        p.join()
    conc_wall = time.perf_counter() - t
    crep = verify_ledger(cpath)
    lat_ms = sorted(x * 1000 for x in lat)
    return {
        "events": rep.events, "verified": rep.ok,
        "append_latency_ms": {"median": round(statistics.median(lat_ms), 3),
                              "p95": round(lat_ms[int(0.95 * (len(lat_ms) - 1))], 3), "max": round(lat_ms[-1], 3)},
        "append_throughput_events_per_s": round(n_events / append_total, 1),
        "verify_seconds": round(verify_s, 3), "export_seconds": round(export_s, 3),
        "storage_bytes_per_event": round(size / max(1, rep.events), 1),
        "concurrent": {"writers": workers, "events": crep.events, "wall_seconds": round(conc_wall, 2),
                       "chain_valid": crep.ok, "exit_codes": [p.exitcode for p in ps]},
        "durability": "WAL, synchronous=FULL",
    }  # fmt: skip
