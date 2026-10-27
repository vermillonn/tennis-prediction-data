"""Phase 5A benchmarks on the real TML snapshot (conversion, identity, reconciliation, compatibility, readiness)
and on synthetic fixtures (schedule normalization). Wall-clock timings are operational metadata."""

from __future__ import annotations

import json
import resource
import time
from datetime import UTC, date, datetime, timedelta

import polars as pl

from tennis_data.operations import tml
from tennis_data.operations.compatibility import study
from tennis_data.prospective.backfill import load_archive, parse_store
from tennis_data.prospective.identity import IdentityResolver, load_archive_players
from tennis_data.prospective.providers.fixture import FORMAT, FixtureProvider
from tennis_data.prospective.schema import check_matches


def _t(fn, reps=1):
    best = None
    out = None
    for _ in range(reps):
        s = time.perf_counter()
        out = fn()
        d = time.perf_counter() - s
        best = d if best is None else min(best, d)
    return out, round(best, 4)


def synthetic_fixtures(archive_players: pl.DataFrame, n: int) -> dict:
    """``n`` synthetic scheduled fixtures between real archive player names (SYNTHETIC; never ingested)."""
    ps = archive_players.filter(pl.col("birth_date").is_not_null()).head(2000).to_dicts()
    t0 = datetime(2030, 1, 7, 10, tzinfo=UTC)
    players = [{"id": f"s{i}", "first_name": p["first_name"], "last_name": p["last_name"], "country": p["country"],
                "birth_date": p["birth_date"].isoformat()} for i, p in enumerate(ps)]  # fmt: skip
    matches = [{"id": f"fx{i}", "tournament_id": "t1", "round": "R32", "best_of": 3, "status": "scheduled",
                "scheduled_start_at": (t0 + timedelta(minutes=10 * i)).isoformat(),
                "player1_id": f"s{(2 * i) % len(ps)}", "player2_id": f"s{(2 * i + 1) % len(ps)}"} for i in range(n)]  # fmt: skip
    return {"schema": FORMAT, "provider": "synthetic_bench", "generated_at": t0.isoformat(), "players": players,
            "tournaments": [{"id": "t1", "name": "Synthetic", "level": "A", "surface": "Hard",
                             "start_date": "2030-01-07"}], "matches": matches}  # fmt: skip


def run_benchmark(cfg) -> dict:
    ws = cfg.workspace_prospective()
    d = cfg.workspace_dir / "tml" / cfg.tml["commit"]
    files = {n: (d / n).read_bytes() for n in cfg.tml["files"]}
    (doc, log), t_conv = _t(lambda: tml.convert(files, cfg.tml["level_map"], "2026-01-01T00:00:00+00:00"), 3)
    parsed, t_parse = _t(lambda: parse_store(ws))
    m = parsed["matches"]
    _, t_check = _t(lambda: check_matches(m), 3)
    arch = load_archive_players(ws)
    res = IdentityResolver(ws, arch, root=cfg.workspace_dir / "bench_identity")
    dec, t_id = _t(lambda: res.resolve(parsed["players"]))
    hist, _ = load_archive(ws)
    _, t_comp = _t(lambda: study(m, dec, hist, cfg))
    sched = {}
    for n in (1000, 10000):
        fx = synthetic_fixtures(arch, n)
        (p, _m, f, _e), t_norm = _t(lambda fx=fx: FixtureProvider(None).parse(fx, "bench", datetime.now(UTC)))
        rr = IdentityResolver(ws, arch, root=cfg.workspace_dir / "bench_identity")
        _, t_res = _t(lambda p=p, rr=rr: rr.resolve(p))
        sched[str(n)] = {"fixtures": f.height, "normalize_seconds": t_norm, "identity_seconds": t_res,
                         "fixtures_per_second": round(f.height / max(t_norm, 1e-9))}  # fmt: skip
    from tennis_data.operations import readiness

    _, t_ready = _t(lambda: readiness.evaluate({"coverage": {"weeks_total": 21}, "gate": {}}, cfg.required_checks), 5)
    recs = sum(v["rows"] for v in log["files"].values())
    out = {"real_tml_snapshot": {"records": recs, "players": log["players"],
                                 "conversion_seconds": t_conv, "conversion_records_per_second": round(recs / t_conv),
                                 "parse_store_seconds": t_parse, "schema_validation_seconds": t_check,
                                 "identity_resolution_seconds": t_id, "compatibility_study_seconds": t_comp},
           "synthetic_schedule_normalization": sched, "readiness_evaluation_seconds": t_ready,
           "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
           "measured_on": date.today().isoformat()}  # fmt: skip
    (cfg.workspace_dir / "benchmark.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    return out
