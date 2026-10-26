"""Reproducible historical backfill: raw snapshots -> identity -> reconciliation -> Phase 1 validation -> ledger.

The backfill never touches the pinned archive or any Phase 1-3B output. It writes the accepted new rows and their
audit tables under ``data/prospective/backfill``. The state snapshot for inference is archive + accepted rows.

Chronology (``weekly_batch_v1``, the frozen Phase 2 policy): a backfilled result becomes state-eligible at the close
of its release batch, i.e. at 00:00 UTC on the Saturday after the release batch. That time is *inferred* from the
event start date (``eligibility_basis = inferred_weekly_batch``). Provider timestamps are stored but not used to
release results earlier; no timestamp-based policy is enabled in Phase 4A.

Corrections: a new ``source_version`` supersedes the old one (both kept in ``versions``). Because the state is a
pure function of the accepted table, the corrected state is a full deterministic replay; ``replay_from_batch``
records the earliest batch whose state can differ (the earliest changed, added or removed row).
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime, timedelta
from datetime import time as dtime

import polars as pl

from tennis_data.features.observations import STAT_INPUT_COLUMNS, feature_frame
from tennis_data.pipeline import validate_frame
from tennis_data.prospective.config import SCHEMA_VERSION, STATE_ENGINE_VERSION, ProspectiveConfig
from tennis_data.prospective.identity import IdentityResolver
from tennis_data.prospective.providers.fixture import FORMAT as FIXTURE_FORMAT
from tennis_data.prospective.providers.fixture import FixtureProvider
from tennis_data.prospective.providers.raw_store import RawStore
from tennis_data.prospective.providers.sackmann_csv import SackmannCsvProvider
from tennis_data.prospective.reconciliation import reconcile, to_raw_rows
from tennis_data.prospective.schema import MATCH_SCHEMA, PLAYER_SCHEMA
from tennis_data.prospective.state import StateSnapshot, make_snapshot, table_hash
from tennis_data.storage.parquet import write_single
from tennis_data.validation.reference import load_players

PREDICTION_MODE = "historical_reconstruction"


def raw_store(cfg: ProspectiveConfig) -> RawStore:
    return RawStore(cfg.prospective_dir / "raw")


def ingest_file(cfg: ProspectiveConfig, path, fmt: str, retrieved_at: datetime | None = None) -> dict:
    """Store one authorized file immutably (idempotent). Parsing happens in ``run_backfill`` from the store."""
    store = raw_store(cfg)
    content = open(path, "rb").read()
    if fmt == FIXTURE_FORMAT:
        doc = json.loads(content)
        provider = doc["provider"]
        if retrieved_at is None and doc.get("generated_at"):
            retrieved_at = datetime.fromisoformat(doc["generated_at"])
    elif fmt == "sackmann_csv":
        provider = "sackmann_csv"
    else:
        raise ValueError(f"unknown import format {fmt!r}")
    rec = store.put(provider, "file", {"name": str(path).rsplit("/", 1)[-1], "format": fmt}, content,
                    retrieved_at or datetime.now(UTC))  # fmt: skip
    return {"provider": rec.provider, "sha256": rec.sha256, "retrieved_at": rec.retrieved_at, "bytes": rec.bytes}


def parse_store(cfg: ProspectiveConfig):
    """Re-parse every stored snapshot (so the result depends only on the store's content, not on run history)."""
    store = raw_store(cfg)
    players, matches, fixtures, errors, used = [], [], [], [], []
    for rec in store.records():
        fmt = rec.params.get("format")
        content = store.get(rec.provider, rec.sha256)
        retrieved = datetime.fromisoformat(rec.retrieved_at)
        if fmt == FIXTURE_FORMAT:
            p, m, f, e = FixtureProvider(store).parse(json.loads(content), rec.sha256, retrieved)
            fixtures.append(f)
        elif fmt == "sackmann_csv":
            p, m, _, e = SackmannCsvProvider(store).parse(
                pl.read_csv(content, infer_schema=False), rec.sha256, retrieved
            )
        else:
            errors.append({"provider": rec.provider, "record_id": rec.sha256, "field": "format", "code": "unparsed",
                           "message": f"no verified parser for format {fmt!r}"})  # fmt: skip
            continue
        players.append(p)
        matches.append(m)
        errors += [e.__dict__ for e in e]
        used.append(f"{rec.provider}:{rec.sha256}")
    cat = lambda fs, schema: pl.concat(fs, how="vertical") if fs else pl.DataFrame(schema=schema)  # noqa: E731
    from tennis_data.prospective.schema import FIXTURE_SCHEMA

    return {"players": cat(players, PLAYER_SCHEMA), "matches": cat(matches, MATCH_SCHEMA),
            "fixtures": cat(fixtures, FIXTURE_SCHEMA), "errors": errors, "snapshots": sorted(used),
            "snapshot_id": store.snapshot_id()}  # fmt: skip


def load_archive(cfg: ProspectiveConfig) -> tuple[pl.DataFrame, pl.Series]:
    paths = cfg.data.paths
    files = sorted((paths.validated_dir / "matches").rglob("*.parquet"))
    hist = pl.concat([pl.read_parquet(f) for f in files], how="vertical")
    q = paths.quarantine_dir / "matches.parquet"
    qids = pl.read_parquet(q, columns=["match_id"])["match_id"] if q.exists() else pl.Series([], dtype=pl.String)
    return hist, qids


def _state_eligible_at(release_batch) -> datetime:
    return datetime.combine(release_batch + timedelta(days=7), dtime(0, 0), tzinfo=UTC)


def ledger(new_validated: pl.DataFrame, provenance: pl.DataFrame, cfg: ProspectiveConfig) -> pl.DataFrame:
    if new_validated.height == 0:
        return pl.DataFrame()
    ff = feature_frame(new_validated.select(list(STAT_INPUT_COLUMNS)), cfg.features)
    led = ff.select("match_id", "tournament_id", "tournament_date", "prediction_batch", "release_batch", "event_weeks",
                    "pool", "is_target", "is_update", "exclusion_reason", "stats_eligible", "stats_exclusion_reason",
                    "match_status", "stats_status")  # fmt: skip
    led = led.join(provenance, on="match_id", how="left", validate="1:1")
    hs, ps = cfg.historical_snapshot_end, cfg.prospective_start
    return led.with_columns(
        pl.col("release_batch").map_elements(_state_eligible_at, return_dtype=MATCH_SCHEMA["retrieved_at"])
        .alias("state_eligible_at"),
        pl.lit("inferred_weekly_batch").alias("eligibility_basis"),
        pl.lit(PREDICTION_MODE).alias("prediction_mode"),
        pl.when(pl.col("tournament_date") <= hs).then(pl.lit("overlaps_archive_period"))
        .when(pl.col("tournament_date") <= ps).then(pl.lit("backfill_gap"))
        .otherwise(pl.lit("after_prospective_start")).alias("period"),
    ).sort("prediction_batch", "match_id")  # fmt: skip


def replay_from(prev: pl.DataFrame | None, cur: pl.DataFrame) -> str | None:
    """Earliest prediction batch whose state can change between two accepted tables (None if identical)."""

    def keyed(df):
        if df is None or df.height == 0:
            return {}
        return {r["match_id"]: (r["prediction_batch"], json.dumps(r, default=str, sort_keys=True))
                for r in df.select(list(STAT_INPUT_COLUMNS) + ["prediction_batch"]).iter_rows(named=True)}  # fmt: skip

    a, b = keyed(prev), keyed(cur)
    changed = [v[0] for k, v in a.items() if k not in b or b[k][1] != v[1]] + [v[0] for k, v in b.items() if k not in a]
    return str(min(changed)) if changed else None


def run_backfill(cfg: ProspectiveConfig, write: bool = True) -> dict:
    t0 = time.perf_counter()
    parsed = parse_store(cfg)
    hist, qids = load_archive(cfg)
    resolver = IdentityResolver(cfg)
    decisions = resolver.resolve(parsed["players"])
    rec = reconcile(parsed["matches"], decisions, hist, cfg)
    names = {r["player_id"]: f"{r['first_name'] or ''} {r['last_name'] or ''}".strip()
             for r in resolver.archive.iter_rows(named=True)}  # fmt: skip
    names.update({r["canonical_player_id"]: f"{r['first_name'] or ''} {r['last_name'] or ''}".strip()
                  for r in resolver._pending_registry.iter_rows(named=True)})  # fmt: skip
    raw_rows, provenance = to_raw_rows(rec["accepted"], names, cfg)
    clash = raw_rows.filter(pl.col("tourney_id").is_in(hist["tournament_id"].unique().implode())).height
    if raw_rows.height:
        validated, p1_quarantine = validate_frame(raw_rows, cfg.data, load_players(cfg.data))
        collide = validated.filter(pl.col("match_id").is_in(hist["match_id"].implode()))
        if collide.height:
            raise AssertionError(f"backfill match ids collide with the archive: {collide['match_id'].to_list()[:3]}")
        validated = validated.with_columns(pl.col("season").cast(pl.Int32))
    else:
        validated = hist.head(0)
        p1_quarantine = pl.DataFrame()
    led = ledger(validated, provenance, cfg)
    led_e = led if led.height else None
    out_dir = cfg.prospective_dir / "backfill"
    prev_path = out_dir / "matches.parquet"
    prev = pl.read_parquet(prev_path) if prev_path.exists() else None
    if prev is not None and prev.height:
        prev = prev.join(pl.read_parquet(out_dir / "ledger.parquet").select("match_id", "prediction_batch"), on="match_id",
                         how="left")  # fmt: skip
    cur = validated.join(led.select("match_id", "prediction_batch"), on="match_id", how="left") if led.height else None
    combined = pl.concat([hist.select(list(STAT_INPUT_COLUMNS)), validated.select(list(STAT_INPUT_COLUMNS))])
    snapshot = make_snapshot(combined, qids, {"archive_commit": cfg.data.source.commit,
                                              "backfill_raw_snapshot": parsed["snapshot_id"]})  # fmt: skip
    summary = {
        "schema_version": SCHEMA_VERSION, "state_engine_version": STATE_ENGINE_VERSION,
        "configuration_hash": cfg.configuration_hash(), "chronology_policy": cfg.chronology_policy,
        "raw_snapshot_id": parsed["snapshot_id"], "raw_snapshots": parsed["snapshots"],
        "provider_records": parsed["matches"].height, "provider_players": parsed["players"].height,
        "provider_fixtures": parsed["fixtures"].height, "parse_errors": len(parsed["errors"]),
        "identity": decisions.group_by("category", "usable").len().sort("category", "usable").to_dicts(),
        "reconciliation_quarantine": rec["quarantine"].group_by("reason").len().sort("reason").to_dicts(),
        "reconciliation_notes": rec["notes"].group_by("reason").len().sort("reason").to_dicts(),
        "accepted_records": len(rec["accepted"]),
        "rows_in_archive_tournaments": clash,
        "phase1_validated": validated.height, "phase1_quarantined": p1_quarantine.height,
        "with_complete_stats": int(validated.filter(pl.col("stats_status") == "complete").height),
        "state_eligible_updates": int(led["is_update"].sum()) if led.height else 0,
        "stats_eligible": int(led["stats_eligible"].sum()) if led.height else 0,
        "by_period": led.group_by("period").len().sort("period").to_dicts() if led.height else [],
        "state_snapshot_id": snapshot.snapshot_id, "state_components": snapshot.components,
        "accepted_table_hash": table_hash(validated.select(list(STAT_INPUT_COLUMNS))) if validated.height else None,
    }  # fmt: skip
    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        resolver.persist(decisions)
        write_single(validated, out_dir / "matches.parquet")
        for name, df in (("phase1_quarantine", p1_quarantine), ("reconciliation_quarantine", rec["quarantine"]),
                         ("reconciliation_notes", rec["notes"]), ("versions", rec["versions"]),
                         ("provenance", provenance), ("ledger", led), ("identity_decisions", decisions)):  # fmt: skip
            write_single(df if df.height or df.width else pl.DataFrame({"empty": []}), out_dir / f"{name}.parquet")
        (out_dir / "backfill_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n"
        )
    # Relative to the previous run's accepted table, so it is run history (operational), not part of the summary.
    replay_batch = replay_from(prev, cur)
    if write:
        (out_dir / "performance.json").write_text(json.dumps({"seconds": round(time.perf_counter() - t0, 2),
                                                              "replay_from_batch": replay_batch}) + "\n")  # fmt: skip
    return {"summary": summary, "replay_from_batch": replay_batch, "validated": validated, "ledger": led_e, "decisions": decisions, "snapshot": snapshot,
            "reconciliation": rec, "parsed": parsed, "phase1_quarantine": p1_quarantine}  # fmt: skip


def load_state_snapshot(cfg: ProspectiveConfig) -> StateSnapshot:
    """Checkpoint restoration: archive + the persisted accepted backfill rows."""
    hist, qids = load_archive(cfg)
    path = cfg.prospective_dir / "backfill" / "matches.parquet"
    summary_path = cfg.prospective_dir / "backfill" / "backfill_summary.json"
    parts = [hist.select(list(STAT_INPUT_COLUMNS))]
    comp = {"archive_commit": cfg.data.source.commit}
    if path.exists():
        acc = pl.read_parquet(path)
        if acc.height and "match_id" in acc.columns:
            parts.append(acc.select(list(STAT_INPUT_COLUMNS)))
        comp["backfill_raw_snapshot"] = json.loads(summary_path.read_text())["raw_snapshot_id"]
    snap = make_snapshot(pl.concat(parts), qids, comp)
    if path.exists():
        expected = json.loads(summary_path.read_text())["state_snapshot_id"]
        if snap.snapshot_id != expected:
            raise AssertionError(f"restored state {snap.snapshot_id} != recorded {expected}")
    return snap
