"""Versioned, provider-neutral record schemas (``provider-match-1``).

Provider adapters map their responses onto these frames. Provider-specific fields stay in the immutable raw
snapshot (``raw_sha256`` links a record to it) and never enter the canonical match schema.

Time fields are timezone-aware UTC and null when the provider does not supply them. Nothing is inferred:

- ``event_start_date``: tournament (event) start date, the only date the historical archive has.
- ``event_start_at``: match start time, only if the provider states it.
- ``event_completed_at``: match end time, only if the provider states it.
- ``source_published_at``: when the provider says it published this version of the record.
- ``retrieved_at``: when this system downloaded or imported the record (the ``ingested_at`` of the ledger).
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from tennis_data.prospective.config import SCHEMA_VERSION

UTC = pl.Datetime("us", "UTC")
STATS = ("svpt", "1stIn", "1stWon", "2ndWon", "ace", "df", "SvGms", "bpSaved", "bpFaced")
RESULT_STATUSES = ("completed", "retired", "walkover", "default", "abandoned", "unknown")
FIXTURE_STATUSES = ("scheduled", "postponed", "cancelled")
LEVELS = ("G", "M", "A", "F", "D", "O")  # Phase 1 tournament_level codes in the atp_tour pool
# Counting convention of serve statistics. Only "sackmann_v1" (svpt = service points; 1stWon + 2ndWon = service
# points won) is known to be compatible with the Phase 3A estimators; anything else is excluded from statistics.
COMPATIBLE_STAT_DEFINITIONS = ("sackmann_v1",)

PLAYER_SCHEMA = {
    "provider": pl.String, "provider_player_id": pl.String, "first_name": pl.String, "last_name": pl.String,
    "country": pl.String, "birth_date": pl.Date, "hand": pl.String, "declared_canonical_id": pl.Int64,
    "raw_sha256": pl.String, "retrieved_at": UTC,
}  # fmt: skip

_COMMON = {
    "provider": pl.String, "provider_match_id": pl.String, "source_version": pl.Int64,
    "provider_tournament_id": pl.String, "tournament_name": pl.String, "tournament_level": pl.String,
    "surface": pl.String, "draw_size": pl.Int64, "event_start_date": pl.Date, "event_start_at": UTC,
    "round": pl.String, "best_of": pl.Int64,
}  # fmt: skip

MATCH_SCHEMA = {
    **_COMMON,
    "event_completed_at": UTC, "source_published_at": UTC, "status": pl.String, "score": pl.String,
    "winner_provider_id": pl.String, "loser_provider_id": pl.String, "minutes": pl.Int64,
    **{f"{side}_{s}": pl.Int64 for side in ("w", "l") for s in STATS},
    "stats_definition": pl.String, "raw_sha256": pl.String, "retrieved_at": UTC,
}  # fmt: skip

FIXTURE_SCHEMA = {
    **_COMMON,
    "status": pl.String, "player1_provider_id": pl.String, "player2_provider_id": pl.String,
    "raw_sha256": pl.String, "retrieved_at": UTC,
}  # fmt: skip


@dataclass(frozen=True)
class RecordError:
    provider: str
    record_id: str | None
    field: str
    code: str
    message: str


def frame(rows: list[dict], schema: dict) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=schema, orient="row") if rows else pl.DataFrame(schema=schema)


def check_matches(df: pl.DataFrame) -> list[RecordError]:
    """Structural validation before identity/reconciliation. Value-level rules are Phase 1's."""
    errs: list[RecordError] = []
    if list(df.schema.items()) != list(MATCH_SCHEMA.items()):
        raise ValueError(f"match frame does not follow {SCHEMA_VERSION}")
    for r in df.iter_rows(named=True):
        rid, prov = r["provider_match_id"], r["provider"]

        def err(fieldname, code, msg, rid=rid, prov=prov):
            errs.append(RecordError(prov, rid, fieldname, code, msg))

        for k in ("provider_match_id", "provider_tournament_id", "event_start_date", "round", "status"):
            if r[k] is None:
                err(k, "missing", f"{k} is required")
        if r["status"] is not None and r["status"] not in RESULT_STATUSES:
            err("status", "invalid_status", f"result status {r['status']!r} not in {RESULT_STATUSES}")
        if r["winner_provider_id"] is None or r["loser_provider_id"] is None:
            err("winner_provider_id", "missing_players", "winner and loser are required for a result")
        elif r["winner_provider_id"] == r["loser_provider_id"]:
            err("winner_provider_id", "identical_players", "winner equals loser")
        if r["tournament_level"] is not None and r["tournament_level"] not in LEVELS:
            err("tournament_level", "unmapped_level", f"level {r['tournament_level']!r} is not an ATP tour level")
        if r["source_version"] is None or r["source_version"] < 1:
            err("source_version", "invalid_version", "source_version must be >= 1")
        for side in ("w", "l"):
            v = {s: r[f"{side}_{s}"] for s in STATS}
            if any(x is not None and x < 0 for x in v.values()):
                err(f"{side}_stats", "negative_stat", "negative serve statistic")
            if v["svpt"] is not None and v["1stIn"] is not None and v["1stIn"] > v["svpt"]:
                err(f"{side}_1stIn", "first_in_exceeds_points", "first serves in > service points")
            if None not in (v["svpt"], v["1stWon"], v["2ndWon"]) and v["1stWon"] + v["2ndWon"] > v["svpt"]:
                err(f"{side}_svpt", "won_exceeds_played", "service points won > service points")
        if r["event_completed_at"] is not None and r["event_start_at"] is not None:
            if r["event_completed_at"] < r["event_start_at"]:
                err("event_completed_at", "completed_before_start", "completion precedes start")
    return errs
