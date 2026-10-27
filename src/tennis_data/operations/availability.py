"""Prospective state-availability safeguard ``state_availability_v1`` (proposed, separately versioned, NOT active).

The frozen weekly batching (``weekly_batch_v1``) releases an event's results at the close of its release batch. It
is an approximation built for event start dates: with per-match dates, some late-round results (weekend finals,
Davis Cup Saturday rubbers, events starting mid-week) were played on or after the day the frozen policy makes them
state-eligible. For the archive and every historical evaluation this policy stays as it is. For *prospective*
forecasts this module defines a stricter rule: a result may enter the state used by a forecast created at time
``T`` only if it was known to be available before ``T``::

    available_at = max(frozen state_eligible_at, ingested_at, plausible_end_at)

- ``state_eligible_at`` (frozen): the safeguard is never earlier than the frozen policy;
- ``ingested_at``: when the result was actually held. Without it there is no evidence: the row is never available
  (``no_availability_evidence``);
- ``plausible_end_at``: 00:00 UTC ``match_day_buffer_days`` after the match date, or, without a match date,
  ``undated_after_release_batch_days`` after the start of the release batch;
- an ingestion time more than ``ingest_before_match_tolerance_days`` before the match date is impossible and
  excludes the row (``ingested_before_match_date``).

A row is usable for a forecast created at ``T`` if ``available_at <= T``. The policy hash covers every parameter.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import polars as pl

from tennis_data.config import PROJECT_ROOT

DEFAULT_PATH = PROJECT_ROOT / "config" / "availability.toml"
UTC_TS = pl.Datetime("us", "UTC")


@dataclass(frozen=True)
class AvailabilityPolicy:
    version: str = "state_availability_v1"
    enabled: bool = False
    match_day_buffer_days: int = 2
    undated_after_release_batch_days: int = 21
    ingest_before_match_tolerance_days: int = 1

    def policy_hash(self) -> str:
        payload = {k: v for k, v in asdict(self).items() if k != "enabled"}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def load_policy(path: str | Path | None = None) -> AvailabilityPolicy:
    with Path(path or DEFAULT_PATH).open("rb") as fh:
        d = tomllib.load(fh)["policy"]
    p = AvailabilityPolicy(**d)
    if p.version != "state_availability_v1":
        raise ValueError(f"unknown availability policy {p.version!r}")
    if min(p.match_day_buffer_days, p.undated_after_release_batch_days) < 1 or p.ingest_before_match_tolerance_days < 0:
        raise ValueError("availability margins must be positive")
    return p


def _midnight(col: pl.Expr, days: int) -> pl.Expr:
    return (col.cast(pl.Datetime("us")) + pl.duration(days=days)).dt.replace_time_zone("UTC")


def annotate(df: pl.DataFrame, policy: AvailabilityPolicy) -> pl.DataFrame:
    """Add ``plausible_end_at``, ``availability_status`` and ``available_at`` (null unless the status is ``ok``).

    Required columns: ``state_eligible_at`` (UTC), ``release_batch`` (date), ``match_date`` (date, nullable),
    ``ingested_at`` (UTC, nullable).
    """
    plausible = (
        pl.when(pl.col("match_date").is_not_null())
        .then(_midnight(pl.col("match_date"), policy.match_day_buffer_days))
        .otherwise(_midnight(pl.col("release_batch"), policy.undated_after_release_batch_days))
    )
    early = _midnight(pl.col("match_date"), -policy.ingest_before_match_tolerance_days)
    status = (pl.when(pl.col("ingested_at").is_null()).then(pl.lit("no_availability_evidence"))
              .when(pl.col("match_date").is_not_null() & (pl.col("ingested_at") < early))
              .then(pl.lit("ingested_before_match_date"))
              .otherwise(pl.lit("ok")))  # fmt: skip
    out = df.with_columns(plausible.cast(UTC_TS).alias("plausible_end_at"), status.alias("availability_status"))
    latest = pl.max_horizontal(pl.col("state_eligible_at").cast(UTC_TS), pl.col("ingested_at").cast(UTC_TS),
                               pl.col("plausible_end_at"))  # fmt: skip
    return out.with_columns(
        pl.when(pl.col("availability_status") == "ok").then(latest).otherwise(None).cast(UTC_TS).alias("available_at")
    )


def usable(df: pl.DataFrame, forecast_created_at: datetime) -> pl.Series:
    """Rows (already annotated) whose result may enter the state of a forecast created at ``forecast_created_at``."""
    if forecast_created_at.tzinfo is None:
        raise ValueError("forecast_created_at must be timezone-aware")
    return (pl.col("available_at").is_not_null() & (pl.col("available_at") <= forecast_created_at)).alias("usable")


def evaluate(df: pl.DataFrame, policy: AvailabilityPolicy, forecast_times: list[datetime]) -> dict:
    """Effect of the safeguard on annotated rows: delays against the frozen policy and usability at given times."""
    a = annotate(df, policy)
    ok = a.filter(pl.col("availability_status") == "ok")
    delay = ok.select(((pl.col("available_at") - pl.col("state_eligible_at").cast(UTC_TS)).dt.total_hours() / 24)
                      .alias("d"))["d"]  # fmt: skip
    played_late = ok.filter(pl.col("match_date").is_not_null()
                            & (pl.col("match_date") >= pl.col("state_eligible_at").dt.date()))  # fmt: skip
    return {"policy": asdict(policy), "policy_hash": policy.policy_hash(), "rows": a.height,
            "status": dict(sorted(a.group_by("availability_status").len().iter_rows())),
            "never_earlier_than_frozen": bool(ok.filter(pl.col("available_at") < pl.col("state_eligible_at")).height
                                              == 0),
            "delayed_vs_frozen": int((delay > 0).sum()), "max_delay_days": float(delay.max() or 0),
            "frozen_played_on_or_after_release": played_late.height,
            "safeguard_played_on_or_after_available": int(ok.filter(pl.col("match_date").is_not_null() & (
                pl.col("match_date").cast(pl.Datetime("us")).dt.replace_time_zone("UTC") >= pl.col("available_at")))
                .height),
            "usable_at": {t.isoformat(): int(a.select(usable(a, t)).to_series().sum()) for t in forecast_times},
            "latest_available_at": str(ok["available_at"].max()) if ok.height else None}  # fmt: skip


def as_known_at(df: pl.DataFrame, policy: AvailabilityPolicy, forecast_created_at: datetime) -> pl.DataFrame:
    """The state input of a forecast created at ``forecast_created_at``: per match, the latest source version that
    was available by then. A correction ingested later never reaches an earlier forecast; a match whose every
    version is unavailable is absent. Requires ``match_id`` and ``source_version`` besides the ``annotate`` columns."""
    a = annotate(df, policy)
    known = a.filter(usable(a, forecast_created_at))
    return known.sort("match_id", "source_version").unique("match_id", keep="last", maintain_order=True)
