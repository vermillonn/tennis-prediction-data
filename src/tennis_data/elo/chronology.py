"""Weekly batching: when a match is predicted, and when its result may update ratings.

The source has no match dates, only each event's start date (``tournament_date``). So:

- ``prediction_batch``: the start date of the calendar week (week_start weekday 00:00 to
  six days later 24:00) that contains ``tournament_date``. Every match of an event is predicted
  from the rating state frozen at the start of that batch.
- ``release_batch``: the batch whose close applies the event's results. For one-week events it is
  the prediction batch; for events that span N weeks (multi_week rules) it is N-1 batches later,
  so a result is never applied before the event can have finished.

Neither value is a match timestamp. They are conservative bounds derived from the event start date.
"""

from __future__ import annotations

import polars as pl

from tennis_data.elo.config import Chronology


def batch_start(date_col: str, chronology: Chronology) -> pl.Expr:
    # Polars weekday(): Monday = 1 ... Sunday = 7.
    offset = (pl.col(date_col).dt.weekday() - 1 - chronology.week_start_index) % 7
    return (pl.col(date_col) - pl.duration(days=offset)).alias("prediction_batch")


def event_weeks(chronology: Chronology) -> pl.Expr:
    weeks = pl.lit(1, dtype=pl.Int32)
    for rule in reversed(chronology.multi_week):
        cond = pl.col("tournament_level").is_in(list(rule.levels))
        if rule.min_draw_size is not None:
            cond = cond & (pl.col("draw_size") >= rule.min_draw_size).fill_null(False)
        weeks = pl.when(cond).then(pl.lit(rule.weeks, dtype=pl.Int32)).otherwise(weeks)
    return weeks.alias("event_weeks")


def add_batches(df: pl.DataFrame, chronology: Chronology) -> pl.DataFrame:
    return df.with_columns(batch_start("tournament_date", chronology), event_weeks(chronology)).with_columns(
        (pl.col("prediction_batch") + pl.duration(days=7 * (pl.col("event_weeks") - 1))).alias("release_batch")
    )
