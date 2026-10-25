"""Load raw match files and map them onto the canonical schema.

Every raw value is read as a string. Casts are non-strict, and every cell that
is present in the source but fails to parse is recorded as a parse issue rather
than silently becoming null. No value is imputed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from tennis_data.config import Config
from tennis_data.ingestion.fetch import RawIntegrityError, sha256_file
from tennis_data.ingestion.manifest import load_manifest, manifest_path
from tennis_data.ingestion.sources import selected_files
from tennis_data.normalization.ordering import assign_players
from tennis_data.normalization.schema import (
    CANONICAL_COLUMNS,
    PLAYER_FIELDS,
    REQUIRED_SOURCE_VALUES,
    ROUND_ORDER,
    SOURCE_MATCH_COLUMNS,
    STAT_COLUMNS,
    STAT_FIELDS,
)
from tennis_data.normalization.score import match_status

# Helper columns carried alongside the canonical ones until the final split.
ERRORS = "_errors"
WARNINGS = "_warnings"
RAW_RECORD = "_raw_record"
LOSER_ID = "_loser_id"
CONTENT_KEY = "_content_key"  # all source fields except match_num, for duplicate detection

# Source columns whose parse failure makes the row unusable (-> quarantine);
# failures elsewhere are warnings and the value is left null.
CRITICAL_NUMERIC = {"winner_id", "loser_id", "match_num", "best_of"} | {
    f"{p}_{s}" for p in "wl" for s in STAT_FIELDS.values()
}


class SourceSchemaError(ValueError):
    """A raw file does not have the expected columns."""


class MissingRawDataError(FileNotFoundError):
    """Selected raw files have not been ingested."""


@dataclass(frozen=True)
class RawMatchFile:
    path: Path
    rel_path: str
    family: str
    season: int
    sha256: str


def raw_match_files(config: Config) -> list[RawMatchFile]:
    """The selected match files, verified against the manifest."""
    manifest = load_manifest(manifest_path(config))
    if manifest is None:
        raise MissingRawDataError(f"No raw manifest at {manifest_path(config)}. Run `tennis-data ingest` first.")
    files = []
    missing = []
    for sf in selected_files(config):
        if sf.kind != "matches":
            continue
        entry = manifest.files.get(sf.path)
        path = config.raw_source_dir / sf.path
        if entry is None or not path.exists():
            missing.append(sf.path)
            continue
        if sha256_file(path) != entry["sha256"]:
            raise RawIntegrityError(f"{path} does not match its manifest checksum; re-run `tennis-data ingest`.")
        files.append(RawMatchFile(path, sf.path, sf.family, sf.season, entry["sha256"]))
    if missing:
        preview = ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else "")
        raise MissingRawDataError(
            f"{len(missing)} selected match file(s) not ingested ({preview}). Run `tennis-data ingest`."
        )
    return files


def read_raw_file(f: RawMatchFile, repo: str, commit: str) -> pl.DataFrame:
    df = pl.read_csv(f.path, infer_schema=False, encoding="utf8")
    missing = [c for c in SOURCE_MATCH_COLUMNS if c not in df.columns]
    if missing:
        raise SourceSchemaError(
            f"{f.rel_path} is missing expected column(s): {missing}. Present columns: {df.columns}. "
            "The source layout changed; update SOURCE_MATCH_COLUMNS and the mapping in normalize.py "
            "after inspecting the new files."
        )
    return df.select(
        *[pl.col(c).str.strip_chars().replace("", None).alias(c) for c in SOURCE_MATCH_COLUMNS],
        pl.lit(f.family).alias("source_family"),
        pl.lit(repo).alias("source_repo"),
        pl.lit(commit).alias("source_commit"),
        pl.lit(f.rel_path).alias("source_file"),
        pl.int_range(1, pl.len() + 1, dtype=pl.Int64).alias("source_row_number"),
        pl.lit(f.sha256).alias("source_sha256"),
        pl.lit(f.season, dtype=pl.Int32).alias("season"),
    )


def extra_columns(f: RawMatchFile) -> list[str]:
    header = pl.read_csv(f.path, infer_schema=False, n_rows=0).columns
    return [c for c in header if c not in SOURCE_MATCH_COLUMNS]


def load_raw(config: Config) -> pl.DataFrame:
    files = raw_match_files(config)
    frames = [read_raw_file(f, config.source.repo, config.source.commit) for f in files]
    return pl.concat(frames, how="vertical")


def _to_int(col: str) -> tuple[pl.Expr, pl.Expr]:
    """(parsed Int64 value, parse-failure flag). Accepts '185' and '185.0', rejects '18.5'."""
    f = pl.col(col).cast(pl.Float64, strict=False)
    ok = f.is_not_null() & f.is_finite() & (f == f.round())
    value = pl.when(ok).then(f).otherwise(None).cast(pl.Int64)
    return value, pl.col(col).is_not_null() & ~ok


def _to_float(col: str) -> tuple[pl.Expr, pl.Expr]:
    f = pl.col(col).cast(pl.Float64, strict=False)
    ok = f.is_not_null() & f.is_finite()
    return pl.when(ok).then(f).otherwise(None), pl.col(col).is_not_null() & ~ok


def _issue_list(flags: list[tuple[str, pl.Expr]]) -> pl.Expr:
    if not flags:
        return pl.lit([], dtype=pl.List(pl.String))
    return pl.concat_list([pl.when(e).then(pl.lit(code)).otherwise(None) for code, e in flags]).list.drop_nulls()


def to_canonical(raw: pl.DataFrame, config: Config) -> pl.DataFrame:
    """Map raw rows to canonical columns plus helper columns (_errors, _warnings, _raw_record)."""
    v = config.validation
    errors: list[tuple[str, pl.Expr]] = []
    warnings: list[tuple[str, pl.Expr]] = []

    for col in REQUIRED_SOURCE_VALUES:
        errors.append((f"missing_{col}", pl.col(col).is_null()))

    parsed: dict[str, pl.Expr] = {}
    int_cols = (
        ["winner_id", "loser_id", "match_num", "best_of", "draw_size", "minutes"]
        + [f"{p}_{s}" for p in ("winner", "loser") for s in ("seed", "ht", "rank", "rank_points")]
        + [f"{p}_{s}" for p in "wl" for s in STAT_FIELDS.values()]
    )
    for col in int_cols:
        value, failed = _to_int(col)
        parsed[col] = value
        (errors if col in CRITICAL_NUMERIC else warnings).append((f"unparseable_{col}", failed))
    for col in ("winner_age", "loser_age"):
        value, failed = _to_float(col)
        parsed[col] = value
        warnings.append((f"unparseable_{col}", failed))

    date = pl.col("tourney_date").str.strptime(pl.Date, "%Y%m%d", strict=False)
    errors.append(("unparseable_tourney_date", pl.col("tourney_date").is_not_null() & date.is_null()))

    surface_map = {s.lower(): s for s in v.allowed_surfaces}
    surface_lower = pl.col("surface").str.to_lowercase()
    surface = (
        pl.when(surface_lower.is_in(list(surface_map)))
        .then(surface_lower.replace_strict(surface_map, default=None))
        .otherwise(pl.col("surface"))
    )
    warnings.append(("missing_surface", pl.col("surface").is_null()))
    warnings.append(
        (
            "surface_case_normalized",
            surface_lower.is_in(list(surface_map)) & ~pl.col("surface").is_in(v.allowed_surfaces),
        )
    )
    errors.append(("invalid_surface", pl.col("surface").is_not_null() & ~surface_lower.is_in(list(surface_map))))

    df = raw.with_columns(
        pl.struct(SOURCE_MATCH_COLUMNS).struct.json_encode().alias(RAW_RECORD),
        pl.struct([c for c in SOURCE_MATCH_COLUMNS if c != "match_num"]).struct.json_encode().alias(CONTENT_KEY),
        _issue_list(errors).alias(ERRORS),
        _issue_list(warnings).alias(WARNINGS),
        *[expr.alias(f"_p_{c}") for c, expr in parsed.items()],
        date.alias("tournament_date"),
        surface.alias("surface"),
        pl.when(pl.col("tourney_id").is_not_null() & pl.col("match_num").is_not_null())
        .then(pl.concat_str([pl.col("source_family"), pl.col("tourney_id"), pl.col("match_num")], separator=":"))
        .alias("match_id"),
    )

    # Deterministic A/B assignment (see ordering.py); computed in Python because it uses SHA-256.
    a_ids: list[int | None] = []
    for mid, w, l in zip(df["match_id"], df["_p_winner_id"], df["_p_loser_id"], strict=True):
        a_ids.append(assign_players(mid, w, l)[0] if mid is not None and w is not None and l is not None else None)
    df = df.with_columns(pl.Series("player_a_id", a_ids, dtype=pl.Int64))
    a_won = pl.col("player_a_id") == pl.col("_p_winner_id")

    def side(a_expr: pl.Expr, b_expr: pl.Expr, is_a: bool) -> pl.Expr:
        return pl.when(a_won).then(a_expr if is_a else b_expr).otherwise(b_expr if is_a else a_expr)

    player_exprs = []
    for s, is_a in (("a", True), ("b", False)):
        player_exprs.append(side(pl.col("_p_winner_id"), pl.col("_p_loser_id"), is_a).alias(f"player_{s}_id"))
        for suffix, (src, dtype, _) in PLAYER_FIELDS.items():
            w = pl.col(f"_p_winner_{src}") if f"_p_winner_{src}" in df.columns else pl.col(f"winner_{src}")
            l = pl.col(f"_p_loser_{src}") if f"_p_loser_{src}" in df.columns else pl.col(f"loser_{src}")
            player_exprs.append(side(w, l, is_a).cast(dtype).alias(f"player_{s}_{suffix}"))
        for suffix, src in STAT_FIELDS.items():
            player_exprs.append(side(pl.col(f"_p_w_{src}"), pl.col(f"_p_l_{src}"), is_a).alias(f"player_{s}_{suffix}"))

    n_stats = pl.sum_horizontal([pl.col(c).is_not_null().cast(pl.Int32) for c in STAT_COLUMNS])
    status_map = {s: match_status(s) for s in raw["score"].unique().to_list() if s is not None}

    df = df.with_columns(player_exprs).with_columns(
        pl.col("tourney_id").alias("tournament_id"),
        pl.col("tourney_name").alias("tournament_name"),
        pl.col("tourney_level").alias("tournament_level"),
        pl.col("_p_draw_size").alias("draw_size"),
        pl.col("round").replace_strict(ROUND_ORDER, default=None, return_dtype=pl.Int32).alias("round_order"),
        pl.col("_p_match_num").alias("match_num"),
        pl.col("_p_best_of").cast(pl.Int32).alias("best_of"),
        pl.lit(None, dtype=pl.Date).alias("match_date"),
        pl.lit(None, dtype=pl.Datetime("us", "UTC")).alias("match_start_time"),
        pl.lit("tournament_week").alias("time_precision"),
        pl.col("_p_winner_id").alias("winner_id"),
        pl.col("_p_loser_id").alias(LOSER_ID),
        (pl.col("player_a_id") == pl.col("_p_winner_id")).alias("player_a_won"),
        pl.col("score").replace_strict(status_map, default="unknown").fill_null("unknown").alias("match_status"),
        pl.col("_p_minutes").alias("minutes"),
        pl.when(n_stats == len(STAT_COLUMNS)).then(pl.lit("complete"))
        .when(n_stats == 0).then(pl.lit("none"))
        .otherwise(pl.lit("partial")).alias("stats_status"),
    )  # fmt: skip

    canonical = [c.name for c in CANONICAL_COLUMNS if c.name != "quality_flags"]
    return df.select(*canonical, LOSER_ID, ERRORS, WARNINGS, RAW_RECORD, CONTENT_KEY)
