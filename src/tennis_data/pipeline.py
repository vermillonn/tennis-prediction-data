"""End-to-end build: raw snapshot -> canonical table -> validation -> Parquet."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from tennis_data import __version__
from tennis_data.config import Config
from tennis_data.ingestion.manifest import load_manifest, manifest_path
from tennis_data.normalization.normalize import (
    ERRORS,
    RAW_RECORD,
    WARNINGS,
    extra_columns,
    load_raw,
    raw_match_files,
    to_canonical,
)
from tennis_data.normalization.schema import CANONICAL_SCHEMA, QUARANTINE_EXTRA
from tennis_data.storage.parquet import sort_canonical, tree_checksums, write_partitioned, write_single
from tennis_data.validation.contract import enforce_contract
from tennis_data.validation.duplicates import detect_duplicates
from tennis_data.validation.reference import (
    apply_identity_checks,
    load_players,
    load_rankings,
    rank_discrepancies,
    rankings_duplicates,
)
from tennis_data.validation.rules import RULES_BY_CODE, apply_row_rules


@dataclass
class BuildResult:
    validated_rows: int
    quarantined_rows: int
    summary_path: Path


def _sorted_unique(col: str) -> pl.Expr:
    return pl.col(col).list.unique().list.sort()


def validate(config: Config) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    """Return (validated, quarantine, context) without writing anything."""
    raw = load_raw(config)
    df = to_canonical(raw, config)
    df = apply_row_rules(df, config)
    players = load_players(config)
    df = apply_identity_checks(df, players)
    df = detect_duplicates(df, config)

    unknown_codes = sorted(
        {c for codes in df[ERRORS].to_list() + df[WARNINGS].to_list() for c in codes} - set(RULES_BY_CODE)
    )
    if unknown_codes:
        raise RuntimeError(f"Validation produced unregistered codes {unknown_codes}; add them to rules.RULES")

    has_error = pl.col(ERRORS).list.len() > 0
    validated = enforce_contract(
        sort_canonical(df.filter(~has_error).with_columns(_sorted_unique(WARNINGS).alias("quality_flags")))
    )

    category_map = {code: r.category for code, r in RULES_BY_CODE.items()}
    quarantine = sort_canonical(
        df.filter(has_error).with_columns(
            _sorted_unique(WARNINGS).alias("quality_flags"),
            _sorted_unique(ERRORS).alias("quarantine_reasons"),
            pl.col(ERRORS)
            .list.eval(pl.element().replace_strict(category_map, return_dtype=pl.String))
            .list.unique()
            .list.sort()
            .alias("quarantine_categories"),
            pl.col(RAW_RECORD).alias("raw_record"),
        )
    ).select(
        *[pl.col(c).cast(t) for c, t in CANONICAL_SCHEMA.items()],
        *[pl.col(c).cast(t) for c, t in QUARANTINE_EXTRA.items()],
    )

    context = {
        "players": players,
        "rankings": load_rankings(config),
        "source_rows": raw.height,
        "extra_columns": {f.rel_path: cols for f in raw_match_files(config) if (cols := extra_columns(f))},
    }
    return validated, quarantine, context


def build(config: Config) -> BuildResult:
    validated, quarantine, ctx = validate(config)
    if validated.height + quarantine.height != ctx["source_rows"]:
        raise RuntimeError("Row accounting failed: validated + quarantined != source rows")

    paths = config.paths
    write_partitioned(validated, paths.validated_dir / "matches")
    paths.quarantine_dir.mkdir(parents=True, exist_ok=True)
    write_single(quarantine, paths.quarantine_dir / "matches.parquet")

    details = paths.reports_dir / "details"
    details.mkdir(parents=True, exist_ok=True)
    rankings = ctx["rankings"]
    rank_dup = None
    if rankings is not None:
        write_single(rank_discrepancies(validated, rankings), details / "rank_discrepancies.parquet")
        rank_dup = rankings_duplicates(rankings)

    manifest = load_manifest(manifest_path(config))
    summary = {
        "pipeline_version": __version__,
        "source": manifest.source,
        "selection": {
            "families": list(config.selection.families),
            "start_year": config.selection.start_year,
            "end_year": config.selection.end_year,
        },
        "input_files": {f.rel_path: f.sha256 for f in raw_match_files(config)},
        "rows": {
            "source": ctx["source_rows"],
            "validated": validated.height,
            "quarantined": quarantine.height,
        },
        "source_extra_columns": ctx["extra_columns"],
        "rankings_duplicate_keys": rank_dup,
        "outputs": {
            "validated": tree_checksums(paths.validated_dir / "matches"),
            "quarantine": tree_checksums(paths.quarantine_dir),
        },
    }
    summary_path = paths.reports_dir / "build_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return BuildResult(validated.height, quarantine.height, summary_path)
