"""Data-quality report: CSV tables plus a Markdown summary, computed with DuckDB over the outputs.

The report contains no wall-clock timestamps, so it is reproducible byte-for-byte.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb

from tennis_data.config import Config
from tennis_data.storage.query import _sql_path, connect
from tennis_data.validation.reference import RANK_TOLERANCE_DAYS
from tennis_data.validation.rules import RULES

PLAYED = "match_status <> 'walkover'"


def _all_rows_view(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE OR REPLACE VIEW all_rows AS
        SELECT season, source_family, surface, tournament_level, tournament_id, match_status, stats_status,
               quality_flags, 'validated' AS disposition FROM matches
        UNION ALL
        SELECT season, source_family, surface, tournament_level, tournament_id, match_status, stats_status,
               quality_flags, 'quarantined' FROM quarantine
        """
    )


def _queries(config: Config) -> dict[str, str]:
    v = config.validation
    expected = " ".join(f"WHEN '{fam}' THEN {yr}" for fam, yr in sorted(v.stats_expected_from.items()))
    expected_expr = f"CASE source_family {expected} ELSE 9999 END" if expected else "9999"
    return {
        "matches_by_season": """
            SELECT season, source_family,
                   count(*) AS source_rows,
                   count(*) FILTER (WHERE disposition = 'validated') AS validated,
                   count(*) FILTER (WHERE disposition = 'quarantined') AS quarantined,
                   count(*) FILTER (WHERE disposition = 'validated' AND match_status = 'completed') AS completed,
                   count(*) FILTER (WHERE disposition = 'validated' AND match_status = 'retired') AS retired,
                   count(*) FILTER (WHERE disposition = 'validated' AND match_status = 'walkover') AS walkover,
                   count(*) FILTER (WHERE disposition = 'validated' AND match_status = 'default') AS "default",
                   count(*) FILTER (WHERE disposition = 'validated'
                                    AND match_status IN ('abandoned', 'unfinished', 'unknown')) AS other_status,
                   count(DISTINCT tournament_id) AS tournaments
            FROM all_rows GROUP BY ALL ORDER BY season, source_family""",
        "matches_by_surface": """
            SELECT coalesce(surface, '(missing)') AS surface, count(*) AS matches,
                   round(100.0 * count(*) / sum(count(*)) OVER (), 2) AS pct,
                   min(season) AS first_season, max(season) AS last_season
            FROM matches GROUP BY ALL ORDER BY matches DESC, surface""",
        "tournament_level_coverage": f"""
            SELECT tournament_level, source_family, count(DISTINCT tournament_id) AS tournaments, count(*) AS matches,
                   min(season) AS first_season, max(season) AS last_season,
                   round(100.0 * count(*) FILTER (WHERE stats_status = 'complete' AND {PLAYED})
                         / nullif(count(*) FILTER (WHERE {PLAYED}), 0), 2) AS pct_played_with_stats,
                   round(100.0 * count(*) FILTER (WHERE player_a_rank IS NOT NULL AND player_b_rank IS NOT NULL)
                         / count(*), 2) AS pct_both_ranked
            FROM matches GROUP BY ALL ORDER BY matches DESC, tournament_level, source_family""",
        "serve_stats_coverage_by_season": f"""
            WITH c AS (
                SELECT season, source_family,
                       count(*) FILTER (WHERE {PLAYED}) AS played_matches,
                       count(*) FILTER (WHERE {PLAYED} AND stats_status = 'complete') AS complete,
                       count(*) FILTER (WHERE {PLAYED} AND stats_status = 'partial') AS partial,
                       count(*) FILTER (WHERE {PLAYED} AND stats_status = 'none') AS missing
                FROM matches GROUP BY ALL)
            SELECT *, round(100.0 * complete / nullif(played_matches, 0), 2) AS pct_complete,
                   season >= {expected_expr} AS stats_expected,
                   season >= {expected_expr}
                     AND coalesce(complete / nullif(played_matches, 0), 0) < {v.min_stats_coverage} AS under_covered
            FROM c ORDER BY season, source_family""",
        "flagged_rows": "SELECT count(*) FILTER (WHERE len(quality_flags) > 0) AS rows FROM matches",
        "match_status": """
            SELECT match_status, disposition, count(*) AS rows FROM all_rows GROUP BY ALL
            ORDER BY match_status, disposition""",
        "quarantine_reasons": """
            SELECT reason, count(*) AS rows FROM (SELECT unnest(quarantine_reasons) AS reason FROM quarantine)
            GROUP BY ALL ORDER BY rows DESC, reason""",
        "quarantine_categories": """
            SELECT category, count(*) AS rows FROM (SELECT unnest(quarantine_categories) AS category FROM quarantine)
            GROUP BY ALL ORDER BY rows DESC, category""",
        "quarantine_by_season": """
            SELECT season, source_family, count(*) AS rows, string_agg(DISTINCT r, ', ' ORDER BY r) AS reasons
            FROM (SELECT season, source_family, unnest(quarantine_reasons) AS r FROM quarantine)
            GROUP BY ALL ORDER BY season, source_family""",
        "warning_flags": """
            SELECT flag, disposition, count(*) AS rows FROM (SELECT unnest(quality_flags) AS flag, disposition FROM all_rows)
            GROUP BY ALL ORDER BY rows DESC, flag, disposition""",
        "duplicates_and_conflicts": """
            SELECT r AS reason, match_id, duplicate_of, tournament_id, tournament_name, season, round,
                   winner_id, player_a_id, player_b_id, score, source_file, source_row_number
            FROM (SELECT *, unnest(quarantine_reasons) AS r FROM quarantine)
            WHERE r IN ('duplicate_record', 'conflicting_record', 'conflicting_match_id', 'player_repeated_in_round')
            ORDER BY r, tournament_id, round, source_file, source_row_number""",
        "player_identity_coverage": """
            WITH ids AS (SELECT player_a_id AS id, quality_flags FROM matches
                         UNION ALL SELECT player_b_id, quality_flags FROM matches)
            SELECT metric, value FROM (
            SELECT 0 AS ord, 'distinct players (validated)' AS metric, count(DISTINCT id)::BIGINT AS value FROM ids
            UNION ALL SELECT 1, 'player appearances (validated)', count(*) FROM ids
            UNION ALL SELECT 2, 'matches with a player missing from atp_players.csv',
                             count(*) FILTER (WHERE list_contains(quality_flags, 'player_not_in_master')) FROM matches
            UNION ALL SELECT 3, 'matches with a player unnamed in atp_players.csv',
                             count(*) FILTER (WHERE list_contains(quality_flags, 'player_unnamed_in_master')) FROM matches
            UNION ALL SELECT 4, 'matches with a name differing from atp_players.csv',
                             count(*) FILTER (WHERE list_contains(quality_flags, 'player_name_mismatch_master')) FROM matches
            UNION ALL SELECT 5, 'matches quarantined for identity reasons',
                             count(*) FILTER (WHERE list_contains(quarantine_categories, 'identity')) FROM quarantine
            UNION ALL SELECT 6, 'matches with both players ranked (match file)',
                             count(*) FILTER (WHERE player_a_rank IS NOT NULL AND player_b_rank IS NOT NULL) FROM matches
            UNION ALL SELECT 7, 'matches with exactly one player ranked',
                             count(*) FILTER (WHERE (player_a_rank IS NULL) <> (player_b_rank IS NULL)) FROM matches
            UNION ALL SELECT 8, 'matches with neither player ranked',
                             count(*) FILTER (WHERE player_a_rank IS NULL AND player_b_rank IS NULL) FROM matches) ORDER BY ord""",
    }


def _rank_queries(path: Path) -> dict[str, str]:
    src = f"read_parquet('{_sql_path(path)}')"
    return {
        "rank_discrepancies_by_season": f"""
            SELECT season,
                   count(*) FILTER (WHERE comparison = 'equal') AS equal,
                   count(*) FILTER (WHERE comparison = 'different') AS different,
                   count(*) FILTER (WHERE comparison = 'equals_later_list') AS equals_later_list,
                   count(*) FILTER (WHERE comparison = 'missing_in_match_file') AS missing_in_match_file,
                   count(*) FILTER (WHERE comparison = 'missing_in_rankings_file') AS missing_in_rankings_file,
                   count(*) FILTER (WHERE comparison = 'both_missing') AS both_missing,
                   round(100.0 * count(*) FILTER (WHERE comparison = 'different')
                         / nullif(count(*) FILTER (WHERE comparison IN ('equal', 'different')), 0), 2)
                     AS pct_different_when_both_present
            FROM {src} GROUP BY ALL ORDER BY season""",
        "rank_discrepancy_summary": f"""
            SELECT comparison, count(*) AS player_appearances,
                   round(median(ranking_age_days), 1) AS median_ranking_age_days,
                   max(later_list_days_after) AS max_days_list_published_after
            FROM {src} GROUP BY ALL ORDER BY player_appearances DESC, comparison""",
        "rank_later_list_lag": f"""
            SELECT later_list_days_after AS days_after_tournament_date, count(*) AS player_appearances
            FROM {src} WHERE comparison = 'equals_later_list' GROUP BY ALL ORDER BY 1""",
    }


def _markdown_table(columns: list[str], rows: list[tuple], limit: int | None = None) -> str:
    def fmt(x) -> str:
        if x is None:
            return ""
        if isinstance(x, bool):
            return "yes" if x else "no"
        return str(x).replace("|", "\\|")

    shown = rows if limit is None else rows[:limit]
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(fmt(x) for x in r) + " |" for r in shown]
    if limit is not None and len(rows) > limit:
        lines.append(f"\n_{len(rows) - limit} more rows in the CSV._")
    return "\n".join(lines)


def generate_report(config: Config) -> Path:
    reports = config.paths.reports_dir
    tables_dir = reports / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads((reports / "build_summary.json").read_text())

    con = connect(config)
    _all_rows_view(con)
    queries = _queries(config)
    rank_path = reports / "details" / "rank_discrepancies.parquet"
    if rank_path.exists():
        queries.update(_rank_queries(rank_path))

    results: dict[str, tuple[list[str], list[tuple]]] = {}
    for name, sql in queries.items():
        rel = con.sql(sql)
        results[name] = (rel.columns, rel.fetchall())
        con.execute(f"COPY ({sql}) TO '{_sql_path(tables_dir / f'{name}.csv')}' (HEADER, DELIMITER ',')")
    con.close()

    report = reports / "data_quality_report.md"
    report.write_text(_render(config, summary, results))
    return report


def _validated_flag_count(results, flag: str) -> int:
    return sum(r[2] for r in results["warning_flags"][1] if r[0] == flag and r[1] == "validated")


def _render(config: Config, summary: dict, results: dict) -> str:
    src = summary["source"]
    rows = summary["rows"]
    t = results
    under = [r for r in t["serve_stats_coverage_by_season"][1] if r[-1]]
    dup_rows = t["duplicates_and_conflicts"][1]
    n_dup = sum(1 for r in dup_rows if r[0] == "duplicate_record")
    n_conflict = sum(1 for r in dup_rows if r[0] != "duplicate_record")
    out = [
        "# Data-quality report: ATP singles matches",
        "",
        "Generated by `tennis-data report`. Reproducible: contains no timestamps; every number is "
        "recomputed from the Parquet outputs.",
        "",
        "## Source",
        "",
        f"- Repository: [{src['repo']}]({src['repo_url']}) at commit `{src['commit']}` "
        f"(commit verified: {'yes' if src['commit_verified'] else 'no'}, mode: {src['mode']})",
        f"- Families: {', '.join(summary['selection']['families'])}; seasons "
        f"{summary['selection']['start_year']}-{summary['selection']['end_year']}; "
        f"{len(summary['input_files'])} match files",
        f"- License: {src['license']}. {src['attribution']}",
        "",
        "## Headline counts",
        "",
        _markdown_table(
            ["metric", "rows"],
            [
                ("source rows", rows["source"]),
                ("validated rows", rows["validated"]),
                ("quarantined rows", rows["quarantined"]),
                ("  of which duplicate copies", n_dup),
                (
                    "  of which conflict flags (conflicting_record, conflicting_match_id, player_repeated_in_round)",
                    n_conflict,
                ),
                ("validated rows with at least one warning flag", t["flagged_rows"][1][0][0]),
            ],
        ),
        "",
        "Every source row is accounted for: `validated + quarantined = source`. Quarantined rows are kept in "
        "`data/quarantine/matches.parquet` with their reasons and the untouched raw record.",
        "",
        "## Matches by season",
        "",
        _markdown_table(*t["matches_by_season"]),
        "",
        "## Matches by surface (validated)",
        "",
        _markdown_table(*t["matches_by_surface"]),
        "",
        "## Tournament-level coverage (validated)",
        "",
        _markdown_table(*t["tournament_level_coverage"]),
        "",
        "## Player identity coverage",
        "",
        _markdown_table(*t["player_identity_coverage"]),
        "",
        "## Serve-statistics coverage by season",
        "",
        "Played matches only (walkovers excluded). Missing statistics are left null; nothing is imputed. "
        f"`under_covered` marks seasons on or after the family's expected start year "
        f"({', '.join(f'{k}: {v}' for k, v in sorted(config.validation.stats_expected_from.items()))}) "
        f"with complete-stats coverage below {config.validation.min_stats_coverage:.0%}.",
        "",
        _markdown_table(*t["serve_stats_coverage_by_season"]),
        "",
        f"Under-covered seasons: {', '.join(f'{r[0]} ({r[1]}, {r[6]}%)' for r in under) or 'none'}.",
        "",
        "## Match status",
        "",
        _markdown_table(*t["match_status"]),
        "",
        "## Quarantine",
        "",
        "By category:",
        "",
        _markdown_table(*t["quarantine_categories"]),
        "",
        "By reason (a row can have several reasons):",
        "",
        _markdown_table(*t["quarantine_reasons"]),
        "",
        "By season:",
        "",
        _markdown_table(*t["quarantine_by_season"]),
        "",
        "## Duplicates and conflicts",
        "",
        _markdown_table(*t["duplicates_and_conflicts"], limit=40),
        "",
        "## Warning flags (rows kept, flagged in `quality_flags`)",
        "",
        _markdown_table(*t["warning_flags"]),
        "",
        "## Source discrepancies",
        "",
    ]
    if "rank_discrepancy_summary" in t:
        out += [
            "Match-file rank (as of `tournament_date`) versus the latest `atp_rankings_*.csv` entry published on "
            f"or up to {RANK_TOLERANCE_DAYS} days before `tournament_date`, per player appearance in validated "
            "matches. `missing_in_rankings_file` includes unranked players and periods without published rankings "
            "(1982, and intermittent coverage before 1985):",
            "",
            _markdown_table(*t["rank_discrepancy_summary"]),
            "",
            "`equals_later_list`: the match-file rank equals the list published *after* `tournament_date` "
            "(lag below). A 1-2 day lag is an event recorded as starting on a weekend that used the following "
            "Monday's list: no result of the event itself is encoded, but the list was not yet public on the "
            "event's first day. A lag of 7+ days would mean the rank could include the event's own results; for a "
            "player whose rank did not move, an equal later rank can also be coincidence. Strictly point-in-time "
            "features should take ranks from the rankings files with `ranking_date <= tournament_date`.",
            "",
            _markdown_table(*t["rank_later_list_lag"]),
            "",
            "By season: `tables/rank_discrepancies_by_season.csv`. Per-appearance detail: "
            "`details/rank_discrepancies.parquet`.",
            "",
        ]
    out += [
        f"- Name mismatches against `atp_players.csv`: "
        f"{_validated_flag_count(t, 'player_name_mismatch_master')} validated rows "
        "(after trimming and whitespace/case normalisation).",
        f"- Surface values differing only in case: {_validated_flag_count(t, 'surface_case_normalized')} validated rows.",
        f"- Duplicate (ranking_date, player) keys in rankings files: {summary.get('rankings_duplicate_keys')}.",
        f"- Unexpected extra columns in source files: {summary['source_extra_columns'] or 'none'}.",
        "",
        "## Validation rules",
        "",
        _markdown_table(["code", "severity", "category", "description"],
                        [(r.code, r.severity, r.category, r.description) for r in RULES]),
        "",
    ]  # fmt: skip
    return "\n".join(out)
