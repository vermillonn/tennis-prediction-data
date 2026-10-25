"""Command-line interface: `tennis-data <command> [--config PATH]`."""

from __future__ import annotations

import argparse
import sys

from tennis_data.config import ConfigError, load_config
from tennis_data.ingestion.fetch import IngestionError, describe, ingest, verify_raw


def _cmd_ingest(args, config) -> int:
    result = ingest(config)
    print(f"raw snapshot: {config.raw_source_dir}")
    print(describe(result))
    return 0


def _cmd_verify(args, config) -> int:
    problems = verify_raw(config)
    for p in problems:
        print(p, file=sys.stderr)
    print("raw snapshot OK" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


def _cmd_build(args, config) -> int:
    from tennis_data.pipeline import build
    from tennis_data.storage.query import build_database

    result = build(config)
    db = build_database(config)
    print(f"validated rows:   {result.validated_rows}  -> {config.paths.validated_dir / 'matches'}")
    print(f"quarantined rows: {result.quarantined_rows}  -> {config.paths.quarantine_dir / 'matches.parquet'}")
    print(f"build summary:    {result.summary_path}")
    print(f"duckdb views:     {db}")
    return 0


def _cmd_report(args, config) -> int:
    from tennis_data.reporting.quality import generate_report

    print(f"report: {generate_report(config)}")
    return 0


def _cmd_docs(args, config) -> int:
    from tennis_data.reporting.docs import write_dictionary

    print(f"data dictionary: {write_dictionary()}")
    return 0


def _cmd_run(args, config) -> int:
    for step in (_cmd_ingest, _cmd_build, _cmd_report):
        if step(args, config):
            return 1
    return 0


def _cmd_elo_backtest(args, config) -> int:
    from tennis_data.elo.backtest import backtest
    from tennis_data.elo.config import load_elo_config
    from tennis_data.elo.report import write_report

    cfg = load_elo_config(args.elo_config)
    result = backtest(cfg)
    report = write_report(cfg, result)
    overall = result["test"]["overall"]
    for r in overall.iter_rows(named=True):
        print(
            f"{r['model']:<32} n={r['n']:<6} log_loss={r['log_loss']:.4f} brier={r['brier']:.4f} accuracy={r['accuracy']:.4f}"
        )
    print(f"predictions: {cfg.elo_dir / 'predictions'}")
    print(f"report:      {report}")
    return 0


def _elo_dir_if_built():
    from tennis_data.elo.config import DEFAULT_ELO_CONFIG, load_elo_config

    return load_elo_config(DEFAULT_ELO_CONFIG).elo_dir if DEFAULT_ELO_CONFIG.exists() else None


def _cmd_query(args, config) -> int:
    from tennis_data.storage.query import connect

    con = connect(config, elo_dir=_elo_dir_if_built())
    rel = con.sql(args.sql)
    if rel is None:
        return 0
    if args.csv:
        import csv

        writer = csv.writer(sys.stdout)
        writer.writerow(rel.columns)
        writer.writerows(rel.fetchall())
    else:
        rel.show(max_rows=args.max_rows, max_width=args.max_width)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tennis-data", description=__doc__)
    parser.add_argument("--config", help="TOML config (default: config/default.toml)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ingest", help="download/import the selected raw files and record checksums")
    sub.add_parser("verify", help="re-check raw files against the manifest")
    sub.add_parser("build", help="normalize, validate and write Parquet + DuckDB views")
    sub.add_parser("report", help="write the data-quality report from the built outputs")
    sub.add_parser("docs", help="regenerate docs/data_dictionary.md from the schema")
    sub.add_parser("run", help="ingest + build + report")
    eb = sub.add_parser("elo-backtest", help="tune Elo on validation, write predictions, evaluate on the test period")
    eb.add_argument("--elo-config", help="Elo TOML config (default: config/elo.toml)")
    q = sub.add_parser(
        "query", help="run SQL against the DuckDB views (matches, quarantine, player_matches, players, elo_*)"
    )
    q.add_argument("sql")
    q.add_argument("--csv", action="store_true", help="print CSV instead of a table")
    q.add_argument("--max-rows", type=int, default=40)
    q.add_argument("--max-width", type=int, default=200)
    args = parser.parse_args(argv)

    handlers = {
        "ingest": _cmd_ingest, "verify": _cmd_verify, "build": _cmd_build, "report": _cmd_report,
        "docs": _cmd_docs, "run": _cmd_run, "query": _cmd_query, "elo-backtest": _cmd_elo_backtest,
    }  # fmt: skip
    try:
        config = load_config(args.config)
        return handlers[args.command](args, config)
    except (ConfigError, IngestionError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
