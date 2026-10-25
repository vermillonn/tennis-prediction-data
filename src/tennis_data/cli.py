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


def _cmd_features_build(args, config) -> int:
    from tennis_data.features.config import load_feature_config
    from tennis_data.features.dictionary import write_dictionary
    from tennis_data.features.pipeline import build_features
    from tennis_data.features.report import write_report

    cfg = load_feature_config(args.features_config)
    built = build_features(cfg)
    report = write_report(cfg, built)
    perf = built["performance"]
    print(f"feature rows: {built['table'].height}  columns: {len(built['table'].columns)}")
    print(f"configuration_hash: {cfg.configuration_hash()}")
    print(
        f"runtime: {perf['total_seconds']:.1f}s  peak RSS: {perf['peak_rss_mb']:.0f} MB  opponent fits: {perf['opponent_fits']}"
    )
    print(f"features: {cfg.features_dir / 'serve_return'}")
    print(f"report:   {report}")
    if args.write_docs:
        print(f"dictionary: {write_dictionary(cfg)}")
    return 0


def _cmd_features_docs(args, config) -> int:
    from tennis_data.features.config import load_feature_config
    from tennis_data.features.dictionary import write_dictionary

    print(f"feature dictionary: {write_dictionary(load_feature_config(args.features_config))}")
    return 0


def _cmd_model_backtest(args, config) -> int:
    from tennis_data.modeling.config import load_modeling_config
    from tennis_data.modeling.pipeline import run_modeling

    cfg = load_modeling_config(args.modeling_config)
    res = run_modeling(cfg, use_cache=not args.no_cache)
    perf, fid = res["perf"], res["final_id"]
    print(f"configuration_hash: {cfg.configuration_hash()}  frozen_hash: {res['frozen']['frozen_hash'][:16]}")
    print(f"selected: {fid}  (lowest development log loss: {res['sel']['lowest']})")
    for name, block in (("development", res["devres"]), ("exploratory", res["exres"])):
        b = block["boot"][f"{fid} vs M0"]
        print(f"{name}: n={b['rows']}  dLL vs surface Elo {b['log_loss_diff']:+.5f}  "
              f"95% CI [{b['ll_ci95'][0]:+.5f}, {b['ll_ci95'][1]:+.5f}]")  # fmt: skip
    print(
        f"runtime: {perf['seconds']['total']:.1f}s  peak RSS: {perf['peak_rss_mb']:.0f} MB  fits: {perf['logistic_fits']}"
    )
    print(f"reports: {cfg.reports_dir}")
    return 0


def _cmd_prospective_import(args, config) -> int:
    from tennis_data.prospective.backfill import ingest_file
    from tennis_data.prospective.config import load_prospective_config

    cfg = load_prospective_config(args.prospective_config)
    rec = ingest_file(cfg, args.path, args.format)
    print(
        f"stored {rec['provider']} snapshot {rec['sha256'][:16]} ({rec['bytes']} bytes, retrieved {rec['retrieved_at']})"
    )
    return 0


def _cmd_prospective_backfill(args, config) -> int:
    from tennis_data.prospective.backfill import run_backfill
    from tennis_data.prospective.config import load_prospective_config

    s = run_backfill(load_prospective_config(args.prospective_config))["summary"]
    print(f"raw snapshot {s['raw_snapshot_id']}: {s['provider_records']} records, {s['accepted_records']} accepted, "
          f"{s['phase1_validated']} validated; state {s['state_snapshot_id']}")  # fmt: skip
    return 0


def _cmd_prospective_run(args, config) -> int:
    from tennis_data.prospective.config import load_prospective_config
    from tennis_data.prospective.pipeline import run_phase4a

    res = run_phase4a(load_prospective_config(args.prospective_config))
    rp, ready = res["replay"]["summary"], res["readiness"]
    print(f"replay: {rp['cases']} cases, max error {rp['max_abs_error']:.1e}, passed={rp['passed']}")
    print(f"freshness probe: {res['freshness']['status']}")
    print(f"ready for prospective predictions: {ready['ready_for_prospective_predictions']}")
    for b in ready["blockers"]:
        print(f"  blocker: {b}")
    print(f"reports: {res['cfg'].reports_dir}")
    return 0


def _cmd_prospective_predict(args, config) -> int:
    import json as _json
    from datetime import date as _date

    from tennis_data.prospective.config import load_prospective_config
    from tennis_data.prospective.pipeline import predict_upcoming
    from tennis_data.prospective.state import UpcomingMatch

    items = _json.loads(open(args.fixtures).read())
    fixtures = [UpcomingMatch(**{**i, "tournament_date": _date.fromisoformat(i["tournament_date"])}) for i in items]
    out = predict_upcoming(load_prospective_config(args.prospective_config), fixtures, args.mode)
    out.write_csv(sys.stdout)
    return 0


def _modeling_dir_if_built():
    from tennis_data.modeling.config import DEFAULT_MODELING_CONFIG, load_modeling_config

    return load_modeling_config(DEFAULT_MODELING_CONFIG).modeling_dir if DEFAULT_MODELING_CONFIG.exists() else None


def _features_dir_if_built():
    from tennis_data.features.config import DEFAULT_FEATURES_CONFIG, load_feature_config

    return load_feature_config(DEFAULT_FEATURES_CONFIG).features_dir if DEFAULT_FEATURES_CONFIG.exists() else None


def _elo_dir_if_built():
    from tennis_data.elo.config import DEFAULT_ELO_CONFIG, load_elo_config

    return load_elo_config(DEFAULT_ELO_CONFIG).elo_dir if DEFAULT_ELO_CONFIG.exists() else None


def _cmd_query(args, config) -> int:
    from tennis_data.storage.query import connect

    con = connect(
        config,
        elo_dir=_elo_dir_if_built(),
        features_dir=_features_dir_if_built(),
        modeling_dir=_modeling_dir_if_built(),
    )
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
    fb = sub.add_parser("features-build", help="build Phase 3A serve/return features and the coverage report")
    fb.add_argument("--features-config", help="feature TOML config (default: config/features.toml)")
    fb.add_argument("--write-docs", action="store_true", help="also regenerate docs/feature_dictionary.md")
    fd = sub.add_parser("features-docs", help="regenerate docs/feature_dictionary.md from the feature registry")
    fd.add_argument("--features-config", help="feature TOML config (default: config/features.toml)")
    mb = sub.add_parser("model-backtest", help="Phase 3B: develop, select, freeze and evaluate models on top of Elo")
    mb.add_argument("--modeling-config", help="modeling TOML config (default: config/modeling.toml)")
    mb.add_argument("--no-cache", action="store_true", help="rebuild the joined modeling dataset")
    for name, helptext in (("prospective-import", "store an authorized export immutably in the Phase 4A raw store"),
                           ("prospective-backfill", "reconcile stored snapshots and extend the state (Phase 4A)"),
                           ("prospective-run", "Phase 4A: gap audit, backfill, replay verification, readiness reports"),
                           ("prospective-predict", "price upcoming matches with the frozen model (refuses stale state)")):  # fmt: skip
        sp = sub.add_parser(name, help=helptext)
        sp.add_argument("--prospective-config", help="config/prospective.toml by default")
        if name == "prospective-import":
            sp.add_argument("path")
            sp.add_argument("--format", required=True, choices=["tennis-fixture-provider-1", "sackmann_csv"])
        if name == "prospective-predict":
            sp.add_argument("fixtures", help="JSON list of upcoming matches (UpcomingMatch fields)")
            sp.add_argument(
                "--mode",
                default="prospective",
                choices=["prospective", "exploratory_retrospective", "historical_reconstruction"],
            )
    q = sub.add_parser(
        "query",
        help="run SQL against the DuckDB views (matches, quarantine, player_matches, players, elo_*, serve_return_features, model_*)",
    )
    q.add_argument("sql")
    q.add_argument("--csv", action="store_true", help="print CSV instead of a table")
    q.add_argument("--max-rows", type=int, default=40)
    q.add_argument("--max-width", type=int, default=200)
    args = parser.parse_args(argv)

    handlers = {
        "ingest": _cmd_ingest, "verify": _cmd_verify, "build": _cmd_build, "report": _cmd_report,
        "docs": _cmd_docs, "run": _cmd_run, "query": _cmd_query, "elo-backtest": _cmd_elo_backtest,
        "features-build": _cmd_features_build, "features-docs": _cmd_features_docs,
        "model-backtest": _cmd_model_backtest, "prospective-import": _cmd_prospective_import,
        "prospective-backfill": _cmd_prospective_backfill, "prospective-run": _cmd_prospective_run,
        "prospective-predict": _cmd_prospective_predict,
    }  # fmt: skip
    try:
        config = load_config(args.config)
        return handlers[args.command](args, config)
    except (ConfigError, IngestionError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
