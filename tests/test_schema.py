"""Stable schema and byte-reproducible outputs."""

from __future__ import annotations

import polars as pl

from conftest import match_row
from tennis_data.cli import main
from tennis_data.normalization.schema import CANONICAL_SCHEMA, QUARANTINE_EXTRA
from tennis_data.pipeline import build
from tennis_data.reporting.docs import DICTIONARY_PATH, render_dictionary
from tennis_data.reporting.quality import generate_report
from tennis_data.storage.parquet import tree_checksums
from tennis_data.storage.query import connect


def _rows():
    return [
        match_row(1, 100, 200), match_row(2, 300, 400, stats=False), match_row(3, 500, 600, score="W/O", stats=False),
        match_row(4, 100, 200), match_row(5, 700, 800, w_ace="-2"), match_row(6, 900, 300, surface=None, round="R16"),
    ]  # fmt: skip


def test_output_schema_matches_canonical(make_pipeline):
    cfg = make_pipeline(_rows())
    build(cfg)
    files = sorted((cfg.paths.validated_dir / "matches").rglob("*.parquet"))
    assert [f.parent.name for f in files] == ["season=2020"]
    for f in files:
        assert dict(pl.read_parquet_schema(f)) == CANONICAL_SCHEMA
    q = pl.read_parquet_schema(cfg.paths.quarantine_dir / "matches.parquet")
    assert dict(q) == {**CANONICAL_SCHEMA, **QUARANTINE_EXTRA}


def test_builds_are_byte_reproducible(make_pipeline):
    outputs = []
    for run in ("a", "b"):
        cfg = make_pipeline(_rows(), subdir=run)
        build(cfg)
        generate_report(cfg)
        outputs.append(
            (
                tree_checksums(cfg.paths.validated_dir),
                tree_checksums(cfg.paths.quarantine_dir),
                (cfg.paths.reports_dir / "data_quality_report.md").read_text(),
                tree_checksums(cfg.paths.reports_dir / "tables"),
            )
        )
    assert outputs[0] == outputs[1]


def test_rebuild_replaces_rather_than_appends(make_pipeline):
    cfg = make_pipeline(_rows())
    build(cfg)
    first = tree_checksums(cfg.paths.validated_dir)
    build(cfg)
    assert tree_checksums(cfg.paths.validated_dir) == first
    con = connect(cfg)
    assert con.execute("SELECT count(*) FROM matches").fetchone()[0] == 4
    assert con.execute("SELECT count(*) FROM quarantine").fetchone()[0] == 2


def test_duckdb_views(make_pipeline):
    cfg = make_pipeline(_rows())
    build(cfg)
    con = connect(cfg)
    n = con.execute("SELECT count(*) FROM matches").fetchone()[0]
    assert con.execute("SELECT count(*) FROM player_matches").fetchone()[0] == 2 * n
    assert con.execute("SELECT count(*) FROM player_matches WHERE won").fetchone()[0] == n
    assert con.execute("SELECT count(*) FROM players").fetchone()[0] > 0


def test_cli_run_and_query(make_pipeline, capsys):
    cfg = make_pipeline(_rows())
    assert main(["--config", str(cfg.config_path), "run"]) == 0
    capsys.readouterr()
    assert main(["--config", str(cfg.config_path), "query", "--csv", "SELECT count(*) AS n FROM matches"]) == 0
    assert capsys.readouterr().out.split() == ["n", "4"]
    report = (cfg.paths.reports_dir / "data_quality_report.md").read_text()
    assert "| quarantined rows | 2 |" in report


def test_data_dictionary_is_current():
    assert DICTIONARY_PATH.read_text() == render_dictionary(), "run `tennis-data docs` to regenerate"
