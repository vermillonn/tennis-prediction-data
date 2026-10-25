"""Statistical coverage and quality report for the Phase 3A feature table."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import polars.selectors as cs

from tennis_data.elo.backtest import period_expr
from tennis_data.features.config import FeatureConfig
from tennis_data.reporting.quality import _markdown_table


def _t(df: pl.DataFrame, digits: int = 4, limit: int | None = None) -> str:
    def fmt(v):
        return round(v, digits) if isinstance(v, float) else v

    return _markdown_table(df.columns, [tuple(fmt(v) for v in r) for r in df.rows()], limit=limit)


def _pct(num: pl.Expr, den: pl.Expr) -> pl.Expr:
    return pl.when(den > 0).then(100.0 * num / den).otherwise(None)


def coverage_tables(cfg: FeatureConfig, frame: pl.DataFrame, table: pl.DataFrame) -> dict[str, pl.DataFrame]:
    frame = frame.with_columns(period_expr(cfg.elo).alias("period"))
    table = table.with_columns(period_expr(cfg.elo).alias("period"))
    targets = frame.filter("is_target")
    k = cfg.shrinkage
    out: dict[str, pl.DataFrame] = {}

    out["match_counts"] = pl.DataFrame(
        {
            "metric": [
                "validated matches (all statuses)",
                "validated matches in rating pools",
                "Phase 2 target matches (feature rows)",
                "matches with eligible serve/return statistics (any status)",
                "target matches with eligible statistics",
                "Elo update-eligible matches without eligible statistics (still used by Elo)",
            ],
            "matches": [
                frame.height,
                frame.filter(pl.col("pool").is_not_null()).height,
                targets.height,
                frame.filter("stats_eligible").height,
                targets.filter("stats_eligible").height,
                frame.filter(pl.col("is_update") & ~pl.col("stats_eligible")).height,
            ],
        }
    )

    def by(col: str) -> pl.DataFrame:
        return (
            targets.with_columns(pl.col(col).cast(pl.String).fill_null("(missing)"))
            .group_by(col)
            .agg(pl.len().alias("target_matches"), pl.col("stats_eligible").sum().alias("with_stats"))
            .with_columns(_pct(pl.col("with_stats"), pl.col("target_matches")).alias("pct_with_stats"))
            .sort(col)
        )

    out["coverage_by_season"] = by("season")
    out["coverage_by_surface"] = by("surface")
    out["coverage_by_level"] = by("tournament_level")

    excl = frame.filter(pl.col("pool").is_not_null())
    out["stats_exclusions"] = (
        excl.with_columns(pl.col("stats_exclusion_reason").fill_null("eligible"))
        .group_by("stats_exclusion_reason")
        .agg(pl.len().alias("matches"), pl.col("is_update").sum().alias("of_which_elo_update_eligible"))
        .sort("matches", descending=True)
    )

    # Player appearances (two per feature row).
    pa = pl.concat(
        [
            table.select(
                "period", "surface",
                pl.col(f"{s}_service_points_observed").alias("sp"), pl.col(f"{s}_return_points_observed").alias("rp"),
                pl.col(f"{s}_surface_service_points").alias("ssp"), pl.col(f"{s}_has_stats_history").alias("hist"),
                pl.col(f"{s}_service_prior_dominated").alias("sdom"), pl.col(f"{s}_return_prior_dominated").alias("rdom"),
                (pl.col(f"{s}_surface_prior_weight") > 0.5).alias("surfdom"),
            )
            for s in ("a", "b")
        ]
    )  # fmt: skip
    out["appearances_by_period"] = (
        pa.group_by("period")
        .agg(
            pl.len().alias("player_appearances"),
            (~pl.col("hist")).sum().alias("no_prior_statistics"),
            pl.col("sdom").sum().alias("service_prior_dominated"),
            pl.col("rdom").sum().alias("return_prior_dominated"),
            pl.col("surfdom").sum().alias("surface_prior_dominated"),
        )
        .with_columns(
            _pct(pl.col("no_prior_statistics"), pl.col("player_appearances")).alias("pct_no_prior_statistics"),
            _pct(pl.col("service_prior_dominated"), pl.col("player_appearances")).alias("pct_service_prior_dominated"),
            _pct(pl.col("surface_prior_dominated"), pl.col("player_appearances")).alias("pct_surface_prior_dominated"),
        )
        .sort("period")
    )
    qs = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]

    def dist(col: str, name: str) -> pl.DataFrame:
        return (
            pa.filter(pl.col(col).is_not_null())
            .group_by("period")
            .agg([pl.col(col).quantile(q, "nearest").alias(f"q{int(q * 100):02d}") for q in qs])
            .with_columns(pl.lit(name).alias("measure"))
            .select("measure", "period", *[f"q{int(q * 100):02d}" for q in qs])
            .sort("period")
        )

    out["sample_size_distribution"] = pl.concat(
        [
            dist("sp", "prior service points"),
            dist("rp", "prior return points"),
            dist("ssp", "prior surface service points"),
        ]
    )

    def both(cond_a: pl.Expr, cond_b: pl.Expr) -> pl.Expr:
        return (cond_a & cond_b).sum()

    sp = lambda s: pl.col(f"{s}_service_points_observed")  # noqa: E731
    out["prediction_coverage"] = (
        table.group_by("period")
        .agg(
            pl.len().alias("feature_rows"),
            pl.col("stats_eligible").sum().alias("rows_whose_own_match_has_stats"),
            both(sp("a") > 0, sp("b") > 0).alias("both_players_any_history"),
            both(sp("a") >= k.kappa_service, sp("b") >= k.kappa_service).alias(
                f"both_players_ge_{int(k.kappa_service)}_points"
            ),
            both(sp("a") >= 2000, sp("b") >= 2000).alias("both_players_ge_2000_points"),
            ((sp("a") > 0) ^ (sp("b") > 0)).sum().alias("exactly_one_player_history"),
        )
        .with_columns(
            _pct(pl.col("rows_whose_own_match_has_stats"), pl.col("feature_rows")).alias("pct_own_match_has_stats"),
            _pct(pl.col("both_players_any_history"), pl.col("feature_rows")).alias("pct_both_any_history"),
            _pct(pl.col(f"both_players_ge_{int(k.kappa_service)}_points"), pl.col("feature_rows")).alias(
                "pct_both_ge_kappa"
            ),
        )
        .sort("period")
    )

    feats = [c for c in table.columns if c not in ("period",)]
    miss = table.select([pl.col(c).null_count().alias(c) for c in feats]).row(0, named=True)
    miss_test = table.filter(pl.col("period") == "test").select([pl.col(c).null_count().alias(c) for c in feats])
    miss_val = table.filter(pl.col("period") == "validation").select([pl.col(c).null_count().alias(c) for c in feats])
    n_test = table.filter(pl.col("period") == "test").height
    n_val = table.filter(pl.col("period") == "validation").height
    out["missingness"] = pl.DataFrame(
        {
            "column": feats,
            "pct_null_all": [100.0 * miss[c] / table.height for c in feats],
            "pct_null_validation": [100.0 * miss_val[c][0] / n_val if n_val else None for c in feats],
            "pct_null_test": [100.0 * miss_test[c][0] / n_test if n_test else None for c in feats],
        }
    )
    return out


def _write_tables(tables_dir: Path, tables: dict[str, pl.DataFrame]) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.with_columns(cs.float().round(6)).write_csv(tables_dir / f"{name}.csv")


def write_report(cfg: FeatureConfig, built: dict) -> Path:
    frame, table, snap = built["frame"], built["table"], built["snap"]
    tables = coverage_tables(cfg, frame, table)
    fit_log = snap.fit_log
    if fit_log.height:
        tables["opponent_fits"] = fit_log.select(
            pl.len().alias("fits"),
            pl.col("iterations").mean().alias("mean_iterations"),
            pl.col("iterations").max().alias("max_iterations"),
            pl.col("converged").mean().alias("share_converged"),
            pl.col("observations").max().alias("max_observations_in_a_fit"),
        )
    _write_tables(cfg.reports_dir / "tables", tables)
    perf = built.get("performance", {})
    summary = built.get("summary", {})
    elo_join = None
    elo_preds = cfg.elo.elo_dir / "predictions"
    if elo_preds.exists():
        ids = pl.concat(
            [pl.read_parquet(p, columns=["match_id"]) for p in sorted(elo_preds.rglob("*.parquet"))]
        ).unique()
        elo_join = (table.height, table.join(ids, on="match_id", how="semi").height, ids.height)

    k, o = cfg.shrinkage, cfg.opponent
    mc = dict(tables["match_counts"].rows())
    lines = [
        "# Phase 3A coverage and quality report: serve/return features",
        "",
        "Generated by `tennis-data features-build`. Counts are recomputed from the outputs. Runtime figures are "
        "measured on the machine that ran the build and vary between runs; everything else is reproducible.",
        "",
        f"- Feature version `{cfg.full_version}`, configuration hash `{cfg.configuration_hash()}`",
        f"- Feature rows: **{table.height}** (one per Phase 2 target match), {len(table.columns)} columns; "
        f"player snapshots: {2 * table.height}",
        f"- Join to Phase 2 Elo predictions: {elo_join[1]} of {elo_join[0]} feature rows match "
        f"{elo_join[2]} distinct Elo match_ids"
        if elo_join
        else "- Phase 2 Elo predictions not found; join not checked",
        "",
        "## 1. Two different coverage concepts",
        "",
        "**Match-statistics coverage** is the share of matches whose own serve/return statistics are present and "
        "valid. **Prediction coverage** is the share of feature rows (pre-match snapshots) for which useful "
        "*prior* statistics exist for the players. A match can have perfect statistics and still be predicted from "
        "an empty history (e.g. a debutant), and vice versa.",
        "",
        _t(tables["match_counts"]),
        "",
        f"Share of target matches with eligible statistics: "
        f"{100.0 * mc['target matches with eligible statistics'] / mc['Phase 2 target matches (feature rows)']:.2f}%.",
        "",
        "### Prediction coverage by evaluation period",
        "",
        _t(tables["prediction_coverage"], 2),
        "",
        "## 2. Match-statistics coverage",
        "",
        "### By season",
        "",
        _t(tables["coverage_by_season"], 2),
        "",
        "### By surface",
        "",
        _t(tables["coverage_by_surface"], 2),
        "",
        "### By tournament level",
        "",
        _t(tables["coverage_by_level"], 2),
        "",
        "### Exclusions (matches in rating pools)",
        "",
        "`stats_eligible` is independent of Elo eligibility: rows excluded here may still update Elo "
        "(`of_which_elo_update_eligible`). Rows Phase 1 quarantined (e.g. for impossible statistics) are not in "
        "the validated table at all and never reach this pipeline.",
        "",
        _t(tables["stats_exclusions"]),
        "",
        "## 3. Historical sample sizes and prior dominance (player appearances)",
        "",
        _t(tables["appearances_by_period"], 2),
        "",
        f"A service estimate is *prior-dominated* when the player has fewer prior service points than "
        f"kappa_service = {k.kappa_service:g} (the prior carries more than half the weight); likewise for return "
        f"(kappa {k.kappa_return:g}) and surface (kappa {k.kappa_surface_service:g}). Appearances in matches with a "
        "missing surface have no surface estimate and are not counted in `surface_prior_dominated`.",
        "",
        _t(tables["sample_size_distribution"], 1),
        "",
        "## 4. Missingness of every exported column (%)",
        "",
        "Warm-up rows (before 1991) have no statistical history at all, which dominates `pct_null_all`.",
        "",
        _t(tables["missingness"], 2),
        "",
        "## 5. Opponent-adjustment fits",
        "",
        _t(tables["opponent_fits"], 2) if "opponent_fits" in tables else "Opponent model disabled.",
        "",
        f"Refit every {o.refit_interval_weeks} batches on released observations only; ridge kappa serve "
        f"{o.kappa_serve:g}, return {o.kappa_return:g}, surface {o.kappa_surface:g}; observation half-life "
        f"{o.half_life_weeks:g} weeks. Per-fit log: `data/features/opponent_fit_log.parquet`.",
        "",
        "## 6. Runtime and memory (this run; not reproducible)",
        "",
        _markdown_table(
            ["measure", "value"], [(k_, round(v, 2) if isinstance(v, float) else v) for k_, v in sorted(perf.items())]
        )
        if perf
        else "Not measured.",
        "",
        "Peak memory is the process maximum resident set size reported by the OS (includes the Python runtime, "
        "Polars and the full validated table).",
        "",
        "## 7. Selection bias from uneven coverage",
        "",
        "- **Era:** no tour-level statistics exist before 1991. Every player starts the validation period with an "
        "empty statistical history, so early-1990s features are mostly prior-driven, and careers that began before "
        "1991 are only partly observed.",
        "- **Event type:** Davis Cup matches are far less often covered than tour events (see by-level table). A "
        "player's history therefore over-represents tour-level matches, and players who mainly play Davis Cup or "
        "lower tiers (not in the default pool) have thinner and less representative histories.",
        "- **Survivorship:** players accumulate statistics by staying on tour; strong players reach later rounds "
        "and generate more points. Sample size is correlated with ability, so `has_stats_history` and point counts "
        "carry information beyond the rates themselves, and missingness is not at random.",
        "- **Retirements and flags:** matches with a retirement, a walkover or flagged statistics never contribute. "
        "Retirements are not random (injury, fatigue), so their exclusion slightly favours healthy-match data.",
        "- **Population prior:** the prior is the rate of the observed population, which over-represents tour regulars.",
        "",
        f"Summary file: `reports/features/feature_summary.json` (rows {summary.get('rows')}, output checksums).",
        "",
    ]
    path = cfg.reports_dir / "coverage_report.md"
    path.write_text("\n".join(lines))
    return path
