"""Synthetic Phase 1 -> 2 -> 3A build and a small Phase 3B config for end-to-end modeling tests."""

from __future__ import annotations

from pathlib import Path

from elo_helpers import build_phase1, synthetic_season_rows, write_elo_config

YEARS = tuple(range(2008, 2021))


def feature_config(tmp: Path, elo_config: Path) -> Path:
    default = (Path(__file__).resolve().parents[1] / "config" / "features.toml").read_text()
    body = default[default.index("[features]") :]
    body = body.replace("min_prior_points = 20000", "min_prior_points = 500").replace(
        "min_fit_points = 20000", "min_fit_points = 500"
    )
    text = (
        f'elo_config = "{elo_config}"\n\n[outputs]\nfeatures_dir = "{tmp / "features"}"\n'
        f'reports_dir = "{tmp / "reports_features"}"\n\n{body}'
    )
    path = tmp / "features.toml"
    path.write_text(text)
    return path


def modeling_config(tmp: Path, features_config: Path, **over) -> Path:
    default = (Path(__file__).resolve().parents[1] / "config" / "modeling.toml").read_text()
    body = default[default.index("[model]") :]
    repl = {
        'train_start = "1991-01-01"': 'train_start = "2008-01-01"',
        "first_prediction_year = 1994": "first_prediction_year = 2010",
        "development_years = [2002, 2009]": "development_years = [2013, 2016]",
        "exploratory_years = [2010, 2026]": "exploratory_years = [2017, 2020]",
        "folds = [[2002, 2004], [2005, 2007], [2008, 2009]]": "folds = [[2013, 2014], [2015, 2016]]",
        "isotonic_min_rows = 5000": "isotonic_min_rows = 50",
        "bootstrap_samples = 2000": f"bootstrap_samples = {over.get('bootstrap_samples', 100)}",
    }
    for a, b in repl.items():
        assert a in body, a
        body = body.replace(a, b)
    text = (
        f'features_config = "{features_config}"\n\n[outputs]\nmodeling_dir = "{tmp / "modeling"}"\n'
        f'reports_dir = "{tmp / "reports_modeling"}"\n\n{body}'
    )
    path = tmp / "modeling.toml"
    path.write_text(text)
    return path


def build_upstream(tmp: Path, archive) -> Path:
    """Phase 1 + Phase 2 backtest + Phase 3A features on synthetic seasons; returns the modeling config path."""
    from tennis_data.elo.backtest import backtest
    from tennis_data.elo.config import load_elo_config
    from tennis_data.elo.report import write_report as write_elo_report
    from tennis_data.features.config import load_feature_config
    from tennis_data.features.pipeline import build_features
    from tennis_data.features.report import write_report as write_feature_report

    data_cfg = build_phase1(tmp, archive, synthetic_season_rows(years=YEARS, stats=True))
    elo_path = write_elo_config(tmp, data_cfg)
    elo_cfg = load_elo_config(elo_path)
    write_elo_report(elo_cfg, backtest(elo_cfg))
    fcfg = load_feature_config(feature_config(tmp, elo_path))
    built = build_features(fcfg)
    write_feature_report(fcfg, built)
    return modeling_config(tmp, feature_config(tmp, elo_path))
