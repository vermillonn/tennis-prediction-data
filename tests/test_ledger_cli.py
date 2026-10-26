"""CLI wiring and read-only DuckDB views of the ledger export."""

from __future__ import annotations

import json
from dataclasses import asdict

import duckdb

from ledger_helpers import ledger_cfg, request, result, writer
from tennis_data.ledger.export import export


def _config(tmp_path):
    text = (
        f'prospective_config = "{ledger_cfg(tmp_path).prospective.config_path}"\n[outputs]\n'
        f'ledger_path = "{tmp_path / "ledger.sqlite"}"\nexport_dir = "{tmp_path / "export"}"\n'
        f'checkpoint_dir = "{tmp_path / "cps"}"\nreports_dir = "{tmp_path / "rep"}"\n[ledger]\n'
        'forecast_slots = ["pre_match"]\n'
    )
    p = tmp_path / "ledger.toml"
    p.write_text(text)
    return str(p)


def test_cli_end_to_end(tmp_path, capsys):
    from tennis_data.cli import main

    c = _config(tmp_path)
    assert main(["ledger-init", "--ledger-config", c, "--ledger-id", "cli-ledger"]) == 0
    req = asdict(request(ledger_cfg(tmp_path), match_id="synthetic:cli"))
    (tmp_path / "req.json").write_text(json.dumps(req))
    assert main(["ledger-commit-synthetic", "--ledger-config", c, str(tmp_path / "req.json")]) == 0
    assert main(["ledger-commit-synthetic", "--ledger-config", c, str(tmp_path / "req.json")]) == 0  # idempotent
    (tmp_path / "res.json").write_text(json.dumps(result(req["player_a_id"])))
    assert main(["ledger-record", "--ledger-config", c, "MATCH_RESULT_RECORDED", "synthetic:cli",
                 str(tmp_path / "res.json")]) == 0  # fmt: skip
    capsys.readouterr()
    assert main(["ledger-verify", "--ledger-config", c]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["ok"] and rep["events"] == 2
    assert main(["ledger-lookup", "--ledger-config", c, "synthetic:cli"]) == 0
    look = json.loads(capsys.readouterr().out)
    assert look[0]["match_status"] == "decided" and look[0]["mode"] == "SYNTHETIC"
    assert main(["ledger-checkpoint", "--ledger-config", c, "--record-event"]) == 0
    cp_path = json.loads(capsys.readouterr().out)["checkpoint"]
    assert main(["ledger-verify-checkpoint", "--ledger-config", c, cp_path]) == 0
    assert main(["ledger-export", "--ledger-config", c]) == 0
    assert main(["ledger-conflicts", "--ledger-config", c]) == 0
    pros = {**req, "mode": "PROSPECTIVE", "match_id": "tour:2026-1:1"}
    (tmp_path / "pros.json").write_text(json.dumps(pros))
    assert main(["ledger-commit-synthetic", "--ledger-config", c, str(tmp_path / "pros.json")]) == 2


def test_duckdb_views_are_read_only(tmp_path):
    from tennis_data.storage.query import ledger_views

    w = writer(tmp_path)
    w.commit_prediction(request(w.cfg, match_id="synthetic:v"))
    export(w.path, tmp_path / "export")
    con = duckdb.connect()
    ledger_views(con, tmp_path / "export")
    assert con.execute("SELECT count(*), any_value(mode) FROM ledger_predictions").fetchone() == (1, "SYNTHETIC")
    try:
        con.execute("UPDATE ledger_events SET mode = 'PROSPECTIVE'")
        raise AssertionError("view must not be updatable")
    except duckdb.Error:
        pass
