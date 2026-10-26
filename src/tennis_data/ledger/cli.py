"""``tennis-data ledger-*`` commands. Writes go through ``LedgerWriter`` (the trusted service); reads are read-only."""

from __future__ import annotations

import json
import sys
from pathlib import Path

COMMANDS = {
    "ledger-init": "create the ledger database (no-op if it exists)",
    "ledger-commit-synthetic": "commit a SYNTHETIC prediction from a JSON request file",
    "ledger-record": "append a match-status event (postponed, cancelled, corrected, result, void)",
    "ledger-verify": "verify the whole hash chain and lifecycle",
    "ledger-lookup": "list predictions of a match with derived status and timing",
    "ledger-events": "list every event of a match",
    "ledger-export": "export all events (JSONL + Parquet)",
    "ledger-checkpoint": "write a checkpoint of the current chain head",
    "ledger-verify-checkpoint": "verify the ledger against an externally kept checkpoint",
    "ledger-conflicts": "list rejected idempotency conflicts",
    "ledger-benchmark": "benchmark appends, verification and export on a synthetic ledger",
}


def register(sub) -> None:
    for name, helptext in COMMANDS.items():
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--ledger-config", help="config/ledger.toml by default")
        if name == "ledger-init":
            p.add_argument("--ledger-id")
        if name == "ledger-commit-synthetic":
            p.add_argument("request", help="JSON object with PredictionRequest fields (mode must be SYNTHETIC)")
        if name == "ledger-record":
            p.add_argument("event_type", choices=["MATCH_METADATA_CORRECTED", "MATCH_CANCELLED", "MATCH_POSTPONED",
                                                  "MATCH_RESULT_RECORDED", "PREDICTION_VOIDED"])  # fmt: skip
            p.add_argument("match_id")
            p.add_argument("payload", help="JSON file with the event payload")
            p.add_argument("--ref", help="referenced prediction event id (PREDICTION_VOIDED)")
        if name in ("ledger-lookup", "ledger-events"):
            p.add_argument("match_id")
        if name == "ledger-checkpoint":
            p.add_argument("--previous", help="previous checkpoint file to chain to")
            p.add_argument("--record-event", action="store_true", help="also append LEDGER_CHECKPOINT_CREATED")
        if name == "ledger-verify-checkpoint":
            p.add_argument("checkpoint")
        if name == "ledger-benchmark":
            p.add_argument("--events", type=int, default=10000)
            p.add_argument("--out", help="directory for the benchmark ledger (default: a temporary directory)")


def run(args) -> int:
    from tennis_data.ledger.writer import load_ledger_config

    cfg = load_ledger_config(args.ledger_config)
    cmd = args.command
    if cmd == "ledger-benchmark":
        from tennis_data.ledger.benchmark import run_benchmark

        print(json.dumps(run_benchmark(cfg, args.events, Path(args.out) if args.out else None), indent=2))
        return 0
    if cmd in ("ledger-init", "ledger-commit-synthetic", "ledger-record", "ledger-checkpoint"):
        from tennis_data.ledger.writer import LedgerWriter, PredictionRequest

        w = LedgerWriter(cfg)
        if cmd == "ledger-init":
            print(json.dumps(w.init(args.ledger_id), indent=2))
        elif cmd == "ledger-commit-synthetic":
            req = PredictionRequest(**json.loads(Path(args.request).read_text()))
            if req.mode != "SYNTHETIC":
                print("error: this command only commits SYNTHETIC predictions", file=sys.stderr)
                return 2
            print(json.dumps(w.commit_prediction(req).__dict__, indent=2))
        elif cmd == "ledger-record":
            r = w.record(args.event_type, args.match_id, json.loads(Path(args.payload).read_text()), args.ref)
            print(json.dumps(r.__dict__, indent=2))
        elif cmd == "ledger-checkpoint":
            from tennis_data.ledger.checkpoint import make_checkpoint, write_checkpoint

            prev = json.loads(Path(args.previous).read_text()) if args.previous else None
            cp = make_checkpoint(w.con, prev)
            path = write_checkpoint(cp, cfg.checkpoint_dir / f"checkpoint-{cp['sequence']:08d}.json")
            if args.record_event:
                w.record_checkpoint_event({"checkpoint_hash": cp["checkpoint_hash"], "covered_seq": cp["sequence"],
                                           "covered_head_hash": cp["head_hash"],
                                           "previous_checkpoint_hash": cp["previous_checkpoint_hash"]})  # fmt: skip
            print(json.dumps({"checkpoint": str(path), **cp}, indent=2))
        w.close()
        return 0
    if cmd == "ledger-verify":
        from tennis_data.ledger.verification import verify_ledger

        rep = verify_ledger(cfg.ledger_path, cfg.probability_tolerance).as_dict()
        print(json.dumps(rep, indent=2))
        return 0 if rep["ok"] else 1
    if cmd == "ledger-verify-checkpoint":
        from tennis_data.ledger.checkpoint import verify_checkpoint

        res = verify_checkpoint(cfg.ledger_path, json.loads(Path(args.checkpoint).read_text()))
        print(json.dumps(res, indent=2))
        return 0 if res["ok"] else 1
    if cmd == "ledger-export":
        from tennis_data.ledger.export import export

        print(json.dumps(export(cfg.ledger_path, cfg.export_dir), indent=2))
        return 0
    from tennis_data.ledger.reader import LedgerReader

    rd = LedgerReader(cfg.ledger_path)
    if cmd == "ledger-lookup":
        out = [{"event_id": p["event"]["event_id"], "seq": p["event"]["seq"], "mode": p["event"]["mode"],
                "recorded_at": p["event"]["recorded_at"], "probability_a": p["event"]["probability_a"],
                "player_a_id": p["event"]["payload"]["match"]["player_a_id"], "voided": p["voided"],
                "timing": p["timing"], "match_status": p["match_status"]} for p in rd.predictions_for_match(args.match_id)]  # fmt: skip
    elif cmd == "ledger-events":
        out = rd.events_for_match(args.match_id)
    else:
        out = rd.conflicts()
    print(json.dumps(out, indent=2, default=str))
    return 0
