"""Exports that preserve every event (voided predictions, cancellations, corrections included) and re-import them.

- ``events.jsonl``: one canonical JSON object per event in sequence order, preceded by a header line with the ledger
  metadata; re-importable and verifiable with no database at all.
- ``events.parquet``: the same rows for analysis (DuckDB views read it; files cannot write back to the ledger).
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from tennis_data.ledger import storage
from tennis_data.ledger.hashing import canonical
from tennis_data.ledger.verification import verify_events

HEADER = "tennis-ledger-export-1"


def export(path, out_dir: Path) -> dict:
    con = storage.connect(path, read_only=True)
    try:
        m, events = storage.meta(con), storage.all_events(con)
    finally:
        con.close()
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [canonical({"format": HEADER, "meta": m})]
    for e in events:
        lines.append(canonical({**e, "payload": json.loads(e["payload"]), "refs": json.loads(e["refs"])}))
    (out_dir / "events.jsonl").write_bytes(b"\n".join(lines) + b"\n")
    rows = [{**e, "payload": e["payload"], "refs": e["refs"]} for e in events]
    schema = {"seq": pl.Int64, "event_id": pl.String, "event_type": pl.String, "schema_version": pl.String,
              "recorded_at": pl.String, "match_id": pl.String, "mode": pl.String, "idempotency_key": pl.String,
              "probability_a": pl.Float64, "probability_b": pl.Float64, "ref_event_id": pl.String,
              "payload": pl.String, "refs": pl.String, "previous_hash": pl.String, "event_hash": pl.String}  # fmt: skip
    pl.DataFrame(rows, schema=schema, orient="row").write_parquet(out_dir / "events.parquet")
    head = events[-1]["event_hash"] if events else m["genesis_hash"]
    return {"events": len(events), "head_hash": head, "ledger_id": m["ledger_id"]}


def read_jsonl(path: Path) -> tuple[dict, list[dict]]:
    lines = Path(path).read_bytes().splitlines()
    head = json.loads(lines[0])
    if head.get("format") != HEADER:
        raise ValueError("not a ledger export")
    events = [json.loads(x) for x in lines[1:] if x.strip()]
    for e in events:  # restore the stored text form so verification also checks canonical payloads
        e["payload"] = canonical(e["payload"]).decode("utf-8")
        e["refs"] = canonical(e["refs"]).decode("utf-8")
    return head["meta"], events


def import_jsonl(path: Path, db_path: Path) -> dict:
    """Rebuild a ledger database from an export. The chain trigger accepts only an unbroken, correctly linked
    sequence, and the result is verified before returning."""
    meta, events = read_jsonl(path)
    rep = verify_events(meta, events)
    if not rep.ok:
        raise ValueError(f"export does not verify: {rep.issues[:3]}")
    if Path(db_path).exists():
        raise FileExistsError(db_path)
    con = storage.connect(db_path)
    try:
        storage.create_schema(con)
        con.execute("BEGIN IMMEDIATE")
        con.execute("INSERT INTO ledger_meta VALUES (1, ?, ?, ?, ?)",
                    (meta["ledger_id"], meta["schema_version"], meta["genesis_hash"], meta["created_at"]))  # fmt: skip
        for e in events:
            con.execute(f"INSERT INTO events ({', '.join(storage.COLUMNS)}) VALUES ({', '.join('?' * len(storage.COLUMNS))})",
                        tuple(e[c] for c in storage.COLUMNS))  # fmt: skip
        con.execute("COMMIT")
        return verify_events(storage.meta(con), storage.all_events(con)).as_dict()
    finally:
        con.close()
