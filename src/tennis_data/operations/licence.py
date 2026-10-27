"""Per-file source and licence records for a registered owner export (manifest only; files and hashes untouched).

The owner states the conditions of a source; they are recorded per file, with the date and the statement, next to
the original SHA-256. Before and after writing, every stored file is re-hashed against the manifest: a mismatch
aborts without writing. Earlier records are kept in ``licence_history``. Recording a condition never grants more
than it says: commercial use stays *unverified* and raw redistribution stays *prohibited* unless the owner records
otherwise.

``python -m tennis_data.operations.licence --label L --roles tml_matches,tml_players,excluded`` records the
TennisMyLife conditions given by the owner on 2026-10-26.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tennis_data.operations import exports

TENNISMYLIFE = {
    "source": "TennisMyLife (stats.tennismylife.org; TML-Database match and player files)",
    "attribution": "Match data: TennisMyLife (TML-Database), https://stats.tennismylife.org",
    "research_use": "allowed for research-oriented investigation, subject to the source's applicable terms",
    "raw_redistribution": "prohibited without authorization (raw files stay git-ignored; only aggregates published)",
    "commercial_use": "unverified (no commercial permission on record)",
    "status": "research_use_subject_to_terms",
}


def _stored(d: Path, f: dict, label: str) -> Path:
    if f["role"] in ("tml_matches", "tml_players"):
        return d / "workspace" / "tml" / label / f["stored_as"]
    return d / "files" / f["stored_as"]


def verify_hashes(d: Path, man: dict) -> dict:
    out = {}
    for f in man["files"]:
        p = _stored(d, f, man["label"])
        out[f["stored_as"]] = p.exists() and hashlib.sha256(p.read_bytes()).hexdigest() == f["sha256"]
    return out


def record(cfg, label: str, roles: list[str], conditions: dict, stated_on: str, statement: str) -> dict:
    d = exports.export_dir(cfg, label)
    path = d / "manifest.json"
    man = json.loads(path.read_text())
    bad = [k for k, ok in verify_hashes(d, man).items() if not ok]
    if bad:
        raise ValueError(f"stored files do not match the manifest hashes: {bad}")
    unknown = set(roles) - {f["role"] for f in man["files"]}
    if unknown:
        raise ValueError(f"no file with role(s) {sorted(unknown)}")
    rec = {**conditions, "stated_on": stated_on, "statement": statement}
    hashes_before = {f["stored_as"]: f["sha256"] for f in man["files"]}
    for f in man["files"]:
        if f["role"] in roles and f.get("licence") != rec:
            if f.get("licence"):
                man.setdefault("licence_history", []).append({"file": f["stored_as"], "previous": f["licence"]})
            f["licence"] = rec
    for f in man["files"]:
        f.setdefault("licence", {"status": "not verified"})
    man["licence_status"] = "per file (see files[].licence)"
    assert {f["stored_as"]: f["sha256"] for f in man["files"]} == hashes_before
    path.write_text(json.dumps(man, indent=2, sort_keys=True) + "\n")
    if not all(verify_hashes(d, man).values()):
        raise RuntimeError("stored files changed while recording the licence")
    return man


def main(argv=None) -> int:
    from tennis_data.operations.config import load_operations_config

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--label", required=True)
    ap.add_argument("--roles", required=True, help="comma-separated file roles the TennisMyLife conditions apply to")
    ap.add_argument("--stated-on", required=True)
    ap.add_argument("--operations-config")
    a = ap.parse_args(argv)
    man = record(load_operations_config(a.operations_config), a.label, a.roles.split(","), TENNISMYLIFE, a.stated_on,
                 "Owner instruction (Phase 5A final safety review): research-oriented investigation allowed subject to "
                 "applicable terms; unauthorized raw redistribution prohibited; commercial permission unverified.")  # fmt: skip
    print(json.dumps([{k: f[k] for k in ("original_name", "role", "sha256")} | {"licence": f["licence"]["status"]}
                      for f in man["files"]], indent=1))  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
