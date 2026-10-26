"""Owner-supplied exports: registration with provenance, a dedicated workspace, and roles per file.

Possessing a file does not establish a right to redistribute it. Registered files are copied into the git-ignored
workspace (``<workspace>/exports/<label>/``) with their SHA-256, size, original name, the owner's statement of
origin and acquisition date, and a licence status that stays ``"not verified"`` unless the owner states otherwise.
Only aggregates ever reach the repository.

Roles:

- ``tml_matches`` / ``tml_players``: TML-format match files (Sackmann columns, ATP codes) and the player file. They
  are converted with ``operations.tml`` and imported through the unchanged Phase 4A path;
- ``crosscheck_season`` / ``crosscheck_vbt``: independent result sources used only to corroborate outcomes and
  coverage (``operations.crosscheck``). They are never imported;
- ``excluded``: files outside the frozen rating population (e.g. Challenger). They are registered for provenance
  only and never parsed into any state.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

from tennis_data.evaluation_prospective.safeio import read_bounded
from tennis_data.ledger.hashing import ts

ROLES = {"tml_matches": None, "tml_players": "ATP_Database.csv", "crosscheck_season": "crosscheck_season.csv",
         "crosscheck_vbt": "crosscheck_vbt.csv", "excluded": None}  # fmt: skip
MAX_BYTES = 256 * 1024 * 1024


def export_dir(cfg, label: str) -> Path:
    if not label or "/" in label or label.startswith("."):
        raise ValueError(f"invalid export label {label!r}")
    return cfg.workspace_dir / "exports" / label


def export_config(cfg, label: str, gap_only: bool = False):
    """Operations config for one export: its own workspace and TML input set; Phase 4A outputs stay untouched.

    ``gap_only=False``: the whole file, for validation (identity, compatibility, coverage). ``gap_only=True``: only
    events starting after the archive's last event week, in a separate workspace: the backfill set.
    """
    d = export_dir(cfg, label)
    man = json.loads((d / "manifest.json").read_text())
    tml_files = sorted(f["stored_as"] for f in man["files"] if f["role"] in ("tml_matches", "tml_players"))
    tml = {**cfg.tml, "commit": label, "files": tml_files, "provider": man["provider"],
           "source_dir": str(d / "workspace" / "tml" / label)}  # fmt: skip
    if gap_only:
        tml["min_event_start"] = (cfg.archive_last_week + timedelta(days=1)).isoformat()
    return replace(cfg, workspace_dir=d / ("workspace_gap" if gap_only else "workspace"), reports_dir=d / "reports",
                   tml=tml)  # fmt: skip


def register(cfg, label: str, files: list[tuple[Path, str]], *, provider: str, origin: str, acquired_on: str,
             licence: str = "not verified", now: datetime | None = None) -> dict:  # fmt: skip
    """Copy files into the export workspace with a provenance manifest (idempotent for identical bytes)."""
    d = export_dir(cfg, label)
    when = ts(now or datetime.now(UTC))
    entries = []
    tml_dir = d / "workspace" / "tml" / label
    tml_dir.mkdir(parents=True, exist_ok=True)
    (d / "files").mkdir(parents=True, exist_ok=True)
    tml_manifest = {"repository": f"owner-supplied:{label}", "commit": label, "files": {}}
    for path, role in files:
        if role not in ROLES:
            raise ValueError(f"unknown role {role!r} (allowed: {sorted(ROLES)})")
        data = read_bounded(Path(path), MAX_BYTES)
        sha = hashlib.sha256(data).hexdigest()
        name = ROLES[role] or re.sub(r"^[0-9a-f]{8}-", "", Path(path).name)  # drop an upload-tool prefix
        target = (tml_dir if role in ("tml_matches", "tml_players") else d / "files") / name
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != sha:
            raise FileExistsError(f"{target} already holds different content")
        target.write_bytes(data)
        e = {"original_name": Path(path).name, "stored_as": name, "role": role, "sha256": sha, "bytes": len(data)}
        entries.append(e)
        if role in ("tml_matches", "tml_players"):
            tml_manifest["files"][name] = {"sha256": sha, "bytes": len(data), "url": f"owner-supplied:{e['original_name']}",
                                           "retrieved_at": f"{acquired_on}T00:00:00.000000Z"}  # fmt: skip
    man = {"format": "tennis-owner-export-1", "label": label, "provider": provider, "origin": origin,
           "acquired_on": acquired_on, "licence_status": licence, "registered_at": when,
           "redistribution": "not permitted unless the licence says otherwise; raw files stay git-ignored",
           "files": sorted(entries, key=lambda e: e["stored_as"])}  # fmt: skip
    (d / "manifest.json").write_text(json.dumps(man, indent=2, sort_keys=True) + "\n")
    (tml_dir / "manifest.json").write_text(json.dumps(tml_manifest, indent=2, sort_keys=True) + "\n")
    return man
