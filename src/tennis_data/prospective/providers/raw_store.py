"""Immutable, content-addressed store of raw provider responses.

Each payload is written once as ``<root>/<provider>/<sha256>.bin`` with a metadata file ``<sha256>.json``
(endpoint, parameters with secrets removed, retrieval time, HTTP status, bytes). Writing the same bytes again
is a no-op, so repeated ingestion is idempotent; nothing is ever modified in place.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

SECRET_KEYS = ("api_key", "apikey", "key", "token", "authorization", "x-api-key")


@dataclass(frozen=True)
class RawRecord:
    provider: str
    endpoint: str
    params: dict
    retrieved_at: str  # ISO 8601 UTC
    sha256: str
    bytes: int
    http_status: int | None
    path: str


def redact(params: dict) -> dict:
    return {k: ("<redacted>" if k.lower() in SECRET_KEYS else v) for k, v in sorted(params.items())}


class RawStore:
    def __init__(self, root: Path):
        self.root = Path(root)

    def put(self, provider: str, endpoint: str, params: dict, content: bytes, retrieved_at: datetime,
            http_status: int | None = None) -> RawRecord:  # fmt: skip
        if retrieved_at.tzinfo is None:
            raise ValueError("retrieved_at must be timezone-aware (UTC)")
        sha = hashlib.sha256(content).hexdigest()
        d = self.root / provider
        d.mkdir(parents=True, exist_ok=True)
        blob, meta = d / f"{sha}.bin", d / f"{sha}.json"
        if blob.exists():
            if hashlib.sha256(blob.read_bytes()).hexdigest() != sha:
                raise RuntimeError(f"raw snapshot {blob} was modified on disk")
            return RawRecord(**json.loads(meta.read_text()))
        tmp = blob.with_suffix(".tmp")
        tmp.write_bytes(content)
        tmp.replace(blob)
        rec = RawRecord(provider, endpoint, redact(params), retrieved_at.isoformat(), sha, len(content), http_status,
                        str(blob.relative_to(self.root)))  # fmt: skip
        meta.write_text(json.dumps(asdict(rec), indent=2, sort_keys=True) + "\n")
        return rec

    def get(self, provider: str, sha: str) -> bytes:
        blob = self.root / provider / f"{sha}.bin"
        content = blob.read_bytes()
        if hashlib.sha256(content).hexdigest() != sha:
            raise RuntimeError(f"raw snapshot {blob} failed its checksum")
        return content

    def records(self, provider: str | None = None) -> list[RawRecord]:
        out = []
        for meta in sorted(self.root.glob(f"{provider or '*'}/*.json")):
            out.append(RawRecord(**json.loads(meta.read_text())))
        return sorted(out, key=lambda r: (r.provider, r.retrieved_at, r.sha256))

    def snapshot_id(self, provider: str | None = None) -> str:
        """Hash of every stored payload hash: identifies the exact raw input of a backfill."""
        shas = sorted(f"{r.provider}:{r.sha256}" for r in self.records(provider))
        return hashlib.sha256("\n".join(shas).encode()).hexdigest()[:16]
