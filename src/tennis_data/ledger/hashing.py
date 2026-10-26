"""Canonical serialization and the event hash chain (``ledger-1``).

Canonical JSON (the bytes that are hashed):

- UTF-8; ``json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)``;
- allowed value types: ``None``, ``bool``, ``int``, ``float`` (finite only), ``str``, ``list``, ``dict`` with ``str``
  keys. Anything else (dates, tuples, numpy scalars, sets, bytes) is rejected: callers convert first;
- floats are written with Python's shortest round-trip representation (``repr``), so a float hashes the same on every
  platform that implements IEEE 754 binary64; ``-0.0`` is normalized to ``0.0``;
- timestamps are strings in the single form ``YYYY-MM-DDTHH:MM:SS.ffffffZ`` (UTC, microseconds, ``Z``), produced by
  ``ts()``; naive datetimes are rejected.

Chain:

    genesis_hash = SHA256("tennis-ledger/genesis/v1\\0" + ledger_id)
    event_hash   = SHA256("tennis-ledger/event/v1\\0" + canonical(event_body) + "\\0" + previous_event_hash)

where ``event_body`` holds every stored field of the event except ``event_hash`` itself (sequence, event id, type,
schema version, recorded time, match id, idempotency key, payload, references and the previous hash).
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime

EVENT_TAG = b"tennis-ledger/event/v1\x00"
GENESIS_TAG = b"tennis-ledger/genesis/v1\x00"
CHECKPOINT_TAG = b"tennis-ledger/checkpoint/v1\x00"


class CanonicalizationError(ValueError):
    pass


def _check(v, path="$"):
    if v is None or isinstance(v, bool | str):
        return v
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        if not math.isfinite(v):
            raise CanonicalizationError(f"non-finite float at {path}")
        return 0.0 if v == 0 else v
    if isinstance(v, list):
        return [_check(x, f"{path}[{i}]") for i, x in enumerate(v)]
    if isinstance(v, dict):
        out = {}
        for k, x in v.items():
            if not isinstance(k, str):
                raise CanonicalizationError(f"non-string key at {path}")
            out[k] = _check(x, f"{path}.{k}")
        return out
    raise CanonicalizationError(f"unsupported type {type(v).__name__} at {path}")


def canonical(obj) -> bytes:
    return json.dumps(_check(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")  # fmt: skip


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ts(t: datetime) -> str:
    """The one timestamp form used in events."""
    if t.tzinfo is None:
        raise CanonicalizationError("naive datetime: timestamps must be timezone-aware")
    return t.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def parse_ts(s: str) -> datetime:
    if not isinstance(s, str) or not s.endswith("Z") or len(s) != 27:
        raise CanonicalizationError(f"timestamp not in canonical form: {s!r}")
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=UTC)


def genesis_hash(ledger_id: str) -> str:
    return sha256_hex(GENESIS_TAG + ledger_id.encode("utf-8"))


def event_hash(body: dict, previous_hash: str) -> str:
    if "event_hash" in body:
        raise CanonicalizationError("event body must not contain event_hash")
    return sha256_hex(EVENT_TAG + canonical(body) + b"\x00" + previous_hash.encode("ascii"))


def content_hash(obj) -> str:
    return sha256_hex(canonical(obj))
