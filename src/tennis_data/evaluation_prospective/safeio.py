"""Untrusted-input handling shared by every Phase 4C reader.

Ledger exports, outcome snapshots, checkpoint records and receipts are treated as untrusted data:

- files are size-limited before reading and must be regular files (no symlinks, devices or directories);
- JSON is parsed with duplicate keys, ``NaN`` and ``Infinity`` rejected; nothing is ever unpickled or executed;
- relative paths named inside packages or receipts are resolved under an explicit root and refused if absolute, if
  they contain ``..`` or a drive, or if they escape the root through a symlink.
"""

from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath


class UntrustedInputError(ValueError):
    pass


def _no_dupes(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            raise UntrustedInputError(f"duplicate JSON key {k!r}")
        out[k] = v
    return out


def _no_constants(name):
    raise UntrustedInputError(f"non-finite number {name} is not allowed")


def strict_json(data: bytes | str):
    try:
        return json.loads(data, object_pairs_hook=_no_dupes, parse_constant=_no_constants)
    except (json.JSONDecodeError, UnicodeDecodeError, RecursionError) as e:
        raise UntrustedInputError(f"invalid JSON: {e}") from e


def read_bounded(path: Path, max_bytes: int) -> bytes:
    p = Path(path)
    if p.is_symlink():
        raise UntrustedInputError(f"{p.name}: symbolic links are not accepted")
    if not p.is_file():
        raise UntrustedInputError(f"{p}: not a regular file")
    size = p.stat().st_size
    if size > max_bytes:
        raise UntrustedInputError(f"{p.name}: {size} bytes exceeds the limit of {max_bytes}")
    data = p.read_bytes()
    if len(data) > max_bytes:  # changed while reading
        raise UntrustedInputError(f"{p.name}: grew beyond the limit while reading")
    return data


def safe_relpath(rel: str) -> PurePosixPath:
    if not isinstance(rel, str) or not rel or "\x00" in rel or "\\" in rel:
        raise UntrustedInputError(f"invalid relative path {rel!r}")
    p = PurePosixPath(rel)
    if p.is_absolute() or any(part in ("..", "") for part in p.parts) or ":" in p.parts[0]:
        raise UntrustedInputError(f"path {rel!r} escapes its root")
    return p


def resolve_under(root: Path, rel: str) -> Path:
    """``root / rel`` if it stays inside ``root`` after resolving symlinks."""
    p = safe_relpath(rel)
    base = Path(root).resolve()
    full = (base / Path(*p.parts)).resolve()
    if os.path.commonpath([base, full]) != str(base):
        raise UntrustedInputError(f"path {rel!r} escapes its root")
    return full


def write_new(path: Path, data: bytes) -> Path:
    """Create a file that must not exist yet (never overwrite evidence)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    return path
