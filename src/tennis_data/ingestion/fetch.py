"""Download or import raw archive files, verify them and record provenance.

Raw files are written once and never modified. Re-running ingestion re-verifies
checksums and skips files that are already present and intact, so repeated runs
neither duplicate data nor alter the manifest.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from tennis_data.config import Config
from tennis_data.ingestion.manifest import Manifest, load_manifest, manifest_path, save_manifest
from tennis_data.ingestion.sources import KNOWN_COVERAGE, SourceFile, selected_files

Fetcher = Callable[[str, Path, int], None]


class IngestionError(RuntimeError):
    """Base class for ingestion failures."""


class MissingSourceFileError(IngestionError):
    """A selected file does not exist at the pinned source."""


class RawIntegrityError(IngestionError):
    """A raw file on disk no longer matches its recorded checksum."""


class SourceCommitError(IngestionError):
    """A local clone is not at the pinned commit."""


@dataclass
class IngestResult:
    fetched: list[str]
    verified: list[str]

    @property
    def total(self) -> int:
        return len(self.fetched) + len(self.verified)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def source_url(config: Config, rel_path: str) -> str:
    src = config.source
    if src.mode == "local":
        return f"file://{Path(src.local_path).resolve() / rel_path}"
    return f"{src.base_url}/{src.repo}/{src.commit}/{rel_path}"


def http_fetch(url: str, dest: Path, timeout: int) -> None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp, dest.open("wb") as out:
            shutil.copyfileobj(resp, out)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise FileNotFoundError(url) from exc
        raise IngestionError(f"HTTP {exc.code} while downloading {url}: {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise IngestionError(
            f"Network error downloading {url}: {exc.reason}. Check connectivity/proxy, or set "
            "[source].mode = 'local' with local_path pointing at a clone of the archive."
        ) from exc


def local_fetch(url: str, dest: Path, timeout: int) -> None:
    src = Path(url.removeprefix("file://"))
    if not src.is_file():
        raise FileNotFoundError(str(src))
    shutil.copyfile(src, dest)


def _missing_file_message(config: Config, sf: SourceFile, url: str) -> str:
    msg = f"Source file not found: {sf.path}\n  looked at: {url}\n  pinned commit: {config.source.commit}"
    if sf.kind == "matches" and sf.family in KNOWN_COVERAGE:
        lo, hi = KNOWN_COVERAGE[sf.family]
        msg += (
            f"\n  family '{sf.family}' is known to cover {lo}-{hi} at the default commit; "
            "adjust [selection].start_year/end_year or families in your config."
        )
    elif sf.kind in ("players", "rankings"):
        msg += "\n  set [selection].include_players / include_rankings = false to skip reference files."
    return msg


def _verify_local_commit(config: Config) -> bool:
    """Return True if the local clone is verified to be at the pinned commit."""
    src = config.source
    root = Path(src.local_path)
    if not root.is_dir():
        raise IngestionError(f"[source].local_path does not exist or is not a directory: {root}")
    if not src.verify_local_commit:
        return False
    try:
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SourceCommitError(
            f"Cannot read git HEAD of {root}. Use a git clone of {src.repo}, or set "
            "[source].verify_local_commit = false (provenance will be marked unverified)."
        ) from exc
    if head != src.commit:
        raise SourceCommitError(
            f"Local clone {root} is at {head}, but config pins {src.commit}. "
            f"Run `git -C {root} checkout {src.commit}` or update [source].commit."
        )
    return True


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def ingest(config: Config, fetcher: Fetcher | None = None) -> IngestResult:
    """Fetch every selected file into the raw snapshot directory, idempotently."""
    src = config.source
    commit_verified = _verify_local_commit(config) if src.mode == "local" else True
    if fetcher is None:
        fetcher = local_fetch if src.mode == "local" else http_fetch

    raw_root = config.raw_source_dir
    raw_root.mkdir(parents=True, exist_ok=True)
    mpath = manifest_path(config)
    manifest = load_manifest(mpath) or Manifest.new(config, commit_verified)
    manifest.check_matches(config)

    fetched: list[str] = []
    verified: list[str] = []
    for sf in selected_files(config):
        dest = raw_root / sf.path
        entry = manifest.files.get(sf.path)
        if dest.exists() and entry is not None:
            actual = sha256_file(dest)
            if actual != entry["sha256"]:
                raise RawIntegrityError(
                    f"Raw file {dest} has sha256 {actual}, but the manifest recorded {entry['sha256']}. "
                    "Raw files must never be edited. Delete the file and re-run `tennis-data ingest` "
                    "to restore it from the pinned commit."
                )
            verified.append(sf.path)
            continue

        url = source_url(config, sf.path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=dest.parent, prefix=".partial-")
        os.close(fd)
        tmp = Path(tmp_name)
        try:
            try:
                fetcher(url, tmp, src.timeout_seconds)
            except FileNotFoundError as exc:
                raise MissingSourceFileError(_missing_file_message(config, sf, url)) from exc
            digest = sha256_file(tmp)
            if entry is not None and entry["sha256"] != digest:
                raise RawIntegrityError(
                    f"Re-fetched {sf.path} has sha256 {digest}, but the manifest recorded "
                    f"{entry['sha256']} for the same commit. The source is not reproducible; investigate "
                    "before continuing."
                )
            if dest.exists() and sha256_file(dest) != digest:
                raise RawIntegrityError(
                    f"{dest} exists without a manifest entry and differs from the source. Remove it and re-run ingest."
                )
            os.replace(tmp, dest)
        finally:
            tmp.unlink(missing_ok=True)

        if entry is None:
            manifest.files[sf.path] = {
                "kind": sf.kind,
                "family": sf.family,
                "season": sf.season,
                "source_url": url,
                "sha256": digest,
                "bytes": dest.stat().st_size,
                "retrieved_at": _now(),
            }
        fetched.append(sf.path)
        # Persist after each file so an interrupted run keeps its provenance.
        save_manifest(mpath, manifest)

    save_manifest(mpath, manifest)
    return IngestResult(fetched, verified)


def verify_raw(config: Config) -> list[str]:
    """Re-check every manifest entry against disk. Returns a list of problems (empty = OK)."""
    manifest = load_manifest(manifest_path(config))
    if manifest is None:
        return [f"No manifest at {manifest_path(config)}; run `tennis-data ingest` first."]
    problems = []
    for rel, entry in sorted(manifest.files.items()):
        path = config.raw_source_dir / rel
        if not path.exists():
            problems.append(f"missing: {rel}")
        elif sha256_file(path) != entry["sha256"]:
            problems.append(f"checksum mismatch: {rel}")
    return problems


def describe(result: IngestResult) -> str:
    return json.dumps({"fetched": len(result.fetched), "already_present_verified": len(result.verified)})
