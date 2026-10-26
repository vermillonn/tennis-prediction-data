"""External checkpoint anchoring: publication backends, receipt verification and evidence-based trust levels.

An *anchor record* (``tennis-ledger-anchor-1``) states that a Phase 4B checkpoint was handed to a publication backend:

    {format, ledger_id, checkpoint_sequence, chain_head_hash, checkpoint_schema_version, previous_checkpoint_hash,
     checkpoint_hash, hash_algorithm, checkpoint, publisher {backend, identity}, published_at, receipt}

``published_at`` is the publishing host's clock (a claim, never evidence). ``receipt`` is whatever the backend returned.
A record says nothing about its own trustworthiness: trust levels are recomputed by ``verify_anchor`` from the
receipt evidence each time, never read from a stored label.

Trust levels (weakest to strongest, plus INVALID):

- ``INVALID``: the checkpoint or its evidence contradicts the snapshot, or a retained trusted checkpoint.
- ``LOCAL_ONLY``: the chain verifies internally, but no external evidence verifies (no anchor, fixture anchors,
  unreachable or unverifiable remote, untrusted TSA).
- ``EXTERNALLY_PUBLISHED``: the exact checkpoint bytes were fetched back from an independently hosted location
  (GitHub commit). Publication *time* and the host's administrative independence are not attested.
- ``INDEPENDENTLY_TIMESTAMPED``: an RFC 3161 token, verified against a TSA the verifier trusts, binds the checkpoint
  hash to a time. It proves the checkpoint existed no later than that time; it says nothing about predictions after
  the checkpoint's sequence.

Publication never overwrites: a record file is created with O_EXCL, and the GitHub backend refuses to replace a file
with different content. A failed publication leaves the ledger untouched and writes no receipt.
"""

from __future__ import annotations

import base64
import hashlib
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from tennis_data.evaluation_prospective import timestamping
from tennis_data.evaluation_prospective.integrity import check_checkpoint
from tennis_data.evaluation_prospective.safeio import (
    UntrustedInputError,
    read_bounded,
    resolve_under,
    strict_json,
    write_new,
)
from tennis_data.evaluation_prospective.snapshot import LedgerSnapshot
from tennis_data.ledger.checkpoint import FORMAT as CHECKPOINT_FORMAT
from tennis_data.ledger.hashing import canonical, parse_ts, sha256_hex, ts

ANCHOR_FORMAT = "tennis-ledger-anchor-1"
TRUST_LEVELS = ("INVALID", "LOCAL_ONLY", "EXTERNALLY_PUBLISHED", "INDEPENDENTLY_TIMESTAMPED")
RECEIPT_TYPES = ("local_fixture", "github_contents", "rfc3161")
ANCHOR_KEYS = {"format", "ledger_id", "checkpoint_sequence", "chain_head_hash", "checkpoint_schema_version",
               "previous_checkpoint_hash", "checkpoint_hash", "hash_algorithm", "checkpoint", "publisher",
               "published_at", "receipt"}  # fmt: skip

# (method, url, headers, body, timeout) -> (status, body)
Transport = Callable[[str, str, dict, bytes | None, float], tuple[int, bytes]]


class PublicationError(RuntimeError):
    """Publication failed or its evidence could not be verified; no receipt was recorded."""


def urllib_transport(method: str, url: str, headers: dict, body: bytes | None, timeout: float) -> tuple[int, bytes]:
    if not url.startswith("https://"):
        raise PublicationError("only https URLs are allowed")
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (https enforced above)
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read() if e.fp else b""
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise PublicationError(f"network error for {url.split('?')[0]}: {type(e).__name__}") from e


def checkpoint_bytes(cp: dict) -> bytes:
    """The exact bytes that are published (identical to ``ledger.checkpoint.write_checkpoint``)."""
    return canonical(cp) + b"\n"


# --------------------------------------------------------------------------------------------------- publishers
class LocalFixturePublisher:
    """Deterministic test backend: copies the checkpoint into a local directory. Never external evidence."""

    backend = "local_fixture"

    def __init__(self, root: Path):
        self.root = Path(root)

    def publish(self, cp: dict, evidence_dir: Path) -> dict:
        rel = f"{cp['ledger_id']}/{cp['sequence']:08d}-{cp['checkpoint_hash'][:16]}.json"
        data = checkpoint_bytes(cp)
        target = resolve_under(self.root, rel)
        if target.exists():
            if target.read_bytes() != data:
                raise PublicationError(f"fixture publication {rel} exists with different content")
        else:
            write_new(target, data)
        return {"type": "local_fixture", "path": rel, "content_sha256": sha256_hex(data)}


class GitHubContentsPublisher:
    """Commits the checkpoint file to an explicitly configured repository through the GitHub Contents API.

    Fails closed: refuses to overwrite an existing path with different content, and after the commit fetches the
    file back *at the returned commit* and compares the bytes; any mismatch or error raises and no receipt exists.
    The token is read from the environment at call time and never logged or stored.
    """

    backend = "github_contents"

    def __init__(self, repository: str, branch: str, path_prefix: str, token_env: str,
                 api_base: str = "https://api.github.com", transport: Transport = urllib_transport,
                 timeout: float = 30.0):  # fmt: skip
        if not repository or repository.count("/") != 1:
            raise PublicationError("[publishers.github].repository must be 'owner/repo' (explicitly configured)")
        self.repository, self.branch, self.prefix = repository, branch, path_prefix.strip("/")
        self.token_env, self.api, self.transport, self.timeout = token_env, api_base.rstrip("/"), transport, timeout

    def _headers(self) -> dict:
        tok = os.environ.get(self.token_env)
        if not tok:
            raise PublicationError(f"set the {self.token_env} environment variable (least-privilege token)")
        return {"Authorization": f"Bearer {tok}", "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "tennis-data-ledger-anchor"}  # fmt: skip

    def path_for(self, cp: dict) -> str:
        return f"{self.prefix}/{cp['ledger_id']}/{cp['sequence']:08d}-{cp['checkpoint_hash']}.json"

    def _get(self, path: str, ref: str, headers: dict) -> tuple[int, dict | None]:
        st, body = self.transport("GET", f"{self.api}/repos/{self.repository}/contents/{path}?ref={ref}", headers,
                                  None, self.timeout)  # fmt: skip
        if st == 404:
            return st, None
        if st != 200:
            raise PublicationError(f"GitHub GET {path}: HTTP {st}")
        doc = strict_json(body)
        if not isinstance(doc, dict):
            raise PublicationError(f"GitHub GET {path}: not a file")
        return st, doc

    @staticmethod
    def _content(doc: dict) -> bytes:
        if doc.get("encoding") != "base64" or not isinstance(doc.get("content"), str):
            raise PublicationError("unexpected GitHub contents response")
        return base64.b64decode(doc["content"])

    def publish(self, cp: dict, evidence_dir: Path) -> dict:
        h = self._headers()
        data, path = checkpoint_bytes(cp), self.path_for(cp)
        st, doc = self._get(path, self.branch, h)
        if doc is not None and self._content(doc) != data:
            raise PublicationError(
                f"{path} already exists in {self.repository} with different content; not overwritten"
            )
        if doc is None:
            body = canonical({"message": f"ledger checkpoint {cp['ledger_id']} #{cp['sequence']}",
                              "content": base64.b64encode(data).decode("ascii"), "branch": self.branch})  # fmt: skip
            st, resp = self.transport("PUT", f"{self.api}/repos/{self.repository}/contents/{path}", h, body,
                                      self.timeout)  # fmt: skip
            if st not in (200, 201):
                raise PublicationError(f"GitHub PUT {path}: HTTP {st} (nothing recorded)")
            r = strict_json(resp)
            commit, blob = r.get("commit", {}), r.get("content", {})
        else:  # identical content already published (idempotent re-publication): locate its commit
            st, commits = self.transport("GET", f"{self.api}/repos/{self.repository}/commits?path={path}"
                                         f"&sha={self.branch}&per_page=100", h, None, self.timeout)  # fmt: skip
            lst = strict_json(commits) if st == 200 else []
            if not lst:
                raise PublicationError(f"cannot locate the commit that published {path}")
            commit, blob = lst[-1], {"sha": doc.get("sha")}
        sha = commit.get("sha")
        if not isinstance(sha, str) or len(sha) != 40:
            raise PublicationError("GitHub response has no commit sha")
        _, back = self._get(path, sha, h)
        if back is None or self._content(back) != data:
            raise PublicationError("published content could not be read back identically at the commit")
        committed = (commit.get("committer") or commit.get("commit", {}).get("committer") or {}).get("date")
        return {"type": "github_contents", "repository": self.repository, "branch": self.branch, "path": path,
                "commit_sha": sha, "blob_sha": blob.get("sha"), "content_sha256": sha256_hex(data),
                "api_committed_at": committed}  # fmt: skip


class Rfc3161Publisher:
    """Requests an RFC 3161 token for the checkpoint hash from a configured TSA and verifies it before recording."""

    backend = "rfc3161"

    def __init__(self, tsa_url: str, trusted_ca_files: list[Path], transport: Transport = urllib_transport,
                 timeout: float = 30.0):  # fmt: skip
        if not tsa_url.startswith("https://"):
            raise PublicationError("[publishers.rfc3161].tsa_url must be an explicitly configured https URL")
        self.url, self.cas, self.transport, self.timeout = tsa_url, list(trusted_ca_files), transport, timeout

    def publish(self, cp: dict, evidence_dir: Path) -> dict:
        q = timestamping.make_query(cp["checkpoint_hash"])
        st, token = self.transport("POST", self.url, {"Content-Type": "application/timestamp-query"}, q, self.timeout)
        if st != 200 or not token:
            raise PublicationError(f"TSA returned HTTP {st}")
        v = timestamping.verify_token(token, cp["checkpoint_hash"], self.cas)
        if not v["ok"]:
            raise PublicationError(f"TSA token does not verify ({v['status']}: {v['problem']}); nothing recorded")
        rel = f"tokens/{cp['ledger_id']}/{cp['sequence']:08d}-{cp['checkpoint_hash'][:16]}.tsr"
        target = resolve_under(evidence_dir, rel)
        if not target.exists():
            write_new(target, token)
        elif target.read_bytes() != token:  # keep the first token; store the new one beside it
            rel = f"{rel[:-4]}-{sha256_hex(token)[:12]}.tsr"
            write_new(resolve_under(evidence_dir, rel), token)
        return {"type": "rfc3161", "tsa_url": self.url, "token_file": rel, "token_sha256": sha256_hex(token),
                "imprint": cp["checkpoint_hash"], "hash_algorithm": "sha256"}  # fmt: skip


# ---------------------------------------------------------------------------------------------- records on disk
def make_record(cp: dict, backend: str, identity: str, receipt: dict, published_at: datetime) -> dict:
    return {"format": ANCHOR_FORMAT, "ledger_id": cp["ledger_id"], "checkpoint_sequence": cp["sequence"],
            "chain_head_hash": cp["head_hash"], "checkpoint_schema_version": cp["format"],
            "previous_checkpoint_hash": cp["previous_checkpoint_hash"], "checkpoint_hash": cp["checkpoint_hash"],
            "hash_algorithm": "sha256", "checkpoint": cp, "publisher": {"backend": backend, "identity": identity},
            "published_at": ts(published_at), "receipt": receipt}  # fmt: skip


def check_checkpoint_body(cp) -> None:
    """Only a well-formed Phase 4B checkpoint whose hash matches its body can be published."""
    from tennis_data.evaluation_prospective.integrity import CHECKPOINT_KEYS
    from tennis_data.ledger.checkpoint import checkpoint_hash

    if not isinstance(cp, dict) or set(cp) != CHECKPOINT_KEYS or cp.get("format") != CHECKPOINT_FORMAT:
        raise PublicationError("not a tennis-ledger-checkpoint-1 checkpoint; nothing was published")
    try:
        ok = checkpoint_hash(cp) == cp["checkpoint_hash"]
    except (ValueError, TypeError) as e:
        raise PublicationError(f"checkpoint cannot be hashed: {e}") from e
    if not ok:
        raise PublicationError("checkpoint_hash does not match its body; nothing was published")


def publish(cp: dict, publisher, evidence_dir: Path, identity: str, now: datetime | None = None) -> tuple[dict, Path]:
    """Publish and write the anchor record (new file only). On failure, a failure note is written and it re-raises."""
    when = now or datetime.now(UTC)
    check_checkpoint_body(cp)
    try:
        receipt = publisher.publish(cp, evidence_dir)
    except Exception as e:
        note = canonical({"checkpoint_hash": cp.get("checkpoint_hash"), "backend": publisher.backend,
                          "failed_at": ts(when), "error": f"{type(e).__name__}: {e}"[:500]})  # fmt: skip
        fail = evidence_dir / "failures" / f"{ts(when).replace(':', '')}-{publisher.backend}.json"
        if not fail.exists():
            write_new(fail, note + b"\n")
        raise
    rec = make_record(cp, publisher.backend, identity, receipt, when)
    name = f"{cp['sequence']:08d}-{cp['checkpoint_hash'][:16]}-{publisher.backend}.json"
    path = evidence_dir / "records" / cp["ledger_id"] / name
    data = canonical(rec) + b"\n"
    if path.exists():
        if path.read_bytes() != data:
            path = path.with_name(f"{path.stem}-{sha256_hex(data)[:12]}.json")
            write_new(path, data)
    else:
        write_new(path, data)
    return rec, path


def load_records(directory: Path | None, max_bytes: int) -> list[tuple[str, dict]]:
    """Every ``*.json`` anchor record under ``directory/records`` (sorted; unreadable ones are returned as errors)."""
    out = []
    if directory is None or not (Path(directory) / "records").exists():
        return out
    base = Path(directory) / "records"
    for p in sorted(base.rglob("*.json")):
        rel = p.relative_to(directory).as_posix()
        try:
            out.append((rel, strict_json(read_bounded(p, max_bytes))))
        except UntrustedInputError as e:
            out.append((rel, {"_error": str(e)}))
    return out


# ------------------------------------------------------------------------------------------------- verification
def verify_anchor(rec: dict, snap: LedgerSnapshot, *, evidence_dir: Path | None, fixture_root: Path | None = None,
                  tsa_ca_files: list[Path] = (), online: bool = False, transport: Transport = urllib_transport,
                  max_bytes: int = 1024 * 1024, api_base: str = "https://api.github.com") -> dict:  # fmt: skip
    """Recompute the trust level of one anchor record from its evidence."""
    out = {"checkpoint_sequence": None, "checkpoint_hash": None, "backend": None, "level": "INVALID",
           "receipt_status": None, "attested_time": None, "claimed_published_at": None, "problems": []}  # fmt: skip
    if "_error" in rec:
        out["problems"].append(rec["_error"])
        return out
    if set(rec) != ANCHOR_KEYS or rec.get("format") != ANCHOR_FORMAT:
        out["problems"].append("not a tennis-ledger-anchor-1 record")
        return out
    cp = rec["checkpoint"]
    if (not isinstance(rec["checkpoint_sequence"], int) or isinstance(rec["checkpoint_sequence"], bool)
            or not isinstance(rec["checkpoint_hash"], str) or not isinstance(rec["publisher"], dict)
            or not isinstance(rec["published_at"], str)):  # fmt: skip
        out["problems"].append("anchor record fields have invalid types")
        return out
    out.update(checkpoint_sequence=rec["checkpoint_sequence"], checkpoint_hash=rec["checkpoint_hash"],
               backend=rec["publisher"].get("backend"), claimed_published_at=rec["published_at"])  # fmt: skip
    probs = check_checkpoint(cp, snap)
    if not probs and (rec["ledger_id"], rec["checkpoint_sequence"], rec["chain_head_hash"], rec["checkpoint_hash"],
                      rec["previous_checkpoint_hash"], rec["checkpoint_schema_version"]) != (
            cp["ledger_id"], cp["sequence"], cp["head_hash"], cp["checkpoint_hash"], cp["previous_checkpoint_hash"],
            CHECKPOINT_FORMAT):  # fmt: skip
        probs.append("record fields differ from the embedded checkpoint")
    if rec["hash_algorithm"] != "sha256":
        probs.append("unsupported hash algorithm")
    if probs:
        out["problems"] = probs
        return out
    r = rec["receipt"] if isinstance(rec["receipt"], dict) else {}
    kind = r.get("type")
    data = checkpoint_bytes(cp)
    if kind == "local_fixture":
        status = "fixture_not_found"
        if fixture_root is not None:
            try:
                p = resolve_under(fixture_root, r.get("path", ""))
                if p.is_file() and read_bounded(p, max_bytes) == data:
                    status = "fixture_matches"
                elif p.is_file():
                    status = "fixture_content_differs"
            except UntrustedInputError as e:
                status = f"invalid_path: {e}"
        out["receipt_status"] = status
        if status == "fixture_content_differs":
            out["problems"].append("fixture publication differs from the checkpoint")
            return out
        out["level"] = "LOCAL_ONLY"  # a fixture is never independent evidence
        return out
    if kind == "github_contents":
        out["level"] = "LOCAL_ONLY"
        if r.get("content_sha256") != sha256_hex(data):
            out["problems"].append("receipt content hash differs from the checkpoint")
            out["level"] = "INVALID"
            return out
        if not online:
            out["receipt_status"] = "not_checked_offline"
            return out
        st, body = None, None
        try:
            url = f"{api_base.rstrip('/')}/repos/{r['repository']}/contents/{r['path']}?ref={r['commit_sha']}"
            st, body = transport("GET", url, {"Accept": "application/vnd.github+json",
                                             "User-Agent": "tennis-data-ledger-verify"}, None, 30.0)  # fmt: skip
        except (PublicationError, KeyError, TypeError) as e:
            out["receipt_status"] = f"unreachable: {e}"
            return out
        if st != 200:
            out["receipt_status"] = f"remote_http_{st}"
            return out
        try:
            doc = strict_json(body)
            remote = base64.b64decode(doc["content"]) if doc.get("encoding") == "base64" else None
        except (UntrustedInputError, KeyError, ValueError, TypeError):
            remote = None
        if remote != data:
            out["receipt_status"] = "remote_content_differs"
            out["problems"].append("content at the published commit differs from the checkpoint")
            out["level"] = "INVALID"
            return out
        out["receipt_status"] = "remote_content_matches"
        out["level"] = "EXTERNALLY_PUBLISHED"
        try:  # deletion or history rewriting on the branch is reported (the commit itself may still be served)
            url = f"{api_base.rstrip('/')}/repos/{r['repository']}/contents/{r['path']}?ref={r['branch']}"
            st, _ = transport("GET", url, {"Accept": "application/vnd.github+json",
                                          "User-Agent": "tennis-data-ledger-verify"}, None, 30.0)  # fmt: skip
            if st != 200:
                out["receipt_status"] += "; removed_from_branch"
        except (PublicationError, KeyError, TypeError):
            out["receipt_status"] += "; branch_unreachable"
        return out
    if kind == "rfc3161":
        out["level"] = "LOCAL_ONLY"
        if r.get("imprint") != rec["checkpoint_hash"]:
            out["problems"].append("token imprint is not this checkpoint")
            out["level"] = "INVALID"
            return out
        if evidence_dir is None:
            out["receipt_status"] = "token_missing"
            return out
        try:
            tok = read_bounded(resolve_under(evidence_dir, r.get("token_file", "")), max_bytes)
        except (UntrustedInputError, FileNotFoundError) as e:
            out["receipt_status"] = f"token_missing: {e}"
            return out
        if hashlib.sha256(tok).hexdigest() != r.get("token_sha256"):
            out["receipt_status"] = "token_hash_differs"
            out["problems"].append("token file differs from the receipt")
            out["level"] = "INVALID"
            return out
        v = timestamping.verify_token(tok, rec["checkpoint_hash"], list(tsa_ca_files))
        out["receipt_status"] = v["status"] if v["ok"] else f"{v['status']}: {v['problem']}"
        if v["ok"]:
            out["level"], out["attested_time"] = "INDEPENDENTLY_TIMESTAMPED", v["gen_time"]
        elif v["status"] == "verification_failed" and v["problem"] == "imprint mismatch":
            out["problems"].append("token imprint does not match the checkpoint")
            out["level"] = "INVALID"
        return out
    out["problems"].append(f"unknown receipt type {kind!r}")
    return out


def compare_trusted(trusted: list[tuple[str, dict]], snap: LedgerSnapshot) -> list[dict]:
    """Checkpoints an auditor retained earlier must all agree with the snapshot (no rewrite, fork or truncation)."""
    out = []
    for name, cp in trusted:
        probs = [cp["_error"]] if "_error" in cp else check_checkpoint(cp, snap)
        out.append({"source": name, "sequence": cp.get("sequence"), "checkpoint_hash": cp.get("checkpoint_hash"),
                    "ok": not probs, "problems": probs})  # fmt: skip
    return out


def load_trusted(directory: Path | None, max_bytes: int) -> list[tuple[str, dict]]:
    out = []
    if directory is None or not Path(directory).exists():
        return out
    for p in sorted(Path(directory).glob("*.json")):
        if p.name.endswith(".meta.json"):
            continue
        try:
            out.append((p.name, strict_json(read_bounded(p, max_bytes))))
        except UntrustedInputError as e:
            out.append((p.name, {"_error": str(e)}))
    return out


def coverage(anchors: list[dict], snap: LedgerSnapshot | None = None) -> dict:
    """Snapshot-level trust plus, per event sequence, the strongest verified evidence covering it.

    An event at sequence ``s`` is covered by every valid anchor with ``checkpoint_sequence >= s``. Its attested time
    is the earliest attested time among INDEPENDENTLY_TIMESTAMPED anchors covering it; its published flag is whether
    any EXTERNALLY_PUBLISHED (or stronger) anchor covers it.
    """
    invalid = [a for a in anchors if a["level"] == "INVALID"]
    # conflicting anchors for the same sequence (fork evidence)
    by_seq: dict = {}
    for a in anchors:
        if a["checkpoint_sequence"] is not None:
            by_seq.setdefault(a["checkpoint_sequence"], set()).add(a["checkpoint_hash"])
    forks = sorted(s for s, hs in by_seq.items() if len(hs) > 1)
    valid = [a for a in anchors if a["level"] != "INVALID"]
    ts_anchors = sorted([a["checkpoint_sequence"], a["attested_time"]] for a in valid
                        if a["level"] == "INDEPENDENTLY_TIMESTAMPED")  # fmt: skip
    # a TSA token is not a publication of the bytes: only read-back remote copies count as published
    pub_max = max((a["checkpoint_sequence"] for a in valid if a["level"] == "EXTERNALLY_PUBLISHED"), default=0)
    return {"invalid_anchors": len(invalid), "forked_sequences": forks, "timestamped": ts_anchors,
            "published_through_seq": pub_max,
            "timestamped_through_seq": max((s for s, _ in ts_anchors), default=0)}  # fmt: skip


def evidence_for(seq: int, cov: dict) -> tuple[str, str | None]:
    """(strongest level covering ``seq``, earliest attested time) for one event, given ``coverage``."""
    times = [t for s, t in cov["timestamped"] if s >= seq]
    if times:
        return "INDEPENDENTLY_TIMESTAMPED", min(times, key=parse_ts)
    if cov["published_through_seq"] >= seq:
        return "EXTERNALLY_PUBLISHED", None
    return "LOCAL_ONLY", None


def checkpoint_from_snapshot(snap: LedgerSnapshot, previous: dict | None = None) -> dict:
    """The Phase 4B checkpoint of a snapshot's head, computed from the export alone (no database access)."""
    from tennis_data.ledger.checkpoint import checkpoint_hash
    from tennis_data.ledger.storage import SCHEMA_VERSION

    body = {"format": CHECKPOINT_FORMAT, "ledger_id": snap.ledger_id, "schema_version": SCHEMA_VERSION,
            "sequence": snap.head_seq, "head_hash": snap.head_hash,
            "previous_checkpoint_hash": previous["checkpoint_hash"] if previous else None}  # fmt: skip
    return {**body, "checkpoint_hash": checkpoint_hash(body)}


def make_publisher(cfg, backend: str, transport: Transport = urllib_transport):
    """A publisher for an explicitly enabled backend of config/evaluation.toml (``[publishers.<name>]``)."""
    pubs = cfg.publishers
    if backend == "local_fixture":
        return LocalFixturePublisher(cfg.config_path.parent.parent / pubs.get("local_fixture", {}).get(
            "root", "data/ledger/fixture_publications"))  # fmt: skip
    p = pubs.get("github" if backend == "github_contents" else backend, {})
    if not p.get("enabled"):
        raise PublicationError(f"publisher {backend} is not enabled in {cfg.config_path.name}; nothing was published")
    if backend == "github_contents":
        return GitHubContentsPublisher(p.get("repository", ""), p.get("branch", "ledger-checkpoints"),
                                       p.get("path_prefix", "checkpoints"), p.get("token_env", ""),
                                       p.get("api_base", "https://api.github.com"), transport)  # fmt: skip
    if backend == "rfc3161":
        return Rfc3161Publisher(p.get("tsa_url", ""), [Path(x) for x in p.get("trusted_ca_files", [])], transport)
    raise PublicationError(f"unknown backend {backend}")
