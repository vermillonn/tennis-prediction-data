"""Independent verification of an audit package: needs only the package, the verifier's trust anchors and this code.

Checks, in order (each reported, machine-readable):

1. ``manifest``: the manifest parses, every listed path is safe and present with the listed SHA-256, and no unlisted
   file is present; optionally the manifest hash equals one obtained independently (e.g. a published reference).
2. ``inputs``: the ledger export and outcome snapshot parse under the strict loaders.
3. ``protocol``: optionally, the package protocol equals the verifier's own protocol configuration.
4. ``recompute``: chain integrity, checkpoint and anchor evidence, eligibility, outcome reconciliation, metrics,
   calibration and uncertainty are recomputed from the inputs; ``results.json`` must be byte-identical.
5. ``reports``: every report is re-rendered from the recomputed results and must be byte-identical.

Summary metrics are never taken from a report or from ``results.json``: they are recomputed. The original SQLite
ledger is not needed.
"""

from __future__ import annotations

from pathlib import Path

from tennis_data.evaluation_prospective import anchoring
from tennis_data.evaluation_prospective.config import DEFAULT_LIMITS, Limits
from tennis_data.evaluation_prospective.pipeline import PACKAGE_FORMAT, evaluate_package_inputs, results_bytes
from tennis_data.evaluation_prospective.report import render_all
from tennis_data.evaluation_prospective.safeio import UntrustedInputError, read_bounded, resolve_under, strict_json
from tennis_data.evaluation_prospective.snapshot import SnapshotChanged
from tennis_data.ledger.hashing import canonical, sha256_hex


def verify_package(pkg: Path, *, tsa_ca_files: list[Path] = (), online: bool = False,
                   transport=anchoring.urllib_transport, expected_manifest_sha256: str | None = None,
                   protocol: dict | None = None, limits: Limits = DEFAULT_LIMITS) -> dict:  # fmt: skip
    pkg = Path(pkg)
    checks: list[dict] = []

    def check(name, ok, detail=""):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})
        return ok

    def done(extra=None):
        return {"ok": all(c["ok"] for c in checks), "package": pkg.name, "checks": checks, **(extra or {})}

    try:
        mbytes = read_bounded(pkg / "manifest.json", limits.max_receipt_bytes * 64)
        man = strict_json(mbytes)
    except (UntrustedInputError, FileNotFoundError) as e:
        check("manifest", False, f"unreadable manifest: {e}")
        return done()
    if expected_manifest_sha256 is not None:
        check("manifest_reference", sha256_hex(mbytes) == expected_manifest_sha256,
              "manifest hash equals the independently obtained reference")  # fmt: skip
    if not isinstance(man, dict) or man.get("format") != PACKAGE_FORMAT or not isinstance(man.get("files"), dict):
        check("manifest", False, "not a tennis-evaluation-package-1 manifest")
        return done()
    bad = []
    for rel, h in sorted(man["files"].items()):
        try:
            p = resolve_under(pkg, rel)
            if p.is_symlink() or not p.is_file():
                bad.append(f"{rel}: missing")
            elif sha256_hex(read_bounded(p, limits.max_export_bytes)) != h:
                bad.append(f"{rel}: content differs from the manifest")
        except UntrustedInputError as e:
            bad.append(f"{rel}: {e}")
    listed = set(man["files"]) | {"manifest.json"}
    extra = sorted(p.relative_to(pkg).as_posix() for p in pkg.rglob("*")
                   if (p.is_file() or p.is_symlink()) and p.relative_to(pkg).as_posix() not in listed)  # fmt: skip
    bad += [f"{x}: not listed in the manifest" for x in extra]
    check("manifest", not bad, "; ".join(bad[:20]) or f"{len(man['files'])} files match their hashes")
    if bad:
        return done()
    try:
        prot = strict_json(read_bounded(pkg / "inputs" / "protocol.json", limits.max_receipt_bytes))
    except (UntrustedInputError, FileNotFoundError) as e:
        check("inputs", False, f"protocol: {e}")
        return done()
    if protocol is not None:
        check("protocol", canonical(protocol) == canonical(prot),
              "package protocol equals the verifier's protocol configuration")  # fmt: skip
    idn = man.get("identity") or {}
    cas_now = sorted(sha256_hex(Path(c).read_bytes()) for c in tsa_ca_files)
    if cas_now != idn.get("tsa_trust_anchor_sha256", []):
        check("trust_anchors", False, "the verifier's TSA trust anchors differ from the evaluation's: timestamp levels "
              "are recomputed under the verifier's anchors and may legitimately differ")  # fmt: skip
    if ("online" if online else "offline") != idn.get("anchor_verification"):
        check("anchor_mode", False, f"package anchors were verified {idn.get('anchor_verification')}; this "
              f"verification is {'online' if online else 'offline'}")  # fmt: skip
    try:
        res = evaluate_package_inputs(pkg, tsa_ca_files=tsa_ca_files, online=online, transport=transport,
                                      cutoff=idn.get("evaluation_cutoff"), limits=limits, protocol=prot)  # fmt: skip
    except (UntrustedInputError, SnapshotChanged, ValueError, KeyError, TypeError) as e:
        check("inputs", False, f"inputs do not load: {e}")
        return done()
    check("inputs", True, "ledger export, outcome snapshot and evidence load under the strict parsers")
    rb = results_bytes(res)
    same = rb == (pkg / "results.json").read_bytes()
    check("recompute", same and sha256_hex(rb) == man.get("results_sha256"),
          "recomputed results are byte-identical to results.json" if same else
          "recomputed results differ from results.json (metrics, classification or evidence changed)")  # fmt: skip
    rendered = render_all(res, man.get("provenance") or {})
    diffs = [n for n, data in rendered.items() if not (pkg / "reports" / n).is_file()
             or (pkg / "reports" / n).read_bytes() != data]  # fmt: skip
    check(
        "reports", not diffs, f"re-rendered reports differ: {diffs}" if diffs else "all reports re-render identically"
    )
    integ = res["integrity"]
    check("chain", integ["ok"] is not False, "ledger chain verified" if integ["ok"] else
          ("no ledger in package" if integ["ok"] is None else f"chain invalid: {integ['issues'][:3]}"))  # fmt: skip
    check("evidence", integ.get("evidence_ok", True) is not False,
          "anchors and retained checkpoints consistent with the snapshot")  # fmt: skip
    summary = {"ledger_id": idn.get("ledger_id"), "head_hash": idn.get("ledger_head_hash"),
               "primary_n": res["scores"]["primary"]["n"], "operational_n": res["scores"]["operational"]["n"],
               "primary_delta_ll": (res["scores"]["primary"]["paired"] or {}).get("log_loss_diff"),
               "operational_delta_ll": (res["scores"]["operational"]["paired"] or {}).get("log_loss_diff"),
               "snapshot_head_evidence": res["anchoring"]["snapshot_head_evidence"]}  # fmt: skip
    return done({"recomputed": summary})
