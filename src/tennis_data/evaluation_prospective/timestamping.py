"""RFC 3161 time-stamp tokens: request construction and offline verification with the system ``openssl`` binary.

A time-stamp response binds a message imprint (here: the 64-hex ``checkpoint_hash``, used directly as a SHA-256
digest) to a time asserted by a time-stamping authority (TSA). It is independent evidence only when:

1. the signature verifies against a TSA certificate chain that the *verifier* chose to trust (``ca_files``; the
   evidence never names its own trust anchor), and
2. the imprint equals the checkpoint hash being checked.

``openssl ts -verify`` checks the signature, the certificate chain, the ``timeStamping`` key usage and the imprint.
No Python dependency is added and nothing from a token is executed. If ``openssl`` is missing, verification fails
closed (``verifier_unavailable``): a token is never accepted unverified.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from tennis_data.ledger.hashing import ts

HEX64 = re.compile(r"^[0-9a-f]{64}$")
_TIME = re.compile(r"^Time stamp: (\w{3})\s+(\d{1,2}) (\d{2}):(\d{2}):(\d{2})(?:\.(\d+))? (\d{4}) GMT$", re.M)
_STATUS = re.compile(r"^Status: (.+)$", re.M)
_TSA = re.compile(r"^TSA: (.+)$", re.M)
MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
TIMEOUT_S = 20


def openssl_path() -> str | None:
    return shutil.which("openssl")


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, timeout=TIMEOUT_S, check=False)  # noqa: S603 (fixed argv)


def make_query(digest_hex: str) -> bytes:
    """DER TimeStampReq for a SHA-256 digest, asking the TSA to include its certificate."""
    if not HEX64.match(digest_hex or ""):
        raise ValueError("digest must be 64 lowercase hex characters")
    exe = openssl_path()
    if exe is None:
        raise RuntimeError("openssl is not available: cannot build an RFC 3161 request")
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "q.tsq"
        r = _run([exe, "ts", "-query", "-digest", digest_hex, "-sha256", "-cert", "-out", str(out)])
        if r.returncode != 0:
            raise RuntimeError(f"openssl ts -query failed: {r.stderr.decode(errors='replace')[-300:]}")
        return out.read_bytes()


def parse_gen_time(text: str) -> str | None:
    m = _TIME.search(text)
    if not m:
        return None
    mon, day, hh, mm, ss, frac, year = m.groups()
    us = int((frac or "0")[:6].ljust(6, "0"))
    t = datetime(int(year), MONTHS.index(mon) + 1, int(day), int(hh), int(mm), int(ss), us, tzinfo=UTC)
    return ts(t)


def verify_token(token: bytes, digest_hex: str, ca_files: list[Path]) -> dict:
    """Offline verification of a DER TimeStampResp against the verifier's trusted CA certificates."""
    res = {"ok": False, "status": None, "gen_time": None, "tsa": None, "problem": None}
    if not HEX64.match(digest_hex or ""):
        res["problem"] = "invalid imprint"
        return res
    exe = openssl_path()
    if exe is None:
        res["status"], res["problem"] = "verifier_unavailable", "openssl is not installed"
        return res
    cas = [Path(c) for c in ca_files if Path(c).is_file()]
    if not cas:
        res["status"], res["problem"] = "no_trusted_tsa", "no trusted TSA CA certificate was supplied by the verifier"
        return res
    with tempfile.TemporaryDirectory() as d:
        tok = Path(d) / "token.tsr"
        tok.write_bytes(token)
        bundle = Path(d) / "ca.pem"
        bundle.write_bytes(b"\n".join(c.read_bytes() for c in cas))
        info = _run([exe, "ts", "-reply", "-in", str(tok), "-text"])
        text = info.stdout.decode("utf-8", errors="replace")
        if info.returncode != 0:
            res["status"], res["problem"] = "unreadable_token", "not an RFC 3161 time-stamp response"
            return res
        st = _STATUS.search(text)
        if not st or not st.group(1).startswith("Granted"):
            res["status"], res["problem"] = "not_granted", (st.group(1) if st else "no status")
            return res
        v = _run([exe, "ts", "-verify", "-digest", digest_hex, "-in", str(tok), "-CAfile", str(bundle)])
        out = (v.stdout + v.stderr).decode("utf-8", errors="replace")
        if v.returncode != 0 or "Verification: OK" not in out:
            reason = "imprint mismatch" if "imprint mismatch" in out else "signature or certificate chain not trusted"
            res["status"], res["problem"] = "verification_failed", reason
            return res
    res.update(ok=True, status="verified", gen_time=parse_gen_time(text),
               tsa=(_TSA.search(text).group(1).strip() if _TSA.search(text) else None))  # fmt: skip
    if res["gen_time"] is None:
        res.update(ok=False, status="verification_failed", problem="no generation time in token")
    return res
