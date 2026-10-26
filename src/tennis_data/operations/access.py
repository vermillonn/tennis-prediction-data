"""Provider access audit: one permitted request per provider, classified into distinct, actionable outcomes.

A provider counts as *verified* only when an actual data request succeeded and the returned schema was inspected
(``DATA_VERIFIED``). A reachable documentation page or a 200 from a host whose data format was not inspected is
only ``REACHABLE``. The outcomes keep different problems apart, because each needs a different remedy:

| status | meaning | remedy |
|---|---|---|
| DATA_VERIFIED | data returned and its schema inspected | none |
| REACHABLE | host answered, data not verified | write a schema check / authorized request |
| NETWORK_BLOCKED | the environment's egress proxy refused the connection | allow the host in the environment's network settings |
| DNS_FAILURE | the host name did not resolve from this environment | network settings / host name |
| CREDENTIALS_ABSENT | the provider requires a key and none is configured | provide the key in the named environment variable |
| AUTH_REJECTED | a configured key was refused (invalid, expired or not entitled) | fix the key or the subscription |
| RATE_LIMITED | HTTP 429 | wait; respect the documented limit |
| NOT_FOUND | HTTP 404 (file or endpoint absent) | the source no longer publishes it |
| SCHEMA_MISMATCH | data returned but not in the expected format | adapter work |
| NOT_PROBED | deliberately not requested (terms do not permit automated access) | none / licence |
| ERROR | any other failure | inspect detail |

Credentials are read from environment variables and attached only to the provider's own API request; their values
are never printed, stored or written to reports (only "present"/"absent"). Probes never retry and never work around
a refusal.
"""

from __future__ import annotations

import io
import os
import socket
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime

import polars as pl

from tennis_data.ledger.hashing import ts
from tennis_data.normalization.schema import SOURCE_MATCH_COLUMNS
from tennis_data.operations.config import Provider

STATUSES = ("DATA_VERIFIED", "REACHABLE", "NETWORK_BLOCKED", "DNS_FAILURE", "CREDENTIALS_ABSENT", "AUTH_REJECTED",
            "RATE_LIMITED", "NOT_FOUND", "SCHEMA_MISMATCH", "NOT_PROBED", "ERROR")  # fmt: skip
MAX_PROBE_BYTES = 64 * 1024 * 1024


class NetworkFailure(Exception):
    def __init__(self, kind: str, detail: str):
        super().__init__(detail)
        self.kind, self.detail = kind, detail


Transport = Callable[[str, dict, float], tuple[int, bytes]]


def urllib_transport(url: str, headers: dict, timeout: float) -> tuple[int, bytes]:
    req = urllib.request.Request(url, headers={"User-Agent": "tennis-data-access-audit", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (https URLs from config)
            return resp.status, resp.read(MAX_PROBE_BYTES)
    except urllib.error.HTTPError as e:
        return e.code, b""
    except urllib.error.URLError as e:
        reason = e.reason
        if isinstance(reason, socket.gaierror):
            raise NetworkFailure("dns", "host name did not resolve") from e
        text = str(reason)
        if "Tunnel connection failed" in text:
            raise NetworkFailure(
                "proxy", f"egress proxy refused the connection ({text.split(':', 1)[-1].strip()})"
            ) from e
        raise NetworkFailure("other", type(reason).__name__) from e
    except (TimeoutError, OSError) as e:
        raise NetworkFailure("other", type(e).__name__) from e


def _with_auth(url: str, auth: str | None, key: str) -> tuple[str, dict]:
    if not auth:
        return url, {}
    where, name = auth.split(":", 1)
    if where == "header":
        return url, {name: key}
    if where == "query":
        sep = "&" if urllib.parse.urlparse(url).query else "?"
        return f"{url}{sep}{urllib.parse.urlencode({name: key})}", {}
    raise ValueError(f"unknown auth style {auth!r}")


def verify_sackmann_csv(body: bytes) -> dict:
    df = pl.read_csv(io.BytesIO(body), infer_schema=False)
    missing = [c for c in SOURCE_MATCH_COLUMNS if c not in df.columns]
    if missing:
        return {"ok": False, "detail": f"missing Sackmann columns {missing[:5]}"}
    dates = [d for d in df["tourney_date"].to_list() if d]
    return {"ok": True, "rows": df.height, "first_tourney_date": min(dates) if dates else None,
            "last_tourney_date": max(dates) if dates else None,
            "rows_with_serve_stats": int(df.filter(pl.col("w_svpt").is_not_null() & (pl.col("w_svpt") != "")).height)}  # fmt: skip


VERIFIERS = {"sackmann_csv": verify_sackmann_csv}


def probe(provider: Provider, transport: Transport = urllib_transport, env: dict | None = None,
          timeout: float = 20.0) -> dict:  # fmt: skip
    env = os.environ if env is None else env
    cred = None
    if provider.credential_env:
        cred = "present" if env.get(provider.credential_env) else "absent"
    out = {"provider": provider.name, "kind": provider.kind, "url": provider.probe or None,
           "credential_env": provider.credential_env, "credentials": cred, "status": None, "http_status": None,
           "detail": "", "data": None}  # fmt: skip
    if not provider.probe:
        out.update(status="NOT_PROBED", detail=provider.terms)
        return out
    url, headers = provider.probe, {}
    if cred == "present":
        url, headers = _with_auth(provider.probe, provider.auth, env[provider.credential_env])
    try:
        status, body = transport(url, headers, timeout)
    except NetworkFailure as e:
        out.update(status={"proxy": "NETWORK_BLOCKED", "dns": "DNS_FAILURE"}.get(e.kind, "ERROR"), detail=e.detail)
        return out
    out["http_status"] = status
    if status in (401, 403):
        if cred == "present":
            out.update(status="AUTH_REJECTED", detail="the configured key was refused")
        elif provider.credential_env:
            out.update(status="CREDENTIALS_ABSENT", detail=f"set {provider.credential_env}")
        else:
            out.update(status="AUTH_REJECTED", detail="the host refused an unauthenticated request")
    elif status == 429:
        out.update(status="RATE_LIMITED", detail="HTTP 429")
    elif status == 404:
        out.update(status="NOT_FOUND", detail="HTTP 404")
    elif status == 200:
        if provider.verify in VERIFIERS:
            try:
                v = VERIFIERS[provider.verify](body)
            except Exception as e:  # noqa: BLE001 (any parse failure is a schema mismatch)
                v = {"ok": False, "detail": type(e).__name__}
            out.update(status="DATA_VERIFIED" if v["ok"] else "SCHEMA_MISMATCH", data=v,
                       detail="" if v["ok"] else v["detail"])  # fmt: skip
        elif provider.credential_env and cred == "absent":
            out.update(status="REACHABLE", detail="answered without credentials; data not verified")
        else:
            out.update(status="REACHABLE", detail="data format not inspected")
    else:
        out.update(status="ERROR", detail=f"HTTP {status}")
    # a 200 without credentials from an API that requires them is not access to data
    if provider.kind == "api" and out["status"] == "REACHABLE" and cred != "present":
        out["detail"] = f"API host reachable; data requires {provider.credential_env}"
    return out


def audit(providers, transport: Transport = urllib_transport, env: dict | None = None,
          now: datetime | None = None) -> dict:  # fmt: skip
    results = [probe(p, transport, env) for p in providers]
    return {"format": "tennis-provider-access-1", "probed_at": ts(now or datetime.now(UTC)), "results": results,
            "verified_sources": [r["provider"] for r in results if r["status"] == "DATA_VERIFIED"]}  # fmt: skip
