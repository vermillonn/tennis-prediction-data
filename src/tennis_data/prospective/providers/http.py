"""HTTP access for authenticated providers: keys from the environment, rate limits, bounded retries.

Never bypasses access control: 401/403 stop immediately (``AccessDenied``) and are not retried. 429 and 5xx are
retried with exponential backoff up to ``max_retries``; the rate limiter enforces the documented request rate.
"""

from __future__ import annotations

import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime

from tennis_data.prospective.providers.raw_store import RawRecord, RawStore


class ProviderUnavailable(RuntimeError):
    """The provider cannot be used (no credentials, network policy, outage)."""


class AccessDenied(ProviderUnavailable):
    """401/403: authentication or licence does not permit the request. Never retried or worked around."""


class NotVerifiedError(NotImplementedError):
    """The response layout of this provider has not been verified against real responses."""


Transport = Callable[[str, dict, float], tuple[int, bytes]]


def urllib_transport(url: str, headers: dict, timeout: float) -> tuple[int, bytes]:
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (https only, see build_url)
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read() if e.fp else b""
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise ProviderUnavailable(f"network error for {url.split('?')[0]}: {e}") from e


class RateLimiter:
    def __init__(self, per_second: float, clock=time.monotonic, sleep=time.sleep):
        self.interval = 1.0 / per_second if per_second > 0 else 0.0
        self.clock, self.sleep = clock, sleep
        self.last: float | None = None

    def wait(self) -> None:
        now = self.clock()
        if self.last is not None and now - self.last < self.interval:
            self.sleep(self.interval - (now - self.last))
            now = self.clock()
        self.last = now


class HttpClient:
    def __init__(self, provider: str, store: RawStore, *, api_key_env: str, requests_per_second: float,
                 max_retries: int, backoff_seconds: float, transport: Transport = urllib_transport,
                 sleep=time.sleep, clock=time.monotonic, now=lambda: datetime.now(UTC), timeout: float = 30.0):  # fmt: skip
        self.provider, self.store = provider, store
        self.api_key_env = api_key_env
        self.limiter = RateLimiter(requests_per_second, clock=clock, sleep=sleep)
        self.max_retries, self.backoff, self.sleep = max_retries, backoff_seconds, sleep
        self.transport, self.now, self.timeout = transport, now, timeout

    def api_key(self) -> str:
        key = os.environ.get(self.api_key_env)
        if not key:
            raise ProviderUnavailable(f"{self.provider}: set the {self.api_key_env} environment variable")
        return key

    def get(self, url: str, endpoint: str, params: dict, headers: dict | None = None) -> RawRecord:
        if not url.startswith("https://"):
            raise ValueError("only https URLs are allowed")
        attempt = 0
        while True:
            self.limiter.wait()
            status, body = self.transport(url, headers or {}, self.timeout)
            if status in (401, 403):
                raise AccessDenied(f"{self.provider}: HTTP {status} for {endpoint}; check the key and licence")
            if status == 429 or status >= 500:
                if attempt >= self.max_retries:
                    raise ProviderUnavailable(f"{self.provider}: HTTP {status} after {attempt} retries ({endpoint})")
                self.sleep(self.backoff * 2**attempt)
                attempt += 1
                continue
            if status != 200:
                raise ProviderUnavailable(f"{self.provider}: HTTP {status} for {endpoint}")
            return self.store.put(self.provider, endpoint, params, body, self.now(), status)
