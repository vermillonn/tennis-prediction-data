"""Sportradar Tennis v3 (raw retrieval only).

Verified from public documentation only: the URL pattern
``https://api.sportradar.com/tennis/{access_level}/v3/{language_code}/{feed}.{format}``, key-based access, a
trial quota (documented as 1,000 requests per rolling 30 days at 1 QPS) and HTTP 429 when limits are exceeded.

Not verified: any response layout. This environment could not reach api.sportradar.com (network policy) and no
key was supplied, so ``normalize`` refuses to run rather than guess field names. Raw payloads can still be
retrieved and stored immutably once access exists; the mapping must then be written against real responses
and fixture-tested before any record reaches the state engine.
"""

from __future__ import annotations

from tennis_data.prospective.config import HttpProviderConfig
from tennis_data.prospective.providers.http import HttpClient, NotVerifiedError
from tennis_data.prospective.providers.raw_store import RawRecord, RawStore

NAME = "sportradar"


class SportradarProvider:
    name = NAME
    verified = False

    def __init__(self, cfg: HttpProviderConfig, store: RawStore, **client_kw):
        self.cfg = cfg
        self.client = HttpClient(NAME, store, api_key_env=cfg.api_key_env, requests_per_second=cfg.requests_per_second,
                                 max_retries=cfg.max_retries, backoff_seconds=cfg.backoff_seconds, **client_kw)  # fmt: skip

    def url(self, feed: str) -> str:
        return self.cfg.base_url.format(access_level=self.cfg.access_level, language=self.cfg.language, feed=feed)

    def fetch_raw(self, feed: str) -> RawRecord:
        """Retrieve one feed and store it immutably. The key is sent as a header and never stored."""
        key = self.client.api_key()
        return self.client.get(self.url(feed), feed, {"feed": feed}, headers={"x-api-key": key})

    def normalize(self, raw: bytes):
        raise NotVerifiedError("Sportradar response layout not verified; write and fixture-test the mapping first")
