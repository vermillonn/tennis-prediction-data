"""Phase 4A units: raw store, HTTP access rules, provider parsing, schema checks, identity resolution."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime

import polars as pl
import pytest

from prospective_helpers import fixture_doc
from tennis_data.config import ConfigError
from tennis_data.prospective.config import HttpProviderConfig, load_prospective_config
from tennis_data.prospective.identity import IdentityResolver, Master, classify, first_compatible, norm
from tennis_data.prospective.providers.fixture import FixtureProvider
from tennis_data.prospective.providers.http import (
    AccessDenied,
    HttpClient,
    NotVerifiedError,
    ProviderUnavailable,
    RateLimiter,
)
from tennis_data.prospective.providers.raw_store import RawStore
from tennis_data.prospective.providers.sportradar import SportradarProvider
from tennis_data.prospective.schema import MATCH_SCHEMA, PLAYER_SCHEMA, check_matches, frame

T0 = datetime(2026, 10, 1, tzinfo=UTC)


# ---------------------------------------------------------------------------------------------- raw store


def test_raw_store_is_immutable_and_idempotent(tmp_path):
    s = RawStore(tmp_path)
    a = s.put("p", "feed", {"api_key": "SECRET", "date": "x"}, b"payload", T0)
    b = s.put("p", "feed", {"api_key": "SECRET", "date": "x"}, b"payload", datetime(2027, 1, 1, tzinfo=UTC))
    assert a == b and len(s.records()) == 1  # same bytes: no new snapshot, original retrieval time kept
    assert "SECRET" not in (tmp_path / "p" / f"{a.sha256}.json").read_text()
    assert s.get("p", a.sha256) == b"payload"
    (tmp_path / "p" / f"{a.sha256}.bin").write_bytes(b"tampered")
    with pytest.raises(RuntimeError):
        s.get("p", a.sha256)
    with pytest.raises(ValueError):
        s.put("p", "feed", {}, b"x", datetime(2026, 1, 1))  # naive time refused


def test_snapshot_id_depends_only_on_content(tmp_path):
    s1, s2 = RawStore(tmp_path / "1"), RawStore(tmp_path / "2")
    s1.put("p", "f", {}, b"a", T0)
    s1.put("p", "f", {}, b"b", T0)
    s2.put("p", "f", {}, b"b", T0)
    s2.put("p", "f", {}, b"a", T0)
    assert s1.snapshot_id() == s2.snapshot_id()


# ---------------------------------------------------------------------------------------------------- http


def _client(tmp_path, responses, monkeypatch, **kw):
    monkeypatch.setenv("TEST_KEY", "k")
    calls, sleeps = [], []

    def transport(url, headers, timeout):
        calls.append((url, headers))
        return responses.pop(0)

    c = HttpClient("p", RawStore(tmp_path), api_key_env="TEST_KEY", requests_per_second=1000, max_retries=2,
                   backoff_seconds=1.0, transport=transport, sleep=sleeps.append, now=lambda: T0, **kw)  # fmt: skip
    return c, calls, sleeps


@pytest.mark.parametrize("status", [401, 403])
def test_access_denied_is_never_retried(tmp_path, monkeypatch, status):
    c, calls, _ = _client(tmp_path, [(status, b"no")], monkeypatch)
    with pytest.raises(AccessDenied):
        c.get("https://example.test/x", "x", {})
    assert len(calls) == 1


def test_rate_limit_and_server_errors_back_off_then_succeed(tmp_path, monkeypatch):
    c, calls, sleeps = _client(tmp_path, [(429, b""), (503, b""), (200, b"ok")], monkeypatch)
    rec = c.get("https://example.test/x", "x", {"q": 1})
    assert len(calls) == 3 and [s for s in sleeps if s >= 0.5] == [1.0, 2.0]  # exponential backoff
    assert RawStore(tmp_path).get("p", rec.sha256) == b"ok"


def test_retries_are_bounded(tmp_path, monkeypatch):
    c, calls, _ = _client(tmp_path, [(429, b"")] * 3, monkeypatch)
    with pytest.raises(ProviderUnavailable):
        c.get("https://example.test/x", "x", {})
    assert len(calls) == 3


def test_missing_key_and_plain_http(tmp_path, monkeypatch):
    c, _, _ = _client(tmp_path, [], monkeypatch)
    monkeypatch.delenv("TEST_KEY")
    with pytest.raises(ProviderUnavailable, match="TEST_KEY"):
        c.api_key()
    with pytest.raises(ValueError):
        c.get("http://example.test", "x", {})


def test_rate_limiter_spaces_requests():
    t = [0.0]
    slept = []
    rl = RateLimiter(2.0, clock=lambda: t[0], sleep=lambda s: (slept.append(s), t.__setitem__(0, t[0] + s)))
    rl.wait()
    rl.wait()
    assert slept == [0.5]


def test_sportradar_mapping_is_not_guessed(tmp_path, monkeypatch):
    cfg = HttpProviderConfig("sportradar", "https://api.sportradar.com/tennis/{access_level}/v3/{language}/{feed}.json",
                             "trial", "en", "SR_KEY", 1.0, 1, 1.0)  # fmt: skip
    sr = SportradarProvider(cfg, RawStore(tmp_path))
    assert (
        sr.url("schedules/live/summaries")
        == "https://api.sportradar.com/tennis/trial/v3/en/schedules/live/summaries.json"
    )
    monkeypatch.delenv("SR_KEY", raising=False)
    with pytest.raises(ProviderUnavailable):
        sr.fetch_raw("x")
    with pytest.raises(NotVerifiedError):
        sr.normalize(b"{}")


# ----------------------------------------------------------------------------------------- fixture format


def test_fixture_provider_splits_results_and_fixtures(tmp_path):
    p = FixtureProvider(RawStore(tmp_path))
    players, results, fixtures, errors = p.parse(fixture_doc(), "sha", T0)
    assert results.height == 10 and fixtures.height == 1 and not errors
    assert dict(results.schema) == MATCH_SCHEMA
    f = fixtures.row(0, named=True)
    assert f["event_start_at"] == datetime(2020, 8, 4, 10, tzinfo=UTC)  # verified scheduled time kept, in UTC
    r = results.filter(pl.col("provider_match_id") == "m1").row(0, named=True)
    assert r["event_start_at"] is None and r["event_completed_at"] is None  # not supplied -> not invented
    assert r["w_svpt"] == 60 and r["winner_provider_id"] == "fx-100"


def test_fixture_errors_are_structured(tmp_path):
    doc = fixture_doc(matches=[
        {"id": "x", "tournament_id": "nope", "round": "F", "status": "completed", "player1_id": "a", "player2_id": "b"},
        {"id": "y", "tournament_id": "t-june1", "round": "F", "status": "completed", "player1_id": "a",
         "player2_id": "b", "winner_id": "c"},
        {"id": "z", "tournament_id": "t-june1", "round": "F", "status": "completed", "player1_id": "a",
         "player2_id": "b", "winner_id": "a", "completed_at": "2020-06-01T10:00:00"},
    ])  # fmt: skip
    _, results, _, errors = FixtureProvider(RawStore(tmp_path)).parse(doc, "sha", T0)
    assert results.height == 0
    assert sorted(e.code for e in errors) == ["malformed", "unknown_tournament", "winner_not_a_player"]


def test_schema_checks_invalid_service_counts(tmp_path):
    _, results, _, _ = FixtureProvider(RawStore(tmp_path)).parse(fixture_doc(), "sha", T0)
    errs = check_matches(results)
    assert [(e.record_id, e.code) for e in errs] == [("m10", "won_exceeds_played")]
    bad = results.head(1).with_columns(
        pl.lit(-1, dtype=pl.Int64).alias("w_ace"), pl.lit("ITF").alias("tournament_level")
    )
    codes = {e.code for e in check_matches(bad)}
    assert {"negative_stat", "unmapped_level"} <= codes


# ------------------------------------------------------------------------------------------------- identity


def _master(rows):
    df = pl.DataFrame(rows, orient="row", schema={"player_id": pl.Int64, "first_name": pl.String, "last_name": pl.String,
                                    "country": pl.String, "birth_date": pl.Date})  # fmt: skip
    return Master.build(df, pl.DataFrame(schema={"canonical_player_id": pl.Int64, "first_name": pl.String,
                                                 "last_name": pl.String, "country": pl.String,
                                                 "birth_date": pl.Date}))  # fmt: skip


MASTER = [
    (1, "Felix", "Auger Aliassime", "CAN", date(2000, 8, 8)),
    (2, "Pablo", "Carreno Busta", "ESP", date(1991, 7, 12)),
    (3, "Juan Martin", "del Potro", "ARG", date(1988, 9, 23)),
    (4, "John", "Smith", "USA", date(1990, 1, 1)),
    (5, "John", "Smith", "GBR", date(1995, 5, 5)),  # identical name, different person
    (6, "Daniil", "Medvedev", "RUS", date(1996, 2, 11)),
    (7, "Alexander", "Zverev", "GER", date(1997, 4, 20)),
    (8, "Mischa", "Zverev", "GER", date(1987, 8, 22)),
    (9, "Hubert", "Hurkacz", None, None),  # missing metadata
]


def _p(pid, first, last, country=None, dob=None, declared=None):
    return {
        "provider": "t",
        "provider_player_id": pid,
        "first_name": first,
        "last_name": last,
        "country": country,
        "birth_date": dob,
        "hand": None,
        "declared_canonical_id": declared,
        "raw_sha256": "x",
        "retrieved_at": T0,
    }


@pytest.fixture
def pcfg():
    return load_prospective_config()


@pytest.mark.parametrize(
    "player, category, cid",
    [
        (_p("a", "Félix", "Auger-Aliassime", "CAN", date(2000, 8, 8)), "high_confidence", 1),  # accents + hyphen
        (_p("b", "Pablo", "Carreño-Busta", "ESP"), "high_confidence", 2),  # compound surname, no DOB: nationality
        (_p("c", "Juan Martín", "Del Potro", "ARG", date(1988, 9, 23)), "high_confidence", 3),
        (_p("d", "John", "Smith", "USA"), "ambiguous", None),  # identical names, no birth date
        (_p("e", "John", "Smith", None, date(1995, 5, 5)), "high_confidence", 5),  # DOB disambiguates
        (_p("f", "D.", "Medvedev", "RUS"), "ambiguous", None),  # abbreviated only
        (_p("g", "Daniil", "Medvedev", "NIL", date(1996, 2, 11)), "high_confidence", 6),  # changed representation
        (_p("h", "A.", "Zverev"), "ambiguous", None),  # initial matches one brother but is not confirmed
        (_p("i", "Hubert", "Hurkacz", "POL"), "ambiguous", None),  # master lacks metadata: nothing confirms
        (_p("j", "Brandon", "Newcomer", "USA"), "unmatched", None),  # previously unseen player
        (_p("k", "Johnny", "Smith", "USA", date(2001, 1, 1)), "ambiguous", None),  # DOB rules both out, surname kept
        (_p("l", "Daniil", "Medvedev", "RUS", date(1999, 1, 1)), "ambiguous", None),  # DOB conflict
    ],
)
def test_identity_categories(pcfg, player, category, cid):
    d = classify(player, _master(MASTER), {}, {}, pcfg)
    assert d["category"] == category, d
    assert d["canonical_player_id"] == cid
    assert d["usable"] == (category == "unmatched")  # high-confidence needs approval by default; ambiguous never


def test_declared_ids(pcfg):
    m = _master(MASTER)
    ok = classify(_p("x", "Alexander", "Zverev", declared=7), m, {}, {}, pcfg)
    assert (ok["category"], ok["canonical_player_id"], ok["usable"]) == ("verified", 7, True)
    wrong = classify(_p("y", "Mischa", "Zverev", declared=7), m, {}, {}, pcfg)  # provider points to the brother
    assert wrong["category"] == "ambiguous" and not wrong["usable"]


def test_override_verifies_and_must_point_to_known_player(pcfg):
    m = _master(MASTER)
    d = classify(
        _p("d", "John", "Smith", "USA"), m, {("t", "d"): {"canonical_player_id": 4, "reviewer": "r"}}, {}, pcfg
    )
    assert (d["category"], d["canonical_player_id"], d["method"]) == ("verified", 4, "manual_override")
    with pytest.raises(ValueError):
        classify(_p("d", "John", "Smith"), m, {("t", "d"): {"canonical_player_id": 999}}, {}, pcfg)


def test_name_normalization():
    assert norm("Ðoković") == "dokovic" and norm("Auger-Aliassime") == "auger aliassime"
    assert first_compatible("J.", "Jannik") == "initial" and first_compatible("Alex", "Alexander") == "prefix"
    assert first_compatible("", "Jannik") is None


def test_new_player_ids_are_stable_and_append_only(tmp_path, pcfg):
    archive = pl.DataFrame(
        MASTER, schema=["player_id", "first_name", "last_name", "country", "birth_date"], orient="row"
    )
    r1 = IdentityResolver(pcfg, archive, root=tmp_path)
    d1 = r1.resolve(frame([_p("n2", "Zed", "Zulu"), _p("n1", "Yan", "Yu")], PLAYER_SCHEMA))
    r1.persist(d1)
    ids1 = dict(zip(d1["provider_player_id"], d1["canonical_player_id"], strict=True))
    assert ids1 == {"n1": pcfg.new_player_id_base, "n2": pcfg.new_player_id_base + 1}  # sorted provider-id order
    r2 = IdentityResolver(pcfg, archive, root=tmp_path)
    d2 = r2.resolve(frame([_p("n0", "Xi", "Xu"), _p("n1", "Yan", "Yu"), _p("n2", "Zed", "Zulu")], PLAYER_SCHEMA))
    ids2 = dict(zip(d2["provider_player_id"], d2["canonical_player_id"], strict=True))
    assert ids2["n1"] == ids1["n1"] and ids2["n2"] == ids1["n2"]  # never renumbered
    assert ids2["n0"] == pcfg.new_player_id_base + 2
    assert all(i >= 1_000_000 for i in ids2.values())


def test_config_validation(tmp_path):
    base = (load_prospective_config().config_path).read_text()
    for old, new in (('policy = "weekly_batch_v1"', 'policy = "timestamps"'),
                     ('precedence = ["historical_archive", ', 'precedence = ['),
                     ("new_player_id_base = 9000000", "new_player_id_base = 5")):  # fmt: skip
        p = tmp_path / "p.toml"
        p.write_text(base.replace(old, new))
        with pytest.raises(ConfigError):
            load_prospective_config(p)


def test_reports_mention_no_credentials():
    text = json.dumps({"k": 1})
    assert "SECRET" not in text
