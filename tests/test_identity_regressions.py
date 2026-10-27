"""Phase 5A regression: names in another order or split differently must not create new canonical players.

Found on real TML-Database records (ATP codes) against the pinned archive: "Yunchaokete Bu" is stored in the archive
as first="Bu", last="Yunchaokete"; "Mukund Sasikumar" as "Sasi Kumar Mukund"; "M Rifki Fitriadi" as
first="Fitriadi", last="M Rifqi". Before the fix, Phase 4A found no surname candidate and assigned each a new canonical ID,
which would have split their histories. They must now be held as ambiguous for review.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import polars as pl
import pytest

from tennis_data.prospective.config import load_prospective_config
from tennis_data.prospective.identity import Master, classify

T0 = datetime(2026, 1, 1, tzinfo=UTC)
ARCHIVE = [
    (207352, "Bu", "Yunchaokete", "CHN", date(2002, 1, 19)),
    (124154, "Sasi Kumar", "Mukund", "IND", date(1997, 1, 14)),
    (207134, "Fitriadi", "M Rifqi", "INA", date(1999, 1, 23)),  # archive stores it swapped too
    (126888, "Federico", "Bonacia", "ITA", date(1997, 1, 14)),  # same birth date as Sasikumar, unrelated name
    (1, "Jannik", "Sinner", "ITA", date(2001, 8, 16)),
]


def _master():
    df = pl.DataFrame(ARCHIVE, orient="row", schema={"player_id": pl.Int64, "first_name": pl.String,
                                                     "last_name": pl.String, "country": pl.String,
                                                     "birth_date": pl.Date})  # fmt: skip
    reg = pl.DataFrame(schema={"canonical_player_id": pl.Int64, "first_name": pl.String, "last_name": pl.String,
                               "country": pl.String, "birth_date": pl.Date})  # fmt: skip
    return Master.build(df, reg)


def _p(first, last, country=None, dob=None):
    return {"provider": "tml_github", "provider_player_id": f"{first}-{last}", "first_name": first, "last_name": last,
            "country": country, "birth_date": dob, "hand": None, "declared_canonical_id": None, "raw_sha256": "x",
            "retrieved_at": T0}  # fmt: skip


@pytest.mark.parametrize("player, candidates", [
    (_p("Yunchaokete", "Bu", "CHN", date(2002, 1, 19)), [207352]),  # name order swapped (same tokens)
    (_p("Yunchaokete", "Bu", "CHN"), [207352]),  # swapped, even without a birth date
    (_p("Mukund", "Sasikumar", "IND", date(1997, 1, 14)), [124154]),  # swapped and split surname
    (_p("M", "Rifki Fitriadi", "INA", date(1999, 1, 23)), [207134]),  # transliteration, birth date confirms
])  # fmt: skip
def test_variant_names_are_held_not_created(player, candidates):
    d = classify(player, _master(), {}, {}, load_prospective_config())
    assert d["category"] == "ambiguous" and d["method"] == "name_order_or_spelling_variant"
    assert d["candidates"] == candidates and not d["usable"] and d["canonical_player_id"] is None


def test_genuinely_new_players_still_get_new_ids():
    cfg = load_prospective_config()
    m = _master()
    new = classify(_p("Fares", "Zakaria", "EGY", date(2001, 5, 13)), m, {}, {}, cfg)
    assert new["category"] == "unmatched" and new["usable"]
    # a shared birth date alone (unrelated name) is not evidence of the same person
    other = classify(_p("Carlos", "Newman", "ESP", date(1997, 1, 14)), m, {}, {}, cfg)
    assert other["category"] == "unmatched" and other["usable"]
    # ordinary surname matching is unchanged
    s = classify(_p("Jannik", "Sinner", "ITA", date(2001, 8, 16)), m, {}, {}, cfg)
    assert s["category"] == "high_confidence" and s["canonical_player_id"] == 1
