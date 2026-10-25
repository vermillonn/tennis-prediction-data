"""Duplicate and conflicting-record detection."""

from __future__ import annotations

from conftest import match_row


def _reasons(q, match_num):
    return q.filter(q["match_num"] == match_num)["quarantine_reasons"][0].to_list()


def test_exact_duplicate_keeps_first_and_quarantines_copy(validate_rows):
    rows = [match_row(1, 100, 200), match_row(2, 300, 400), match_row(7, 100, 200)]  # 7 = copy of 1
    v, q = validate_rows(rows)
    assert sorted(v["match_num"].to_list()) == [1, 2]
    assert q.height == 1
    assert _reasons(q, 7) == ["duplicate_record"]
    assert q["duplicate_of"][0] == "tour:2020-0001:1"


def test_duplicate_detection_ignores_row_order_of_unrelated_rows(validate_rows):
    rows = [match_row(2, 300, 400), match_row(1, 100, 200), match_row(9, 100, 200), match_row(3, 500, 600)]
    v, q = validate_rows(rows)
    assert q["match_num"].to_list() == [9]  # first occurrence in the file (match_num 1) is kept
    assert v.height == 3


def test_conflicting_records_are_all_quarantined(validate_rows):
    rows = [
        match_row(1, 100, 200),
        match_row(2, 200, 100, score="7-5 6-3"),  # same pair, same knockout round, different result
        match_row(3, 300, 400),
    ]
    v, q = validate_rows(rows)
    assert v["match_num"].to_list() == [3]
    assert sorted(q["match_num"].to_list()) == [1, 2]
    assert all("conflicting_record" in r for r in q["quarantine_reasons"].to_list())


def test_round_robin_rematch_is_legitimate(validate_rows):
    rows = [
        match_row(1, 100, 200, round="RR"),
        match_row(2, 200, 100, round="RR", score="7-5 6-3"),
    ]
    v, q = validate_rows(rows)
    assert v.height == 2 and q.height == 0


def test_player_in_two_matches_of_one_knockout_round(validate_rows):
    rows = [match_row(1, 100, 200), match_row(2, 100, 300), match_row(3, 500, 600)]
    v, q = validate_rows(rows)
    assert v["match_num"].to_list() == [3]
    assert all("player_repeated_in_round" in r for r in q["quarantine_reasons"].to_list())


def test_same_pair_in_different_tournaments_is_fine(validate_rows):
    rows = [match_row(1, 100, 200), match_row(1, 100, 200, tourney_id="2020-0002", tourney_name="Other")]
    v, q = validate_rows(rows)
    assert v.height == 2 and q.height == 0


def test_every_source_row_is_accounted_for(validate_rows):
    rows = [match_row(1, 100, 200), match_row(7, 100, 200), match_row(2, 300, 400, surface="Sand"),
            match_row(3, 500, 600, w_ace="-1"), match_row(4, 700, 800)]  # fmt: skip
    v, q = validate_rows(rows)
    assert v.height + q.height == len(rows)
    assert set(v["match_id"]).isdisjoint(set(q["match_id"]))
