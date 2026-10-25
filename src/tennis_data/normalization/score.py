"""Interpret source score strings.

Scores are written from the winner's perspective ("6-4 3-6 7-6(5)"). Outcome
markers observed at the pinned commit: RET, W/O, Walkover, DEF, Def., ABD,
"Played and unfinished", "?"-garbled sets, NA, UNK and nulls.
"""

from __future__ import annotations

import re
from functools import cache

MATCH_STATUSES = ("completed", "retired", "walkover", "default", "abandoned", "unfinished", "unknown")

_SET = re.compile(r"(\d+)-(\d+)(?:\(\d+\))?")
_COMPLETE = re.compile(r"^\d+-\d+(?:\(\d+\))?(?:\s+\d+-\d+(?:\(\d+\))?)*$")


@cache
def match_status(score: str | None) -> str:
    if score is None:
        return "unknown"
    s = score.strip()
    if not s:
        return "unknown"
    upper = s.upper()
    if "W/O" in upper or upper.startswith("WALKOVER"):
        return "walkover"
    if re.search(r"\bDEF\b", upper) or "DEF." in upper:
        return "default"
    if re.search(r"\bRET\b", upper):
        return "retired"
    if re.search(r"\b(ABD|ABN)\b", upper):
        return "abandoned"
    if "UNFINISHED" in upper or "IN PROGRESS" in upper:
        return "unfinished"
    if "?" in s:
        return "unknown"
    if _COMPLETE.match(s):
        return "completed"
    return "unknown"


@cache
def sets_won(score: str | None) -> tuple[int, int] | None:
    """(sets won by winner, sets won by loser) for a fully parseable score, else None."""
    if score is None or not _COMPLETE.match(score.strip()):
        return None
    w = l = 0
    for a, b in _SET.findall(score):
        if int(a) > int(b):
            w += 1
        elif int(b) > int(a):
            l += 1
    return w, l


def consistent_with_best_of(score: str | None, best_of: int | None) -> bool | None:
    """For a completed score: does the winner have exactly the sets needed and the loser fewer?"""
    parsed = sets_won(score)
    if parsed is None or best_of is None:
        return None
    needed = best_of // 2 + 1
    w, l = parsed
    return w == needed and l < needed
