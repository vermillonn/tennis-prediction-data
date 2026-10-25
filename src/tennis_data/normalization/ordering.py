"""Deterministic, winner-independent assignment of player A and player B.

Sorting by player ID would be deterministic but not neutral: IDs are allocated
roughly chronologically, so "lower ID" correlates with age and era. Instead the
choice is taken from a cryptographic hash of the match identifier, which is fixed
by the source, independent of the result, and stable across runs and platforms.
Roughly half of all matches therefore have player A as the winner.
"""

from __future__ import annotations

import hashlib


def a_is_lower_id(match_id: str) -> bool:
    """True if player A is the player with the lower ID for this match."""
    return hashlib.sha256(match_id.encode("utf-8")).digest()[0] % 2 == 0


def assign_players(match_id: str, id_1: int, id_2: int) -> tuple[int, int]:
    """Return (player_a_id, player_b_id). Symmetric in its two ID arguments."""
    lo, hi = (id_1, id_2) if id_1 <= id_2 else (id_2, id_1)
    return (lo, hi) if a_is_lower_id(match_id) else (hi, lo)
