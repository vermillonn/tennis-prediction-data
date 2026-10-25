"""Which archive files a configuration selects, and how source file names are parsed."""

from __future__ import annotations

import re
from dataclasses import dataclass

from tennis_data.config import Config

ATP_DIR = "atp"

FAMILY_PATTERNS = {
    "tour": "atp_matches_{year}.csv",
    "qual_chall": "atp_matches_qual_chall_{year}.csv",
    "futures": "atp_matches_futures_{year}.csv",
}
_FAMILY_REGEX = {
    "tour": re.compile(r"^atp_matches_(\d{4})\.csv$"),
    "qual_chall": re.compile(r"^atp_matches_qual_chall_(\d{4})\.csv$"),
    "futures": re.compile(r"^atp_matches_futures_(\d{4})\.csv$"),
}

# Coverage observed at the pinned commit; used only to make error messages actionable.
KNOWN_COVERAGE = {"tour": (1968, 2026), "qual_chall": (1978, 2026), "futures": (1991, 2026)}

PLAYERS_FILE = f"{ATP_DIR}/atp_players.csv"
RANKINGS_FILES = tuple(f"{ATP_DIR}/atp_rankings_{d}.csv" for d in ("70s", "80s", "90s", "00s", "10s", "20s", "current"))
# Licence and documentation shipped with the archive, kept for provenance.
METADATA_FILES = ("LICENSE", "README.md", f"{ATP_DIR}/UPSTREAM_README.md", f"{ATP_DIR}/matches_data_dictionary.txt")


@dataclass(frozen=True)
class SourceFile:
    path: str  # repository-relative path, e.g. "atp/atp_matches_2024.csv"
    kind: str  # "matches" | "players" | "rankings" | "metadata"
    family: str | None = None
    season: int | None = None


def parse_match_filename(name: str) -> tuple[str, int] | None:
    """Return (family, season) for a singles match file name, else None."""
    for family, regex in _FAMILY_REGEX.items():
        m = regex.match(name)
        if m:
            return family, int(m.group(1))
    return None


def selected_files(config: Config) -> list[SourceFile]:
    sel = config.selection
    files = [SourceFile(p, "metadata") for p in METADATA_FILES]
    for family in sel.families:
        for year in sel.years(family):
            name = FAMILY_PATTERNS[family].format(year=year)
            files.append(SourceFile(f"{ATP_DIR}/{name}", "matches", family, year))
    if sel.include_players:
        files.append(SourceFile(PLAYERS_FILE, "players"))
    if sel.include_rankings:
        files.extend(SourceFile(p, "rankings") for p in RANKINGS_FILES)
    return files
