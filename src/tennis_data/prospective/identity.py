"""Cross-source player identity: provider player IDs -> canonical (archive) player IDs.

Evidence, strongest first:

1. a reviewed entry in ``config/identity_overrides.toml``                      -> verified (manual_override)
2. a provider ID that already owns a new-player ID from the registry          -> unmatched (new_player, kept)
3. a provider-declared canonical ID that exists and whose name agrees         -> verified (declared_id)
4. a unique exact name match confirmed by birth date (or, without dates, by
   nationality)                                                               -> high_confidence
5. several surviving candidates, a declared ID whose name disagrees, or only
   an abbreviated/partial name match                                          -> ambiguous
6. no candidate sharing the surname                                           -> unmatched (new canonical ID)

Birth dates that disagree eliminate a candidate. Nationality never eliminates one (players change national
representation); it is recorded as evidence. High-confidence matches are usable only when the policy auto-approves
them or a reviewer adds an override. Ambiguous identities are never resolved automatically: matches involving them
are held out of the state engine until a reviewer decides.

New players receive IDs from a reserved range (``new_player_id_base`` and up) in a deterministic order; assigned IDs
are persisted and never reused or changed, and archive IDs are never altered.
"""

from __future__ import annotations

import json
import re
import tomllib
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from tennis_data.ingestion.sources import PLAYERS_FILE
from tennis_data.prospective.config import ProspectiveConfig
from tennis_data.storage.parquet import write_single

CATEGORIES = ("verified", "high_confidence", "ambiguous", "unmatched")
DECISION_SCHEMA = {
    "provider": pl.String, "provider_player_id": pl.String, "category": pl.String, "canonical_player_id": pl.Int64,
    "method": pl.String, "usable": pl.Boolean, "candidates": pl.List(pl.Int64), "evidence": pl.String,
    "policy_version": pl.String,
}  # fmt: skip
REGISTRY_SCHEMA = {
    "canonical_player_id": pl.Int64, "provider": pl.String, "provider_player_id": pl.String, "first_name": pl.String,
    "last_name": pl.String, "country": pl.String, "birth_date": pl.Date, "policy_version": pl.String,
}  # fmt: skip


def norm(s: str | None) -> str:
    """Accent-, case- and punctuation-insensitive name key ('Auger-Aliassime' == 'auger aliassime')."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    for a, b in (("ß", "ss"), ("ø", "o"), ("đ", "d"), ("ð", "d"), ("ł", "l"), ("æ", "ae"), ("œ", "oe"), ("ı", "i")):
        s = s.replace(a, b)
    s = re.sub(r"[-'’`.,]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def first_compatible(a: str, b: str) -> str | None:
    """'exact', 'initial' (C. vs Carlos), 'prefix' (Alex vs Alexander) or None."""
    a, b = norm(a), norm(b)
    if not a or not b:
        return None  # a missing first name is no evidence of compatibility
    if a == b:
        return "exact"
    ta, tb = a.split(), b.split()
    if (len(ta[0]) == 1 and tb[0].startswith(ta[0])) or (len(tb[0]) == 1 and ta[0].startswith(tb[0])):
        return "initial"
    if ta[0].startswith(tb[0]) or tb[0].startswith(ta[0]):
        return "prefix"
    return None


def surname_keys(last: str) -> set[str]:
    """Keys under which a (possibly compound) surname is indexed: full, each token, and the last token."""
    n = norm(last)
    toks = [t for t in n.split() if t not in ("de", "del", "da", "van", "von", "der", "di", "la", "le")]
    return {n, *toks} - {""}


@dataclass
class Master:
    """Canonical players: the archive master plus previously assigned new players."""

    df: pl.DataFrame  # player_id, first_name, last_name, country, birth_date, source
    by_surname: dict

    @classmethod
    def build(cls, players: pl.DataFrame, registry: pl.DataFrame) -> Master:
        reg = registry.select(pl.col("canonical_player_id").alias("player_id"), "first_name", "last_name", "country",
                              "birth_date", pl.lit("new_player_registry").alias("source"))  # fmt: skip
        df = pl.concat([players.with_columns(pl.lit("archive").alias("source")), reg], how="vertical_relaxed")
        by: dict = {}
        for r in df.iter_rows(named=True):
            for k in surname_keys(r["last_name"]):
                by.setdefault(k, []).append(r)
        return cls(df, by)

    def get(self, pid: int) -> dict | None:
        m = self.df.filter(pl.col("player_id") == pid)
        return m.row(0, named=True) if m.height else None


def load_archive_players(cfg: ProspectiveConfig) -> pl.DataFrame:
    path = cfg.data.raw_source_dir / PLAYERS_FILE
    df = pl.read_csv(path, infer_schema=False)
    return df.select(
        pl.col("player_id").cast(pl.Int64), pl.col("name_first").alias("first_name"),
        pl.col("name_last").alias("last_name"), pl.col("ioc").alias("country"),
        pl.col("dob").str.strptime(pl.Date, "%Y%m%d", strict=False).alias("birth_date"),
    ).unique("player_id", keep="first", maintain_order=True)  # fmt: skip


def load_overrides(path: Path) -> dict[tuple[str, str], dict]:
    if not path.exists():
        return {}
    with path.open("rb") as fh:
        d = tomllib.load(fh)
    out = {}
    for m in d.get("mapping", []):
        key = (m["provider"], str(m["provider_player_id"]))
        if key in out:
            raise ValueError(f"duplicate identity override for {key}")
        out[key] = m
    return out


def surname_match(a: str | None, b: str | None) -> str | None:
    """'exact' (same normalized surname), 'partial' (one compound surname contains the other) or None."""
    na, nb = norm(a), norm(b)
    if not na or not nb:
        return None
    if na == nb:
        return "exact"
    ta, tb = set(na.split()), set(nb.split())
    return "partial" if ta <= tb or tb <= ta else None


def _name_ok(p: dict, c: dict) -> tuple[bool, str | None]:
    surname = surname_match(p["last_name"], c["last_name"])
    first = first_compatible(p["first_name"] or "", c["first_name"] or "")
    return surname is not None and first is not None, first


def classify(p: dict, master: Master, overrides: dict, prior: dict, cfg: ProspectiveConfig) -> dict:
    key = (p["provider"], p["provider_player_id"])
    ev: dict = {}

    def out(category, cid, method, usable, candidates=()):
        return {"provider": key[0], "provider_player_id": key[1], "category": category, "canonical_player_id": cid,
                "method": method, "usable": usable, "candidates": sorted(set(candidates)),
                "evidence": json.dumps(ev, sort_keys=True, default=str), "policy_version": cfg.identity_policy_version}  # fmt: skip

    if key in overrides:
        o = overrides[key]
        cid = int(o["canonical_player_id"])
        if master.get(cid) is None:
            raise ValueError(f"override {key} points to unknown canonical id {cid}")
        ev["reviewer"], ev["note"] = o.get("reviewer"), o.get("note")
        return out("verified", cid, "manual_override", True, [cid])

    declared = p.get("declared_canonical_id")
    if declared is not None:
        c = master.get(int(declared))
        if c is not None:
            ok, first = _name_ok(p, c)
            dob_conflict = (
                p["birth_date"] is not None and c["birth_date"] is not None and p["birth_date"] != c["birth_date"]
            )
            ev.update(declared_id=int(declared), first_name_match=first, dob_conflict=dob_conflict)
            if ok and not dob_conflict:
                return out("verified", int(declared), "declared_id", True, [int(declared)])
            return out("ambiguous", None, "declared_id_contradicted", False, [int(declared)])
        ev["declared_id_unknown"] = int(declared)

    pool = {}
    for k in surname_keys(p["last_name"] or ""):
        for c in master.by_surname.get(k, ()):
            pool[c["player_id"]] = c
    if not pool:
        if cfg.accept_unmatched_as_new:
            return out("unmatched", None, "new_player", True)
        return out("unmatched", None, "no_candidate", False)
    survivors = []
    for cid, c in sorted(pool.items()):
        ok, first = _name_ok(p, c)
        if not ok:
            continue
        if p["birth_date"] is not None and c["birth_date"] is not None and p["birth_date"] != c["birth_date"]:
            continue  # different person
        exact = norm(f"{p['first_name']} {p['last_name']}") == norm(f"{c['first_name']} {c['last_name']}")
        survivors.append((cid, c, first, exact))
    ev["surname_candidates"] = sorted(pool)
    ev["name_candidates"] = [s[0] for s in survivors]
    if not survivors:
        # shares a surname with someone but no compatible first name: could be a misspelling -> never auto-new
        return out("ambiguous", None, "surname_only", False, sorted(pool))
    exact = [s for s in survivors if s[3]]
    if len(exact) == 1:
        cid, c, first, _ = exact[0]
        dob_match = p["birth_date"] is not None and p["birth_date"] == c["birth_date"]
        country_match = bool(p["country"]) and p["country"] == c["country"]
        others = [s[0] for s in survivors if s[0] != cid]
        ev.update(first_name_match=first, exact_name=True, dob_match=dob_match, country_match=country_match,
                  country_differs=bool(p["country"] and c["country"] and p["country"] != c["country"]),
                  other_compatible=others)  # fmt: skip
        # other compatible (non-exact) names are tolerated only when the birth date confirms the exact one
        if dob_match or (not others and p["birth_date"] is None and country_match):
            return out("high_confidence", cid, "name_dob" if dob_match else "name_country",
                       self_approved(cfg), [cid, *others])  # fmt: skip
        return out("ambiguous", None, "unconfirmed_name_match", False, [s[0] for s in survivors])
    if len(survivors) > 1 or len(exact) > 1:
        return out("ambiguous", None, "multiple_candidates", False, [s[0] for s in survivors])
    return out("ambiguous", None, "weak_name_match", False, [survivors[0][0]])


def self_approved(cfg: ProspectiveConfig) -> bool:
    return cfg.auto_approve_high_confidence


class IdentityResolver:
    def __init__(self, cfg: ProspectiveConfig, archive_players: pl.DataFrame | None = None, root: Path | None = None):
        self.cfg = cfg
        self.dir = (root or cfg.prospective_dir) / "identity"
        self.registry_path, self.decisions_path = self.dir / "new_players.parquet", self.dir / "decisions.parquet"
        self.archive = archive_players if archive_players is not None else load_archive_players(cfg)
        self.registry = (pl.read_parquet(self.registry_path) if self.registry_path.exists()
                         else pl.DataFrame(schema=REGISTRY_SCHEMA))  # fmt: skip
        prior = (
            pl.read_parquet(self.decisions_path)
            if self.decisions_path.exists()
            else pl.DataFrame(schema=DECISION_SCHEMA)
        )
        self.prior = {(r["provider"], r["provider_player_id"]): r for r in prior.iter_rows(named=True)}
        self.overrides = load_overrides(cfg.overrides_path)

    def resolve(self, players: pl.DataFrame) -> pl.DataFrame:
        """Classify every provider player (deterministic order) and assign new IDs. Does not write anything."""
        master = Master.build(self.archive, self.registry)
        rows = sorted(players.unique(["provider", "provider_player_id"], keep="first").iter_rows(named=True),
                      key=lambda r: (r["provider"], r["provider_player_id"]))  # fmt: skip
        decisions, new_reg = [], []
        known_new = {(r["provider"], r["provider_player_id"]): r["canonical_player_id"]
                     for r in self.registry.iter_rows(named=True)}  # fmt: skip
        next_id = max([self.cfg.new_player_id_base - 1, *known_new.values()]) + 1
        for p in rows:
            key = (p["provider"], p["provider_player_id"])
            if key in known_new and key not in self.overrides:
                # this provider ID already owns a canonical ID from the new-player registry
                # (recorded exactly as when the ID was first assigned, so re-runs give identical decisions)
                decisions.append({"provider": key[0], "provider_player_id": key[1], "category": "unmatched",
                                  "canonical_player_id": known_new[key], "method": "new_player", "usable": True,
                                  "candidates": [], "evidence": "{}",
                                  "policy_version": self.cfg.identity_policy_version})  # fmt: skip
                continue
            d = classify(p, master, self.overrides, self.prior, self.cfg)
            if d["category"] == "unmatched" and d["usable"]:
                d["canonical_player_id"] = next_id
                new_reg.append({"canonical_player_id": next_id, "provider": key[0], "provider_player_id": key[1],
                                "first_name": p["first_name"], "last_name": p["last_name"], "country": p["country"],
                                "birth_date": p["birth_date"], "policy_version": self.cfg.identity_policy_version})  # fmt: skip
                next_id += 1
            decisions.append(d)
        self._pending_registry = pl.concat([self.registry, pl.DataFrame(new_reg, schema=REGISTRY_SCHEMA)])
        return pl.DataFrame(decisions, schema=DECISION_SCHEMA).sort("provider", "provider_player_id")

    def persist(self, decisions: pl.DataFrame) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        reg = getattr(self, "_pending_registry", self.registry)
        old = {r["canonical_player_id"]: r for r in self.registry.iter_rows(named=True)}
        for r in reg.iter_rows(named=True):  # append-only: existing assignments never change
            if r["canonical_player_id"] in old and old[r["canonical_player_id"]] != r:
                raise AssertionError("new-player registry entries may not change")
        write_single(reg.sort("canonical_player_id"), self.registry_path)
        merged = {(r["provider"], r["provider_player_id"]): r for r in self.prior.values()}
        merged.update({(r["provider"], r["provider_player_id"]): r for r in decisions.iter_rows(named=True)})
        write_single(pl.DataFrame(list(merged.values()), schema=DECISION_SCHEMA).sort("provider", "provider_player_id"),
                     self.decisions_path)  # fmt: skip
        self.registry = reg
        self.prior = merged
