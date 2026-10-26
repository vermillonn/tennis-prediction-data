"""The trusted ledger writer: validation, idempotency, mode gates, timing checks and atomic appends.

Every append runs in one ``BEGIN IMMEDIATE`` transaction: validate → check idempotency → read the chain head (the
write lock is already held) → assign the next sequence → hash → insert → commit. Any exception rolls the whole
transaction back, so no partial event and no half-advanced head can exist.

Commit times come from the writer's clock, never from the caller. ``SystemClock`` is the only clock trusted for
PROSPECTIVE records; events written with an injected clock carry ``audit.clock = "injected"``.
"""

from __future__ import annotations

import json
import sqlite3
import tomllib
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

from tennis_data.config import PROJECT_ROOT, ConfigError
from tennis_data.ledger import storage
from tennis_data.ledger.events import (
    NAMESPACES,
    ORIENTATION_RULE,
    EventValidationError,
    check_mode_namespace,
    check_probabilities,
    validate_payload,
)
from tennis_data.ledger.hashing import canonical, content_hash, event_hash, genesis_hash, parse_ts, sha256_hex, ts
from tennis_data.ledger.lifecycle import MatchState, TransitionError, apply
from tennis_data.normalization.ordering import assign_players
from tennis_data.prospective.config import ProspectiveConfig, load_prospective_config

DEFAULT_LEDGER_CONFIG = PROJECT_ROOT / "config" / "ledger.toml"


EVENT_PREFIX = {"PREDICTION_COMMITTED": "pred", "MATCH_METADATA_CORRECTED": "corr", "MATCH_CANCELLED": "canc",
                "MATCH_POSTPONED": "post", "MATCH_RESULT_RECORDED": "rslt", "PREDICTION_VOIDED": "void",
                "LEDGER_CHECKPOINT_CREATED": "ckpt"}  # fmt: skip


class SystemClock:
    """Trusted operational clock (the host's UTC time)."""

    trusted = True

    def now(self) -> datetime:
        return datetime.now(UTC)


class FixedClock:
    """Deterministic clock for tests and synthetic artifacts. Never trusted for PROSPECTIVE records."""

    trusted = False

    def __init__(self, start: datetime, step: timedelta = timedelta(seconds=1)):
        self.t, self.step = start, step

    def now(self) -> datetime:
        t = self.t
        self.t = t + self.step
        return t


class LedgerError(RuntimeError):
    pass


class PredictionRejected(LedgerError):
    """The request may not be recorded (mode gate, timing, identity, model or validation)."""


class IdempotencyConflict(LedgerError):
    """Same idempotency key, different content. The original record is kept; the attempt is logged."""


@dataclass(frozen=True)
class LedgerConfig:
    ledger_path: Path
    export_dir: Path
    checkpoint_dir: Path
    reports_dir: Path
    schema_version: str
    probability_tolerance: float
    busy_timeout_ms: int
    forecast_slots: tuple[str, ...]
    max_schedule_age_hours: float
    prospective: ProspectiveConfig


def load_ledger_config(path: str | Path | None = None, root: str | Path | None = None,
                       prospective: ProspectiveConfig | None = None) -> LedgerConfig:  # fmt: skip
    p = Path(path) if path else DEFAULT_LEDGER_CONFIG
    if not p.exists():
        raise ConfigError(f"Ledger config not found: {p}")
    base = Path(root) if root else PROJECT_ROOT
    with p.open("rb") as fh:
        d = tomllib.load(fh)

    def resolve(x: str) -> Path:
        q = Path(x)
        return q if q.is_absolute() else base / q

    o, lg, pr = d.get("outputs", {}), d.get("ledger", {}), d.get("prospective", {})
    cfg = LedgerConfig(
        ledger_path=resolve(o.get("ledger_path", "data/ledger/ledger.sqlite")),
        export_dir=resolve(o.get("export_dir", "data/ledger/export")),
        checkpoint_dir=resolve(o.get("checkpoint_dir", "data/ledger/checkpoints")),
        reports_dir=resolve(o.get("reports_dir", "reports/ledger")),
        schema_version=lg.get("schema_version", storage.SCHEMA_VERSION),
        probability_tolerance=float(lg.get("probability_tolerance", 1e-12)),
        busy_timeout_ms=int(lg.get("busy_timeout_ms", 30000)),
        forecast_slots=tuple(lg.get("forecast_slots", ["pre_match"])),
        max_schedule_age_hours=float(pr.get("max_schedule_age_hours", 24)),
        prospective=prospective
        or load_prospective_config(
            resolve(d.get("prospective_config", "config/prospective.toml")), root=base
        ),  # fmt: skip
    )
    if cfg.schema_version != storage.SCHEMA_VERSION:
        raise ConfigError(f"ledger schema {cfg.schema_version} is not supported (expected {storage.SCHEMA_VERSION})")
    return cfg


@dataclass(frozen=True)
class PredictionRequest:
    """Everything a prediction commitment states except what the ledger itself assigns (commit time, sequence,
    hashes, idempotency key, timing flags)."""

    mode: str  # SYNTHETIC | HISTORICAL_REPLAY | PROSPECTIVE
    forecast_slot: str
    match_id: str
    player_a_id: int
    player_b_id: int
    tournament_id: str
    surface: str | None
    round: str | None
    best_of: int | None
    scheduled_start_at: str | None  # canonical UTC timestamp, from the schedule source
    scheduled_start_verified: bool
    schedule_snapshot_id: str | None
    schedule_retrieved_at: str | None
    model_id: str
    model_version: str
    frozen_hash: str
    configuration_hash: str
    feature_schema_version: str
    state_snapshot_id: str
    feature_state_hash: str
    source_snapshots: dict
    prediction_created_at: str
    prediction_cutoff: str | None
    probability_a: float
    probability_b: float
    baseline_elo_probability_a: float
    feature_availability: dict
    freshness_status: str
    freshness: dict = field(default_factory=dict)
    provider_match_ids: dict = field(default_factory=dict)
    inference_engine_version: str = ""
    git_commit: str | None = None


# Fields that make two submissions "the same forecast". Creation time and git commit are excluded so that an
# identical re-submission is recognized; anything that could change the probability is included.
CONTENT_FIELDS = (
    "mode", "forecast_slot", "match_id", "player_a_id", "player_b_id", "tournament_id", "surface", "round", "best_of",
    "scheduled_start_at", "scheduled_start_verified", "schedule_snapshot_id", "model_id", "model_version",
    "frozen_hash", "configuration_hash", "feature_schema_version", "state_snapshot_id", "feature_state_hash",
    "source_snapshots", "prediction_cutoff", "probability_a", "probability_b", "baseline_elo_probability_a",
    "feature_availability", "freshness_status", "provider_match_ids", "inference_engine_version",
)  # fmt: skip


def idempotency_key(req: PredictionRequest) -> str:
    """Match identity + mode + forecast slot + frozen model identity + schedule version."""
    return content_hash({"match_id": req.match_id, "mode": req.mode, "forecast_slot": req.forecast_slot,
                         "model_id": req.model_id, "frozen_hash": req.frozen_hash,
                         "configuration_hash": req.configuration_hash,
                         "schedule_snapshot_id": req.schedule_snapshot_id})  # fmt: skip


def request_content_hash(req: PredictionRequest) -> str:
    d = asdict(req)
    return content_hash({k: d[k] for k in CONTENT_FIELDS})


@dataclass(frozen=True)
class Receipt:
    ledger_id: str
    seq: int
    event_id: str
    event_hash: str
    previous_hash: str
    recorded_at: str
    idempotency_key: str | None
    status: str  # "committed" or "existing"


def check_prospective_timing(req: PredictionRequest, now: datetime, cfg: LedgerConfig) -> None:
    """Pre-match rules for PROSPECTIVE records (applied after the status gate)."""
    if not req.scheduled_start_verified or req.scheduled_start_at is None:
        raise PredictionRejected("no verified scheduled start: a prospective record cannot be certified pre-match")
    start = parse_ts(req.scheduled_start_at)
    if req.prediction_cutoff is None:
        raise PredictionRejected("prospective records need a prediction cutoff")
    cutoff = parse_ts(req.prediction_cutoff)
    if cutoff > start:
        raise PredictionRejected("prediction cutoff is after the scheduled start")
    if now >= cutoff:
        raise PredictionRejected("commit time is not before the prediction cutoff")
    if now >= start:
        raise PredictionRejected("commit time is not before the scheduled start")
    if parse_ts(req.prediction_created_at) > now:
        raise PredictionRejected("prediction creation time is in the future of the ledger clock")
    if req.schedule_retrieved_at is None:
        raise PredictionRejected("schedule snapshot time unknown")
    if now - parse_ts(req.schedule_retrieved_at) > timedelta(hours=cfg.max_schedule_age_hours):
        raise PredictionRejected("schedule snapshot is too old")
    if req.freshness_status == "fail":
        raise PredictionRejected("model/state freshness fails")


class LedgerWriter:
    def __init__(self, cfg: LedgerConfig, clock=None, path: Path | None = None):
        self.cfg = cfg
        self.clock = clock or SystemClock()
        self.path = Path(path or cfg.ledger_path)
        self.con = storage.connect(self.path, cfg.busy_timeout_ms)

    def close(self):
        self.con.close()

    # ---------------------------------------------------------------------------------------------- lifecycle
    def init(self, ledger_id: str | None = None) -> dict:
        storage.create_schema(self.con)
        row = self.con.execute("SELECT * FROM ledger_meta").fetchone()
        if row is not None:
            if ledger_id is not None and row["ledger_id"] != ledger_id:
                raise LedgerError(f"ledger already initialized as {row['ledger_id']}")
            return dict(row)
        lid = ledger_id or f"ledger-{uuid.uuid4()}"
        self.con.execute("INSERT INTO ledger_meta VALUES (1, ?, ?, ?, ?)",
                         (lid, storage.SCHEMA_VERSION, genesis_hash(lid), ts(self.clock.now())))  # fmt: skip
        return storage.meta(self.con)

    # ----------------------------------------------------------------------------------------------- appends
    def _append(self, event_type: str, match_id: str | None, payload: dict, *, mode=None, idem=None, pa=None,
                pb=None, ref=None, event_id=None, extra_check=None) -> Receipt:  # fmt: skip
        con = self.con
        con.execute("BEGIN IMMEDIATE")
        try:
            meta = storage.meta(con)
            seq, prev = storage.head(con)
            seq += 1
            now = self.clock.now()
            recorded = ts(now)
            last = con.execute("SELECT recorded_at FROM events ORDER BY seq DESC LIMIT 1").fetchone()
            if last is not None and recorded < last["recorded_at"]:
                raise LedgerError(
                    f"clock regression: {recorded} is earlier than the chain head ({last['recorded_at']})"
                )
            if extra_check is not None:
                payload = extra_check(now, payload)
            validate_payload(event_type, payload, self.cfg.probability_tolerance)
            if match_id is not None:
                _check_transition(con, match_id, {"seq": seq, "event_type": event_type, "event_id": event_id or "?",
                                                  "recorded_at": recorded, "match_id": match_id, "payload": payload,
                                                  "ref_event_id": ref})  # fmt: skip
            eid = event_id or f"{EVENT_PREFIX[event_type]}-" + sha256_hex(
                canonical([meta["ledger_id"], seq, event_type, match_id, payload]))[:32]  # fmt: skip
            body = {"seq": seq, "event_id": eid, "event_type": event_type, "schema_version": storage.SCHEMA_VERSION,
                    "recorded_at": recorded, "match_id": match_id, "mode": mode, "idempotency_key": idem,
                    "probability_a": pa, "probability_b": pb, "ref_event_id": ref, "payload": payload,
                    "refs": {"ref_event_id": ref}, "previous_hash": prev}  # fmt: skip
            h = event_hash(body, prev)
            con.execute(
                f"INSERT INTO events ({', '.join(storage.COLUMNS)}) VALUES ({', '.join('?' * len(storage.COLUMNS))})",
                (seq, eid, event_type, storage.SCHEMA_VERSION, recorded, match_id, mode, idem, pa, pb, ref,
                 canonical(payload).decode("utf-8"), canonical(body["refs"]).decode("utf-8"), prev, h),
            )  # fmt: skip
            con.execute("COMMIT")
        except BaseException:
            con.execute("ROLLBACK")
            raise
        return Receipt(meta["ledger_id"], seq, eid, h, prev, recorded, idem, "committed")

    def commit_prediction(self, req: PredictionRequest) -> Receipt:
        self._precheck(req)
        idem = idempotency_key(req)
        chash = request_content_hash(req)
        existing = self._existing(idem)
        if existing is not None:
            return self._resolve_existing(existing, idem, chash, req)
        pcfg = self.cfg.prospective
        trusted = getattr(self.clock, "trusted", False)

        def finalize(now: datetime, _payload):
            if req.mode == "PROSPECTIVE":
                if not trusted:
                    raise PredictionRejected("PROSPECTIVE records require the trusted system clock")
                check_prospective_timing(req, now, self.cfg)
            sched = parse_ts(req.scheduled_start_at) if req.scheduled_start_at else None
            return {
                "match": {"match_id": req.match_id, "provider_match_ids": dict(req.provider_match_ids),
                          "player_a_id": req.player_a_id, "player_b_id": req.player_b_id,
                          "tournament_id": req.tournament_id, "surface": req.surface, "round": req.round,
                          "best_of": req.best_of, "scheduled_start_at": req.scheduled_start_at,
                          "scheduled_start_verified": req.scheduled_start_verified,
                          "schedule_snapshot_id": req.schedule_snapshot_id,
                          "schedule_retrieved_at": req.schedule_retrieved_at},
                "prediction": {"model_id": req.model_id, "model_version": req.model_version,
                               "frozen_hash": req.frozen_hash, "configuration_hash": req.configuration_hash,
                               "feature_schema_version": req.feature_schema_version,
                               "state_snapshot_id": req.state_snapshot_id, "feature_state_hash": req.feature_state_hash,
                               "source_snapshots": dict(req.source_snapshots),
                               "prediction_created_at": req.prediction_created_at,
                               "prediction_cutoff": req.prediction_cutoff, "probability_a": req.probability_a,
                               "probability_b": req.probability_b,
                               "baseline_elo_probability_a": req.baseline_elo_probability_a,
                               "feature_availability": dict(req.feature_availability),
                               "freshness_status": req.freshness_status, "freshness": dict(req.freshness)},
                "audit": {"mode": req.mode, "forecast_slot": req.forecast_slot,
                          "inference_engine_version": req.inference_engine_version, "git_commit": req.git_commit,
                          "ledger_schema_version": storage.SCHEMA_VERSION, "idempotency_key": idem,
                          "content_hash": chash, "orientation": ORIENTATION_RULE,
                          "clock": "system" if trusted else "injected",
                          "prospective_gate": [pcfg.backfill_status, pcfg.prospective_predictions],
                          "timing": {"generated_before_scheduled_start":
                                     None if sched is None else parse_ts(req.prediction_created_at) < sched,
                                     "committed_before_scheduled_start": None if sched is None else now < sched}},
            }  # fmt: skip

        try:
            return self._append("PREDICTION_COMMITTED", req.match_id, {}, mode=req.mode, idem=idem,
                                pa=req.probability_a, pb=req.probability_b, event_id=f"pred-{idem[:32]}",
                                extra_check=finalize)  # fmt: skip
        except sqlite3.IntegrityError as e:
            # A concurrent writer committed the same key between our check and our insert.
            existing = self._existing(idem)
            if existing is None:
                raise LedgerError(f"ledger write failed: {e}") from e
            return self._resolve_existing(existing, idem, chash, req)

    def _precheck(self, req: PredictionRequest) -> None:
        pcfg = self.cfg.prospective
        if req.mode == "PROSPECTIVE" and (
            pcfg.prospective_predictions != "ENABLED" or pcfg.backfill_status != "COMPLETE"
        ):
            raise PredictionRejected(
                f"PROSPECTIVE predictions are {pcfg.prospective_predictions} (backfill {pcfg.backfill_status}); "
                "the ledger refuses prospective commitments until the reviewed status gate is ENABLED"
            )
        try:
            check_mode_namespace(req.mode, req.match_id)
            check_probabilities(req.probability_a, req.probability_b, self.cfg.probability_tolerance)
        except EventValidationError as e:
            raise PredictionRejected(str(e)) from e
        if req.forecast_slot not in self.cfg.forecast_slots:
            raise PredictionRejected(f"forecast slot {req.forecast_slot!r} not in {self.cfg.forecast_slots}")
        if (req.player_a_id, req.player_b_id) != assign_players(_orientation_key(req.match_id), req.player_a_id,
                                                                  req.player_b_id):  # fmt: skip
            raise PredictionRejected("player orientation does not follow the documented rule; swap A and B "
                                     "(and the probabilities) before committing")  # fmt: skip
        if not req.frozen_hash.startswith(pcfg.expected_frozen_hash):
            raise PredictionRejected(
                f"frozen hash {req.frozen_hash[:16]} is not the expected {pcfg.expected_frozen_hash}"
            )
        if req.configuration_hash != pcfg.modeling.configuration_hash():
            raise PredictionRejected("configuration hash differs from the frozen modeling configuration")
        if not req.state_snapshot_id or not req.feature_state_hash:
            raise PredictionRejected("missing player-state snapshot")

    def _existing(self, idem: str):
        return self.con.execute("SELECT * FROM events WHERE idempotency_key = ?", (idem,)).fetchone()

    def _resolve_existing(self, row, idem: str, chash: str, req: PredictionRequest) -> Receipt:
        old = json.loads(row["payload"])["audit"]["content_hash"]
        if old == chash:
            return Receipt(storage.meta(self.con)["ledger_id"], row["seq"], row["event_id"], row["event_hash"],
                           row["previous_hash"], row["recorded_at"], idem, "existing")  # fmt: skip
        self.con.execute(
            "INSERT INTO idempotency_conflicts (detected_at, idempotency_key, existing_event_id, existing_content_hash,"
            " rejected_content_hash, rejected_request) VALUES (?, ?, ?, ?, ?, ?)",
            (ts(self.clock.now()), idem, row["event_id"], old, chash, canonical(asdict(req)).decode("utf-8")),
        )
        raise IdempotencyConflict(f"idempotency key {idem[:12]} already committed as {row['event_id']} with different "
                                  "content; the original forecast is kept and this attempt was logged")  # fmt: skip

    # ------------------------------------------------------------------------------------- lifecycle events
    def record(self, event_type: str, match_id: str, payload: dict, ref_event_id: str | None = None) -> Receipt:
        if event_type in ("PREDICTION_COMMITTED", "LEDGER_CHECKPOINT_CREATED"):
            raise LedgerError(f"use the dedicated method for {event_type}")
        return self._append(event_type, match_id, payload, ref=ref_event_id)

    def record_checkpoint_event(self, payload: dict) -> Receipt:
        return self._append("LEDGER_CHECKPOINT_CREATED", None, payload)


def _orientation_key(match_id: str) -> str:
    """Orientation uses the canonical id without the SYNTHETIC/HISTORICAL_REPLAY namespace prefix."""
    for prefix in NAMESPACES.values():
        if match_id.startswith(prefix):
            return match_id[len(prefix) :]
    return match_id


def orient(match_id: str, p1: int, p2: int, prob_p1: float, elo_p1: float) -> dict:
    """Helper for callers: player A/B and A's probabilities under the documented orientation rule."""
    a, b = assign_players(_orientation_key(match_id), p1, p2)
    if a == p1:
        return {"player_a_id": a, "player_b_id": b, "probability_a": prob_p1, "probability_b": 1.0 - prob_p1,
                "baseline_elo_probability_a": elo_p1}  # fmt: skip
    return {"player_a_id": a, "player_b_id": b, "probability_a": 1.0 - prob_p1, "probability_b": prob_p1,
            "baseline_elo_probability_a": 1.0 - elo_p1}  # fmt: skip


def _check_transition(con, match_id: str, new_event: dict) -> None:
    rows = [dict(r) for r in con.execute("SELECT * FROM events WHERE match_id = ? ORDER BY seq", (match_id,))]
    for r in rows:
        r["payload"] = json.loads(r["payload"])
    by_id = {r["event_id"]: r for r in rows}
    st = MatchState(match_id)
    for r in rows:
        apply(st, r, by_id)
    try:
        apply(st, new_event, by_id)
    except TransitionError as e:
        if new_event["event_type"] == "PREDICTION_COMMITTED":
            raise PredictionRejected(str(e)) from e
        raise
