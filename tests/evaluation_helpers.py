"""Phase 4C fixtures: synthetic ledgers (deterministic clocks), outcome snapshots, a local RFC 3161 TSA and a fake
GitHub Contents API. Nothing here reaches a real provider or network.

PROSPECTIVE fixture records are written through the real ``LedgerWriter`` with a *test-only* prospective config whose
status gate is ENABLED and a deterministic clock flagged as trusted. This models a ledger produced by a future,
enabled deployment; the repository's own gate stays DISABLED and is never changed by these helpers.
"""

from __future__ import annotations

import base64
import json
import shutil
import subprocess
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

from tennis_data.evaluation_prospective.config import load_evaluation_config
from tennis_data.ledger.hashing import canonical, ts
from tennis_data.ledger.writer import FixedClock, LedgerWriter, PredictionRequest, orient

T0 = datetime(2030, 3, 4, 9, 0, tzinfo=UTC)  # a Monday, after any real TSA time used by the tests
OPENSSL = shutil.which("openssl")


class TrustedTestClock(FixedClock):
    """Deterministic clock that the writer treats as trusted. Tests only: models an enabled deployment's clock."""

    trusted = True


def eval_cfg(tmp_path: Path, prospective_enabled: bool = False):
    cfg = load_evaluation_config()
    lcfg = replace(cfg.ledger, ledger_path=tmp_path / "ledger.sqlite", export_dir=tmp_path / "export",
                   checkpoint_dir=tmp_path / "checkpoints", reports_dir=tmp_path / "ledger_reports")  # fmt: skip
    if prospective_enabled:
        lcfg = replace(lcfg, prospective=replace(lcfg.prospective, backfill_status="COMPLETE",
                                                 prospective_predictions="ENABLED"))  # fmt: skip
    return replace(cfg, ledger=lcfg, snapshot_dir=tmp_path / "snapshots", package_dir=tmp_path / "packages",
                   reports_dir=tmp_path / "reports", evidence_dir=tmp_path / "anchors",
                   trusted_checkpoint_dir=tmp_path / "trusted")  # fmt: skip


class World:
    """A PROSPECTIVE-enabled test ledger with a settable deterministic clock."""

    def __init__(self, tmp_path: Path, ledger_id: str = "ledger-eval-test"):
        self.cfg = eval_cfg(tmp_path, prospective_enabled=True)
        self.clock = TrustedTestClock(T0 - timedelta(days=10), timedelta(0))
        self.w = LedgerWriter(self.cfg.ledger, self.clock)
        self.w.init(ledger_id)
        self.tmp = tmp_path

    def at(self, t: datetime):
        self.clock.t = t
        return self

    def predict(self, mid, start, commit, prob=0.62, elo=0.58, p1=104925, p2=207989, slot="pre_match", sched="s1",
                mode="PROSPECTIVE", **over):  # fmt: skip
        o = orient(mid, p1, p2, prob, elo)
        req = PredictionRequest(
            mode=mode, forecast_slot=slot, match_id=mid, tournament_id=over.pop("tournament_id", "t-2030-1"),
            surface="Hard", round="R32", best_of=3, scheduled_start_at=ts(start), scheduled_start_verified=True,
            schedule_snapshot_id=sched, schedule_retrieved_at=ts(commit - timedelta(hours=1)), model_id="M4A+raw",
            model_version="m3b-1+engine1", frozen_hash=self.cfg.ledger.prospective.expected_frozen_hash + "0" * 48,
            configuration_hash=self.cfg.protocol["approved_configuration_hash"], feature_schema_version="sr-1+engine2",
            state_snapshot_id="state-1", feature_state_hash="fs-1", source_snapshots={"archive": "x"},
            prediction_created_at=ts(commit - timedelta(minutes=1)),
            prediction_cutoff=ts(start - timedelta(minutes=30)) if mode == "PROSPECTIVE" else None,
            feature_availability={"a_service_points_observed": 4000, "b_service_points_observed": 900,
                                  "global_match_count_a": 150, "global_match_count_b": 40},
            freshness_status="ok", inference_engine_version="test", git_commit="test", **o,
        )  # fmt: skip
        if over:
            req = replace(req, **over)
        return self.at(commit).w.commit_prediction(req)

    def record(self, t, event_type, mid, payload, ref=None):
        return self.at(t).w.record(event_type, mid, payload, ref)

    def start(self, t_rec, mid, actual, provider="feed_a"):
        return self.record(t_rec, "MATCH_METADATA_CORRECTED", mid, {
            "kind": "actual_start", "changes": {"actual_start_at": ts(actual)}, "reason": "began",
            "source": {"provider": provider}})  # fmt: skip

    def result(self, t, mid, winner, status="completed", score="6-4 6-4", supersedes=None, reason=None):
        return self.record(t, "MATCH_RESULT_RECORDED", mid, {
            "status": status, "winner_id": winner, "score": score, "source": {"provider": "feed_a"},
            "supersedes_event_id": supersedes, "correction_reason": reason})  # fmt: skip

    def close(self):
        self.w.close()


def scored_match(world: World, mid: str, start: datetime, winner: int | None = 104925, prob=0.62, elo=0.58,
                 p1=104925, p2=207989, actual_delay=timedelta(minutes=5), **kw):  # fmt: skip
    """Forecast 20 h before, actual start shortly after the scheduled start, completed result 3 h later."""
    r = world.predict(mid, start, start - timedelta(hours=20), prob=prob, elo=elo, p1=p1, p2=p2, **kw)
    world.start(start + timedelta(minutes=10), mid, start + actual_delay)
    if winner is not None:
        world.result(start + timedelta(hours=3), mid, winner)
    return r


def outcome_record(mid, provider, status, winner, players, version=1, score="6-4 6-4", start=None, level="A",
                   retrieved=None, record=None):  # fmt: skip
    return {"match_id": mid, "provider": provider, "provider_record_id": record or f"{provider}-{mid}",
            "source_version": version, "status": status, "winner_id": winner, "player_ids": list(players),
            "score": score, "actual_start_at": ts(start) if start else None, "tournament_id": "t-2030-1",
            "tournament_level": level, "retrieved_at": retrieved or ts(T0 + timedelta(days=60))}  # fmt: skip


def write_outcomes(path: Path, records: list[dict], snapshot_id="test-outcomes") -> Path:
    head = {"format": "tennis-outcome-snapshot-1", "snapshot_id": snapshot_id, "created_at": ts(T0),
            "description": "synthetic test outcomes"}  # fmt: skip
    path.write_bytes(b"\n".join(canonical(x) for x in [head, *records]) + b"\n")
    return path


# ------------------------------------------------------------------------------------------ local RFC 3161 TSA
TSA_CNF = """[ req ]
distinguished_name = dn
prompt = no
[ dn ]
CN = Test TSA
[ tsa_ext ]
basicConstraints = critical,CA:false
keyUsage = critical,digitalSignature
extendedKeyUsage = critical,timeStamping
[ ca_ext ]
basicConstraints = critical,CA:true
keyUsage = critical,keyCertSign
[ tsa ]
default_tsa = tsa_config
[ tsa_config ]
dir = .
serial = ./serial
signer_cert = ./tsa.crt
signer_key = ./tsa.key
signer_digest = sha256
default_policy = 1.2.3.4.1
digests = sha256
accuracy = secs:1
ess_cert_id_alg = sha256
"""


class LocalTSA:
    """A throwaway time-stamping authority for tests (its CA is trusted only where a test passes it explicitly)."""

    def __init__(self, d: Path, name: str = "Test Root"):
        self.d = d
        d.mkdir(parents=True, exist_ok=True)
        (d / "tsa.cnf").write_text(TSA_CNF)
        (d / "serial").write_text("01\n")

        def run(*a):
            subprocess.run([OPENSSL, *a], cwd=d, check=True, capture_output=True)

        run("req", "-x509", "-new", "-newkey", "rsa:2048", "-nodes", "-keyout", "ca.key", "-out", "ca.crt", "-days",
            "3650", "-subj", f"/CN={name}", "-config", "tsa.cnf", "-extensions", "ca_ext")  # fmt: skip
        run("req", "-new", "-newkey", "rsa:2048", "-nodes", "-keyout", "tsa.key", "-out", "tsa.csr", "-subj",
            "/CN=Test TSA", "-config", "tsa.cnf")  # fmt: skip
        run("x509", "-req", "-in", "tsa.csr", "-CA", "ca.crt", "-CAkey", "ca.key", "-CAcreateserial", "-out", "tsa.crt",
            "-days", "3650", "-extfile", "tsa.cnf", "-extensions", "tsa_ext")  # fmt: skip
        self.ca = d / "ca.crt"

    def respond(self, query: bytes) -> bytes:
        (self.d / "q.tsq").write_bytes(query)
        subprocess.run([OPENSSL, "ts", "-reply", "-config", "tsa.cnf", "-queryfile", "q.tsq", "-out", "r.tsr"],
                       cwd=self.d, check=True, capture_output=True)  # fmt: skip
        return (self.d / "r.tsr").read_bytes()

    def transport(self, method, url, headers, body, timeout):
        assert method == "POST" and headers["Content-Type"] == "application/timestamp-query"
        return 200, self.respond(body)


# ---------------------------------------------------------------------------------------- fake GitHub contents
class FakeGitHub:
    """In-memory GitHub Contents API: files per ref, commits, and switches to simulate tampering or outages."""

    def __init__(self):
        self.files: dict = {}  # (ref, path) -> bytes
        self.commits = 0
        self.down = False
        self.serve_other_content = False
        self.calls: list = []

    def transport(self, method, url, headers, body, timeout):
        self.calls.append((method, url))
        if self.down:
            from tennis_data.evaluation_prospective.anchoring import PublicationError

            raise PublicationError("network error (simulated)")
        path_ref = url.split("/contents/", 1)[1] if "/contents/" in url else None
        if method == "GET" and path_ref:
            path, ref = path_ref.split("?ref=")
            data = self.files.get((ref, path))
            if data is None:
                return 404, b"{}"
            if self.serve_other_content:
                data = data + b" "
            return 200, json.dumps({"encoding": "base64", "content": base64.b64encode(data).decode(),
                                    "sha": "blob"}).encode()  # fmt: skip
        if method == "PUT" and path_ref:
            req = json.loads(body)
            data = base64.b64decode(req["content"])
            if (req["branch"], path_ref) in self.files:
                return 422, b"{}"
            self.commits += 1
            sha = f"{self.commits:040d}"
            self.files[(req["branch"], path_ref)] = data
            self.files[(sha, path_ref)] = data
            return 201, json.dumps({"content": {"sha": "blob"}, "commit": {
                "sha": sha, "committer": {"date": "2030-01-01T00:00:00Z"}}}).encode()  # fmt: skip
        return 404, b"{}"
