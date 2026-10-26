"""Concurrency, rollback and crash recovery of the SQLite ledger."""

from __future__ import annotations

import multiprocessing as mp
import os
import threading
from datetime import timedelta

import pytest

from ledger_helpers import T0, ledger_cfg, request, writer
from tennis_data.ledger import storage
from tennis_data.ledger.reader import LedgerReader
from tennis_data.ledger.verification import verify_ledger
from tennis_data.ledger.writer import LedgerError, LedgerWriter, SystemClock


def _worker(tmp_path, wid, n, barrier):
    w = LedgerWriter(ledger_cfg(tmp_path), SystemClock())
    barrier.wait()
    for i in range(n):
        w.commit_prediction(request(w.cfg, match_id=f"synthetic:w{wid}-{i}", prob_p1=0.55))
    w.close()


def test_concurrent_process_writers_form_one_valid_chain(tmp_path):
    writer(tmp_path).close()
    ctx = mp.get_context("spawn")
    workers, n = 4, 40
    barrier = ctx.Barrier(workers)
    ps = [ctx.Process(target=_worker, args=(tmp_path, k, n, barrier)) for k in range(workers)]
    for p in ps:
        p.start()
    for p in ps:
        p.join(120)
        assert p.exitcode == 0
    rep = verify_ledger(tmp_path / "ledger.sqlite")
    assert rep.ok, rep.issues[:3]
    assert rep.events == workers * n
    con = storage.connect(tmp_path / "ledger.sqlite", read_only=True)
    seqs = [r[0] for r in con.execute("SELECT seq FROM events ORDER BY seq")]
    assert seqs == list(range(1, workers * n + 1))


def test_concurrent_identical_requests_commit_once(tmp_path):
    writer(tmp_path).close()
    results, errors = [], []

    def go():
        w = LedgerWriter(ledger_cfg(tmp_path), SystemClock())
        try:
            results.append(w.commit_prediction(request(w.cfg, match_id="synthetic:race")))
        except Exception as e:  # noqa: BLE001
            errors.append(e)
        finally:
            w.close()

    ts = [threading.Thread(target=go) for _ in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errors
    assert sorted(r.status for r in results).count("committed") == 1
    assert len({r.event_id for r in results}) == 1
    assert verify_ledger(tmp_path / "ledger.sqlite").events == 1


class _FailingCommit:
    """Connection proxy whose COMMIT fails, as a disk error would."""

    def __init__(self, con):
        self.con = con

    def execute(self, sql, *a):
        if sql == "COMMIT":
            raise OSError("simulated failure at commit")
        return self.con.execute(sql, *a)


def test_failed_transaction_leaves_nothing_behind(tmp_path):
    w = writer(tmp_path)
    w.commit_prediction(request(w.cfg, match_id="synthetic:a"))
    head = LedgerReader(w.path).head()
    real = w.con
    w.con = _FailingCommit(real)
    with pytest.raises(OSError):
        w.commit_prediction(request(w.cfg, match_id="synthetic:b"))
    w.con = real
    assert LedgerReader(w.path).head() == head
    r = w.commit_prediction(request(w.cfg, match_id="synthetic:b"))  # the ledger is usable and the seq is reused
    assert r.seq == 2 and verify_ledger(w.path).ok


def _crash_mid_write(path):
    con = storage.connect(path)
    con.execute("BEGIN IMMEDIATE")
    seq, prev = storage.head(con)
    row = dict(con.execute("SELECT * FROM events WHERE seq = ?", (seq,)).fetchone())
    row.update(seq=seq + 1, previous_hash=prev, event_id="crash-event-0001", event_hash="c" * 64,
               idempotency_key="d" * 64)  # fmt: skip
    con.execute(f"INSERT INTO events ({', '.join(storage.COLUMNS)}) VALUES ({', '.join('?' * len(storage.COLUMNS))})",
                tuple(row[c] for c in storage.COLUMNS))  # fmt: skip
    os._exit(1)  # process dies inside the transaction, before COMMIT


def test_crash_inside_a_transaction_is_rolled_back(tmp_path):
    w = writer(tmp_path)
    for i in range(3):
        w.commit_prediction(request(w.cfg, match_id=f"synthetic:c{i}"))
    w.close()
    p = mp.get_context("spawn").Process(target=_crash_mid_write, args=(tmp_path / "ledger.sqlite",))
    p.start()
    p.join(60)
    assert p.exitcode == 1
    rep = verify_ledger(tmp_path / "ledger.sqlite")
    assert rep.ok and rep.events == 3  # the half-written event never became visible
    w2 = writer(tmp_path, start=T0 + timedelta(hours=1))
    assert w2.commit_prediction(request(w2.cfg, match_id="synthetic:after")).seq == 4
    assert verify_ledger(tmp_path / "ledger.sqlite").ok


def test_writer_refuses_a_clock_that_runs_backwards(tmp_path):
    w = writer(tmp_path)
    w.commit_prediction(request(w.cfg, match_id="synthetic:t1"))
    w2 = writer(tmp_path)  # a new writer whose (test) clock restarts earlier than the head
    with pytest.raises(LedgerError, match="clock regression"):
        w2.commit_prediction(request(w2.cfg, match_id="synthetic:t2"))
    assert verify_ledger(w.path).ok
