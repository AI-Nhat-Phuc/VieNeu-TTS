"""vieneu/run_ahead.py: later text chunks generate on borrowed slots, audio comes
back in order, slots go back, a waiting request preempts the look-ahead."""
import threading
import time

import pytest

from vieneu.run_ahead import RunAhead, run_ahead


class Lender:
    def __init__(self, spare):
        self.spare = spare
        self.lent = 0
        self.waiting = False
        self.lock = threading.Lock()

    def borrow(self):
        with self.lock:
            if self.lent >= self.spare:
                return False
            self.lent += 1
            return True

    def give_back(self):
        with self.lock:
            self.lent -= 1
            assert self.lent >= 0

    def wanted(self):
        return self.waiting


def chunk(i, n=3, delay=0.02, starts=None):
    def make():
        if starts is not None:
            starts.append(i)
        for k in range(n):
            time.sleep(delay)
            yield (i, k)
    return make


def play(per_chunk):
    return [sub for subs in per_chunk for sub in subs]


def test_chunks_run_at_once_and_come_back_in_order():
    lender = Lender(spare=3)
    ra = RunAhead([chunk(i) for i in range(6)], lender)
    out = play(ra)
    assert out == [(i, k) for i in range(6) for k in range(3)]
    assert ra.peak == 4                  # its own slot + three borrowed
    assert lender.lent == 0              # every borrowed slot went back


def test_no_spare_slot_keeps_one_chunk_at_a_time():
    lender = Lender(spare=0)
    ra = RunAhead([chunk(i) for i in range(4)], lender)
    assert play(ra) == [(i, k) for i in range(4) for k in range(3)]
    assert ra.peak == 1


def test_without_lender_nothing_runs_ahead():
    starts = []
    per_chunk = run_ahead([chunk(i, starts=starts) for i in range(3)], None)
    first = next(per_chunk)
    assert list(first) == [(0, 0), (0, 1), (0, 2)]
    assert starts == [0]


def test_waiting_request_takes_the_slots_back():
    lender = Lender(spare=3)
    starts = []
    ra = RunAhead([chunk(i, n=10, starts=starts) for i in range(5)], lender)
    it = iter(ra)
    subs = next(it)
    assert next(subs) == (0, 0)
    time.sleep(0.05)
    assert lender.lent == 3
    lender.waiting = True
    deadline = time.time() + 2
    while lender.lent and time.time() < deadline:
        time.sleep(0.01)
    assert lender.lent == 0               # chunks 1-3 dropped their work
    lender.waiting = False
    out = [(0, 0)] + list(subs) + [s for rest in it for s in rest]
    assert out == [(i, k) for i in range(5) for k in range(10)]   # still complete, in order
    assert starts.count(1) >= 2           # chunk 1 was generated again
    assert lender.lent == 0


def test_failed_chunk_raises_in_order_and_starts_nothing_after():
    starts = []

    def bad():
        starts.append(2)
        raise RuntimeError("boom")
        yield  # pragma: no cover

    lender = Lender(spare=5)
    makers = [chunk(0, starts=starts), chunk(1, starts=starts), bad,
              chunk(3, starts=starts), chunk(4, starts=starts)]
    got = []
    with pytest.raises(RuntimeError, match="boom"):
        for subs in RunAhead(makers, lender):
            got.extend(subs)
    assert got == [(i, k) for i in range(2) for k in range(3)]
    assert lender.lent == 0


def test_leaving_early_stops_every_chunk():
    lender = Lender(spare=3)
    ra = RunAhead([chunk(i, n=50) for i in range(5)], lender)
    it = iter(ra)
    next(next(it))
    it.close()
    assert all(not t.is_alive() for t in ra._threads)
    assert lender.lent == 0
