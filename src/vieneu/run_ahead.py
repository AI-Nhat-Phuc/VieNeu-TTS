"""
Run ahead: speak the next text chunks of one call on stream slots nobody uses.
=============================================================================

``infer_stream`` splits a long input into text chunks and speaks them one after
another, so a single caller kept one stream of the llama.cpp frame scheduler
busy while the other five idled (each stream runs ~1.1-3x real time; six
together 6.9x). ``RunAhead`` starts chunk ``i+1``, ``i+2``... in threads on
slots it *borrows* from a ``lender`` while chunk ``i`` is still being played,
buffers their audio and hands it back in order.

The lender decides who gets spare capacity (the server keeps one slot free for
the next request and lends nothing while a request is queued). When it says a
request is waiting (``wanted()``), the chunk furthest ahead gives its slot back
at once — its partial audio is dropped and it is generated again later — so a
new listener never waits for somebody else's look-ahead.
"""
import queue
import threading
from typing import Callable, Iterable, Iterator, List, Optional, Protocol


class Lender(Protocol):
    def borrow(self) -> bool:
        """Take one spare stream slot if there is one; never blocks."""

    def give_back(self) -> None:
        """Return a slot ``borrow`` handed out."""

    def wanted(self) -> bool:
        """True while another request waits for a slot."""


_DONE = object()


class _Failed:
    def __init__(self, exc: BaseException):
        self.exc = exc


class RunAhead:
    """Iterate audio iterators of ``makers`` (one per text chunk) in order while
    up to ``1 + borrowed`` of them generate at once.

    The call itself owns one slot (the caller took it); every further chunk
    that runs at the same time needs a slot from ``lender``. Chunks start in
    order, so the one being played is always running.
    """

    def __init__(self, makers: List[Callable[[], Iterable]], lender: Lender,
                 join_timeout: float = 10.0):
        self.makers = makers
        self.lender = lender
        self.join_timeout = join_timeout
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._queues = [queue.SimpleQueue() for _ in makers]
        self._threads: List[threading.Thread] = []
        self._limit = len(makers)     # nothing past a failed chunk is started
        self._started = 0             # chunks [0, started) are running or done
        self._running = 0             # threads alive, each on a stream slot
        self._live = set()            # chunks generating that were not preempted
        self._borrowed = 0
        self.peak = 0                 # most chunks generating at once (for logs / tests)

    # ── scheduling (caller holds the lock) ──────────────────────────────────

    def _give_back_surplus(self) -> None:
        while self._borrowed and 1 + self._borrowed > self._running:
            self._borrowed -= 1
            self.lender.give_back()

    def _pump(self) -> None:
        if self.lender.wanted():
            self._give_back_surplus()
        while self._started < self._limit and not self._stop.is_set():
            if self._running >= 1 + self._borrowed:
                if self.lender.wanted() or not self.lender.borrow():
                    break
                self._borrowed += 1
            i = self._started
            self._started += 1
            self._running += 1
            self._live.add(i)
            self.peak = max(self.peak, self._running)
            t = threading.Thread(target=self._work, args=(i,), daemon=True, name=f"run-ahead-{i}")
            self._threads.append(t)
            t.start()
        self._give_back_surplus()

    def _preempt(self, i: int) -> bool:
        """Chunk ``i`` drops its work if it is the furthest one ahead (never the
        one being played: a chunk before it is still running)."""
        with self._lock:
            if self._stop.is_set() or i != max(self._live) or min(self._live) == i:
                return False
            self._live.discard(i)
            self._started -= 1
            self._queues[i] = queue.SimpleQueue()
            return True

    # ── workers ─────────────────────────────────────────────────────────────

    def _work(self, i: int) -> None:
        q = self._queues[i]
        preempted = False
        try:
            it = iter(self.makers[i]())
            try:
                for sub in it:
                    if self._stop.is_set():
                        break
                    if self.lender.wanted() and self._preempt(i):
                        preempted = True
                        break
                    q.put(sub)
            finally:
                close = getattr(it, "close", None)
                if close is not None:
                    close()
            if not preempted:
                q.put(_DONE)
        except BaseException as e:  # noqa: BLE001 - re-raised in the caller's thread
            with self._lock:
                self._limit = min(self._limit, i + 1)
            q.put(_Failed(e))
        finally:
            with self._lock:
                self._running -= 1
                if not preempted:
                    self._live.discard(i)
                self._pump()

    # ── consumer ────────────────────────────────────────────────────────────

    def _drain(self, i: int) -> Iterator:
        while True:
            item = self._queues[i].get()
            if item is _DONE:
                return
            if isinstance(item, _Failed):
                raise item.exc
            yield item

    def __iter__(self) -> Iterator[Iterator]:
        with self._lock:
            self._pump()
        try:
            for i in range(len(self.makers)):
                yield self._drain(i)
        finally:
            self.close()

    def close(self) -> None:
        """Stop every chunk and wait for its thread, so no stream outlives the
        call's own slot."""
        self._stop.set()
        for t in list(self._threads):
            if t is not threading.current_thread():
                t.join(self.join_timeout)


def run_ahead(makers: List[Callable[[], Iterable]], lender: Optional[Lender]) -> Iterator[Iterable]:
    """One audio iterator per chunk: generated ahead when ``lender`` is given and
    there is more than one chunk, else one by one as before."""
    if lender is None or len(makers) < 2:
        for make in makers:
            yield make()
        return
    yield from RunAhead(makers, lender)
