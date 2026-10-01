"""
Frame scheduler for the v3 Turbo ONNX engine with a llama.cpp backbone.
=======================================================================
Steps every active stream in lockstep, one frame per round:

    backbone  one ``llama_decode`` for all streams (+ prefills of new ones)
    acoustic  one batched acoustic-head pass for all streams (16 slots)

so each round costs two calls whatever the number of streams, instead of 17
per stream. Streams join at frame boundaries and leave on EOS, at their frame
cap, or when their consumer stops iterating. Codec decoding stays in the stream
threads, where it overlaps with the next rounds.
"""
from __future__ import annotations

import logging
import queue
import threading
from typing import Iterator, List, Optional, Tuple

import numpy as np

from .acoustic_batched import AcousticHead, acoustic_frames

logger = logging.getLogger("Vieneu.V3Turbo.ONNX")

_DONE = object()


class _Job:
    __slots__ = ("prompt", "anchor", "params", "max_frames", "out", "seq", "h", "embed", "pos", "t",
                 "cancelled")

    def __init__(self, prompt, anchor, params, max_frames):
        self.prompt, self.anchor, self.params, self.max_frames = prompt, anchor, params, max_frames
        self.out: "queue.SimpleQueue" = queue.SimpleQueue()
        self.seq: Optional[int] = None
        self.h = self.embed = None
        self.pos = self.t = 0
        self.cancelled = False


class FrameScheduler:
    def __init__(self, engine, head: AcousticHead, backbone):
        self.eng, self.head, self.bb = engine, head, backbone
        self._free = list(range(backbone.n_seq_max))
        self._pending: List[_Job] = []
        self._active: List[_Job] = []
        self._cv = threading.Condition()
        threading.Thread(target=self._run, name="vieneu-frame-scheduler", daemon=True).start()

    # ── stream side ───────────────────────────────────────────────────────────
    def generate(self, prompt_embeds: np.ndarray, anchor, params, max_new_frames: int
                 ) -> Iterator[Tuple[List[int], bool]]:
        """Yield ``(codes, eos)`` per frame. ``prompt_embeds`` (T, H);
        ``params`` = (temperature, top_k, top_p, repetition_penalty, history)."""
        job = _Job(np.ascontiguousarray(prompt_embeds, np.float32), anchor, params, int(max_new_frames))
        if job.max_frames <= 0:
            return
        with self._cv:
            self._pending.append(job)
            self._cv.notify()
        try:
            while True:
                item = job.out.get()
                if item is _DONE:
                    return
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            job.cancelled = True   # consumer stopped early: the scheduler drops it

    # ── scheduler thread ──────────────────────────────────────────────────────
    def _admit(self) -> List[_Job]:
        """Pending jobs that get a KV sequence this round, within the token budget."""
        new, budget = [], self.bb.n_batch - len(self._active)
        with self._cv:
            while not self._pending and not self._active:
                self._cv.wait()
            while self._pending and self._free and self._pending[0].prompt.shape[0] <= budget:
                job = self._pending.pop(0)
                if job.cancelled:
                    continue
                job.seq = self._free.pop()
                budget -= job.prompt.shape[0]
                new.append(job)
        return new

    def _release(self, job: _Job, item=_DONE) -> None:
        job.out.put(item)
        if job.seq is not None:
            try:
                self.bb.reset(job.seq)
            finally:
                with self._cv:
                    self._free.append(job.seq)
                job.seq = None

    def _run(self) -> None:
        e = self.eng
        while True:
            new = self._admit()
            try:
                for job in [j for j in self._active if j.cancelled]:
                    self._active.remove(job)
                    self._release(job)
                for job in new:
                    self.bb.reset(job.seq)
                rows = [(j.seq, j.prompt, 0) for j in new] + [(j.seq, j.embed, j.pos) for j in self._active]
                if rows:
                    hs = self.bb.decode(rows)
                    for j, h in zip(new + self._active, hs):
                        j.h = h
                self._active += new
                if not self._active:
                    continue
                res = acoustic_frames(e, self.head, np.stack([j.h for j in self._active]),
                                      [j.params for j in self._active])
                keep = []
                for j, (codes, eos) in zip(self._active, res):
                    j.out.put((codes, eos))
                    j.t += 1
                    if eos or j.cancelled or j.t >= j.max_frames:
                        self._release(j)
                        continue
                    slot = np.full((e.n_vq + 1,), e.audio_pad, dtype=np.int64)
                    slot[0] = e.sgs
                    slot[1:] = codes
                    j.embed = e._embed_rows(slot[None], j.anchor)[0]       # (1, H)
                    j.pos = j.prompt.shape[0] + j.t - 1
                    keep.append(j)
                self._active = keep
            except Exception as ex:  # fail every stream in flight; the next ones start clean
                logger.exception("frame scheduler round failed")
                for j in self._active + [j for j in new if j not in self._active]:
                    self._release(j, ex)
                self._active = []
