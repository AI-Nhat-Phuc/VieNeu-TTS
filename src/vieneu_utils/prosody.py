"""Speed and pitch for streamed speech, set per request or per sentence.

The v3 Turbo model has no speed or pitch control of its own (its speaking
style comes from the reference voice), so both are applied to the audio it
makes, chunk by chunk, without waiting for the whole utterance:

* pitch: resample by ``2**(semitones/12)`` (soxr) — shifts pitch and tempo;
* speed: WSOLA time-stretch that also undoes the tempo change of the pitch step.

Inline tags in the text change them from that point on::

    Bình thường. [speed=1.3] Câu này nhanh hơn. [pitch=-3 speed=0.85] Trầm và chậm.
    [voice=Mai Anh] Đổi giọng giữa chừng. [reset] Về lại giá trị của request.

Keys: ``speed`` (0.25-4, best 0.7-1.5), ``pitch`` (semitones, -12..12),
``voice`` (a voice name). Other bracketed text (``[cười]``, ``[thở dài]``...) is
left in the text for the model.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from typing import List, Optional

import numpy as np

SPEED_RANGE = (0.25, 4.0)
PITCH_RANGE = (-12.0, 12.0)

_KEYS = ("speed", "pitch", "voice")
_TAG_RE = re.compile(r"\[([^\[\]]*)\]")
_KEY_RE = re.compile(r"(?:^|\s+)(speed|pitch|voice)\s*=\s*", re.IGNORECASE)


@dataclass(frozen=True)
class Prosody:
    speed: float = 1.0
    pitch: float = 0.0
    voice: Optional[str] = None


@dataclass(frozen=True)
class Part:
    text: str
    prosody: Prosody


class ProsodyError(ValueError):
    pass


def _parse_tag(body: str) -> Optional[dict]:
    """``"speed=1.2 voice=Mai Anh"`` -> {"speed": "1.2", "voice": "Mai Anh"};
    ``"reset"``/``"/"`` -> {}; anything else (a cue like ``cười``) -> None."""
    s = body.strip()
    if s.lower() in ("reset", "/"):
        return {}
    pieces = _KEY_RE.split(s)
    # split -> [before, key1, val1, key2, val2, ...]; a real tag has nothing before key1
    if len(pieces) < 3 or pieces[0].strip():
        return None
    return {pieces[i].lower(): pieces[i + 1].strip() for i in range(1, len(pieces), 2)}


def _check(p: Prosody) -> Prosody:
    if not (SPEED_RANGE[0] <= p.speed <= SPEED_RANGE[1]) or math.isnan(p.speed):
        raise ProsodyError(f"speed must be in {SPEED_RANGE[0]}..{SPEED_RANGE[1]}, got {p.speed}")
    if not (PITCH_RANGE[0] <= p.pitch <= PITCH_RANGE[1]) or math.isnan(p.pitch):
        raise ProsodyError(f"pitch must be in {PITCH_RANGE[0]}..{PITCH_RANGE[1]} semitones, got {p.pitch}")
    return p


def split_prosody(text: str, base: Prosody = Prosody()) -> List[Part]:
    """Cut ``text`` at its prosody tags; each part keeps the settings in force
    there. Parts with nothing to say are dropped; neighbours with the same
    settings are joined. Raises ProsodyError on a bad value."""
    _check(base)
    parts: List[Part] = []
    cur, pos = base, 0

    def add(seg: str, p: Prosody) -> None:
        if not re.search(r"\w", seg):
            return
        if parts and parts[-1].prosody == p:
            parts[-1] = Part(parts[-1].text + seg, p)
        else:
            parts.append(Part(seg, p))

    for m in _TAG_RE.finditer(text):
        kv = _parse_tag(m.group(1))
        if kv is None:
            continue
        add(text[pos:m.start()], cur)
        pos = m.end()
        if not kv:
            cur = base
            continue
        upd = {}
        try:
            if "speed" in kv:
                upd["speed"] = float(kv["speed"])
            if "pitch" in kv:
                upd["pitch"] = float(kv["pitch"])
        except ValueError:
            raise ProsodyError(f"bad number in tag [{m.group(1)}]") from None
        if "voice" in kv:
            upd["voice"] = kv["voice"] or base.voice
        cur = _check(replace(cur, **upd))
    add(text[pos:], cur)
    return parts


def has_tags(text: str) -> bool:
    return any(_parse_tag(m.group(1)) is not None for m in _TAG_RE.finditer(text))


class Wsola:
    """Streaming WSOLA time-stretch: ``rate`` > 1 is faster (shorter).

    Feed float32 mono chunks; each call returns the output that is final so
    far, ``last=True`` flushes the rest. Output length is ``round(n_in / rate)``.
    """

    def __init__(self, rate: float, sr: int = 48000):
        self.rate = float(rate)
        self.n = n = max(256, int(sr * 0.04) // 2 * 2)     # 40 ms frames
        self.hs = n // 2                                     # synthesis hop
        self.ha = self.hs * self.rate                        # analysis hop
        self.tol = n // 4                                    # search ±10 ms
        self.dec = max(1, sr // 12000)                       # search on a 12 kHz view
        self.win = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)).astype(np.float32)
        # Input starts with hs zeros so the first frame's fade-in falls on silence.
        self.buf = np.zeros(self.hs + self.tol, np.float32)  # input from abs pos `off`
        self.off = -self.tol                                 # abs pos of buf[0]
        self.n_in = 0
        self.k = 0
        self.prev = 0                                        # abs start of frame k-1
        self.ola = np.zeros(n, np.float32)                   # output from k*hs on
        self.out_n = 0                                       # samples emitted
        self.skip = self.hs                                  # drop the padding's output

    def _get(self, a: int, length: int) -> np.ndarray:
        i = a - self.off
        return self.buf[i:i + length]

    def _frames(self, end: int) -> List[np.ndarray]:
        n, hs, tol = self.n, self.hs, self.tol
        out = []
        while True:
            nom = int(round(self.k * self.ha))
            if nom + tol + n > end or (self.k and self.prev + hs + n > end):
                break
            if self.k == 0:
                start = nom
            else:
                target = self._get(self.prev + hs, n)[::self.dec]
                lo = max(nom - tol, self.off)
                region = self._get(lo, n + (nom + tol - lo))[::self.dec]
                c = np.correlate(region, target, mode="valid")
                start = lo + int(np.argmax(c)) * self.dec if len(c) else nom
            frame = self._get(start, n) * self.win
            self.ola += frame
            out.append(self.ola[:hs].copy())
            self.ola = np.concatenate([self.ola[hs:], np.zeros(hs, np.float32)])
            self.prev = start
            self.k += 1
        # keep only the input later frames can still read
        keep_from = min(int(round(self.k * self.ha)) - tol, self.prev + hs) if self.k else self.off
        drop = keep_from - self.off
        if drop > 0:
            self.buf = self.buf[drop:]
            self.off += drop
        return out

    def _emit(self, pieces: List[np.ndarray], limit: Optional[int] = None) -> np.ndarray:
        y = np.concatenate(pieces) if pieces else np.zeros(0, np.float32)
        if self.skip:
            d = min(self.skip, len(y))
            y, self.skip = y[d:], self.skip - d
        if limit is not None:
            y = y[:max(0, limit - self.out_n)]
        self.out_n += len(y)
        return y

    def __call__(self, x: np.ndarray, last: bool = False) -> np.ndarray:
        x = np.asarray(x, np.float32)
        self.n_in += len(x)
        self.buf = np.concatenate([self.buf, x])
        if not last:
            return self._emit(self._frames(self.off + len(self.buf)))
        want = int(round(self.n_in / self.rate))
        pieces = []
        pad = np.zeros(self.n + 2 * self.tol + int(self.ha) + self.hs, np.float32)
        while self.out_n + sum(map(len, pieces)) - self.skip < want:
            self.buf = np.concatenate([self.buf, pad])
            got = self._frames(self.off + len(self.buf))
            if not got:
                break
            pieces += got
        pieces.append(self.ola)
        y = self._emit(pieces, limit=want)
        short = want - self.out_n          # never short: pad with silence
        if short > 0:
            y =np.concatenate([y, np.zeros(short, np.float32)])
            self.out_n += short
        return y


class Shifter:
    """Speed and pitch for one stream of 48 kHz float32 chunks (see module doc).
    Identity — the chunks pass through untouched — at speed 1, pitch 0."""

    def __init__(self, speed: float = 1.0, pitch: float = 0.0, sr: int = 48000):
        r = 2.0 ** (pitch / 12.0)
        self.rs = None
        if abs(r - 1.0) > 1e-6:
            import soxr
            # sr -> sr/r samples per second, played back at sr: pitch and tempo x r
            self.rs = soxr.ResampleStream(sr, sr / r, 1, dtype="float32", quality="HQ")
        a = speed / r                       # what is left for the time-stretch
        self.ts = Wsola(a, sr) if abs(a - 1.0) > 1e-6 else None

    @property
    def identity(self) -> bool:
        return self.rs is None and self.ts is None

    def __call__(self, x: np.ndarray, last: bool = False) -> np.ndarray:
        y = np.asarray(x, np.float32)
        if self.rs is not None:
            y = self.rs.resample_chunk(y, last=last)
        if self.ts is not None:
            y = self.ts(y, last=last)
        return y
