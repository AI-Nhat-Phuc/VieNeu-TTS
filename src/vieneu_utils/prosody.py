"""Speed, pitch, volume and vocal effort for streamed speech, per request or per sentence.

The v3 Turbo model has no speed or pitch control of its own: it speaks the way
its reference voice speaks. So the natural way ("native", the default) is to
reshape the reference and let the model follow it; measured over 3 voices x 2
texts it follows a sped-up reference ~83 % of the way in log-speed (x1.25 ->
x1.23, x1.5 -> x1.40, x0.8 -> x0.85) and a pitched one ~72 % with the 50/50 embedding blend (+-3, +-6
semitones). The small rest is applied to the output with the DSP below, which
is also the whole job in "dsp" mode:

* pitch: resample by ``2**(semitones/12)`` (soxr) — shifts pitch and tempo;
* speed: WSOLA time-stretch that also undoes the tempo change of the pitch step.

The model ignores the reference's loudness (the codec normalises it), so
``volume`` is a gain on the output. ``effort`` (0-2) makes the reference
higher, faster, brighter and saturated, as a raised / shouting voice is; the
model then speaks with more strain (1 = raised, 2 = shouting). A shout stays
loud to the end, where a narrator trails off, so with effort the reference is
built from its energetic opening only, each clause is generated on its own
(the model starts every clause with fresh breath) and a leveler holds the
loudness steady across the sentence.

Inline tags in the text change them from that point on::

    Bình thường. [speed=1.3] Câu này nhanh hơn. [pitch=-3 speed=0.85] Trầm và chậm.
    [effort=2 volume=1.5] Cút ra khỏi đây! [voice=Mai Anh] Đổi giọng. [reset] Về như cũ.

Keys: ``speed`` (0.25-4, best 0.7-1.5), ``pitch`` (semitones, -12..12),
``volume`` (gain, 0.1-4), ``effort`` (alias ``shout``, 0-2), ``voice`` (a voice
name). Other bracketed text (``[cười]``, ``[thở dài]``...) is left for the model.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from typing import List, Optional

import numpy as np

SPEED_RANGE = (0.25, 4.0)
PITCH_RANGE = (-12.0, 12.0)
VOLUME_RANGE = (0.1, 4.0)
EFFORT_RANGE = (0.0, 2.0)

_NUM_KEYS = {"speed": "speed", "pitch": "pitch", "volume": "volume", "effort": "effort", "shout": "effort"}
_TAG_RE = re.compile(r"\[([^\[\]]*)\]")
_KEY_RE = re.compile(r"(?:^|\s+)(speed|pitch|volume|effort|shout|voice)\s*=\s*", re.IGNORECASE)


@dataclass(frozen=True)
class Prosody:
    speed: float = 1.0
    pitch: float = 0.0
    voice: Optional[str] = None
    volume: float = 1.0
    effort: float = 0.0


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
    for name, (lo, hi), unit in (("speed", SPEED_RANGE, ""), ("pitch", PITCH_RANGE, " semitones"),
                                 ("volume", VOLUME_RANGE, ""), ("effort", EFFORT_RANGE, "")):
        v = getattr(p, name)
        if math.isnan(v) or not (lo <= v <= hi):
            raise ProsodyError(f"{name} must be in {lo:g}..{hi:g}{unit}, got {v:g}")
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
            for k, field in _NUM_KEYS.items():
                if k in kv:
                    upd[field] = float(kv[k])
        except ValueError:
            raise ProsodyError(f"bad number in tag [{m.group(1)}]") from None
        if "voice" in kv:
            upd["voice"] = kv["voice"] or base.voice
        cur = _check(replace(cur, **upd))
    add(text[pos:], cur)
    return parts


def has_tags(text: str) -> bool:
    return any(_parse_tag(m.group(1)) is not None for m in _TAG_RE.finditer(text))


# ── native mode: reshape the reference, let the model follow ────────────────

K_SPEED = 0.83          # log(achieved speed) / log(reference speed), measured
K_PITCH = 0.72          # achieved / reference semitones, measured with the 50/50 embedding blend
                        # (formant-kept reference; 0.97 with the reshaped clip's own embedding)
REF_SPEED_RANGE = (0.6, 1.7)   # beyond this the reshaped reference itself degrades
REF_PITCH_RANGE = (-9.0, 9.0)


@dataclass(frozen=True)
class Plan:
    """How one part is made: the reference reshaped by ``ref_*``, then the
    model's output shifted by ``res_*`` (what the model will not do) and gained."""
    ref_speed: float = 1.0
    ref_pitch: float = 0.0
    tilt_db: float = 0.0
    drive: float = 1.0
    res_speed: float = 1.0
    res_pitch: float = 0.0
    gain: float = 1.0
    head: bool = False                 # reference = its energetic opening, looped
    level_db: Optional[float] = None   # hold loudness at this dBFS (None = off)
    by_clause: bool = False            # one generation per clause

    @property
    def reshapes_reference(self) -> bool:
        return (self.ref_speed, self.ref_pitch, self.tilt_db, self.drive, self.head) != (1.0, 0.0, 0.0, 1.0, False)


def _q(x: float, step: float) -> float:
    return round(round(x / step) * step, 6)


def plan_for(p: Prosody, native: bool = True) -> Plan:
    """Split ``p`` between the reference (native) and the output DSP. Reference
    values are quantised (speed 0.05, pitch 0.5 st) so few variants get built;
    the residual makes up the exact target from the quantised values."""
    gain = p.volume
    e = p.effort
    strain = dict(level_db=-18.0 + 2.0 * e, by_clause=True) if e > 0 else {}
    if not native:
        return Plan(res_speed=p.speed, res_pitch=p.pitch, gain=gain, **strain)
    lo, hi = REF_SPEED_RANGE
    rs = _q(min(hi, max(lo, p.speed ** (1 / K_SPEED))), 0.05)
    rp = _q(min(REF_PITCH_RANGE[1], max(REF_PITCH_RANGE[0], p.pitch / K_PITCH)), 0.5)
    res_s = p.speed / rs ** K_SPEED
    res_p = p.pitch - K_PITCH * rp
    # strain on top, not compensated
    rs_e = _q(min(hi, rs * (1 + 0.12 * e)), 0.05)
    rp_e = _q(min(12.0, rp + 3.5 * e), 0.5)
    return Plan(ref_speed=rs_e, ref_pitch=rp_e, tilt_db=_q(5.0 * e, 0.5), drive=_q(1 + e, 0.25),
                res_speed=1.0 if abs(res_s - 1) < 0.02 else res_s,
                res_pitch=0.0 if abs(res_p) < 0.25 else res_p, gain=gain, head=e > 0, **strain)


def reshape_reference(wav: np.ndarray, sr: int, plan: Plan) -> np.ndarray:
    """The reference clip as ``plan`` wants it: tempo, pitch, brightness, strain."""
    y = np.asarray(wav, np.float32)
    if plan.head:
        y = energetic_head(y, sr)
    if plan.ref_pitch != 0.0:
        y = pitch_shift_keep_formants(y, sr, plan.ref_pitch)
    if plan.ref_speed != 1.0:
        ts = Wsola(plan.ref_speed, sr)
        y = np.concatenate([ts(y), ts(np.zeros(0, np.float32), last=True)])
    if plan.tilt_db:
        from scipy.signal import butter, sosfilt
        hp = sosfilt(butter(2, 1500, "highpass", fs=sr, output="sos"), y).astype(np.float32)
        y = y + (10 ** (plan.tilt_db / 20) - 1) * hp
    if plan.drive != 1.0:
        y = np.tanh(plan.drive * y / (np.abs(y).max() + 1e-9))
    return (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32)


def _envelope(spec: np.ndarray, lifter: int) -> np.ndarray:
    """Smooth spectral envelope per frame (low-quefrency cepstrum): the
    formants, without the harmonics of the voice's pitch."""
    c = np.fft.irfft(np.log(np.abs(spec) + 1e-7), axis=0)
    c[lifter:-lifter] = 0
    return np.exp(np.fft.rfft(c, axis=0).real)


def pitch_shift_keep_formants(wav: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    """Shift the pitch of a whole clip by ``semitones`` keeping its length and
    its formants (the vocal tract's resonances), so it still sounds like the
    same person rather than a smaller / larger one. Resample + WSOLA moves the
    formants with the pitch; each STFT frame then gets the original clip's
    spectral envelope back. Measured on voice identity (speaker-encoder cosine
    to the unshifted voice, +3 st): 0.44 -> 0.69 (female), 0.33 -> 0.61 (male).
    Offline (whole clip); for references, not for the live stream."""
    from scipy.signal import istft, stft
    x = np.asarray(wav, np.float32)
    sh = Shifter(1.0, semitones, sr)
    y = np.concatenate([sh(x), sh(np.zeros(0, np.float32), last=True)])[: len(x)]
    y = np.pad(y, (0, len(x) - len(y)))
    n_fft, hop = 2048 * sr // 48000 or 2048, 256 * sr // 48000 or 256
    lifter = max(8, sr // 1000)                     # 1 ms: below any voice's pitch period
    _, _, X = stft(x, nperseg=n_fft, noverlap=n_fft - hop, boundary="even")
    _, _, Y = stft(y, nperseg=n_fft, noverlap=n_fft - hop, boundary="even")
    g = np.clip(_envelope(X, lifter) / _envelope(Y, lifter), 0.05, 20.0)
    _, z = istft(Y * g, nperseg=n_fft, noverlap=n_fft - hop, boundary=True)
    z = np.asarray(z[: len(x)], np.float32)
    return np.pad(z, (0, len(x) - len(z)))


def energetic_head(wav: np.ndarray, sr: int, frac: float = 0.45, total_s: float = 6.5) -> np.ndarray:
    """The first ``frac`` of the speech in ``wav`` (before a narrator's voice
    falls), looped with short pauses to ``total_s``."""
    y = np.asarray(wav, np.float32)
    hop = max(1, sr // 100)
    n = len(y) // hop
    if n < 10:
        return y
    rms = np.sqrt((y[:n * hop].reshape(n, hop) ** 2).mean(1))
    act = np.where(rms > 0.1 * rms.max())[0]
    a = act[0] * hop
    b = a + max(int((act[-1] - act[0]) * hop * frac), sr // 2)
    head = y[a:b].copy()
    f = min(len(head) // 4, sr // 100)
    if f:
        ramp = np.linspace(0, 1, f, dtype=np.float32)
        head[:f] *= ramp
        head[-f:] *= ramp[::-1]
    unit = np.concatenate([head, np.zeros(int(0.15 * sr), np.float32)])
    reps = int(np.ceil(total_s * sr / len(unit)))
    return np.tile(unit, reps)[: int(total_s * sr)]


_CLAUSE_RE = re.compile(r"[^,;:!?.…]+[,;:!?.…]*")


def clauses(text: str) -> List[str]:
    """``text`` cut after each , ; : ! ? . (the mark stays with its clause).
    Bracketed cues stay whole; pieces without a word join the one before."""
    out: List[str] = []
    for c in _CLAUSE_RE.findall(text):
        if out and (not re.search(r"\w", c) or out[-1].count("[") > out[-1].count("]")):
            out[-1] += c
        elif c.strip():
            out.append(c)
    return [c.strip() for c in out if c.strip()] or [text]


class Leveler:
    """Streaming loudness leveler: follows the RMS (fast attack, slow release)
    and gains speech toward ``target_db`` dBFS, at most ``max_boost_db`` up;
    pauses (under -40 dBFS) keep the gain they had. Ends in a soft limiter.
    No look-ahead, so no added latency."""

    def __init__(self, target_db: float, sr: int = 48000, max_boost_db: float = 12.0,
                 attack: float = 0.01, release: float = 0.35):
        self.hop = max(1, sr // 200)
        self.target = 10 ** (target_db / 20)
        self.max_g = 10 ** (max_boost_db / 20)
        self.a_att = math.exp(-self.hop / (attack * sr))
        self.a_rel = math.exp(-self.hop / (release * sr))
        self.gate = 10 ** (-40 / 20)
        self.env, self.g = None, 1.0
        self.carry = np.zeros(0, np.float32)

    def __call__(self, x: np.ndarray, last: bool = False) -> np.ndarray:
        x = np.concatenate([self.carry, np.asarray(x, np.float32)])
        size = len(x)                       # real samples; the last hop may be padded
        n = len(x) // self.hop
        if last and size > n * self.hop:
            x = np.concatenate([x, np.zeros((n + 1) * self.hop - size, np.float32)])
            n += 1
        body, self.carry = x[:n * self.hop], x[n * self.hop:]
        if n == 0:
            return np.zeros(0, np.float32)
        out_len = min(size, n * self.hop)
        rms = np.sqrt((body.reshape(n, self.hop) ** 2).mean(1)) + 1e-9
        gains = np.empty(n, np.float32)
        for i in range(n):
            r = rms[i]
            if self.env is None:
                self.env = r
            a = self.a_att if r > self.env else self.a_rel
            self.env = a * self.env + (1 - a) * r
            if self.env > self.gate:
                self.g = 0.9 * self.g + 0.1 * min(self.max_g, self.target / self.env)
            gains[i] = self.g
        return np.tanh(body * np.repeat(gains, self.hop))[:out_len].astype(np.float32)


def apply_gain(x: np.ndarray, gain: float) -> np.ndarray:
    """Output gain; above 1 a soft limiter keeps peaks under full scale."""
    if gain == 1.0:
        return x
    y = x * np.float32(gain)
    if gain > 1.0:
        y = np.tanh(y)
    return y.astype(np.float32)


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
            y = np.concatenate([y, np.zeros(short, np.float32)])
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
