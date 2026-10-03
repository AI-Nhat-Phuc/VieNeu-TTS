"""
Audio tabs of the desktop monitor.

Studio   write a text, have the local server read it sentence by sentence, listen
         with a transcript that follows the voice, then save WAV + SRT on this
         machine or upload them to the R2 bucket.
Falevon  the narration Falevon has already stored in R2 (production ``audio/``
         and staging ``audio-staging/``), grouped by world and chapter, played
         chapter after chapter without gaps, with a link to each chapter.

Playback uses the Windows waveOut API through ctypes (gapless queueing and a
real play position, no extra packages); R2 is spoken to with SigV4 over urllib.
Settings come from the environment or the repo's ``.env`` with the names
story-services uses: AUDIO_S3_ENDPOINT, AUDIO_S3_BUCKET, AUDIO_S3_ACCESS_KEY_ID,
AUDIO_S3_SECRET_ACCESS_KEY, AUDIO_PUBLIC_BASE_URL (optional), AUDIO_S3_REGION.
"""
from __future__ import annotations

import collections
import ctypes
import datetime as dt
import hashlib
import hmac
import io
import json
import os
import queue
import re
import threading
import time
import tkinter as tk
import unicodedata
import urllib.parse
import urllib.request
import webbrowser
import xml.etree.ElementTree as ET
from ctypes import wintypes
from tkinter import filedialog, ttk

import numpy as np

ENV_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
SITES = {"Production": ("https://www.falevon.com", "AUDIO_S3_PREFIX_PROD", "audio"),
         "Staging": ("https://staging.falevon.com", "AUDIO_S3_PREFIX_STAGING", "audio-staging")}
SENTENCE_GAP_S, PARAGRAPH_GAP_S = 0.28, 0.7


def env_value(name: str, default: str = "") -> str:
    """``name`` from the environment, else from the repo's ``.env``."""
    if os.environ.get(name):
        return os.environ[name]
    try:
        with open(ENV_FILE, encoding="utf-8") as f:
            for line in f:
                k, _, v = line.strip().partition("=")
                if k.strip() == name and not line.lstrip().startswith("#"):
                    return v.strip().strip('"').strip("'")
    except OSError:
        pass
    return default


# ── text ──────────────────────────────────────────────────────────────────────
def split_sentences(text: str, max_chars: int = 220) -> list[tuple[str, bool]]:
    """``[(sentence, ends_paragraph)]``: sentences on . ! ? … ; too long ones cut at commas, then spaces."""
    out = []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n|\r?\n", text) if p.strip()]
    for pi, para in enumerate(paragraphs):
        parts = [s.strip() for s in re.split(r"(?<=[.!?…])\s+(?=\S)", para) if s.strip()]
        merged = []
        for s in parts:                       # a tiny fragment ("Ừ.", "Hết.") joins the sentence before it
            if merged and len(s) < 8:
                merged[-1] = f"{merged[-1]} {s}"
            else:
                merged.append(s)
        pieces = []
        for s in merged:
            while len(s) > max_chars:
                cut = max(s.rfind(", ", 0, max_chars), s.rfind("; ", 0, max_chars))
                cut = cut + 1 if cut > max_chars // 3 else (s.rfind(" ", 0, max_chars) if s.rfind(" ", 0, max_chars) > 0
                                                              else max_chars)
                pieces.append(s[:cut].strip())
                s = s[cut:].strip()
            if s:
                pieces.append(s)
        for i, s in enumerate(pieces):
            out.append((s, i == len(pieces) - 1 and pi < len(paragraphs) - 1))
    return out


def srt_time(t: float) -> str:
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def to_srt(segments) -> str:
    return "\n".join(f"{i}\n{srt_time(a)} --> {srt_time(b)}\n{txt}\n" for i, (a, b, txt) in enumerate(segments, 1))


def slugify(text: str, n: int = 6) -> str:
    t = unicodedata.normalize("NFKD", text.replace("đ", "d").replace("Đ", "D"))
    t = "".join(c for c in t if not unicodedata.combining(c)).lower()
    words = re.findall(r"[a-z0-9]+", t)[:n]
    return "-".join(words) or "audio"


def wav_bytes(pcm: np.ndarray, rate: int) -> bytes:
    import wave
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(np.asarray(pcm, dtype="<i2").tobytes())
    return b.getvalue()


def fmt_time(s: float) -> str:
    s = max(0, int(s))
    return f"{s // 60}:{s % 60:02d}" if s < 3600 else f"{s // 3600}:{s // 60 % 60:02d}:{s % 60:02d}"


# ── playback: waveOut ─────────────────────────────────────────────────────────
class _WAVEFORMATEX(ctypes.Structure):
    _fields_ = [("wFormatTag", wintypes.WORD), ("nChannels", wintypes.WORD), ("nSamplesPerSec", wintypes.DWORD),
                ("nAvgBytesPerSec", wintypes.DWORD), ("nBlockAlign", wintypes.WORD),
                ("wBitsPerSample", wintypes.WORD), ("cbSize", wintypes.WORD)]


class _WAVEHDR(ctypes.Structure):
    _fields_ = [("lpData", ctypes.c_void_p), ("dwBufferLength", wintypes.DWORD),
                ("dwBytesRecorded", wintypes.DWORD), ("dwUser", ctypes.c_size_t), ("dwFlags", wintypes.DWORD),
                ("dwLoops", wintypes.DWORD), ("lpNext", ctypes.c_void_p), ("reserved", ctypes.c_size_t)]


class _MMTIME(ctypes.Structure):
    _fields_ = [("wType", wintypes.UINT), ("u", wintypes.DWORD * 2)]


class WavePlayer:
    """Mono 16-bit PCM out of the default device. ``enqueue`` appends without a gap;
    ``position`` is the sample being heard. Call from the UI thread only."""

    WHDR_DONE, TIME_SAMPLES = 0x1, 0x2

    def __init__(self):
        self.winmm = ctypes.windll.winmm
        self.h = ctypes.c_void_p()
        self.rate = 0
        self.pending: list = []          # (header, buffer) until the device is done with them
        self.queued = 0                  # samples handed to the device since open
        self.paused = False

    def _open(self, rate: int) -> None:
        self.close()
        fmt = _WAVEFORMATEX(1, 1, rate, rate * 2, 2, 16, 0)
        r = self.winmm.waveOutOpen(ctypes.byref(self.h), wintypes.UINT(0xFFFFFFFF), ctypes.byref(fmt), 0, 0, 0)
        if r != 0:
            self.h = ctypes.c_void_p()
            raise OSError(f"waveOutOpen failed ({r}): no audio output device?")
        self.rate, self.queued, self.paused = rate, 0, False

    def enqueue(self, pcm: np.ndarray, rate: int) -> None:
        if not self.h or rate != self.rate:
            self._open(rate)
        data = np.ascontiguousarray(pcm, dtype="<i2").tobytes()
        if not data:
            return
        buf = ctypes.create_string_buffer(data, len(data))
        hdr = _WAVEHDR(ctypes.cast(buf, ctypes.c_void_p), len(data), 0, 0, 0, 0, None, 0)
        self.winmm.waveOutPrepareHeader(self.h, ctypes.byref(hdr), ctypes.sizeof(hdr))
        self.winmm.waveOutWrite(self.h, ctypes.byref(hdr), ctypes.sizeof(hdr))
        self.pending.append((hdr, buf))
        self.queued += len(data) // 2

    def reap(self) -> None:
        """Release buffers the device has finished with."""
        keep = []
        for hdr, buf in self.pending:
            if hdr.dwFlags & self.WHDR_DONE:
                self.winmm.waveOutUnprepareHeader(self.h, ctypes.byref(hdr), ctypes.sizeof(hdr))
            else:
                keep.append((hdr, buf))
        self.pending = keep

    def position(self) -> int:
        if not self.h:
            return 0
        t = _MMTIME(self.TIME_SAMPLES)
        self.winmm.waveOutGetPosition(self.h, ctypes.byref(t), ctypes.sizeof(t))
        pos = int(t.u[0]) // 2 if t.wType == 0x4 else int(t.u[0])     # TIME_BYTES on drivers without samples
        return min(pos, self.queued)

    @property
    def playing(self) -> bool:
        return bool(self.h) and bool(self.pending)

    def pause(self, on: bool) -> None:
        if self.h:
            (self.winmm.waveOutPause if on else self.winmm.waveOutRestart)(self.h)
            self.paused = on

    def close(self) -> None:
        if self.h:
            self.winmm.waveOutReset(self.h)
            self.reap()
            self.winmm.waveOutClose(self.h)
        self.h, self.pending, self.queued, self.rate, self.paused = ctypes.c_void_p(), [], 0, 0, False


# ── R2 (S3 API, SigV4) ────────────────────────────────────────────────────────
def sigv4_headers(method, host, path, query, headers, payload_hash, access_key, secret_key, region, now):
    """Authorization (+ x-amz-*) headers for one S3 request (AWS Signature V4)."""
    amz_date, date = now.strftime("%Y%m%dT%H%M%SZ"), now.strftime("%Y%m%d")
    h = {k.lower(): str(v).strip() for k, v in headers.items()}
    h.update({"host": host, "x-amz-content-sha256": payload_hash, "x-amz-date": amz_date})
    signed = ";".join(sorted(h))
    q = "&".join(f"{urllib.parse.quote(k, safe='-_.~')}={urllib.parse.quote(str(v), safe='-_.~')}"
                 for k, v in sorted(query.items()))
    creq = "\n".join([method, path, q, "".join(f"{k}:{h[k]}\n" for k in sorted(h)), signed, payload_hash])
    scope = f"{date}/{region}/s3/aws4_request"
    sts = "\n".join(["AWS4-HMAC-SHA256", amz_date, scope, hashlib.sha256(creq.encode()).hexdigest()])
    key = f"AWS4{secret_key}".encode()
    for part in (date, region, "s3", "aws4_request"):
        key = hmac.new(key, part.encode(), hashlib.sha256).digest()
    sig = hmac.new(key, sts.encode(), hashlib.sha256).hexdigest()
    out = {k: v for k, v in h.items() if k != "host"}
    out["authorization"] = f"AWS4-HMAC-SHA256 Credential={access_key}/{scope}, SignedHeaders={signed}, Signature={sig}"
    return out


class R2:
    NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"

    def __init__(self):
        self.bucket = env_value("AUDIO_S3_BUCKET")
        # Cloudflare's dashboard shows the S3 URL with "/<bucket>" on the end; requests add the bucket
        # themselves, so only the origin is kept (a doubled bucket fails the signature).
        u = urllib.parse.urlparse(env_value("AUDIO_S3_ENDPOINT").rstrip("/"))
        self.endpoint = f"{u.scheme}://{u.netloc}" if u.netloc else ""
        self.ak, self.sk = env_value("AUDIO_S3_ACCESS_KEY_ID"), env_value("AUDIO_S3_SECRET_ACCESS_KEY")
        self.region = env_value("AUDIO_S3_REGION", "auto")
        self.public = env_value("AUDIO_PUBLIC_BASE_URL").rstrip("/")

    def endpoint_problem(self) -> str:
        """Why the endpoint cannot be the S3 API, or "" (a bucket's public domain serves files only)."""
        host = urllib.parse.urlparse(self.endpoint).netloc
        if host and not host.endswith((".r2.cloudflarestorage.com", ".amazonaws.com")) and                 not host.startswith(("localhost", "127.0.0.1")):
            return (f"AUDIO_S3_ENDPOINT ({host}) looks like the bucket's public domain, which serves files "
                    "but cannot list or upload. Use the S3 API URL from Cloudflare → R2 → bucket → Settings "
                    "(https://<account-id>.r2.cloudflarestorage.com) and put this domain in AUDIO_PUBLIC_BASE_URL.")
        return ""

    def missing(self) -> list[str]:
        return [n for n, v in (("AUDIO_S3_ENDPOINT", self.endpoint), ("AUDIO_S3_BUCKET", self.bucket),
                               ("AUDIO_S3_ACCESS_KEY_ID", self.ak), ("AUDIO_S3_SECRET_ACCESS_KEY", self.sk)) if not v]

    def _request(self, method, key="", query=None, body=b"", headers=None, timeout=30):
        query = query or {}
        path = "/" + self.bucket + ("/" + urllib.parse.quote(key, safe="/-_.~") if key else "")
        host = urllib.parse.urlparse(self.endpoint).netloc
        h = sigv4_headers(method, host, path, query, headers or {}, hashlib.sha256(body).hexdigest(), self.ak,
                          self.sk, self.region, dt.datetime.now(dt.timezone.utc))
        url = self.endpoint + path + ("?" + urllib.parse.urlencode(query) if query else "")
        # urllib's default User-Agent is refused by Cloudflare's browser check (error 1010) on a
        # custom-domain endpoint; it is not a signed header, so any name works.
        req = urllib.request.Request(url, data=body if method == "PUT" else None, method=method,
                                     headers={**h, "User-Agent": "vieneu-monitor/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()

    def list(self, prefix: str):
        """Every object under ``prefix``: ``[(key, size, last_modified)]``."""
        out, token = [], None
        while True:
            q = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
            if token:
                q["continuation-token"] = token
            root = ET.fromstring(self._request("GET", query=q))
            for c in root.iter(self.NS + "Contents"):
                out.append((c.findtext(self.NS + "Key"), int(c.findtext(self.NS + "Size") or 0),
                            c.findtext(self.NS + "LastModified") or ""))
            if root.findtext(self.NS + "IsTruncated") != "true":
                return out
            token = root.findtext(self.NS + "NextContinuationToken")

    def get(self, key: str) -> bytes:
        if self.public:
            url = f"{self.public}/{urllib.parse.quote(key, safe='/-_.~')}"
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "vieneu-monitor"}),
                                        timeout=30) as r:
                return r.read()
        return self._request("GET", key)

    def put(self, key: str, data: bytes, content_type: str) -> str:
        self._request("PUT", key, body=data, headers={"content-type": content_type}, timeout=120)
        return f"{self.public}/{urllib.parse.quote(key, safe='/-_.~')}" if self.public else f"r2://{self.bucket}/{key}"


def decode_audio(data: bytes) -> tuple[np.ndarray, int]:
    import soundfile as sf
    x, rate = sf.read(io.BytesIO(data), dtype="int16", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1).astype("int16")
    return x, rate


def http_get_json(url: str, timeout: float = 10.0, headers: dict | None = None):
    req = urllib.request.Request(url, headers={"User-Agent": "vieneu-monitor", **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


# ── shared bits of UI ────────────────────────────────────────────────────────
class _Tab(ttk.Frame):
    def __init__(self, parent, colors):
        super().__init__(parent, padding=(0, 8, 0, 0))
        self.C = colors

    def _text(self, parent, **kw):
        return tk.Text(parent, bg="#0d1117", fg=self.C["fg"], insertbackground=self.C["fg"], borderwidth=0,
                       padx=10, pady=8, font=("Segoe UI", 11), wrap="word", selectbackground="#264f78", **kw)

    def _tree(self, parent, columns, widths, height=10):
        t = ttk.Treeview(parent, columns=columns, height=height, style="Dark.Treeview")
        for col, (title, w, anchor) in zip(("#0",) + tuple(columns), widths):
            t.heading(col, text=title, anchor=anchor)
            t.column(col, width=w, anchor=anchor, stretch=col in ("#0", "text"))
        sy = ttk.Scrollbar(parent, orient="vertical", command=t.yview)
        t.configure(yscrollcommand=sy.set)
        return t, sy


def style_trees(style: ttk.Style, c: dict) -> None:
    style.configure("Dark.Treeview", background="#0d1117", fieldbackground="#0d1117", foreground=c["fg"],
                    rowheight=26, borderwidth=0, font=("Segoe UI", 10))
    style.configure("Dark.Treeview.Heading", background=c["panel"], foreground=c["muted"], borderwidth=0,
                    font=("Segoe UI", 9))
    style.map("Dark.Treeview", background=[("selected", "#264f78")], foreground=[("selected", "#ffffff")])
    style.configure("TCombobox", fieldbackground=c["panel"], background=c["panel"], foreground=c["fg"],
                    arrowcolor=c["fg"], selectbackground=c["panel"], selectforeground=c["fg"])
    style.map("TCombobox", fieldbackground=[("readonly", c["panel"])], foreground=[("readonly", c["fg"])],
              selectbackground=[("readonly", c["panel"])], selectforeground=[("readonly", c["fg"])])
    for opt, val in (("background", c["panel"]), ("foreground", c["fg"]), ("selectBackground", "#264f78")):
        style.master.option_add(f"*TCombobox*Listbox.{opt}", val)      # the drop-down list
    style.configure("Accent.TButton", background="#238636", foreground="#ffffff")
    style.map("Accent.TButton", background=[("active", "#2ea043"), ("disabled", "#21262d")])
    style.configure("TRadiobutton", background=c["bg"], foreground=c["fg"])
    style.configure("TCheckbutton", background=c["bg"], foreground=c["fg"])


# ── Studio: write → generate → listen → publish ──────────────────────────────
class StudioTab(_Tab):
    def __init__(self, parent, colors, local_url: str, key: str):
        super().__init__(parent, colors)
        self.local_url, self.key = local_url, key
        self.player = WavePlayer()
        self.pcm, self.rate = np.zeros(0, np.int16), 48000
        self.segments: list = []         # (start_s, end_s, text)
        self.play_from = 0               # sample the current playback started at
        self.gen_id = 0
        self.generating = False
        self.r2 = R2()
        self._build()
        threading.Thread(target=self._load_voices, daemon=True).start()
        self.after(100, self._tick)

    def _build(self):
        C = self.C
        top = ttk.Frame(self)
        top.pack(fill="both", expand=True)
        left = ttk.Frame(top)
        left.pack(side="left", fill="both", expand=True)
        ttk.Label(left, text="Text to read", foreground=C["muted"]).pack(anchor="w")
        self.text = self._text(left, height=10)
        self.text.pack(fill="both", expand=True, pady=(2, 6))
        self.text.insert("1.0", "Xin chào! Đây là VieNeu-TTS chạy ngay trên máy này.\n\n"
                                "Bạn có thể gõ một đoạn văn bất kỳ, chọn giọng đọc, rồi bấm Generate.")
        row = ttk.Frame(left)
        row.pack(fill="x")
        ttk.Label(row, text="Voice").pack(side="left")
        self.voice = ttk.Combobox(row, width=22, state="readonly", values=["(default)"])
        self.voice.set("(default)")
        self.voice.pack(side="left", padx=(6, 14))
        ttk.Label(row, text="Sample rate").pack(side="left")
        self.rate_box = ttk.Combobox(row, width=8, state="readonly", values=["48000", "24000"])
        self.rate_box.set("48000")
        self.rate_box.pack(side="left", padx=(6, 14))
        self.b_gen = ttk.Button(row, text="Generate", style="Accent.TButton", command=self._generate)
        self.b_gen.pack(side="left")
        self.b_cancel = ttk.Button(row, text="Cancel", command=self._cancel, state="disabled")
        self.b_cancel.pack(side="left", padx=6)
        self.l_status = ttk.Label(row, text="", foreground=C["muted"])
        self.l_status.pack(side="left", padx=8)

        right = ttk.Frame(top, padding=(12, 0, 0, 0))
        right.pack(side="left", fill="both", expand=True)
        ttk.Label(right, text="Transcript (click a line to play from there)", foreground=C["muted"]).pack(anchor="w")
        tf = ttk.Frame(right)
        tf.pack(fill="both", expand=True, pady=(2, 0))
        self.tr, sy = self._tree(tf, ("start", "text"), [("#", 40, "e"), ("time", 70, "e"), ("sentence", 420, "w")])
        self.tr.column("text", stretch=True)
        sy.pack(side="right", fill="y")
        self.tr.pack(fill="both", expand=True)
        self.tr.tag_configure("now", background="#1f3a2a", foreground="#7ee787")
        self.tr.tag_configure("todo", foreground=C["muted"])
        self.tr.bind("<<TreeviewSelect>>", self._seek_selected)

        wave = ttk.Frame(self, padding=(0, 10, 0, 0))
        wave.pack(fill="x")
        self.wave = tk.Canvas(wave, height=96, bg="#0d1117", highlightthickness=0)
        self.wave.pack(fill="x")
        self.wave.bind("<Button-1>", self._seek_click)
        self.wave.bind("<Configure>", lambda e: self._draw_wave(full=True))

        bar = ttk.Frame(self, padding=(0, 8, 0, 0))
        bar.pack(fill="x")
        self.b_play = ttk.Button(bar, text="▶  Play", command=self._toggle, state="disabled")
        self.b_play.pack(side="left")
        ttk.Button(bar, text="■  Stop", command=self._stop).pack(side="left", padx=6)
        self.l_time = ttk.Label(bar, text="0:00 / 0:00", foreground=C["muted"])
        self.l_time.pack(side="left", padx=10)
        self.b_r2 = ttk.Button(bar, text="Upload to R2", command=self._upload, state="disabled")
        self.b_r2.pack(side="right")
        self.b_save = ttk.Button(bar, text="Save WAV + SRT…", command=self._save, state="disabled")
        self.b_save.pack(side="right", padx=6)
        self.url = tk.StringVar()
        self.e_url = ttk.Entry(bar, textvariable=self.url, width=46, state="readonly")
        self.e_url.pack(side="right", padx=6)
        self.l_pub = ttk.Label(bar, text="", foreground=C["muted"], wraplength=520)
        self.l_pub.pack(side="right")
        self._wave_cache = None

    # voices -----------------------------------------------------------------
    def _load_voices(self):
        try:
            body = http_get_json(self.local_url + "/v1/voices", 5,
                                 {"Authorization": f"Bearer {self.key}"} if self.key else None)
            names = ["(default)"] + [v["id"] for v in body.get("data", [])]
            self.after(0, lambda: self.voice.configure(values=names))
        except Exception as e:  # server down: keep "(default)", say why
            self.after(0, lambda: self.l_status.configure(text=f"voices unavailable: {str(e)[:60]}"))

    # generate ---------------------------------------------------------------
    def _generate(self):
        text = self.text.get("1.0", "end").strip()
        if not text:
            self.l_status.configure(text="write something first")
            return
        self._stop()
        self.gen_id += 1
        gid, rate = self.gen_id, int(self.rate_box.get())
        voice = None if self.voice.get() == "(default)" else self.voice.get()
        sentences = split_sentences(text)
        self.pcm, self.rate, self.segments = np.zeros(0, np.int16), rate, []
        self.tr.delete(*self.tr.get_children())
        for i, (s, _) in enumerate(sentences):
            self.tr.insert("", "end", iid=str(i), text=str(i + 1), values=("", s), tags=("todo",))
        self.generating = True
        self.b_gen.configure(state="disabled")
        self.b_cancel.configure(state="normal")
        for b in (self.b_play, self.b_save, self.b_r2):
            b.configure(state="disabled")
        self.url.set("")
        self.l_pub.configure(text="")
        threading.Thread(target=self._gen_worker, args=(gid, sentences, voice, rate), daemon=True).start()

    def _gen_worker(self, gid, sentences, voice, rate):
        t0 = time.time()
        for i, (s, para_end) in enumerate(sentences):
            if gid != self.gen_id:
                return
            self.after(0, lambda i=i: self.l_status.configure(text=f"reading sentence {i + 1}/{len(sentences)}…"))
            body = {"input": s, "response_format": "pcm", "sample_rate": rate}
            if voice:
                body["voice"] = voice
            req = urllib.request.Request(self.local_url + "/v1/audio/speech", data=json.dumps(body).encode(),
                                         method="POST", headers={"Content-Type": "application/json",
                                                                 "User-Agent": "vieneu-studio",
                                                                 **({"Authorization": f"Bearer {self.key}"}
                                                                    if self.key else {})})
            try:
                with urllib.request.urlopen(req, timeout=300) as r:
                    audio = np.frombuffer(r.read(), dtype="<i2")
            except Exception as e:
                msg = str(getattr(e, "code", "") or e)[:80]
                self.after(0, lambda: self._gen_done(gid, f"failed at sentence {i + 1}: {msg}"))
                return
            gap = np.zeros(int(rate * (PARAGRAPH_GAP_S if para_end else SENTENCE_GAP_S)), np.int16) \
                if i < len(sentences) - 1 else np.zeros(0, np.int16)
            self.after(0, lambda a=audio, g=gap, s=s, i=i: self._add_sentence(gid, i, s, a, g))
        self.after(0, lambda: self._gen_done(gid, f"done in {time.time() - t0:.1f} s"))

    def _add_sentence(self, gid, i, s, audio, gap):
        if gid != self.gen_id:
            return
        old = len(self.pcm)
        start = old / self.rate
        self.segments.append((start, start + len(audio) / self.rate, s))
        self.pcm = np.concatenate([self.pcm, audio, gap])
        self.tr.item(str(i), values=(fmt_time(start), s), tags=())
        self._draw_wave(full=True)
        self.b_play.configure(state="normal")
        if i == 0:                                 # start listening while the rest is generated
            self._play_from(0)
        elif self.player.h and self.play_from + self.player.queued == old:   # playing up to here: extend it
            self.player.enqueue(np.concatenate([audio, gap]), self.rate)

    def _gen_done(self, gid, msg):
        if gid != self.gen_id:
            return
        self.generating = False
        self.l_status.configure(text=msg)
        self.b_gen.configure(state="normal")
        self.b_cancel.configure(state="disabled")
        if len(self.pcm):
            self.b_save.configure(state="normal")
            self.b_r2.configure(state="normal")

    def _cancel(self):
        self.gen_id += 1
        self._gen_done(self.gen_id, "cancelled")

    # playback ---------------------------------------------------------------
    def _play_from(self, sample: int):
        self.player.close()
        if sample >= len(self.pcm):
            return
        self.play_from = sample
        self.player.enqueue(self.pcm[sample:], self.rate)
        self.b_play.configure(text="❚❚  Pause")

    def _toggle(self):
        if self.player.playing:
            self.player.pause(not self.player.paused)
            self.b_play.configure(text="▶  Play" if self.player.paused else "❚❚  Pause")
        else:
            self._play_from(0)

    def _stop(self):
        self.player.close()
        self.b_play.configure(text="▶  Play")

    def _now(self) -> float:
        return (self.play_from + self.player.position()) / self.rate if self.player.playing else -1.0

    def _seek_selected(self, _e=None):
        sel = self.tr.selection()
        if sel and int(sel[0]) < len(self.segments):
            self._play_from(int(self.segments[int(sel[0])][0] * self.rate))

    def _seek_click(self, e):
        if len(self.pcm):
            w = max(1, self.wave.winfo_width())
            self._play_from(int(len(self.pcm) * e.x / w))

    def _draw_wave(self, full=False):
        c, C = self.wave, self.C
        w, h = max(c.winfo_width(), 100), max(c.winfo_height(), 40)
        if full or self._wave_cache != (w, len(self.pcm)):
            c.delete("all")
            self._wave_cache = (w, len(self.pcm))
            if not len(self.pcm):
                c.create_text(w / 2, h / 2, text="the waveform appears here as the sentences are read",
                              fill=C["muted"], font=("Segoe UI", 9))
                return
            total = len(self.pcm)
            for i, (a, b, _) in enumerate(self.segments):      # alternate shading per sentence
                x0, x1 = a * self.rate / total * w, b * self.rate / total * w
                c.create_rectangle(x0, 0, x1, h, fill="#111a24" if i % 2 else "#0f1620", outline="", tags="seg")
            n = int(w)
            env = np.abs(self.pcm[: total // n * n].astype(np.float32)).reshape(n, -1).max(axis=1) / 32768 \
                if total >= n else np.zeros(0)
            pts_t, pts_b = [], []
            for x, v in enumerate(env):
                a = (h / 2 - 6) * min(1.0, v * 1.4)
                pts_t += [x, h / 2 - a]
                pts_b = [x, h / 2 + a] + pts_b
            if len(pts_t) >= 4:
                c.create_polygon(*pts_t, *pts_b, fill="#1f6feb", outline="#58a6ff", tags="wave")
        c.delete("head")
        now = self._now()
        if now >= 0 and len(self.pcm):
            x = now * self.rate / len(self.pcm) * w
            c.create_line(x, 0, x, h, fill="#f0883e", width=2, tags="head")

    def _tick(self):
        try:
            self.player.reap()
            if self.player.h and not self.player.pending and not self.player.paused and not self.generating:
                self._stop()                       # (while generating, the next sentence will extend it)
            now = self._now()
            total = len(self.pcm) / self.rate if len(self.pcm) else 0
            self.l_time.configure(text=f"{fmt_time(max(0, now))} / {fmt_time(total)}")
            self._draw_wave()
            cur = next((i for i, (a, b, _) in enumerate(self.segments) if a <= now < b + 0.3), None)
            for iid in self.tr.tag_has("now"):
                if iid != str(cur):
                    self.tr.item(iid, tags=())
            if cur is not None and "now" not in self.tr.item(str(cur), "tags"):
                self.tr.item(str(cur), tags=("now",))
                self.tr.see(str(cur))
        finally:
            self.after(80, self._tick)

    # publish ----------------------------------------------------------------
    def _bundle(self):
        text = " ".join(s for _, _, s in self.segments)
        return text, wav_bytes(self.pcm, self.rate), to_srt(self.segments)

    def _save(self):
        text, wav, srt = self._bundle()
        path = filedialog.asksaveasfilename(defaultextension=".wav", initialfile=slugify(text) + ".wav",
                                            filetypes=[("WAV audio", "*.wav")])
        if not path:
            return
        with open(path, "wb") as f:
            f.write(wav)
        with open(os.path.splitext(path)[0] + ".srt", "w", encoding="utf-8") as f:
            f.write(srt)
        self.l_pub.configure(text=f"saved {os.path.basename(path)} + .srt")

    def _upload(self):
        missing = self.r2.missing()
        if missing:
            self.l_pub.configure(text="R2 not configured: add " + ", ".join(missing) + " to .env")
            return
        text, wav, srt = self._bundle()
        digest = hashlib.sha256(wav).hexdigest()[:10]
        base = f"{env_value('MANUAL_AUDIO_PREFIX', 'manual').strip('/')}/{dt.date.today():%Y-%m-%d}/" \
               f"{slugify(text)}-{digest}"
        meta = json.dumps({"text": text, "sample_rate": self.rate, "voice": self.voice.get(),
                           "segments": [{"start": round(a, 3), "end": round(b, 3), "text": s}
                                        for a, b, s in self.segments]}, ensure_ascii=False).encode()
        self.b_r2.configure(state="disabled")
        self.l_pub.configure(text="uploading…")

        def run():
            try:
                url = self.r2.put(base + ".wav", wav, "audio/wav")
                self.r2.put(base + ".srt", srt.encode(), "application/x-subrip; charset=utf-8")
                self.r2.put(base + ".json", meta, "application/json")
                msg, ok = "uploaded (URL copied)", url
            except Exception as e:
                msg, ok = self.r2.endpoint_problem() or f"upload failed: {str(getattr(e, 'code', '') or e)[:70]}", None
            self.after(0, lambda: self._uploaded(msg, ok))
        threading.Thread(target=run, daemon=True).start()

    def _uploaded(self, msg, url):
        self.l_pub.configure(text=msg)
        self.b_r2.configure(state="normal")
        if url:
            self.url.set(url)
            self.clipboard_clear()
            self.clipboard_append(url)


# ── Falevon library: stored narration by world / chapter ─────────────────────
class LibraryTab(_Tab):
    PREFETCH = 24                        # decoded chunks kept ahead of the voice

    def __init__(self, parent, colors):
        super().__init__(parent, colors)
        self.r2 = R2()
        self.player = WavePlayer()
        self.worlds: dict = {}           # world_id -> {title, slug, chapters: [chapter dicts in order]}
        self.timeline: list = []         # (start_sample, end_sample, chapter, chunk_idx) of what was queued
        self.buffer: queue.Queue = queue.Queue(maxsize=self.PREFETCH)
        self.play_id = 0
        self.now_chapter = None
        self._build()
        self.after(300, self.refresh)
        self.after(100, self._tick)

    def _build(self):
        C = self.C
        bar = ttk.Frame(self)
        bar.pack(fill="x")
        self.site = tk.StringVar(value="Production")
        for name in SITES:
            ttk.Radiobutton(bar, text=name, value=name, variable=self.site,
                            command=self.refresh).pack(side="left", padx=(0, 10))
        ttk.Button(bar, text="Refresh", command=self.refresh).pack(side="left", padx=6)
        self.auto_next = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="Continue into the next chapter", variable=self.auto_next).pack(side="left", padx=12)
        self.l_status = ttk.Label(bar, text="", foreground=C["muted"], wraplength=760)
        self.l_status.pack(side="left", padx=8)

        tf = ttk.Frame(self, padding=(0, 8, 0, 0))
        tf.pack(fill="both", expand=True)
        self.tree, sy = self._tree(tf, ("sentences", "codec", "size", "updated"),
                                   [("world / chapter", 520, "w"), ("sentences", 90, "e"), ("codec", 70, "center"),
                                    ("size", 90, "e"), ("updated", 150, "w")], height=14)
        sy.pack(side="right", fill="y")
        self.tree.pack(fill="both", expand=True)
        self.tree.tag_configure("world", font=("Segoe UI Semibold", 10), foreground="#e6edf3")
        self.tree.tag_configure("playing", background="#1f3a2a", foreground="#7ee787")
        self.tree.bind("<Double-1>", lambda e: self._play_selected())

        player = ttk.Frame(self, style="Card.TFrame", padding=(12, 10))
        player.pack(fill="x", pady=(8, 0))
        top = tk.Frame(player, bg=C["panel"])
        top.pack(fill="x")
        for text, fn in (("⏮", self._prev), ("▶  Play", self._play_selected), ("❚❚", self._pause),
                         ("■", self._stop), ("⏭", self._next)):
            b = ttk.Button(top, text=text, command=fn, width=8 if len(text) > 2 else 4)
            b.pack(side="left", padx=(0, 4))
            if text.startswith("▶"):
                self.b_play = b
        self.l_now = tk.Label(top, text="Pick a chapter and press Play (or double-click it)", bg=C["panel"],
                              fg=C["fg"], font=("Segoe UI Semibold", 11), anchor="w")
        self.l_now.pack(side="left", padx=12, fill="x", expand=True)
        ttk.Button(top, text="Open on Falevon ↗", command=self._open_link).pack(side="right")
        self.prog = tk.Canvas(player, height=26, bg=C["panel"], highlightthickness=0)
        self.prog.pack(fill="x", pady=(8, 0))
        self.prog.bind("<Button-1>", self._seek_click)

    # data -------------------------------------------------------------------
    def refresh(self):
        site = self.site.get()
        missing = self.r2.missing()
        self.tree.delete(*self.tree.get_children())
        if missing:
            self.l_status.configure(text="R2 not configured: add " + ", ".join(missing) + " to .env "
                                         "(the values story-services uses)")
            return
        self.l_status.configure(text=f"listing {site} narration in R2…")
        threading.Thread(target=self._load, args=(site,), daemon=True).start()

    def _load(self, site):
        url, var, default = SITES[site]
        prefix = env_value(var, default).strip("/") + "/"
        try:
            objects = self.r2.list(prefix)
        except Exception as e:
            msg = self.r2.endpoint_problem() or f"R2 listing failed: {str(getattr(e, 'code', '') or e)[:90]}"
            self.after(0, lambda: self.l_status.configure(text=msg))
            return
        worlds = group_objects(objects, prefix)
        for wid, w in worlds.items():                  # titles and order from the public novel page data
            try:
                d = http_get_json(f"{url}/api/worlds/{wid}/novel", 8).get("data") or {}
            except Exception:
                d = {}
            w["title"] = d.get("title") or f"world {wid[:8]}"
            w["slug"] = d.get("world_slug") or wid
            meta = {c.get("story_id"): c for c in d.get("chapters") or []}
            for ch in w["chapters"]:
                m = meta.get(ch["id"], {})
                ch["title"] = m.get("title") or f"chapter {ch['id'][:8]}"
                ch["number"] = m.get("chapter_number") or 10_000
                ch["link"] = f"{url}/worlds/{w['slug']}/novel?story={ch['id']}"
                ch["world"] = w
            w["chapters"].sort(key=lambda c: (c["number"], c["modified"]))
        self.after(0, lambda: self._show(site, worlds, len(objects)))

    def _show(self, site, worlds, n_objects):
        if site != self.site.get():
            return
        self.worlds = worlds
        self.tree.delete(*self.tree.get_children())
        for wid, w in sorted(worlds.items(), key=lambda kv: kv[1]["title"]):
            n = sum(len(c["chunks"]) for c in w["chapters"])
            wi = self.tree.insert("", "end", iid=wid, text=f"  {w['title']}", open=True, tags=("world",),
                                  values=(n, "", "", ""))
            for ch in w["chapters"]:
                size = sum(s for _, _, s in ch["chunks"])
                label = ch["title"] + (f"   ·  {ch['variants']} versions" if ch["variants"] > 1 else "")
                self.tree.insert(wi, "end", iid=f"{wid}/{ch['id']}", text=f"  {label}",
                                 values=(len(ch["chunks"]), ch["codec"], f"{size / 1024:.0f} KB",
                                         ch["modified"][:16].replace("T", " ")))
        chapters = sum(len(w["chapters"]) for w in worlds.values())
        self.l_status.configure(text=f"{site}: {len(worlds)} worlds · {chapters} narrated chapters · "
                                     f"{n_objects} objects" if worlds else f"{site}: no narration stored yet")

    def _chapter(self, iid):
        if not iid or "/" not in iid:
            w = self.worlds.get(iid)
            return w["chapters"][0] if w and w["chapters"] else None
        wid, cid = iid.split("/", 1)
        return next((c for c in self.worlds.get(wid, {}).get("chapters", []) if c["id"] == cid), None)

    # playback -----------------------------------------------------------------
    def _play_selected(self):
        sel = self.tree.selection()
        ch = self._chapter(sel[0]) if sel else None
        if ch is None and self.now_chapter is not None and self.player.paused:
            self._pause()
            return
        if ch is not None:
            self._play(ch, 0)

    def _play(self, ch, from_chunk: int):
        self._stop()
        self.play_id += 1
        pid = self.play_id
        self.buffer = queue.Queue(maxsize=self.PREFETCH)
        self.timeline, self.now_chapter = [], ch
        self.l_now.configure(text=f"{ch['title']}  —  loading…")
        threading.Thread(target=self._fetch, args=(pid, ch, from_chunk), daemon=True).start()

    def _fetch(self, pid, ch, from_chunk):
        """Download + decode chunks in order, chapter after chapter, a little ahead of the voice."""
        while ch is not None and pid == self.play_id:
            for i in range(from_chunk, len(ch["chunks"])):
                if pid != self.play_id:
                    return
                try:
                    pcm, rate = decode_audio(self.r2.get(ch["chunks"][i][1]))
                except Exception as e:
                    pcm, rate = None, str(e)[:60]
                while pid == self.play_id:
                    try:
                        self.buffer.put((pid, ch, i, pcm, rate), timeout=0.5)
                        break
                    except queue.Full:
                        continue
            from_chunk = 0
            ch = self._next_of(ch) if self.auto_next.get() else None
        if pid == self.play_id:
            self.buffer.put((pid, None, -1, None, 0))      # end of the playlist

    def _next_of(self, ch, step=1):
        chapters = ch["world"]["chapters"]
        i = chapters.index(ch) + step
        return chapters[i] if 0 <= i < len(chapters) else None

    def _pause(self):
        if self.player.playing:
            self.player.pause(not self.player.paused)

    def _stop(self):
        self.play_id += 1
        self.player.close()
        self.timeline = []
        self._mark_playing(None)

    def _next(self):
        cur = self._current()
        if cur:
            nxt = self._next_of(cur[2])
            if nxt:
                self._play(nxt, 0)

    def _prev(self):
        cur = self._current()
        if cur:
            ch, idx = cur[2], cur[3]
            prv = self._next_of(ch, -1) if idx < 2 else ch      # like a CD player: restart, or the one before
            self._play(prv or ch, 0)

    def _current(self):
        pos = self.player.position()
        for item in self.timeline:
            if item[0] <= pos < item[1]:
                return item
        return self.timeline[-1] if self.timeline else None

    def _seek_click(self, e):
        cur = self._current()
        if cur:
            ch = cur[2]
            w = max(1, self.prog.winfo_width())
            self._play(ch, min(len(ch["chunks"]) - 1, int(len(ch["chunks"]) * e.x / w)))

    def _open_link(self):
        cur = self._current()
        ch = cur[2] if cur else None
        if ch is None:
            sel = self.tree.selection()
            ch = self._chapter(sel[0]) if sel else None
        if ch:
            webbrowser.open(ch["link"])

    def _mark_playing(self, ch):
        for iid in self.tree.tag_has("playing"):
            self.tree.item(iid, tags=())
        if ch is not None:
            iid = f"{ch['world']['id']}/{ch['id']}"
            if self.tree.exists(iid):
                self.tree.item(iid, tags=("playing",))
                self.tree.see(iid)

    def _tick(self):
        try:
            self.player.reap()
            # hand decoded chunks to the device; a rate change waits until the device is drained
            while True:
                try:
                    pid, ch, i, pcm, rate = self.buffer.queue[0]
                except IndexError:
                    break
                if pid != self.play_id:
                    self.buffer.get_nowait()
                    continue
                if ch is None:                              # end of the playlist
                    if not self.player.pending:
                        self.buffer.get_nowait()
                        self.l_now.configure(text="Finished.")
                        self._stop()
                    break
                if pcm is None:                             # a chunk that would not load: skip it
                    self.buffer.get_nowait()
                    self.l_status.configure(text=f"skipped {ch['title']} #{i + 1}: {rate}")
                    continue
                if self.player.h and rate != self.player.rate and self.player.pending:
                    break
                self.buffer.get_nowait()
                if rate != self.player.rate:
                    self.timeline = []
                start = self.player.queued if self.player.h and rate == self.player.rate else 0
                self.player.enqueue(pcm, rate)
                self.timeline.append((start, start + len(pcm), ch, i))
            self._draw_progress()
        finally:
            self.after(100, self._tick)

    def _draw_progress(self):
        c, C = self.prog, self.C
        c.delete("all")
        w, h = max(c.winfo_width(), 100), max(c.winfo_height(), 10)
        cur = self._current() if self.player.h else None
        if not cur:
            return
        pos = self.player.position()
        ch, idx = cur[2], cur[3]
        if ch is not self.now_chapter or not self.tree.tag_has("playing"):
            self.now_chapter = ch
            self._mark_playing(ch)
        n = len(ch["chunks"])
        loaded = {i for _, _, c2, i in self.timeline if c2 is ch}
        frac = (pos - cur[0]) / max(1, cur[1] - cur[0])
        seg_w = w / max(1, n)
        for i in range(n):
            x0 = i * seg_w
            fill = "#3fb950" if i < idx else ("#2d333b" if i not in loaded else "#30475e")
            c.create_rectangle(x0 + 1, 8, x0 + seg_w - 1, h - 8, fill=fill, outline="")
        x = (idx + frac) * seg_w
        c.create_rectangle(idx * seg_w + 1, 8, x, h - 8, fill="#3fb950", outline="")
        c.create_oval(x - 6, h / 2 - 6, x + 6, h / 2 + 6, fill="#f0883e", outline="")
        rate = self.player.rate or 1
        chapter_samples = [(a, b) for a, b, c2, _ in self.timeline if c2 is ch]
        heard = (pos - chapter_samples[0][0]) / rate if chapter_samples else 0
        state = "paused" if self.player.paused else "playing"
        self.l_now.configure(text=f"{ch['world']['title']}  ·  {ch['title']}   —   sentence {idx + 1}/{n}   "
                                  f"·   {fmt_time(heard)}   ({state})")


def group_objects(objects, prefix: str) -> dict:
    """R2 keys ``<prefix><world>/<chapter>/<content_hash>/<i>.<ext>`` → worlds with their chapters.

    A chapter narrated several times (edited text, another narrator) has several
    hashes; the most recently written one is played."""
    tree: dict = collections.defaultdict(lambda: collections.defaultdict(dict))
    for key, size, modified in objects:
        parts = key[len(prefix):].split("/")
        if len(parts) != 4 or parts[0] == "previews":
            continue
        world, chapter, h, name = parts
        stem, _, ext = name.partition(".")
        if not stem.isdigit():
            continue
        v = tree[world][chapter].setdefault(h, {"chunks": [], "modified": ""})
        v["chunks"].append((int(stem), key, size))
        v["modified"] = max(v["modified"], modified)
    worlds = {}
    for world, chapters in tree.items():
        out = []
        for cid, variants in chapters.items():
            best = max(variants.values(), key=lambda v: v["modified"])
            chunks = sorted(best["chunks"])
            out.append({"id": cid, "chunks": chunks, "modified": best["modified"], "variants": len(variants),
                        "codec": chunks[0][1].rsplit(".", 1)[-1] if chunks else ""})
        worlds[world] = {"id": world, "chapters": out}
    return worlds
