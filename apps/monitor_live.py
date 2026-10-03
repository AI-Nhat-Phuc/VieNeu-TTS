"""
The "Live" tab of the desktop monitor: an animated picture of what the speech
server is doing, drawn on one Tk canvas from ``GET /debug/activity``.

    clients ──► Cloudflare ──► API gate ──► Text ─► Phonemes ─► Backbone ─► Acoustic ─► Codec ─► PCM
       ▲                                                                                     │
       └──────────────────────────────── audio ◄─────────────────────────────────────────────┘

Every request gets its own colour. Its real text flows in as letters, turns into
phoneme symbols, then codec tokens, and comes back out as audio drops sized by
the loudness the server actually produced. Below the pipeline each live request
has a lane (voice, the normalized text chunk it is speaking, a live waveform,
speed) and finished ones slide into the "Recent" list with their outline.
"""
from __future__ import annotations

import math
import random
import time
import tkinter.font as tkfont

PALETTE = ["#58a6ff", "#f778ba", "#3fb950", "#f0883e", "#bc8cff", "#39c5cf", "#e3b341", "#ff7b72"]
PHONEMES = "aəɛeiɨoɔuŋɲʔʂʐχɣθðʃ"
STATE_STYLE = {"queued": ("QUEUED", "#d29922"), "prefill": ("THINKING", "#bc8cff"),
               "streaming": ("SPEAKING", "#3fb950")}
END_STYLE = {"done": "#3fb950", "cancelled": "#d29922", "error": "#f85149", "rejected": "#f85149"}
CHARS_PER_AUDIO_S = 14.0            # rough Vietnamese reading speed, for the reading cursor
TEXT_SPEED, AUDIO_SPEED = 300.0, 340.0   # particle speeds, px/s


# ── small drawing helpers ─────────────────────────────────────────────────────
def _rgb(h):
    return int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)


def mix(a: str, b: str, t: float) -> str:
    t = max(0.0, min(1.0, t))
    (r1, g1, b1), (r2, g2, b2) = _rgb(a), _rgb(b)
    return "#%02x%02x%02x" % (round(r1 + (r2 - r1) * t), round(g1 + (g2 - g1) * t), round(b1 + (b2 - b1) * t))


def rrect(c, x0, y0, x1, y1, r, **kw):
    r = min(r, (x1 - x0) / 2, (y1 - y0) / 2)
    pts = [x0 + r, y0, x1 - r, y0, x1, y0, x1, y0 + r, x1, y1 - r, x1, y1, x1 - r, y1, x0 + r, y1,
           x0, y1, x0, y1 - r, x0, y0 + r, x0, y0]
    return c.create_polygon(pts, smooth=True, **kw)


def bezier(p0, p1, p2, n=12):
    return [((1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t * t * p2[0],
             (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t * t * p2[1]) for t in (i / n for i in range(n + 1))]


class Path:
    """A polyline walked by distance."""

    def __init__(self, pts):
        self.pts = pts
        self.cum = [0.0]
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            self.cum.append(self.cum[-1] + math.hypot(x1 - x0, y1 - y0))
        self.length = self.cum[-1] or 1.0

    def at(self, d):
        d = max(0.0, min(d, self.length))
        lo, hi = 0, len(self.cum) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self.cum[mid] <= d:
                lo = mid
            else:
                hi = mid
        seg = (self.cum[hi] - self.cum[lo]) or 1.0
        t = (d - self.cum[lo]) / seg
        (x0, y0), (x1, y1) = self.pts[lo], self.pts[hi]
        return x0 + (x1 - x0) * t, y0 + (y1 - y0) * t

    def flat(self):
        return [v for p in self.pts for v in p]


def fit(font, text: str, width: float) -> str:
    """``text`` cut (with …) to fit ``width`` pixels; binary search, so long texts stay cheap."""
    if font.measure(text) <= width:
        return text
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if font.measure(text[:mid] + "…") <= width:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo] + "…" if lo else ""


def short_agent(agent: str) -> str:
    a = agent.lower()
    for key, name in (("openai/python", "openai-py"), ("openai/js", "openai-js"), ("httpx", "httpx"),
                      ("aiohttp", "aiohttp"), ("python-requests", "requests"), ("node", "node"), ("undici", "node"),
                      ("curl", "curl"), ("powershell", "powershell"), ("mozilla", "browser"), ("vercel", "vercel")):
        if key in a:
            return name
    return agent.split("/")[0][:14] or "client"


def initials(voice: str) -> str:
    words = [w for w in voice.replace("-", " ").split() if w]
    return ("".join(w[0] for w in words[:2]) or "♪").upper()


def ago(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s ago"
    if seconds < 3600:
        return f"{seconds / 60:.0f} min ago"
    return f"{seconds / 3600:.1f} h ago"


class _Tagged:
    """A canvas whose create_* calls all carry one tag."""

    def __init__(self, canvas, tag):
        self._c, self._tag = canvas, tag

    def __getattr__(self, name):
        f = getattr(self._c, name)
        return (lambda *a, **kw: f(*a, tags=self._tag, **kw)) if name.startswith("create_") else f


# ── the view ──────────────────────────────────────────────────────────────────
class LiveView:
    STAGES = [("Text", "normalize · split ≤256 chars", "Aa"),
              ("Phonemes", "G2P · emotions", "/ə/"),
              ("Backbone", "", "LM"),
              ("Acoustic head", "", "≋"),
              ("Codec", "MOSS decoder · CPU", "∿"),
              ("PCM 48 kHz", "", "▶")]

    def __init__(self, canvas, colors, public_host: str):
        self.c, self.C, self.host = canvas, colors, public_host
        self.f_title = tkfont.Font(family="Segoe UI Semibold", size=10)
        self.f_small = tkfont.Font(family="Segoe UI", size=8)
        self.f_text = tkfont.Font(family="Segoe UI", size=10)
        self.f_big = tkfont.Font(family="Segoe UI Light", size=20)
        self.f_glyph = tkfont.Font(family="Segoe UI Semibold", size=11)
        self.f_mono = tkfont.Font(family="Consolas", size=9)
        self.snap, self.snap_t, self.error = None, 0.0, "waiting for the server…"
        self.streams: dict = {}          # rid -> per-stream animation state
        self.clients: dict = {}          # client ip -> {agent, via, country, seen}
        self.recent_seen: dict = {}      # rid -> time it appeared in Recent
        self.particles: list = []
        self.flash = [0.0] * len(self.STAGES)
        self.gpu = self.cpu = 0.0
        self.gpu_name = "GPU"
        self.t = 0.0
        self.used_colors: dict = {}
        self.ping_t = 0.0
        self._recent_key = None

    # data in (UI thread) ---------------------------------------------------
    def set_system(self, cpu: float, gpu: float | None, gpu_name: str | None):
        self.cpu = cpu
        self.gpu = gpu or 0.0
        if gpu_name:
            self.gpu_name = gpu_name.replace(" Series", "").replace("Radeon ", "")

    def ping(self):
        """A public /health check went through the tunnel."""
        self._spawn("ping", None, PALETTE[0], route="ping")

    def set_error(self, msg: str):
        self.error = msg

    def update(self, snap: dict):
        now = time.time()
        self.snap, self.snap_t, self.error = snap, now, None
        live_ids = set()
        for rec in snap.get("live", []):
            rid = rec["id"]
            live_ids.add(rid)
            s = self.streams.get(rid)
            if s is None or s.get("gone"):
                s = self.streams[rid] = {"rec": rec, "color": self._color(rid), "env": [], "env_n": 0,
                                         "shown": 0.0, "born": self.t, "gone": None, "tacc": 0.0, "tchar": 0,
                                         "seg": -1, "seg_audio0": 0.0, "aspawn": 0}
            self._touch_client(rec)
            if rec.get("segment", -1) != s["seg"]:
                s["seg"], s["seg_audio0"], s["tchar"] = rec.get("segment", -1), rec.get("audio_s", 0.0), 0
                self.flash[0] = self.flash[1] = 1.0
            new = rec.get("env_n", 0) - s["env_n"]
            if new > 0:
                tail = rec.get("env") or []
                s["env"].extend(tail[-new:] if new <= len(tail) else tail)
                s["env_n"] = rec["env_n"]
            s["rec"] = rec
        for rid, s in self.streams.items():
            if rid not in live_ids and not s["gone"]:
                s["gone"] = self.t
                self.used_colors.pop(rid, None)
        first = not self.recent_seen         # what was already there when we connected does not slide in
        for rec in snap.get("recent", []):
            self.recent_seen.setdefault(rec["id"], -10.0 if first else self.t)
            self._touch_client(rec, seen=rec.get("t_end") or now)

    def _color(self, rid):
        taken = set(self.used_colors.values())
        free = [p for p in PALETTE if p not in taken] or PALETTE
        self.used_colors[rid] = free[0]
        return free[0]

    def _touch_client(self, rec, seen=None):
        ip = rec.get("client") or "?"
        c = self.clients.setdefault(ip, {"seen": 0.0})
        c.update(agent=short_agent(rec.get("agent", "")), via=rec.get("via", ""), country=rec.get("country", ""))
        c["seen"] = max(c["seen"], seen or time.time())

    # particles ---------------------------------------------------------------
    def _spawn(self, kind, stream, color, route, glyph="", amp=0.0):
        self.particles.append({"kind": kind, "stream": stream, "color": color, "route": route, "d": 0.0,
                               "glyph": glyph, "amp": amp, "jit": random.uniform(-5, 5)})

    # frame ---------------------------------------------------------------------
    def frame(self, dt: float) -> bool:
        """Draw one frame; True while something moves (the caller then draws more often)."""
        self.t += dt
        c = self.c
        c.delete("!recent")                # the Recent list is redrawn only when it changes
        W, H = max(c.winfo_width(), 600), max(c.winfo_height(), 400)
        lay = self._layout(W, H)
        self._advance(dt, lay)
        self._draw_routes(lay)
        self._draw_particles(lay)          # under the boxes: they go in as one thing, come out as the next
        self._draw_pipeline(lay)
        self._draw_lanes(lay)
        self._draw_recent(lay)
        self.flash = [max(0.0, f - dt * 1.6) for f in self.flash]
        return bool(lay["live"] or self.particles or any(self.flash))

    def _layout(self, W, H):
        cy = 112
        live = [rid for rid, s in self.streams.items() if not s["gone"]]
        # clients: the ones talking now, then the most recent ones (max 4)
        active_ips = {self.streams[r]["rec"].get("client") for r in live}
        ips = sorted(self.clients, key=lambda ip: (ip not in active_ips, -self.clients[ip]["seen"]))[:4]
        cxs = 78
        gap = 62 if len(ips) <= 3 else 50
        client_pos = {ip: (cxs, cy + (i - (len(ips) - 1) / 2) * gap) for i, ip in enumerate(ips)}
        cloud, gate = (232, cy), (378, cy)
        x0, x1 = 470, W - 24
        n = len(self.STAGES)
        step = (x1 - x0) / n
        bw = min(150, step - 16)
        stages = [(x0 + step * (i + 0.5), cy) for i in range(n)]
        routes = {}
        for ip, p in client_pos.items():
            inbound = bezier(p, ((p[0] + cloud[0]) / 2, cy - 10), (cloud[0], cy - 10)) + \
                [(gate[0], cy - 10), (gate[0] + 44, cy)] + [(x, cy) for x, _ in stages[:5]]
            retx = stages[5][0]
            back = [stages[4], stages[5]] + bezier((retx, cy + 34), (retx, cy + 84), (retx - 40, cy + 84), 6) + \
                [(gate[0] + 60, cy + 84)] + bezier((gate[0] + 60, cy + 84), (gate[0], cy + 84), (gate[0], cy + 30), 6) + \
                [(gate[0] - 30, cy + 10), (cloud[0], cy + 10)] + bezier((cloud[0], cy + 10), ((p[0] + cloud[0]) / 2, cy + 10), p)
            routes[("in", ip)] = Path(inbound)
            routes[("queue", ip)] = Path(inbound[:len(inbound) - 5])   # stops at the gate
            routes[("out", ip)] = Path(back)
        routes["ping"] = Path([(cloud[0] + 30, cy - 4), (gate[0] - 30, cy - 4), (gate[0] - 30, cy + 4),
                               (cloud[0] + 30, cy + 4)])
        recent = (self.snap or {}).get("recent", [])
        recent_rows = min(len(recent), max(3, int((H - 470) / 22))) if recent else 0
        recent_h = 34 + recent_rows * 22 if recent else 0
        return {"W": W, "H": H, "cy": cy, "client_pos": client_pos, "cloud": cloud, "gate": gate, "stages": stages,
                "bw": bw, "routes": routes, "lanes_y": 246, "recent_y": H - recent_h - 6, "recent_rows": recent_rows,
                "live": live}

    def _advance(self, dt, lay):
        routes = lay["routes"]
        for rid, s in list(self.streams.items()):
            if s["gone"] is not None:
                if self.t - s["gone"] > 0.8:
                    del self.streams[rid]
                continue
            rec, ip = s["rec"], s["rec"].get("client")
            st = rec.get("state")
            s["tacc"] += dt
            # text flowing in: the real characters of the chunk being spoken
            if ("in", ip) in routes and s["tacc"] > (0.55 if st == "queued" else 0.24):
                s["tacc"] = 0.0
                segs = rec.get("segments") or []
                txt = segs[s["seg"]] if 0 <= s["seg"] < len(segs) else rec.get("text", "")
                txt = txt.replace(" ", "") or "…"
                ch = txt[s["tchar"] % len(txt)]
                s["tchar"] += 1
                self._spawn("text", rid, s["color"], ("queue" if st == "queued" else "in", ip), glyph=ch)
            # audio flowing out, at the rate it is produced, sized by its loudness
            target = len(s["env"])
            s["shown"] += max(0.0, target - s["shown"]) * min(1.0, dt * 2.2) + (dt * 4 if s["shown"] < target else 0)
            s["shown"] = min(s["shown"], target)
            while s["aspawn"] + 6 <= s["shown"]:   # one drop per 0.3 s of audio
                s["aspawn"] += 6
                amp = max(s["env"][s["aspawn"] - 6:s["aspawn"]] or [0])
                if ("out", ip) in routes:
                    self._spawn("audio", rid, s["color"], ("out", ip), amp=amp)
                self.flash[5] = min(1.0, self.flash[5] + 0.25)
        alive = []
        for p in self.particles:
            path = routes.get(p["route"])
            if path is None:
                continue
            p["d"] += dt * (AUDIO_SPEED if p["kind"] == "audio" else 160 if p["kind"] == "ping" else TEXT_SPEED)
            if p["d"] < path.length:
                alive.append(p)
        self.particles = alive[-400:]

    # pipeline ----------------------------------------------------------------
    def _draw_pipeline(self, lay):
        c, C, cy = self.c, self.C, lay["cy"]
        W = lay["W"]
        eng = (self.snap or {}).get("engine", {})
        live = [self.streams[r] for r in lay["live"]]
        speaking = sum(1 for s in live if s["rec"].get("state") == "streaming")
        thinking = sum(1 for s in live if s["rec"].get("state") == "prefill")
        busy = (speaking + thinking) / max(1, eng.get("max_streams", 1))


        # clients
        for ip, (x, y) in lay["client_pos"].items():
            info = self.clients[ip]
            mine = [s for s in live if s["rec"].get("client") == ip]
            r = 17
            if mine:
                pr = r + 5 + 3 * math.sin(self.t * 5)
                c.create_oval(x - pr, y - pr, x + pr, y + pr, outline=mine[0]["color"], width=2)
            c.create_oval(x - r, y - r, x + r, y + r, fill="#2d333b", outline=C["muted"] if not mine else C["fg"])
            for i, s in enumerate(mine[:6]):                       # one coloured arc per stream
                c.create_arc(x - r - 1, y - r - 1, x + r + 1, y + r + 1, start=90 + i * 60, extent=50,
                             style="arc", outline=s["color"], width=3)
            c.create_text(x, y, text="⌂" if info["via"] == "direct" else "◉",
                          fill=C["fg"] if mine else C["muted"], font=self.f_glyph)
            label = info["agent"] + (f" · {info['country']}" if info.get("country") else "")
            c.create_text(x, y + r + 9, text=label, fill=C["fg"] if mine else C["muted"], font=self.f_small)
            c.create_text(x, y + r + 21, text=ip, fill=C["muted"], font=self.f_small)
        if not lay["client_pos"]:
            c.create_text(78, cy, text="no clients yet", fill=C["muted"], font=self.f_small)

        # cloud (the tunnel)
        x, y = lay["cloud"]
        glow = mix("#2d333b", C["accent"], 0.25 + 0.5 * (self.t - self.ping_t < 0.6) + 0.25 * busy)
        for dx, dy, r in ((-26, 6, 18), (-6, -8, 24), (18, 0, 20), (30, 10, 13), (-2, 12, 16)):
            c.create_oval(x + dx - r - 2, y + dy - r - 2, x + dx + r + 2, y + dy + r + 2, fill=glow, outline="")
        for dx, dy, r in ((-26, 6, 18), (-6, -8, 24), (18, 0, 20), (30, 10, 13), (-2, 12, 16)):
            c.create_oval(x + dx - r, y + dy - r, x + dx + r, y + dy + r, fill="#1f2630", outline="")
        c.create_text(x + 2, y + 2, text="Cloudflare", fill=C["fg"], font=self.f_title)
        c.create_text(x, y + 42, text=self.host, fill=C["muted"], font=self.f_small)

        # the API gate with its stream slots
        x, y = lay["gate"]
        ms = eng.get("max_streams", 0) or 1
        w = max(70, ms * 13 + 20)
        rrect(c, x - w / 2, y - 30, x + w / 2, y + 30, 12, fill="#1f2630",
              outline=mix("#30363d", C["ok"], busy), width=2)
        c.create_text(x, y - 16, text="API :8000", fill=C["fg"], font=self.f_title)
        slots = [s for s in live if s["rec"].get("state") in ("prefill", "streaming")]
        for i in range(ms):
            sx = x - (ms - 1) * 13 / 2 + i * 13
            col = slots[i]["color"] if i < len(slots) else "#30363d"
            rr = 4.5 + (1.2 * math.sin(self.t * 8 + i) if i < len(slots) else 0)
            c.create_oval(sx - rr, y + 6 - rr, sx + rr, y + 6 + rr, fill=col, outline="")
        queued = [s for s in live if s["rec"].get("state") == "queued"]
        for i, s in enumerate(queued[:8]):
            qx = x - w / 2 - 12 - i * 11
            c.create_oval(qx - 4, y + 22 - 4, qx + 4, y + 22 + 4, fill=s["color"], outline="")
        c.create_text(x, y + 42, text=f"{len(slots)}/{ms} streams" + (f" · {len(queued)} queued" if queued else ""),
                      fill=C["muted"], font=self.f_small)

        # stages
        gpu_stage = eng.get("backbone_gpu")
        subs = {2: f"{eng.get('backbone', 'llama.cpp')} · " + (f"{self.gpu_name} Vulkan" if gpu_stage else "CPU"),
                3: f"CPU · {eng.get('acoustic', 'int8')} batched" if eng.get("acoustic") != "per-stream" else
                "CPU · per stream", 5: "→ " + ", ".join(sorted({s['rec'].get('format', 'wav') for s in live}) or
                                                          ["wav / pcm"])}
        loads = {2: (self.gpu / 100 if gpu_stage else self.cpu / 100, "GPU" if gpu_stage else "CPU"),
                 4: (self.cpu / 100, "CPU")}
        activity = [self.flash[0], self.flash[1], busy, speaking / ms, speaking / ms, self.flash[5]]
        bw, bh = lay["bw"], 66
        for i, ((name, sub, glyph), (sx, sy)) in enumerate(zip(self.STAGES, lay["stages"])):
            a = min(1.0, activity[i])
            pulse = a * (0.65 + 0.35 * math.sin(self.t * 7 + i))
            accent = [C["accent"], C["cpu"], C["gpu"], "#39c5cf", "#3fb950", "#e3b341"][i]
            if pulse > 0.05:
                g = 5 + 5 * pulse
                rrect(c, sx - bw / 2 - g, sy - bh / 2 - g, sx + bw / 2 + g, sy + bh / 2 + g, 18,
                      fill=mix(C["bg"], accent, 0.18 * pulse), outline="")
            rrect(c, sx - bw / 2, sy - bh / 2, sx + bw / 2, sy + bh / 2, 12, fill="#1f2630",
                  outline=mix("#30363d", accent, 0.3 + 0.7 * pulse), width=2)
            c.create_text(sx - bw / 2 + 12, sy - 16, anchor="w", text=glyph, fill=mix(C["muted"], accent, 0.4 + pulse),
                          font=self.f_glyph)
            c.create_text(sx - bw / 2 + 40, sy - 16, anchor="w", text=name, fill=C["fg"], font=self.f_title)
            sub = fit(self.f_small, subs.get(i, sub), bw - 16)
            c.create_text(sx - bw / 2 + 10, sy + 4, anchor="w", text=sub, fill=C["muted"], font=self.f_small)
            if i in loads:
                v, lab = loads[i]
                bx0, bx1, by = sx - bw / 2 + 10, sx + bw / 2 - 40, sy + 21
                c.create_rectangle(bx0, by - 3, bx1, by + 3, fill="#2d333b", outline="")
                c.create_rectangle(bx0, by - 3, bx0 + (bx1 - bx0) * max(0, min(1, v)), by + 3,
                                   fill=accent, outline="")
                c.create_text(sx + bw / 2 - 8, by, anchor="e", text=f"{lab} {v * 100:.0f}%", fill=C["muted"],
                              font=self.f_small)
            elif i in (0, 1):
                n_seg = sum(len(s["rec"].get("segments") or []) for s in live)
                c.create_text(sx - bw / 2 + 10, sy + 21, anchor="w",
                              text=f"{n_seg} chunks in flight" if n_seg else "idle", fill=C["muted"],
                              font=self.f_small)
            elif i == 5:
                kb = sum(s["rec"].get("audio_s", 0) for s in live)
                c.create_text(sx - bw / 2 + 10, sy + 21, anchor="w", text=f"{kb:.1f} s produced now",
                              fill=C["muted"], font=self.f_small)

        # status / totals, top right
        snap = self.snap or {}
        fresh = time.time() - self.snap_t < 3
        dot = C["ok"] if fresh else C["bad"]
        c.create_oval(W - 18, 10, W - 10, 18, fill=dot, outline="")
        c.create_text(W - 24, 14, anchor="e", fill=C["muted"], font=self.f_small,
                      text=(f"served {snap.get('served', 0)} · {snap.get('audio_s', 0) / 60:.1f} min of audio since "
                            f"start" if fresh else "no live data"))
        c.create_text(470, 14, anchor="w", fill=C["muted"], font=self.f_small,
                      text="how a request becomes audio  —  letters → phonemes → tokens → sound")

    def _draw_routes(self, lay):
        c = self.c
        for key, path in lay["routes"].items():
            if key == "ping":
                continue
            if key[0] == "in":
                c.create_line(*path.flat(), fill="#262c36", width=2, smooth=True)
            elif key[0] == "out":
                c.create_line(*path.flat(), fill="#232a33", width=2, smooth=True, dash=(3, 5))

    # particles -------------------------------------------------------------
    def _draw_particles(self, lay):
        c, routes = self.c, lay["routes"]
        stages = lay["stages"]
        for p in self.particles:
            path = routes[p["route"]]
            x, y = path.at(p["d"])
            if p["kind"] == "ping":
                c.create_oval(x - 2, y - 2, x + 2, y + 2, fill="#6e7681", outline="")
                continue
            if p["kind"] == "audio":
                a = min(1.0, (p["amp"] / 0.18) ** 0.7)
                r = 2 + 3.5 * a
                if a > 0.45:                           # loud bits ripple
                    rr = r + 3 + 2 * math.sin(p["d"] / 6)
                    c.create_oval(x - rr, y - rr, x + rr, y + rr, outline=mix(self.C["bg"], p["color"], 0.5))
                c.create_oval(x - r, y - r, x + r, y + r, fill=p["color"], outline="")
                continue
            y += p["jit"] * (0.0 if p["route"][0] == "queue" else 1.0) * min(1.0, p["d"] / 200)
            if x < stages[1][0]:                       # still letters
                c.create_text(x, y, text=p["glyph"], fill=p["color"], font=self.f_glyph)
            elif x < stages[2][0]:                     # phonemes
                c.create_text(x, y, text=PHONEMES[hash(p["glyph"]) % len(PHONEMES)], fill=p["color"],
                              font=self.f_glyph)
            elif x < stages[3][0]:                     # codec tokens
                c.create_rectangle(x - 3, y - 3, x + 3, y + 3, fill=p["color"], outline="")
            else:                                      # acoustic features: a little wiggle
                h = 3 + 4 * abs(math.sin(p["d"] / 9))
                c.create_line(x - 6, y, x - 3, y - h, x, y + h, x + 3, y - h, x + 6, y, fill=p["color"], width=2)

    # lanes -------------------------------------------------------------------
    def _draw_lanes(self, lay):
        c, C, W = self.c, self.C, lay["W"]
        y0 = lay["lanes_y"]
        y_end = lay["recent_y"] - 8
        c.create_text(16, y0, anchor="w", text="LIVE REQUESTS", fill=C["muted"], font=self.f_small)
        c.create_line(110, y0, W - 16, y0, fill="#262c36")
        streams = sorted(self.streams.items(), key=lambda kv: kv[1]["rec"].get("t_start", 0))
        y = y0 + 12
        lane_h = 84
        if self.error and not self.snap:
            c.create_text(W / 2, (y0 + y_end) / 2, text=self.error, fill=C["warn"], font=self.f_text,
                          width=W - 120, justify="center")
            return
        if not streams:
            mid = (y0 + 12 + y_end) / 2
            pts = []
            for i in range(0, int(W - 160), 6):
                a = 6 * math.sin(i / 40 + self.t * 1.5) * math.sin(i / 230 + self.t * 0.4)
                pts += [80 + i, mid + 26 + a]
            if len(pts) >= 4:
                c.create_line(*pts, fill="#2d333b", width=2, smooth=True)
            last = (self.snap or {}).get("recent") or []
            when = f" · last request {ago(time.time() - (last[0].get('t_end') or 0))}" if last else ""
            c.create_text(W / 2, mid - 6, text="Waiting for someone to ask for a voice…",
                          fill=C["muted"], font=self.f_big)
            c.create_text(W / 2, mid + 52, text=f"idle{when}", fill=C["muted"], font=self.f_small)
            if self.error:
                c.create_text(W / 2, mid + 70, text=self.error, fill=C["warn"], font=self.f_small)
            return
        shown = 0
        for rid, s in streams:
            if y + lane_h > y_end:
                c.create_text(W / 2, y_end - 4, text=f"+{len(streams) - shown} more", fill=C["muted"],
                              font=self.f_small)
                break
            self._draw_lane(s, y, lane_h, W)
            y += lane_h + 6
            shown += 1

    def _draw_lane(self, s, y, h, W):
        c, C = self.c, self.C
        rec, col = s["rec"], s["color"]
        fade = 1.0 if s["gone"] is None else max(0.0, 1 - (self.t - s["gone"]) / 0.8)
        grow = min(1.0, (self.t - s["born"]) / 0.35)
        x0 = 16 + (1 - grow) * -40 + (1 - fade) * 60
        bg = mix(self.C["bg"], "#1f2630", fade)
        rrect(c, x0, y, W - 16, y + h, 12, fill=bg, outline=mix(C["bg"], col, 0.45 * fade), width=1)
        c.create_rectangle(x0 + 2, y + 10, x0 + 6, y + h - 10, fill=mix(C["bg"], col, fade), outline="")
        st = rec.get("state", "")
        if s["gone"] is not None:
            st = "done"
        # avatar: spins while thinking, breathes with the voice while speaking
        ax, ay = x0 + 40, y + h / 2
        level = (s["env"][int(s["shown"]) - 1] if s["shown"] >= 1 and s["env"] else 0.0)
        if st == "prefill" or st == "queued":
            c.create_arc(ax - 25, ay - 25, ax + 25, ay + 25, start=(self.t * 300) % 360, extent=100, style="arc",
                         outline=col, width=3)
        elif st == "streaming":
            rr = 24 + 12 * min(1.0, (level / 0.18) ** 0.7)
            c.create_oval(ax - rr, ay - rr, ax + rr, ay + rr, outline=mix(C["bg"], col, 0.6), width=2)
            rr2 = 24 + 6 * math.sin(self.t * 3)
            c.create_oval(ax - rr2, ay - rr2, ax + rr2, ay + rr2, outline=mix(C["bg"], col, 0.25))
        c.create_oval(ax - 20, ay - 20, ax + 20, ay + 20, fill=mix("#1f2630", col, 0.45 * fade), outline="")
        c.create_text(ax, ay, text=initials(rec.get("voice") or ""), fill=mix(C["bg"], "#ffffff", fade),
                      font=self.f_title)

        # header
        tx = x0 + 74
        label, chip = STATE_STYLE.get(st, ("DONE", C["ok"]))
        cw = self.f_small.measure(label) + 14
        rrect(c, tx, y + 9, tx + cw, y + 25, 8, fill=mix(C["bg"], chip, 0.3 * fade), outline="")
        c.create_text(tx + cw / 2, y + 17, text=label, fill=mix(C["bg"], chip, fade), font=self.f_small)
        who = rec.get("client", "?") + (f" ({rec['country']})" if rec.get("country") else "")
        head = (f"{rec.get('voice') or 'default voice'}   ·   {short_agent(rec.get('agent', ''))} {who}   ·   "
                f"{rec.get('format', '')} {rec.get('rate', 48000) // 1000} kHz   ·   {rec['id']}")
        c.create_text(tx + cw + 10, y + 17, anchor="w", text=head, fill=mix(C["bg"], C["fg"], fade),
                      font=self.f_small)

        # chunk progress blocks + the chunk being spoken, with a reading cursor
        segs = rec.get("segments") or []
        si = rec.get("segment", -1)
        text_w = int(W * 0.52) - (tx - 16)
        bx = tx
        for i in range(min(len(segs), 40)):
            if i < si:
                fill = mix(C["bg"], col, 0.75 * fade)
            elif i == si:
                fill = mix(C["bg"], col, (0.6 + 0.4 * math.sin(self.t * 6)) * fade)
            else:
                fill = "#30363d"
            c.create_rectangle(bx, y + 33, bx + 14, y + 38, fill=fill, outline="")
            bx += 17
        if segs:
            c.create_text(bx + 4, y + 35, anchor="w", text=f"chunk {si + 1}/{len(segs)}", fill=C["muted"],
                          font=self.f_small)
        cur = segs[si] if 0 <= si < len(segs) else rec.get("text", "")
        spoken_n = int((rec.get("audio_s", 0) - s["seg_audio0"]) * CHARS_PER_AUDIO_S) if st == "streaming" else 0
        spoken_n = max(0, min(len(cur), spoken_n))
        self._draw_reading(cur, spoken_n, tx, y + 58, text_w, col, fade)

        # waveform of what the server has produced, scrolling
        wx0, wx1 = int(W * 0.54), W - 190
        wy = y + h / 2
        c.create_line(wx0, wy, wx1, wy, fill="#2d333b")
        pts_n = max(2, (wx1 - wx0) // 3)
        env = s["env"][:int(s["shown"])][-pts_n:]
        if len(env) >= 2:
            top, bot = [], []
            start = wx1 - len(env) * 3
            for i, v in enumerate(env):
                a = (h / 2 - 10) * min(1.0, (v / 0.2) ** 0.7)
                top += [start + i * 3, wy - a]
                bot = [start + i * 3, wy + a] + bot
            c.create_polygon(*top, *bot, fill=mix(C["bg"], col, 0.45 * fade),
                             outline=mix(C["bg"], col, fade), smooth=True)
        elif st != "streaming":
            dots = "·" * (1 + int(self.t * 3) % 3)
            c.create_text((wx0 + wx1) / 2, wy, text=("waiting for a free stream" if st == "queued" else
                                                     "preparing the voice, first audio coming") + dots,
                          fill=C["muted"], font=self.f_small)

        # numbers
        now = time.time()
        el = now - rec.get("t_start", now)
        gen = now - (rec.get("t_slot") or now)
        audio = rec.get("audio_s", 0.0)
        speed = audio / gen if gen > 0.3 and audio else 0.0
        rows = [("elapsed", f"{el:.1f} s"),
                ("first audio", f"{rec['ttfa']} ms" if rec.get("ttfa") is not None else "…"),
                ("audio", f"{audio:.1f} s"),
                ("speed", f"{speed:.2f}× real time" if speed else "…")]
        for i, (k, v) in enumerate(rows):
            yy = y + 14 + i * 17
            c.create_text(W - 176, yy, anchor="w", text=k, fill=C["muted"], font=self.f_small)
            colv = (C["ok"] if speed >= 1 else C["warn"]) if k == "speed" and speed else C["fg"]
            c.create_text(W - 26, yy, anchor="e", text=v, fill=mix(C["bg"], colv, fade), font=self.f_small)

    def _draw_reading(self, text, spoken, x, y, width, col, fade):
        """The chunk text on one line: the part already voiced in the stream colour."""
        c, C = self.c, self.C
        text = " ".join(text.split())
        if not text:
            return
        # keep the reading cursor in view: drop words from the left as it advances
        start = 0
        while start < spoken and self.f_text.measure(text[start:spoken]) > width * 0.55:
            nxt = text.find(" ", start + 1)
            if nxt < 0:
                break
            start = nxt + 1
        said, rest = text[start:spoken], text[spoken:]
        if start:
            said = "…" + said
        w1 = self.f_text.measure(said)
        rest = fit(self.f_text, rest, width - w1)
        c.create_text(x, y, anchor="w", text=said, fill=mix(C["bg"], col, fade), font=self.f_text)
        c.create_text(x + w1, y, anchor="w", text=rest, fill=mix(C["bg"], C["muted"], fade), font=self.f_text)
        if 0 < spoken < len(text):
            blink = 0.5 + 0.5 * math.sin(self.t * 10)
            c.create_line(x + w1 + 1, y - 9, x + w1 + 1, y + 9, fill=mix(C["bg"], col, blink * fade), width=2)

    # recent ------------------------------------------------------------------
    def _draw_recent(self, lay):
        C, W = self.C, lay["W"]
        recent = (self.snap or {}).get("recent", [])[:lay["recent_rows"]]
        sliding = any(self.t - self.recent_seen.get(r["id"], -10.0) < 0.6 for r in recent)
        key = (W, lay["recent_y"], tuple(r["id"] for r in recent))
        if key == self._recent_key and not sliding:
            return
        self._recent_key = key
        self.c.delete("recent")
        c = _Tagged(self.c, "recent")
        if not recent:
            return
        y = lay["recent_y"]
        c.create_text(16, y + 10, anchor="w", text="RECENT", fill=C["muted"], font=self.f_small)
        c.create_line(70, y + 10, W - 16, y + 10, fill="#262c36")
        y += 28
        now = time.time()
        for rec in recent:
            appear = self.recent_seen.get(rec["id"], -10.0)
            k = min(1.0, (self.t - appear) / 0.5)
            ease = 1 - (1 - k) ** 3
            x = 16 + (1 - ease) * 120
            col = END_STYLE.get(rec.get("state"), C["muted"])
            if k < 1:
                c.create_rectangle(16, y - 10, W - 16, y + 10, fill=mix(C["bg"], col, 0.25 * (1 - k)), outline="")
            c.create_oval(x, y - 4, x + 8, y + 4, fill=col, outline="")
            t_end = rec.get("t_end") or now
            cells = [(x + 16, time.strftime("%H:%M:%S", time.localtime(t_end)), C["muted"]),
                     (x + 80, (rec.get("voice") or "default")[:16], C["fg"])]
            for cx, txt, fg in cells:
                c.create_text(cx, y, anchor="w", text=txt, fill=mix(C["bg"], fg, ease), font=self.f_small)
            stats = (f"{rec.get('audio_s', 0):.1f} s audio   ttfa {rec['ttfa']} ms   "
                     f"rtf {(t_end - (rec.get('t_slot') or t_end)) / rec['audio_s']:.2f}"
                     if rec.get("audio_s") and rec.get("ttfa") is not None else (rec.get("error") or rec.get("state")))
            stats += f"   {short_agent(rec.get('agent', ''))} {rec.get('client', '')}"
            sx = W - 120 - self.f_small.measure(stats) - 14
            text = " ".join(rec.get("text", "").split())
            tx0, tw = x + 190, sx - (x + 190) - 16
            text = fit(self.f_small, text, tw)
            c.create_text(tx0, y, anchor="w", text=f"“{text}”" if text else "", fill=mix(C["bg"], C["muted"], ease),
                          font=self.f_small)
            c.create_text(sx, y, anchor="w", text=stats, fill=mix(C["bg"], C["muted"], ease), font=self.f_small)
            env = rec.get("env") or []
            if env:
                ex0, ew = W - 112, 96
                m = max(env) or 1
                pts = []
                for i, v in enumerate(env):
                    pts += [ex0 + i * ew / len(env), y - 8 * v / m]
                for i, v in reversed(list(enumerate(env))):
                    pts += [ex0 + i * ew / len(env), y + 8 * v / m]
                c.create_polygon(*pts, fill=mix(C["bg"], col, 0.35 * ease), outline=mix(C["bg"], col, 0.8 * ease))
            y += 22
