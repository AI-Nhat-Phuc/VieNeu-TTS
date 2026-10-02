"""
VieNeu-TTS Monitor — a small desktop app (Tkinter, Windows) for a self-hosted
``apps/openai_speech.py`` server: live status (incl. GPU load / VRAM), a load chart, request counts, the
server / tunnel / watchdog logs, and buttons to restart things.

    .venv\\Scripts\\pythonw.exe -m apps.monitor_desktop \\
        --logs E:\\vieneu-logs --public-url https://tts.example.com --task VieNeu-TTS

It reads ``<logs>/server.err.log`` + ``server.out.log`` (app log + access log),
``tunnel.err.log`` and ``watchdog.log`` (the files a watchdog script writes when
it runs the server and ``cloudflared``) and
polls ``<local-url>/health`` and ``<public-url>/health``. Restart buttons kill
the process and rely on that watchdog to start it again; Start/Stop drive the
Windows scheduled task. Nothing here needs admin rights or extra packages.
"""
from __future__ import annotations

import argparse
import collections
import ctypes
import json
import os
import re
import subprocess
import threading
import time
import tkinter as tk
import urllib.request
from ctypes import wintypes
from tkinter import ttk

HISTORY_S = 300          # chart window
POLL_S = 2.0
LOG_TAIL_BYTES = 200_000
NO_WINDOW = 0x08000000   # CREATE_NO_WINDOW for helper processes

COLORS = {"bg": "#16181d", "panel": "#1f232b", "fg": "#e6e8eb", "muted": "#8b93a1",
          "ok": "#3fb950", "warn": "#d29922", "bad": "#f85149", "accent": "#58a6ff", "cpu": "#bc8cff",
          "gpu": "#f0883e"}


# ── system metrics (no psutil) ────────────────────────────────────────────────
class _FT(ctypes.Structure):
    _fields_ = [("lo", wintypes.DWORD), ("hi", wintypes.DWORD)]


class _MEM(ctypes.Structure):
    _fields_ = [("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


class SystemStats:
    def __init__(self):
        self._prev = self._times()

    @staticmethod
    def _times():
        idle, kern, user = _FT(), _FT(), _FT()
        ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kern), ctypes.byref(user))
        v = lambda f: (f.hi << 32) | f.lo
        return v(idle), v(kern) + v(user)          # kernel time includes idle

    def cpu(self) -> float:
        idle, total = self._times()
        d_idle, d_total = idle - self._prev[0], total - self._prev[1]
        self._prev = (idle, total)
        return 100.0 * (1 - d_idle / d_total) if d_total > 0 else 0.0

    @staticmethod
    def ram():
        m = _MEM()
        m.dwLength = ctypes.sizeof(_MEM)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.dwMemoryLoad, (m.ullTotalPhys - m.ullAvailPhys) / 2**30, m.ullTotalPhys / 2**30


class _PdhValue(ctypes.Structure):
    _fields_ = [("CStatus", wintypes.DWORD), ("pad", wintypes.DWORD), ("doubleValue", ctypes.c_double)]


class _PdhItem(ctypes.Structure):
    _fields_ = [("szName", wintypes.LPWSTR), ("FmtValue", _PdhValue)]


class _DxgiDesc(ctypes.Structure):
    _fields_ = [("Description", ctypes.c_wchar * 128), ("VendorId", ctypes.c_uint), ("DeviceId", ctypes.c_uint),
                ("SubSysId", ctypes.c_uint), ("Revision", ctypes.c_uint), ("DedicatedVideoMemory", ctypes.c_size_t),
                ("DedicatedSystemMemory", ctypes.c_size_t), ("SharedSystemMemory", ctypes.c_size_t),
                ("LuidLow", wintypes.DWORD), ("LuidHigh", wintypes.LONG), ("Flags", ctypes.c_uint)]


def _com_call(obj, index, restype, *args):
    """Call vtable slot ``index`` of a COM object pointer."""
    fn = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents[index]
    argtypes = [ctypes.c_void_p if type(a).__name__ == "CArgObject" else type(a) for a in args]  # byref() -> pointer
    return ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)(fn)(obj, *args)


def dxgi_adapters() -> dict:
    """``{"luid_0x…_0x…": (name, dedicated VRAM bytes)}`` for the hardware adapters (DXGI, no admin)."""
    out = {}
    try:
        iid = (ctypes.c_byte * 16).from_buffer_copy(          # IID_IDXGIFactory1 770aae78-f26f-4dba-a829-253c83d1b387
            bytes.fromhex("78ae0a776ff2ba4da829253c83d1b387"))
        factory = ctypes.c_void_p()
        if ctypes.windll.dxgi.CreateDXGIFactory1(ctypes.byref(iid), ctypes.byref(factory)) != 0:
            return out
        i = 0
        while True:
            adapter = ctypes.c_void_p()
            if _com_call(factory, 12, ctypes.c_long, ctypes.c_uint(i), ctypes.byref(adapter)) != 0:  # EnumAdapters1
                break
            d = _DxgiDesc()
            if _com_call(adapter, 10, ctypes.c_long, ctypes.byref(d)) == 0 and not d.Flags & 2:  # GetDesc1, skip software
                out["luid_0x%08X_0x%08X" % (d.LuidHigh & 0xFFFFFFFF, d.LuidLow)] = (d.Description,
                                                                                    d.DedicatedVideoMemory)
            _com_call(adapter, 2, ctypes.c_ulong)                                                  # Release
            i += 1
        _com_call(factory, 2, ctypes.c_ulong)
    except Exception:
        pass
    return out


class GpuStats:
    """GPU load and VRAM from the Windows "GPU Engine" / "GPU … Memory" performance counters
    (what Task Manager shows; works for AMD/Intel/NVIDIA, needs Windows 10 1709+)."""

    COUNTERS = {"engine": r"\GPU Engine(*)\Utilization Percentage",
                "adapter": r"\GPU Adapter Memory(*)\Dedicated Usage",
                "process": r"\GPU Process Memory(*)\Dedicated Usage"}

    def __init__(self):
        self.adapters = dxgi_adapters()
        self.ok, self.pdh, self.query, self.handles = False, None, wintypes.HANDLE(), {}
        try:
            self.pdh = ctypes.windll.pdh
            if self.pdh.PdhOpenQueryW(None, 0, ctypes.byref(self.query)) != 0:
                return
            for key, path in self.COUNTERS.items():
                h = wintypes.HANDLE()
                if self.pdh.PdhAddEnglishCounterW(self.query, path, 0, ctypes.byref(h)) != 0:
                    return
                self.handles[key] = h
            self.pdh.PdhCollectQueryData(self.query)            # utilization is a rate: needs two samples
            self.ok = True
        except Exception:
            pass

    def _values(self, key):
        size, n = wintypes.DWORD(0), wintypes.DWORD(0)
        fmt = 0x200 | 0x8000                                    # PDH_FMT_DOUBLE | PDH_FMT_NOCAP100
        self.pdh.PdhGetFormattedCounterArrayW(self.handles[key], fmt, ctypes.byref(size), ctypes.byref(n), None)
        if not size.value:
            return []
        buf = (ctypes.c_byte * size.value)()
        if self.pdh.PdhGetFormattedCounterArrayW(self.handles[key], fmt, ctypes.byref(size), ctypes.byref(n),
                                                 buf) != 0:
            return []
        items = ctypes.cast(buf, ctypes.POINTER(_PdhItem))
        return [(items[i].szName, items[i].FmtValue.doubleValue) for i in range(n.value)
                if items[i].FmtValue.CStatus in (0, 1)]         # PDH_CSTATUS_VALID_DATA / NEW_DATA

    def sample(self, pids=()) -> dict | None:
        """Busiest adapter: ``{name, util, engine, used, total, proc}`` (bytes; ``proc`` = VRAM of ``pids``)."""
        if not self.ok or self.pdh.PdhCollectQueryData(self.query) != 0:
            return None
        used = dict(self._values("adapter"))                    # luid_…_phys_0 -> bytes
        if not used:
            return None
        engines = collections.defaultdict(float)                # (luid, engine type) -> % summed over processes
        for name, v in self._values("engine"):
            m = re.search(r"(luid_\w+?_phys_\d+).*engtype_(.*)$", name)
            if m:
                engines[(m.group(1), m.group(2))] += v
        inst = max(used, key=lambda k: (k.rsplit("_phys", 1)[0] in self.adapters, used[k]))
        luid = inst.rsplit("_phys", 1)[0]
        util, engine = max(((v, e) for (l, e), v in engines.items() if l == inst), default=(0.0, ""))
        want = {f"pid_{p}_" for p in pids}
        proc = sum(v for name, v in self._values("process")
                   if inst in name and any(name.startswith(w) for w in want))
        name, total = self.adapters.get(luid, ("GPU", 0))
        return {"name": name, "util": min(util, 100.0), "engine": engine, "used": used[inst],
                "total": total, "proc": proc}


# ── helpers ───────────────────────────────────────────────────────────────────
def http_json(url: str, timeout: float = 4.0):
    t = time.perf_counter()
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "vieneu-monitor"}),
                                    timeout=timeout) as r:
            return json.loads(r.read().decode()), (time.perf_counter() - t) * 1000, None
    except Exception as e:  # unreachable, 5xx, tunnel 502 ...
        return None, None, str(getattr(e, "code", "") or e)[:80]


def run_ps(cmd: str, timeout: float = 20) -> str:
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True,
                           timeout=timeout, creationflags=NO_WINDOW)
        return (r.stdout or r.stderr).strip()
    except Exception as e:
        return str(e)


def kill_matching(pattern: str) -> str:
    """Kill processes whose command line matches ``pattern`` (the watchdog restarts them)."""
    return run_ps("Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match '%s' -and "
                  "$_.ProcessId -ne $PID } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force; "
                  "$_.ProcessId }" % pattern)


def server_pids() -> list[int]:
    """PIDs of the running ``apps.openai_speech`` processes (for their VRAM)."""
    out = run_ps("Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | Where-Object { "
                 "$_.CommandLine -match 'apps.openai_speech' } | ForEach-Object { $_.ProcessId }")
    return [int(x) for x in out.split() if x.isdigit()]


def process_alive(image: str) -> bool:
    try:
        out = subprocess.run(["tasklist", "/FI", f"IMAGENAME eq {image}", "/NH"], capture_output=True,
                             text=True, timeout=5, creationflags=NO_WINDOW).stdout
        return image.lower() in out.lower()
    except Exception:
        return False


class LogTail:
    """Incremental reader of a growing (or truncated/recreated) log file."""

    def __init__(self, path: str):
        self.path, self.pos, self.partial = path, 0, b""

    def read_new(self) -> str:
        """Complete new lines since the last call (a line still being written waits)."""
        try:
            size = os.path.getsize(self.path)
        except OSError:
            return ""
        if size < self.pos:              # rotated / recreated
            self.pos, self.partial = 0, b""
        if self.pos == 0 and size > LOG_TAIL_BYTES:
            self.pos = size - LOG_TAIL_BYTES
        if size == self.pos:
            return ""
        with open(self.path, "rb") as f:
            f.seek(self.pos)
            data = f.read()
        self.pos += len(data)            # what was read, not the size seen before (the file grows)
        data, nl = self.partial + data, data.rfind(b"\n")
        if nl < 0:
            self.partial = data
            return ""
        cut = len(self.partial) + nl + 1
        data, self.partial = data[:cut], data[cut:]
        return data.decode("utf-8", errors="replace")


ACCESS_RE = re.compile(r'"(GET|POST|DELETE|PUT) (\S+) HTTP/[\d.]+" (\d{3})')
DONE_RE = re.compile(r" done: ttfa=(\S+) .*?rtf=(\S+)")


# ── UI ────────────────────────────────────────────────────────────────────────
class Monitor(tk.Tk):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.title("VieNeu-TTS Monitor")
        self.geometry("1240x760")
        self.minsize(820, 560)
        self.configure(bg=COLORS["bg"])
        self.sys = SystemStats()
        self.gpu = GpuStats()
        self.hist = collections.deque()          # (t, active, max, cpu, gpu %)
        self.req_done = collections.deque(maxlen=200)     # (time, ttfa_ms, rtf) per finished speech request
        self.req_codes = collections.Counter()            # HTTP status of non-/health requests
        self.state = {}
        self._style()
        self._build()
        j = lambda fn: LogTail(os.path.join(args.logs, fn))
        self.tails = {"Server": [j("server.err.log"), j("server.out.log")],
                      "Tunnel": [j("tunnel.err.log")], "Watchdog": [j("watchdog.log")]}
        threading.Thread(target=self._poll_loop, daemon=True).start()
        self.after(500, self._tick)

    # layout
    def _style(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        c = COLORS
        s.configure(".", background=c["bg"], foreground=c["fg"], fieldbackground=c["panel"], font=("Segoe UI", 10))
        s.configure("Card.TFrame", background=c["panel"])
        s.configure("Card.TLabel", background=c["panel"], foreground=c["muted"], font=("Segoe UI", 9))
        s.configure("Val.TLabel", background=c["panel"], foreground=c["fg"], font=("Segoe UI Semibold", 13))
        s.configure("Sub.TLabel", background=c["panel"], foreground=c["muted"], font=("Segoe UI", 9))
        s.configure("TButton", background=c["panel"], foreground=c["fg"], borderwidth=0, padding=(10, 5))
        s.map("TButton", background=[("active", "#2d333b")])
        s.configure("TNotebook", background=c["bg"], borderwidth=0)
        s.configure("TNotebook.Tab", background=c["panel"], foreground=c["muted"], padding=(14, 5))
        s.map("TNotebook.Tab", background=[("selected", "#2d333b")], foreground=[("selected", c["fg"])])
        s.configure("TEntry", foreground=c["fg"], insertcolor=c["fg"])

    def _card(self, parent, col, title):
        f = ttk.Frame(parent, style="Card.TFrame", padding=(12, 8))
        f.grid(row=0, column=col, sticky="nsew", padx=4)
        ttk.Label(f, text=title, style="Card.TLabel").pack(anchor="w")
        dot_row = tk.Frame(f, bg=COLORS["panel"])
        dot_row.pack(anchor="w", fill="x")
        dot = tk.Canvas(dot_row, width=12, height=12, bg=COLORS["panel"], highlightthickness=0)
        dot.pack(side="left", pady=(4, 0))
        dot.create_oval(2, 2, 11, 11, fill=COLORS["muted"], outline="", tags="d")
        val = ttk.Label(dot_row, text="…", style="Val.TLabel")
        val.pack(side="left", padx=(6, 0))
        sub = ttk.Label(f, text="", style="Sub.TLabel")
        sub.pack(anchor="w")
        return dot, val, sub

    def _build(self):
        top = ttk.Frame(self, padding=(8, 8, 8, 4))
        top.pack(fill="x")
        for i in range(6):
            top.columnconfigure(i, weight=1)
        self.c_server = self._card(top, 0, "Server (local)")
        self.c_public = self._card(top, 1, "Public URL")
        self.c_tunnel = self._card(top, 2, "Cloudflare tunnel")
        self.c_task = self._card(top, 3, "Auto-start task")
        self.c_sys = self._card(top, 4, "CPU / RAM")
        self.c_gpu = self._card(top, 5, "GPU / VRAM")

        mid = ttk.Frame(self, padding=(12, 4))
        mid.pack(fill="x")
        self.chart = tk.Canvas(mid, height=150, bg=COLORS["panel"], highlightthickness=0)
        self.chart.pack(side="left", fill="x", expand=True)
        side = ttk.Frame(mid, style="Card.TFrame", padding=(12, 8))
        side.pack(side="left", fill="y", padx=(8, 0))
        ttk.Label(side, text="Speech requests (in the loaded log)", style="Card.TLabel").pack(anchor="w")
        self.l_reqs = ttk.Label(side, text="0", style="Val.TLabel")
        self.l_reqs.pack(anchor="w")
        self.l_perf = ttk.Label(side, text="", style="Sub.TLabel", justify="left")
        self.l_perf.pack(anchor="w")
        self.l_codes = ttk.Label(side, text="", style="Sub.TLabel", justify="left")
        self.l_codes.pack(anchor="w", pady=(4, 0))

        bar = ttk.Frame(self, padding=(12, 4))
        bar.pack(fill="x")
        for text, fn in (("Restart server", self._restart_server), ("Restart tunnel", self._restart_tunnel),
                         ("Start task", self._start_task), ("Stop all", self._stop_all),
                         ("Open logs folder", lambda: os.startfile(self.args.logs)),
                         ("Open public /health", lambda: os.startfile(self.args.public_url + "/health"))):
            ttk.Button(bar, text=text, command=fn).pack(side="left", padx=(0, 6))
        self.l_action = ttk.Label(bar, text="", foreground=COLORS["muted"])
        self.l_action.pack(side="left", padx=8)

        logs = ttk.Frame(self, padding=(12, 4, 12, 12))
        logs.pack(fill="both", expand=True)
        filt = ttk.Frame(logs)
        filt.pack(fill="x", pady=(0, 4))
        ttk.Label(filt, text="Filter:").pack(side="left")
        self.filter = tk.StringVar()
        ttk.Entry(filt, textvariable=self.filter, width=30).pack(side="left", padx=6)
        self.filter.trace_add("write", lambda *_: self._refilter())
        self.autoscroll = tk.BooleanVar(value=True)
        ttk.Checkbutton(filt, text="Auto-scroll", variable=self.autoscroll).pack(side="left", padx=8)
        self.hide_health = tk.BooleanVar(value=True)
        ttk.Checkbutton(filt, text="Hide /health checks", variable=self.hide_health,
                        command=self._refilter).pack(side="left", padx=8)
        ttk.Button(filt, text="Clear view", command=self._clear_logs).pack(side="right")
        self.nb = ttk.Notebook(logs)
        self.nb.pack(fill="both", expand=True)
        self.texts, self.lines = {}, {}
        for name in ("Server", "Tunnel", "Watchdog"):
            frame = ttk.Frame(self.nb)
            t = tk.Text(frame, bg="#0d1117", fg=COLORS["fg"], insertbackground=COLORS["fg"], wrap="none",
                        font=("Consolas", 9), borderwidth=0, padx=8, pady=6)
            sy = ttk.Scrollbar(frame, orient="vertical", command=t.yview)
            t.configure(yscrollcommand=sy.set)
            sy.pack(side="right", fill="y")
            t.pack(fill="both", expand=True)
            t.tag_configure("err", foreground=COLORS["bad"])
            t.tag_configure("warn", foreground=COLORS["warn"])
            t.tag_configure("ok", foreground=COLORS["ok"])
            self.nb.add(frame, text=name)
            self.texts[name], self.lines[name] = t, collections.deque(maxlen=5000)

    # background polling (network / processes), results go to self.state
    def _poll_loop(self):
        n = 0
        while True:
            st = {}
            st["local"], st["local_ms"], st["local_err"] = http_json(self.args.local_url + "/health", 3)
            if n % 3 == 0 or "public" not in self.state:
                st["public"], st["public_ms"], st["public_err"] = http_json(self.args.public_url + "/health", 8)
            else:
                st.update({k: self.state.get(k) for k in ("public", "public_ms", "public_err")})
            if n % 5 == 0 or "task" not in self.state:
                st["task"] = run_ps(f"(Get-ScheduledTask -TaskName '{self.args.task}' -ErrorAction "
                                    f"SilentlyContinue).State") or "not found"
                st["tunnel_proc"] = process_alive("cloudflared.exe")
                st["server_pids"] = server_pids()
            else:
                st.update({k: self.state.get(k) for k in ("task", "tunnel_proc", "server_pids")})
            self.state = st
            n += 1
            time.sleep(POLL_S)

    # UI tick: cards, chart, logs
    def _set(self, card, level, value, sub=""):
        dot, val, lab = card
        dot.itemconfigure("d", fill=COLORS[level])
        val.configure(text=value)
        lab.configure(text=sub)

    def _tick(self):
        st, now = self.state, time.time()
        h = st.get("local")
        if h:
            lvl = "ok" if h.get("status") == "ok" else "bad"
            self._set(self.c_server, lvl, f"{h.get('active', 0)}/{h.get('max_streams', '?')} streams",
                      f"waiting {h.get('waiting', 0)} · {h.get('backend', '')} · {st.get('local_ms', 0):.0f} ms")
        elif st:
            self._set(self.c_server, "bad", "down", st.get("local_err") or "")
        p = st.get("public")
        host = self.args.public_url.split("//")[-1]
        if p:
            self._set(self.c_public, "ok", "reachable", f"{host} · {st.get('public_ms', 0):.0f} ms")
        elif st:
            self._set(self.c_public, "bad", "unreachable", f"{host} · {st.get('public_err') or ''}")
        if st:
            alive = st.get("tunnel_proc")
            self._set(self.c_tunnel, "ok" if alive and p else ("warn" if alive else "bad"),
                      "running" if alive else "stopped", "cloudflared.exe")
            task = st.get("task") or "?"
            self._set(self.c_task, "ok" if task == "Running" else ("warn" if task == "Ready" else "bad"),
                      task, self.args.task)
        cpu = self.sys.cpu()
        load, used, total = self.sys.ram()
        self._set(self.c_sys, "ok" if cpu < 85 else "warn", f"{cpu:.0f}% · {load}%", f"RAM {used:.1f}/{total:.1f} GB")

        g = self.gpu.sample(st.get("server_pids") or ())
        if g:
            gb = lambda b: b / 2**30
            vram = f"{gb(g['used']):.1f}/{gb(g['total']):.1f} GB" if g["total"] else f"{gb(g['used']):.1f} GB"
            hot = g["total"] and g["used"] > 0.9 * g["total"]
            self._set(self.c_gpu, "warn" if g["util"] >= 90 or hot else "ok", f"{g['util']:.0f}% · {vram}",
                      f"{g['name'].replace(' Series', '')} · server {gb(g['proc']):.2f} GB · {g['engine'] or 'idle'}")
        else:
            self._set(self.c_gpu, "muted", "n/a", "no GPU performance counters")
        self.hist.append((now, h.get("active", 0) if h else 0, h.get("max_streams", 0) if h else 0, cpu,
                          g["util"] if g else None))
        while self.hist and now - self.hist[0][0] > HISTORY_S:
            self.hist.popleft()
        self._draw_chart(now)
        self._read_logs()
        done = list(self.req_done)
        self.l_reqs.configure(text=f"{len(done)}  ({sum(1 for t, _, _ in done if now - t < 60)} in last 60 s)")
        last = done[-20:]
        ttfa = [x for _, x, _ in last if x is not None]
        rtf = [r for _, _, r in last if r is not None]
        self.l_perf.configure(text=(f"last {len(last)}: TTFA avg {sum(ttfa) / len(ttfa):.0f} ms\n"
                                    f"RTF avg {sum(rtf) / len(rtf):.2f} (<1 = faster than real time)")
                              if ttfa and rtf else "no requests yet")
        bad = sum(n for c, n in self.req_codes.items() if not c.startswith("2"))
        self.l_codes.configure(text=("HTTP " + "  ".join(f"{c}×{n}" for c, n in sorted(self.req_codes.items()))
                                     + (f"   ⚠ {bad} errors" if bad else "")) if self.req_codes else "")
        self.after(int(POLL_S * 1000), self._tick)

    def _draw_chart(self, now):
        c = self.chart
        c.delete("all")
        w, hgt = max(c.winfo_width(), 200), max(c.winfo_height(), 100)
        pad_l, pad_b, pad_t = 34, 18, 10
        ph = hgt - pad_b - pad_t
        cap = max([r[2] for r in self.hist] + [1])
        for i in range(5):
            y = pad_t + ph * i / 4
            c.create_line(pad_l, y, w - 8, y, fill="#2d333b")
        c.create_text(4, pad_t, anchor="nw", text=str(cap), fill=COLORS["accent"], font=("Segoe UI", 8))
        c.create_text(4, pad_t + ph - 10, anchor="nw", text="0", fill=COLORS["muted"], font=("Segoe UI", 8))
        c.create_text(w - 10, hgt - 4, anchor="se", text="streams (blue) · CPU % (purple) · GPU % (orange) · last 5 min",
                      fill=COLORS["muted"], font=("Segoe UI", 8))
        if len(self.hist) < 2:
            return
        x = lambda t: pad_l + (w - 8 - pad_l) * (1 - (now - t) / HISTORY_S)
        pts_a, pts_c, pts_g = [], [], []
        for t, a, m, cpu, gpu in self.hist:
            pts_a += [x(t), pad_t + ph * (1 - a / cap)]
            pts_c += [x(t), pad_t + ph * (1 - cpu / 100)]
            if gpu is not None:
                pts_g += [x(t), pad_t + ph * (1 - gpu / 100)]
        c.create_line(*pts_c, fill=COLORS["cpu"], width=1.5, smooth=True)
        if len(pts_g) >= 4:
            c.create_line(*pts_g, fill=COLORS["gpu"], width=1.5, smooth=True)
        c.create_line(*pts_a, fill=COLORS["accent"], width=2)

    def _visible(self, line: str, filt: str) -> bool:
        if self.hide_health.get() and '/health ' in line:
            return False
        return not filt or filt in line.lower()

    def _read_logs(self):
        filt = self.filter.get().lower()
        for name, tails in self.tails.items():
            data = "".join(t.read_new() for t in tails)
            if not data:
                continue
            t = self.texts[name]
            for line in data.splitlines():
                if name == "Server":
                    self._account(line)
                self.lines[name].append(line)
                if self._visible(line, filt):
                    t.insert("end", line + "\n", self._tag(line))
            if int(float(t.index("end-1c").split(".")[0])) > 6000:
                t.delete("1.0", "1000.0")
            if self.autoscroll.get():
                t.see("end")

    def _account(self, line: str) -> None:
        m = ACCESS_RE.search(line)
        if m and not m.group(2).startswith("/health"):
            self.req_codes[m.group(3)] += 1
        m = DONE_RE.search(line)
        if m:
            num = lambda v: None if v in (None, "-", "None") else float(v.rstrip("ms"))
            self.req_done.append((time.time(), num(m.group(1)), num(m.group(2))))

    @staticmethod
    def _tag(line: str) -> str:
        l = line.lower()
        if "error" in l or "traceback" in l or " err " in l or "exited" in l or '" 5' in line:
            return "err"
        if "warn" in l or '" 4' in line:                     # warnings, 4xx responses
            return "warn"
        if "✅" in line or "registered tunnel connection" in l or "ready in" in l:
            return "ok"
        return ""

    def _refilter(self):
        filt = self.filter.get().lower()
        for name, t in self.texts.items():
            t.delete("1.0", "end")
            for line in self.lines[name]:
                if not filt or filt in line.lower():
                    t.insert("end", line + "\n", self._tag(line))
            t.see("end")

    def _clear_logs(self):
        for name, t in self.texts.items():
            t.delete("1.0", "end")
            self.lines[name].clear()

    # actions (in a thread so the UI never blocks)
    def _act(self, label, fn):
        self.l_action.configure(text=f"{label}…")

        def run():
            out = fn()
            self.after(0, lambda: self.l_action.configure(text=f"{label}: done {out or ''}"[:120]))
        threading.Thread(target=run, daemon=True).start()

    def _restart_server(self):
        self._act("Restart server", lambda: "pid " + (kill_matching("apps.openai_speech") or "none").replace("\n", ","))

    def _restart_tunnel(self):
        self._act("Restart tunnel", lambda: "pid " + (kill_matching("cloudflared.exe.*tunnel") or "none").replace("\n", ","))

    def _start_task(self):
        self._act("Start task", lambda: run_ps(f"Start-ScheduledTask -TaskName '{self.args.task}'"))

    def _stop_all(self):
        def stop():
            run_ps(f"Stop-ScheduledTask -TaskName '{self.args.task}'")
            kill_matching("vieneu-autostart|serve_public.ps1|apps.openai_speech|cloudflared.exe.*tunnel")
            return "(server + tunnel stopped; Start task to bring them back)"
        self._act("Stop all", stop)


def main():
    ap = argparse.ArgumentParser(description="VieNeu-TTS desktop monitor")
    ap.add_argument("--logs", default=os.environ.get("VIENEU_LOG_DIR", "logs"), help="directory of the log files")
    ap.add_argument("--local-url", default=os.environ.get("VIENEU_LOCAL_URL", "http://127.0.0.1:8000"))
    ap.add_argument("--public-url", default=os.environ.get("VIENEU_PUBLIC_URL", "http://127.0.0.1:8000"))
    ap.add_argument("--task", default=os.environ.get("VIENEU_TASK_NAME", "VieNeu-TTS"), help="scheduled task name")
    args = ap.parse_args()
    args.public_url = args.public_url.rstrip("/")
    args.local_url = args.local_url.rstrip("/")
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)   # crisp text on HiDPI screens
    except Exception:
        pass
    Monitor(args).mainloop()


if __name__ == "__main__":
    main()
