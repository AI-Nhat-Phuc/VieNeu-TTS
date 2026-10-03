"""apps/monitor_desktop.py: the incremental log reader and the log-line parsers
(no window is opened)."""
import sys

import pytest

pytest.importorskip("tkinter")

from apps.monitor_desktop import ACCESS_RE, DONE_RE, GpuStats, LogTail


def test_logtail_returns_each_complete_line_once(tmp_path):
    p = tmp_path / "server.err.log"
    p.write_bytes(b"one\ntw")
    t = LogTail(str(p))
    assert t.read_new() == "one\n"            # "tw" is still being written
    assert t.read_new() == ""
    with open(p, "ab") as f:
        f.write(b"o\nthree\n")
    assert t.read_new() == "two\nthree\n"
    assert t.read_new() == ""


def test_logtail_follows_a_recreated_file(tmp_path):
    p = tmp_path / "w.log"
    p.write_bytes(b"old line that is long\n")
    t = LogTail(str(p))
    assert t.read_new() == "old line that is long\n"
    p.write_bytes(b"new\n")                    # truncated / recreated by a restart
    assert t.read_new() == "new\n"


def test_logtail_missing_file_is_empty(tmp_path):
    assert LogTail(str(tmp_path / "nope.log")).read_new() == ""


def test_parsers_match_real_server_lines():
    done = ("2026-10-02 08:28:35,030 INFO vieneu.api: spk-40ba8e1d done: ttfa=404ms total=1.83s "
            "audio=2.96s rtf=0.62 active=2")
    m = DONE_RE.search(done)
    assert m and m.group(1) == "404ms" and m.group(2) == "0.62"
    m = ACCESS_RE.search('INFO:     127.0.0.1:57083 - "POST /v1/audio/speech HTTP/1.1" 200 OK')
    assert m and m.groups() == ("POST", "/v1/audio/speech", "200")
    assert ACCESS_RE.search('INFO:     115.72.48.131:0 - "GET /health HTTP/1.1" 200 OK').group(2) == "/health"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows performance counters")
def test_gpu_sample_is_none_or_complete():
    g = GpuStats().sample([0])
    assert g is None or (set(g) == {"name", "util", "engine", "used", "total", "proc"}
                         and 0 <= g["util"] <= 100 and g["proc"] == 0)


def _rec(rid, state, env_n, segment=0):
    return {"id": rid, "state": state, "voice": "Mai Anh", "text": "Xin chào các bạn.", "client": "1.2.3.4",
            "via": "cloudflare", "country": "VN", "agent": "python-httpx/0.27", "format": "wav", "rate": 48000,
            "t_start": 0.0, "t_slot": 0.0, "ttfa": 300, "audio_s": env_n / 20, "segments": ["xin chào", "các bạn"],
            "segment": segment, "env": [0.1] * min(env_n, 200), "env_n": env_n, "t_end": None, "error": None}


def test_live_view_follows_a_request_to_recent():
    tk = pytest.importorskip("tkinter")
    from apps.monitor_desktop import COLORS
    from apps.monitor_live import LiveView
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display")
    root.withdraw()
    try:
        c = tk.Canvas(root, width=1300, height=800)
        v = LiveView(c, COLORS, "tts.example.com")
        eng = {"max_streams": 6, "backbone": "llama.cpp", "backbone_gpu": True, "acoustic": "int8"}
        v.update({"live": [_rec("spk-1", "streaming", 30)], "recent": [], "engine": eng})
        v.update({"live": [_rec("spk-1", "streaming", 250, segment=1)], "recent": [], "engine": eng})
        s = v.streams["spk-1"]
        assert len(s["env"]) == 230 and s["seg"] == 1      # 30 known + the 200 newest of the 220 new points
        for _ in range(40):
            assert v.frame(0.05)                           # something is moving
        assert any(p["kind"] == "text" for p in v.particles)
        assert any(p["kind"] == "audio" for p in v.particles)
        done = dict(_rec("spk-1", "done", 250), t_end=1.0, env=[0.1] * 64)
        v.update({"live": [], "recent": [done], "engine": eng})
        for _ in range(30):
            v.frame(0.05)
        assert "spk-1" not in v.streams and c.find_withtag("recent")
    finally:
        root.destroy()
