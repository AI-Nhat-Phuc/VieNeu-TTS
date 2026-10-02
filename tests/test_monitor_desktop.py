"""apps/monitor_desktop.py: the incremental log reader and the log-line parsers
(no window is opened)."""
import pytest

pytest.importorskip("tkinter")

from apps.monitor_desktop import ACCESS_RE, DONE_RE, LogTail


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
