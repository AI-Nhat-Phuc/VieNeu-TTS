"""apps/monitor_audio.py: sentence splitting, SRT, R2 signing and key grouping, and
the waveOut player (fed silence, so nothing is heard)."""
import datetime as dt
import sys
import time

import numpy as np
import pytest

pytest.importorskip("tkinter")

from apps.monitor_audio import (WavePlayer, chapter_transcript, group_objects, narrated_sentences, sigv4_headers,
                                slugify, split_sentences, strip_markup, to_srt)


def test_sentences_follow_punctuation_and_paragraphs():
    out = split_sentences("Xin chào. Hôm nay trời đẹp quá! Ừ.\n\nĐoạn hai… Hết.")
    assert out == [("Xin chào.", False), ("Hôm nay trời đẹp quá! Ừ.", True), ("Đoạn hai… Hết.", False)]


def test_long_sentences_are_cut_at_commas():
    s = ", ".join(["một đoạn khá dài"] * 30) + "."
    parts = [p for p, _ in split_sentences(s, max_chars=120)]
    assert all(len(p) <= 120 for p in parts) and " ".join(parts).replace(" ,", ",") == s


def test_srt_and_slug():
    assert to_srt([(0.0, 1.5, "Một"), (1.78, 62.25, "Hai")]) == (
        "1\n00:00:00,000 --> 00:00:01,500\nMột\n\n2\n00:00:01,780 --> 00:01:02,250\nHai\n")
    assert slugify("Đây là VieNeu-TTS, chạy ngay trên máy!") == "day-la-vieneu-tts-chay-ngay"


def test_sigv4_matches_the_aws_example():
    # "GET Object" example from the AWS Signature Version 4 documentation for S3.
    h = sigv4_headers("GET", "examplebucket.s3.amazonaws.com", "/test.txt", {}, {"Range": "bytes=0-9"},
                      "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                      "AKIAIOSFODNN7EXAMPLE", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", "us-east-1",
                      dt.datetime(2013, 5, 24, tzinfo=dt.timezone.utc))
    assert h["authorization"] == (
        "AWS4-HMAC-SHA256 Credential=AKIAIOSFODNN7EXAMPLE/20130524/us-east-1/s3/aws4_request, "
        "SignedHeaders=host;range;x-amz-content-sha256;x-amz-date, "
        "Signature=f0e8bdb87c964420e857bd35b5d6ed310bd44f0170aba48dd91039c6036bdb41")


def test_objects_group_by_world_and_chapter_newest_version_wins():
    objs = [("audio/w1/c1/old/0.wav", 10, "2026-10-01T00:00:00Z"),
            ("audio/w1/c1/new/1.opus", 5, "2026-10-02T00:00:01Z"),
            ("audio/w1/c1/new/0.opus", 5, "2026-10-02T00:00:00Z"),
            ("audio/w1/c2/h/0.opus", 5, "2026-10-02T00:00:00Z"),
            ("audio/previews/narr/abc.opus", 1, "2026-10-02T00:00:00Z"),
            ("audio/w1/c2/h/manifest.json", 1, "2026-10-02T00:00:00Z")]
    w = group_objects(objs, "audio/")
    c1 = next(c for c in w["w1"]["chapters"] if c["id"] == "c1")
    assert [k for _, k, _ in c1["chunks"]] == ["audio/w1/c1/new/0.opus", "audio/w1/c1/new/1.opus"]
    assert c1["variants"] == 2 and c1["codec"] == "opus"
    assert len(w) == 1 and len(w["w1"]["chapters"]) == 2


@pytest.mark.skipif(sys.platform != "win32", reason="waveOut")
def test_player_queues_without_gaps_and_reports_position():
    p = WavePlayer()
    try:
        p.enqueue(np.zeros(4800, np.int16), 48000)      # 2 × 0.1 s of silence
        p.enqueue(np.zeros(4800, np.int16), 48000)
    except OSError:
        pytest.skip("no audio output device")
    assert p.queued == 9600 and p.playing
    deadline = time.time() + 3
    while p.playing and time.time() < deadline:
        time.sleep(0.05)
        p.reap()
    assert not p.playing and p.position() == 9600
    p.close()


def test_reading_time_from_wav_sizes_and_opus_bitrate():
    from apps.monitor_audio import chapter_seconds, reading_time, wav_bytes, wav_layout
    one = wav_bytes(np.zeros(24000, np.int16), 24000)        # 1 s at 24 kHz
    assert wav_layout(one[:256]) == (48000, 44)
    chunks = [(0, "a/0.wav", len(one)), (1, "a/1.wav", len(one)), (2, "a/2.wav", 44 + 24000)]
    assert chapter_seconds(chunks, "wav", one[:256]) == (2.5, False)
    assert chapter_seconds(chunks, "wav", b"not a wav") == (None, False)
    secs, approx = chapter_seconds([(0, "a/0.opus", 5000)], "opus")
    assert approx and secs == pytest.approx(1.0)               # 40 kbit/s default
    assert reading_time(2.5) == "0:02" and reading_time(3725, True) == "≈ 1:02:05" and reading_time(None) == "–"


def test_names_follow_falevon_including_private_and_deleted():
    from apps.monitor_audio import group_objects, resolve_names
    site = "https://x"
    api = {f"{site}/api/worlds/w1/novel": (200, {"title": "Truyện", "world_slug": "truyen",
                                                 "chapters": [{"story_id": "c1", "title": "Chương 1",
                                                               "chapter_number": 1}]}),
           f"{site}/api/stories/c2": (403, None),               # a draft in a public world
           f"{site}/api/stories/c3": (200, {"title": "Ngoại truyện", "order": 9}),
           f"{site}/api/worlds/w2/novel": (404, None), f"{site}/api/worlds/w2": (404, None)}
    calls = []

    def fetch(url):
        calls.append(url)
        return api[url]
    objs = [(f"audio/{w}/{c}/h/0.wav", 100, "2026-10-02T00:00:00Z")
            for w, c in (("w1", "c2"), ("w1", "c1"), ("w1", "c3"), ("w2", "c9"))]
    worlds = group_objects(objs, "audio/")
    for w in worlds.values():
        resolve_names(w, site, fetch)
    w1, w2 = worlds["w1"], worlds["w2"]
    assert (w1["title"], w1["state"]) == ("Truyện", "ok")
    assert [(c["title"], c["state"]) for c in w1["chapters"]] == [
        ("Chương 1", "ok"), ("Ngoại truyện", "ok"), ("private chapter · c2", "private")]
    assert w1["chapters"][0]["link"] == "https://x/worlds/truyen/novel?story=c1"
    assert w2["state"] == "deleted" and w2["title"] == "deleted world · w2"
    assert w2["chapters"][0]["state"] == "deleted" and w2["chapters"][0]["link"] == ""
    assert f"{site}/api/stories/c9" not in calls             # a deleted world's chapters are not looked up


def test_transcript_lines_are_the_narrated_sentences():
    body = "<p>**Mưa** rơi.​‌ Anh nói: \"Ừ.\" Rồi đi.</p><p>[Hết](http://x)!</p>"   # watermark bits too
    assert strip_markup(body, "html") == 'Mưa rơi. Anh nói: "Ừ." Rồi đi.\n\nHết!'
    assert narrated_sentences("  Chương   1 ", strip_markup(body, "html")) == [
        "Chương 1", "Mưa rơi.", 'Anh nói: "Ừ." Rồi đi.', "Hết!"]   # a quote ending in . does not cut
    long = ", ".join(["một đoạn khá dài"] * 20) + "."
    assert all(len(s) <= 160 for s in narrated_sentences("", long))


def test_transcript_says_why_it_is_missing():
    pages = {"https://f/api/stories/a": (200, {"title": "T", "content": "Một. Hai.", "format": "plain"}),
             "https://f/api/stories/b": (200, {"title": "T", "content": "", "secure_content": True}),
             "https://f/api/stories/c": (403, None)}
    fetch = pages.__getitem__
    assert chapter_transcript("https://f", "a", fetch) == (["T", "Một.", "Hai."], "")
    assert chapter_transcript("https://f", "b", fetch)[1].endswith("(early access)")
    assert chapter_transcript("https://f", "c", fetch) == ([], "the chapter is private")
