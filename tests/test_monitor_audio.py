"""apps/monitor_audio.py: sentence splitting, SRT, R2 signing and key grouping, and
the waveOut player (fed silence, so nothing is heard)."""
import datetime as dt
import sys
import time

import numpy as np
import pytest

pytest.importorskip("tkinter")

from apps.monitor_audio import (WavePlayer, group_objects, sigv4_headers, slugify, split_sentences,
                                to_srt)


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
