"""apps/openai_speech.py with a stand-in model: request limits, voice enrolment,
stream slots, health, auth and the listening address. ``Engine.__init__`` runs
for real."""
import gc
import logging
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from apps import openai_speech as api


class FakeTTS:
    backend = "onnx"

    def __init__(self):
        self._preset_voices = {"Hải Đăng": {}, "Mai Anh": {}}
        self._voice_aliases = {"Minh Quân": "Hải Đăng"}
        self.stream_calls = []
        self.texts = []
        self.enrolled = []

    def resolve_voice_name(self, name):
        if name in self._preset_voices:
            return name
        target = self._voice_aliases.get(name)
        return target if target in self._preset_voices else None

    def infer_stream(self, text, **kw):
        self.stream_calls.append(kw)
        self.texts.append(text)
        yield np.zeros(4800, dtype=np.float32)

    def _apply_watermark(self, wav):
        return wav

    def add_voice(self, name, path, **kw):
        self.enrolled.append(name)
        self._preset_voices[name] = {}


@pytest.fixture
def eng(monkeypatch):
    for var in ("VIENEU_API_KEY", "VIENEU_MAX_STREAMS", "VIENEU_QUEUE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(api, "Vieneu", lambda **kw: FakeTTS())
    e = api.Engine()
    monkeypatch.setattr(api, "ENGINE", e)
    return e


@pytest.fixture
def client(eng):
    with TestClient(api.app) as c:
        yield c


def _speech(client, **body):
    return client.post("/v1/audio/speech", json={"input": "Xin chào.", **body})


def test_valid_request_streams_wav(client, eng):
    r = _speech(client, voice="Mai Anh", temperature=0.5, top_k=50)
    assert r.status_code == 200
    assert r.content[:4] == b"RIFF" and len(r.content) == 44 + 4800 * 2
    assert eng.tts.stream_calls[-1]["temperature"] == 0.5
    assert eng.tts.stream_calls[-1]["top_k"] == 50
    gc.collect()                               # the drop-time release must be a no-op now
    assert eng.active == 0                     # the stream slot was given back, once


def test_unread_response_still_frees_its_slot(eng):
    # The client left while its request was queued: the body is never iterated,
    # so the stream's own finally never runs. On CPU there is a single slot, and
    # losing it meant 429 for every later request.
    resp = api.speech(api.SpeechRequest(input="Xin chào."))
    assert eng.active == 1
    del resp
    gc.collect()
    assert eng.active == 0
    assert eng._gate.acquire(blocking=False)
    eng._gate.release()


@pytest.mark.parametrize("field, value", [
    ("top_k", 10 ** 30),          # overflowed the GPU scheduler's int64 slot and killed it
    ("top_k", 0),
    ("temperature", 5.0),
    ("top_p", 0.0),
    ("repetition_penalty", 0.0),
    ("max_chars", 0),             # ZeroDivisionError in the chunker, after the 200 was sent
])
def test_out_of_range_sampling_is_rejected_up_front(client, eng, field, value):
    r = _speech(client, **{field: value})
    assert r.status_code == 400
    assert r.json()["error"]["message"].startswith(f"{field}:")   # OpenAI's error shape
    assert eng.tts.stream_calls == [{"apply_watermark": False}]   # only Engine's warm-up ran


def test_nan_temperature_is_rejected(client, eng):
    # json.loads accepts NaN; it used to reach torch.multinomial. Echoing it back
    # in the error body is what turned FastAPI's default 422 into a 500.
    r = client.post("/v1/audio/speech", content=b'{"input": "x", "temperature": NaN}',
                    headers={"Content-Type": "application/json"})
    assert r.status_code == 400 and "temperature" in r.json()["error"]["message"]


@pytest.mark.parametrize("name", ["Hải Đăng", "Minh Quân"])   # the default voice, an alias of it
def test_builtin_voice_cannot_be_replaced(client, eng, name):
    r = client.post("/v1/voices", data={"name": name}, files={"file": ("a.wav", b"RIFF", "audio/wav")})
    assert r.status_code == 409
    assert eng.tts.enrolled == []


def test_new_voice_is_enrolled(client, eng):
    r = client.post("/v1/voices", data={"name": "Giọng của tôi"},
                    files={"file": ("me.wav", b"RIFF", "audio/wav")})
    assert r.status_code == 200
    assert eng.tts.enrolled == ["Giọng của tôi"]


def test_dead_scheduler_fails_health_and_new_requests(client, eng):
    assert client.get("/health").status_code == 200
    eng.sched = SimpleNamespace(error=RuntimeError("worker died"))
    r = client.get("/health")
    assert r.status_code == 503 and r.json()["status"] == "error"
    assert _speech(client).status_code == 503
    assert eng.active == 0 and eng.waiting == 0


def test_api_key_is_enforced_when_set(client, monkeypatch):
    monkeypatch.setenv("VIENEU_API_KEY", "s3cret")
    assert client.get("/v1/voices").status_code == 401
    assert client.get("/v1/voices", headers={"Authorization": "Bearer nope"}).status_code == 401
    assert client.get("/v1/voices", headers={"Authorization": "Bearer s3cret"}).status_code == 200


def test_listens_on_loopback_by_default(monkeypatch):
    seen = {}
    monkeypatch.setattr(api.uvicorn, "run", lambda app, **kw: seen.update(kw))
    monkeypatch.delenv("HOST", raising=False)
    api.main()
    assert seen["host"] == "127.0.0.1"


def test_open_public_bind_is_logged(monkeypatch, caplog):
    monkeypatch.setattr(api.uvicorn, "run", lambda app, **kw: None)
    monkeypatch.setenv("HOST", "0.0.0.0")
    monkeypatch.delenv("VIENEU_API_KEY", raising=False)
    with caplog.at_level(logging.WARNING, logger="vieneu.api"):
        api.main()
    assert "without VIENEU_API_KEY" in caplog.text


def test_activity_follows_a_request_from_start_to_done(client, eng, monkeypatch):
    monkeypatch.setattr(api, "LOOPBACK", ("testclient",))     # TestClient's peer address
    assert _speech(client, voice="Mai Anh").status_code == 200
    r = client.get("/debug/activity")
    assert r.status_code == 200
    body = r.json()
    assert body["live"] == [] and body["engine"]["max_streams"] == eng.max_streams
    done = body["recent"][0]
    assert done["state"] == "done" and done["voice"] == "Mai Anh" and done["text"] == "Xin chào."
    assert done["audio_s"] == pytest.approx(0.1) and done["ttfa"] is not None and len(done["env"]) == 64


def test_activity_records_the_callers_own_source(client, eng, monkeypatch):
    monkeypatch.setattr(api, "LOOPBACK", ("testclient",))
    r = client.post("/v1/audio/speech", json={"input": "Xin chào."},
                    headers={"X-VieNeu-Source": "story-services/drain" + "x" * 60})
    assert r.status_code == 200
    assert _speech(client).status_code == 200
    recent = client.get("/debug/activity").json()["recent"]
    assert recent[0]["source"] == ""                     # no header: nothing to show
    assert recent[1]["source"] == ("story-services/drain" + "x" * 60)[:40]   # unprintable dropped, capped


def test_activity_is_not_served_through_a_tunnel(client, monkeypatch):
    monkeypatch.setattr(api, "LOOPBACK", ("testclient",))
    # cloudflared connects from loopback too; its headers give it away.
    assert client.get("/debug/activity", headers={"CF-Connecting-IP": "1.2.3.4"}).status_code == 404
    assert client.get("/debug/activity", headers={"X-Forwarded-For": "1.2.3.4"}).status_code == 404


def test_health_reports_the_voice_revision(client):
    from vieneu_utils.vi_prosody import VOICE_REVISION
    assert client.get("/health").json()["voice_revision"] == VOICE_REVISION


@pytest.fixture
def exits(monkeypatch):
    """Records the scheduled process exit instead of running it."""
    seen = []

    class Timer:
        def __init__(self, delay, fn, args=()):
            seen.append((fn, args))

        def start(self):
            pass

    monkeypatch.setattr(api.threading, "Timer", Timer)
    return seen


def test_backbone_crash_returns_503_and_restarts_the_process(client, eng, exits):
    lite_bb = SimpleNamespace(fault=None)
    eng._lite_runner = SimpleNamespace(bb=lite_bb)

    def crash(text, **kw):
        lite_bb.fault = RuntimeError("llama_decode crashed in native code")
        raise lite_bb.fault
        yield

    eng.tts.infer_stream = crash
    r = _speech(client)
    assert r.status_code == 503                # not an empty 200
    assert client.get("/health").status_code == 503
    assert exits == [(api.os._exit, (70,))]    # once, however many requests saw it
    assert _speech(client).status_code == 503
    assert len(exits) == 1
    assert eng.active == 0


def test_exit_on_fault_can_be_disabled(client, eng, exits, monkeypatch):
    monkeypatch.setenv("VIENEU_EXIT_ON_FAULT", "0")
    eng.sched = SimpleNamespace(error=RuntimeError("worker died"))
    eng.exit_if_broken()
    assert exits == []


def test_failure_before_any_audio_is_a_500(client, eng, exits):
    def bad(text, **kw):
        raise ValueError("bad voice file")
        yield

    eng.tts.infer_stream = bad
    r = _speech(client)
    assert r.status_code == 500 and "ValueError" in r.json()["error"]["message"]
    assert exits == [] and eng.active == 0


def test_keep_warm_speaks_only_when_no_stream_runs(eng):
    calls = len(eng.tts.stream_calls)
    eng.acquire()                               # a listener is being served
    assert eng.keep_warm() is False
    assert len(eng.tts.stream_calls) == calls
    eng.release()
    before = eng.last_used
    assert eng.keep_warm() is True
    assert len(eng.tts.stream_calls) == calls + 1
    assert eng.last_used >= before
    assert eng.active == 0                      # not counted as a request
    assert eng._gate.acquire(blocking=False)    # and its slot was given back
    eng._gate.release()


def test_keep_warm_and_pin_are_off_by_default(eng, monkeypatch):
    for var in ("VIENEU_KEEP_WARM_SECONDS", "VIENEU_PIN_MB"):
        monkeypatch.delenv(var, raising=False)
    started = []
    monkeypatch.setattr(api.threading, "Thread", lambda **kw: started.append(kw) or SimpleNamespace(start=lambda: None))
    api._startup()
    assert started == []
    monkeypatch.setenv("VIENEU_KEEP_WARM_SECONDS", "180")
    api._startup()
    assert [kw["name"] for kw in started] == ["keep-warm"]


def test_no_wait_request_is_refused_at_once_when_every_slot_is_taken(client, eng):
    import time as _time

    for _ in range(eng.max_streams):
        eng.acquire()                           # every slot is taken
    t = _time.monotonic()
    r = client.post("/v1/audio/speech", json={"input": "Xin chào."}, headers={"X-VieNeu-No-Wait": "1"})
    assert r.status_code == 429
    assert _time.monotonic() - t < 1            # not the 10 s queue
    for _ in range(eng.max_streams):
        eng.release()
    assert _speech(client).status_code == 200   # a normal request still gets a slot


def test_lending_keeps_one_slot_free_and_stops_for_a_waiter(monkeypatch, eng):
    monkeypatch.setattr(eng, "max_streams", 4)
    monkeypatch.setattr(eng, "run_ahead", 3)
    monkeypatch.setattr(eng, "_gate", api.threading.BoundedSemaphore(4))
    eng.acquire()                                  # the request's own slot
    lender = api._Lender(eng)
    assert lender.borrow() and lender.borrow()
    assert not lender.borrow()                     # the last free slot stays for the next request
    assert eng.lent == 2 and api.health()["lent"] == 2
    lender.give_back()
    eng.waiting = 1                                # someone is queued: lend nothing, ask back
    assert lender.wanted() and not lender.borrow()
    eng.waiting = 0
    lender.give_back()
    eng.release()
    assert eng.lent == 0 and eng.active == 0
    for _ in range(4):
        assert eng._gate.acquire(blocking=False)   # every slot came back


# ── merging identical requests ───────────────────────────────────────────────

def _req(headers=None):
    from starlette.requests import Request
    raw = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    return Request({"type": "http", "headers": raw, "client": ("testclient", 1)})


def _body(resp) -> bytes:
    import asyncio

    async def read():
        return b"".join([c async for c in resp.body_iterator])
    return asyncio.run(read())


@pytest.fixture
def noisy(eng):
    """infer_stream yields three random chunks: two separate takes never match."""
    def stream(text, **kw):
        eng.tts.stream_calls.append(kw)
        for _ in range(3):
            yield np.random.uniform(-0.5, 0.5, 4800).astype(np.float32)
    eng.tts.infer_stream = stream
    eng.tts.stream_calls.clear()
    return eng


def test_same_speech_at_the_same_time_is_generated_once(noisy):
    a = api.speech(api.SpeechRequest(input="Xin chào.", voice="Mai Anh"), _req())
    b = api.speech(api.SpeechRequest(input="Xin chào.", voice="Mai Anh"), _req())
    assert noisy.active == 1                       # B took no slot of its own
    assert b.headers["X-VieNeu-Merged-Into"] == a.headers["X-Request-Id"]
    body_a, body_b = _body(a), _body(b)
    assert body_a == body_b and len(body_a) == 44 + 3 * 4800 * 2
    assert len(noisy.tts.stream_calls) == 1
    gc.collect()
    assert noisy.active == 0 and api.MERGES.live == {}


def test_alias_merges_with_its_voice(noisy):
    a = api.speech(api.SpeechRequest(input="Xin chào.", voice="Hải Đăng"), _req())
    b = api.speech(api.SpeechRequest(input="Xin chào.", voice="Minh Quân"), _req())
    assert "X-VieNeu-Merged-Into" in b.headers
    assert _body(a) == _body(b)


@pytest.mark.parametrize("other", [{"input": "Tạm biệt."}, {"temperature": 0.5}, {"sample_rate": 24000}])
def test_different_speech_is_not_merged(noisy, other):
    a = api.speech(api.SpeechRequest(input="Xin chào."), _req())
    with pytest.raises(api.HTTPException) as e:    # the single CPU slot is A's
        api.speech(api.SpeechRequest(**{"input": "Xin chào.", **other}), _req({"X-VieNeu-No-Wait": "1"}))
    assert e.value.status_code == 429
    _body(a)


def test_no_merge_header_asks_for_a_fresh_take(noisy):
    a = api.speech(api.SpeechRequest(input="Xin chào."), _req())
    with pytest.raises(api.HTTPException):
        api.speech(api.SpeechRequest(input="Xin chào."),
                   _req({"X-VieNeu-No-Wait": "1", "X-VieNeu-No-Merge": "1"}))
    _body(a)


def test_merged_request_keeps_its_audio_when_the_first_caller_leaves(noisy):
    a = api.speech(api.SpeechRequest(input="Xin chào."), _req())
    b = api.speech(api.SpeechRequest(input="Xin chào."), _req())
    del a
    gc.collect()
    assert noisy.active == 1                       # still generating for B
    assert len(_body(b)) == 44 + 3 * 4800 * 2
    gc.collect()
    assert noisy.active == 0


def test_generation_stops_when_every_caller_leaves(noisy):
    a = api.speech(api.SpeechRequest(input="Xin chào."), _req())
    b = api.speech(api.SpeechRequest(input="Xin chào."), _req())
    del a, b
    gc.collect()
    assert noisy.active == 0 and api.MERGES.live == {}
    assert noisy._gate.acquire(blocking=False)
    noisy._gate.release()


def test_late_request_past_the_window_generates_its_own(noisy, monkeypatch):
    monkeypatch.setenv("VIENEU_MERGE_WINDOW", "0.05")   # less than one chunk
    a = api.speech(api.SpeechRequest(input="Xin chào."), _req())
    with pytest.raises(api.HTTPException):         # not merged: queues for the slot A holds
        api.speech(api.SpeechRequest(input="Xin chào."), _req({"X-VieNeu-No-Wait": "1"}))
    _body(a)


def test_merged_request_is_shown_on_the_monitor(noisy, monkeypatch):
    a = api.speech(api.SpeechRequest(input="Xin chào."), _req())
    b = api.speech(api.SpeechRequest(input="Xin chào."), _req())
    live = {r["id"]: r for r in api.ACTIVITY.snapshot()["live"]}
    rb = live[b.headers["X-Request-Id"]]
    assert rb["state"] == "merged" and rb["merged"] == a.headers["X-Request-Id"]
    _body(a), _body(b)
    done = api.ACTIVITY.snapshot()["recent"]
    assert {r["state"] for r in done[:2]} == {"done"}
    assert all(r["audio_s"] == pytest.approx(0.3) for r in done[:2])


def _in_threads(*calls):
    import threading
    out = [None] * len(calls)

    def run(i, fn):
        try:
            out[i] = fn()
        except Exception as e:  # noqa: BLE001
            out[i] = e
    ts = [threading.Thread(target=run, args=(i, fn)) for i, fn in enumerate(calls)]
    for t in ts:
        t.start()
        t.join(0.2)                               # the first is queued before the second comes
    return ts, out


def test_identical_requests_queued_together_share_one_slot(noisy):
    noisy.acquire()                               # the single CPU slot is busy
    ask = lambda: api.speech(api.SpeechRequest(input="Xin chào."), _req())   # noqa: E731
    ts, out = _in_threads(ask, ask)
    assert noisy.waiting == 1                     # only the first queues for a slot
    noisy.release()
    for t in ts:
        t.join(5)
    a, b = out
    assert _body(a) == _body(b) and len(noisy.tts.stream_calls) == 1
    gc.collect()
    assert noisy.active == 0


def test_merged_request_shares_the_429_of_the_one_it_waited_with(noisy, monkeypatch):
    monkeypatch.setattr(noisy, "queue_timeout", 0.5)
    noisy.acquire()
    ask = lambda: api.speech(api.SpeechRequest(input="Xin chào."), _req())   # noqa: E731
    ts, out = _in_threads(ask, ask)
    for t in ts:
        t.join(5)
    noisy.release()
    assert all(isinstance(e, api.HTTPException) and e.status_code == 429 for e in out)
    states = [r["state"] for r in api.ACTIVITY.snapshot()["recent"][:2]]
    assert states == ["rejected", "rejected"] and api.MERGES.live == {}


# ── speed / pitch / inline tags ──────────────────────────────────────────────

def _fresh(eng):
    """Forget the startup warm-up call, so a test sees only its own."""
    eng.tts.stream_calls.clear()
    eng.tts.texts.clear()


def _samples(r):
    assert r.status_code == 200, r.text
    return len(r.content) // 2          # pcm s16le


def test_speed_shortens_the_audio_and_is_no_longer_ignored(client, eng):
    plain = _samples(_speech(client, response_format="pcm"))
    r = _speech(client, response_format="pcm", speed=2.0)
    assert _samples(r) == plain // 2
    assert "X-VieNeu-Ignored" not in r.headers


def test_pitch_keeps_the_length(client, eng):
    plain = _samples(_speech(client, response_format="pcm"))
    assert _samples(_speech(client, response_format="pcm", pitch=-4)) == plain


def test_plain_request_is_one_untouched_call(client, eng):
    _fresh(eng)
    _speech(client, input="Xin chào [cười] bạn.")
    assert eng.tts.texts == ["Xin chào [cười] bạn."]
    assert eng.tts.stream_calls[0]["apply_watermark"] == eng.watermark


def test_inline_tags_split_the_input_by_voice_and_speed(client, eng):
    _fresh(eng)
    r = _speech(client, response_format="pcm",
                input="Một. [speed=2 voice=Mai Anh] Hai. [reset] Ba.")
    assert [t.strip() for t in eng.tts.texts] == ["Một.", "Hai.", "Ba."]
    assert [c["voice"] for c in eng.tts.stream_calls] == [None, "Mai Anh", None]
    assert _samples(r) == 4800 + 2400 + 4800


@pytest.mark.parametrize("body", [{"speed": 5}, {"pitch": 13}, {"input": "a [speed=0.1] b"},
                                  {"input": "a [pitch=x] b"}, {"input": "a [voice=Nobody] b"}])
def test_bad_prosody_is_rejected_up_front(client, eng, body):
    _fresh(eng)
    assert _speech(client, **body).status_code in (400, 422)
    assert eng.tts.stream_calls == []


def test_instructions_are_still_reported_as_ignored(client, eng):
    assert _speech(client, instructions="vui vẻ").headers["X-VieNeu-Ignored"] == "instructions"


# ── native mode: the reference is reshaped, the model speaks it ─────────────

@pytest.fixture
def native(eng, monkeypatch):
    plans = []

    def voice_variant(voice, plan):
        plans.append((voice, plan))
        return {"speaker_emb": np.zeros(4, np.float32), "codes": np.zeros((2, 16), np.int64)}
    eng.tts.voice_variant = voice_variant
    _fresh(eng)
    return plans


def test_native_speed_reshapes_the_reference(client, eng, native):
    r = _speech(client, response_format="pcm", speed=1.3, voice="Mai Anh")
    assert r.status_code == 200
    (voice, plan), = native
    assert voice == "Mai Anh" and plan.ref_speed > 1.3 and plan.res_speed == 1.0
    assert isinstance(eng.tts.stream_calls[0]["voice"], dict)   # the reshaped voice
    assert eng.tts.stream_calls[0]["apply_watermark"] == eng.watermark   # no DSP after


def test_native_mode_can_be_turned_off(client, eng, native, monkeypatch):
    monkeypatch.setenv("VIENEU_PROSODY", "dsp")
    assert _samples(_speech(client, response_format="pcm", speed=2.0)) == 2400
    assert native == []


def test_volume_alone_keeps_the_voice_and_gains_the_output(client, eng, native):
    eng.tts.infer_stream = lambda text, **kw: iter([np.full(4800, 0.1, np.float32)])
    pcm = np.frombuffer(_speech(client, response_format="pcm", volume=0.5).content, np.int16)
    assert native == [] and abs(pcm.mean() / 32768 - 0.05) < 1e-3


def test_effort_tag_strains_the_voice_for_that_part_only(client, eng, native):
    _speech(client, input="Bình tĩnh. [shout=2] Cút ra khỏi đây! [reset] Xong.")
    (_, plan), = native
    assert plan.ref_pitch > 0 and plan.tilt_db > 0 and plan.drive > 1
    assert [isinstance(c["voice"], dict) for c in eng.tts.stream_calls] == [False, True, False]


@pytest.mark.parametrize("body", [{"volume": 0}, {"effort": 3}, {"input": "a [effort=5] b"},
                                  {"input": "a [volume=9] b"}])
def test_bad_volume_or_effort_is_rejected(client, eng, body):
    _fresh(eng)
    assert _speech(client, **body).status_code in (400, 422)
    assert eng.tts.stream_calls == []
