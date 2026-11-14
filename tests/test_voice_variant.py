"""Voice variants (reshaped references): kept on disk across restarts, and
built ahead in the background only while the server is idle."""
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from vieneu.v3turbo import V3TurboVieNeuTTS
from vieneu_utils.prosody import Prosody, plan_for


class FakeEngine:
    def __init__(self):
        self.builds = 0

    def _decode_codes(self, codes):
        return (np.sin(np.arange(48000 * 3) / 7) * 0.3).astype(np.float32)

    def prepare_reference(self, wav, sr, denoise):
        self.builds += 1
        return np.ones(4, np.float32), np.zeros((5, 16), np.int64)


def _tts():
    t = V3TurboVieNeuTTS.__new__(V3TurboVieNeuTTS)          # no model: only what voice_variant uses
    t.engine, t.backend, t.sample_rate = FakeEngine(), "onnx", 48000
    t._preset_voices = {"Mai Anh": {"speaker_emb": np.zeros(4, np.float32), "codes": np.zeros((5, 16), np.int64)}}
    t._voice_aliases, t._default_voice = {}, "Mai Anh"
    return t


def test_variant_survives_a_restart_through_the_disk(tmp_path, monkeypatch):
    monkeypatch.setenv("VIENEU_VARIANT_DIR", str(tmp_path))
    plan = plan_for(Prosody(effort=2))
    first = _tts()
    v = first.voice_variant("Mai Anh", plan)
    first.voice_variant("Mai Anh", plan)                  # memory hit
    assert first.engine.builds == 1 and len(list(tmp_path.glob("*.npz"))) == 1
    again = _tts()                                        # a new process
    assert np.array_equal(again.voice_variant("Mai Anh", plan)["codes"], v["codes"])
    assert again.engine.builds == 0


def test_an_edited_voice_is_built_again(tmp_path, monkeypatch):
    monkeypatch.setenv("VIENEU_VARIANT_DIR", str(tmp_path))
    plan = plan_for(Prosody(speed=1.3))
    _tts().voice_variant("Mai Anh", plan)
    t = _tts()
    t._preset_voices["Mai Anh"]["codes"] = np.ones((5, 16), np.int64)
    t.voice_variant("Mai Anh", plan)
    assert t.engine.builds == 1


def test_memory_only_when_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("VIENEU_VARIANT_DIR", "0")
    monkeypatch.chdir(tmp_path)
    _tts().voice_variant("Mai Anh", plan_for(Prosody(pitch=3)))
    assert list(tmp_path.rglob("*.npz")) == []


# ── prewarm ──────────────────────────────────────────────────────────────────

from apps import openai_speech as api   # noqa: E402


def _eng(names=("Mai Anh", "Hải Đăng")):
    built = []
    tts = SimpleNamespace(_preset_voices={n: {} for n in names},
                          resolve_voice_name=lambda n: n if n in names else None,
                          voice_variant=lambda v, p: built.append((v, p)))
    return SimpleNamespace(tts=tts, active=0, waiting=0), built


def test_prewarm_covers_every_voice_and_setting(monkeypatch):
    monkeypatch.delenv("VIENEU_PREWARM", raising=False)
    monkeypatch.delenv("VIENEU_PREWARM_VOICES", raising=False)
    eng, _ = _eng()
    jobs = api._prewarm_jobs(eng)
    assert len(jobs) == 2 * len(api.DEFAULT_PREWARM.split(";"))


def test_prewarm_settings_and_voices_are_configurable(monkeypatch):
    monkeypatch.setenv("VIENEU_PREWARM", "effort=2; bogus=1 ;speed=1.2 pitch=1;volume=2")
    monkeypatch.setenv("VIENEU_PREWARM_VOICES", "Mai Anh, Nobody")
    eng, _ = _eng()
    jobs = api._prewarm_jobs(eng)   # bogus is skipped; volume alone needs no variant
    assert [v for v, _ in jobs] == ["Mai Anh", "Mai Anh"]
    monkeypatch.setenv("VIENEU_PREWARM", "0")
    assert api._prewarm_jobs(eng) == []


def test_prewarm_waits_while_someone_is_listening():
    eng, built = _eng()
    eng.active = 1
    jobs = [("Mai Anh", plan_for(Prosody(effort=1)))]
    th = threading.Thread(target=api._prewarm_loop, args=(eng, jobs, 0.01), daemon=True)
    th.start()
    time.sleep(0.1)
    assert built == []                       # not while a stream runs
    eng.active = 0
    th.join(2)
    assert len(built) == 1
