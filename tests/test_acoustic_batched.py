"""Batched acoustic head + frame scheduler: the generated ONNX graph must equal a
plain numpy port of ``AcousticDecoder.cached_step`` (modeling_v3_turbo.py) — at
batch 1 and in a batch, where each row must not depend on its neighbours — the
vectorised sampler must match the engine's, and the scheduler must give every
concurrent stream exactly what it gets alone."""
import threading

import numpy as np
import pytest

pytest.importorskip("onnx")

from vieneu._v3_turbo_engine.acoustic_batched import AcousticHead, build_acoustic_onnx, sample_rows
from vieneu._v3_turbo_engine.lite_scheduler import FrameScheduler

H, NH, F, NVQ, EPS = 64, 4, 96, 16, 1e-6


def _rms(x, w):
    return x / np.sqrt(np.mean(x * x, -1, keepdims=True) + EPS) * w


def _lin(x, w, quant):
    """x @ w.T — exactly, or with the graph's int8 scheme (per-channel weights,
    per-row 7-bit activations around zero point 64)."""
    w = w.T
    if quant == "fp32":
        return x @ w
    ws = np.maximum(np.abs(w).max(0), 1e-12) / 127.0
    wq = np.clip(np.round(w / ws), -127, 127)
    xs = np.maximum(np.abs(x).max(-1, keepdims=True), 1e-12) / 63.0
    xq = np.clip(np.round(x / xs) + 64, 0, 127) - 64
    return (xq @ wq) * xs * ws


def ref_step(t, x, pos0, past, n_layers, quant="fp32"):
    """numpy port of AcousticDecoder.cached_step (float64 for a tight reference)."""
    p = "acoustic_decoder."
    B, S, _ = x.shape
    hd = H // NH
    x = x.astype(np.float64) + t[p + "slot_pos_emb.weight"][pos0:pos0 + S]
    P = 0 if past is None else past[0].shape[2]
    mask = np.triu(np.full((S, P + S), -np.inf), k=P + 1)
    new = []
    for i in range(n_layers):
        q_ = f"{p}layers.{i}."
        qkv = _lin(_rms(x, t[q_ + "norm1.weight"]), t[q_ + "attn.qkv.weight"], quant).reshape(B, S, 3, NH, hd)
        q, k, v = (qkv[:, :, j].transpose(0, 2, 1, 3) for j in range(3))
        q, k = _rms(q, t[q_ + "attn.q_norm.weight"]), _rms(k, t[q_ + "attn.k_norm.weight"])
        if past is not None:
            k, v = np.concatenate([past[2 * i], k], 2), np.concatenate([past[2 * i + 1], v], 2)
        new += [k, v]
        a = q @ k.transpose(0, 1, 3, 2) / np.sqrt(hd) + mask
        a = np.exp(a - a.max(-1, keepdims=True))
        a /= a.sum(-1, keepdims=True)
        x = x + _lin((a @ v).transpose(0, 2, 1, 3).reshape(B, S, H), t[q_ + "attn.o_proj.weight"], quant)
        n2 = _rms(x, t[q_ + "norm2.weight"])
        g = _lin(n2, t[q_ + "ff_gate.weight"], quant)
        x = x + _lin(g / (1 + np.exp(-g)) * _lin(n2, t[q_ + "ff_up.weight"], quant), t[q_ + "ff_down.weight"], quant)
    return _rms(x, t[p + "norm.weight"]), new


def weights(n_layers, seed=0):
    rng = np.random.default_rng(seed)
    p = "acoustic_decoder."
    t = {p + "slot_pos_emb.weight": rng.standard_normal((NVQ + 1, H)), p + "norm.weight": 1 + 0.1 * rng.standard_normal(H)}
    for i in range(n_layers):
        q_ = f"{p}layers.{i}."
        t.update({q_ + "norm1.weight": 1 + 0.1 * rng.standard_normal(H), q_ + "norm2.weight": 1 + 0.1 * rng.standard_normal(H),
                  q_ + "attn.q_norm.weight": 1 + 0.1 * rng.standard_normal(H // NH),
                  q_ + "attn.k_norm.weight": 1 + 0.1 * rng.standard_normal(H // NH),
                  q_ + "attn.qkv.weight": rng.standard_normal((3 * H, H)) / 8,
                  q_ + "attn.o_proj.weight": rng.standard_normal((H, H)) / 8,
                  q_ + "ff_gate.weight": rng.standard_normal((F, H)) / 8,
                  q_ + "ff_up.weight": rng.standard_normal((F, H)) / 8,
                  q_ + "ff_down.weight": rng.standard_normal((H, F)) / 10})
    return {k: v.astype(np.float32) for k, v in t.items()}


@pytest.fixture(params=[(1, "fp32"), (2, "fp32"), (1, "int8"), (2, "int8")],
                ids=["1-layer-fp32", "2-layer-fp32", "1-layer-int8", "2-layer-int8"])
def head(request, tmp_path):
    n, quant = request.param
    t = weights(n)
    path = build_acoustic_onnx(t, n, NH, EPS, str(tmp_path / "ac.onnx"), quant)
    return AcousticHead(path, n, NH, H), t, n, quant


def test_graph_matches_reference_over_a_full_frame(head):
    h, t, n, quant = head
    t64 = {k: v.astype(np.float64) for k, v in t.items()}
    rng = np.random.default_rng(1)
    worst = 0.0

    def check(out, ref):
        nonlocal worst
        worst = max(worst, float(np.max(np.linalg.norm(out - ref, axis=-1) / np.linalg.norm(ref, axis=-1))))

    x = rng.standard_normal((3, 2, H)).astype(np.float32)
    out, past = h.step(x, 0, None)
    ref, rpast = ref_step(t64, x, 0, None, n, quant)
    check(out, ref)
    for ch in range(1, NVQ):                                  # 15 cached single-slot steps
        x = rng.standard_normal((3, 1, H)).astype(np.float32)
        out, past = h.step(x, ch + 1, past)
        ref, rpast = ref_step(t64, x, ch + 1, rpast, n, quant)
        check(out, ref)
    # same arithmetic as the reference (int8: up to a rare float32 rounding flip)
    assert worst < (1e-4 if quant == "fp32" else 2e-3), worst
    assert past[0].shape == (3, NH, NVQ + 1, H // NH)


def test_int8_graph_stays_close_to_fp32(tmp_path):
    t = weights(1)
    h8 = AcousticHead(build_acoustic_onnx(t, 1, NH, EPS, str(tmp_path / "q.onnx"), "int8"), 1, NH, H)
    h32 = AcousticHead(build_acoustic_onnx(t, 1, NH, EPS, str(tmp_path / "f.onnx"), "fp32"), 1, NH, H)
    x = np.random.default_rng(7).standard_normal((4, 2, H)).astype(np.float32)
    a, b = h8.step(x, 0, None)[0], h32.step(x, 0, None)[0]
    assert np.max(np.linalg.norm(a - b, axis=-1) / np.linalg.norm(b, axis=-1)) < 0.15


def test_rows_do_not_depend_on_batch_neighbours(head):
    h = head[0]
    x = np.random.default_rng(2).standard_normal((4, 2, H)).astype(np.float32)
    batched, _ = h.step(x, 0, None)
    alone, _ = h.step(x[2:3], 0, None)
    np.testing.assert_array_equal(batched[2:3], alone)


class FakeEngine:
    n_vq, sgs, eos_speech, audio_pad = NVQ, 1, 2, 31

    def __init__(self):
        rng = np.random.default_rng(3)
        self.text_emb = rng.standard_normal((8, H)).astype(np.float32)
        self.text_emb[self.eos_speech] = -1e3          # EOS never wins: streams run to their cap
        self.audio_emb = rng.standard_normal((NVQ, 32, H)).astype(np.float32)

    def _embed_rows(self, rows, anchor):                      # (T, n_vq+1) → (1, T, H)
        return (self.audio_emb[0][rows[:, 1:] % 32].sum(1) + anchor)[None].astype(np.float32)


class FakeBackbone:
    """Hidden state from the row's own content only (batch-independent)."""
    n_batch = 4096

    def __init__(self, n_seq_max=3):
        self.n_seq_max, self.calls = n_seq_max, []

    def reset(self, seq):
        pass

    def decode(self, rows):
        self.calls.append(len(rows))
        return np.stack([np.tanh(0.05 * e.sum(0) + 0.1 * (p0 + len(e))) for _, e, p0 in rows]).astype(np.float32)


def test_scheduler_streams_match_solo_runs_and_free_their_slots(head):
    h = head[0]
    eng, rng = FakeEngine(), np.random.default_rng(4)
    prompts = [rng.standard_normal((5 + i, H)).astype(np.float32) for i in range(5)]
    anchors = [rng.standard_normal(H).astype(np.float32) for _ in range(5)]
    greedy = (0.0, 0, 1.0, 1.0, None)

    def run(sched, i, cap=6):
        return list(sched.generate(prompts[i], anchors[i], greedy, cap))

    solo = [run(FrameScheduler(eng, h, FakeBackbone()), i) for i in range(5)]
    bb = FakeBackbone(n_seq_max=3)                            # 5 streams > 3 sequences
    sched = FrameScheduler(eng, h, bb)
    got = {}
    th = [threading.Thread(target=lambda i=i: got.__setitem__(i, run(sched, i))) for i in range(5)]
    [x.start() for x in th]
    [x.join(20) for x in th]
    assert [got[i] for i in range(5)] == solo
    for frames in solo:                                       # cap respected, no EOS
        assert len(frames) == 6 and not any(eos for _, eos in frames)
        assert all(len(c) == NVQ for c, _ in frames)
    assert max(bb.calls) > 1                                  # streams really shared rounds
    # a consumer that stops early gives its sequence back
    it = sched.generate(prompts[0], anchors[0], greedy, 50)
    next(it)
    it.close()
    for _ in range(100):
        if len(sched._free) == 3 and not sched._active:
            break
        threading.Event().wait(0.02)
    assert sorted(sched._free) == [0, 1, 2]


def _ref_sample(logits, t, k, p, r, prev):
    from vieneu._v3_turbo_engine.onnx_runtime_lite import OnnxV3LiteEngine
    return OnnxV3LiteEngine._sample(None, logits, t, k, p, r, prev)   # does not use self


class _Hist(list):
    def __getitem__(self, ch):
        return set(list.__getitem__(self, ch))


def test_batched_sampler_greedy_and_repetition_penalty_match_engine():
    rng = np.random.default_rng(5)
    logits = rng.standard_normal((6, 50)).astype(np.float32) * 3
    hists = [_Hist([[int(i) for i in rng.integers(0, 50, 8)]]) for _ in range(6)]
    params = [(0.0, 25, 0.95, 1.0 + 0.3 * (b % 3), hists[b]) for b in range(6)]
    got = sample_rows(logits, params, 0)
    want = [_ref_sample(logits[b], *params[b][:4], hists[b][0]) for b in range(6)]
    assert got.tolist() == want


def test_batched_sampler_distribution_matches_engine():
    rng = np.random.default_rng(6)
    row = (rng.standard_normal(40) * 2).astype(np.float32)
    settings = [(0.8, 25, 0.95, 1.0), (1.0, 5, 1.0, 1.0), (0.7, 0, 0.8, 1.0)]   # mixed per row
    n = 20000
    np.random.seed(0)
    got = np.stack([sample_rows(np.tile(row, (3, 1)), [s + (None,) for s in settings], 0)
                    for _ in range(n)])
    for j, s in enumerate(settings):
        ref = np.array([_ref_sample(row, *s, None) for _ in range(n)])
        pg, pr = np.bincount(got[:, j], minlength=40) / n, np.bincount(ref, minlength=40) / n
        assert np.abs(pg - pr).max() < 0.015, (s, np.abs(pg - pr).max())
        assert set(np.flatnonzero(pg)) <= set(np.flatnonzero(pr) if s[1] else range(40))
