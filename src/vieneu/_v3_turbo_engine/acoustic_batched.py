"""
Batched acoustic head for the v3 Turbo ONNX engine.
====================================================
Every frame, each stream runs the acoustic decoder 16 times (one slot per VQ
code, sampling in between). The shipped ``vieneu_acoustic_cached.onnx`` takes
batch 1 only, so N streams cost 16·N small calls per frame, from N threads that
contend for the cores and the GIL.

``build_acoustic_onnx`` writes the same network (``AcousticDecoder.cached_step``
in modeling_v3_turbo.py) as an ONNX graph with a batch dimension, from the
original weights in ``model.safetensors``. It is fp32 throughout, which is closer
to the trained model than the shipped graph (dynamic int8, ~2-6 % off), and a
row's result does not depend on the other rows in its batch. ``acoustic_frames``
runs one frame for a whole batch of streams (see ``lite_scheduler``).
"""
from __future__ import annotations

import json
import os
import struct
from typing import List, Optional, Tuple

import numpy as np


def read_safetensors(path: str, prefix: str = "") -> dict:
    """float32 arrays for the tensors whose name starts with ``prefix``."""
    out = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        base = 8 + n
        for k, v in hdr.items():
            if k == "__metadata__" or not k.startswith(prefix):
                continue
            s, e = v["data_offsets"]
            f.seek(base + s)
            raw = f.read(e - s)
            if v["dtype"] == "BF16":
                a = (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32)
            elif v["dtype"] == "F16":
                a = np.frombuffer(raw, np.float16).astype(np.float32)
            elif v["dtype"] == "F32":
                a = np.frombuffer(raw, np.float32)
            else:
                raise ValueError(f"{k}: unsupported dtype {v['dtype']}")
            out[k] = a.reshape(v["shape"])
    return out


def build_acoustic_onnx(tensors: dict, n_layers: int, n_heads: int, eps: float, out_path: str,
                        quant: str = "int8") -> str:
    """Acoustic decoder step as ONNX, batched.

    Inputs:  x (B, S, H) token embeddings · pos (S,) int64 slot ids ·
             mask (S, P+S) additive · past_k_i / past_v_i (B, nH, P, hd)
    Outputs: hidden (B, S, H) · present_k_i / present_v_i (B, nH, P+S, hd)

    ``quant="int8"`` (default): per-channel int8 weights and per-ROW dynamic
    activations (7-bit, uint8 around a fixed zero point 64, corrected with the
    weight column sums), so the weights stay cache-resident and a row's result
    never depends on its batch neighbours — unlike ONNX dynamic quantization,
    whose one scale per tensor couples the rows. 7 bits because x86 AVX2 U8S8
    kernels add u8·s8 pairs in int16, which 255·127·2 would overflow.
    ``"fp32"``: plain MatMul.
    """
    try:
        import onnx
        from onnx import TensorProto, helper, numpy_helper
    except ImportError as e:
        raise ImportError("building the batched acoustic graph needs the `onnx` package: "
                          "uv pip install onnx") from e

    p = "acoustic_decoder."
    pos_tab = np.asarray(tensors[p + "slot_pos_emb.weight"], np.float32)
    H = pos_tab.shape[1]
    hd = H // n_heads
    nodes, inits = [], []
    cnt = [0]

    def name(s):
        cnt[0] += 1
        return f"{s}_{cnt[0]}"

    def const(a, s):
        n = name(s)
        a = np.asarray(a)
        # (np.ascontiguousarray would turn a 0-d index into shape (1,))
        inits.append(numpy_helper.from_array(np.ascontiguousarray(a) if a.ndim else a, n))
        return n

    def op(t, ins, **kw):
        o = name(t.lower())
        nodes.append(helper.make_node(t, ins, [o], **kw))
        return o

    eps_c = const(np.array(eps, np.float32), "eps")

    def rms(x, w):
        ms = op("ReduceMean", [op("Mul", [x, x])], axes=[-1], keepdims=1)
        return op("Mul", [op("Div", [x, op("Sqrt", [op("Add", [ms, eps_c])])]), const(w, "norm_w")])

    qmax = const(np.array(63.0, np.float32), "qmax")
    zp = const(np.array(64.0, np.float32), "zp")
    c127 = const(np.array(127.0, np.float32), "c127")
    c0 = const(np.array(0.0, np.float32), "c0")
    tiny = const(np.array(1e-12, np.float32), "tiny")

    def lin(x, w):  # torch Linear weight (out, in) → x @ w.T
        w = np.asarray(w, np.float32).T                                    # (K, N)
        if quant == "fp32":
            return op("MatMul", [x, const(w, "w")])
        ws = np.maximum(np.abs(w).max(0), 1e-12) / 127.0                     # per output channel
        wq = np.clip(np.round(w / ws), -127, 127).astype(np.int8)
        corr = 64.0 * wq.astype(np.int64).sum(0).astype(np.float32)          # zero-point term
        xs = op("Div", [op("Max", [op("ReduceMax", [op("Abs", [x])], axes=[-1], keepdims=1), tiny]), qmax])
        xq = op("Cast", [op("Clip", [op("Add", [op("Round", [op("Div", [x, xs])]), zp]), c0, c127])],
                to=TensorProto.UINT8)
        acc = op("Sub", [op("Cast", [op("MatMulInteger", [xq, const(wq, "wq")])], to=TensorProto.FLOAT),
                         const(corr, "wcorr")])
        return op("Mul", [op("Mul", [acc, xs]), const(ws.astype(np.float32), "wscale")])

    x = op("Add", ["x", op("Gather", [const(pos_tab, "slot_pos"), "pos"], axis=0)])
    heads_shape = const(np.array([0, 0, 3, n_heads, hd], np.int64), "shape")
    flat_shape = const(np.array([0, 0, H], np.int64), "shape")
    idx = [const(np.array(i, np.int64), "idx") for i in range(3)]
    scale = const(np.array(1.0 / np.sqrt(hd), np.float32), "scale")
    outputs = []
    for i in range(n_layers):
        q_ = f"{p}layers.{i}."
        qkv = op("Reshape", [lin(rms(x, tensors[q_ + "norm1.weight"]), tensors[q_ + "attn.qkv.weight"]), heads_shape])
        q, k, v = (op("Transpose", [op("Gather", [qkv, idx[j]], axis=2)], perm=[0, 2, 1, 3]) for j in range(3))
        q = rms(q, tensors[q_ + "attn.q_norm.weight"])
        k = rms(k, tensors[q_ + "attn.k_norm.weight"])
        k = op("Concat", [f"past_k_{i}", k], axis=2)
        v = op("Concat", [f"past_v_{i}", v], axis=2)
        nodes.append(helper.make_node("Identity", [k], [f"present_k_{i}"]))
        nodes.append(helper.make_node("Identity", [v], [f"present_v_{i}"]))
        att = op("Add", [op("Mul", [op("MatMul", [q, op("Transpose", [k], perm=[0, 1, 3, 2])]), scale]), "mask"])
        o = op("Reshape", [op("Transpose", [op("MatMul", [op("Softmax", [att], axis=-1), v])], perm=[0, 2, 1, 3]),
                           flat_shape])
        x = op("Add", [x, lin(o, tensors[q_ + "attn.o_proj.weight"])])
        n2 = rms(x, tensors[q_ + "norm2.weight"])
        g = lin(n2, tensors[q_ + "ff_gate.weight"])
        ff = op("Mul", [op("Mul", [g, op("Sigmoid", [g])]), lin(n2, tensors[q_ + "ff_up.weight"])])
        x = op("Add", [x, lin(ff, tensors[q_ + "ff_down.weight"])])
        outputs += [f"present_k_{i}", f"present_v_{i}"]
    nodes.append(helper.make_node("Identity", [rms(x, tensors[p + "norm.weight"])], ["hidden"]))

    f = TensorProto.FLOAT
    kv = lambda n, P: helper.make_tensor_value_info(n, f, ["B", n_heads, P, hd])
    g = helper.make_graph(
        nodes, "vieneu_acoustic_batched",
        [helper.make_tensor_value_info("x", f, ["B", "S", H]),
         helper.make_tensor_value_info("pos", TensorProto.INT64, ["S"]),
         helper.make_tensor_value_info("mask", f, ["S", "T"])]
        + [kv(f"past_{t}_{i}", "P") for i in range(n_layers) for t in "kv"],
        [helper.make_tensor_value_info("hidden", f, ["B", "S", H])]
        + [kv(o, "T") for o in outputs],
        inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    tmp = out_path + ".tmp"
    onnx.save(m, tmp)
    os.replace(tmp, out_path)
    return out_path


class AcousticHead:
    """ORT session over the batched graph, with a cached-step API."""

    def __init__(self, onnx_path: str, n_layers: int, n_heads: int, hidden: int, sess_options=None):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(onnx_path, sess_options, providers=["CPUExecutionProvider"])
        self.L, self.nH, self.hd = n_layers, n_heads, hidden // n_heads
        self._outs = ["hidden"] + [f"present_{t}_{i}" for i in range(n_layers) for t in "kv"]

    def step(self, x: np.ndarray, pos0: int, past: Optional[list]):
        """x (B, S, H) at slots pos0.. → (hidden (B, S, H), past)."""
        B, S, _ = x.shape
        P = 0 if past is None else past[0].shape[2]
        feed = {"x": np.ascontiguousarray(x, np.float32),
                "pos": np.arange(pos0, pos0 + S, dtype=np.int64),
                "mask": np.triu(np.full((S, P + S), -np.inf, np.float32), k=P + 1)}
        empty = np.zeros((B, self.nH, 0, self.hd), np.float32)
        for i in range(self.L):
            feed[f"past_k_{i}"] = empty if past is None else past[2 * i]
            feed[f"past_v_{i}"] = empty if past is None else past[2 * i + 1]
        out = self.sess.run(self._outs, feed)
        return out[0], out[1:]


def sample_rows(logits: np.ndarray, params, ch: int) -> np.ndarray:
    """Row-wise ``OnnxV3LiteEngine._sample`` for (B, V) logits: repetition
    penalty → temperature → top-k → top-p → draw. Same distribution per row;
    rows may use different settings. ``params[b]`` is
    ``(temperature, top_k, top_p, repetition_penalty, history | None)``."""
    logits = np.array(logits, dtype=np.float32)
    B, V = logits.shape
    temp = np.empty(B, np.float32)
    topk = np.empty(B, np.int64)
    topp = np.empty(B, np.float32)
    for b, (t, k, p, r, hist) in enumerate(params):
        prev = hist[ch] if hist is not None else None
        if prev and not np.isclose(r, 1.0):
            idx = np.fromiter(prev, dtype=np.int64, count=len(prev))
            sel = logits[b, idx]
            logits[b, idx] = np.where(sel < 0, sel * r, sel / r)
        temp[b] = t if (t and t > 0) else 0.0
        topk[b] = int(k) if (k and 0 < int(k) < V) else V
        topp[b] = p if (p and p < 1.0) else 1.0
    out = logits.argmax(-1)
    s = np.flatnonzero(temp > 0)
    if s.size:
        lg = logits[s] / temp[s, None]
        K = int(topk[s].max())
        cand = np.argpartition(lg, -K, axis=-1)[:, -K:] if K < V else np.broadcast_to(np.arange(V), lg.shape)
        cs = np.take_along_axis(lg, cand, -1)
        order = np.argsort(-cs, axis=-1, kind="stable")
        cand, cs = np.take_along_axis(cand, order, -1), np.take_along_axis(cs, order, -1)
        cs = np.where(np.arange(K)[None] < topk[s, None], cs, -np.inf)       # each row's own k
        pr = np.exp(cs - cs[:, :1])
        pr /= pr.sum(-1, keepdims=True)
        pr = pr * ((np.cumsum(pr, -1) - pr) < topp[s, None])                # nucleus
        c = np.cumsum(pr, -1)
        u = np.random.random(s.size) * c[:, -1]
        pick = np.minimum((c <= u[:, None]).sum(-1), K - 1)
        out[s] = cand[np.arange(s.size), pick]
    return out


def acoustic_frames(eng, head: AcousticHead, hs: np.ndarray, params) -> List[Tuple[List[int], bool]]:
    """One frame for B streams: backbone hidden states (B, H) → per stream
    ``(16 codes, eos)``. ``eng`` supplies the tied heads (text_emb, audio_emb)."""
    B = len(hs)
    txt = eng.text_emb[eng.sgs]
    x = np.stack([np.stack([h, txt]) for h in hs]).astype(np.float32)   # (B, 2, H)
    out, past = head.step(x, 0, None)
    slot0 = out[:, 0]
    codes: List[List[int]] = [[] for _ in range(B)]

    def samp(ch, vecs):
        got = sample_rows(vecs @ eng.audio_emb[ch].T, params, ch)        # (B,)
        for b in range(B):
            c = int(got[b])
            hist = params[b][4]
            if hist is not None:
                hist[ch].add(c)
            codes[b].append(c)

    samp(0, out[:, 1])
    for ch in range(1, eng.n_vq):
        emb = eng.audio_emb[ch - 1][[c[-1] for c in codes]]              # (B, H)
        out, past = head.step(emb[:, None].astype(np.float32), ch + 1, past)
        samp(ch, out[:, 0])
    eos = (slot0 @ eng.text_emb.T).argmax(-1) == eng.eos_speech
    return [(codes[b], bool(eos[b])) for b in range(B)]
