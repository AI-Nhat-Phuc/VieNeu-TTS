"""llama.cpp backbone helpers: the safetensors → qwen3 GGUF converter (on a tiny
synthetic model) and the StepBatcher that merges concurrent streams' decode
steps. Neither needs the llama.cpp binaries."""
import json
import struct
import threading
import time

import numpy as np
import pytest

from vieneu._v3_turbo_engine.llamacpp_backbone import StepBatcher, convert_backbone_gguf

CFG = dict(num_hidden_layers=2, hidden_size=64, intermediate_size=128, num_attention_heads=4,
           num_key_value_heads=2, head_dim=16, max_position_embeddings=256, rope_theta=10000.0,
           rms_norm_eps=1e-6, bos_token_id=1, eos_token_id=2)


def _write_bf16_safetensors(path, tensors):
    hdr, blobs, off = {}, [], 0
    for k, a in tensors.items():
        raw = (np.ascontiguousarray(a, np.float32).view(np.uint32) >> 16).astype(np.uint16).tobytes()
        hdr[k] = {"dtype": "BF16", "shape": list(a.shape), "data_offsets": [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    h = json.dumps(hdr).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h + b"".join(blobs))


def _bf16(a):
    return ((np.asarray(a, np.float32).view(np.uint32) >> 16) << 16).view(np.float32)


@pytest.fixture
def tiny(tmp_path):
    rng = np.random.default_rng(0)
    H, F, L = CFG["hidden_size"], CFG["intermediate_size"], CFG["num_hidden_layers"]
    q, kv = CFG["num_attention_heads"] * CFG["head_dim"], CFG["num_key_value_heads"] * CFG["head_dim"]
    t = {"text_embeddings.weight": rng.standard_normal((40, H)),
         "semantic_backbone.norm.weight": rng.standard_normal(H),
         "acoustic_decoder.norm.weight": rng.standard_normal(H)}  # not part of the backbone
    shapes = {"input_layernorm": (H,), "post_attention_layernorm": (H,),
              "self_attn.q_proj": (q, H), "self_attn.k_proj": (kv, H), "self_attn.v_proj": (kv, H),
              "self_attn.o_proj": (H, q), "self_attn.q_norm": (CFG["head_dim"],),
              "self_attn.k_norm": (CFG["head_dim"],), "mlp.gate_proj": (F, H),
              "mlp.up_proj": (F, H), "mlp.down_proj": (H, F)}
    for i in range(L):
        for n, s in shapes.items():
            t[f"semantic_backbone.layers.{i}.{n}.weight"] = rng.standard_normal(s)
    st, cfg = tmp_path / "model.safetensors", tmp_path / "config.json"
    _write_bf16_safetensors(st, t)
    cfg.write_text(json.dumps(CFG), encoding="utf-8")
    return st, cfg, t


@pytest.mark.parametrize("qtype", ["f32", "q8_0"])
def test_convert_backbone_gguf(tiny, tmp_path, qtype):
    gguf = pytest.importorskip("gguf")
    st, cfg, t = tiny
    out = convert_backbone_gguf(str(st), str(cfg), str(tmp_path / f"bb-{qtype}.gguf"), qtype)
    r = gguf.GGUFReader(out)
    assert r.fields["general.architecture"].contents() == "qwen3"
    assert r.fields["qwen3.block_count"].contents() == CFG["num_hidden_layers"]
    assert r.fields["qwen3.attention.head_count_kv"].contents() == CFG["num_key_value_heads"]
    tensors = {x.name: x for x in r.tensors}
    assert len(tensors) == 3 + 11 * CFG["num_hidden_layers"]
    assert "blk.1.attn_k_norm.weight" in tensors and "output_norm.weight" in tensors
    q = tensors["blk.0.attn_q.weight"]
    if qtype == "f32":
        src = _bf16(t["semantic_backbone.layers.0.self_attn.q_proj.weight"])
        np.testing.assert_array_equal(np.asarray(q.data).reshape(src.shape), src)
    else:
        assert q.tensor_type == gguf.GGMLQuantizationType.Q8_0
        # norms stay f32
        assert tensors["blk.0.attn_norm.weight"].tensor_type == gguf.GGMLQuantizationType.F32


class FakeBackbone:
    """decode() returns, per row, the row's embedding sum + pos0 (so results are
    checkable) and records how many rows each call carried."""

    n_batch = 64

    def __init__(self, n_seq_max=4):
        self.n_seq_max = n_seq_max
        self.calls, self.resets = [], []

    def reset(self, seq):
        self.resets.append(seq)

    def decode(self, rows):
        self.calls.append(len(rows))
        time.sleep(0.01)  # long enough for other streams to queue up behind it
        return np.array([[e.sum() + p0] for _, e, p0 in rows], np.float32)


def test_step_batcher_merges_streams_and_recycles_sequences():
    bb = FakeBackbone(n_seq_max=4)
    b = StepBatcher(bb)
    results, errors = {}, []

    def stream(k):
        try:
            h, seq = b.begin(np.full((3, 1), k, np.float32))
            got = [float(h[0])]
            for t in range(5):
                got.append(float(b.step(seq, np.array([float(k)]), 3 + t)[0]))
            b.end(seq)
            results[k] = got
        except Exception as e:  # pragma: no cover - surfaced below
            errors.append(e)

    th = [threading.Thread(target=stream, args=(k,)) for k in range(6)]  # 6 streams > 4 seqs
    [x.start() for x in th]
    [x.join(10) for x in th]
    assert not errors
    for k in range(6):
        assert results[k] == [3.0 * k] + [k + 3.0 + t for t in range(5)]
    assert max(bb.calls) > 1                      # steps of different streams were merged
    assert sum(bb.calls) == 6 * 6                 # every prefill/step decoded exactly once
    assert sorted(b._free) == [0, 1, 2, 3]        # all sequences returned to the pool


def test_step_batcher_propagates_decode_errors():
    bb = FakeBackbone()
    bb.decode = lambda rows: (_ for _ in ()).throw(RuntimeError("boom"))
    b = StepBatcher(bb)
    with pytest.raises(RuntimeError, match="boom"):
        b.begin(np.zeros((2, 1), np.float32))
    assert len(b._free) == bb.n_seq_max           # the failed prefill gave its seq back
