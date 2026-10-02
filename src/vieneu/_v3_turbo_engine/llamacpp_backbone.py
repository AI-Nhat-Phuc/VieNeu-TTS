"""
llama.cpp (ggml) runner for the v3 Turbo semantic backbone.
============================================================
The backbone is a stock Qwen3Model, so llama.cpp runs it natively once converted
to a ``qwen3`` GGUF (``convert_backbone_gguf`` below; the engine does it once). Inputs are the
engine's precomputed ``inputs_embeds``; outputs are the final-norm hidden states.

Why: on GPUs without CUDA (AMD Polaris/RDNA, Intel) llama.cpp's Vulkan backend is
the only fast path, and it batches many streams in one decode call. Even on CPU,
ggml's Q8_0 decode step is ~2x faster than ONNX Runtime int8.

Bound with ctypes against the prebuilt ``llama.dll``/``libllama.so`` from a
llama.cpp release (no compiler, no Python wheel). Only the leading fields of the
param structs are mirrored; the rest is padding — large structs are passed and
returned through a hidden pointer, so a bigger buffer is safe. Field order is
from llama.h at release b11321.
"""
from __future__ import annotations

import ctypes as C
import logging
import os
import sys
import threading
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

_PAD = 512


class _ModelParams(C.Structure):
    _fields_ = [("devices", C.c_void_p), ("tensor_buft_overrides", C.c_void_p),
                ("n_gpu_layers", C.c_int32), ("_pad", C.c_uint8 * _PAD)]


class _CtxParams(C.Structure):
    _fields_ = [
        ("n_ctx", C.c_uint32), ("n_batch", C.c_uint32), ("n_ubatch", C.c_uint32),
        ("n_seq_max", C.c_uint32), ("n_rs_seq", C.c_uint32), ("n_outputs_max", C.c_uint32),
        ("n_outputs_max_per_seq", C.c_uint32), ("n_threads", C.c_int32), ("n_threads_batch", C.c_int32),
        ("ctx_type", C.c_int), ("rope_scaling_type", C.c_int), ("pooling_type", C.c_int),
        ("attention_type", C.c_int), ("flash_attn_type", C.c_int),
        ("rope_freq_base", C.c_float), ("rope_freq_scale", C.c_float), ("yarn_ext_factor", C.c_float),
        ("yarn_attn_factor", C.c_float), ("yarn_beta_fast", C.c_float), ("yarn_beta_slow", C.c_float),
        ("yarn_orig_ctx", C.c_uint32), ("defrag_thold", C.c_float),
        ("cb_eval", C.c_void_p), ("cb_eval_user_data", C.c_void_p),
        ("type_k", C.c_int), ("type_v", C.c_int),
        ("abort_callback", C.c_void_p), ("abort_callback_data", C.c_void_p),
        ("embeddings", C.c_bool), ("offload_kqv", C.c_bool), ("no_perf", C.c_bool),
        ("op_offload", C.c_bool), ("swa_full", C.c_bool), ("kv_unified", C.c_bool),
        ("_pad", C.c_uint8 * _PAD),
    ]


class _Batch(C.Structure):
    _fields_ = [("n_tokens", C.c_int32), ("token", C.POINTER(C.c_int32)), ("embd", C.POINTER(C.c_float)),
                ("pos", C.POINTER(C.c_int32)), ("n_seq_id", C.POINTER(C.c_int32)),
                ("seq_id", C.POINTER(C.POINTER(C.c_int32))), ("logits", C.POINTER(C.c_int8))]


_LIB = None
_LIB_LOCK = threading.Lock()

# llama.cpp prints every tensor and buffer at load time; keep only warnings and
# errors, through Python logging. ggml_log_level: DEBUG=1 INFO=2 WARN=3 ERROR=4,
# CONT=5 continues the previous message.
_log = logging.getLogger("Vieneu.llama.cpp")
_LOG_CB_T = C.CFUNCTYPE(None, C.c_int, C.c_char_p, C.c_void_p)
_last_level = [0]


# Expected with embeddings input (llama.cpp then outputs every token): logged on each prefill.
_BENIGN = ("were not marked as outputs",)


def _on_llama_log(level, text, _user):
    if level != 5:
        _last_level[0] = level
    if _last_level[0] >= 3:
        msg = (text or b"").decode("utf-8", "replace").rstrip()
        if msg and not any(b in msg for b in _BENIGN):
            _log.log(logging.ERROR if _last_level[0] >= 4 else logging.WARNING, msg)


_LOG_CB = _LOG_CB_T(_on_llama_log)   # module-level: must outlive the library


def _load_lib(lib_dir: str):
    global _LIB
    with _LIB_LOCK:
        if _LIB is not None:
            return _LIB
        d = str(Path(lib_dir).resolve())
        if sys.platform == "win32":
            os.add_dll_directory(d)
            lib = lambda n: C.CDLL(os.path.join(d, f"{n}.dll"))
        else:
            lib = lambda n: C.CDLL(os.path.join(d, f"lib{n}.so"))
        ggml_base, ggml, llama = lib("ggml-base"), lib("ggml"), lib("llama")
        # Quiet logging first, so backend/device init goes through it too.
        for fn in (ggml_base.ggml_log_set, llama.llama_log_set):
            fn.argtypes = [_LOG_CB_T, C.c_void_p]
            fn.restype = None
            fn(_LOG_CB, None)
        # Release builds ship the CPU/Vulkan backends as loadable modules.
        ggml.ggml_backend_load_all_from_path.argtypes = [C.c_char_p]
        ggml.ggml_backend_load_all_from_path(d.encode())
        L = llama
        L.llama_backend_init.restype = None
        L.llama_model_default_params.restype = _ModelParams
        L.llama_context_default_params.restype = _CtxParams
        L.llama_model_load_from_file.argtypes = [C.c_char_p, _ModelParams]
        L.llama_model_load_from_file.restype = C.c_void_p
        L.llama_init_from_model.argtypes = [C.c_void_p, _CtxParams]
        L.llama_init_from_model.restype = C.c_void_p
        L.llama_model_n_embd.argtypes = [C.c_void_p]
        L.llama_model_n_embd.restype = C.c_int32
        L.llama_batch_init.argtypes = [C.c_int32, C.c_int32, C.c_int32]
        L.llama_batch_init.restype = _Batch
        L.llama_batch_free.argtypes = [_Batch]
        L.llama_batch_free.restype = None
        L.llama_decode.argtypes = [C.c_void_p, _Batch]
        L.llama_decode.restype = C.c_int32
        L.llama_get_embeddings_ith.argtypes = [C.c_void_p, C.c_int32]
        L.llama_get_embeddings_ith.restype = C.POINTER(C.c_float)
        L.llama_get_memory.argtypes = [C.c_void_p]
        L.llama_get_memory.restype = C.c_void_p
        L.llama_memory_seq_rm.argtypes = [C.c_void_p, C.c_int32, C.c_int32, C.c_int32]
        L.llama_memory_seq_rm.restype = C.c_bool
        L.llama_free.argtypes = [C.c_void_p]
        L.llama_model_free.argtypes = [C.c_void_p]
        L.llama_backend_init()
        _LIB = L
        return L


class LlamaBackbone:
    """Qwen3 backbone in llama.cpp. One KV-cache sequence per concurrent stream.

    ``decode`` takes rows of (seq, embeds (T, H), first position) and returns the
    last hidden state of each row, all in a single ``llama_decode`` call — so N
    streams cost one GPU dispatch per step instead of N.
    """

    def __init__(self, gguf_path: str, lib_dir: str, n_gpu_layers: int = 99,
                 n_seq_max: int = 1, n_ctx_per_seq: int = 2048, threads: int = 0,
                 n_batch: int = 2048):
        L = _load_lib(lib_dir)
        self.L = L
        mp = L.llama_model_default_params()
        mp.n_gpu_layers = int(n_gpu_layers)
        self.model = L.llama_model_load_from_file(str(gguf_path).encode(), mp)
        if not self.model:
            raise RuntimeError(f"llama.cpp could not load {gguf_path}")
        cp = L.llama_context_default_params()
        cp.n_seq_max = int(n_seq_max)
        cp.n_ctx = int(n_ctx_per_seq) * int(n_seq_max)
        cp.n_batch = cp.n_ubatch = int(n_batch)
        thr = int(threads) if threads and threads > 0 else max((os.cpu_count() or 8) // 2, 1)
        cp.n_threads = cp.n_threads_batch = thr
        cp.embeddings = True
        cp.pooling_type = 0          # NONE: per-token hidden states
        # One KV buffer for all sequences: a decode of N streams is one ubatch
        # (split per-sequence caches cost ~1.5-3x per step on Vulkan).
        cp.kv_unified = True
        cp.no_perf = True
        self.ctx = L.llama_init_from_model(self.model, cp)
        if not self.ctx:
            raise RuntimeError("llama.cpp context creation failed")
        self.mem = L.llama_get_memory(self.ctx)
        self.H = L.llama_model_n_embd(self.model)
        self.n_seq_max = int(n_seq_max)
        self.n_batch = int(n_batch)
        self._batch = L.llama_batch_init(self.n_batch, self.H, 1)
        self._lock = threading.Lock()

    def reset(self, seq: int) -> None:
        with self._lock:
            self.L.llama_memory_seq_rm(self.mem, int(seq), -1, -1)

    def decode(self, rows: Sequence[Tuple[int, np.ndarray, int]]) -> np.ndarray:
        """rows: (seq_id, embeds (T, H) float32, pos0). Returns (len(rows), H)."""
        n = sum(int(e.shape[0]) for _, e, _ in rows)
        if n > self.n_batch:
            raise ValueError(f"batch of {n} tokens exceeds n_batch={self.n_batch}")
        flat = np.ascontiguousarray(np.concatenate([e for _, e, _ in rows], 0), dtype=np.float32)
        # The batch buffer is shared, so filling it is part of the critical section.
        with self._lock:
            b = self._batch
            C.memmove(b.embd, flat.ctypes.data, flat.nbytes)
            out_idx = []
            k = 0
            for seq, e, p0 in rows:
                for t in range(e.shape[0]):
                    b.pos[k] = p0 + t
                    b.n_seq_id[k] = 1
                    b.seq_id[k][0] = seq
                    b.logits[k] = 0
                    k += 1
                b.logits[k - 1] = 1
                out_idx.append(k - 1)
            b.n_tokens = n
            rc = self.L.llama_decode(self.ctx, b)
            if rc != 0:
                raise RuntimeError(f"llama_decode failed ({rc})")
            out = np.empty((len(rows), self.H), np.float32)
            for j, i in enumerate(out_idx):
                p = self.L.llama_get_embeddings_ith(self.ctx, i)
                C.memmove(out[j].ctypes.data, p, self.H * 4)
        return out

    def close(self) -> None:
        if getattr(self, "ctx", None):
            self.L.llama_batch_free(self._batch)
            self.L.llama_free(self.ctx)
            self.L.llama_model_free(self.model)
            self.ctx = self.model = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class StepBatcher:
    """Merges the backbone calls of concurrent streams into one ``llama_decode``.

    Each stream thread calls ``begin`` (prefill) / ``step`` / ``end``; a worker
    thread drains whatever is queued and decodes it together. Streams spend most
    of a frame in the CPU acoustic head, so their steps arrive staggered and the
    worker batches them with no added wait.
    """

    def __init__(self, bb: LlamaBackbone):
        self.bb = bb
        self._free = list(range(bb.n_seq_max))
        self._free_cv = threading.Condition()
        self._q: List[list] = []
        self._cv = threading.Condition()
        threading.Thread(target=self._run, name="llamacpp-batcher", daemon=True).start()

    def _submit(self, seq: int, embeds: np.ndarray, pos0: int) -> np.ndarray:
        item = [seq, embeds, pos0, threading.Event(), None, None]
        with self._cv:
            self._q.append(item)
            self._cv.notify()
        item[3].wait()
        if item[5] is not None:
            raise item[5]
        return item[4]

    def _run(self) -> None:
        while True:
            with self._cv:
                while not self._q:
                    self._cv.wait()
                take, n = [], 0
                while self._q and (not take or n + self._q[0][1].shape[0] <= self.bb.n_batch):
                    it = self._q.pop(0)
                    take.append(it)
                    n += it[1].shape[0]
            try:
                out = self.bb.decode([(it[0], it[1], it[2]) for it in take])
                for j, it in enumerate(take):
                    it[4] = out[j]
            except Exception as e:  # surface to every waiting stream
                for it in take:
                    it[5] = e
            for it in take:
                it[3].set()

    def begin(self, prompt_embeds: np.ndarray) -> Tuple[np.ndarray, int]:
        """Prefill (T, H) on a free sequence → (last hidden (H,), seq)."""
        with self._free_cv:
            while not self._free:
                self._free_cv.wait()
            seq = self._free.pop()
        try:
            self.bb.reset(seq)
            return self._submit(seq, prompt_embeds, 0), seq
        except BaseException:
            self.end(seq)
            raise

    def step(self, seq: int, embed: np.ndarray, pos: int) -> np.ndarray:
        return self._submit(seq, embed.reshape(1, -1), pos)

    def end(self, seq: int) -> None:
        self.bb.reset(seq)
        with self._free_cv:
            self._free.append(seq)
            self._free_cv.notify()


def convert_backbone_gguf(st_path: str, cfg_path: str, out_path: str, qtype: str = "q8_0") -> str:
    """Write the v3 Turbo semantic backbone (Qwen3) from ``model.safetensors`` as
    a llama.cpp ``qwen3`` GGUF. ``qtype``: q8_0 (default) | f16 | f32."""
    import json
    import struct

    try:
        import gguf
    except ImportError as e:
        raise ImportError("converting the backbone to GGUF needs the `gguf` package: "
                          "uv pip install gguf  (or set VIENEU_LLAMACPP_GGUF to a converted file)") from e

    with open(st_path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        base = 8 + n

        def tensor(name: str) -> np.ndarray:
            v = hdr[name]
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
                raise ValueError(f"{name}: unsupported dtype {v['dtype']}")
            return a.reshape(v["shape"])

        c = json.load(open(cfg_path, encoding="utf-8"))
        L, H = c["num_hidden_layers"], c["hidden_size"]
        emb = tensor("text_embeddings.weight")
        V = emb.shape[0]
        qt = {"q8_0": gguf.GGMLQuantizationType.Q8_0, "f16": gguf.GGMLQuantizationType.F16,
              "f32": gguf.GGMLQuantizationType.F32}[qtype]
        ft = {"q8_0": gguf.LlamaFileType.MOSTLY_Q8_0, "f16": gguf.LlamaFileType.MOSTLY_F16,
              "f32": gguf.LlamaFileType.ALL_F32}[qtype]

        tmp = out_path + ".tmp"
        w = gguf.GGUFWriter(tmp, "qwen3")
        w.add_name("vieneu-v3-turbo-backbone")
        w.add_context_length(c["max_position_embeddings"])
        w.add_embedding_length(H)
        w.add_block_count(L)
        w.add_feed_forward_length(c["intermediate_size"])
        w.add_head_count(c["num_attention_heads"])
        w.add_head_count_kv(c["num_key_value_heads"])
        w.add_key_length(c["head_dim"])
        w.add_value_length(c["head_dim"])
        w.add_rope_freq_base(c["rope_theta"])
        w.add_layer_norm_rms_eps(c["rms_norm_eps"])
        w.add_file_type(ft)
        # Placeholder vocab: the backbone is fed inputs_embeds, never token ids.
        w.add_tokenizer_model("llama")
        w.add_token_list([f"<t{i}>" for i in range(V)])
        w.add_token_scores([0.0] * V)
        w.add_token_types([1] * V)
        w.add_bos_token_id(c.get("bos_token_id", 1))
        w.add_eos_token_id(c.get("eos_token_id", 2))

        def put(name: str, a: np.ndarray) -> None:
            a = np.ascontiguousarray(a, dtype=np.float32)
            if a.ndim == 2 and qt != gguf.GGMLQuantizationType.F32:
                q = gguf.quants.quantize(a, qt)
                w.add_tensor(name, q, raw_shape=q.shape, raw_dtype=qt)
            else:
                w.add_tensor(name, a)

        put("token_embd.weight", emb)
        put("output.weight", emb)  # unused head (we read hidden states)
        put("output_norm.weight", tensor("semantic_backbone.norm.weight"))
        names = {"input_layernorm": "attn_norm", "post_attention_layernorm": "ffn_norm",
                 "self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k",
                 "self_attn.v_proj": "attn_v", "self_attn.o_proj": "attn_output",
                 "self_attn.q_norm": "attn_q_norm", "self_attn.k_norm": "attn_k_norm",
                 "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up", "mlp.down_proj": "ffn_down"}
        for i in range(L):
            for hf, gg in names.items():
                put(f"blk.{i}.{gg}.weight", tensor(f"semantic_backbone.layers.{i}.{hf}.weight"))
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()
    os.replace(tmp, out_path)
    return out_path
