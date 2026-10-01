# v3 Turbo on llama.cpp (Vulkan GPU / ggml CPU) with batched streams

v3 Turbo's semantic backbone is a stock **Qwen3** (12 layers, hidden 768, ~104 M
parameters). Without CUDA, the ONNX engine runs it on the CPU. llama.cpp runs the
same network on **any Vulkan GPU** (AMD Radeon including Polaris/RX 4xx–5xx, Intel
Arc/iGPU, NVIDIA), or on its ggml CPU kernels.

On top of that, a **frame scheduler** (`lite_scheduler.py`) steps every concurrent
stream in lockstep. Each frame round makes:

- **one** `llama_decode` for all streams, including the prefills of streams that
  just joined
- **one** pass of a batched acoustic-head graph for all streams (16 slots)

So the round cost no longer grows as 17 calls per stream. The MOSS codec, voice
cloning and the prompt build are unchanged.

## Setup

1. Download a llama.cpp release for your platform from
   <https://github.com/ggml-org/llama.cpp/releases>, e.g.
   `llama-<build>-bin-win-vulkan-x64.zip`. Unzip it anywhere. Verified with build
   `b11321`; the binding mirrors that build's `llama.h` structs.
2. Install the one-time converters: `uv pip install gguf onnx`
3. Run the API (or the SDK) with:

   ```bash
   VIENEU_BACKEND=onnx VIENEU_LLAMACPP_LIB=/path/to/llama-b11321-bin-win-vulkan-x64 \
   VIENEU_THREADS=2 uv run python -m apps.openai_speech
   ```

On first start, two files are written next to `update/model.safetensors` in the
Hugging Face cache, in about 3 s:

- `backbone-q8_0.gguf`, bit-identical to `llama-quantize q8_0`
- `acoustic-batched-int8-v1.onnx`

| Variable | Default | |
|---|---|---|
| `VIENEU_LLAMACPP_LIB` | unset (off) | directory holding `llama.dll` / `libllama.so` |
| `VIENEU_LLAMACPP_NGL` | `99` | layers on the GPU; `0` = ggml CPU |
| `VIENEU_LLAMACPP_TYPE` | `q8_0` | `q8_0` \| `f16` \| `f32` for the auto-conversion |
| `VIENEU_LLAMACPP_GGUF` | auto | use an already converted GGUF |
| `VIENEU_LLAMACPP_SEQS` | `16` | KV-cache sequences, i.e. the stream cap |
| `VIENEU_ACOUSTIC_BATCHED` | `1` | `0` = per-stream acoustic head (shipped graph), no scheduler |
| `VIENEU_ACOUSTIC_QUANT` | `int8` | `int8` \| `fp32` for the batched acoustic graph |
| `VIENEU_ACOUSTIC_THREADS` | cores/4 (max 4) | ONNX threads of the batched acoustic graph |
| `VIENEU_THREADS` | engine default | ONNX threads per call (codec); `2` is best with several streams |

With the scheduler the API defaults to `max_streams=6`; with
`VIENEU_ACOUSTIC_BATCHED=0` it defaults to 4.

## The batched acoustic head

The shipped `vieneu_acoustic_cached.onnx` accepts batch 1 only, and is
dynamic-int8 even in the "fp32" folder. `build_acoustic_onnx` writes the same
network (`AcousticDecoder.cached_step`) with a batch dimension, from the original
weights.

The default `int8` scheme uses:

- per-channel int8 weights, about 7 MB, which stay in the L3 cache. The fp32
  weights are 28 MB and are memory-bound.
- **per-row** dynamic activations. ONNX dynamic quantization uses one scale per
  tensor, which would make a stream's output depend on its batch neighbours. Here
  each row has its own scale, so a stream's output is **bit-identical** whatever
  it is batched with.
- 7-bit activations (zero point 64). x86 AVX2 U8S8 kernels sum u8·s8 pairs in
  int16, and full 8-bit values (255·127·2) overflow it.

Measured against the fp32 graph over a full 16-slot frame, the hidden-state error
was 12 % for this graph and 17 % for the shipped one.

Acoustic head CPU time per frame (2 threads):

| batch | 1 | 4 | 8 |
|---|---|---|---|
| shipped graph, one call per stream | 5.8 ms | 4 × 5.8 | 8 × 5.8 |
| batched int8 graph | 4.6 ms | 9.5 ms | 16.0 ms |

## Accuracy of the backbone

Teacher-forced against the ONNX fp32 backbone (same codes fed to both, 29 steps),
the relative error of the hidden state was:

- F32 GGUF: ≤ 0.05 %
- Q8_0 GGUF: ≤ 0.6 %

Greedy decoding is chaotic in both engines: 2·10⁻⁴ noise on pure ONNX already
changes most codes. So the right check is per-step agreement, not code identity.

## Measurements (Ryzen 5 3600 + Radeon RX 590 8 GB, Windows 10)

Backbone decode alone (`llama-batched-bench`, Q8_0, tokens/s):

| parallel streams | 1 | 4 | 8 | 16 |
|---|---|---|---|---|
| Vulkan (RX 590) | 87 | 292 | 504 | **751** |
| ggml CPU (6 threads) | 213 | 291 | 337 | — |

The audio runs at 12.5 frames/s. On Polaris a single Vulkan step has a latency
floor of about 10 ms, so the GPU only pays off once streams are batched. The
context uses one unified KV cache (`kv_unified`). With per-sequence caches, a
step for 8 streams took 44 ms instead of 14 ms.

End to end (`infer_stream`, N threads started together, `VIENEU_THREADS=2`; ×RT =
generation speed relative to real-time playback, higher is better):

| setup | streams | total ×RT | per stream ×RT |
|---|---|---|---|
| ONNX fp32 (no llama.cpp) | 1 | 1.07 | 1.07 |
| ONNX int8, one engine | 2 | 2.74 | 1.37 |
| llama.cpp Vulkan, per-stream acoustic head | 4 | 5.15 | 1.29 |
| llama.cpp Vulkan, per-stream acoustic head | 8 | 5.92 | 0.74 |
| **llama.cpp Vulkan + frame scheduler** | 1 | 2.22 | 2.22 |
| **llama.cpp Vulkan + frame scheduler** | 4 | 5.57 | 1.39 |
| **llama.cpp Vulkan + frame scheduler** | **6** | **6.86** | **1.14** |
| **llama.cpp Vulkan + frame scheduler** | 8 | 7.80 | 0.98 |

Generating frames alone, without decoding audio, the scheduler reaches 10× real
time at 8 streams and 15× at 16. What remains is the MOSS streaming codec. Its
ONNX graph takes batch 1 only and carries about 12 MB of attention state per
stream, costing about 7.5 ms of CPU per frame plus about 13 ms per call.

Two approaches were measured and dropped:

- **DirectML** (`onnxruntime-directml`) on the same card was slower than the CPU:
  RTF 3.4 with every graph on the GPU, i.e. slower than real time.
- **Splitting streams over two scheduler threads**, to overlap the GPU step with
  the CPU acoustic pass, lowered throughput because the CPU was already the limit.
