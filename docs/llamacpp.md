# v3 Turbo backbone on llama.cpp (Vulkan GPU / ggml CPU)

v3 Turbo's semantic backbone is a stock **Qwen3** (12 layers, hidden 768, ~104 M
parameters). Without CUDA, the ONNX engine runs it on the CPU. llama.cpp runs the
same network on **any Vulkan GPU** (AMD Radeon including Polaris/RX 4xx–5xx, Intel
Arc/iGPU, NVIDIA), or on its ggml CPU kernels, and decodes the steps of every
concurrent stream **in one batched call**.

Only the backbone moves. The acoustic head, the MOSS codec, sampling and voice
cloning stay on ONNX Runtime, so the audio pipeline is unchanged.

## Setup

1. Download a llama.cpp release for your platform from
   <https://github.com/ggml-org/llama.cpp/releases>, e.g.
   `llama-<build>-bin-win-vulkan-x64.zip`. Unzip it anywhere. Verified with build
   `b11321`; the binding mirrors that build's `llama.h` structs.
2. Install the converter dependency, which is only needed once:
   `uv pip install gguf`
3. Run the API (or the SDK) with:

   ```bash
   VIENEU_BACKEND=onnx VIENEU_LLAMACPP_LIB=/path/to/llama-b11321-bin-win-vulkan-x64 \
   VIENEU_THREADS=2 VIENEU_MAX_STREAMS=4 uv run python -m apps.openai_speech
   ```

On first start the backbone is converted from `update/model.safetensors` to
`backbone-q8_0.gguf`, next to it in the Hugging Face cache. This takes about 2 s,
and the output is bit-identical to `llama-quantize q8_0`.

| Variable | Default | |
|---|---|---|
| `VIENEU_LLAMACPP_LIB` | unset (off) | directory holding `llama.dll` / `libllama.so` |
| `VIENEU_LLAMACPP_NGL` | `99` | layers on the GPU; `0` = ggml CPU |
| `VIENEU_LLAMACPP_TYPE` | `q8_0` | `q8_0` \| `f16` \| `f32` for the auto-conversion |
| `VIENEU_LLAMACPP_GGUF` | auto | use an already converted GGUF |
| `VIENEU_LLAMACPP_SEQS` | `16` | KV-cache sequences, i.e. the stream cap |
| `VIENEU_THREADS` | engine default | ONNX threads per call; `2` is best with several streams |

With the llama.cpp backbone the API defaults to `max_streams=4`.

## Accuracy

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
floor of about 10 ms, so the GPU only pays off once streams are batched.

End to end (`infer_stream`, N threads started together; ×RT = generation speed relative to real-time playback, higher is better):

| setup | streams | total ×RT | per stream ×RT |
|---|---|---|---|
| ONNX fp32 (previous default) | 1 | 1.07 | 1.07 |
| ONNX int8, one engine | 2 | 2.74 | 1.37 |
| llama.cpp CPU | 1 | 2.91 | 2.91 |
| **llama.cpp Vulkan, `VIENEU_THREADS=2`** | **4** | **5.15** | **1.29** |
| llama.cpp Vulkan, `VIENEU_THREADS=2` | 8 | 5.92 | 0.74 |

Past 4 streams the limit is the CPU side: the acoustic head (16 small ONNX calls
per frame), the streaming codec and Python. The backbone is no longer the limit.

For comparison, DirectML (`onnxruntime-directml`) on the same card was slower than
the CPU (RTF 3.4 with every graph on the GPU, i.e. slower than real time). That path was
dropped.
