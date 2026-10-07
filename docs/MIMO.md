# Strata-MiMo: MiMo-V2.6-Flash-RL on one RTX PRO 6000

Strata's engine ported to XiaomiMiMo's MiMo-V2.6-Flash-RL (309B total, 15B active, MXFP4 experts): the attention,
the dense layer, the routers and ~6,600 of the 12,032 experts on one 96 GB GPU, the rest of the experts in RAM and
computed by the CPU, MTP speculative decoding, behind Strata's own server (OpenAI and Anthropic APIs).  Coding work
was the target.  Every number below was measured on one RTX PRO 6000 Blackwell Max-Q (GPU 0), a Threadripper 9960X
and 512 GB of DDR5-6400, Windows 11.

## What it does

| | Strata-MiMo | llama.cpp `5ad1c5d`, best setting |
| --- | ---: | ---: |
| Output, short coding prompts | 117-125 tok/s | 33 tok/s |
| Output after a 19K-token code prompt (through the API) | 86.6 tok/s | - |
| Output after a 68K-token code prompt | 86.9 tok/s | - |
| Prompt reading | 3,300 tok/s (17,881 tokens in 5.5 s, 67,723 in 20.5 s) | 96-178 tok/s |
| First token after a 19,457-token prompt (API) | 6.2 s | - |
| Context | 256K (8-bit KV) | |

Quality: against llama.cpp on the same tokens the final logits agree on the top token at 98.4% of positions (KL mean
0.0026) and the perplexity is the same within noise (1.7942 vs 1.7896); see [MIMO_DECODER.md](MIMO_DECODER.md).

## Running it

Build (Windows, CUDA 12.9, VS 2022 Build Tools; the pinned ggml is fetched on the first configure):

```
cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 ^
      -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared
cmake --build build --target strata-mimo
```

The model: `ggml-org/MiMo-V2.6-Flash-RL-GGUF`, the MXFP4 files (167 GB) and `mtp-MiMo-V2.6-Flash-RL-Q8_0.gguf`
(2.4 GB).  The tokenizer for the server: `python tools/strata_tokenizer.py --gguf <shard 1> --out <dir>`.  The
expert profile: `python tools/mimo/make_profile.py <out.prof> <moe-trace file>` (the one used here came from
the model's own answers to 44 coding prompts).

A server config (JSON, as `serve/server.py --config` reads it):

```json
{
  "exe": "D:/repos/Strata-MiMo/build/strata-mimo.exe",
  "args": ["--native", "<...>/MiMo-V2.6-Flash-RL-MXFP4-00001-of-00002.gguf",
           "--mtp", "<...>/mtp-MiMo-V2.6-Flash-RL-Q8_0.gguf",
           "--expert-profile", "<...>/coding-answers.prof",
           "--max-context", "262144"],
  "cwd": "D:/repos/Strata-MiMo",
  "tokenizer": "<dir>/tokenizer",
  "model_name": "mimo-v2.6-flash-rl",
  "env": {"CUDA_VISIBLE_DEVICES": "0"}
}
```

then `python serve/server.py --engine strata --config <that file> --host 127.0.0.1 --port 8096`.  Start-up takes
about 70 s (the 150 GB pinned RAM copy of the experts).  The engine needs ~150 GB of free RAM and the whole GPU.

## Where things are

- [MIMO_PORT_PLAN.md](MIMO_PORT_PLAN.md): the plan and its steps.
- [MIMO_BASELINE.md](MIMO_BASELINE.md), [MIMO_ROUTING.md](MIMO_ROUTING.md): llama.cpp on this machine; expert routing on coding text.
- [MIMO_DECODER.md](MIMO_DECODER.md): the decoder (v0 to v2 + MTP), parity, the VRAM traps on Windows.
- [MIMO_GPU_EXPERTS.md](MIMO_GPU_EXPERTS.md), [MIMO_CPU_EXPERTS.md](MIMO_CPU_EXPERTS.md), [MIMO_ATTN.md](MIMO_ATTN.md),
  [MIMO_PREFILL_MOE.md](MIMO_PREFILL_MOE.md): the kernels and their measurements.
- Code: `src/mimo/` (model, decoder, prompt path, router, attention, experts), `src/program/mimo_serve.cpp` (the
  engine behind the server), `src/program/mimo_run.cpp` (benchmarks and dumps), `tools/mimo/` (traces, profiles,
  parity, benchmarks).

## Not done yet

- Residency that adapts DURING generation (today it follows the profile, and a long prompt's own routing).
- setup.py: downloading MiMo and writing the config; the start-up time.
- When the server is killed hard, the engine process can outlive it (seen on this machine; Strata's job object is
  meant to prevent that).
- llama.cpp's large-batch MXFP4 MoE path is 11-31% off the exact result on this machine (MIMO_DECODER.md): not
  reported upstream yet.
