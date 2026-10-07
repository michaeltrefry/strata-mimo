# MiMo-V2.6-Flash-RL: llama.cpp baseline

The speed Strata-MiMo has to beat. Measured 2026-10-07.

**Setup:** one RTX PRO 6000 Blackwell Max-Q (96 GB, GPU 0 only), AMD Ryzen Threadripper 9960X (24 cores, 24
threads used), 512 GB RAM, Windows 11. llama.cpp `5ad1c5d` built with CUDA 12.9 for sm_120, flash attention on.
Model: `ggml-org/MiMo-V2.6-Flash-RL-GGUF` revision `a5d1269`, MXFP4 experts (155.87 GiB), everything else Q8_0.

## Experts split between GPU and RAM (`llama-bench`, 2,048-token prompt, 128 generated tokens, 2 runs)

`-ncmoe N` keeps the experts of the first N of the 47 MoE layers in RAM; the rest of the model is on the GPU.

| MoE layers on the GPU | Prompt tok/s | Output tok/s |
| ---: | ---: | ---: |
| 0 | 95.99 | 18.38 |
| 17 | 138.44 | 28.04 |
| 21 | 148.42 | 31.72 |
| 23 | 160.59 | 31.63 |
| 25 | 178.33 | 35.86 |

25 layers is about the limit: a real server needs room for its KV cache.

## MTP speculative decoding (`llama-server`, 23 MoE layers on the GPU, 16K context, one slot)

Four prompts (code, explanation, story, C++ class), 512 generated tokens each, temperature 0, after one warm-up
request. MTP head: `mtp-MiMo-V2.6-Flash-RL-Q8_0.gguf`. Warm = prompts 2-4 (the first still pages experts in).

| Drafting | Output tok/s, all 4 | Warm prompts | Drafts accepted |
| --- | ---: | ---: | ---: |
| none | 29.41 | 31.8-33.4 | - |
| MTP, 1 token | **33.01** | 34.0-39.3 | 85.5% |
| MTP, 2 tokens | 30.25 | 29.3-36.4 | 62.4% |
| MTP, 3 tokens | 26.14 | 23.8-31.9 | 46.2% |

## What these numbers say

- With all experts in RAM, 5.0 GB of MXFP4 expert weights per token at 18.4 tok/s works out to about 92 GB/s of
  effective RAM bandwidth.
- Drafting more than one token makes llama.cpp slower: every extra token checked picks its own experts, and half of
  them are computed from RAM.
- llama.cpp places experts by whole layer. Strata-MiMo's target is at least 80 tok/s output with 4-bit experts, which
  needs the experts held on the GPU to serve about 75% of the routing choices (see the plan).
