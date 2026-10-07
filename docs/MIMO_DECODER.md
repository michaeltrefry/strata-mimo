# MiMo-V2.6-Flash decoder: where it stands

Measured 2026-10-07 on one RTX PRO 6000 Blackwell Max-Q (96 GB, GPU 0), Threadripper 9960X (24 cores), 4 x 128 GB
DDR5-6400, Windows 11; MXFP4 GGUF from ggml-org; 32K context, 8-bit KV; the 311-token coding prompt of
`E:\strata-mimo\ref300`, then greedy generation. `strata-mimo-run` (src/program/mimo_run.cpp).

## Speed

| Version | What it does | Output tok/s | Prompt tok/s |
| --- | --- | ---: | ---: |
| llama.cpp `5ad1c5d` (best: 23 MoE layers on the GPU, MTP 1 token) | | 33.0 | - |
| v0 | every expert on the GPU; misses copied from the GGUF mapping per window | 1.05 | 1.6 |
| v1 | misses on the CPU (CpuExperts) at the same time as the GPU's resident experts; one host round trip per layer | 47.7 | 168 |
| v2 | one CUDA graph per window; the CPU handoff inside it (doorbell + mapped flag) | 70-76 | 188 |
| v2, slots packed | expert slots allocated 64 at a time (see below) | 87.2 | 188 |
| v2 + MTP | 2-token verify windows, drafts from the MTP GGUF's first layer | **124.9** | 188 |

**Fresh prompts at 256K context** (not in the profile's text; 6,672 resident experts; MTP; 1,024 generated tokens
each):

| Prompt | Output tok/s | Resident | Drafts accepted |
| --- | ---: | ---: | ---: |
| Rust lock-free SPSC ring buffer with tests | 117.1 | 88.3% | 91.2% |
| Finding and fixing a FastAPI memory leak (more prose) | 100.7 | 86.4% | 73.4% |
| Dijkstra with a binary heap in TypeScript | 105.3 | 83.8% | 95.4% |

**Two VRAM placement traps on Windows** (WDDM lets an allocation past the VRAM succeed in system memory, where the
GPU reads it at PCIe speed): one cudaMalloc per 12.75 MiB expert is rounded up to 14 MiB of 2 MiB pages, which
pushed about 8.6 GiB of "resident" experts into system memory (fixed: slots in chunks of 64); and the MTP layer,
loaded after the expert budget was taken, ran its GEMVs at 46 GB/s (fixed: load it first).  `strata-mimo-run`
prints the VRAM left after setup (1.5 GiB by design).

With MTP the plain-greedy and speculative outputs agree until a near-tie (on the test prompt: 121 tokens, then a
line wrap in a comment comes out one way or the other); speculative output is always the main model's own pick
in a 2-token window.

6,903 resident experts (82 GiB, the most used in the coding-answers profile); 89.7% of the routed experts in
generation are resident, the CPU computes the rest in 4.2 ms per token.

## Correctness

Against llama.cpp on the same 311 tokens (reference dumped with 8-token ubatches, see below): the final logits have
the same top-1 token at 98.4% of positions, KL mean 0.0026 (max 0.06); next-token perplexity 1.7942 vs 1.7896
(difference 0.0025 +- 0.004 per token).  The MoE output of layer 1 is 0.7-1.5% from an exact float computation of
the official formula (Xiaomi's modeling_mimo_v2.py: sigmoid gating, normalized top-8, plain SiLU MLP,
routed_scaling_factor null), the cost of 8-bit activations.

llama.cpp `5ad1c5d`'s MoE for MXFP4 is 11-31% off that exact result when it runs a large batch (311 tokens, as in
prompt processing) on this machine, on the CPU and with experts on the GPU alike, and 0.9% off at 4 tokens: the
reference dumps use `-ub 8`.

## Where a v2 token's time goes (T = 1)

From Nsight Systems (`--cuda-graph-trace=node`), medians per MoE layer: attention + router 98 us (qkv GEMV 41 us,
attn_output 23 us, the rest small kernels), the GPU's resident experts 71 us (the kernel's own benchmark: 71 us for
8 experts), then the GPU waits for the CPU's share: median 39 us, mean 59 us, p90 154 us, p99 333 us - layers over
200 us hold 40% of the waiting.  With the CPU work replaced by nothing the graph runs at 84.8 tok/s; by an 89 us
busy wait (the CPU's mean per layer) at 77.3 tok/s.

## Next

- Prompt processing: 188 tok/s comes from running the prompt through 8-token decode windows; long prompts (a
  50K-token coding context would take over 4 minutes) need the batched prefill path (GEMM + MXFP4 MMQ).  Its MoE part
  is done: [MIMO_PREFILL_MOE.md](MIMO_PREFILL_MOE.md) (layer-major over slices of up to 32K tokens: 3,061-8,107 tok/s
  for the experts at S = 4K-32K; compute-bound from ~8K).
- The server (serve/), setup, and an expert profile from the user's own sessions.
- The CPU's slow layers and an adaptive expert cache (LRU measured at 92% vs 84-90% static).
