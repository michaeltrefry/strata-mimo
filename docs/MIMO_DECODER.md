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

- The CPU's slow layers (the tail above) and the CPU share itself: residency that adapts to the session (an LRU of
  the same size served 92% in the routing study).
- MTP speculative decoding: a 2-token window reads the attention and dense weights once for both tokens.
- Fewer small kernels around attention and the router.
