# Strata-MiMo: engine port plan (decode first)

Goal: MiMo-V2.6-Flash-RL on one RTX PRO 6000 (96 GB) + 512 GB RAM, coding workload, 256K context with 8-bit KV,
MXFP4 experts, **at least 80 tok/s output**. Measurements behind the plan: [MIMO_BASELINE.md](MIMO_BASELINE.md)
(llama.cpp: 33 tok/s) and [MIMO_ROUTING.md](MIMO_ROUTING.md) (84.7% GPU hit rate from a static profile, 92% with a
per-layer LRU; CPU MXFP4 at ~100 GB/s).

## Approach

MiMo is a plain pre-norm transformer (sliding-window + global attention with sinks, top-8 MoE, one dense layer). Strata's
layer stack is all qwen4exp (Gated DeltaNet, QSA indexer, hyper-connections, PLE, shared expert, softmax top-10), so
this fork gets a **new `src/mimo/` layer stack and driver** and keeps Strata's kernels and expert machinery:

- Decode protocol: `Verifier` (`src/core/verify.cpp`): one CUDA graph per window, per layer a doorbell to the host,
  the CPU expert pool computing the misses while `native_expert_grouped` runs the GPU-resident experts, a device spin
  on a mapped host flag, combine. No per-layer launches.
- Reused: `native_mmvq.cu` (Q8_0 GEMV), `iq_kernels` (grouped experts, embedding rows), `native_moe_combine`,
  `sampler.cu`, the rope tables, `kv_q8`, `ExpertLayout` native mode, `pinned.cu`, `ExpertCache`, expert profiles
  (STRP), `ExpertDispatch`, the CPU `ExpertPool` (ggml-cpu traits, MXFP4 included), the `--serve` protocol.
- Not used: GDN, QSA, GR/hyper-connections, PLE/n-gram, draft vocab, the qwen4exp snapshot format.

## Steps (critical path to a first measured tok/s: 1-7)

| # | Step | New / changed | ~LOC |
| --- | --- | --- | ---: |
| 1 | Geometry from GGUF metadata (`mimo2.*`), GGUF-direct upload of the non-expert weights, `ExpertLayout` from tensor offsets, accept arch `mimo2` | `include/strata/mimo/geometry.hpp`, `src/mimo/model.cpp`, `gguf_reader.hpp` | 700 |
| 2 | CPU experts at n_embd 4096 / n_ff 2048: runtime H/NE instead of constants, `kNativeActBytes` >= 4352, `kNativeHBytes` >= 2176 | `expert_source.cpp`, `pool.*`, `expert_layout.cpp`, `cpu/expert.hpp`, `native_expert.hpp` | 200 |
| 3 | GPU MXFP4 (type 39) experts for T = 1..8 from ggml-cuda `vec_dot_mxfp4_q8_1`, own kernels and slot format ([MIMO_GPU_EXPERTS.md](MIMO_GPU_EXPERTS.md)) | `src/mimo/kernels/expert_mxfp4.cu` + parity test | 400 |
| 4 | Router: F32 GEMV, sigmoid, select on p + bias, top-8, weights = unbiased p normalized | `src/mimo/router.cu` | 300 |
| 5 | Attention: Q8_0 qkv, NEOX rope on 64 of 192 dims (two bases), int8 KV (global: linear; SWA: ring of 128 + window), split-K flash decode with sinks, value scale 0.707 | `src/mimo/attn.cu` + NumPy reference test | 1,000 |
| 6 | Window graph (T = 1..8) and driver: dense layer 0, router -> doorbell -> CPU misses in parallel with GPU hits -> combine; head, sampler; expert profile -> resident slots; `--serve` | `src/mimo/window.cpp`, `src/program/mimo_main.cpp` | 2,200 |
| 7 | Parity against llama.cpp: per-layer dumps (`attn_norm`, `Qcur`, `ffn_moe_topk`, `ffn_moe_out`, `l_out`, logits) on fixed tokens; all-CPU, all-GPU and profile splits; past the 128-token window and at long context; KL against `D:\mimo-kld-base.bin` | `tools/mimo_dump.cpp`, `tools/mimo_parity.py` | 400 |
| 8 | MTP (3 nextn layers, separate GGUF), prefill (batched GEMM + MXFP4 MMQ), conversation snapshot, server (eos 151645, template), setup | | 3,100 |

## Speed estimate, corrected

Within a layer, attention and the router have to finish before the CPU can start that layer's missed experts, so the
CPU time does not fully overlap with the GPU's work (the earlier estimate assumed it did). Per token at an 84.7% hit rate:
~3.8 ms of attention/dense/head on the GPU in series with ~7.7 ms of CPU misses at 100 GB/s, so about 12 ms,
**~80 tok/s**, which is just the target. What moves it up:

- Residency that adapts to the session (per-layer LRU measured at 92%): ~8 ms, ~120 tok/s.
- Faster CPU experts (130 GB/s seen with more experts in flight; AVX-512 multi-token kernels, large pages, prefetch).
- MTP: a 2-token window reads attention/dense weights once for both tokens; the misses roughly double.
- 6-bit attention (Q6_K) if its quality holds: 5-bit (Q5_K) measured mean KLD 0.044 and 93.96% same top token
  against the original on 49K tokens of coding answers (noise floor: 0.000000 / 100%), so 8-bit stays for now.
