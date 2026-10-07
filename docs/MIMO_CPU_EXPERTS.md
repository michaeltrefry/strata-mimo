# MiMo-V2.6-Flash: the CPU's experts (port plan step 2)

The routed experts that are not resident on the GPU are computed on the CPU by Strata's `ExpertPool`, reading the
MXFP4 experts where they are (the GGUF mapping, or a RAM copy of a layer with the same layout). Measured 2026-10-07 on
a Threadripper 9960X (24 cores, Zen 5, AVX-512), 4 x 128 GB DDR5-6400 (205 GB/s peak), Windows 11, with other jobs
running on the machine at the same time (so every number is the best of 3 runs, with the spread).

## What there is

| Piece | File |
| --- | --- |
| `CpuExperts`: one call per MoE layer, F32 hidden states (or Q8_0) + expert ids + router weights in, F32 weighted sums out | `include/strata/mimo/cpu_experts.hpp`, `src/mimo/cpu_experts.cpp` |
| `ExpertPool::run_moe`: two phases per layer (gate/up rows with the intermediate quantized by the task that computed it; then down rows, each task owning a range of output rows and summing every expert into it) | `include/strata/kernels/cpu/pool.hpp`, `src/kernels/cpu/pool.cpp` |
| MXFP4 x Q8_0 rows, AVX-512, 1-8 tokens per weight row, bitwise ggml's `ggml_vec_dot_mxfp4_q8_0` | `include/strata/kernels/cpu/mxfp4_avx512.hpp`, `src/kernels/cpu/mxfp4_avx512.cpp` |
| Runtime n_embd / n_ff in the native expert path (buffers sized for 4096 / 2048; matrices given by pointer, not one blob) | `native_expert.*`, `pool.*`, `expert_source.cpp` |
| Parity test, benchmark | `tests/mimo/cpu_experts_parity.cpp`, `tools/mimo/cpu_experts_bench.cpp` |

No repack: an expert is `ExpertSlices::matrix(m, e)` (gate, up, down in three GGUF tensors). `ExpertLayout` (one blob
per expert) is not used for MiMo; `CpuExperts` keeps the per-layer `ExpertSlices` instead.

## Parity

`mimo_cpu_experts_parity <shard>`: layers 1, 24, 47; windows of T = 1..4 tokens, each token with its own random 1-8
experts (later tokens reuse some of token 0's) and normalized random weights, x ~ N(0, 1); 90 token outputs.

| Check | Result |
| --- | --- |
| AVX-512 kernel rows vs `ggml_vec_dot_mxfp4_q8_0`, 1-8 tokens, 64,800 rows | 0 not bitwise equal |
| Pool vs float reference with the pool's Q8_0 activations (x and the intermediate through Q8_0, double sums) | max relative L2 error 5.6e-4 |
| Pool vs the float reference (MXFP4 dequantized with ggml's `to_float`, F32 x, double sums) | max relative L2 error 3.9e-2, max abs error / max abs ref 6.6e-2 |

The float gap is the Q8_0 activation contract, which is ggml-cpu's for MXFP4 experts: putting only x through Q8_0
moves the float reference itself by up to 3.8e-2 on these random inputs. The 5.6e-4 is the intermediate rounding to a
different Q8_0 value in a few places, because the reference forms it in double.

## Speed

`mimo_cpu_experts_bench <shard>`: the experts of 24 MoE layers (76.5 GiB) copied into `VirtualAlloc` memory (4 KiB
pages, not locked); per window 47 `run_layer` calls (layer l reads the copy of layer (l - 1) % 24 + 1), M random
experts per token, F32 in, F32 sums out, activation quantization included. T = 2: the second token repeats
round(M x 2.2 / 8) of the first token's experts (MIMO_ROUTING.md: 13.6-13.9 distinct of 16). GB/s counts each
distinct expert once. Pool: 23 workers, one per physical core, plus the calling thread.

Ceiling: every core streaming a disjoint slice of the same copy with plain loads reads **162.5 GB/s** (24 threads;
128.6 with 16, 70.5 with 8).

### Before and after (ms per window: the range of the best-of-3 over interleaved runs)

| Experts per layer | T | Distinct | Before: ggml dot | After: AVX-512 kernel | ms per token after | GB/s after |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 1 | 4.69-4.98 | 4.76-4.98 | 4.8-5.0 | 126-132 |
| 2 | 1 | 2 | 8.83-9.52 | 9.03-9.28 | 9.0-9.3 | 135-139 |
| 3 | 1 | 3 | 12.99-14.17 | 13.20-13.75 | 13.2-13.8 | 137-143 |
| 4 | 1 | 4 | 17.18-19.01 | 17.17-18.45 | 17.2-18.5 | 136-146 |
| 8 | 1 | 8 | 33.70-39.21 | 33.41-36.80 | 33.4-36.8 | 137-150 |
| 1 | 2 | 2 | 8.91-9.56 | 8.88-9.42 | 4.4-4.7 | 133-142 |
| 2 | 2 | 3 | 13.24-14.47 | 13.12-14.13 | 6.6-7.1 | 133-144 |
| 4 | 2 | 7 | 29.59-32.17 | 29.56-32.71 | 14.8-16.4 | 134-149 |
| 8 | 2 | 14 | 58.37-63.93 | 58.10-62.73 | 29.1-31.4 | 140-151 |

"Before" is already the engine's pool (the standalone `tools/mimo/mxfp4-cpu-bench.cpp` with a static row split and
spin barriers measured 98-111 GB/s, MIMO_ROUTING.md); the pool's dynamic claims and the host working in the drain
are what moved it to 130-150 GB/s, 80-93% of the 162.5 GB/s plain-read ceiling. At this thread count the dot product
is not the limit: the kernel changes nothing within the noise. At an 85% GPU hit rate (about 1.2 misses per layer
per token) this is about 5.5-6 ms of CPU time per token, against the 7.7 ms the port plan assumed at 100 GB/s.

### Each change, measured (ms per window, best of 3; M = experts per layer)

| Change | M=1 T=1 | M=4 T=1 | M=8 T=1 | M=4 T=2 | M=8 T=2 | Kept |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| ggml dot, pool protocol (park waits), 3 tasks per thread | 4.69 | 17.18 | 33.70 | 29.59 | 58.37 | baseline |
| AVX-512 kernel | 4.76-5.02 | 17.17-18.68 | 33.41-37.33 | 29.56-33.04 | 58.10-64.45 | yes (bitwise ggml; faster when compute-bound, below) |
| no park waits around the phases (epoch-tagged claims make them redundant) | 4.67-4.96 | 17.15-17.27 | 33.41-33.93 | 29.56-31.40 | 58.10-62.01 | opt-in only: no measurable gain |
| 2 tasks per thread (vs 3) | 4.95 | 18.13 | 36.01 | 34.31 | 68.24 | no |
| 1 task per thread | 5.26 | 18.55 | 36.43 | 35.53 | 71.47 | no |
| 4 tasks per thread | 5.04 | 17.79 | 35.21 | 31.29 | 67.21 | no |
| software prefetch 512 B ahead | 4.75 | 17.79 | 35.38 | 32.15 | 63.80 | no (default 0) |
| software prefetch 1 KiB / 4 KiB | 5.15 / 7.06 | 23.40 / 27.21 | 41.19 / 42.58 | 35.00 / 54.20 | 99.52 / 85.80 | no |
| workers unpinned | 5.60 | 20.69 | 41.46 | 40.10 | 73.83 | no (one pinned worker per physical core stays) |
| 47 workers (SMT siblings too) | 10.02 | 24.24 | 39.63 | 35.30 | 81.34 | no |
| large pages | - | - | - | - | - | not measurable: this account lacks SeLockMemoryPrivilege |

The prefetch rows and the unpinned/SMT rows ran while other jobs loaded the machine (their spread was 20-50%), so
they are "not better", not precise. The pool's defaults are the kept column: AVX-512 kernel, park waits on, 3 tasks
per thread, no prefetch, one pinned worker per physical core.

### Where the kernel matters: tokens that share experts, or few cores

Windows whose tokens all pick the same M experts (`--share 1`), ms per window:

| Threads | M | T | ggml dot | AVX-512 kernel | Speed-up |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 24 | 8 | 2 | 36.68 | 34.45 | 1.06x |
| 24 | 8 | 4 | 54.74 | 40.23 | 1.36x |
| 24 | 8 | 8 | 103.73 | 71.38 | 1.45x |
| 24 | 1 | 8 | 13.76 | 7.00 | 1.97x |
| 2 | 1 | 1 | 16.04 | 13.53 | 1.19x |
| 2 | 1 | 2 | 32.23 | 17.17 | 1.88x |
| 2 | 1 | 8 | 124.71 | 50.62 | 2.46x |

The kernel decodes a pair of blocks once for all tokens of the window; ggml's dot decodes them again per token. The
scales are kept one float per block (not pre-broadcast) so that 8 tokens' activations stay in L1: pre-broadcast
scales made T = 8 take 92.4 ms at 2 threads (now 50.6).

## The API for step 6 (the decode driver)

```cpp
strata::kernels::cpu::ExpertPool pool;            // once: 23 workers on physical cores 1..23
strata::mimo::CpuExperts cpu_experts(model, pool);  // once: per-layer formats, experts read from the GGUF mapping
cpu_experts.copy_layer(l, ram, 16);                // optional, per layer: read from a RAM copy instead

// per MoE layer of a window of n_tok <= 8 tokens:
cpu_experts.run_layer(l, n_tok, x, 8, ids, w, out);
//   x   [n_tok][4096] F32: the layer's ffn_norm output (what the router saw)
//   ids [n_tok][8]  the router's top-8, with every expert the GPU computes replaced by -1
//   w   [n_tok][8]  the router weights (normalized p of the picked experts, MIMO geometry.hpp)
//   out [n_tok][4096] F32: sum of w * expert(x) over the ids >= 0 (zeros if none); overwritten
// run_layer_q(l, n_tok, act, stride, 8, ids, w, out) takes x already as Q8_0 rows (act_bytes() = 4352 each).
```

`run_layer` blocks until done and the calling thread works in the pool (it should be the host thread pinned to
physical core 0, `pin_current_thread(detect_cpu_topology(true).host_core)`). Experts two tokens share are read once.

Signalling: reuse the Verifier's protocol (`src/core/verify.cpp`, `Verifier::run` and the `post` lambda), with one
change: the CPU returns one summed row per token, not one row per (token, slot).

1. GPU, per MoE layer: router, then `doorbell_publish(x, ids, w, ...)` copies x, ids and w into mapped host memory and
   raises the sequence word (`ring = layer step + 1`).
2. Host: spins on the sequence word (`_mm_pause`, with the 20 s / `cudaStreamQuery` guards of `Verifier::run`), masks
   the resident experts to -1 from its copy of the residency table, calls `run_layer` with `out` pointing at the
   mapped per-token rows, then `_mm_sfence()` and stores `ring` to the mapped flag (`h_flag_`).
3. GPU, meanwhile: its resident experts (`native_expert_grouped`), then `wait_flag_ge(m_flag, ring)`,
   `copy_from_mapped` of n_tok x 4096 floats (16 KiB per token), and the combine adds it to the GPU's weighted sum.

So `PoolMultiFn` becomes `(user, x, ids, w, n_tok, k, out_sum, layer)`; the weights are applied on the CPU and the
GPU's combine reads n_tok rows instead of n_tok x 8. The GPU can mask the hits itself and publish the masked ids,
which spares the host the residency lookup.

## Not resolved

- Large pages: `VirtualAlloc(MEM_LARGE_PAGES)` needs SeLockMemoryPrivilege ("Lock pages in memory" in the local
  security policy, then a new logon); this account does not have it, so the effect is unmeasured. The copy was not
  locked either (`lock_resident` would keep Windows from trimming it under memory pressure).
- Reading the experts in place from the GGUF mapping works (the parity test does) but its speed was not measured: the
  benchmark used a RAM copy. File-backed pages of a 167 GB file can be evicted; a copy (or a locked mapping) cannot.
- About 10% of the plain-read ceiling is lost at one expert per layer (126-134 vs 162.5 GB/s): two barriers per layer
  and the tail of each phase. Overlapping a layer's down phase with the next expert's gate/up would need the work
  per expert to be split differently.
- The Qwen paths were rebuilt but not re-run (`expert_source.cpp` now takes a native layer's width from its format;
  the Q2_0 path still uses the compiled 2560).
