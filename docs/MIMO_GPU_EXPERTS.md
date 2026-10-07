# MiMo-V2.6-Flash: the GPU-resident MXFP4 experts

Step 3 of [MIMO_PORT_PLAN.md](MIMO_PORT_PLAN.md): the experts that sit in VRAM slots, computed on the GPU for a
window of 1 to 8 tokens. Code: `src/mimo/kernels/expert_mxfp4.cu`, `include/strata/mimo/expert_mxfp4.hpp`.
Test and benchmark: `tests/mimo/expert_mxfp4_test.cpp` (`mimo_expert_mxfp4_test`); the dense Q8_0 numbers below:
`tests/mimo/dense_q8_0_bench.cu` (`mimo_dense_q8_0_bench`).

Measured 2026-10-07 on GPU 0: RTX PRO 6000 Blackwell Max-Q (96 GB GDDR7), driver 596.36, CUDA 12.9, sm_120,
Windows 11. Another agent's jobs shared the GPU at times; tables give the best and the spread of repeated runs.

## What a call does

`mxfp4_experts(slot_ptr, slots, weights, x, out, T, K, scratch, stream)` computes, for each token t and each of its
K routed pairs whose slot is resident (`slots[t][k] >= 0`), `out[t] += weights[t][k] * down(silu(gate x) * up x)`.
Pairs with slot -1 are skipped; the CPU adds their share to the same `out` later. Everything per token is read on
the device and the launch shapes depend only on T and K, so one call per layer can be captured in the window's CUDA
graph (checked: a replay with rewritten slot / weight / x buffers is bitwise equal to a direct call).

Three kernels:

| Kernel | Grid | Work |
| --- | --- | --- |
| prepare | T blocks | x of each token to q8_1 (ggml's `quantize_q8_1`) |
| gate_up | 64 x T*K blocks | block (j, e): rows 32j..32j+31 of gate and up of entry e's expert, for every token that picked it (the first entry with a slot does the work, the others return); SiLU * up; those 32 values are one q8_1 block of h |
| down | 2,048 blocks | 2 output rows per block; the 8 warps take the window's distinct experts in turn; weight * down(h) summed per token in shared memory, added to `out` in a fixed order |

An expert shared by several tokens of a window is read once. There are no atomics: the same input gives the same
bits. gate_up and down are launched with programmatic dependent launch: gate_up finds its expert while prepare runs,
and down builds the window's grouping and loads its first weight rows while gate_up finishes. A token that lists
the same slot twice keeps only the first.

Scratch (`mxfp4_experts_scratch_bytes`): x as q8_1 (4,608 B per token), h as q8_1 per (token, k) pair; 184,320 B
at T = 8, K = 8.

## Slot format

A slot is one expert, 13,369,344 bytes as in the GGUF, but re-laid out:

    [gate codes | up codes | down codes | gate scales | up scales | down scales]

- codes: each row's 16-byte `qs` blocks in order (gate/up rows 2,048 B, down rows 1,024 B), every row 16-byte
  aligned, so a lane reads whole blocks with 128-bit loads.
- scales: each row's E8M0 bytes permuted so that one 32-bit (gate/up) or 16-bit (down) load gives a lane the
  scales of its blocks (lane i of a warp handles blocks i, i + 32, ...).

`mxfp4_slot_from_gguf` builds it on the host (1.02 ms per expert on one thread, from the mapped file: a slot filled
through a staging buffer pays this instead of a memcpy), `mxfp4_slot_relayout` on the device (13.7 us per expert,
for experts copied raw). The GGUF and the CPU path that reads it are unchanged.

Why re-lay out: in the GGUF a block is 17 bytes (scale byte + 16 code bytes), so blocks start at every byte offset
and a lane can only assemble its codes from byte or 16-bit loads (what ggml's `get_int_b1` does), several
instructions per 4 bytes. The re-layout costs nothing at run time and makes every weight load a coalesced 128-bit
load. It was chosen on that reasoning, not on an A/B measurement of a GGUF-layout kernel.

## Accuracy (16 real experts of layers 1, 24 and 47, random activations)

Two CPU references on ggml's own MXFP4 dequantizer (`ggml_get_type_traits(GGML_TYPE_MXFP4)->to_float`), T = 1..8,
every token with 8 resident pairs and with a random 0..8 of them resident; error = max |gpu - ref| / max |ref| per
token:

| Reference | Worst over 16 windows | Typical |
| --- | ---: | ---: |
| same arithmetic in double (x and h quantized to q8_1 as ggml does) | 1.16e-4 | 1e-6 |
| plain float (no activation quantization) | 3.1e-2 | 1-2e-2 |

The first is the kernel check (gate 1e-3): its outliers are a q8_1 value of h rounding the other way after a sum
in another order. The second is what q8_1 activations cost, the same as llama.cpp's CUDA path.

## Speed

One call = one layer's resident experts. 47 calls with different experts in one graph, cycling over 192 slots
(2.5 GB, so nothing is still in the 128 MB L2), 30 repeats; GB/s = distinct experts x 13,369,344 B / time. For
T > 1 each token picks its 8 from a pool (T = 2: 28 slots, 13.7 distinct of 16, as two consecutive tokens on coding
text in [MIMO_ROUTING.md](MIMO_ROUTING.md); T = 4: 40; T = 8: 64). A plain streaming read of 1 GiB on this card
reaches 1,627 GB/s (`mimo_dense_q8_0_bench`, first line).

| Case | Experts read | Best us | Median us | Worst us | Best GB/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| T=1, 8 resident | 8.0 | 70.9 | 71.2 | 86.4 | 1,508 |
| T=1, 6 resident | 6.0 | 56.7 | 57.0 | 69.1 | 1,414 |
| T=2, 8 resident | 13.7 | 124.0 | 127.6 | 141.9 | 1,477 |
| T=2, 6 resident | 10.8 | 102.0 | 104.3 | 119.1 | 1,411 |
| T=4, 8 resident | 23.6 | 211.2 | 221.8 | 236.4 | 1,491 |
| T=4, 6 resident | 19.1 | 172.6 | 180.0 | 194.2 | 1,477 |
| T=8, 8 resident | 42.5 | 378.7 | 418.5 | 425.3 | 1,500 |

What moved the T=1, 8-resident case (same benchmark):

| Change | us |
| --- | ---: |
| first version: grouping and x quantization in every gate_up block, 163 registers (1 block per SM) | 79.1 |
| 80 registers (3 blocks per SM), weights decoded per use | 83.5 |
| grouping in a separate kernel with shuffles in rolled loops (5 us for one warp) | 92-96 |
| grouping by ballots / unrolled shared-memory loops, gate_up started with programmatic dependent launch | 75.2 |
| gate_up finds its expert itself from `slots` (no wait for prepare), down groups for itself and prefetches before waiting, accumulators sized by T (no spills) | 70.9 |

Tried and not kept: 128-thread gate_up blocks (72.5), 1 or 4 down rows per block (72.4 / 84.0, 4 rows spill),
L2 256-byte prefetch hint (71.6), loading gate_up's first row before waiting for prepare (71.9, spills).

## Strata's Q8_0 MMVQ on MiMo's dense matrices

`native_q8_0_mmvq` (llama.cpp's MMVQ, `src/kernels/cuda/native_mmvq.cu`), synthetic Q8_0 weights, each shape
cycled over >= 512 MB of copies, 32 calls per graph, 20 repeats (`mimo_dense_q8_0_bench`). Best us / best GB/s:

| Shape | MB | T=1 | T=2 | T=3 | T=4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| attn_qkv global 4096 -> 13568 | 59.0 | 36.3 / 1,625 | 36.6 / 1,615 | 37.4 / 1,577 | 38.3 / 1,541 |
| attn_qkv SWA 4096 -> 14848 | 64.6 | 41.1 / 1,571 | 41.2 / 1,570 | 41.8 / 1,547 | 43.1 / 1,501 |
| attn_output 8192 -> 4096 | 35.7 | 23.8 / 1,500 | 23.9 / 1,494 | 25.2 / 1,413 | 25.8 / 1,380 |
| dense ffn gate or up 4096 -> 16384 | 71.3 | 45.1 / 1,580 | 45.2 / 1,577 | 45.8 / 1,557 | 47.5 / 1,503 |
| dense ffn down 16384 -> 4096 | 71.3 | 46.1 / 1,546 | 46.4 / 1,536 | 49.7 / 1,435 | 52.5 / 1,358 |
| output 4096 -> 152576 | 664.0 | 405.3 / 1,639 | 444.7 / 1,493 | 473.5 / 1,402 | 497.7 / 1,334 |

At T = 1 and 2 it runs at 92-101% of the streaming read; it needs no work for MiMo. At T = 3-4 the long-row shapes
and the head drop to 1,330-1,440 GB/s: one block per row re-reads every column's q8_1 activation for every row, so
at T = 4 the activation traffic through L1 is ~4x the weight bytes. Several rows per block sharing the activation
loads would fix it; it matters only for 4-token verify windows (and the head is read once per window).

## Not done here

- The router (step 4) should hand over `slots` and `weights` directly; the slot table comes from the expert cache.
- A GGUF-layout kernel for an A/B of the re-layout (reasoned above, not measured).
- HIP: the kernels use inline PTX loads and programmatic dependent launch; CUDA only for now.
