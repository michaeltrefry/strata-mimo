# MiMo attention (port step 5)

Decode attention for MiMo-V2.6-Flash (prompt processing: [below](#prompt-processing-prefill)):
`include/strata/mimo/attn.hpp`, `src/mimo/attn.cu`, test and benchmark `tests/cuda/mimo_attn_test.cpp`
(`mimo_attn_test`, `mimo_attn_test --bench`). The math is llama.cpp's
(`src/models/mimo2.cpp`): NEOX rope on dims 0..63 of the 192-dim q and k heads, scale 1/sqrt(192), GQA (64 query heads
over 4 KV heads on the 9 global layers, 8 on the 39 sliding-window layers), causal mask, window 128 (a key is masked
when pos_q - pos_k >= 128), and on SWA layers one sink logit per query head in the softmax denominator. Windows of
T = 1..8 consecutive query tokens (speculative verify).

## KV cache

int8 codes with one fp16 scale per 64 values per position and KV head (Strata's kv_q8 rule). K's first 64-value group
is exactly the roped part, so the roped and unroped dims get their own scales. 332 bytes per position and KV head.

| Layers | Per token | At 262,144 positions |
| --- | ---: | ---: |
| 9 global (4 KV heads, by position) | 11,952 B | 3.13 GB |
| 39 SWA (8 KV heads, ring of 256 slots) | fixed | 26.5 MB |
| split-K scratch (shared) | fixed | 12.3 MB |

A rejected speculative token needs no undo: the next window starts at the accepted length and overwrites it, and
attention never reads past pos0 + T. The 256-slot ring keeps every key a later window can read (it needs
n_swa + T - 1 = 135 slots).

## Accuracy (synthetic data, `mimo_attn_test`)

Errors are max |difference| / rms of the reference output, CPU reference in double precision.

| | Range over 21 cases (T = 1..8, 1 to 65,536 positions) |
| --- | --- |
| kernel vs the same int8 cache on the CPU (the kernel's own f16 roundings of Q, P, V) | 0.10% - 1.05% |
| int8 cache vs unquantized keys and values (this KV format) | 1.3% - 31% |
| llama.cpp q8_0 cache vs unquantized, same data | 1.3% - 21% |
| llama.cpp f16 cache vs unquantized, same data | 0.10% - 1.3% |

The synthetic queries are peaky on purpose (score standard deviation up to ~3.7), which magnifies the int8 error;
the format costs about what llama.cpp's q8_0 cache costs. Also checked: the sink identity out(sink) = out(no sink) x Z / (Z + e^sink)
(2.5e-4), speculative sequences with rejected tokens over 600 positions (the ring wraps twice), rmsnorm (4e-7) and the
residual add (bit-exact to ggml's scale-then-add).

## Speed (RTX PRO 6000 Blackwell Max-Q, 188 SMs, CUDA 12.9, `mimo_attn_test --bench`)

One layer, attention kernels only (split + combine), inside a CUDA graph, KV read from DRAM (each call reads KV the
previous calls did not touch). Best of 5 graph launches; the medians were within 5% of it.

| Layer | Positions | T = 1 | T = 2 | T = 4 | T = 8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| global | 4,096 | 11.8 us | 13.4 us | 17.7 us | 39.2 us |
| global | 32,768 | 39.4 us (1.11 TB/s) | 40.6 us | 57.6 us | 108.2 us |
| global | 131,072 | 126.2 us (1.38 TB/s) | 132.0 us | 206.7 us | 343.8 us |
| global | 262,144 | 251.5 us (1.38 TB/s) | 279.5 us | 427.3 us | 674.1 us |
| SWA | any (128 + T - 1 keys) | 8.3 us | 9.6 us | 9.1 us | 10.2 us |

Around it, per layer: rmsnorm 2.2 us, q/k/v split + rope + KV append 1.8 us, residual add 1.1 us (T = 1, graph).
So per token at T = 1, attention without the two GEMVs: ~0.93 ms at 32K positions, ~2.8 ms at 256K.

T >= 4 at long context is limited by instruction issue (8 warps per SM, ~670 instructions per 16-key slice), not by
memory or the tensor cores; the next step there is dequantizing each K/V tile once into shared memory for all the
query tiles. Short contexts and SWA layers are latency-bound (each kernel in the graph costs ~1 us on its own).

## Prompt processing (prefill)

`mimo_qkv_rope_append_prompt` + `mimo_attn_prompt` (same files; test and benchmark
`tests/cuda/mimo_attn_prompt_test.cpp`, `mimo_attn_prompt_test`, `mimo_attn_prompt_test --bench`): a chunk of P = 1 to
4,096 tokens at positions pos0 .. pos0 + P - 1, with earlier chunks or decode windows already in the cache. Same
cache, same rope and quantizer as decode: any mix of chunk sizes and decode windows writes the same int8 codes.

**The SWA ring.** The ring holds 256 positions, but the queries of a chunk longer than 129 tokens need more keys than
that (each one the 127 before it). So an SWA chunk goes through a linear staging buffer in the cache's format
(`AttnPromptScratch::swa`, shared by all SWA layers, 8 x (4,096 + 128) rows = 11.2 MB). The append first copies the
ring's last 128 positions before pos0 into the staging, then writes the chunk's k and v into the staging and its
last min(P, 256) positions into the ring. The attention reads only the staging. Afterwards the ring holds positions
>= pos0 + P - 256, which is what the next decode window or chunk needs. Global layers read and write their cache by
position.

**The attention kernel.** Flash attention on mma.sync (f16 inputs, f32 accumulation). A block takes 128 query rows of
one KV head (8 tokens x 16 query heads on a global layer, 16 x 8 on an SWA layer) and streams 64-key tiles. Each
tile's int8 codes are converted once into f16 tiles in shared memory, alongside the MMAs of the tile before.
q comes from the append as fp16; K becomes fp16(code x scale), whereas decode keeps the codes exact and scales the
f32 dot product. When the prior context is long and the blocks would not fill the GPU evenly, the keys are split
(split-K, merged by a second kernel). That needs `part_bytes` of partial buffer: with the default 64 MiB, a
2,048-token chunk runs unsplit; with 512 MiB it splits in 2, 4% faster (11.09 against 11.44 ms at 32K).

### Accuracy (synthetic data, `mimo_attn_prompt_test`)

Max |difference| / rms of the reference output, CPU reference in double precision on the cache contents and the
GPU's fp16 q (llama.cpp's formula, as in the decode test). 24 cases: P = 1, 7, 128, 129, 1,000, 2,048 at pos0 = 0 and
3,000 (prior context written by the decode path's append), global and SWA (sinks); up to 94 sampled tokens per case.

| | Result |
| --- | --- |
| kernel vs reference, SWA | 0.20% - 0.96% |
| kernel vs reference, global | 0.10% - 1.30% (1.2-1.3% at pos0 = 3,000) |
| fp16 q vs the CPU rope | 99.98-100% the same fp16 value; the rest within 2e-5 of the rms past the half ulp |
| chunk codes vs the CPU quantizer | at most 4 values off by one step per case (a value on a rounding boundary) |
| SWA ring after the chunk | its last 256 positions byte-identical to the staging, older slots untouched |

**Chunked consistency:** 300 tokens of decode-written context, then 2,048 tokens processed three ways:

| | SWA layer | Global layer |
| --- | ---: | ---: |
| 16 chunks of 128 vs one chunk of 2,048 | 0 (bit-identical) | 0.15% (split-K boundaries) |
| 256 decode windows of 8 vs one chunk of 2,048 | 1.06% | 1.78% |
| cache rows that differ (all positions the cache still holds) | 0 | 0 |

The decode comparison holds two independent sets of f16 roundings (q, P and V on both sides, K only in the prompt
kernel), each about as large as its own error against the reference.

### Speed (RTX PRO 6000 Blackwell Max-Q, 188 SMs, CUDA 12.9, `mimo_attn_prompt_test --bench`)

One layer, best of 5 runs (stream-timed, no graph). Append = q/k/v split + rope + int8 append from F32 qkv (SWA:
plus the ring copy). TFLOPS = 2 x 64 heads x (192 + 128) x the keys each query actually attends, over the attention
time.

| Layer | P | pos0 | Append | Attention | TFLOPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| global | 512 | 0 | 16 us | 44 us | 123 |
| global | 512 | 32,768 | 17 us | 3.12 ms | 222 |
| global | 512 | 131,072 | 17 us | 12.31 ms | 224 |
| global | 2,048 | 0 | 95 us | 390 us | 221 |
| global | 2,048 | 32,768 | 97 us | 11.44 ms | 248 |
| global | 2,048 | 131,072 | 97 us | 47.86 ms | 232 |
| global | 4,096 | 0 | 227 us | 1.54 ms | 223 |
| global | 4,096 | 32,768 | 224 us | 24.42 ms | 239 |
| global | 4,096 | 131,072 | 226 us | 94.29 ms | 237 |
| SWA | 512 | any | 21-23 us | 36-38 us | 66-73 |
| SWA | 2,048 | any | 109-114 us | 101-109 us | 99-104 |
| SWA | 4,096 | any | 247-249 us | 220-230 us | 93-96 |

**A 2,048-token chunk at pos0 = 0, all 48 layers (9 global + 39 SWA), append + attention: 12.6 ms.** At 128K of
prior context the 9 global layers take 9 x 48 ms = 431 ms per 2,048-token chunk; their attention grows linearly with
the context.

Where the time goes (P = 2,048 at 32K on a global layer, measured by switching parts off): the same loop without
loads and without the int8 conversion runs at ~300 TFLOPS (the f16 MMA peak measured on this card is 444); the loads
and the conversion cost about 1.2 ms each of the 11.4 ms. SWA layers do less useful work per tile: a block's 16
tokens read 143 keys in 3 tiles, 192 keys of work for the 128 each query sees. Skipping the n-tiles a warp cannot see
(predicated mma) was slower: 14.0 against 11.5 ms on the global case. The SWA append runs at 1.65 TB/s (122 MB of F32
qkv in, 50 MB of fp16 q out), close to the card's DRAM bandwidth; fusing it into the qkv GEMM's epilogue would remove
that traffic.
