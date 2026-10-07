# MiMo attention (port step 5)

Decode attention for MiMo-V2.6-Flash: `include/strata/mimo/attn.hpp`, `src/mimo/attn.cu`, test and benchmark
`tests/cuda/mimo_attn_test.cpp` (`mimo_attn_test`, `mimo_attn_test --bench`). The math is llama.cpp's
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
