# MiMo-V2.6-Flash: the routed experts of a prompt chunk (prefill)

Part of step 8 of [MIMO_PORT_PLAN.md](MIMO_PORT_PLAN.md): the MoE part of prompt processing for a chunk of P = 1 to
4,096 tokens, one layer per call. Code: `include/strata/mimo/prefill_moe.hpp`, `src/mimo/prefill_moe.cu`. Test:
`tests/mimo/prefill_moe_test.cpp` (`mimo_prefill_moe_test`); benchmark: `tools/mimo/prefill_moe_bench.cpp`
(`mimo_prefill_moe_bench`); pinned-memory probe: `tools/mimo/pinned_probe.cpp` (`mimo_pinned_probe`).

Measured 2026-10-07 on GPU 0: RTX PRO 6000 Blackwell Max-Q (96 GB, sm_120, PCIe 5.0 x16), CUDA 12.9, Windows 11;
Threadripper 9960X, 4 x 128 GB DDR5-6400; the MXFP4 GGUF from ggml-org. Nothing else ran on GPU 0.

## What a call does

`PrefillMoe::run_layer` computes, for one MoE layer, `out[t] = sum over the 8 picks of w * down(silu(gate x) * up x)`
for every token of the chunk:

1. **Grouping on the device.** A histogram of the P x 8 picks; the 256 counts go to the host (the one host sync per
   layer), which orders the used experts (resident ones first, then the ones in RAM) and splits them into groups. A
   placement kernel then lays the picks out in that order, stable (pick order within an expert, no atomics), and
   writes each pick's row. x is quantized once in that row order.
2. **Copies.** The used experts that are not resident are copied from RAM (GGUF layout, gate | up | down, 13.4 MB,
   three `cudaMemcpyAsync` each) into a ring of 16 device staging buffers on a second stream. Events in both
   directions: a conversion waits for its copy, a copy waits until its buffer has been converted. The ring carries over
   from one layer to the next.
3. **Conversion to Q8_0.** Every used expert is converted on the device into Q8_0, from its VRAM slot (the decoder's
   slot format) or from the staging buffer. This is exact: an MXFP4 block's 32 values are `kvalues_mxfp4` (integers
   -12..12) times 2^(e - 128), which is a Q8_0 block with d = 2^(e - 128) as fp16 - exact for e in 104..143. Every block
   of all 47 layers has e in that range (the benchmark counts them: 0 outside; layer 24 spans 118..125).
4. **Products.** llama.cpp's Q8_0 MMQ (int8 tensor cores, q8_1 activations) for gate|up of a group of experts in one
   launch, SiLU * up, h to q8_1, the down MMQ, one F32 row per pick. Groups of 16 resident or 8 copied experts; the
   resident groups are spread over the waits for the copied ones, all before the last copied group, so the ring keeps
   moving and only one group is left after the last copy.
5. **Combine.** Per token, the 8 rows times their weights, summed in pick order. Only experts with picks run.

The same input gives the same bits (checked). Device memory: 1.56 GiB at P <= 4,096 (one `cudaMalloc`; `init` refuses
when less than that plus 256 MiB is free, because on Windows an allocation past the free VRAM lands in system memory
instead of failing).

### Why Q8_0 and not llama.cpp's MXFP4 MMQ

On sm_120 llama.cpp's MXFP4 MMQ runs FP4 x FP4 (block-scaled MMA) and quantizes the activations to FP4. The test's
A/B (gate|up of 16 experts x 128 rows of layer 24, against double products of the float x):

| Path | rel. L2 error, median | worst | time, x quantization included |
| --- | ---: | ---: | ---: |
| Q8_0 weights (converted), q8_1 activations - this path | 0.54% | 0.59% | 0.338 ms |
| llama.cpp MXFP4 MMQ, FP4 activations | 11.9% | 12.6% | 0.280 ms |

FP4 activations cost 20x the error for 17% less time (and would spare the conversion, 0.3 ms for these 16 experts).
The MXFP4 instance is built
only with `-DSTRATA_MMQ_MXFP4=ON` (its block-scaled MMA needs the `sm_120a` target, so it gets its own library).

Resident experts: running MMQ directly on the slot format would need a new kernel; converting slot -> Q8_0 costs
22 us per expert (GGUF -> Q8_0: 20 us; first versions that wrote 2-byte pieces took 145-150 us, now the 34-byte blocks
are assembled in shared memory and stored 128 bits at a time). That is ~5 ms per layer when all 256 experts are used,
hidden behind the copies (below). Converting slot -> GGUF layout would buy nothing: MMQ still needs Q8_0.

## Correctness

`mimo_prefill_moe_test <shard> [--layer L] [--resident share]`: the layer's experts copied into pinned RAM, a random
share put into slots, random x (N(0, 1) times the layer's `ffn_norm` weights), random routing over all 256 experts.
Error per token = max |gpu - ref| / max |ref|; every token checked for P <= 333, 96 random tokens at 2,048.

| Layer, resident | P | Used / copied | vs q8 reference: worst / median | vs float: worst max / worst rel. L2 |
| --- | ---: | ---: | ---: | ---: |
| 24, 128 of 256 | 1 | 8 / 5 | 5.0e-08 / 5.0e-08 | 0.23% / 0.99% |
| | 8 | 60 / 28 | 6.3e-05 / 2.5e-07 | 0.70% / 1.1% |
| | 333 | 256 / 128 | 4.4e-04 / 1.2e-07 | 2.5% / 1.2% |
| | 2,048 | 256 / 128 | 1.5e-04 / 1.2e-07 | 1.8% / 1.2% |
| 1, 77 of 256 | 333 | 256 / 179 | 3.6e-04 / 1.9e-07 | 3.4% / 3.1% |
| | 2,048 | 256 / 179 | 7.9e-05 / 1.9e-07 | 3.2% / 2.4% |
| 47, 77 of 256 | 333 | 256 / 179 | 2.6e-04 / 2.4e-07 | 8.4% / 9.2% |
| | 2,048 | 256 / 179 | 4.7e-04 / 2.2e-07 | 17% / 24% |

- q8 reference: llama.cpp MMQ's arithmetic in double (x and h through `quantize_mmq_q8_1`'s rounding, scale kept as a
  float). Gate 2e-3; the outliers are a q8_1 value of h rounding the other way after a sum in another order.
- Float reference: no activation quantization. The gap is the q8_1 activations, the contract of the decoder's
  kernels and of llama.cpp; it grows with outliers in x (layer 47's `ffn_norm` weights make the synthetic x much
  spikier than in layer 1). Real activations of layer 1 measured 0.7-1.5% in MIMO_DECODER.md.
- Also checked: the two converters give the same bytes, and those bytes are exactly ggml's dequantized values (3
  experts, 0 differ); a repeated call is bitwise equal; with every expert resident or none the output moves by at most
  1.7e-4 of its max (MMQ splits the work differently with other groups, which changes the order of some sums).

## Pinned memory (`mimo_pinned_probe`)

The decoder's RAM copy is one `malloc` per MoE layer (3,422,552,064 B). H2D rates of 13.4 MB experts:

| Host memory | Pinning 47 layers (149.8 GiB) | H2D GB/s | CPU read, 24 threads |
| --- | --- | ---: | ---: |
| `malloc`, not pinned | - | 15-21 | 145.5 GB/s |
| `malloc` + `cudaHostRegister` after filling | works for all 47; 0.02 s per layer; unregister 2 s | 44-46 (one copy of 13.4 MB: 46-49) | - |
| `malloc` + `cudaHostRegister` before filling | 0.27 s per layer | 45 | - |
| `cudaHostAlloc` per layer, then filled | works for all 47; 0.9 s per layer (17.6 s); free 6 s | **56-57** | 141.5 GB/s |
| `cudaMallocHost` 428 MB (reference) | | 57.2 | |

So pinning the whole RAM copy works on this machine (Windows 11, 512 GB), but registered `malloc` memory DMAs ~20%
slower than memory allocated pinned (4 KiB pages scattered in physical memory, presumably). Allocating the decoder's
per-layer copies with `cudaHostAlloc` instead of `malloc` gives the full rate and the CPU experts read it as fast (it
is cached, not write-combined). Two streams do not help (one stream already saturates). About 2 of 47 single
measurements per run dropped to 5-7 GB/s for no visible reason (another process's PCIe traffic?); a best-of-2 hides
them in the benchmark.

## Speed (`mimo_prefill_moe_bench`)

All 47 MoE layers; residency as the decoder chooses it from `E:\strata-mimo\coding-answers.prof` at 55% (6,618 of
12,032 experts, 112..202 per layer); routing drawn per token from that profile's per-layer scores (8 distinct experts,
probability proportional to the score; tokens independent); RAM copy pinned with `cudaHostAlloc`. Best of 2 per layer
and mode; times exclude attention and the router. "Compute" = the same calls with the copies skipped; "copy" = with
the conversions and products skipped; overlap = (copy + compute - full) / min(copy, compute).

| P | Used / copied per layer | GB copied per chunk | Full ms | Compute ms | Copy ms | Overlap | H2D GB/s | **MoE tok/s** | If no copies |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 241 / 101 | 63.2 | 1,154 | 538 | 1,131 | 0.96 | 56.0 | **444** | 952 |
| 1,024 | 248 / 107 | 67.2 | 1,234 | 602 | 1,205 | 0.95 | 55.8 | **830** | 1,702 |
| 2,048 | 252 / 111 | 69.6 | 1,283 | 692 | 1,246 | 0.95 | 55.7 | **1,596** | 2,961 |
| 4,096 | 254 / 113 | 71.0 | 1,325 | 889 | 1,266 | 0.93 | 55.8 | **3,091** | 4,606 |

Per layer at P = 4,096: 20-35 ms (54..142 experts copied); routing and the host's plan 0.2-0.4 ms; after the last
copy 1 ms. Other settings, all 47 layers:

| Change | P | Full ms | MoE tok/s |
| --- | ---: | ---: | ---: |
| RAM pinned with `cudaHostRegister` (45 GB/s) instead of `cudaHostAlloc` | 2,048 | 1,569 | 1,306 |
| uniform routing (every expert used, 115 copied per layer) | 512 / 2,048 / 4,096 | 1,316 / 1,325 / 1,349 | 389 / 1,545 / 3,036 |

What the first version of each piece cost (3 layers, P = 2,048, full ms): converters writing 2-byte pieces 159.9 ->
shared-memory converters 91.0 -> resident groups spread over the copy waits and 16 staging buffers instead of 8 (the
copy stream had stalled on a full ring while the GPU did all resident experts first; H2D 38 -> 56 GB/s) 62.7.

### The bottleneck

**PCIe bytes.** Already at P = 512 nearly every expert of a layer has picks (241 of 256), so a chunk copies ~65-71 GB
whatever its size, at 56 GB/s: ~1.2 s per chunk. The GPU's own work (conversions ~5 ms per layer, fixed; MMQ 5 ms per
layer at P = 512, 11 ms at 4,096, i.e. ~146 TOPS of int8 at P = 4,096; Nsight Systems, layer 1) is hidden under the
copies at 93-96%. Doubling the chunk nearly doubles tok/s; compute would catch up with the copies only around P = 8K
(extrapolated: at 4,096 it is 70% of the copy time, and 4 of 47 layers are already compute-bound).

What would move it, for the prompt driver:

- **Larger chunks** (as above), within the VRAM the decoder leaves: the buffers grow 0.24 GiB per 1,024 tokens
  (0.6 GiB fixed: staging and the Q8_0 group buffers).
- **Each layer's copies once per prompt instead of once per chunk** (layer-major prefill: run all chunks of a long
  prompt through layer l before layer l + 1, keeping layer l's ~115 copied experts, 1.5 GB in GGUF layout, on the
  GPU). Then a 50K-token prompt pays ~1.2 s of copies in total, and the MoE runs at the compute rate (~4,600 tok/s at
  P = 4,096, more with the conversion of each copied expert also done once).
- Copies that start before the routing is known (all non-resident experts, in a fixed order) would overlap the copies
  with the layer's attention: at P >= 1,024, 248-254 of 256 experts have picks, so at most ~8 copies per layer would
  be wasted.

## The API for the prompt driver

```cpp
strata::mimo::PrefillMoeConfig cfg;   // max_tokens 4096, k 8, group 16, group_ram 8, staging 16
strata::mimo::PrefillMoe moe(cfg);
std::string err;
if (!moe.init(err)) ...               // 1.56 GiB of VRAM at max_tokens 4096 (moe.device_bytes())
// once: the decoder's per-layer RAM copies pinned - best allocated with cudaHostAlloc (56 GB/s);
// PrefillMoe::register_host(ptr, bytes, err) pins an existing malloc'd copy (45 GB/s)

// per MoE layer of a chunk, after ffn_norm and the router, on the prompt stream:
strata::mimo::PrefillMoeLayer L;
L.n_tok = P;                      // 1..max_tokens
L.x = ffn_norm_out;               // device [P][4096] F32
L.ids = router_ids;               // device [P][8] int32 (the router's top-8)
L.w = router_weights;             // device [P][8] F32
L.out = moe_out;                  // device [P][4096] F32, overwritten (add it to the residual)
L.resident = resident_[l].data(); // host [256]: expert -> slot or -1 (MimoDecoder's table)
L.slot_addr = slot_addr_.data();  // host: slot -> device address (MimoDecoder's)
L.ram = cpu_experts.source(l);    // host: where layer l's experts are in RAM (GGUF layout)
moe.run_layer(L, stream);         // waits for the routing counts once, returns with the work in flight
// moe.times() (after the call; waits for it): total / routing / copy / tail ms, experts used / copied
```

`out` is ready when `stream` reaches it. The host tables are read before `run_layer` returns; x, ids, w must stay
valid until the stream passes the call, and consecutive calls must use the same stream (the host-side plan buffer is
reused once the next call's counts have arrived). `Mode::compute_only` / `copy_only` exist for the benchmark only.

## Not done

- The prompt driver itself (attention over the chunk, the router for P tokens, the dense layer), and layer-major
  scheduling across chunks.
- An MMQ kernel on the slot format (or on MXFP4 with int8 activations), which would drop the conversions and halve the
  bytes the products read; worth it once the copies are gone.
- Copies issued before the routing is known.
- HIP.
