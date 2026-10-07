# MiMo-V2.6-Flash: the routed experts of the prompt (prefill)

Part of step 8 of [MIMO_PORT_PLAN.md](MIMO_PORT_PLAN.md): the MoE part of prompt processing, one layer per call, for a
slice of S = 1 to 32,768 tokens (configurable). The prompt driver runs layer-major over a large slice (layer l over all
S tokens, then layer l + 1), so each non-resident expert crosses PCIe at most once per layer per slice. Code:
`include/strata/mimo/prefill_moe.hpp`, `src/mimo/prefill_moe.cu`. Test: `tests/mimo/prefill_moe_test.cpp`
(`mimo_prefill_moe_test`); benchmark: `tools/mimo/prefill_moe_bench.cpp` (`mimo_prefill_moe_bench`); pinned-memory
probe: `tools/mimo/pinned_probe.cpp` (`mimo_pinned_probe`).

Measured 2026-10-07 on GPU 0: RTX PRO 6000 Blackwell Max-Q (96 GB, sm_120, PCIe 5.0 x16), CUDA 12.9, Windows 11;
Threadripper 9960X, 4 x 128 GB DDR5-6400; the MXFP4 GGUF from ggml-org. Unless said otherwise nothing else ran on GPU 0
(checked with nvidia-smi during the runs).

## What a call does

`PrefillMoe::run_layer` computes, for one MoE layer, `out[t] = sum over the 8 picks of w * down(silu(gate x) * up x)`
for every token of the slice:

1. **Grouping on the device.** A histogram of the S x 8 picks; the 256 counts go to the host (the one host sync per
   layer), which orders the used experts (resident ones first, then the ones in RAM), splits them into groups (16
   resident or 8 copied experts) and each group's rows into tiles of at most `tile_rows` rows (an expert's rows may
   span tiles). A placement kernel lays the picks out in that order, stable (pick order within an expert, no atomics),
   with each pick's token and weight.
2. **Copies.** The used experts that are not resident are copied once from RAM (GGUF layout, gate | up | down,
   13.4 MB, three `cudaMemcpyAsync` each) into a ring of 16 device staging buffers on a second stream. Events in both
   directions: a conversion waits for its copy, a copy waits until its buffer has been converted. The ring carries over
   from one layer to the next.
3. **Conversion to Q8_0**, once per expert and call, from its VRAM slot (the decoder's slot format) or from the staging
   buffer. This is exact: an MXFP4 block's 32 values are `kvalues_mxfp4` (integers -12..12) times 2^(e - 128), which is
   a Q8_0 block with d = 2^(e - 128) as fp16 - exact for e in 104..143. Every block of all 47 layers has e in that range
   (the benchmark counts them: 0 outside; layer 24 spans 118..125).
4. **Products, per tile.** The tile's x rows gathered and quantized to q8_1, llama.cpp's Q8_0 MMQ (int8 tensor cores)
   for gate|up of the tile's experts in one launch, SiLU * up, h to q8_1, the down MMQ. The resident groups are spread
   over the waits for the copied ones, all before the last copied group, so the ring keeps moving.
5. **Into `out`.** Each expert's rows of the tile, times their weights, are added to their tokens' rows - one launch per
   expert, in the processing order, so every output row sums in a fixed order. The same input gives the same bits.

Device memory: 0.6 GiB fixed (16 staging buffers, the Q8_0 images of a group) + 31.5 KiB per tile row + 8 bytes per
pick: **0.72 GiB at S <= 32,768 with the default 4,096-row tiles** (`device_bytes()`; `vram_mib` picks the largest tile
that fits a budget). The tile size hardly matters: 4,096 / 8,192 / 16,384 / 32,768 rows (0.72 / 0.84 / 1.08 / 1.56 GiB)
took 503 / 510 / 514 / 523 ms for 6 layers at S = 32,768. One `cudaMalloc`; `init` refuses when less than that plus
256 MiB is free (on Windows an allocation past the free VRAM lands in system memory instead of failing). The caller
owns x, ids, w and out ([S][4096] F32 each for x and out: 0.5 GiB apiece at 32K).

### Why Q8_0 and not llama.cpp's MXFP4 MMQ

On sm_120 llama.cpp's MXFP4 MMQ runs FP4 x FP4 (block-scaled MMA) and quantizes the activations to FP4. The test's
A/B (gate|up of 16 experts x 128 rows of layer 24, against double products of the float x):

| Path | rel. L2 error, median | worst | time, x quantization included |
| --- | ---: | ---: | ---: |
| Q8_0 weights (converted), q8_1 activations - this path | 0.54% | 0.59% | 0.338-0.450 ms |
| llama.cpp MXFP4 MMQ, FP4 activations | 11.9% | 12.6% | 0.280-0.328 ms |

FP4 activations cost 20x the error for 17-27% less time. The MXFP4 instance is built only with
`-DSTRATA_MMQ_MXFP4=ON` (its block-scaled MMA needs the `sm_120a` target, so it gets its own library).

## Correctness

`mimo_prefill_moe_test <shard> [--layer L] [--resident share]`: the layer's experts copied into pinned RAM, a random
share put into slots, random x (N(0, 1) times the layer's `ffn_norm` weights), random routing over all 256 experts;
P = 8,192 adds a hot expert (every token picks expert 7: 8,192 rows). The test runs with 512-row tiles, so experts
are split across tiles (expert 7 over 16). Error per token = max |gpu - ref| / max |ref|; every token checked for
P <= 333, 96 random tokens above.

| Layer, resident | P | Used / copied | vs q8 reference: worst / median | vs float: worst max / worst rel. L2 |
| --- | ---: | ---: | ---: | ---: |
| 24, 128 of 256 | 1 | 8 / 5 | 8.0e-08 / 8.0e-08 | 0.23% / 0.99% |
| | 8 | 60 / 28 | 6.3e-05 / 2.5e-07 | 0.70% / 1.1% |
| | 333 | 256 / 128 | 4.4e-04 / 1.2e-07 | 2.5% / 1.2% |
| | 2,048 | 256 / 128 | 1.5e-04 / 1.0e-07 | 1.8% / 1.2% |
| | 8,192 | 256 / 128 | 3.4e-04 / 1.0e-07 | 1.3% / 1.2% |
| 47, 77 of 256 | 333 | 256 / 179 | 2.6e-04 / 2.4e-07 | 8.4% / 9.2% |
| | 2,048 | 256 / 179 | 4.7e-04 / 2.1e-07 | 17% / 24% |
| | 8,192 | 256 / 179 | 2.0e-03 / 2.3e-07 | 17% / 23% |

- q8 reference: llama.cpp MMQ's arithmetic in double (x and h through `quantize_mmq_q8_1`'s rounding, scale kept as a
  float). Gate 5e-3; the outliers are a q8_1 value of h rounding the other way after a sum in another order (the
  medians are ~1e-7).
- Float reference: no activation quantization. The gap is the q8_1 activations, the contract of the decoder's
  kernels and of llama.cpp; it grows with outliers in x (layer 47's `ffn_norm` weights make the synthetic x much
  spikier than in layer 24). Real activations of layer 1 measured 0.7-1.5% in MIMO_DECODER.md.
- Also checked: the two converters give the same bytes, and those bytes are exactly ggml's dequantized values (3
  experts, 0 differ); a repeated call is bitwise equal; with every expert resident or none the output moves by at most
  1.7e-4 of its max (other groups and tiles change the order of some sums).

## Pinned memory

H2D rates of 13.4 MB experts (`mimo_pinned_probe`; the decoder's RAM copy is one buffer per MoE layer,
3,422,552,064 B):

| Host memory | H2D GB/s | CPU read, 24 threads |
| --- | ---: | ---: |
| `malloc`, not pinned | 15-21 | 145.5 GB/s |
| `malloc` + `cudaHostRegister` (after or before filling) | 44-46 | - |
| `cudaHostAlloc` per layer | **56-57** | 141.5 GB/s |
| `cudaMallocHost` 428 MB (reference) | 57.2 | |

Pinning all 47 layers (149.8 GiB) works either way on this machine (512 GB RAM). Registered `malloc` memory DMAs ~20%
slower than memory allocated pinned; the CPU experts read both as fast. Two copy streams do not help. About 2 of 47
single 64-expert measurements per probe run dropped to 5-7 GB/s for no visible reason; the benchmark's best-of-2 hides
them.

**Start-up cost of the decoder's RAM copy** (`mimo_prefill_moe_bench <shard> --startup hostalloc|register`: allocate
all 47 layers, fill each with `copy_layer`'s copy - 64 MiB pieces from the GGUF mapping on 16 threads - and pin; the
GGUF in the file cache; two runs each, alternating):

| Way | Allocate | Fill | Register | Total | Release |
| --- | ---: | ---: | ---: | ---: | ---: |
| `malloc` + fill + `cudaHostRegister` | 0.0 s | 41.5 / 59.5 s | 0.9 s | **42.3 / 60.4 s** | 11.7 s |
| `cudaHostAlloc` + fill | 19.0 / 25.7 s | 17.1 / 19.6 s | - | **36.1 / 45.3 s** | 6.8 / 12.5 s |

`cudaHostAlloc` pays for the pages up front, and the fill then runs without page faults, so it is not slower to start
- and 25% faster over PCIe. Use `cudaHostAlloc` for the decoder's copy.

## Speed (`mimo_prefill_moe_bench`)

All 47 MoE layers; residency as the decoder chooses it from `E:\strata-mimo\coding-answers.prof` at 55% (6,618 of
12,032 experts, 112..202 per layer); routing drawn per token from that profile's per-layer scores (8 distinct experts,
probability proportional to the score; tokens independent); RAM copy pinned with `cudaHostAlloc`; 4,096-row tiles.
Best of 2 per layer and mode; times exclude attention and the router. "Compute" = the same calls with the copies
skipped; "copy" = with the conversions and products skipped; overlap = (copy + compute - full) / min(copy, compute)
(above 1: within the noise of the two separate runs, the shorter side fully hidden).

| S | Copied per layer | GB copied per S tokens | Full ms | Compute ms | Copy ms | Overlap | **MoE tok/s** |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 112 | 70.6 | 1,338 | 977 | 1,259 | 0.92 | **3,061** |
| 8,192 | 114 | 71.5 | 1,617 | 1,439 | 1,274 | 0.86 | **5,067** |
| 16,384 | 114 | 71.8 | 2,461 | 2,387 | 1,278 | 0.94 | **6,659** |
| 32,768 | 115 | 71.9 | 4,042 | 4,196 | 1,280 | 1.12 | **8,107** |

(An earlier run with 16,384-row tiles, during which another process used GPU 0 for part of the time: 3,056 / 5,103 /
6,543 / 7,855 tok/s.)

**Where it turns compute-bound: between S = 4,096 and 8,192.** The copies are ~71-72 GB per layer pass whatever S is
(253-255 of 256 experts have picks already at 4,096), 1.28 s at 56 GB/s; compute grows ~linearly (89 ms per layer at
32K). From 8K on, tok/s follows the compute: at 32K the full run (8,107 tok/s) is already at the compute-only rate
(S / compute = 7,809; the two runs differ by a few percent).

Compute per layer at S = 32,768 (Nsight Systems, layer 1; another process held most of GPU 0's memory during this
profile, the sum matches the clean run's ~89 ms per layer): MMQ 71 ms (~186 TOPS of int8 for the 13.2 TFLOP of a
layer's three products), the weighted adds into `out` 7 ms, q8_1 quantization of x and h 4.5 ms, the conversions to
Q8_0 5.7 ms (slot -> Q8_0 23 us, GGUF -> Q8_0 22 us per expert), SiLU * up 3.3 ms, placement 0.5 ms.

**Is a kernel on the slot format worth it now?** For the conversions, no: 5.7 ms of ~90 (6%) at 32K; at 4K they are
~27% of the compute, but hidden under the copies. Such a kernel would also halve the weight bytes the products read, but at S >= 8K the
products are int8-MMA bound, not memory bound. What would move the compute-bound regime is the MMQ itself (71 of 90 ms
at ~186 TOPS); the FP4 MMA is faster but its FP4 activations are 20x less accurate (above).

Earlier, smaller chunks (same benchmark, before tiling; P = 512 / 1,024 / 2,048: 444 / 830 / 1,596 tok/s, 63-70 GB
copied per chunk): bound by the copies, which is why the driver goes layer-major over large slices.

What the first version of each piece cost (3 layers, P = 2,048, full ms): converters writing 2-byte pieces 159.9 ->
shared-memory converters 91.0 -> resident groups spread over the copy waits and 16 staging buffers instead of 8 (the
copy stream had stalled on a full ring while the GPU did all resident experts first; H2D 38 -> 56 GB/s) 62.7.

## The API for the prompt driver

```cpp
strata::mimo::PrefillMoeConfig cfg;   // max_tokens 32768, k 8, group 16, group_ram 8, staging 16, tile_rows 4096
cfg.vram_mib = 0;                     // or a budget: init picks the largest tile that fits
strata::mimo::PrefillMoe moe(cfg);
std::string err;
if (!moe.init(err)) ...               // 0.72 GiB at the defaults (moe.device_bytes(), moe.config().tile_rows)
// once: the decoder's per-layer RAM copies allocated with cudaHostAlloc (56 GB/s);
// PrefillMoe::register_host(ptr, bytes, err) pins an existing malloc'd copy instead (45 GB/s)

// per MoE layer, over the whole slice of S tokens, after ffn_norm and the router, on the prompt stream:
strata::mimo::PrefillMoeLayer L;
L.n_tok = S;                      // 1..max_tokens
L.x = ffn_norm_out;               // device [S][4096] F32
L.ids = router_ids;               // device [S][8] int32 (the router's top-8, distinct per token)
L.w = router_weights;             // device [S][8] F32
L.out = moe_out;                  // device [S][4096] F32, overwritten (add it to the residual)
L.resident = resident_[l].data(); // host [256]: expert -> slot or -1 (MimoDecoder's table)
L.slot_addr = slot_addr_.data();  // host: slot -> device address (MimoDecoder's)
L.ram = cpu_experts.source(l);    // host: where layer l's experts are in RAM (GGUF layout)
moe.run_layer(L, stream);         // waits for the routing counts once, returns with the work in flight
// moe.times() (after the call; waits for it): total / routing / copy / tail ms, experts used / copied, tiles
```

`out` is ready when `stream` reaches it. The host tables are read before `run_layer` returns; x, ids, w must stay
valid until the stream passes the call, and consecutive calls must use the same stream (the host-side plan buffer is
reused once the next call's counts have arrived). `Mode::compute_only` / `copy_only` exist for the benchmark only.

## Not done

- The prompt driver itself (attention over the slice, the router for S tokens, the dense layer).
- Copies issued before the routing is known (would overlap them with the layer's attention; at S >= 4K at most 1-3
  copies per layer would be wasted).
- A faster int8 product than llama.cpp's MMQ for the compute-bound regime.
- HIP.
