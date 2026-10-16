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

## Fused small kernels

Around its GEMVs, attention and experts a MoE layer ran 15-16 small kernels of 1-5 us each.  `src/mimo/fused.cu` does the same
work in fewer launches, with bitwise the same results (the same expressions in the same order; `mimo_fused_test`
compares every output with the separate kernels):

| Before | After |
| --- | --- |
| copy_from_mapped, add_scaled (y += CPU sums), add_scaled (x += y), then the next layer's rmsnorm and quantize_q8_1 | `mimo_add_rmsnorm_q8_1`: reads the CPU's sums straight from mapped memory, writes x, h and h as q8_1 for the qkv GEMV |
| quantize_q8_1 before attn_output | the attention combine kernel writes its output as q8_1 too (an SWA layer at T = 1 has no combine kernel and keeps the separate one) |
| residual_add, rmsnorm (ffn_norm), fill_zero, the doorbell's copy of h, the experts' first kernel (h to q8_1) | `mimo_add_rmsnorm_q8_1` |
| router_logits, router_topk, resident_slots, doorbell_publish (ids, weights, ring) | `mimo_router_fused`: the router_logits blocks, then the last block to finish picks the experts (a warp per token) and rings |

The dense layer and the MTP layer get the same norms (their gate and up GEMVs share one q8_1 input instead of two).
The non-graph path (`--dump`) runs the same kernels.

Kernel time per MoE layer (Nsight Systems, `--cuda-graph-trace=node`, 128 generated tokens; everything except the
GEMVs, the attention split kernel, the experts' two matrix kernels and the wait for the CPU):

| Window | Before | After | Small kernels per MoE layer |
| --- | ---: | ---: | --- |
| T = 1 | 30.8 us | 23.3 us | 15 -> 5 (rope_append and the quantize or combine included) |
| T = 2 (MTP verify) | 35.6 us | 26.5 us | 16 -> 5 |

Alone (`mimo_fused_test --bench`, graphs of 200 calls, T = 1): the end of a layer 7.1 -> 3.5 us, residual + ffn_norm
3.1 -> 3.6 us (it now also writes the doorbell's h and the experts' q8_1), router + slots + doorbell + fill_zero
13.2 -> 9.9 us.  Of the router's 9.9 us, 3.3 are the logits; the last block's hand-off (fence, counter, reading the
logits back) costs about 3.5 us, and the pick 1.3 us.

Speed (311-token prompt, 512 generated, profile `data/mimo-coding-answers.prof`, before and after alternated):

| | Before | After |
| --- | --- | --- |
| Output tok/s, T = 1 | 83.7, 83.9, 83.1, 82.2 | 82.7, 84.4, 85.9 |
| Output tok/s, MTP (T = 2) | 115.4, 115.3, 109.3, 110.8 | 117.4, 113.5, 118.4 |
| GPU only (`MIMO_SKIP_CPU=1`), T = 1 | 113.9, 113.8 | 117.9, 117.9 |
| GPU only, MTP: main window | 11.31, 11.29 ms | 10.95, 10.97 ms |

The GPU's part is 3.5% faster; the full runs move less and vary by a few tok/s from run to run with the CPU's expert
time (4.3-5.1 ms per token across these runs): the GPU now waits longer for the CPU (`wait_flag_ge`).  Greedy output is
unchanged: the same 512 tokens with and without MTP, and every tensor of a `--dump` run of the 311-token prompt is
bitwise the same file (against llama.cpp: top-1 98.71%, KL mean 0.00261, as before).

## Residency that follows the work

The resident set starts from the profile and then moves (`src/mimo/decoder.cpp`, `adapt_*` in DecoderConfig):

- After a prompt of 2,048+ tokens the profile is blended with the prompt's own expert counts (weight 0.5) and the
  difference swapped in from the pinned RAM copy (873 experts in 297 ms after a 67,723-token prompt).
- During generation: counts that halve every 256 tokens; up to 2 swaps per window into 8 spare slots, copied on a
  second stream and committed between windows.  Swapping in bursts stalled the CPU handoff (-30%); a trickle does
  not.
- The batched prompt path's 2.29 GiB of buffers borrow the last 188 expert slots for the prompt, which are refilled
  afterwards; the server saves what it learned (`expert_profile_save`) and starts from it next time.

1,024 generated tokens with MTP at 128K context: short prompt 111.6 -> 124.2 tok/s, 68K code prompt 72.5 -> 114.9.

## 250K context

A 248,835-token prompt (code and docs): read in 111.3 s (2,236 tok/s), then 86.0 tok/s output, 85.6% resident, 84%
of MTP drafts accepted; the answer was correct and detailed.

## Tried and not kept

- Q6_K attention weights (qkv, output; 0.8 GiB more experts resident; KL vs Q8_0 0.0035): with the same forced
  token sequence at 68K (`strata-mimo-run --force`), Q8_0 121.7 / 124.0 tok/s and Q6_K 123.7 / 122.1 - within
  noise, so Q8_0 stays.  Free-running runs differ by more (101-131) because the two models generate different text.
- Two MTP heads (drafting 2 tokens): 117 vs 121 tok/s short, 97.5 vs 109.6 at 68K - opt-in only (`--mtp-heads 2`).
- Raising the CPU pool threads' priority: no change.

## Two sequence slots

Opt-in `strata-mimo --slots 2` allocates separate main/MTP KV and CUDA graph executables for each sequence. Model
weights, expert RAM/cache, CPU pool, activation buffers, handoff buffers and prompt workspace stay shared. One
scheduler thread alternates full MTP verify/accept/draft steps, so shared hidden rows are consumed before another
sequence overwrites them. Graphs bind each slot's fixed KV addresses; resetting a slot does not replace its
allocations. `strata-mimo-run` and the default server still use slot 0.

The tagged protocol advertises `READY <context-per-slot> stop multiplex=1 slots=2`. GEN carries a unique increasing
command id and a slot id before the existing arguments; STOP carries the command id. RESUME, PP, T, DONE and ERR
carry the same command id. A slot is reusable only after its command's terminal response. Expert hit/use counters
are accumulated around each request's own scheduled turns, not copied from the process-wide totals.

Prompts use complete calls of at most `--prefill-quantum` tokens (default 2,048). While the peer is decoding, turns
are capped at 8 prompt tokens; with the default `--batched-min 128` these use the decode-window path and do not
borrow/refill expert slots. Bulk chunks resume after the peer finishes; each bulk call refills borrowed slots before
yielding. Lowering the batched threshold can change that latency trade-off. On an intermediate chunk, the chooser
returns the next **actual prompt token**,
without sampling: MTP writes one position ahead and would otherwise cache a prediction at the chunk boundary.
Only the final chunk samples the first generated token. Prompt routing counts accumulate across chunks and prompt
adaptation runs at completion. Repeated borrow/refill is deliberately retained initially; its cost needs measurement.

Prefix reuse is slot-local. With MTP it replays the token before the first unvalidated token as well: that main
hidden row is needed to replace MTP's one-ahead entry at a divergent suffix or extension boundary. The two-slot
scheduler also accounts for writes beyond the accepted session from speculation and MTP. If required
sliding-window history is no longer in the ring, it reprocesses from position zero;
replaying only a short suffix would itself need older keys that are no longer present. A fresh reset clears the
small rings; a retained-prefix continuation does not clear the MTP history it still needs.

`mimo_serve_test` checks the production scheduler against a CPU fake, without a model or GPU. The attention tests
also compare two independent KV sets using shared scratch and slot-bound graph replay against the CPU reference.
The model-required `mimo_decoder_slots_test` compares fixed-input logits and drafts between isolated and
alternating calls, including divergent/extended/canceled prompt-prefix reuse. Its arguments are the main GGUF,
MTP GGUF, optional expert profile, optional expert budget in GiB, and optional MTP head count (1 or 2). It reads CPU
experts from the existing GGUF mapping rather than allocating a second pinned ~150 GiB copy for validation.
Validated 2026-10-15 on GPU 1 of the same RTX PRO 6000 Blackwell Max-Q PC: synthetic attention/shared-scratch
isolation passed both direct and captured-graph execution; fused kernels remained bitwise equal. The full-model
slot harness passed graph off, graph on and batched prefill with MTP enabled, 1,024 positions per slot, a 64 GiB
expert budget and mapped CPU experts. Controlled divergent/extended/canceled-prefix comparisons reported zero
maximum absolute logit error and matching MTP drafts. Their retained and fresh references use identical main-model
window widths, rather than conflating window-width numerical differences with cache isolation.

These are correctness checks. Separate HTTP validation timings follow.

### HTTP validation (2026-10-15)

GPU 1 on the same RTX PRO 6000 Blackwell Max-Q / Threadripper 9960X / 512 GB Windows 11 PC, one MTP head,
262,144 positions **per slot**, 64 GiB expert budget (5,132 resident experts), coding-answers profile and
`--ram-copy 0`. The server reported 6,320,685,056 bytes of KV across the two slots. GPU 0's existing MiMo server
remained loaded and also served work during this session; CPU/RAM were shared. These are single validation runs,
not an uncontended benchmark or a prediction for the normal pinned-RAM/full-expert-budget configuration.

| HTTP workload, 256 generated tokens per completed response | Wall time | Total output tok/s, including prompts |
| --- | ---: | ---: |
| Two streamed short coding requests; a third queued request disconnected before dispatch | 6.54 s | 78.3 |
| One streamed and one collected short coding response | 5.91 s | 86.6 |
| OpenAI-compatible and Anthropic-compatible streams together | 6.27 s | 81.7 |

Both streams produced model text before either finished. Canceling an active stream left its peer producing its
full response; disconnecting the queued third request removed it without dispatch. An additional client that never
read its SSE response did not stop two ordinary 256-token requests from completing; closing it left both slots idle.

A 47-token request already generating beside a newly arrived 6,228-token prompt exposed the scheduling trade-off:

| Measurement | Bulk prompt turns capped at 2,048 | 8-token prompt turns while peer generates |
| --- | ---: | ---: |
| Existing stream finishes | 28.02 s | 14.78 s |
| Largest gap between its streamed text chunks | 8.467 s | 0.143 s |
| Long prompt's first visible text | 21.35 s | 29.74 s |

The interactive policy is retained: protect the live response, accepting slower prompt reading while the two share
compute. After the peer completes, prefill returns to the configured bulk ceiling. These chunk-gap measurements are
not a fixed latency guarantee. `tools/mimo/serve_slots_smoke.py --base-url <loopback test server> --case all` repeats
the streamed/collected, cancellation, long-prompt and queue checks against an already-loaded two-slot instance; it
never starts/stops a model or changes the launch configuration.

## Next

- An option to split the experts over two GPUs.
- The CPU's slow layers (p99 333 us per layer).
