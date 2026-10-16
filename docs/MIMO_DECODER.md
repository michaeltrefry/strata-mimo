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
sliding-window history is no longer in the live ring, it first looks for a valid slot-local host checkpoint. With
no eligible checkpoint it reprocesses from position zero: replaying an arbitrary short suffix would itself need
older keys that are no longer present. A fresh reset clears the small rings and invalidates their checkpoints;
a retained-prefix continuation does not clear the MTP history it still needs.

`mimo_serve_test` checks the production scheduler against a CPU fake, without a model or GPU. The attention tests
also compare two independent KV sets using shared scratch and slot-bound graph replay against the CPU reference.
The model-required `mimo_decoder_slots_test` compares fixed-input logits and drafts between isolated and
alternating calls, including divergent/extended/canceled prompt-prefix reuse. Its arguments are the main GGUF,
MTP GGUF, optional expert profile, optional expert budget in GiB, and optional MTP head count (1 or 2). It reads CPU
experts from the existing GGUF mapping rather than allocating a second pinned ~150 GiB copy for validation.
The original slot suite, before the checkpoint/service-budget follow-up, was validated 2026-10-15 on GPU 1 of the
same RTX PRO 6000 Blackwell Max-Q PC: synthetic attention/shared-scratch
isolation passed both direct and captured-graph execution; fused kernels remained bitwise equal. The full-model
slot harness passed graph off, graph on and batched prefill with MTP enabled, 1,024 positions per slot, a 64 GiB
expert budget and mapped CPU experts. Controlled divergent/extended/canceled-prefix comparisons reported zero
maximum absolute logit error and matching MTP drafts. Their retained and fresh references use identical main-model
window widths, rather than conflating window-width numerical differences with cache isolation.

These are correctness checks. Separate HTTP validation timings follow.

### Long-turn throughput follow-up

**Status:** CPU/build checks, new real-model checkpoint parity and the 25K/50K/77K/100K multi-turn reuse campaign
have passed. The repeated 77K decode / fresh 28K prefill observations below are current-policy evidence; a separate
round-robin control is needed to isolate the scheduling effect. They are not a normal-production tok/s prediction.
The older short HTTP runs are measurements of the previous equal-turn policy, not the long-turn acceptance gate.

**Reuse checkpoints.** Only main SWA and enabled MTP ring allocations are copied into ordinary pageable host RAM.
The nine global-attention KV caches stay in their existing GPU allocations; captured graphs keep the same addresses.
A one-head snapshot is 27,197,440 bytes, or 27,877,376 bytes with two heads (layout arithmetic, not transfer timings).
Per slot, the policy retains prompt milestones at 4K, 8K, 16K, 32K, 64K, 128K and 256K, plus the latest completed
prompt end, deduplicating coincident entries. The aggregate native 512 MiB budget counts payload, exact-prefix
metadata, journals and an in-flight replacement; allocation/budget misses preserve a safe fallback. Copies complete
on the serialized compute owner before a snapshot is published or used. No periodic decode snapshots or cross-slot
restores are attempted.

A checkpoint boundary C records exact committed tokens `[0,C)`. It replays main token A=C−1 after restoring rings,
so token A must still match, but old MTP lookahead at C can be replaced. Global KV `[0,A)` must remain intact.
Every main write beginning at P invalidates checkpoints with A>P **before enqueue**, including identical-token
rewrites: changing window widths can change numerical KV state. Append and replay at A preserve that checkpoint.
Reset invalidates its generation. Capture is allowed only at the last successful complete prompt-chunk boundary,
checked against the decoder's executed-token journal. Restore clears shared hidden ownership, installs the saved
physical write watermarks, and requires main replay before MTP. Partial CUDA restore failure ends the worker;
missing or mismatching snapshots never authorize unsafe reuse. Geometric anchors bound memory, not the amount of
replay for every arbitrary edit.

**Service budget.** Mixed prefill/decode work uses measured compute-owner time: default decode weight 80%, with a
prefill turn due after 250 ms waiting. Native options are `--decode-service-percent 1..99` and
`--prefill-max-wait-ms 1..60000`, accepted as separate arguments or `--name=value`. The least normalized service
total wins, with decode favored on ties and an overdue prefill turn overriding weights. Debt resets when commands
or phase pairing change. Cancellation is handled first; two decoders or two prefills retain round-robin selection.
The eight-token competing-prefill cap remains. A deadline includes the currently executing atomic main/MTP unit;
it is not a fixed streamed-token or first-token latency guarantee. Deadline overrides may lower the achieved share.

**Metrics.** Reserved, versioned `INFO request_diag_v=1` records attach to their command, not process-wide facts;
ordinary INFO and positional DONE remain compatible. No prompt text or token IDs are logged. Missing diagnostics
from older tagged engines remain unknown. Per-request history exposes:

- `prefix_reason`, `raw_lcp`, `boundary_keep`, prior `written_end`, selected `resume`, checkpoint boundary/id and reason.
- `live_written_end`: final physical watermark for affinity estimates, not the prior watermark in prefix diagnostics.
- `prompt_service_ms`, `decode_service_ms`, phase expert hits/uses and derived service rates/hit rates.
- Checkpoint capture/restore counts, copy times, payload/metadata bytes and retained boundaries.

Active service covers whole compute-owner units, including sampling/checkpoint work, excluding stdout flush and
consumer delay. Existing wall decode timings and client-delivered tok/s are unchanged and still include peer work.
The legacy `hit_rate` is combined prompt/decode routing; compare `decode_hit_rate` with decode-only measurements,
not the combined value. Reasoning wraps sum durations/counters and recompute ratios; gauges take their final value.

Affinity uses terminal-confirmed processed input plus all native-emitted accepted tokens, including drained output.
It clears uncertain state on errors/restart and preserves it when cancellation occurred before native preparation.
Free-slot scoring estimates the better live-ring or matching-anchor boundary. It does not wait indefinitely for a
busy matching slot, and only native exact-prefix/global-dependency checks can authorize reuse.

**Validation.** `mimo_prefix_checkpoint_test` checks metadata/identity/global-write/budget rules without a GPU.
`mimo_serve_test` adds bounded-ring restoration and deterministic clock tests for weighted costs, deadline overrides,
phase accounting and cancellation. The model-required slot test adds ring-aged restore, changed lookahead, branch
invalidation and peer/reset cases with identical main-window traces and the existing numerical tolerance.

The HTTP smoke tool now defaults to a network-free dry run; add `--run` only for an explicitly selected loaded,
isolated loopback instance. Existing `--case all` remains the five short checks. Separate `--case long-prefix` and
`--case matched` campaigns cover 25K/50K/77K/100K rendered prefixes, repeated/assistant/tool/divergent turns and a
77K decoder beside fresh 28K prefill. They require actual native generation beyond 256 tokens, a length finish,
request-local diagnostics and bounded suffix replay from the best known matching before-request anchor, not merely
whichever checkpoint was selected. Valid live-tail continuations are checked separately from aged-ring restores.
Missing evidence, full-prefix replay, truncated SSE delivery or an early stop cannot pass the reuse gate. Matched results distinguish delivered wall throughput, active
service, text-update gaps and peer first-text latency; they do not claim a recovered speed without a matched prior
scheduler baseline. Do not run these campaigns against an occupied service or GPU without approval.

### Long-turn GPU validation (2026-10-16)

GPU 1 on the same RTX PRO 6000 Blackwell Max-Q / Threadripper 9960X / 512 GB Windows 11 PC, one MTP head,
64 GiB expert budget (5,132 resident experts), mapped CPU experts and a live GPU 0 server sharing CPU/RAM.
The controlled checkpoint model suite passed graph off, graph on and batched prefill with 1,024-position caches;
aged-ring restoration/lookahead comparisons reported zero maximum absolute logit error. This is a controlled
matching-window comparison, not a claim that different GEMM/GEMV window schedules are bitwise identical.

The HTTP campaign used two 262,144-position slots. All six turns at each size completed 512 native output tokens
with complete SSE/usage evidence: seed, exact repeat, assistant append, normalized tool turn, divergent suffix and
return to the old branch. The validator checked actual count-endpoint agreement, before-request anchor inventory,
request/slot identity, and processed-token bounds.

| Seed's actual rendered tokens | Exact repeat: processed / reused | Assistant append: processed | Tool turn: processed |
| --- | ---: | ---: | ---: |
| 25,003 | 1 / 25,002 | 20 | 40 |
| 50,001 | 1 / 50,000 | 20 | 40 |
| 77,004 | 1 / 77,003 | 20 | 40 |
| 100,012 | 1 / 100,011 | 20 | 40 |

Observed checkpoint restores took 1.51–3.35 ms. Divergence correctly reused older geometric anchors: 16K, 32K,
64K and 64K respectively, rather than resurrecting an overwritten later prefix. It still had to process 8,620,
17,234, 11,469 and 34,477 tokens respectively. Checkpoints do not turn arbitrary edited history into a one-token
prefill. In the first 25K run, cold seed wall time was 106.18 s and exact-repeat wall time 10.41 s; these include
512 generated tokens and different warm/adaptive-cache conditions, so this is not an isolated decode speed-up.

Two repetitions per policy compared a warmed 77K decoder with a fresh 28K peer prefill, producing 512 tokens
per response. The control linked the same decoder libraries and kept checkpoints enabled, replacing only work
selection with round-robin. Its startup identified `scheduler_policy=round_robin_control` and disabled service
weighting; the reported configured 80% was ignored. Both campaigns passed the overlap, output and reuse gates.

| Measurement | Service-weighted policy | Round-robin control |
| --- | ---: | ---: |
| Median delivered wall rate, solo | 49.2 tok/s | 53.2 tok/s |
| Median delivered wall rate, competing | 32.4 tok/s | 11.6 tok/s |
| Competing / solo delivered rate | 66% | 22% |
| Active decode service / elapsed decode interval, competing | 66–67% | 21–23% |
| Largest streamed-text update gap per competing response | 0.22–0.30 s | 0.31–0.32 s |
| Peer first-text latency | 93.1–96.3 s | 104.4–107.1 s |

Delivered wall rate is all 512 output tokens divided by the full API response duration. The competing median was
2.79 times the control's; solo rates differed by about 8%, and there were only two repetitions per policy.
The decoder's charged-service rates were 48.6–50.3 tok/s weighted and 51.8–55.5 tok/s round-robin: the delivery
improvement came from more compute-owner service, not faster decode kernels. The prefill-wait deadline can override
the 80% weight target, and the peer still progressed. Text-update gaps are not token-by-token latency.

These sequential campaigns used the same GPU, model/profile, expert budget, mapped-expert mode and context settings,
with the same counted fixture recipe but randomized identifiers (77,007 decoder tokens weighted, 77,006 control).
This validates the scheduling improvement in this test configuration, not a full old-release before/after comparison,
a production-speed prediction or a 100 tok/s claim. Each raw artifact keeps `performance_validated=false` because the
individual campaign does not itself compare policies. Evidence: `build-throughput-cpu/gpu1-matched77k-run2.json`,
`build-throughput-cpu/gpu1-matched77k-roundrobin-run1.json` and the private control's
`build-throughput-rr-control/variant.json`; these local build artifacts are not checked into Git.

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

These runs selected the eight-token interactive cap: protect the live response, accepting slower prompt reading
while compute is shared. The later service-budget/checkpoint follow-up above is not represented by this table.
The chunk-gap measurements are not a fixed latency guarantee.
`tools/mimo/serve_slots_smoke.py --base-url <loopback test server> --case all --run` repeats the streamed/collected,
cancellation, long-prompt and queue checks against an already-loaded two-slot instance; without `--run` it only
prints the plan. It never starts/stops a model or changes the launch configuration.

## Next

- An option to split the experts over two GPUs.
- The CPU's slow layers (p99 333 us per layer).
