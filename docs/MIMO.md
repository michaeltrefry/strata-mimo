# Strata-MiMo: MiMo-V2.6-Flash-RL on one RTX PRO 6000

Strata's engine ported to XiaomiMiMo's MiMo-V2.6-Flash-RL (309B total, 15B active, MXFP4 experts): the attention,
the dense layer, the routers and ~6,600 of the 12,032 experts on one 96 GB GPU, the rest of the experts in RAM and
computed by the CPU, MTP speculative decoding, behind Strata's own server (OpenAI and Anthropic APIs).  Coding work
was the target.  The single-request speeds below were measured on one RTX PRO 6000 Blackwell Max-Q (GPU 0), a
Threadripper 9960X and 512 GB of DDR5-6400, Windows 11.

## What it does

| | Strata-MiMo | llama.cpp `5ad1c5d`, best setting |
| --- | ---: | ---: |
| Output, short coding prompts | 117-125 tok/s | 33 tok/s |
| Output after a 19K-token code prompt (through the API) | 86.6 tok/s | - |
| Output after a 68K-token code prompt | 108-115 tok/s | - |
| Output after a 248,835-token prompt | 86.0 tok/s (prompt read in 111 s, 2,236 tok/s) | - |
| Prompt reading | 3,300 tok/s (17,881 tokens in 5.5 s, 67,723 in 20.5 s) | 96-178 tok/s |
| First token after a 19,457-token prompt (API) | 6.2 s | - |
| Context | 256K (8-bit KV) | |

Quality: against llama.cpp on the same tokens the final logits agree on the top token at 98.4% of positions (KL mean
0.0026) and the perplexity is the same within noise (1.7942 vs 1.7896); see [MIMO_DECODER.md](MIMO_DECODER.md).

## Installing it

Setup does everything below by itself (Windows: `cmd`; Linux: `./setup.sh` with the same flags):

```
START-HERE.bat --setup --yes --family mimo --no-start      (what it would do, changing nothing: add --dry-run)
run-mimo-mxfp4.bat                                          (starts the server on http://127.0.0.1:8080)
```

What it needs and checks first: an NVIDIA card with 32 GB of VRAM or more (one card; AMD is refused: the MXFP4 kernels
are CUDA), 192 GB of RAM or more (the engine keeps a pinned ~150 GB copy of the experts), ~170 GB of free disk for
the files it has to download.  Every number on this page is from the 96 GB card; with less VRAM setup warns that the
speed is unmeasured, and on a card other than sm_120 that it is untested.  The context is 256K on a card with 80 GB or
more and 64K below (`--context N` up to 262144).

What it does, each step skipped when it is already done:

1. checks the PC (as for Qwen) and the MiMo limits above; with several cards it takes the one with the most VRAM
   (`--gpu N` for another);
2. installs the Python packages and compiles `strata-mimo` for the card into `engine-mimo/` (build folder
   `build-mimo/`, CUDA 12.8 or newer: 42 s from an empty build folder on this PC, longer on a smaller CPU; compiled again
   after a `git pull` changes the engine's source);
3. gets the three files of `ggml-org/MiMo-V2.6-Flash-RL-GGUF` at revision `a5d1269f` (the two MXFP4 shards and
   `mtp-MiMo-V2.6-Flash-RL-Q8_0.gguf`, 169.8 GB).  A complete copy already on the PC is used instead of a download:
   in the model folder, in another Strata data folder, in `--gguf-dir`, or in Hugging Face's download cache
   (`HF_HUB_CACHE`, `HF_HOME/hub` or `~/.cache/huggingface/hub`, the pinned revision only).  On the same drive it is
   hard-linked into the model folder, on another drive it is used where it is (never copied).  Every file's size and
   SHA-256 are checked against the pinned values once (the 167 GB shard takes a few minutes);
4. extracts the tokenizer from shard 1 (`tools/strata_tokenizer.py`: `pre` = `qwen2`);
5. writes `strata-mimo-mxfp4.json` (below) and `run-mimo-mxfp4.bat` / `.sh`.

`START-HERE.bat` with no flags then starts it like a Qwen install (it asks which one when several are installed);
`UPDATE.bat` recompiles `strata-mimo` when its source changed.  The first start takes about 70 s (the pinned RAM copy
of the experts); the engine needs ~150 GB of free RAM and the whole GPU.

## By hand

Build (Windows, CUDA 12.9, VS 2022 Build Tools; the pinned ggml is fetched on the first configure):

```
cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build --target strata-mimo
```

(Setup's configure, with `-DSTRATA_BUILD_TESTS=OFF`, links without `-DCMAKE_CUDA_RUNTIME_LIBRARY=Shared`; a build with
the tests on needed it on this PC.)  The tokenizer: `python tools/strata_tokenizer.py --gguf <shard 1> --out <dir>`.
The expert profile: `data/mimo-coding-answers.prof`, made with `python tools/mimo/make_profile.py <out.prof>
<moe-trace file>` from the model's own answers to 44 coding prompts.  The config setup writes (`serve/server.py
--config` reads it; the server adds `--serve`):

```json
{
 "exe": "<Strata>/engine-mimo/strata-mimo.exe",
 "args": ["--native", "<...>/MiMo-V2.6-Flash-RL-MXFP4-00001-of-00002.gguf",
          "--mtp", "<...>/mtp-MiMo-V2.6-Flash-RL-Q8_0.gguf",
          "--expert-profile", "<Strata>/data/mimo-coding-answers.prof",
          "--max-context", "262144"],
 "cwd": "<Strata>",
 "tokenizer": "<Strata-data>/packs/mimo-mxfp4/tokenizer",
 "model_name": "mimo-v2.6-flash-rl",
 "log": "<Strata>/strata-mimo-mxfp4.log",
 "lib_dirs": ["<the CUDA toolkit's bin>"],
 "port": 8080, "family": "mimo", "gpu": 0, "gpus_asked": true
}
```

then `python serve/server.py --engine strata --config <that file> --host 127.0.0.1 --port 8080`.

## Two active requests

Single-slot serving remains the default. To enable two active text requests, add `"slots": 2` at the top level of
that server config, or pass `--slots 2` to `serve/server.py`. Rebuild `strata-mimo` first: an older engine without the
multiplex capability is rejected rather than silently running one request at a time. The OpenAI and Anthropic
request formats do not change.

Both requests keep their own KV cache in VRAM, token history, sampler and MTP draft state. They share one model,
one expert cache, one pinned RAM copy and one compute workspace. The engine alternates complete generation steps;
it does not execute two GPU forwards at once, and this is not a claim of doubled throughput. A third request waits
for a free slot (at most 32 waiting requests; further requests receive a queue-full error). Idle-slot affinity uses
terminal-confirmed prompt/output tokens and estimated live-ring/checkpoint availability; the engine still validates
exact reuse. A matching occupied slot does not indefinitely block another free slot. Canceling one request
does not cancel the other. A client that stops consuming output is canceled if its bounded token queue fills,
so it cannot block the other client's stream.

`--max-context` is **per slot**, not divided between requests. From the existing int8 cache layout (allocation
arithmetic, not a two-request measurement):

| Context per slot | Main KV, one slot | Main KV, two slots |
| --- | ---: | ---: |
| 65,536 | 0.754 GiB | 1.508 GiB |
| 131,072 | 1.484 GiB | 2.967 GiB |
| 262,144 | 2.943 GiB | 5.885 GiB |

Each MTP head adds about 0.65 MiB per slot. The second cache is allocated before sizing the resident experts; at
256K per slot it costs about 237 expert slots with the current 12.75 MiB expert layout. Startup reports the actual
resident count and total KV bytes. A fixed `--budget-gib` that no longer fits must be reduced; context is not
silently shortened.

In two-slot mode, prompt processing also yields after complete chunks, even if the second request has not arrived
yet. Native `--prefill-quantum N` (in the config's `args`) sets the bulk chunk ceiling; the default is 2,048 tokens.
While a peer is generating, prompt turns use at most 8 tokens; at the default `--batched-min 128`, these use the
decode-window path instead, avoiding bulk expert-copy/refill pauses in the live stream. Bulk chunks resume when
that peer finishes. When prefill competes with generation, scheduling targets 80% of measured compute-owner
service for decoding, not an equal number of unequal turns. Native `--decode-service-percent` (1–99, default 80)
and `--prefill-max-wait-ms` (1–60000, default 250) in the config's `args` tune this policy. An overdue prefill turn
wins even if that reduces the achieved decode share. Its wait bound includes the current nonpreemptible main/MTP
operation; this is not a hard real-time or token-rate guarantee. Two prefills or two decoders still alternate.
Prompt throughput while sharing can be lower. One-slot mode retains the existing bulk prompt path.

Long-turn reuse also keeps bounded checkpoints of the rolling SWA/MTP rings in ordinary system RAM. The large
global-attention KV stays in VRAM. Each slot retains a prompt-end checkpoint and geometric prompt milestones, at
most eight entries, under a 512 MiB combined native checkpoint budget. A checkpoint restores only when its exact
prefix and required global KV are still valid; otherwise the engine safely reads the missing prefix again. This is
not full-KV RAM swapping or arbitrary conversation storage. [Checkpoint and diagnostic details](MIMO_DECODER.md#long-turn-throughput-follow-up).

[Long-turn GPU validation](MIMO_DECODER.md#long-turn-gpu-validation-2026-10-16) passed checkpoint parity and repeated
25K–100K assistant/tool turns. Exact repeats processed one token instead of the full prefix. In two repetitions per
policy, a warmed 77K decoder beside a fresh 28K prefill delivered a median 32.4 tok/s with service weighting versus
11.6 tok/s with a round-robin control using the same decoder/checkpoints. These isolated GPU 1 measurements used a
64 GiB expert budget and mapped CPU experts; they are not a prediction of production tok/s or a full old-release
comparison. The earlier short-request measurements alone did not validate these workloads.

For validation beside another installation, native `--ram-copy 0` reads CPU experts from the existing GGUF mapping
instead of allocating another pinned ~150 GiB copy. The default is `--ram-copy 1`; the mapped path is not the
configuration used for the speed table above.

The speed table above describes single-slot measurements. [Two-slot validation](MIMO_DECODER.md#http-validation-2026-10-15)
on GPU 1 checked graph/MTP cache isolation, overlapping streams in both API dialects, cancellation, queued requests
and a slow reader. It used a smaller 64 GiB expert budget and mapped CPU experts while the other server stayed
loaded; its timings are not a prediction for the normal full-budget pinned-RAM configuration. The measurements
include the prompt-throughput cost of protecting an existing stream from a long new prompt.

## Starting and stopping

`start-mimo.bat [config]` (Linux: `./start-mimo.sh`) starts the server in its own window on the config's port (8095
when it has none).  The config is the argument, else `STRATA_MIMO_CONFIG`, else setup's `strata-mimo-mxfp4.json` in
the Strata folder, else `strata-mimo.json` there.  It is ready after about 90 s; Ctrl+C or closing the window stops
it.

`stop-mimo.bat` (`./stop-mimo.sh`) stops the server and its engine started from this folder, from wherever they were
started, including an engine left running after its server was killed.  It is a hard stop: the learned expert profile
keeps its last periodic save (every 10 minutes by default) rather than one at exit.

## Where things are

- [MIMO_PORT_PLAN.md](MIMO_PORT_PLAN.md): the plan and its steps.
- [MIMO_BASELINE.md](MIMO_BASELINE.md), [MIMO_ROUTING.md](MIMO_ROUTING.md): llama.cpp on this machine; expert routing on coding text.
- [MIMO_DECODER.md](MIMO_DECODER.md): the decoder (v0 to v2 + MTP), parity, the VRAM traps on Windows.
- [MIMO_GPU_EXPERTS.md](MIMO_GPU_EXPERTS.md), [MIMO_CPU_EXPERTS.md](MIMO_CPU_EXPERTS.md), [MIMO_ATTN.md](MIMO_ATTN.md),
  [MIMO_PREFILL_MOE.md](MIMO_PREFILL_MOE.md): the kernels and their measurements.
- Code: `src/mimo/` (model, decoder, prompt path, router, attention, experts), `src/program/mimo_serve.cpp` (the
  engine behind the server), `src/program/mimo_run.cpp` (benchmarks and dumps), `tools/mimo/` (traces, profiles,
  parity, benchmarks).

## Not done yet

- The start-up time (~70 s, the pinned copy of the experts).
- Setup's MiMo path was run here as far as `--dry-run`, the strata-mimo build and the tokenizer extraction (the
  tokenizer files equal the hand-made ones); a full install on a fresh PC is not yet done.
- When the server is killed hard, the engine process can outlive it (seen on this machine; Strata's job object is
  meant to prevent that).
- llama.cpp's large-batch MXFP4 MoE path is 11-31% off the exact result on this machine (MIMO_DECODER.md): not
  reported upstream yet.
