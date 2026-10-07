// include/strata/mimo/attn.hpp - MiMo-V2.6-Flash decode attention: rmsnorm, q/k/v split + partial NEOX rope,
// int8 KV append, split-K (flash-decoding) attention with GQA, the sliding window and sinks, and the residual add.
//
// One layer, a window of T = 1..8 consecutive query tokens at positions pos0 .. pos0+T-1 (T > 1 = speculative
// verify, causal among themselves). The math is llama.cpp's (src/models/mimo2.cpp, build_attn_mha with sinks,
// ggml soft_max with sinks):
//   q, k = NEOX rope on dims 0..63 of each 192-dim head (pairs i, i+32), theta_i = pos * powf(base^(-2/64), i)
//   s_j  = (q . k_j) / sqrt(192)           over keys j <= pos_q, and on SWA layers pos_q - j < n_swa (128)
//   a    = sum_j exp(s_j - M) v_j / (sum_j exp(s_j - M) + exp(sink_h - M))     (the sink term on SWA layers only)
// then x += value_scale * attn_output(a), with the attn_output GEMV between (not here).
//
// KV storage (one layer): int8 codes with one FP16 scale per 64 values, per position and KV head - the scheme of
// Strata's kv_q8 (scale = fp16(max|x| / 127), code = rint(x / scale) clamped to +-127). The 64-value group lines
// up with the head: K's first group is exactly the 64 roped dims and the other two the unroped ones, whose ranges
// differ, so a scale per group keeps both precise where one scale per 192-value row would not. Layout, head-major
// so a block streaming one KV head reads contiguous memory:
//   k [n_kv][cap][192] int8,  v [n_kv][cap][128] int8,  k_scale [n_kv][cap][4] f16 (3 used),  v_scale [n_kv][cap][2]
// = 332 bytes per position and KV head. Global layers index by position (cap = max context); sliding-window layers
// keep a ring (slot = pos % cap, cap >= n_swa + 8 - 1; 256 is used).
//
// Graph capture: nothing here reads a host value that changes per step. pos0 is read from device memory, the
// context length is pos0 + T, the grid is fixed (sized by the SM count), and blocks past the split count that
// this context needs return at once. T is a launch parameter: one captured graph per window size.
//
// Per layer and window the driver calls (all on one stream, capturable):
//   mimo_rmsnorm(x, attn_norm, h)  ->  attn_qkv GEMV (h -> qkv)  ->  mimo_qkv_rope_append(qkv, pos0, ..., q)
//   ->  mimo_attn_decode(q, pos0, ..., a)  ->  attn_output GEMV (a -> y)  ->  mimo_residual_add(x, y, value_scale)
// with pos0 one device int32 written before the graph launch (the first position of the window).
// Prompt chunks (P up to 4096 tokens) use mimo_qkv_rope_append_prompt + mimo_attn_prompt, declared at the end.
//
// Rollback: a rejected speculative token needs nothing - the next window starts at the accepted length, rewrites
// those positions, and attention never reads past pos0 + T. The ring is safe for the same reason: a window at pos0
// overwrites positions pos0 - cap .. pos0 + T - 1 - cap, all older than the oldest key any later window reads
// (pos0 - n_swa + 1) as long as cap >= n_swa + T - 1. Rewinding a conversation by more than cap - T positions on an
// SWA layer loses keys the ring no longer holds; that needs a re-prefill of the last n_swa tokens.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace strata::mimo {

inline constexpr int kAttnHeadK = 192, kAttnHeadV = 128, kAttnRot = 64, kAttnGroup = 64;
inline constexpr int kAttnMaxT = 8;        // query tokens per window
inline constexpr int kAttnMaxHeads = 64;   // query heads (n_head)
inline constexpr int kSwaRing = 256;       // ring capacity of a sliding-window layer

/// One layer's int8 KV cache in VRAM (layout above).
struct AttnKv {
    int8_t* k = nullptr;
    int8_t* v = nullptr;
    uint16_t* k_scale = nullptr;   // fp16 bits
    uint16_t* v_scale = nullptr;
    int n_kv = 0, cap = 0;
    bool ring = false;             // SWA: slot = pos % cap; global: slot = pos (pos < cap)
};

/// Bytes of one layer's cache: n_kv * cap * 332.
size_t attn_kv_bytes(int n_kv, int cap);
/// One allocation, zeroed. False with `err` on failure.
bool attn_kv_alloc(AttnKv& kv, int n_kv, int cap, bool ring, std::string& err);
void attn_kv_free(AttnKv& kv);

/// The per-layer constants the kernels need.
struct AttnLayer {
    int n_head = 64, n_kv = 4;
    bool swa = false;
    int n_swa = 128;
    float theta_scale = 0;          // powf(rope base, -2 / 64), as ggml computes it
    float scale = 0;                // 1 / sqrt(192)
    const float* sinks = nullptr;   // device F32 [n_head] (SWA layers), nullptr = none
};
/// Fills AttnLayer from the GGUF numbers (rope base: rope_base or rope_base_swa).
AttnLayer attn_layer(int n_head, int n_kv, bool swa, int n_swa, float rope_base, int head_k, const float* sinks);

/// Split-K partials, shared by every layer (one layer runs at a time): one 128-row partial per (split, KV head) block.
struct AttnScratch {
    float* o = nullptr;    // [blocks][128 rows][128]: unnormalized output, block = split * n_kv + KV head
    float* ml = nullptr;   // [blocks][128 rows][2]: running max (log2 units) and denominator
    int blocks = 0;
};
/// blocks <= 0: the SM count (12.3 MB on 188 SMs), which every default launch fits (splits = SMs / n_kv).
bool attn_scratch_alloc(AttnScratch& s, int blocks, std::string& err);
void attn_scratch_free(AttnScratch& s);
/// The split count that fills the GPU for a layer with n_kv KV heads (one 8-warp block per SM).
int attn_default_splits(int n_kv);

/// y[t] = x[t] / sqrt(mean(x[t]^2) + eps) * w, for T rows of n (F32).
void mimo_rmsnorm(const float* x, const float* w, float* y, int n, int T, float eps, void* stream);

/// qkv [T][rows] F32 (rows = 192 * (n_head + n_kv) + 128 * n_kv: q, then k, then v) -> roped q [T][n_head][192] F32,
/// and k (roped) and v quantized into `kv` at positions pos0[0] + t.
void mimo_qkv_rope_append(const float* qkv, const int32_t* pos0, int T, const AttnLayer& L, const AttnKv& kv,
                          float* q_out, void* stream);

/// Attention of the T roped queries over keys [lo, pos0 + T) (lo = 0, or pos0 - n_swa + 1 on SWA layers) of `kv`,
/// which must already hold the window's own keys (mimo_qkv_rope_append first). out: [T][n_head][128] F32.
/// max_splits <= 0 uses attn_default_splits(L.n_kv) (capped at s.blocks / n_kv). SWA layers with T = 1 use one
/// block per KV head and no combine kernel.  out_q8_1: when set, `out` is also written there as Q8_1 blocks
/// (native_quantize_q8_1's codes and scales: the input of the attn_output MMVQ), by the combine kernel.
void mimo_attn_decode(const float* q, const int32_t* pos0, int T, const AttnLayer& L, const AttnKv& kv,
                      const AttnScratch& s, float* out, void* stream, int max_splits = 0, void* out_q8_1 = nullptr);

/// x += scale * y (n floats): the residual add after attn_output, scale = value_scale.
void mimo_residual_add(float* x, const float* y, float scale, int n, void* stream);

// ---- Prompt processing (prefill): a chunk of P tokens at positions pos0 .. pos0+P-1 -----------------------------
//
// Same cache, same math, same quantizer as the decode path above, so a prompt chunk and the decode windows after it
// (or chunks of any other size) leave the same int8 codes and give the same outputs up to the kernels' roundings.
// pos0 is a host int here (prefill is not graph-captured: the grid depends on P and the split-K plan on pos0).
//
// Sliding-window layers: the ring holds 256 positions, but a chunk of P > 129 tokens needs, for its queries, up to
// n_swa - 1 keys before each of them - more than the ring can hold at once while the chunk is being written. So the
// chunk goes through a linear staging buffer (AttnPromptScratch::swa, the same int8 format) that holds positions
// pos0 - n_swa .. pos0 + P - 1 at rows 0 .. n_swa + P - 1:
//   mimo_qkv_rope_append_prompt  1. copies the ring's positions pos0 - n_swa .. pos0 - 1 into staging rows 0..n_swa-1
//                                2. ropes and quantizes the chunk into staging rows n_swa .. n_swa + P - 1, and the
//                                   last min(P, 256) of its positions also into the ring (slot = pos % 256), which
//                                   then holds positions >= pos0 + P - 256: what the next decode window or chunk reads
//   mimo_attn_prompt             reads the staging only (the ring is not read, so step 2 may overwrite it freely).
// The staging is shared by all SWA layers: each layer's append must be followed by its own attention before the
// next SWA layer's append. Global layers read and write their cache by position directly (cap >= pos0 + P).
//
// Attention runs on the tensor cores (mma.sync m16n8k16, f16 in, f32 accumulate), flash-attention style: a block
// takes 128 query rows of one KV head (rows r = t * G + head-in-group, so 8 tokens on a global layer, 16 on an SWA
// one, share every K/V tile) and streams 64-key tiles, each dequantized once into shared memory as f16. Long prior
// contexts with few blocks split the keys (split-K) and merge in a second kernel, through AttnPromptScratch::o / ml.

inline constexpr int kAttnPromptMaxP = 4096;   // default chunk capacity of the prompt scratch

/// The prompt path's scratch, shared by every layer.
struct AttnPromptScratch {
    AttnKv swa;                  // SWA staging: n_kv_swa heads x (max_p + n_swa) linear rows, the cache's format
    int max_p = 0, n_swa = 0;
    float* o = nullptr;          // split-K partials: [rows][128] unnormalized outputs
    float* ml = nullptr;         // [rows][2]: running max (log2 units) and denominator
    size_t part_rows = 0;        // capacity of o / ml in rows (splits x n_kv x padded query rows of one chunk)
};
/// max_p <= 0: kAttnPromptMaxP. part_bytes: the split-K partial buffer (0: 64 MiB; it only limits the split count).
bool attn_prompt_scratch_alloc(AttnPromptScratch& s, int max_p, int n_kv_swa, int n_swa, size_t part_bytes,
                               std::string& err);
void attn_prompt_scratch_free(AttnPromptScratch& s);

/// qkv [P][rows] F32 (the decode path's layout) -> roped q [P][n_head][192] as fp16 bits (q_out), k (roped) and v
/// quantized into the cache (global) or the staging + ring (SWA), as described above. 1 <= P <= s.max_p.
void mimo_qkv_rope_append_prompt(const float* qkv, int pos0, int P, const AttnLayer& L, const AttnKv& kv,
                                 const AttnPromptScratch& s, uint16_t* q_out, void* stream);

/// Causal attention of the P roped queries (fp16, from mimo_qkv_rope_append_prompt of the same layer and chunk)
/// over keys [0, pos0 + t] (global) or (pos0 + t - n_swa, pos0 + t] (SWA, from the staging), GQA, sinks.
/// out: [P][n_head][128] F32. splits <= 0 chooses the split-K count itself (1 unless the prior context is long and
/// the chunk's blocks would not fill the GPU evenly).
void mimo_attn_prompt(const uint16_t* q, int pos0, int P, const AttnLayer& L, const AttnKv& kv,
                      const AttnPromptScratch& s, float* out, void* stream, int splits = 0);

}  // namespace strata::mimo
