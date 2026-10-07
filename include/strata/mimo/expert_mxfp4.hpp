// include/strata/mimo/expert_mxfp4.hpp - MiMo's routed experts (MXFP4, ggml type 39) on the GPU: the resident
// ("hit") experts of one MoE layer for a window of T = 1..8 tokens, summed into the layer's output.
//
// One call computes, for every token t and each of its K routed (slot, weight) pairs with slot >= 0:
//     out[t] += weight * down(silu(gate(x[t])) * up(x[t]))
// The pairs with slot < 0 (experts in RAM) are skipped: the CPU adds their share to `out` afterwards.
//
// Slot format.  A slot holds one expert in a re-laid-out form of the GGUF's MXFP4 blocks (same bytes, same size):
//     [gate codes | up codes | down codes | gate scales | up scales | down scales]
//     codes:  row-major, each row the 16-byte `qs` of its blocks in block order (gate/up rows: 128 blocks = 2048 B,
//             down rows: 64 blocks = 1024 B) - every row 16-byte aligned, read with 128-bit loads
//     scales: each row's E8M0 bytes, permuted so lane i of a warp reads its blocks' scales in one load:
//             gate/up row: byte 4i + k = scale of block i + 32k (k < 4); down row: byte 2i + k = block i + 32k (k < 2)
// 13,369,344 bytes per expert, as in the GGUF.  The GGUF (and the CPU path reading it) is unchanged; a slot is
// filled with mxfp4_slot_from_gguf (host, e.g. into the pinned staging buffer) or mxfp4_slot_relayout (device).
//
// The numbers are llama.cpp's: activations quantized to q8_1 (32 values, scale amax/127 rounded to fp16),
// vec_dot_mxfp4_q8_1's integer dot (kvalues_mxfp4 doubled, E8M0 scale * 0.5), SiLU, the hidden state quantized
// to q8_1 again before the down matrix.  Sums run in a different order than ggml's, so results agree to float
// rounding, not bit for bit; they are deterministic (no atomics), and the same input gives the same bits.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::mimo {

inline constexpr int kMxEmbd = 4096, kMxFF = 2048;             // MiMo-V2.6-Flash n_embd, n_ff_exp
inline constexpr int kMxMaxT = 8, kMxMaxK = 8;                 // window tokens, routed experts per token
inline constexpr size_t kMxGuCodes = (size_t) kMxFF * (kMxEmbd / 2);      // 4 MiB per gate or up matrix
inline constexpr size_t kMxDownCodes = (size_t) kMxEmbd * (kMxFF / 2);    // 4 MiB
inline constexpr size_t kMxGuScales = (size_t) kMxFF * (kMxEmbd / 32);    // 256 KiB
inline constexpr size_t kMxDownScales = (size_t) kMxEmbd * (kMxFF / 32);  // 256 KiB
inline constexpr size_t kMxSlotBytes = 2 * kMxGuCodes + kMxDownCodes + 2 * kMxGuScales + kMxDownScales;
static_assert(kMxSlotBytes == 13369344, "one MXFP4 expert of MiMo-V2.6-Flash");

/// One expert from its three GGUF matrices (gate and up: kMxFF rows of kMxEmbd values, down: kMxEmbd rows of
/// kMxFF; raw block_mxfp4 rows) into the slot format.  Host memory on both sides.
void mxfp4_slot_from_gguf(const uint8_t* gate, const uint8_t* up, const uint8_t* down, uint8_t* slot);
/// The same on the device: `raw` holds gate | up | down in GGUF layout (kMxSlotBytes, e.g. DMA'd from RAM), `slot`
/// receives the slot format.  They must not overlap.  Enqueued on `stream`.
void mxfp4_slot_relayout(const void* raw, void* slot, void* stream);

/// Device scratch for a call with T tokens and K experts per token.
size_t mxfp4_experts_scratch_bytes(int T, int K);

/// The hit experts of one layer.  All pointers are device pointers; everything per token is read on the device,
/// and the launch shapes depend on T and K only, so the call can be captured in a CUDA graph and replayed with new
/// contents in the buffers.
///   slot_ptr  device address of each slot (16-byte aligned), indexed by slot number
///   slots     [T][K] int32: the slot of each routed expert, or < 0 when it is not resident.  A token's slots must be
///             distinct (a repeat within one token is ignored)
///   weights   [T][K] float: the router weights
///   x         [T][kMxEmbd] float: the FFN input (after ffn_norm)
///   out       [T][kMxEmbd] float: added to, never overwritten
/// T in 1..kMxMaxT, K in 1..kMxMaxK.
void mxfp4_experts(const unsigned long long* slot_ptr, const int32_t* slots, const float* weights, const float* x,
                   float* out, int T, int K, void* scratch, void* stream);

}  // namespace strata::mimo
