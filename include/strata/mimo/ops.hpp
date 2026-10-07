// include/strata/mimo/ops.hpp - the small element-wise pieces of MiMo's decode step.  All pointers are device
// pointers; every call enqueues on `s` and reads its sizes from the arguments only, so it can be graph-captured.
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace strata::mimo {

/// out[t * n_embd + i] = dequant(token_embd row ids[t])   (Q8_0 rows: 34 bytes per 32 values); ids on the device.
void embed_q8_0(const void* table, const int32_t* ids, float* out, int T, int n_embd, cudaStream_t s);

/// out[i] = silu(gate[i]) * up[i] for n = T * n_ff values (gate and up are separate buffers).
void silu_mul(const float* gate, const float* up, float* out, int n, cudaStream_t s);

/// x[i] += scale * y[i] for n values.
void add_scaled(float* x, const float* y, float scale, int n, cudaStream_t s);

/// x[i] = 0 for n values (a kernel, so a captured graph needs no memset node).
void fill_zero(float* x, int n, cudaStream_t s);

/// slots[i] = res_layer[ids[i]] for n routed experts: the resident slot, or -1 (computed by the CPU).
void resident_slots(const int32_t* ids, const int32_t* res_layer, int32_t* slots, int n, cudaStream_t s);

/// ids[t] = argmax_v logits[t * n_vocab + v] (ties: the lower id), for T rows.
void argmax_rows(const float* logits, int32_t* ids, int T, int n_vocab, cudaStream_t s);

}  // namespace strata::mimo
