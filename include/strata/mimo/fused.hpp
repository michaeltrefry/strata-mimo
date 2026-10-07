// include/strata/mimo/fused.hpp - the decode window's small steps merged into fewer kernels (docs/MIMO_DECODER.md).
//
// Each call gives the same numbers as the separate kernels it replaces (mimo_residual_add / add_scaled /
// copy_from_mapped, mimo_rmsnorm, native_quantize_q8_1, router_logits + router_topk + resident_slots +
// doorbell_publish): the same expressions in the same order, so the outputs are bitwise equal.  The one exception is
// the q8_1 blocks' second half (the block sum, which Q8_0 MMVQ does not read): it is summed in another order.
// All pointers are device pointers (or mapped host memory where noted); every call can be graph-captured.
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace strata::mimo {

/// For T rows of n floats (n a multiple of 32, at most 16384):
///   x[t] = x[t] + scale * (y[t] + y2[t])     when y is set (y2 may be null; it is read as mapped host memory: the
///                                            CPU experts' sums, after wait_flag_ge)
///   h[t] = rmsnorm(x[t]) * w                 (mimo_rmsnorm)
///   q8   = h as Q8_1 blocks                  when q8 is set (native_quantize_q8_1's layout and values), the input of
///                                            native_*_mmvq for the next GEMV
///   y[t] = 0                                 when zero_y (after it is read): the MoE output's accumulator for
///                                            mxfp4_experts, instead of a fill_zero kernel
///   h_out[t] = h[t]                          when h_out is set (mapped host memory, system-fenced): the doorbell's
///                                            copy of the experts' input, which mimo_router_fused then rings
///   xq       = h in mxfp4_experts' input     when xq is set (n = kMxEmbd; the experts' scratch, see kMxXqBytes):
///              layout                        then mxfp4_experts runs with x = nullptr, without its first kernel
/// With scale = 1 the add equals add_scaled; with y2 it equals copy_from_mapped + add_scaled(y, y2) + add_scaled(x, y).
void mimo_add_rmsnorm_q8_1(float* x, float* y, const float* y2, float scale, const float* w, float* h, void* q8,
                           bool zero_y, int n, int T, float eps, cudaStream_t s, float* h_out = nullptr,
                           void* xq = nullptr);

/// The MoE layer's router in one launch (router_logits, router_topk, resident_slots and doorbell_publish's ids,
/// weights and ring): logits of T <= 8 tokens over n_expert <= 256 experts (gate_inp F32 [n_embd x n_expert], n_embd
/// a multiple of 128; one block per expert, as router_logits), then the last block to finish picks each token's
/// n_used experts and writes ids / weights [T][n_used].
///   res_layer, slots   when set: slots[i] = res_layer[ids[i]] (the resident slot, or -1)
///   ids_out .. seq     when set (mapped host memory): ids and weights copied there and *seq incremented after a
///                      system fence - the doorbell of the host loop (h itself: mimo_add_rmsnorm_q8_1's h_out)
/// `logits` is [T][n_expert] scratch.  Calls on different streams must not overlap (one completion counter).
void mimo_router_fused(const float* gate_inp, const float* bias, const float* h, float* logits, int32_t* ids,
                       float* weights, int T, int n_embd, int n_expert, int n_used, float weights_scale,
                       const int32_t* res_layer, int32_t* slots, int32_t* ids_out, float* w_out, uint32_t* seq,
                       cudaStream_t s);

}  // namespace strata::mimo
