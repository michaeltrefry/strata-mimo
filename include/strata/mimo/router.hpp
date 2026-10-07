// include/strata/mimo/router.hpp - MiMo's expert router for T tokens (llama.cpp build_moe_ffn, sigmoid gating):
//
//   logits = gate_inp . h            (F32 [n_embd x n_expert], h F32)
//   p      = sigmoid(logits)
//   picked = top n_used of (p + exp_probs_b)      (the bias only chooses; it is not part of the weight)
//   w      = p[picked] / max(sum p[picked], 6.103515625e-5)   (* expert_weights_scale when it is set and != 1)
//
// Everything is read from device memory and the launch shape depends only on (T, n_expert), so the two launches
// can be captured in a CUDA graph.  Output order within a token's picks is by descending selection score.
#pragma once

#include <cuda_runtime.h>

namespace strata::mimo {

inline constexpr int kRouterMaxT = 8;
inline constexpr int kRouterMaxUsed = 16;

/// logits[t * n_expert + e] = sum_i w[e * n_embd + i] * h[t * n_embd + i]   for t < T (T <= kRouterMaxT).
/// n_embd must be a multiple of 128.
void router_logits(const float* w, const float* h, float* logits, int T, int n_embd, int n_expert, cudaStream_t s);

/// Picks n_used experts per token from `logits` (see above). ids[t * n_used + j], weights[t * n_used + j].
/// n_expert <= 1024, n_used <= kRouterMaxUsed.
void router_topk(const float* logits, const float* bias, int* ids, float* weights, int T, int n_expert, int n_used,
                 float weights_scale, cudaStream_t s);

}  // namespace strata::mimo
