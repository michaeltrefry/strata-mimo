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

/// The same logits for any number of tokens (prompt chunks): one FP32 GEMM through cuBLAS with its default math
/// (no TF32: a reduced-precision router flips near-tie picks, as llama.cpp's batched GPU router does).  `cublas` is a
/// cublasHandle_t bound to the stream by the caller.
void router_logits_batched(void* cublas, const float* w, const float* h, float* logits, int T, int n_embd, int n_expert);

/// Picks n_used experts per token from `logits` (see above). ids[t * n_used + j], weights[t * n_used + j].
/// n_expert <= 1024, n_used <= kRouterMaxUsed, any T (one block per token).
void router_topk(const float* logits, const float* bias, int* ids, float* weights, int T, int n_expert, int n_used,
                 float weights_scale, cudaStream_t s);

}  // namespace strata::mimo
