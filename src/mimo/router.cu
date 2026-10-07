#include "strata/mimo/router.hpp"

#include <cfloat>
#include <climits>

namespace strata::mimo {
namespace {

constexpr int kLogitThreads = 128;

// One block per expert (enough blocks to fill the GPU): the expert's n_embd weights are read once (float4,
// coalesced) and dotted with all T tokens; the four warps' partial sums meet in shared memory.
template <int T>
__global__ void logits_kernel(const float* __restrict__ w, const float* __restrict__ h, float* __restrict__ out,
                              int n_embd, int n_expert) {
    __shared__ float part[T][kLogitThreads / 32];
    const int e = blockIdx.x;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const float4* wr = reinterpret_cast<const float4*>(w + (size_t) e * n_embd);
    float acc[T];
#pragma unroll
    for (int t = 0; t < T; ++t) acc[t] = 0.f;
    for (int i = threadIdx.x; i < n_embd / 4; i += kLogitThreads) {
        const float4 a = wr[i];
#pragma unroll
        for (int t = 0; t < T; ++t) {
            const float4 b = reinterpret_cast<const float4*>(h + (size_t) t * n_embd)[i];
            acc[t] = fmaf(a.x, b.x, fmaf(a.y, b.y, fmaf(a.z, b.z, fmaf(a.w, b.w, acc[t]))));
        }
    }
#pragma unroll
    for (int t = 0; t < T; ++t) {
        float v = acc[t];
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (lane == 0) part[t][warp] = v;
    }
    __syncthreads();
    if (threadIdx.x < T) {
        float v = 0.f;
        for (int k = 0; k < kLogitThreads / 32; ++k) v += part[threadIdx.x][k];
        out[(size_t) threadIdx.x * n_expert + e] = v;
    }
}

// One block per token, one thread per expert: n_used rounds of a block-wide argmax over the selection score.
// Ties go to the lower expert id.
__global__ void topk_kernel(const float* __restrict__ logits, const float* __restrict__ bias, int* __restrict__ ids,
                            float* __restrict__ weights, int n_expert, int n_used, float weights_scale) {
    __shared__ float s_val[32];
    __shared__ int s_idx[32];
    __shared__ int s_pick[kRouterMaxUsed];
    const int t = blockIdx.x;
    const int e = threadIdx.x;
    const int lane = e & 31, warp = e >> 5, n_warps = (blockDim.x + 31) >> 5;
    float p = 0.f, sel = -FLT_MAX;
    if (e < n_expert) {
        p = 1.f / (1.f + expf(-logits[(size_t) t * n_expert + e]));
        sel = p + (bias ? bias[e] : 0.f);
    }
    for (int j = 0; j < n_used; ++j) {
        float v = sel;
        int idx = e < n_expert ? e : INT_MAX;
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, v, o);
            const int oi = __shfl_xor_sync(0xffffffffu, idx, o);
            if (ov > v || (ov == v && oi < idx)) { v = ov; idx = oi; }
        }
        if (lane == 0) { s_val[warp] = v; s_idx[warp] = idx; }
        __syncthreads();
        if (warp == 0) {
            v = lane < n_warps ? s_val[lane] : -FLT_MAX;
            idx = lane < n_warps ? s_idx[lane] : INT_MAX;
            for (int o = 16; o > 0; o >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffffu, v, o);
                const int oi = __shfl_xor_sync(0xffffffffu, idx, o);
                if (ov > v || (ov == v && oi < idx)) { v = ov; idx = oi; }
            }
            if (lane == 0) s_pick[j] = idx;
        }
        __syncthreads();
        if (e == s_pick[j]) sel = -FLT_MAX;   // out of the next rounds
    }
    // weights: the unbiased p of the picked experts, normalized over the picks
    __shared__ float s_p[kRouterMaxUsed];
    for (int j = 0; j < n_used; ++j)
        if (e == s_pick[j]) s_p[j] = p;
    __syncthreads();
    if (e == 0) {
        float sum = 0.f;
        for (int j = 0; j < n_used; ++j) sum += s_p[j];
        sum = fmaxf(sum, 6.103515625e-5f);
        const bool scale = weights_scale != 0.f && weights_scale != 1.f;
        for (int j = 0; j < n_used; ++j) {
            ids[t * n_used + j] = s_pick[j];
            weights[t * n_used + j] = s_p[j] / sum * (scale ? weights_scale : 1.f);
        }
    }
}

}  // namespace

void router_logits(const float* w, const float* h, float* logits, int T, int n_embd, int n_expert, cudaStream_t s) {
    const dim3 grid(n_expert), block(kLogitThreads);
    switch (T) {
#define STRATA_ROUTER_T(n) case n: logits_kernel<n><<<grid, block, 0, s>>>(w, h, logits, n_embd, n_expert); break;
        STRATA_ROUTER_T(1) STRATA_ROUTER_T(2) STRATA_ROUTER_T(3) STRATA_ROUTER_T(4)
        STRATA_ROUTER_T(5) STRATA_ROUTER_T(6) STRATA_ROUTER_T(7) STRATA_ROUTER_T(8)
#undef STRATA_ROUTER_T
        default: break;
    }
}

void router_topk(const float* logits, const float* bias, int* ids, float* weights, int T, int n_expert, int n_used,
                 float weights_scale, cudaStream_t s) {
    const int threads = (n_expert + 31) / 32 * 32;
    topk_kernel<<<T, threads, 0, s>>>(logits, bias, ids, weights, n_expert, n_used, weights_scale);
}

}  // namespace strata::mimo
