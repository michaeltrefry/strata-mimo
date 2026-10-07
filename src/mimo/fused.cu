// src/mimo/fused.cu - see include/strata/mimo/fused.hpp.  The expressions are copied from the kernels they replace
// (attn.cu rmsnorm_kernel / residual_add_kernel, ops.cu add_scaled_kernel, router.cu logits_kernel / topk_kernel),
// in a translation unit with the same (default) float flags, so the results are bitwise theirs.
#include "strata/mimo/fused.hpp"

#include "q8_1.cuh"
#include "strata/mimo/expert_mxfp4.hpp"
#include "strata/mimo/router.hpp"

#include <algorithm>
#include <cfloat>
#include <climits>
#include <cstdio>
#include <cstdlib>

namespace strata::mimo {
namespace {

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "mimo fused: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

// ---------------------------------------------------------------- add + rmsnorm + q8_1
// One block per row, as rmsnorm_kernel: each thread keeps its float4s of x in registers between the two passes.
// Every load is issued before the first store (x, y and the norm weights): a store between them would serialize the
// loads (the pointers may alias as far as the compiler knows), one memory latency each - PCIe latency for y2.
__global__ void __launch_bounds__(1024) add_rmsnorm_q8_1_kernel(float* __restrict__ x, float* __restrict__ y,
                                                                const float* __restrict__ y2, float scale,
                                                                const float* __restrict__ w, float* __restrict__ h,
                                                                MimoQ81* __restrict__ q8, float* h_out,
                                                                uint8_t* __restrict__ xq, int zero_y, int n,
                                                                float eps) {
    const size_t row = (size_t) blockIdx.x * n;
    float4* xr = reinterpret_cast<float4*>(x + row);
    float4* yr = y ? reinterpret_cast<float4*>(y + row) : nullptr;
    const float4* y2r = y2 ? reinterpret_cast<const float4*>(y2 + row) : nullptr;
    float4* hr = reinterpret_cast<float4*>(h + row);
    const float4* wr = reinterpret_cast<const float4*>(w);
    const int n4 = n / 4;
    const float4 zero = make_float4(0, 0, 0, 0);
    float4 v[4], a[4], b[4], ww[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int i = threadIdx.x + k * blockDim.x;
        const bool in = i < n4;
        v[k] = in ? xr[i] : zero;
        a[k] = in && yr ? yr[i] : zero;
        b[k] = in && y2r ? __ldcv(y2r + i) : zero;   // mapped memory the host wrote: not from a cache
        ww[k] = in ? wr[i] : zero;
    }
    float s = 0.0f;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int i = threadIdx.x + k * blockDim.x;
        if (yr && i < n4) {
            // add_scaled(y, y2, 1): y + 1 * y2 is one rounding of y + y2
            if (y2r) a[k] = make_float4(a[k].x + b[k].x, a[k].y + b[k].y, a[k].z + b[k].z, a[k].w + b[k].w);
            // residual_add_kernel: scale, then add (no fma); with scale = 1 it is add_scaled's x + y
            v[k] = make_float4(
                __fadd_rn(v[k].x, __fmul_rn(a[k].x, scale)), __fadd_rn(v[k].y, __fmul_rn(a[k].y, scale)),
                __fadd_rn(v[k].z, __fmul_rn(a[k].z, scale)), __fadd_rn(v[k].w, __fmul_rn(a[k].w, scale)));
            xr[i] = v[k];
            if (zero_y) yr[i] = zero;
        }
        s += v[k].x * v[k].x + v[k].y * v[k].y + v[k].z * v[k].z + v[k].w * v[k].w;
    }
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    __shared__ float part[32];
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = s;
    __syncthreads();
    if (threadIdx.x < 32) {
        s = threadIdx.x < (blockDim.x >> 5) ? part[threadIdx.x] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        if (threadIdx.x == 0) part[0] = s;
    }
    __syncthreads();
    const float r = rsqrtf(part[0] / (float) n + eps);
    MimoQ81* qr = q8 ? q8 + (size_t) blockIdx.x * (n / 32) : nullptr;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int i = threadIdx.x + k * blockDim.x;
        float4 o = zero;
        if (i < n4) {
            o = make_float4((v[k].x * r) * ww[k].x, (v[k].y * r) * ww[k].y, (v[k].z * r) * ww[k].z,
                            (v[k].w * r) * ww[k].w);
            hr[i] = o;
            if (h_out) reinterpret_cast<float4*>(h_out + row)[i] = o;   // the doorbell's copy (mapped)
        }
        if (qr && k * blockDim.x < n4) q8_1_quad(o, i < n4, qr + i / 8);   // warp-uniform condition
        if (xq) {   // expert_mxfp4.cu prepare_kernel (256 threads: float4 i of the row is its v[k]), same expressions
            float amax = fmaxf(fmaxf(fabsf(o.x), fabsf(o.y)), fmaxf(fabsf(o.z), fabsf(o.w)));
#pragma unroll
            for (int off = 4; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
            const float d = amax / 127.0f;
            int q = 0;
            if (amax != 0.0f)
                q = (int) (uint8_t) (int8_t) roundf(o.x / d) | (int) (uint8_t) (int8_t) roundf(o.y / d) << 8 |
                    (int) (uint8_t) (int8_t) roundf(o.z / d) << 16 | (int) (uint8_t) (int8_t) roundf(o.w / d) << 24;
            const int c = threadIdx.x % 8, bl = threadIdx.x / 8;
            uint8_t* xt = xq + (size_t) blockIdx.x * kMxXqBytes;
            reinterpret_cast<int*>(xt)[((k * 2 + c / 4) * 32 + bl) * 4 + c % 4] = q;
            if (c == 0) reinterpret_cast<float*>(xt + 4096)[k * 32 + bl] = __half2float(__float2half_rn(d));
        }
    }
    if (h_out) __threadfence_system();   // the host sees h before the router rings the doorbell
}

// ---------------------------------------------------------------- router
constexpr int kLogitThreads = 128;          // router_logits' block: one expert per block
constexpr int kMaxExpert = 256, kPerLane = kMaxExpert / 32;

__device__ unsigned int g_router_done = 0;   // blocks finished; the last one resets it

template <int T>
__global__ void __launch_bounds__(kLogitThreads) router_fused_kernel(
    const float* __restrict__ w, const float* __restrict__ h, const float* __restrict__ bias, float* logits,
    int32_t* __restrict__ ids, float* __restrict__ weights, int n_embd, int n_expert, int n_used, float weights_scale,
    const int32_t* __restrict__ res, int32_t* __restrict__ slots, int32_t* ids_out, float* w_out, uint32_t* seq) {
    __shared__ float part[T][kLogitThreads / 32];
    __shared__ int s_last;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int e = blockIdx.x;
    {   // logits_kernel: the same loads, fma chain and reductions
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
            logits[(size_t) threadIdx.x * n_expert + e] = v;
        }
    }
    // loaded by every block while the logits drain: the last one needs them, and it does not know it is last yet
    float bj[kPerLane];
#pragma unroll
    for (int j = 0; j < kPerLane; ++j) bj[j] = bias && lane + 32 * j < n_expert ? bias[lane + 32 * j] : 0.f;
    __shared__ int32_t s_res[kMaxExpert];
    if (res)
        for (int i = threadIdx.x; i < n_expert; i += kLogitThreads) s_res[i] = res[i];
    // the last block to finish sees every block's logits (threadFenceReduction)
    if (threadIdx.x < T) __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) s_last = atomicAdd(&g_router_done, 1u) == gridDim.x - 1;
    __syncthreads();
    if (!s_last) return;
    __threadfence();

    // topk_kernel, one warp per token: n_used rounds of an argmax over (score, lower id); the same picks and weights
    for (int t = warp; t < T; t += kLogitThreads / 32) {
        float p[kPerLane], sel[kPerLane];
#pragma unroll
        for (int j = 0; j < kPerLane; ++j) {
            const int ej = lane + 32 * j;
            p[j] = 0.f;
            sel[j] = -FLT_MAX;
            if (ej < n_expert) {
                p[j] = 1.f / (1.f + expf(-__ldcg(logits + (size_t) t * n_expert + ej)));
                sel[j] = p[j] + (bias ? bj[j] : 0.f);
            }
        }
        int pick = 0;      // lane r keeps pick r
        float pp = 0.f;
        for (int r = 0; r < n_used; ++r) {
            float v = sel[0];
            int idx = lane < n_expert ? lane : INT_MAX;
#pragma unroll
            for (int j = 1; j < kPerLane; ++j)
                if (sel[j] > v) { v = sel[j]; idx = lane + 32 * j; }   // ids grow with j: a tie keeps the lower
            for (int o = 16; o > 0; o >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffffu, v, o);
                const int oi = __shfl_xor_sync(0xffffffffu, idx, o);
                if (ov > v || (ov == v && oi < idx)) { v = ov; idx = oi; }
            }
            float mine = 0.f;
#pragma unroll
            for (int j = 0; j < kPerLane; ++j)
                if (lane + 32 * j == idx) { mine = p[j]; sel[j] = -FLT_MAX; }   // out of the next rounds
            mine = __shfl_sync(0xffffffffu, mine, idx & 31);
            if (lane == r) { pick = idx; pp = mine; }
        }
        float sum = 0.f;   // in pick order, as topk_kernel's thread 0
        for (int r = 0; r < n_used; ++r) sum += __shfl_sync(0xffffffffu, pp, r);
        sum = fmaxf(sum, 6.103515625e-5f);
        const bool scale = weights_scale != 0.f && weights_scale != 1.f;
        if (lane < n_used) {
            const int i = t * n_used + lane;
            const float wt = pp / sum * (scale ? weights_scale : 1.f);
            ids[i] = pick;
            weights[i] = wt;
            if (slots) slots[i] = s_res[pick];
            if (ids_out) {
                ids_out[i] = pick;
                w_out[i] = wt;
            }
        }
    }
    if (threadIdx.x == 0) g_router_done = 0;   // for the next launch (stream order)
    if (seq) {   // doorbell_publish's ring
        __threadfence_system();
        __syncthreads();
        if (threadIdx.x == 0) {
            __threadfence_system();
            *(volatile uint32_t*) seq = *(volatile uint32_t*) seq + 1u;
        }
    }
}

}  // namespace

void mimo_add_rmsnorm_q8_1(float* x, float* y, const float* y2, float scale, const float* w, float* h, void* q8,
                           bool zero_y, int n, int T, float eps, cudaStream_t s, float* h_out, void* xq) {
    if (n % 32 != 0 || n > 16 * 1024) {
        std::fprintf(stderr, "mimo_add_rmsnorm_q8_1: n = %d must be a multiple of 32 and at most 16384\n", n);
        std::exit(1);
    }
    if (xq && n != kMxEmbd) {
        std::fprintf(stderr, "mimo_add_rmsnorm_q8_1: the experts' input needs n = %d\n", kMxEmbd);
        std::exit(1);
    }
    const int threads = std::min(1024, ((n / 4 + 3) / 4 + 31) / 32 * 32);   // mimo_rmsnorm's
    add_rmsnorm_q8_1_kernel<<<T, threads, 0, s>>>(x, y, y2, scale, w, h, static_cast<MimoQ81*>(q8), h_out,
                                                  static_cast<uint8_t*>(xq), zero_y ? 1 : 0, n, eps);
    check("add_rmsnorm_q8_1 launch");
}

void mimo_router_fused(const float* gate_inp, const float* bias, const float* h, float* logits, int32_t* ids,
                       float* weights, int T, int n_embd, int n_expert, int n_used, float weights_scale,
                       const int32_t* res_layer, int32_t* slots, int32_t* ids_out, float* w_out, uint32_t* seq,
                       cudaStream_t s) {
    if (T < 1 || T > kRouterMaxT || n_expert > kMaxExpert || n_used > 32 || n_embd % 128 != 0) {
        std::fprintf(stderr, "mimo_router_fused: T %d, n_expert %d, n_used %d, n_embd %d out of range\n", T, n_expert,
                     n_used, n_embd);
        std::exit(1);
    }
    const dim3 grid((unsigned) n_expert), block(kLogitThreads);
    switch (T) {
#define STRATA_ROUTER_T(n)                                                                                           \
    case n:                                                                                                          \
        router_fused_kernel<n><<<grid, block, 0, s>>>(gate_inp, h, bias, logits, ids, weights, n_embd, n_expert,     \
                                                      n_used, weights_scale, res_layer, slots, ids_out, w_out, seq); \
        break;
        STRATA_ROUTER_T(1) STRATA_ROUTER_T(2) STRATA_ROUTER_T(3) STRATA_ROUTER_T(4)
        STRATA_ROUTER_T(5) STRATA_ROUTER_T(6) STRATA_ROUTER_T(7) STRATA_ROUTER_T(8)
#undef STRATA_ROUTER_T
        default: break;
    }
    check("router_fused launch");
}

}  // namespace strata::mimo
