#include "strata/mimo/ops.hpp"

#include <cfloat>
#include <climits>
#include <cuda_fp16.h>

namespace strata::mimo {
namespace {

// one block per (token, 32-value block group): thread i dequantizes value i of its Q8_0 block
__global__ void embed_q8_0_kernel(const uint8_t* __restrict__ table, const int32_t* __restrict__ ids,
                                  float* __restrict__ out, int n_embd) {
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_embd) return;
    const size_t row_bytes = (size_t) n_embd / 32 * 34;
    const uint8_t* blk = table + (size_t) ids[t] * row_bytes + (size_t) (i / 32) * 34;
    const float d = __half2float(*reinterpret_cast<const __half*>(blk));
    out[(size_t) t * n_embd + i] = d * (float) reinterpret_cast<const int8_t*>(blk + 2)[i % 32];
}

__global__ void silu_mul_kernel(const float* __restrict__ g, const float* __restrict__ u, float* __restrict__ o, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float x = g[i];
        o[i] = x / (1.f + expf(-x)) * u[i];
    }
}

__global__ void add_scaled_kernel(float* __restrict__ x, const float* __restrict__ y, float scale, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += scale * y[i];
}

constexpr int kArgmaxThreads = 1024;

__global__ void argmax_kernel(const float* __restrict__ logits, int32_t* __restrict__ ids, int n_vocab) {
    __shared__ float sv[32];
    __shared__ int si[32];
    const float* row = logits + (size_t) blockIdx.x * n_vocab;
    float best = -FLT_MAX;
    int idx = INT_MAX;
    for (int v = threadIdx.x; v < n_vocab; v += kArgmaxThreads) {
        const float x = row[v];
        if (x > best) { best = x; idx = v; }   // v increases per thread: ties keep the lower id
    }
    for (int o = 16; o > 0; o >>= 1) {
        const float ob = __shfl_xor_sync(0xffffffffu, best, o);
        const int oi = __shfl_xor_sync(0xffffffffu, idx, o);
        if (ob > best || (ob == best && oi < idx)) { best = ob; idx = oi; }
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) { sv[warp] = best; si[warp] = idx; }
    __syncthreads();
    if (warp == 0) {
        best = sv[lane];
        idx = si[lane];
        for (int o = 16; o > 0; o >>= 1) {
            const float ob = __shfl_xor_sync(0xffffffffu, best, o);
            const int oi = __shfl_xor_sync(0xffffffffu, idx, o);
            if (ob > best || (ob == best && oi < idx)) { best = ob; idx = oi; }
        }
        if (lane == 0) ids[blockIdx.x] = idx;
    }
}

}  // namespace

void embed_q8_0(const void* table, const int32_t* ids, float* out, int T, int n_embd, cudaStream_t s) {
    const dim3 grid((n_embd + 255) / 256, T);
    embed_q8_0_kernel<<<grid, 256, 0, s>>>(static_cast<const uint8_t*>(table), ids, out, n_embd);
}

void silu_mul(const float* gate, const float* up, float* out, int n, cudaStream_t s) {
    silu_mul_kernel<<<(n + 255) / 256, 256, 0, s>>>(gate, up, out, n);
}

void add_scaled(float* x, const float* y, float scale, int n, cudaStream_t s) {
    add_scaled_kernel<<<(n + 255) / 256, 256, 0, s>>>(x, y, scale, n);
}

void argmax_rows(const float* logits, int32_t* ids, int T, int n_vocab, cudaStream_t s) {
    argmax_kernel<<<T, kArgmaxThreads, 0, s>>>(logits, ids, n_vocab);
}

}  // namespace strata::mimo
