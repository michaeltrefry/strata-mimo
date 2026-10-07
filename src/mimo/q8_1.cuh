// src/mimo/q8_1.cuh - native_quantize_q8_1's numbers inside other kernels (the fused norms, the attention combine).
//
// native_mmvq.cu is built with --use_fast_math: its d = amax / 127 and x / d are div.approx.ftz, and its
// amax == 0 test flushes denormals.  These helpers do the same in a translation unit without fast math, so a fused
// kernel's codes and scales are bitwise those of native_quantize_q8_1 (the block sum, which Q8_0 MMVQ does not read,
// may be summed in another order).
#pragma once

#include <cfloat>
#include <cstdint>
#include <cuda_fp16.h>

namespace strata::mimo {

struct MimoQ81 {   // llama.cpp block_q8_1: fp16 scale and sum, 32 int8 codes
    half2 ds;
    int8_t qs[32];
};
static_assert(sizeof(MimoQ81) == 36, "block_q8_1");

__device__ __forceinline__ float q8_div(float a, float b) {
    float r;
    asm("div.approx.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

// one code; amax below FLT_MIN is zero, as the fast-math comparison sees it
__device__ __forceinline__ int q8_code(float x, float amax, float d) {
    return amax < FLT_MIN ? 0 : (int) (int8_t) roundf(q8_div(x, d));
}

// The calling warp's 32 values form one block (lane = position in the block); all 32 lanes must call.
__device__ __forceinline__ void q8_1_warp(float x, MimoQ81* blk) {
    float amax = fabsf(x), sum = x;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    const float d = q8_div(amax, 127.0f);
    const int lane = threadIdx.x & 31;
    blk->qs[lane] = (int8_t) q8_code(x, amax, d);
    if (lane == 0) blk->ds = make_half2(d, sum);
}

// Eight consecutive lanes hold one block as four values each (lane % 8 = which four); `valid` must be the same for
// the eight.  All 32 lanes must call.
__device__ __forceinline__ void q8_1_quad(float4 v, bool valid, MimoQ81* blk) {
    float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
    float sum = (v.x + v.y) + (v.z + v.w);
#pragma unroll
    for (int o = 4; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        sum += __shfl_xor_sync(0xffffffffu, sum, o);
    }
    if (!valid) return;
    const float d = q8_div(amax, 127.0f);
    const int c = threadIdx.x & 7;
    const int q = (q8_code(v.x, amax, d) & 0xff) | (q8_code(v.y, amax, d) & 0xff) << 8 |
                  (q8_code(v.z, amax, d) & 0xff) << 16 | (q8_code(v.w, amax, d) & 0xff) << 24;
    reinterpret_cast<int*>(blk->qs)[c] = q;
    if (c == 0) blk->ds = make_half2(d, sum);
}

}  // namespace strata::mimo
