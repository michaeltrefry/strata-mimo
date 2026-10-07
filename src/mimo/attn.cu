// src/mimo/attn.cu - see include/strata/mimo/attn.hpp.
//
// Decode attention is flash-decoding on the tensor cores (mma.sync m16n8k16, f16 inputs, f32 accumulation):
//   grid (split, KV head), 32 * NW threads. A block streams one KV head's keys [chunk lo, chunk hi) through shared
//   memory (cp.async, NST stages of KT keys) and holds all T x G query rows of that head (G = n_head / n_kv query
//   heads per KV head, rows r = t * G + head-in-group), cut into NMT m-tiles of 16 rows. Warp w owns m-tile
//   w % NMT and the 16-key slices w / NMT (+ KSUB) of every tile (KT = 16 * KSUB * SL), with its own online
//   softmax; the KSUB slices of one m-tile merge in shared memory at the end, and the block writes one partial
//   (unnormalized output, max, denominator) per row. A second kernel merges the splits and adds the sink (an SWA
//   layer at T = 1 runs one split per KV head and writes the output directly).
//   T = 1 on a global layer: 16 rows, one m-tile, 8 warps on 8 key slices of a 128-key tile: reads the KV at
//   1.38 TB/s at 128K-256K positions (77% of the card's ~1.8 TB/s; RTX PRO 6000 Max-Q, mimo_attn_test --bench).
//   T = 8 on a global layer: 128 rows, 8 m-tiles, 8 warps on two 16-key slices of a 32-key tile. This is bound by
//   instruction issue and latency (8 warps per SM, ~670 instructions per warp and slice), not by the tensor cores
//   (444 TFLOPS f16 measured on this card; T = 8 at 256K uses about a third): 2.7x the T = 1 time.
//
// Precision: K codes enter the MMA as exact f16 integers; Q is rounded to f16 and each 64-dim group's dot product
// is scaled by its f32 key scale after the MMA. V enters as fp16(code * scale), P as fp16. Scores, max and
// denominators are f32. The test measures what the f16 roundings cost next to the int8 KV error.
//
// Fragment orders (PTX m16n8k16 .row.col: thread lane = 4 g + c holds A rows g, g + 8 x k-indices 2c, 2c+1,
// 2c+8, 2c+9; B k-indices the same x column g; C rows g, g + 8 x columns 2c, 2c + 1):
//   QK^T - the 16 k-indices of k-step s of 32-dim chunk ch are a permutation of its dims: k = 2c + j <-> dim
//          32 ch + 8 c + 4 s + j, k = 2c + 8 + j <-> 32 ch + 8 c + 4 s + 2 + j. Q and K use the same one (the dot
//          product does not care), so each thread reads K as one 8-byte load of 8 consecutive codes per key.
//   P V  - output column g of n-tile j is dim 64 (j / 8) + 8 g + j % 8: one n-tile stays in one 64-dim V group
//          (one scale per key) and each thread reads 8 consecutive codes per key and group.
#include "strata/mimo/attn.hpp"
#include "q8_1.cuh"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace strata::mimo {
namespace {

using kernels::f16_from_f32;
using kernels::f32_from_f16;

constexpr int DK = kAttnHeadK, DV = kAttnHeadV, NROT = kAttnRot;
constexpr int K_STRIDE = 224;      // shared-memory row of 192 K codes: 56 words, conflict-free 8-byte reads
constexpr int V_STRIDE = 144;      // 128 V codes: 36 words
constexpr int MAX_ROWS = 128;      // T x G <= 8 x 16
constexpr int MIN_CHUNK = 128;     // keys per split at least on a global layer (short contexts use fewer splits)
constexpr int MIN_CHUNK_SWA = 32;  // an SWA layer's <= 135 keys: 4-5 splits per KV head, for latency
constexpr int MAX_SPLITS = 1024;   // the combine kernel's shared weights
constexpr float LOG2E = 1.4426950408889634f;
#define NEG_INF (__int_as_float(0xff800000))   // -inf without MSVC's INFINITY overflow warning

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "mimo attn: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

// Which keys a window reads and how they are cut into splits; the attention and the combine kernels both call it
// with the same arguments, so they agree on the split count without a host round trip.
struct Plan {
    int lo, hi, chunk, n;
};
__host__ __device__ inline Plan make_plan(int pos0, int T, bool swa, int n_swa, int max_splits) {
    Plan p;
    p.hi = pos0 + T;
    p.lo = swa ? (pos0 - n_swa + 1 > 0 ? pos0 - n_swa + 1 : 0) : 0;
    const int n = p.hi - p.lo, min_chunk = swa ? MIN_CHUNK_SWA : MIN_CHUNK;
    int ns = (n + min_chunk - 1) / min_chunk;
    ns = ns < 1 ? 1 : (ns > max_splits ? max_splits : ns);
    const int per = (n + ns - 1) / ns;
    p.chunk = (per + 15) / 16 * 16;   // whole 16-key warp slices; a block's last tile may be partly empty
    p.n = (n + p.chunk - 1) / p.chunk;
    return p;
}

template <int NMT>
struct Cfg {
    static constexpr int KSUB = 8 / NMT;   // 8, 4, 2, 2, 1, 1, 1, 1
    static constexpr int NW = NMT * KSUB;
    static constexpr int NT = NW * 32;
    static constexpr int SL = KSUB == 1 ? 2 : 1;   // 16-key slices per warp and tile (2: fewer barriers)
    static constexpr int KT = 16 * KSUB * SL;
    static constexpr int OFF_V = KT * K_STRIDE, OFF_KS = OFF_V + KT * V_STRIDE, OFF_VS = OFF_KS + KT * 8;
    static constexpr int STAGE = OFF_VS + KT * 4;
    static constexpr int NST0 = 97280 / STAGE;
    static constexpr int NST = NST0 < 2 ? 2 : (NST0 > 8 ? 8 : NST0);
    static constexpr int MERGE = KSUB > 1 ? NW * 16 * 2 * 4 + NW * 16 * DV * 4 + NMT * 16 * 4 : 0;
    static constexpr int SMEM = NST * STAGE > MERGE ? NST * STAGE : MERGE;
};

__device__ __forceinline__ void cp_async(void* smem, const void* gmem, bool valid, int bytes) {
    const unsigned sa = (unsigned) __cvta_generic_to_shared(smem);
    const int src = valid ? bytes : 0;   // 0 = zero-fill: keys past the chunk read as code 0, scale 0
    if (bytes == 16)
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(sa), "l"(gmem), "r"(src) : "memory");
    else if (bytes == 8)
        asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n" ::"r"(sa), "l"(gmem), "r"(src) : "memory");
    else
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(sa), "l"(gmem), "r"(src) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory"); }

__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm(   // not volatile: no side effects, so the compiler may interleave independent MMAs
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t h2_bits(__half2 h) { return *reinterpret_cast<uint32_t*>(&h); }
__device__ __forceinline__ __half2 bits_h2(uint32_t u) { return *reinterpret_cast<__half2*>(&u); }
__device__ __forceinline__ uint32_t pack_h2(float lo, float hi) { return h2_bits(__floats2half2_rn(lo, hi)); }

// Four int8 codes -> two half2 (bytes 0,1 and 2,3), exact: 0x64XX is 1024 + XX, so (code + 128) + 1024 - 1152.
__device__ __forceinline__ void i8x4_to_h2(uint32_t w, uint32_t& lo, uint32_t& hi) {
    const uint32_t u = w ^ 0x80808080u;
    const __half2 bias = bits_h2(0x64806480u);   // 1152, 1152
    lo = h2_bits(__hsub2(bits_h2(__byte_perm(u, 0x64646464u, 0x4140)), bias));
    hi = h2_bits(__hsub2(bits_h2(__byte_perm(u, 0x64646464u, 0x4342)), bias));
}

struct KvView {
    const int8_t* k;
    const int8_t* v;
    const uint16_t* ks;
    const uint16_t* vs;
    int cap, ring;
};

template <int NMT>
__global__ void __launch_bounds__(Cfg<NMT>::NT, 1)
    attn_split_kernel(const float* __restrict__ q, const int32_t* __restrict__ pos0p, int T, int G, int n_head,
                      KvView kv, int swa, int n_swa, float scale_log2, int max_splits, float* __restrict__ opart,
                      float* __restrict__ mlpart, const float* __restrict__ sinks, float* __restrict__ out) {
    using C = Cfg<NMT>;
    extern __shared__ __align__(16) uint8_t smem[];
    const int split = blockIdx.x, h = blockIdx.y, n_kv = gridDim.y;
    const int pos0 = __ldg(pos0p);
    const Plan pl = make_plan(pos0, T, swa != 0, n_swa, max_splits);
    if (split >= pl.n) return;
    const int c_lo = pl.lo + split * pl.chunk;
    const int c_hi = min(c_lo + pl.chunk, pl.hi);
    const int n_tiles = (c_hi - c_lo + C::KT - 1) / C::KT;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, c = lane & 3;
    const int mt = warp % NMT, ksi = warp / NMT;
    const int rows = T * G;

    const size_t hoff = (size_t) h * kv.cap;
    const int8_t* kbase = kv.k + hoff * DK;
    const int8_t* vbase = kv.v + hoff * DV;
    const uint16_t* ksbase = kv.ks + hoff * 4;
    const uint16_t* vsbase = kv.vs + hoff * 2;

    auto load_tile = [&](int tile) {
        if (tile < n_tiles) {
            uint8_t* st = smem + (tile % C::NST) * C::STAGE;
            const int base = c_lo + tile * C::KT;
            for (int i = threadIdx.x; i < C::KT * 22; i += C::NT) {
                const int key = i / 22, part = i % 22;
                const int p = base + key;
                const bool ok = p < c_hi;
                const size_t slot = ok ? (size_t) (kv.ring ? p % kv.cap : p) : 0;
                if (part < 12) cp_async(st + key * K_STRIDE + part * 16, kbase + slot * DK + part * 16, ok, 16);
                else if (part < 20)
                    cp_async(st + C::OFF_V + key * V_STRIDE + (part - 12) * 16, vbase + slot * DV + (part - 12) * 16, ok, 16);
                else if (part == 20) cp_async(st + C::OFF_KS + key * 8, ksbase + slot * 4, ok, 8);
                else cp_async(st + C::OFF_VS + key * 4, vsbase + slot * 2, ok, 4);
            }
        }
        cp_commit();
    };
    for (int s = 0; s < C::NST - 1; ++s) load_tile(s);

    // Q fragments of rows r0 = 16 mt + g and r1 = r0 + 8 (zero past the last row)
    const int r0 = 16 * mt + g, r1 = r0 + 8;
    const int t0 = r0 / G, t1 = r1 / G;
    uint32_t qa[6][2][4];
    {
        const float* q0 = r0 < rows ? q + ((size_t) t0 * n_head + h * G + r0 % G) * DK : nullptr;
        const float* q1 = r1 < rows ? q + ((size_t) t1 * n_head + h * G + r1 % G) * DK : nullptr;
#pragma unroll
        for (int ch = 0; ch < 6; ++ch)
#pragma unroll
            for (int s = 0; s < 2; ++s) {
                const int d = 32 * ch + 8 * c + 4 * s;
                const float4 a = q0 ? *reinterpret_cast<const float4*>(q0 + d) : make_float4(0, 0, 0, 0);
                const float4 b = q1 ? *reinterpret_cast<const float4*>(q1 + d) : make_float4(0, 0, 0, 0);
                qa[ch][s][0] = pack_h2(a.x, a.y);
                qa[ch][s][1] = pack_h2(b.x, b.y);
                qa[ch][s][2] = pack_h2(a.z, a.w);
                qa[ch][s][3] = pack_h2(b.z, b.w);
            }
    }
    const int qp0 = pos0 + t0, qp1 = pos0 + t1;

    float o[16][4];
#pragma unroll
    for (int j = 0; j < 16; ++j) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.0f;
    float m0 = NEG_INF, m1 = NEG_INF, l0 = 0.0f, l1 = 0.0f;
    for (int it = 0; it < n_tiles; ++it) {
        cp_wait<C::NST - 2>();
        __syncthreads();
        load_tile(it + C::NST - 1);
        const uint8_t* st = smem + (it % C::NST) * C::STAGE;
#pragma unroll 1
      for (int sl = 0; sl < C::SL; ++sl) {
        const int kb = 16 * (ksi + C::KSUB * sl);   // this warp's 16-key slice of the tile
        const int tpos = c_lo + it * C::KT + kb;     // position of its first key
        if (tpos >= c_hi) break;

        // S = Q K^T, one accumulator per 64-dim K group
        float acc[3][2][4];
#pragma unroll
        for (int a = 0; a < 3; ++a)
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) acc[a][nt][0] = acc[a][nt][1] = acc[a][nt][2] = acc[a][nt][3] = 0.0f;
#pragma unroll
        for (int ch = 0; ch < 6; ++ch)
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                const uint2 w = *reinterpret_cast<const uint2*>(st + (kb + 8 * nt + g) * K_STRIDE + 32 * ch + 8 * c);
                uint32_t b0, b1, b2, b3;
                i8x4_to_h2(w.x, b0, b1);
                i8x4_to_h2(w.y, b2, b3);
                mma16816(acc[ch >> 1][nt], qa[ch][0], b0, b1);
                mma16816(acc[ch >> 1][nt], qa[ch][1], b2, b3);
            }
        float sc[2][4];
#pragma unroll
        for (int nt = 0; nt < 2; ++nt)
#pragma unroll
            for (int kk = 0; kk < 2; ++kk) {
                const int key = kb + 8 * nt + 2 * c + kk;
                const uint2 sb = *reinterpret_cast<const uint2*>(st + C::OFF_KS + key * 8);
                const float s0 = __half2float(__ushort_as_half((unsigned short) (sb.x & 0xFFFF)));
                const float s1 = __half2float(__ushort_as_half((unsigned short) (sb.x >> 16)));
                const float s2 = __half2float(__ushort_as_half((unsigned short) (sb.y & 0xFFFF)));
                const int p = tpos + 8 * nt + 2 * c + kk;
#pragma unroll
                for (int hr = 0; hr < 2; ++hr) {
                    const int i = 2 * hr + kk;
                    const int qp = hr ? qp1 : qp0;
                    const float v = (acc[0][nt][i] * s0 + acc[1][nt][i] * s1 + acc[2][nt][i] * s2) * scale_log2;
                    const bool ok = p < c_hi && p <= qp && (!swa || qp - p < n_swa);
                    sc[nt][i] = ok ? v : NEG_INF;
                }
            }
        // online softmax, rows g (elements 0, 1) and g + 8 (2, 3); the 4 lanes of a quad share a row
        float mx0 = fmaxf(fmaxf(sc[0][0], sc[0][1]), fmaxf(sc[1][0], sc[1][1]));
        float mx1 = fmaxf(fmaxf(sc[0][2], sc[0][3]), fmaxf(sc[1][2], sc[1][3]));
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
        const float mn0 = fmaxf(m0, mx0), mn1 = fmaxf(m1, mx1);
        const float ref0 = mn0 == NEG_INF ? 0.0f : mn0, ref1 = mn1 == NEG_INF ? 0.0f : mn1;
        const float al0 = exp2f(m0 - ref0), al1 = exp2f(m1 - ref1);
        m0 = mn0;
        m1 = mn1;
        float p[2][4];
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            p[nt][0] = exp2f(sc[nt][0] - ref0);
            p[nt][1] = exp2f(sc[nt][1] - ref0);
            p[nt][2] = exp2f(sc[nt][2] - ref1);
            p[nt][3] = exp2f(sc[nt][3] - ref1);
        }
        l0 = l0 * al0 + (p[0][0] + p[0][1] + p[1][0] + p[1][1]);
        l1 = l1 * al1 + (p[0][2] + p[0][3] + p[1][2] + p[1][3]);
        if (!__all_sync(0xffffffffu, al0 == 1.0f && al1 == 1.0f)) {   // most tiles leave every row's max alone
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                o[j][0] *= al0;
                o[j][1] *= al0;
                o[j][2] *= al1;
                o[j][3] *= al1;
            }
        }
        const uint32_t pa[4] = {pack_h2(p[0][0], p[0][1]), pack_h2(p[0][2], p[0][3]), pack_h2(p[1][0], p[1][1]),
                                pack_h2(p[1][2], p[1][3])};

        // O += P V: keys 2c, 2c+1 (b0) and 2c+8, 2c+9 (b1) of the slice, 8 codes per key and V group
        const uint8_t* vs_ = st + C::OFF_V;
        const uint16_t* vsc = reinterpret_cast<const uint16_t*>(st + C::OFF_VS);
        const int ka = kb + 2 * c;
#pragma unroll
        for (int gv = 0; gv < 2; ++gv) {
            const uint2 va = *reinterpret_cast<const uint2*>(vs_ + (ka + 0) * V_STRIDE + 64 * gv + 8 * g);
            const uint2 vb = *reinterpret_cast<const uint2*>(vs_ + (ka + 1) * V_STRIDE + 64 * gv + 8 * g);
            const uint2 vc = *reinterpret_cast<const uint2*>(vs_ + (ka + 8) * V_STRIDE + 64 * gv + 8 * g);
            const uint2 vd = *reinterpret_cast<const uint2*>(vs_ + (ka + 9) * V_STRIDE + 64 * gv + 8 * g);
            const __half2 sab = bits_h2((uint32_t) vsc[(ka + 0) * 2 + gv] | ((uint32_t) vsc[(ka + 1) * 2 + gv] << 16));
            const __half2 scd = bits_h2((uint32_t) vsc[(ka + 8) * 2 + gv] | ((uint32_t) vsc[(ka + 9) * 2 + gv] << 16));
#pragma unroll
            for (int hw = 0; hw < 2; ++hw) {
                const uint32_t A = hw ? va.y : va.x, B = hw ? vb.y : vb.x;
                const uint32_t Cc = hw ? vc.y : vc.x, D = hw ? vd.y : vd.x;
#pragma unroll
                for (int pr = 0; pr < 2; ++pr) {
                    const uint32_t sel = pr ? 0x7362u : 0x5140u;
                    uint32_t x0, x1, y0, y1;
                    i8x4_to_h2(__byte_perm(A, B, sel), x0, x1);    // (key a, key b) of codes jj, jj + 1
                    i8x4_to_h2(__byte_perm(Cc, D, sel), y0, y1);   // (key c, key d)
                    x0 = h2_bits(__hmul2(bits_h2(x0), sab));
                    x1 = h2_bits(__hmul2(bits_h2(x1), sab));
                    y0 = h2_bits(__hmul2(bits_h2(y0), scd));
                    y1 = h2_bits(__hmul2(bits_h2(y1), scd));
                    const int j = 8 * gv + 4 * hw + 2 * pr;
                    mma16816(o[j], pa, x0, y0);
                    mma16816(o[j + 1], pa, x1, y1);
                }
            }
        }
      }
    }
    l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
    l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 2);

    // Epilogue. out == nullptr: one partial per row (unnormalized O, max, denominator) for the combine kernel.
    // out != nullptr (one split, SWA layers): the final output, normalized here with the sink folded in.
    const size_t prow = ((size_t) split * n_kv + h) * MAX_ROWS;   // first partial row of this block
    // 1 / denominator of a row with running max m and sum l, the sink of its query head included, times the
    // exp2(m - M) that rescales O to the max M over the keys and the sink
    auto final_scale = [&](int r, float m, float l) {
        const float sk = sinks ? __ldg(sinks + h * G + r % G) * LOG2E : NEG_INF;
        const float M = fmaxf(m, sk);
        if (M == NEG_INF) return 0.0f;
        const float f = exp2f(m - M);
        return f / (l * f + (sinks ? exp2f(sk - M) : 0.0f));
    };
    auto out_index = [&](int r, int d) { return ((size_t) (r / G) * n_head + h * G + r % G) * DV + d; };
    if constexpr (C::KSUB == 1) {
        const float n0 = out ? final_scale(r0, m0, l0) : 1.0f, n1 = out ? final_scale(r1, m1, l1) : 1.0f;
#pragma unroll
        for (int j = 0; j < 16; ++j)
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int r = i < 2 ? r0 : r1;
                const int d = 64 * (j >> 3) + 8 * (2 * c + (i & 1)) + (j & 7);
                if (r >= rows) continue;
                if (out) out[out_index(r, d)] = o[j][i] * (i < 2 ? n0 : n1);
                else opart[(prow + r) * DV + d] = o[j][i];
            }
        if (c == 0 && !out) {
            if (r0 < rows) { mlpart[(prow + r0) * 2] = m0; mlpart[(prow + r0) * 2 + 1] = l0; }
            if (r1 < rows) { mlpart[(prow + r1) * 2] = m1; mlpart[(prow + r1) * 2 + 1] = l1; }
        }
    } else {
        // merge the KSUB key slices of each m-tile through shared memory (the stages are free now)
        cp_wait<0>();
        __syncthreads();
        float* sml = reinterpret_cast<float*>(smem);   // [NW][16][2]
        float* so = sml + C::NW * 32;                  // [NW][16][128]
        float* snorm = so + C::NW * 16 * DV;           // [NMT * 16]
        if (c == 0) {
            sml[(warp * 16 + g) * 2] = m0;
            sml[(warp * 16 + g) * 2 + 1] = l0;
            sml[(warp * 16 + g + 8) * 2] = m1;
            sml[(warp * 16 + g + 8) * 2 + 1] = l1;
        }
        __syncthreads();
        float M0 = NEG_INF, M1 = NEG_INF;
#pragma unroll
        for (int k = 0; k < C::KSUB; ++k) {
            M0 = fmaxf(M0, sml[((mt + NMT * k) * 16 + g) * 2]);
            M1 = fmaxf(M1, sml[((mt + NMT * k) * 16 + g + 8) * 2]);
        }
        const float f0 = M0 == NEG_INF ? 0.0f : exp2f(m0 - M0), f1 = M1 == NEG_INF ? 0.0f : exp2f(m1 - M1);
#pragma unroll
        for (int j = 0; j < 16; ++j)
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int rr = i < 2 ? g : g + 8;
                const int d = 64 * (j >> 3) + 8 * (2 * c + (i & 1)) + (j & 7);
                so[(warp * 16 + rr) * DV + d] = o[j][i] * (i < 2 ? f0 : f1);
            }
        for (int idx = threadIdx.x; idx < NMT * 16; idx += C::NT) {
            const int m = idx / 16, rr = idx % 16, r = 16 * m + rr;
            if (r >= rows) continue;
            float M = NEG_INF, L = 0.0f;
            for (int k = 0; k < C::KSUB; ++k) M = fmaxf(M, sml[((m + NMT * k) * 16 + rr) * 2]);
            if (M != NEG_INF)
                for (int k = 0; k < C::KSUB; ++k) {
                    const float* e = sml + ((m + NMT * k) * 16 + rr) * 2;
                    L += e[1] * exp2f(e[0] - M);
                }
            if (out) {
                snorm[idx] = final_scale(r, M, L);
            } else {
                mlpart[(prow + r) * 2] = M;
                mlpart[(prow + r) * 2 + 1] = L;
            }
        }
        __syncthreads();
        for (int idx = threadIdx.x; idx < NMT * 16 * DV; idx += C::NT) {
            const int m = idx / (16 * DV), rr = (idx / DV) % 16, d = idx % DV;
            const int r = 16 * m + rr;
            if (r >= rows) continue;
            float acc = 0.0f;
#pragma unroll
            for (int k = 0; k < C::KSUB; ++k) acc += so[((m + NMT * k) * 16 + rr) * DV + d];
            if (out) out[out_index(r, d)] = acc * snorm[m * 16 + rr];
            else opart[(prow + r) * DV + d] = acc;
        }
    }
}

// One block per (query head, token), one thread per output dim: merge the splits, add the sink. The split weights
// exp2(m_s - M) are computed once into shared memory; the O loads are unrolled so several are in flight.
__global__ void __launch_bounds__(DV) attn_combine_kernel(const float* __restrict__ opart,
                                                          const float* __restrict__ mlpart,
                                                          const int32_t* __restrict__ pos0p, int T, int G, int n_kv,
                                                          int swa, int n_swa, int max_splits,
                                                          const float* __restrict__ sinks, float* __restrict__ out,
                                                          MimoQ81* __restrict__ out_q8) {
    __shared__ float w[MAX_SPLITS];
    __shared__ float red[DV / 32];
    const int hq = blockIdx.x, t = blockIdx.y, d = threadIdx.x, n_head = gridDim.x, lane = d & 31, warp = d >> 5;
    const int h = hq / G, r = t * G + hq % G;
    const int ns = make_plan(__ldg(pos0p), T, swa != 0, n_swa, max_splits).n;
    const size_t stride = (size_t) n_kv * MAX_ROWS;   // partial rows between consecutive splits
    const size_t row0 = (size_t) h * MAX_ROWS + r;
    const float sink = sinks ? __ldg(sinks + hq) * LOG2E : NEG_INF;
    float M = sink;
    for (int s = d; s < ns; s += DV) M = fmaxf(M, mlpart[(row0 + s * stride) * 2]);
    for (int o = 16; o > 0; o >>= 1) M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, o));
    if (lane == 0) red[warp] = M;
    __syncthreads();
    M = fmaxf(fmaxf(red[0], red[1]), fmaxf(red[2], red[3]));
    __syncthreads();
    float den = 0.0f;
    for (int s = d; s < ns; s += DV) {
        const float ms = mlpart[(row0 + s * stride) * 2];
        const float f = ms == NEG_INF ? 0.0f : exp2f(ms - M);
        w[s] = f;
        den += f * mlpart[(row0 + s * stride) * 2 + 1];
    }
    for (int o = 16; o > 0; o >>= 1) den += __shfl_xor_sync(0xffffffffu, den, o);
    if (lane == 0) red[warp] = den;
    __syncthreads();
    den = red[0] + red[1] + red[2] + red[3] + (sinks ? exp2f(sink - M) : 0.0f);
    const float* op = opart + row0 * DV + d;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    int s = 0;
    for (; s + 4 <= ns; s += 4) {
        a0 += w[s] * op[(size_t) s * stride * DV];
        a1 += w[s + 1] * op[(size_t) (s + 1) * stride * DV];
        a2 += w[s + 2] * op[(size_t) (s + 2) * stride * DV];
        a3 += w[s + 3] * op[(size_t) (s + 3) * stride * DV];
    }
    for (; s < ns; ++s) a0 += w[s] * op[(size_t) s * stride * DV];
    const size_t i = ((size_t) t * n_head + hq) * DV + d;
    const float o = ((a0 + a1) + (a2 + a3)) / den;
    out[i] = o;
    if (out_q8) q8_1_warp(o, out_q8 + i / 32);   // a warp's 32 dims are one q8_1 block of attn_output's input
}

// NEOX rope, shared by the decode and the prompt append so both write the same codes for the same position:
// theta_j = pos * powf(theta_scale, j), and the pair (x_j, x_{j+32}) rotated with each operation rounded on its own
// (no fma contraction, which the compiler would otherwise choose per call site).
__device__ __forceinline__ void rope_cs(int pos, float theta_scale, int j, float& cs, float& sn) {
    const float theta = (float) pos * powf(theta_scale, (float) j);
    sincosf(theta, &sn, &cs);
}
__device__ __forceinline__ void rope_pair(float x0, float x1, float cs, float sn, float& y0, float& y1) {
    y0 = __fsub_rn(__fmul_rn(x0, cs), __fmul_rn(x1, sn));
    y1 = __fadd_rn(__fmul_rn(x0, sn), __fmul_rn(x1, cs));
}
// kv_q8's rule for one 64-value group: scale = fp16(max|x| / 127), code = rint(x / scale) clamped to +-127
__device__ __forceinline__ int q8_code(float y, float sf) {
    if (!(sf > 0.0f)) return 0;
    const int code = __float2int_rn(y / sf);   // quantize against the STORED scale
    return code < -127 ? -127 : (code > 127 ? 127 : code);
}

// Decode: one block per (head of q, k or v, token), one thread per dim: NEOX rope on dims 0..63 of q and k, k and v
// quantized into the cache (one fp16 scale per 64 values). pos0 from device memory, q as F32.
__global__ void qkv_rope_append_kernel(const float* __restrict__ qkv, const int32_t* __restrict__ pos0p, int n_head,
                                       int n_kv, float theta_scale, KvView kv, float* __restrict__ q_out) {
    const int b = blockIdx.x, t = blockIdx.y, i = threadIdx.x;
    const int rows = DK * (n_head + n_kv) + DV * n_kv;
    const int pos = __ldg(pos0p) + t;
    const float* row = qkv + (size_t) t * rows;
    const bool is_q = b < n_head, is_k = !is_q && b < n_head + n_kv, is_v = !is_q && !is_k;
    const int dim = is_v ? DV : DK;
    const float* src = is_q ? row + (size_t) b * DK
                            : (is_k ? row + (size_t) DK * n_head + (size_t) (b - n_head) * DK
                                    : row + (size_t) DK * (n_head + n_kv) + (size_t) (b - n_head - n_kv) * DV);
    float y = i < dim ? src[i] : 0.0f;
    if (!is_v && i < NROT) {
        float sn, cs, y0, y1;
        rope_cs(pos, theta_scale, i & (NROT / 2 - 1), cs, sn);
        if (i < NROT / 2) rope_pair(y, src[i + NROT / 2], cs, sn, y0, y1), y = y0;
        else rope_pair(src[i - NROT / 2], y, cs, sn, y0, y1), y = y1;
    }
    if (is_q) {
        q_out[((size_t) t * n_head + b) * DK + i] = y;
        return;   // the whole block takes this branch: no barrier below is skipped by part of it
    }
    __shared__ float wmax[DK / 32];
    float a = fabsf(y);
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    if ((i & 31) == 0) wmax[i >> 5] = a;
    __syncthreads();
    if (i >= dim) return;
    const int grp = i / kAttnGroup;
    const uint16_t sbits = f16_from_f32(fmaxf(wmax[2 * grp], wmax[2 * grp + 1]) / 127.0f);
    const int code = q8_code(y, f32_from_f16(sbits));
    const int head = is_k ? b - n_head : b - n_head - n_kv;
    const size_t slot = (size_t) head * kv.cap + (size_t) (kv.ring ? pos % kv.cap : pos);
    if (is_k) {
        const_cast<int8_t*>(kv.k)[slot * DK + i] = (int8_t) code;
        if ((i & 63) == 0) const_cast<uint16_t*>(kv.ks)[slot * 4 + grp] = sbits;
    } else {
        const_cast<int8_t*>(kv.v)[slot * DV + i] = (int8_t) code;
        if ((i & 63) == 0) const_cast<uint16_t*>(kv.vs)[slot * 2 + grp] = sbits;
    }
}

// Prompt: one block of 8 warps per token; the token's 32 rope (cos, sin) pairs once in shared memory, then a warp per
// head (q, k, v), lane l holding dims l and l + 32 of each 64-value group - so the roped pair (l, l + 32) and a
// group's max are one lane and one warp reduction. q is written as fp16 bits; k and v go into `kv` for positions
// >= kv_from (global: all; SWA ring: the chunk's last 256) and, when stage.k is set (SWA), into the staging at row
// pos - stage_base.
__global__ void __launch_bounds__(256) qkv_rope_append_prompt_kernel(const float* __restrict__ qkv, int pos0,
                                                                     int n_head, int n_kv, float theta_scale,
                                                                     KvView kv, int kv_from, KvView stage,
                                                                     int stage_base, uint16_t* __restrict__ q_out) {
    __shared__ float tcs[NROT / 2], tsn[NROT / 2];
    const int t = blockIdx.x, pos = pos0 + t, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (threadIdx.x < NROT / 2) rope_cs(pos, theta_scale, lane, tcs[lane], tsn[lane]);
    __syncthreads();
    const float cs = tcs[lane], sn = tsn[lane];
    const float* row = qkv + (size_t) t * (DK * (n_head + n_kv) + DV * n_kv);
    for (int b = warp; b < n_head + 2 * n_kv; b += 8) {
        const bool is_q = b < n_head, is_k = !is_q && b < n_head + n_kv;
        const int ng = is_q || is_k ? 3 : 2;   // 64-value groups
        const float* src = is_q ? row + (size_t) b * DK
                                : (is_k ? row + (size_t) DK * n_head + (size_t) (b - n_head) * DK
                                        : row + (size_t) DK * (n_head + n_kv) + (size_t) (b - n_head - n_kv) * DV);
        float y[6];
#pragma unroll
        for (int g = 0; g < 3; ++g)
            if (g < ng) {
                y[2 * g] = src[64 * g + lane];
                y[2 * g + 1] = src[64 * g + 32 + lane];
            }
        if (ng == 3) rope_pair(y[0], y[1], cs, sn, y[0], y[1]);   // q and k: group 0 is the roped part
        if (is_q) {
            uint16_t* qo = q_out + ((size_t) t * n_head + b) * DK;
#pragma unroll
            for (int g = 0; g < 3; ++g) {
                qo[64 * g + lane] = f16_from_f32(y[2 * g]);
                qo[64 * g + 32 + lane] = f16_from_f32(y[2 * g + 1]);
            }
            continue;
        }
        const int head = is_k ? b - n_head : b - n_head - n_kv, d = is_k ? DK : DV;
#pragma unroll
        for (int g = 0; g < 3; ++g) {
            if (g >= ng) break;
            float a = fmaxf(fabsf(y[2 * g]), fabsf(y[2 * g + 1]));
            for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
            const uint16_t sbits = f16_from_f32(a / 127.0f);
            const float sf = f32_from_f16(sbits);
            const int8_t c0 = (int8_t) q8_code(y[2 * g], sf), c1 = (int8_t) q8_code(y[2 * g + 1], sf);
            auto put = [&](const KvView& dst, size_t slot) {
                slot += (size_t) head * dst.cap;
                int8_t* codes = const_cast<int8_t*>(is_k ? dst.k : dst.v) + slot * d + 64 * g;
                codes[lane] = c0;
                codes[lane + 32] = c1;
                if (lane == 0) const_cast<uint16_t*>(is_k ? dst.ks : dst.vs)[slot * (is_k ? 4 : 2) + g] = sbits;
            };
            if (pos >= kv_from) put(kv, (size_t) (kv.ring ? pos % kv.cap : pos));
            if (stage.k) put(stage, (size_t) (pos - stage_base));
        }
    }
}

// Prompt, SWA layers: the ring's positions pos0 - n_pre .. pos0 - 1 into the staging rows 0 .. n_pre - 1, before
// the chunk's append may overwrite their slots. One block per (position, KV head); positions < 0 are skipped.
__global__ void ring_to_stage_kernel(KvView ring, KvView stage, int pos0, int n_pre) {
    const int p = pos0 - n_pre + (int) blockIdx.x, h = blockIdx.y, i = threadIdx.x;
    if (p < 0) return;
    const size_t src = (size_t) h * ring.cap + (size_t) (p % ring.cap), dst = (size_t) h * stage.cap + blockIdx.x;
    if (i < DK / 16)
        reinterpret_cast<int4*>(const_cast<int8_t*>(stage.k) + dst * DK)[i] = reinterpret_cast<const int4*>(ring.k + src * DK)[i];
    else if (i < (DK + DV) / 16)
        reinterpret_cast<int4*>(const_cast<int8_t*>(stage.v) + dst * DV)[i - DK / 16] =
            reinterpret_cast<const int4*>(ring.v + src * DV)[i - DK / 16];
    else if (i == (DK + DV) / 16)
        reinterpret_cast<uint2*>(const_cast<uint16_t*>(stage.ks))[dst] = reinterpret_cast<const uint2*>(ring.ks)[src];
    else if (i == (DK + DV) / 16 + 1)
        reinterpret_cast<uint32_t*>(const_cast<uint16_t*>(stage.vs))[dst] = reinterpret_cast<const uint32_t*>(ring.vs)[src];
}

// ---- Prompt attention --------------------------------------------------------------------------------------------
// Flash attention (FA2 layout) on mma.sync m16n8k16: grid (m-block, KV head, split), 8 warps. A block owns 128 query
// rows of one KV head (r = t * G + head-in-group: whole tokens, 8 of them at G = 16), warp w rows 16 w .. 16 w + 15,
// Q in registers as A fragments for all 192 dims. Per 64-key tile: cp.async brings the raw int8 codes and scales
// into one of two raw stages, all 256 threads dequantize them once into f16 K and V tiles in shared memory
// (fp16(code x scale): exactly decode's V rounding; for K decode keeps the codes exact and scales the f32 dot product
// instead, a 2^-11 relative difference), and each warp runs S = Q K^T (ldmatrix B fragments), the online softmax in
// f32, and O += P V (ldmatrix.trans). The conversion of V(it) runs alongside S(it) and that of K(it + 1) alongside
// P V(it) (see the loop). Masking (causal, window, chunk end) is computed only for tiles that need it.
//   RTX PRO 6000 Max-Q (mimo_attn_prompt_test --bench): 220-250 TFLOPS on global layers (the f16 MMA peak measured
//   on this card is 444); without the loads and the conversion the same loop runs at ~300, so the remaining gap is
//   the MMA / softmax core with 8 warps per SM. SWA layers reach ~100: a block's 16 tokens read 143 keys in 3 tiles,
//   192 keys of work for the 128 each query sees.
constexpr int PBM = 128, PBN = 64, PNT = 256;
constexpr int PK_STRIDE = 200;   // f16 per shared K row: 400 B, so the 8 rows of an ldmatrix hit 8 different bank quads
constexpr int PV_STRIDE = 136;   // 272 B
constexpr int PR_V = PBN * DK, PR_KS = PR_V + PBN * DV, PR_VS = PR_KS + PBN * 8, PR_STAGE = PR_VS + PBN * 4;
constexpr int PH_K = 2 * PR_STAGE, PH_V = PH_K + PBN * PK_STRIDE * 2, P_SMEM = PH_V + PBN * PV_STRIDE * 2;   // 85,504 B
constexpr int P_MAX_SPLITS = 16;

struct PromptArgs {
    const uint16_t* q;    // [P][n_head][192] fp16 bits
    KvView kv;            // linear (no ring): key position p at row p - base
    int base;
    int pos0, P, G, n_head, n_mb;
    int swa, n_swa;
    float scale_log2;
    int splits;
    const float* sinks;
    float* out;           // splits == 1: [P][n_head][128]
    float* opart;         // splits > 1: [split][n_kv][rows_pad][128] and [..][2]
    float* mlpart;
    int rows_pad;
};

__device__ __forceinline__ void ldsm_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

__global__ void __launch_bounds__(PNT, 1) attn_prompt_kernel(const PromptArgs a) {
    extern __shared__ __align__(16) uint8_t smem[];
    const int mb = a.n_mb - 1 - (int) blockIdx.x;   // the last tokens (the most keys) start first
    const int h = blockIdx.y, split = blockIdx.z, n_kv = gridDim.y;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, c = lane & 3;
    const int tpb = PBM / a.G;
    const int ta = mb * tpb, tb = min(ta + tpb, a.P);
    const int qlo = a.pos0 + ta, qhi = a.pos0 + tb - 1;   // the block's first and last query positions
    int lo = a.swa ? max(0, qlo - a.n_swa + 1) : 0, hi = qhi + 1;
    if (a.splits > 1) {   // whole tiles per split; a split past the end writes an empty partial
        const int per = ((hi - lo + a.splits - 1) / a.splits + PBN - 1) / PBN * PBN;
        lo += split * per;
        hi = min(lo + per, hi);
    }
    const int n_tiles = hi > lo ? (hi - lo + PBN - 1) / PBN : 0;
    const size_t hoff = (size_t) h * a.kv.cap;
    const int8_t* kbase = a.kv.k + hoff * DK;
    const int8_t* vbase = a.kv.v + hoff * DV;
    const uint8_t* ksbase = reinterpret_cast<const uint8_t*>(a.kv.ks + hoff * 4);
    const uint8_t* vsbase = reinterpret_cast<const uint8_t*>(a.kv.vs + hoff * 2);

    // A tile's rows are contiguous in the cache: K 12,288 B and V 8,192 B copied as 1,280 16-byte chunks, then the
    // scales per key (8 B and 4 B: a head's scale rows are only that aligned for an arbitrary cap); the bytes of keys
    // >= hi are zero-filled (src-size 0), never read.
    auto load = [&](int tile) {
        if (tile < n_tiles) {
            uint8_t* st = smem + (tile & 1) * PR_STAGE;
            const int p0 = lo + tile * PBN, nk = min(PBN, hi - p0);
            const size_t row0 = (size_t) (p0 - a.base);
#pragma unroll
            for (int r = 0; r < (1280 + 2 * PBN + PNT - 1) / PNT; ++r) {
                const int i = tid + PNT * r;
                if (i >= 1280 + 2 * PBN) break;
                if (i >= 1280) {
                    const int key = (i - 1280) & (PBN - 1);
                    if (i < 1280 + PBN) cp_async(st + PR_KS + key * 8, ksbase + (row0 + key) * 8, key < nk, 8);
                    else cp_async(st + PR_VS + key * 4, vsbase + (row0 + key) * 4, key < nk, 4);
                    continue;
                }
                const bool isk = i < 768;
                const uint8_t* src = isk ? reinterpret_cast<const uint8_t*>(kbase) + row0 * DK + i * 16
                                         : reinterpret_cast<const uint8_t*>(vbase) + row0 * DV + (i - 768) * 16;
                const int ok = (isk ? i / 12 : (i - 768) / 8) < nk ? 16 : 0;
                const unsigned sa = (unsigned) __cvta_generic_to_shared(st + i * 16);
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(sa), "l"(src), "r"(ok) : "memory");
            }
        }
        cp_commit();
    };
    // raw stage -> f16 tile: fp16(code x scale), 8 codes per unit; K 24 units per key (6 per thread), V 16 (4)
    auto dequant_k = [&](const uint8_t* rs) {
#pragma unroll
        for (int i = 0; i < 6; ++i) {
            const int u = tid + PNT * i, key = u / 24, c8 = u - 24 * key;
            const uint2 w = *reinterpret_cast<const uint2*>(rs + key * DK + c8 * 8);
            const uint32_t sb = *reinterpret_cast<const uint16_t*>(rs + PR_KS + key * 8 + (c8 >> 3) * 2);
            const __half2 sc = bits_h2(sb | (sb << 16));
            uint32_t x0, x1, x2, x3;
            i8x4_to_h2(w.x, x0, x1);
            i8x4_to_h2(w.y, x2, x3);
            *reinterpret_cast<uint4*>(smem + PH_K + (key * PK_STRIDE + c8 * 8) * 2) =
                make_uint4(h2_bits(__hmul2(bits_h2(x0), sc)), h2_bits(__hmul2(bits_h2(x1), sc)),
                           h2_bits(__hmul2(bits_h2(x2), sc)), h2_bits(__hmul2(bits_h2(x3), sc)));
        }
    };
    auto dequant_v = [&](const uint8_t* rs) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int u = tid + PNT * i, key = u >> 4, c8 = u & 15;
            const uint2 w = *reinterpret_cast<const uint2*>(rs + PR_V + key * DV + c8 * 8);
            const uint32_t sb = *reinterpret_cast<const uint16_t*>(rs + PR_VS + key * 4 + (c8 >> 3) * 2);
            const __half2 sc = bits_h2(sb | (sb << 16));
            uint32_t x0, x1, x2, x3;
            i8x4_to_h2(w.x, x0, x1);
            i8x4_to_h2(w.y, x2, x3);
            *reinterpret_cast<uint4*>(smem + PH_V + (key * PV_STRIDE + c8 * 8) * 2) =
                make_uint4(h2_bits(__hmul2(bits_h2(x0), sc)), h2_bits(__hmul2(bits_h2(x1), sc)),
                           h2_bits(__hmul2(bits_h2(x2), sc)), h2_bits(__hmul2(bits_h2(x3), sc)));
        }
    };
    load(0);
    load(1);

    // Q A fragments of rows rb and rb + 8 (zero past the chunk): k-step s covers dims 16 s .. 16 s + 15
    const int rb = mb * PBM + 16 * warp + g;
    const int t0 = rb / a.G, t1 = (rb + 8) / a.G;
    const bool ok0 = t0 < a.P, ok1 = t1 < a.P;
    const int hq0 = h * a.G + rb % a.G, hq1 = h * a.G + (rb + 8) % a.G;
    uint32_t qa[12][4];
    {
        const uint16_t* q0 = a.q + ((size_t) (ok0 ? t0 : 0) * a.n_head + hq0) * DK;
        const uint16_t* q1 = a.q + ((size_t) (ok1 ? t1 : 0) * a.n_head + hq1) * DK;
#pragma unroll
        for (int s = 0; s < 12; ++s) {
            const int d = 16 * s + 2 * c;
            qa[s][0] = ok0 ? *reinterpret_cast<const uint32_t*>(q0 + d) : 0u;
            qa[s][1] = ok1 ? *reinterpret_cast<const uint32_t*>(q1 + d) : 0u;
            qa[s][2] = ok0 ? *reinterpret_cast<const uint32_t*>(q0 + d + 8) : 0u;
            qa[s][3] = ok1 ? *reinterpret_cast<const uint32_t*>(q1 + d + 8) : 0u;
        }
    }
    const int qp0 = a.pos0 + t0, qp1 = a.pos0 + t1;

    // this lane's ldmatrix row addresses: K (non-trans) rows = keys 8 j + lane % 8, 8-dim column block lane / 8;
    // V (trans) rows = keys lane % 8 + 8 ((lane / 8) % 2), column block lane / 16
    const uint32_t kh_s = (uint32_t) __cvta_generic_to_shared(smem + PH_K);
    const uint32_t vh_s = (uint32_t) __cvta_generic_to_shared(smem + PH_V);
    const uint32_t k_lane = kh_s + (uint32_t) (((lane & 7) * PK_STRIDE + 8 * (lane >> 3)) * 2);
    const uint32_t v_lane = vh_s + (uint32_t) ((((lane & 7) + 8 * ((lane >> 3) & 1)) * PV_STRIDE + 8 * (lane >> 4)) * 2);

    float o[16][4];
#pragma unroll
    for (int j = 0; j < 16; ++j) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.0f;
    float m0 = NEG_INF, m1 = NEG_INF, l0 = 0.0f, l1 = 0.0f;
    // Software pipeline, two barriers per tile. Phase 1 of tile it: S = Q K(it)^T from the f16 K tile while V(it) is
    // dequantized into the f16 V tile; phase 2: O += P V(it) while K(it + 1) is dequantized into the K tile. The two
    // instruction streams are independent, so the tensor-core and the conversion work overlap. Raw stage it & 1
    // holds tile it until its V is converted (phase 1 of it), then takes tile it + 2.
    if (n_tiles > 0) {
        cp_wait<1>();
        __syncthreads();
        dequant_k(smem);
    }
    for (int it = 0; it < n_tiles; ++it) {
        __syncthreads();   // K(it) converted, every warp done with V(it - 1)
        const int p0 = lo + it * PBN;

        // S = Q K^T: 8 n-tiles of 8 keys
        float s[8][4];
#pragma unroll
        for (int j = 0; j < 8; ++j) s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.0f;
#pragma unroll
        for (int kk = 0; kk < 6; ++kk)
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                uint32_t b0, b1, b2, b3;
                ldsm_x4(b0, b1, b2, b3, k_lane + (uint32_t) ((8 * j * PK_STRIDE + 32 * kk) * 2));
                mma16816(s[j], qa[2 * kk], b0, b1);
                mma16816(s[j], qa[2 * kk + 1], b2, b3);
            }
        dequant_v(smem + (it & 1) * PR_STAGE);   // after the MMAs in program order: it issues while they run
        const bool need_mask = p0 + PBN > hi || p0 + PBN - 1 > qlo || (a.swa && p0 <= qhi - a.n_swa);
#pragma unroll
        for (int j = 0; j < 8; ++j)
#pragma unroll
            for (int i = 0; i < 4; ++i) s[j][i] *= a.scale_log2;
        if (need_mask) {   // one branch around the whole mask: the common unmasked tile stays one basic block
#pragma unroll
            for (int j = 0; j < 8; ++j)
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int p = p0 + 8 * j + 2 * c + (i & 1), qp = i < 2 ? qp0 : qp1;
                    s[j][i] = p < hi && p <= qp && (!a.swa || qp - p < a.n_swa) ? s[j][i] : NEG_INF;
                }
        }
        // online softmax, rows g (elements 0, 1) and g + 8 (2, 3); a quad shares a row
        float mx0 = NEG_INF, mx1 = NEG_INF;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            mx0 = fmaxf(mx0, fmaxf(s[j][0], s[j][1]));
            mx1 = fmaxf(mx1, fmaxf(s[j][2], s[j][3]));
        }
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
        const float mn0 = fmaxf(m0, mx0), mn1 = fmaxf(m1, mx1);
        const float ref0 = mn0 == NEG_INF ? 0.0f : mn0, ref1 = mn1 == NEG_INF ? 0.0f : mn1;
        const float al0 = exp2f(m0 - ref0), al1 = exp2f(m1 - ref1);
        m0 = mn0;
        m1 = mn1;
        float rs0 = 0.0f, rs1 = 0.0f;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            s[j][0] = exp2f(s[j][0] - ref0);
            s[j][1] = exp2f(s[j][1] - ref0);
            s[j][2] = exp2f(s[j][2] - ref1);
            s[j][3] = exp2f(s[j][3] - ref1);
            rs0 += s[j][0] + s[j][1];
            rs1 += s[j][2] + s[j][3];
        }
        l0 = l0 * al0 + rs0;
        l1 = l1 * al1 + rs1;
        if (!__all_sync(0xffffffffu, al0 == 1.0f && al1 == 1.0f)) {   // later tiles mostly leave the maxima alone
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                o[j][0] *= al0;
                o[j][1] *= al0;
                o[j][2] *= al1;
                o[j][3] *= al1;
            }
        }

        cp_wait<0>();      // tile it + 1 has landed (this thread's chunks; the barrier publishes everyone's)
        __syncthreads();   // V(it) converted, every warp done with K(it)
        load(it + 2);      // into stage it & 1, free now

        // O += P V: k-step kk = keys 16 kk .. 16 kk + 15 (the S n-tiles 2 kk, 2 kk + 1 are its A fragment)
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            const uint32_t pa[4] = {pack_h2(s[2 * kk][0], s[2 * kk][1]), pack_h2(s[2 * kk][2], s[2 * kk][3]),
                                    pack_h2(s[2 * kk + 1][0], s[2 * kk + 1][1]),
                                    pack_h2(s[2 * kk + 1][2], s[2 * kk + 1][3])};
#pragma unroll
            for (int np = 0; np < 8; ++np) {
                uint32_t b0, b1, b2, b3;
                ldsm_x4_t(b0, b1, b2, b3, v_lane + (uint32_t) ((16 * kk * PV_STRIDE + 16 * np) * 2));
                mma16816(o[2 * np], pa, b0, b1);
                mma16816(o[2 * np + 1], pa, b2, b3);
            }
        }
        if (it + 1 < n_tiles) dequant_k(smem + ((it + 1) & 1) * PR_STAGE);
    }
    l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
    l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 2);

    if (a.splits == 1) {
        // normalize, the sink of the row's query head in the denominator (ggml soft_max with sinks)
        auto final_scale = [&](int hq, float m, float l) {
            const float sk = a.sinks ? __ldg(a.sinks + hq) * LOG2E : NEG_INF;
            const float M = fmaxf(m, sk);
            if (M == NEG_INF) return 0.0f;
            const float f = exp2f(m - M);
            return f / (l * f + (a.sinks ? exp2f(sk - M) : 0.0f));
        };
        const float n0 = final_scale(hq0, m0, l0), n1 = final_scale(hq1, m1, l1);
        float* out0 = a.out + ((size_t) (ok0 ? t0 : 0) * a.n_head + hq0) * DV + 2 * c;
        float* out1 = a.out + ((size_t) (ok1 ? t1 : 0) * a.n_head + hq1) * DV + 2 * c;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            if (ok0) *reinterpret_cast<float2*>(out0 + 8 * j) = make_float2(o[j][0] * n0, o[j][1] * n0);
            if (ok1) *reinterpret_cast<float2*>(out1 + 8 * j) = make_float2(o[j][2] * n1, o[j][3] * n1);
        }
    } else {
        const size_t prow = ((size_t) split * n_kv + h) * a.rows_pad + rb;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            if (ok0) *reinterpret_cast<float2*>(a.opart + prow * DV + 8 * j + 2 * c) = make_float2(o[j][0], o[j][1]);
            if (ok1)
                *reinterpret_cast<float2*>(a.opart + (prow + 8) * DV + 8 * j + 2 * c) = make_float2(o[j][2], o[j][3]);
        }
        if (c == 0) {
            if (ok0) *reinterpret_cast<float2*>(a.mlpart + prow * 2) = make_float2(m0, l0);
            if (ok1) *reinterpret_cast<float2*>(a.mlpart + (prow + 8) * 2) = make_float2(m1, l1);
        }
    }
}

// Split-K merge: one block per (query head, token), one thread per output dim; the sink joins the denominator.
__global__ void __launch_bounds__(DV) attn_prompt_combine_kernel(const float* __restrict__ opart,
                                                                 const float* __restrict__ mlpart, int splits, int G,
                                                                 int n_kv, int rows_pad,
                                                                 const float* __restrict__ sinks,
                                                                 float* __restrict__ out) {
    const int hq = blockIdx.x, t = blockIdx.y, d = threadIdx.x, n_head = gridDim.x;
    const size_t stride = (size_t) n_kv * rows_pad, row0 = (size_t) (hq / G) * rows_pad + (size_t) t * G + hq % G;
    const float sink = sinks ? __ldg(sinks + hq) * LOG2E : NEG_INF;
    float M = sink;
    for (int s = 0; s < splits; ++s) M = fmaxf(M, mlpart[(row0 + s * stride) * 2]);
    float den = sinks ? exp2f(sink - M) : 0.0f, acc = 0.0f;
    for (int s = 0; s < splits; ++s) {
        const float ms = mlpart[(row0 + s * stride) * 2];
        const float f = ms == NEG_INF ? 0.0f : exp2f(ms - M);
        den += f * mlpart[(row0 + s * stride) * 2 + 1];
        acc += f * opart[(row0 + s * stride) * DV + d];
    }
    out[((size_t) t * n_head + hq) * DV + d] = M == NEG_INF ? 0.0f : acc / den;
}

// One block per row; each thread keeps its float4s of x in registers between the two passes (n <= 4 * 4 * 1024).
__global__ void __launch_bounds__(1024) rmsnorm_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                                       float* __restrict__ y, int n, float eps) {
    const float4* xr = reinterpret_cast<const float4*>(x + (size_t) blockIdx.x * n);
    float4* yr = reinterpret_cast<float4*>(y + (size_t) blockIdx.x * n);
    const float4* wr = reinterpret_cast<const float4*>(w);
    const int n4 = n / 4;
    float4 v[4];
    float s = 0.0f;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int i = threadIdx.x + k * blockDim.x;
        v[k] = i < n4 ? xr[i] : make_float4(0, 0, 0, 0);
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
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int i = threadIdx.x + k * blockDim.x;
        if (i < n4) {
            const float4 ww = wr[i];
            yr[i] = make_float4((v[k].x * r) * ww.x, (v[k].y * r) * ww.y, (v[k].z * r) * ww.z, (v[k].w * r) * ww.w);
        }
    }
}

__global__ void residual_add_kernel(float* __restrict__ x, const float* __restrict__ y, float scale, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = __fadd_rn(x[i], __fmul_rn(y[i], scale));   // ggml: scale, then add (no fma)
}

KvView view(const AttnKv& kv) { return KvView{kv.k, kv.v, kv.k_scale, kv.v_scale, kv.cap, kv.ring ? 1 : 0}; }

template <int NMT>
void set_smem_attr() {
    static bool done = false;
    if (done) return;
    cudaFuncSetAttribute(attn_split_kernel<NMT>, cudaFuncAttributeMaxDynamicSharedMemorySize, Cfg<NMT>::SMEM);
    check("cudaFuncSetAttribute");
    done = true;
}
void set_all_attrs() {
    set_smem_attr<1>(); set_smem_attr<2>(); set_smem_attr<3>(); set_smem_attr<4>();
    set_smem_attr<5>(); set_smem_attr<6>(); set_smem_attr<7>(); set_smem_attr<8>();
}

template <int NMT>
void launch_split(const float* q, const int32_t* pos0, int T, const AttnLayer& L, const AttnKv& kv,
                 const AttnScratch& s, int splits, float* out, cudaStream_t st) {
    using C = Cfg<NMT>;
    set_smem_attr<NMT>();
    const dim3 grid((unsigned) splits, (unsigned) L.n_kv);
    attn_split_kernel<NMT><<<grid, C::NT, C::SMEM, st>>>(q, pos0, T, L.n_head / L.n_kv, L.n_head, view(kv),
                                                          L.swa ? 1 : 0, L.n_swa, L.scale * LOG2E, splits, s.o, s.ml,
                                                          L.sinks, out);
    check("attn_split launch");

}

}  // namespace

size_t attn_kv_bytes(int n_kv, int cap) { return (size_t) n_kv * cap * (DK + DV + 8 + 4); }

bool attn_kv_alloc(AttnKv& kv, int n_kv, int cap, bool ring, std::string& err) {
    const size_t n = (size_t) n_kv * cap;
    void* p = nullptr;
    const cudaError_t e = cudaMalloc(&p, attn_kv_bytes(n_kv, cap));
    if (e != cudaSuccess) {
        err = "MiMo KV: cudaMalloc of " + std::to_string(attn_kv_bytes(n_kv, cap) >> 20) + " MiB failed: " +
              cudaGetErrorString(e);
        return false;
    }
    cudaMemset(p, 0, attn_kv_bytes(n_kv, cap));
    uint8_t* b = static_cast<uint8_t*>(p);
    kv.k = reinterpret_cast<int8_t*>(b);
    kv.v = reinterpret_cast<int8_t*>(b + n * DK);
    kv.k_scale = reinterpret_cast<uint16_t*>(b + n * (DK + DV));
    kv.v_scale = reinterpret_cast<uint16_t*>(b + n * (DK + DV + 8));
    kv.n_kv = n_kv;
    kv.cap = cap;
    kv.ring = ring;
    return true;
}

void attn_kv_free(AttnKv& kv) {
    if (kv.k) cudaFree(kv.k);
    kv = AttnKv{};
}

AttnLayer attn_layer(int n_head, int n_kv, bool swa, int n_swa, float rope_base, int head_k, const float* sinks) {
    AttnLayer L;
    L.n_head = n_head;
    L.n_kv = n_kv;
    L.swa = swa;
    L.n_swa = n_swa;
    L.theta_scale = powf(rope_base, -2.0f / (float) NROT);
    L.scale = 1.0f / sqrtf((float) head_k);
    L.sinks = sinks;
    return L;
}

int attn_default_splits(int n_kv) {
    int dev = 0, sms = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    return std::max(1, sms / std::max(1, n_kv));
}

bool attn_scratch_alloc(AttnScratch& s, int blocks, std::string& err) {
    if (blocks <= 0) blocks = attn_default_splits(1);   // the SM count: splits x n_kv of every default launch
    const size_t rows = (size_t) blocks * MAX_ROWS;
    if (cudaMalloc(&s.o, rows * DV * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&s.ml, rows * 2 * sizeof(float)) != cudaSuccess) {
        err = "MiMo attention scratch: cudaMalloc failed";
        attn_scratch_free(s);
        return false;
    }
    s.blocks = blocks;
    set_all_attrs();   // before any graph capture
    return true;
}

void attn_scratch_free(AttnScratch& s) {
    if (s.o) cudaFree(s.o);
    if (s.ml) cudaFree(s.ml);
    s = AttnScratch{};
}

void mimo_rmsnorm(const float* x, const float* w, float* y, int n, int T, float eps, void* stream) {
    if (n % 4 != 0 || n > 16 * 1024) {
        std::fprintf(stderr, "mimo_rmsnorm: n = %d must be a multiple of 4 and at most 16384\n", n);
        std::exit(1);
    }
    const int threads = std::min(1024, ((n / 4 + 3) / 4 + 31) / 32 * 32);
    rmsnorm_kernel<<<T, threads, 0, (cudaStream_t) stream>>>(x, w, y, n, eps);
    check("rmsnorm launch");
}

void mimo_qkv_rope_append(const float* qkv, const int32_t* pos0, int T, const AttnLayer& L, const AttnKv& kv,
                          float* q_out, void* stream) {
    const dim3 grid((unsigned) (L.n_head + 2 * L.n_kv), (unsigned) T);
    qkv_rope_append_kernel<<<grid, DK, 0, (cudaStream_t) stream>>>(qkv, pos0, L.n_head, L.n_kv, L.theta_scale,
                                                                    view(kv), q_out);
    check("qkv_rope_append launch");
}

void mimo_attn_decode(const float* q, const int32_t* pos0, int T, const AttnLayer& L, const AttnKv& kv,
                      const AttnScratch& s, float* out, void* stream, int max_splits, void* out_q8_1) {
    const int G = L.n_head / L.n_kv;
    if (T < 1 || T > kAttnMaxT || T * G > MAX_ROWS || L.n_head > kAttnMaxHeads || L.n_kv > s.blocks) {
        std::fprintf(stderr, "mimo_attn_decode: unsupported T %d / n_head %d / n_kv %d\n", T, L.n_head, L.n_kv);
        std::exit(1);
    }
    int splits = max_splits > 0 ? max_splits : attn_default_splits(L.n_kv);
    splits = std::min({splits, s.blocks / L.n_kv, MAX_SPLITS});
    const cudaStream_t st = (cudaStream_t) stream;
    const int nmt = (T * G + 15) / 16;
    // T = 1 on an SWA layer (128 keys, 8 rows per KV head): one block per KV head takes all keys and writes the
    // output itself, one launch instead of a split + combine pair (8.3 against 8.7 us, see the test's --bench).
    // For T > 1 the split + combine pair is faster (T = 4: 9.1 against 12.8 us): more blocks share the rows' work.
    float* direct = L.swa && T == 1 ? out : nullptr;
    if (direct) splits = 1;

    switch (nmt) {
        case 1: launch_split<1>(q, pos0, T, L, kv, s, splits, direct, st); break;
        case 2: launch_split<2>(q, pos0, T, L, kv, s, splits, direct, st); break;
        case 3: launch_split<3>(q, pos0, T, L, kv, s, splits, direct, st); break;
        case 4: launch_split<4>(q, pos0, T, L, kv, s, splits, direct, st); break;
        case 5: launch_split<5>(q, pos0, T, L, kv, s, splits, direct, st); break;
        case 6: launch_split<6>(q, pos0, T, L, kv, s, splits, direct, st); break;
        case 7: launch_split<7>(q, pos0, T, L, kv, s, splits, direct, st); break;
        default: launch_split<8>(q, pos0, T, L, kv, s, splits, direct, st); break;
    }
    if (direct) {
        if (out_q8_1) kernels::native_quantize_q8_1(out, out_q8_1, L.n_head * DV, T, stream);
        return;
    }
    const dim3 grid((unsigned) L.n_head, (unsigned) T);
    attn_combine_kernel<<<grid, DV, 0, st>>>(s.o, s.ml, pos0, T, G, L.n_kv, L.swa ? 1 : 0, L.n_swa, splits, L.sinks,
                                             out, static_cast<MimoQ81*>(out_q8_1));
    check("attn_combine launch");
}

void mimo_residual_add(float* x, const float* y, float scale, int n, void* stream) {
    residual_add_kernel<<<(n + 255) / 256, 256, 0, (cudaStream_t) stream>>>(x, y, scale, n);
    check("residual_add launch");
}

// ---- prompt path ----------------------------------------------------------------------------------------------

namespace {

int sm_count() {
    static int n = 0;
    if (n == 0) n = attn_default_splits(1);
    return n;
}

void set_prompt_attr() {
    static bool done = false;
    if (done) return;
    cudaFuncSetAttribute(attn_prompt_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, P_SMEM);
    check("cudaFuncSetAttribute (prompt)");
    done = true;
}

// Split-K count for a chunk: only when the prior context dominates (every block reads about the same pos0 + P keys,
// so the wave count decides the time) - the S that minimizes whole waves per unit of work, plus a small cost per
// split for the partials and the merge; at least 1,024 keys per split.
int auto_splits(int blocks, int keys, int sms) {
    int best = 1;
    double best_cost = 1e30;
    for (int S = 1; S <= P_MAX_SPLITS && (S == 1 || keys / S >= 1024); ++S) {
        const double waves = (double) ((blocks * S + sms - 1) / sms);
        const double cost = waves / S + 0.02 * S;
        if (cost < best_cost - 1e-9) {
            best_cost = cost;
            best = S;
        }
    }
    return best;
}

}  // namespace

bool attn_prompt_scratch_alloc(AttnPromptScratch& s, int max_p, int n_kv_swa, int n_swa, size_t part_bytes,
                               std::string& err) {
    s.max_p = max_p > 0 ? max_p : kAttnPromptMaxP;
    s.n_swa = n_swa;
    if (part_bytes == 0) part_bytes = (size_t) 64 << 20;
    if (!attn_kv_alloc(s.swa, n_kv_swa, s.max_p + n_swa, false, err)) return false;
    s.part_rows = part_bytes / ((DV + 2) * sizeof(float));
    if (cudaMalloc(&s.o, s.part_rows * DV * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&s.ml, s.part_rows * 2 * sizeof(float)) != cudaSuccess) {
        err = "MiMo prompt attention scratch: cudaMalloc failed";
        attn_prompt_scratch_free(s);
        return false;
    }
    set_prompt_attr();
    return true;
}

void attn_prompt_scratch_free(AttnPromptScratch& s) {
    attn_kv_free(s.swa);
    if (s.o) cudaFree(s.o);
    if (s.ml) cudaFree(s.ml);
    s = AttnPromptScratch{};
}

void mimo_qkv_rope_append_prompt(const float* qkv, int pos0, int P, const AttnLayer& L, const AttnKv& kv,
                                 const AttnPromptScratch& s, uint16_t* q_out, void* stream) {
    const cudaStream_t st = (cudaStream_t) stream;
    const bool bad = P < 1 || pos0 < 0 ||
                     (L.swa ? (!kv.ring || P > s.max_p || s.n_swa < L.n_swa || s.swa.n_kv != L.n_kv)
                            : (kv.ring || pos0 + P > kv.cap));
    if (bad) {
        std::fprintf(stderr, "mimo_qkv_rope_append_prompt: unsupported chunk (pos0 %d, P %d, swa %d, cap %d)\n", pos0, P,
                     L.swa ? 1 : 0, kv.cap);
        std::exit(1);
    }
    if (L.swa) {
        ring_to_stage_kernel<<<dim3((unsigned) s.n_swa, (unsigned) L.n_kv), 32, 0, st>>>(view(kv), view(s.swa), pos0,
                                                                                        s.n_swa);
        check("ring_to_stage launch");
        qkv_rope_append_prompt_kernel<<<(unsigned) P, 256, 0, st>>>(qkv, pos0, L.n_head, L.n_kv, L.theta_scale, view(kv),
                                                                    pos0 + P - kv.cap, view(s.swa), pos0 - s.n_swa,
                                                                    q_out);
    } else {
        qkv_rope_append_prompt_kernel<<<(unsigned) P, 256, 0, st>>>(qkv, pos0, L.n_head, L.n_kv, L.theta_scale, view(kv),
                                                                    0, KvView{}, 0, q_out);
    }
    check("qkv_rope_append (prompt) launch");
}

void mimo_attn_prompt(const uint16_t* q, int pos0, int P, const AttnLayer& L, const AttnKv& kv,
                      const AttnPromptScratch& s, float* out, void* stream, int splits) {
    const int G = L.n_head / L.n_kv;
    const bool bad = P < 1 || pos0 < 0 || L.n_head % L.n_kv != 0 || PBM % G != 0 || L.n_head > kAttnMaxHeads ||
                     (L.swa ? (P > s.max_p || s.n_swa < L.n_swa || s.swa.n_kv != L.n_kv)
                            : (kv.ring || pos0 + P > kv.cap));
    if (bad) {
        std::fprintf(stderr, "mimo_attn_prompt: unsupported chunk (pos0 %d, P %d, n_head %d, n_kv %d, swa %d)\n", pos0,
                     P, L.n_head, L.n_kv, L.swa ? 1 : 0);
        std::exit(1);
    }
    set_prompt_attr();
    PromptArgs a{};
    a.q = q;
    a.kv = L.swa ? view(s.swa) : view(kv);
    a.kv.ring = 0;
    a.base = L.swa ? pos0 - s.n_swa : 0;
    a.pos0 = pos0;
    a.P = P;
    a.G = G;
    a.n_head = L.n_head;
    a.n_mb = (P * G + PBM - 1) / PBM;
    a.swa = L.swa ? 1 : 0;
    a.n_swa = L.n_swa;
    a.scale_log2 = L.scale * LOG2E;
    a.sinks = L.sinks;
    a.out = out;
    a.rows_pad = a.n_mb * PBM;
    const int blocks = a.n_mb * L.n_kv;
    if (splits <= 0) splits = !L.swa && pos0 >= P ? auto_splits(blocks, pos0 + P, sm_count()) : 1;
    splits = std::min(splits, P_MAX_SPLITS);
    while (splits > 1 && (size_t) splits * L.n_kv * a.rows_pad > s.part_rows) --splits;
    a.splits = splits;
    a.opart = s.o;
    a.mlpart = s.ml;
    const cudaStream_t st = (cudaStream_t) stream;
    attn_prompt_kernel<<<dim3((unsigned) a.n_mb, (unsigned) L.n_kv, (unsigned) splits), PNT, P_SMEM, st>>>(a);
    check("attn_prompt launch");
    if (splits > 1) {
        attn_prompt_combine_kernel<<<dim3((unsigned) L.n_head, (unsigned) P), DV, 0, st>>>(
            s.o, s.ml, splits, G, L.n_kv, a.rows_pad, L.sinks, out);
        check("attn_prompt_combine launch");
    }
}

}  // namespace strata::mimo
