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
#include "strata/kernels/f16_bits.hpp"

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
                                                          const float* __restrict__ sinks, float* __restrict__ out) {
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
    out[((size_t) t * n_head + hq) * DV + d] = ((a0 + a1) + (a2 + a3)) / den;
}

// One block per (head of q, k or v, token), one thread per dim: NEOX rope on dims 0..63 of q and k, k and v
// quantized into the cache (one fp16 scale per 64 values, kv_q8's rule).
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
        const int j = i & (NROT / 2 - 1);
        const float theta = (float) pos * powf(theta_scale, (float) j);
        float sn, cs;
        sincosf(theta, &sn, &cs);
        const float xp = src[i ^ (NROT / 2)];
        y = i < NROT / 2 ? y * cs - xp * sn : xp * sn + y * cs;
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
    const float amax = fmaxf(wmax[2 * grp], wmax[2 * grp + 1]);
    const uint16_t sbits = f16_from_f32(amax / 127.0f);
    const float sf = f32_from_f16(sbits);   // quantize against the STORED scale
    int code = 0;
    if (sf > 0.0f) {
        code = __float2int_rn(y / sf);
        code = code < -127 ? -127 : (code > 127 ? 127 : code);
    }
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
                      const AttnScratch& s, float* out, void* stream, int max_splits) {
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
    if (direct) return;
    const dim3 grid((unsigned) L.n_head, (unsigned) T);
    attn_combine_kernel<<<grid, DV, 0, st>>>(s.o, s.ml, pos0, T, G, L.n_kv, L.swa ? 1 : 0, L.n_swa, splits, L.sinks,
                                             out);
    check("attn_combine launch");
}

void mimo_residual_add(float* x, const float* y, float scale, int n, void* stream) {
    residual_add_kernel<<<(n + 255) / 256, 256, 0, (cudaStream_t) stream>>>(x, y, scale, n);
    check("residual_add launch");
}

}  // namespace strata::mimo
