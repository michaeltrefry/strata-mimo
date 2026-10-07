// src/kernels/cpu/mxfp4_avx512.cpp - MiMo step 2: MXFP4 rows against Q8_0 activations, AVX-512, up to 8 tokens per
// weight row.  See the header for why and for the bitwise contract with ggml's `ggml_vec_dot_mxfp4_q8_0`.
//
// block_mxfp4 is 17 bytes: one E8M0 exponent `e`, then 16 bytes of 4-bit codes; code j's low nibble is value j and
// its high nibble value j + 16.  A value is kvalues_mxfp4[code] * 2^(e - 128) (the table holds twice the E2M1 value,
// so ggml's scale is "e8m0 half").  block_q8_0 is 34 bytes: an fp16 scale, then 32 int8 values.
#include "strata/kernels/cpu/mxfp4_avx512.hpp"

#include <immintrin.h>

#include <cstdlib>
#include <cstring>

namespace strata::kernels::cpu {
namespace {

constexpr int kMaxT = 8;
constexpr int kMaxN = 8192;
constexpr int kBlk = 17;                                     // bytes of one block_mxfp4
constexpr unsigned long long kPairMask = (1ull << 34) - 1;   // two blocks

// ggml_e8m0_to_fp32_half (ggml-impl.h): 2^(e - 128), the two smallest as denormals
struct E8Table {
    float v[256];
    E8Table() {
        for (int x = 0; x < 256; ++x) {
            const uint32_t bits = x < 2 ? 0x00200000u << x : (uint32_t) (x - 1) << 23;
            std::memcpy(&v[x], &bits, 4);
        }
    }
};
const E8Table kE8;

int g_prefetch = [] {
    const char* v = std::getenv("STRATA_MXFP4_PREFETCH");
    return v ? std::atoi(v) : 0;
}();

// The activations of one call: int8 values contiguous per token, and the fp16 scales as floats.  The scales stay
// compact (one float per block, broadcast to a pair's [d0 x 8 | d1 x 8] in the loop) so that eight tokens' rows
// (8 x 4.5 KiB at n = 4096) stay in L1; pre-broadcast scales doubled that and made 8 tokens 1.4x slower than
// 2 x 4.  Unaligned loads throughout (thread_local over-alignment is not reliable on MSVC; an unaligned load of
// aligned data costs nothing).
struct Planar {
    int8_t q[kMaxT][kMaxN];
    float ys[kMaxT][kMaxN / 32];
};
thread_local Planar tl_act;

inline float h2f(uint16_t h) { return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) h))); }

void prepare(const void* const* act, int nt, int nb) {
    for (int t = 0; t < nt; ++t) {
        const uint8_t* y = (const uint8_t*) act[t];
        for (int ib = 0; ib < nb; ++ib) {
            const uint8_t* blk = y + (size_t) ib * 34;
            std::memcpy(&tl_act.q[t][ib * 32], blk + 2, 32);
            uint16_t d;
            std::memcpy(&d, blk, 2);
            tl_act.ys[t][ib] = h2f(d);
        }
    }
}

// ggml's hsum_float_8 of (even-block lanes + odd-block lanes): the same additions in the same order
inline float hsum(__m512 a) {
    const __m256 x = _mm256_add_ps(_mm512_castps512_ps256(a), _mm512_extractf32x8_ps(a, 1));
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

// One weight row against NT tokens.  Per pair of blocks: bytes [0, 34) loaded once; vpermb puts block 0's codes in
// 128-bit lanes 0 and 1 and block 1's in lanes 2 and 3; lanes 1 and 3 take the high nibbles.  Then |value| by table
// and the sign as a mask, so `vpdpbusd` (unsigned x signed) sums |w| * (y with w's sign): ggml's
// _mm256_sign_epi8 / maddubs / madd, the same int32 lanes (four adjacent products each).
template <int NT>
inline void row_dot(const uint8_t* x, int nb, int pf, float* res) {
    const __m512i idx = _mm512_set_epi8(33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18,
                                        33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18,
                                        16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1,
                                        16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1);
    const __m512i tbl = _mm512_set_epi8(12, 8, 6, 4, 3, 2, 1, 0, 12, 8, 6, 4, 3, 2, 1, 0,
                                        12, 8, 6, 4, 3, 2, 1, 0, 12, 8, 6, 4, 3, 2, 1, 0,
                                        12, 8, 6, 4, 3, 2, 1, 0, 12, 8, 6, 4, 3, 2, 1, 0,
                                        12, 8, 6, 4, 3, 2, 1, 0, 12, 8, 6, 4, 3, 2, 1, 0);
    const __m512i m4 = _mm512_set1_epi8(0x0f), s8 = _mm512_set1_epi8(0x08), zero = _mm512_setzero_si512();
    const __m512i pair = _mm512_set_epi32(1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0);
    __m512 acc[NT];
    for (int t = 0; t < NT; ++t) acc[t] = _mm512_setzero_ps();
    for (int ib = 0; ib < nb; ib += 2) {
        const uint8_t* b = x + (size_t) ib * kBlk;
        if (pf) _mm_prefetch((const char*) b + pf, _MM_HINT_T0);
        const __m512i raw = _mm512_maskz_loadu_epi8(kPairMask, b);
        __m512i qq = _mm512_permutexvar_epi8(idx, raw);
        qq = _mm512_mask_blend_epi64(0xCC, qq, _mm512_srli_epi16(qq, 4));
        const __m512i nib = _mm512_and_si512(qq, m4);
        const __m512i aw = _mm512_shuffle_epi8(tbl, nib);
        const __mmask64 neg = _mm512_test_epi8_mask(nib, s8);
        const __m512 wv = _mm512_insertf32x8(_mm512_set1_ps(kE8.v[b[0]]), _mm256_set1_ps(kE8.v[b[kBlk]]), 1);
        for (int t = 0; t < NT; ++t) {
            const __m512i y = _mm512_loadu_si512(&tl_act.q[t][ib * 32]);
            const __m512i sy = _mm512_mask_sub_epi8(y, neg, zero, y);
            const __m512i p = _mm512_dpbusd_epi32(zero, aw, sy);
            // fp16(d_y) * e8m0_half(e), as ggml forms it (the product is commutative, so bitwise the same)
            double d01;   // the pair's two scales, one 8-byte broadcast, then lanes 0-7 <- d0, 8-15 <- d1
            std::memcpy(&d01, &tl_act.ys[t][ib], 8);
            const __m512 yv = _mm512_permutexvar_ps(pair, _mm512_castpd_ps(_mm512_set1_pd(d01)));
            const __m512 s = _mm512_mul_ps(yv, wv);
            acc[t] = _mm512_fmadd_ps(s, _mm512_cvtepi32_ps(p), acc[t]);
        }
    }
    for (int t = 0; t < NT; ++t) res[t] = hsum(acc[t]);
}

template <int NT>
void rows_t(const uint8_t* w, size_t row_bytes, int nb, float* const* out, int r0, int r1) {
    const int pf = g_prefetch;
    float res[NT];
    for (int r = r0; r < r1; ++r) {
        row_dot<NT>(w + (size_t) r * row_bytes, nb, pf, res);
        for (int t = 0; t < NT; ++t) out[t][r] = res[t];
    }
}

template <int NT>
void rows2_t(const uint8_t* gate, const uint8_t* up, size_t row_bytes, int nb, float* const* g, float* const* u,
             int r0, int r1) {
    const int pf = g_prefetch;
    float rg[NT], ru[NT];
    for (int r = r0; r < r1; ++r) {
        row_dot<NT>(gate + (size_t) r * row_bytes, nb, pf, rg);
        row_dot<NT>(up + (size_t) r * row_bytes, nb, pf, ru);
        for (int t = 0; t < NT; ++t) { g[t][r] = rg[t]; u[t][r] = ru[t]; }
    }
}

}  // namespace

int mxfp4_prefetch_bytes() { return g_prefetch; }
void mxfp4_set_prefetch_bytes(int bytes) { g_prefetch = bytes; }

void mxfp4_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out, int r0,
                int r1) {
    if (nt < 1 || nt > kMaxT || n % 64 || n > kMaxN) std::abort();
    const int nb = n / 32;
    prepare(act, nt, nb);
    switch (nt) {
        case 1: rows_t<1>(w, row_bytes, nb, out, r0, r1); break;
        case 2: rows_t<2>(w, row_bytes, nb, out, r0, r1); break;
        case 3: rows_t<3>(w, row_bytes, nb, out, r0, r1); break;
        case 4: rows_t<4>(w, row_bytes, nb, out, r0, r1); break;
        case 5: rows_t<5>(w, row_bytes, nb, out, r0, r1); break;
        case 6: rows_t<6>(w, row_bytes, nb, out, r0, r1); break;
        case 7: rows_t<7>(w, row_bytes, nb, out, r0, r1); break;
        default: rows_t<8>(w, row_bytes, nb, out, r0, r1); break;
    }
}

void mxfp4_rows2(const uint8_t* gate, const uint8_t* up, size_t row_bytes, int n, const void* const* act, int nt,
                 float* const* g, float* const* u, int r0, int r1) {
    if (nt < 1 || nt > kMaxT || n % 64 || n > kMaxN) std::abort();
    const int nb = n / 32;
    prepare(act, nt, nb);
    switch (nt) {
        case 1: rows2_t<1>(gate, up, row_bytes, nb, g, u, r0, r1); break;
        case 2: rows2_t<2>(gate, up, row_bytes, nb, g, u, r0, r1); break;
        case 3: rows2_t<3>(gate, up, row_bytes, nb, g, u, r0, r1); break;
        case 4: rows2_t<4>(gate, up, row_bytes, nb, g, u, r0, r1); break;
        case 5: rows2_t<5>(gate, up, row_bytes, nb, g, u, r0, r1); break;
        case 6: rows2_t<6>(gate, up, row_bytes, nb, g, u, r0, r1); break;
        case 7: rows2_t<7>(gate, up, row_bytes, nb, g, u, r0, r1); break;
        default: rows2_t<8>(gate, up, row_bytes, nb, g, u, r0, r1); break;
    }
}

}  // namespace strata::kernels::cpu
