// src/mimo/kernels/expert_mxfp4.cu - see include/strata/mimo/expert_mxfp4.hpp.
//
// Three kernels per call; the two big ones are bandwidth-bound GEMVs over the slots' bytes:
//   prepare  grid T: each token's x to q8_1 (ggml's quantize_q8_1), in the layout gate_up's lanes read.
//   gate_up  grid (kMxFF / 32, T*K): block (j, e) = 32 hidden rows of entry e's expert, if e is the first entry
//            of the window with that slot (the others return at once).  It serves every entry with that slot (the
//            "members": one per token that picked the expert): each weight row is read and decoded once and applied
//            to all of them.  The block's 32 SiLU(gate) * up values per member are exactly one q8_1 block of h,
//            quantized in place.
//   down     grid (kMxEmbd / kDownRows): block = kDownRows output rows; its warps take the window's distinct
//            experts in turn, sum weight * down(h) per token in shared memory, and add the warps' sums to `out` in
//            a fixed order (no atomics: the same input gives the same bits).
// gate_up and down use programmatic dependent launch: gate_up finds its expert and has its first weight row in
// flight before prepare's x is ready; down builds the window's grouping and loads its first weight rows while
// gate_up finishes.  Everything they read before that (slots, weights, slot table) was written before the call.
//
// The dot products are llama.cpp's vec_dot_mxfp4_q8_1 (ggml/src/ggml-cuda/vecdotq.cuh, MIT, see
// third_party/ggml/LICENSE): kvalues_mxfp4 looked up with byte_perm, dp4a against the q8_1 activation, scale
// E8M0 * 0.5 * d8.  The q8_1 quantizer is ggml's quantize_q8_1 (scale amax / 127, rounded to fp16 for the dot).
#include "strata/mimo/expert_mxfp4.hpp"
#include "strata/kernels/dp4a.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::mimo {
namespace {

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}

constexpr int kGuBlocks = kMxEmbd / 32;      // 128 MXFP4 blocks per gate/up row: 4 per lane
constexpr int kDownBlocks = kMxFF / 32;      // 64 per down row: 2 per lane
constexpr size_t kGuRow = kGuBlocks * 16, kDownRow = kDownBlocks * 16;   // code bytes per row
constexpr size_t kOffUp = kMxGuCodes, kOffDown = 2 * kMxGuCodes;
constexpr size_t kOffGateE = 2 * kMxGuCodes + kMxDownCodes, kOffUpE = kOffGateE + kMxGuScales,
                 kOffDownE = kOffUpE + kMxGuScales;
constexpr int kMaxEntries = kMxMaxT * kMxMaxK;   // 64
constexpr int kThreads = 256, kWarps = kThreads / 32;
constexpr int kGuRows = 32;                      // hidden rows per gate_up block: one q8_1 block of h
constexpr int kGuRowsPerWarp = kGuRows / kWarps;
constexpr int kDownRows = 2;                     // output rows per down block
// resident blocks per SM: caps the registers at 80, 24 warps per SM (more registers made it slower: fewer
// warps, and the compiler spent them on decoding ahead rather than on more loads in flight)
constexpr int kMinBlocks = 3;
static_assert(kMxEmbd % (32 * 32) == 0 && kMxFF % (32 * 32) == 0 && kGuRowsPerWarp * kWarps == kGuRows, "shape");

// ---------------------------------------------------------------- llama.cpp helpers (vecdotq.cuh, common.cuh)
// kvalues_mxfp4 = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12} as four little-endian words
__device__ __forceinline__ int2 mxfp4_lookup(const int q4) {
    constexpr uint32_t t0 = 0x03020100u, t1 = 0x0C080604u, t2 = 0xFDFEFF00u, t3 = 0xF4F8FAFCu;
    uint32_t tmp[2];
    const uint32_t sel = 0x32103210u | ((q4 & 0x88888888u) >> 1);
#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t shift = 16 * i;
        const uint32_t low = __byte_perm(t0, t1, q4 >> shift);
        const uint32_t high = __byte_perm(t2, t3, q4 >> shift);
        tmp[i] = __byte_perm(low, high, sel >> shift);
    }
    return make_int2(__byte_perm(tmp[0], tmp[1], 0x6420), __byte_perm(tmp[0], tmp[1], 0x7531));
}

// ggml_cuda_e8m0_to_fp32(e) * 0.5f, exact: 2^(e - 128)
__device__ __forceinline__ float e8m0_half(const uint32_t e) {
    const uint32_t bits = e >= 2 ? (e - 1) << 23 : (e == 1 ? 0x00400000u : 0x00200000u);
    return __uint_as_float(bits);
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// weights are read once: keep them out of L1
__device__ __forceinline__ uint4 ld_stream(const void* p) {
    uint4 r;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
    return r;
}
__device__ __forceinline__ uint32_t ld_stream_u32(const void* p) {
    uint32_t r;
    asm volatile("ld.global.nc.L1::no_allocate.u32 %0, [%1];" : "=r"(r) : "l"(p));
    return r;
}

// one MXFP4 block (16 code bytes, already looked up: v[l] = values of int l, .x low nibbles, .y high) against a
// q8_1 block's 8 ints: vec_dot_mxfp4_q8_1's integer part
__device__ __forceinline__ int dot_block(const int2 (&v)[4], const int4 a, const int4 b) {
    int s = 0;
    s = STRATA_DP4A(v[0].x, a.x, s); s = STRATA_DP4A(v[0].y, b.x, s);
    s = STRATA_DP4A(v[1].x, a.y, s); s = STRATA_DP4A(v[1].y, b.y, s);
    s = STRATA_DP4A(v[2].x, a.z, s); s = STRATA_DP4A(v[2].y, b.z, s);
    s = STRATA_DP4A(v[3].x, a.w, s); s = STRATA_DP4A(v[3].y, b.w, s);
    return s;
}
__device__ __forceinline__ void lookup4(const uint4 q, int2 (&v)[4]) {
    v[0] = mxfp4_lookup((int) q.x); v[1] = mxfp4_lookup((int) q.y);
    v[2] = mxfp4_lookup((int) q.z); v[3] = mxfp4_lookup((int) q.w);
}

// Programmatic dependent launch (sm_90+): a kernel launched with the attribute may start while the one before it
// still runs; dep_wait() returns once that one has finished and its writes are visible, dep_release() lets the next
// one start.  Without the attribute (or below sm_90) both are no-ops and the launches keep plain stream order.
__device__ __forceinline__ void dep_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    cudaGridDependencySynchronize();
#endif
}
__device__ __forceinline__ void dep_release() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

// The window's entries with slot S, as a 64-bit mask over entry indices (entry = t * K + k).  Every lane gets it.
__device__ __forceinline__ unsigned long long same_slot(const int32_t* __restrict__ slots, int n, int S) {
    const int lane = threadIdx.x & 31;
    const int v0 = lane < n ? slots[lane] : -1, v1 = lane + 32 < n ? slots[lane + 32] : -1;
    return (unsigned long long) __ballot_sync(0xffffffffu, v0 == S) |
           (unsigned long long) __ballot_sync(0xffffffffu, v1 == S) << 32;
}

// A token's repeat of a slot is ignored (a precondition violation): keep each token's first entry of `mask`, so an
// expert has at most one member per token.
__device__ __forceinline__ unsigned long long first_per_token(unsigned long long mask, int T, int K) {
    unsigned long long r = 0;
    const unsigned long long tok = (1ull << K) - 1;
    for (int t = 0; t < T; ++t) {
        const unsigned long long b = mask & (tok << (t * K));
        r |= b & (~b + 1);   // lowest set bit
    }
    return r;
}

// ---------------------------------------------------------------- prepare: x to q8_1
// x of one token as q8_1, laid out for gate_up's lanes: int4 q[k][half][lane] = ints 4*half .. 4*half+3 of block
// lane + 32k (a lane reads its 4 blocks with 128-bit loads), and each block's scale, fp16-rounded as the dot uses it.
struct XQ {
    int4 q[4][2][32];
    float d[kGuBlocks];
};

// block t: token t (thread = 4 consecutive values, 8 threads = one 32-block; value 4c+i of a block is byte i of
// its int c)
__global__ void __launch_bounds__(kThreads) prepare_kernel(const float* __restrict__ x, XQ* __restrict__ xq) {
    dep_release();   // gate_up may start finding its expert
    const float4* xt = (const float4*) (x + (size_t) blockIdx.x * kMxEmbd);
    XQ& o = xq[blockIdx.x];
    float4 v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = xt[j * kThreads + threadIdx.x];
#pragma unroll
    for (int j = 0; j < 4; ++j) {   // block j*32 + tid/8, int tid%8 of it
        float amax = fmaxf(fmaxf(fabsf(v[j].x), fabsf(v[j].y)), fmaxf(fabsf(v[j].z), fabsf(v[j].w)));
#pragma unroll
        for (int off = 4; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
        const float d = amax / 127.0f;
        int q = 0;
        if (amax != 0.0f)
            q = (int) (uint8_t) (int8_t) roundf(v[j].x / d) | (int) (uint8_t) (int8_t) roundf(v[j].y / d) << 8 |
                (int) (uint8_t) (int8_t) roundf(v[j].z / d) << 16 | (int) (uint8_t) (int8_t) roundf(v[j].w / d) << 24;
        const int c = threadIdx.x % 8, bl = threadIdx.x / 8;
        (&o.q[j][c / 4][bl].x)[c % 4] = q;
        if (c == 0) o.d[j * 32 + bl] = __half2float(__float2half_rn(d));
    }
}

// ---------------------------------------------------------------- gate + up, SiLU, q8_1 of h
// MAXM >= T bounds the members (one per token), so their accumulators stay in registers.
template<int MAXM>
__global__ void __launch_bounds__(kThreads, kMinBlocks) gate_up_kernel(const unsigned long long* __restrict__ slot_ptr,
                                                                       const int32_t* __restrict__ slots, int T, int K,
                                                                       const XQ* __restrict__ xq,
                                                                       int8_t* __restrict__ hq, float* __restrict__ hd) {
    __shared__ float sh[MAXM][kGuRows];
    __shared__ int s_ent[MAXM];
    const int n = T * K, e = blockIdx.y;
    const int S = slots[e];
    if (S < 0) return;   // uniform: not resident
    const unsigned long long mine = first_per_token(same_slot(slots, n, S), T, K);
    if (__ffsll((long long) mine) - 1 != e) return;   // uniform: another block serves this expert (or a repeat)
    const int nm = __popcll(mine);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const uint8_t* blob = (const uint8_t*) slot_ptr[S];
    const int r0 = blockIdx.x * kGuRows;

    uint4 gq[4], uq[4];
    uint32_t ge, ue;
    auto load_row = [&](int r) {
        const uint4* gc = (const uint4*) (blob + (size_t) r * kGuRow);
        const uint4* uc = (const uint4*) (blob + kOffUp + (size_t) r * kGuRow);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            gq[k] = ld_stream(gc + lane + 32 * k);
            uq[k] = ld_stream(uc + lane + 32 * k);
        }
        ge = ld_stream_u32(blob + kOffGateE + (size_t) r * kGuBlocks + 4 * lane);
        ue = ld_stream_u32(blob + kOffUpE + (size_t) r * kGuBlocks + 4 * lane);
    };
    if (warp == 0) {       // the members' entries, in order
        unsigned long long m = mine;
        for (int i = 0; i < nm; ++i) {
            if (lane == 0) s_ent[i] = __ffsll((long long) m) - 1;
            m &= m - 1;
        }
    }
    dep_wait();   // prepare_kernel's x
    dep_release();
    __syncthreads();

#pragma unroll 1
    for (int i = 0; i < kGuRowsPerWarp; ++i) {
        const int rl = warp + kWarps * i;
        load_row(r0 + rl);
        // block by block, each decoded once and applied to every member
        float ag[MAXM], au[MAXM];
#pragma unroll
        for (int m = 0; m < MAXM; ++m) ag[m] = au[m] = 0.0f;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            int2 vg[4], vu[4];
            lookup4(gq[k], vg);
            lookup4(uq[k], vu);
            const float sg = e8m0_half((ge >> (8 * k)) & 0xff), su = e8m0_half((ue >> (8 * k)) & 0xff);
#pragma unroll
            for (int m = 0; m < MAXM; ++m) {
                if (m >= nm) break;
                const XQ& xm = xq[s_ent[m] / K];
                const int4 a = __ldg(&xm.q[k][0][lane]), b = __ldg(&xm.q[k][1][lane]);
                const float d8 = __ldg(&xm.d[lane + 32 * k]);
                ag[m] += sg * d8 * (float) dot_block(vg, a, b);
                au[m] += su * d8 * (float) dot_block(vu, a, b);
            }
        }
#pragma unroll
        for (int m = 0; m < MAXM; ++m) {
            if (m >= nm) break;
            const float g = warp_sum(ag[m]), u = warp_sum(au[m]);
            if (lane == 0) sh[m][rl] = g / (1.0f + expf(-g)) * u;
        }
    }
    __syncthreads();
    // h of each member, rows r0 .. r0+31: one q8_1 block
    for (int m = warp; m < nm; m += kWarps) {
        const int me = s_ent[m];
        const float v = sh[m][lane];
        const float amax = warp_max(fabsf(v));
        const float d = amax / 127.0f;
        hq[(size_t) me * kMxFF + r0 + lane] = amax == 0.0f ? 0 : (int8_t) roundf(v / d);
        if (lane == 0) hd[(size_t) me * kDownBlocks + blockIdx.x] = __half2float(__float2half_rn(d));
    }
}

// ---------------------------------------------------------------- down, weighted sum into out
// The window's grouping: group g = the g-th distinct resident slot (in entry order); its entries are the bits of
// mask[g].  Built by warp 0, lane = entries lane and lane + 32; `sl` is 64 ints of shared scratch.
struct Groups {
    int n;
    int slot[kMaxEntries];
    unsigned long long mask[kMaxEntries];
};
__device__ void build_groups(const int32_t* __restrict__ slots, int T, int K, Groups& G, int* sl) {
    const int lane = threadIdx.x & 31, n = T * K;
    const int s0 = lane < n ? slots[lane] : -1, s1 = lane + 32 < n ? slots[lane + 32] : -1;
    sl[lane] = s0;
    sl[lane + 32] = s1;
    __syncwarp();
    unsigned long long m0 = 0, m1 = 0;   // entries with this lane's slots
#pragma unroll
    for (int j = 0; j < kMaxEntries; ++j) {
        const int v = sl[j];
        m0 |= (unsigned long long) (v == s0) << j;
        m1 |= (unsigned long long) (v == s1) << j;
    }
    // an entry starts a group when no earlier entry has its slot
    const bool f0 = s0 >= 0 && __ffsll((long long) m0) - 1 == lane;
    const bool f1 = s1 >= 0 && __ffsll((long long) m1) - 1 == lane + 32;
    const unsigned fa = __ballot_sync(0xffffffffu, f0), fb = __ballot_sync(0xffffffffu, f1);
    if (f0) {
        const int g = __popc(fa & ((1u << lane) - 1));
        G.slot[g] = s0;
        G.mask[g] = first_per_token(m0, T, K);
    }
    if (f1) {
        const int g = __popc(fa) + __popc(fb & ((1u << lane) - 1));
        G.slot[g] = s1;
        G.mask[g] = first_per_token(m1, T, K);
    }
    if (lane == 0) G.n = __popc(fa) + __popc(fb);
}

__global__ void __launch_bounds__(kThreads, kMinBlocks) down_kernel(const unsigned long long* __restrict__ slot_ptr,
                                                                    const int32_t* __restrict__ slots,
                                                                    const float* __restrict__ weights, int T, int K,
                                                                    const int8_t* __restrict__ hq,
                                                                    const float* __restrict__ hd,
                                                                    float* __restrict__ out) {
    __shared__ Groups G;
    __shared__ int sl[kMaxEntries];
    __shared__ float acc[kWarps][kMxMaxT][kDownRows];
    for (int i = threadIdx.x; i < kWarps * kMxMaxT * kDownRows; i += kThreads) (&acc[0][0][0])[i] = 0.0f;
    if (threadIdx.x < 32) build_groups(slots, T, K, G, sl);
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r0 = blockIdx.x * kDownRows;
    uint4 c[kDownRows][2];
    uint32_t se[kDownRows];
    auto load = [&](int g) {
        const uint8_t* blob = (const uint8_t*) slot_ptr[G.slot[g]];
#pragma unroll
        for (int i = 0; i < kDownRows; ++i) {
            const uint4* dc = (const uint4*) (blob + kOffDown + (size_t) (r0 + i) * kDownRow);
            c[i][0] = ld_stream(dc + lane);
            c[i][1] = ld_stream(dc + lane + 32);
            se[i] = *(const uint16_t*) (blob + kOffDownE + (size_t) (r0 + i) * kDownBlocks + 2 * lane);
        }
    };
    if (warp < G.n) load(warp);   // in flight while gate_up_kernel finishes
    dep_wait();                   // gate_up_kernel's h
    for (int g = warp; g < G.n; g += kWarps) {
        if (g != warp) load(g);
        for (unsigned long long mk = G.mask[g]; mk; mk &= mk - 1) {
            const int e = __ffsll((long long) mk) - 1;
            const int4* hr = (const int4*) (hq + (size_t) e * kMxFF);
            const int4 a0 = hr[2 * lane], b0 = hr[2 * lane + 1], a1 = hr[2 * (lane + 32)], b1 = hr[2 * (lane + 32) + 1];
            const float d0 = hd[(size_t) e * kDownBlocks + lane], d1 = hd[(size_t) e * kDownBlocks + lane + 32];
            float sum[kDownRows];
#pragma unroll
            for (int i = 0; i < kDownRows; ++i) {
                int2 v[4];
                lookup4(c[i][0], v);
                sum[i] = e8m0_half(se[i] & 0xff) * d0 * (float) dot_block(v, a0, b0);
                lookup4(c[i][1], v);
                sum[i] += e8m0_half(se[i] >> 8) * d1 * (float) dot_block(v, a1, b1);
            }
#pragma unroll
            for (int i = 0; i < kDownRows; ++i) sum[i] = warp_sum(sum[i]);
            if (lane == 0) {
                const float w = weights[e];
                const int t = e / K;
#pragma unroll
                for (int i = 0; i < kDownRows; ++i) acc[warp][t][i] += w * sum[i];
            }
        }
    }
    __syncthreads();
    if (threadIdx.x < T * kDownRows) {
        const int t = threadIdx.x / kDownRows, i = threadIdx.x % kDownRows;
        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < kWarps; ++w) sum += acc[w][t][i];
        out[(size_t) t * kMxEmbd + r0 + i] += sum;
    }
}

// ---------------------------------------------------------------- slot format
// byte offset of block b's scale inside its row's scale bytes (see the header)
__host__ __device__ inline int gu_scale_pos(int b) { return 4 * (b % 32) + b / 32; }
__host__ __device__ inline int down_scale_pos(int b) { return 2 * (b % 32) + b / 32; }

// one thread per MXFP4 block of the expert: blocks [0, gu) gate, [gu, 2gu) up, then down
__global__ void relayout_kernel(const uint8_t* __restrict__ raw, uint8_t* __restrict__ slot) {
    constexpr long long gu = (long long) kMxFF * kGuBlocks, dn = (long long) kMxEmbd * kDownBlocks;
    const long long n = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= 2 * gu + dn) return;
    const uint8_t* src = raw + n * 17;
    uint8_t *codes, *scale;
    if (n < 2 * gu) {
        const int m = (int) (n / gu);
        const long long nn = n - m * gu;
        const int r = (int) (nn / kGuBlocks), b = (int) (nn % kGuBlocks);
        codes = slot + (m ? kOffUp : 0) + (size_t) r * kGuRow + b * 16;
        scale = slot + (m ? kOffUpE : kOffGateE) + (size_t) r * kGuBlocks + gu_scale_pos(b);
    } else {
        const long long nn = n - 2 * gu;
        const int r = (int) (nn / kDownBlocks), b = (int) (nn % kDownBlocks);
        codes = slot + kOffDown + (size_t) r * kDownRow + b * 16;
        scale = slot + kOffDownE + (size_t) r * kDownBlocks + down_scale_pos(b);
    }
    *scale = src[0];
    uint4 q;
    uint8_t* qb = (uint8_t*) &q;
#pragma unroll
    for (int j = 0; j < 16; ++j) qb[j] = src[1 + j];
    *(uint4*) codes = q;
}

constexpr size_t align256(size_t n) { return (n + 255) & ~(size_t) 255; }

// [x as q8_1 per token | h codes per entry | h scales per entry]
struct Scratch {
    XQ* xq;
    int8_t* hq;
    float* hd;
    size_t bytes;
    Scratch(void* base, int T, int K) {
        const size_t e = (size_t) T * K;
        uint8_t* p = (uint8_t*) base;
        xq = (XQ*) p;
        p += align256((size_t) T * sizeof(XQ));
        hq = (int8_t*) p;
        p += align256(e * kMxFF);
        hd = (float*) p;
        p += align256(e * kDownBlocks * sizeof(float));
        bytes = (size_t) (p - (uint8_t*) base);
    }
};

template<typename... Params, typename... Args>
void launch_pdl(void (*kernel)(Params...), dim3 grid, cudaStream_t s, Args... args) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = dim3(kThreads);
    cfg.stream = s;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    const cudaError_t e = cudaLaunchKernelEx(&cfg, kernel, ((Params) args)...);
    if (e != cudaSuccess) { std::fprintf(stderr, "mxfp4_experts launch: %s\n", cudaGetErrorString(e)); std::exit(1); }
}

}  // namespace

void mxfp4_slot_from_gguf(const uint8_t* gate, const uint8_t* up, const uint8_t* down, uint8_t* slot) {
    for (int m = 0; m < 2; ++m) {
        const uint8_t* src = m ? up : gate;
        uint8_t* codes = slot + (m ? kOffUp : 0);
        uint8_t* scales = slot + (m ? kOffUpE : kOffGateE);
        for (int r = 0; r < kMxFF; ++r)
            for (int b = 0; b < kGuBlocks; ++b) {
                const uint8_t* blk = src + ((size_t) r * kGuBlocks + b) * 17;
                scales[(size_t) r * kGuBlocks + gu_scale_pos(b)] = blk[0];
                std::memcpy(codes + (size_t) r * kGuRow + b * 16, blk + 1, 16);
            }
    }
    for (int r = 0; r < kMxEmbd; ++r)
        for (int b = 0; b < kDownBlocks; ++b) {
            const uint8_t* blk = down + ((size_t) r * kDownBlocks + b) * 17;
            slot[kOffDownE + (size_t) r * kDownBlocks + down_scale_pos(b)] = blk[0];
            std::memcpy(slot + kOffDown + (size_t) r * kDownRow + b * 16, blk + 1, 16);
        }
}

void mxfp4_slot_relayout(const void* raw, void* slot, void* stream) {
    constexpr long long n = 2LL * kMxFF * kGuBlocks + (long long) kMxEmbd * kDownBlocks;
    relayout_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, (cudaStream_t) stream>>>((const uint8_t*) raw,
                                                                                    (uint8_t*) slot);
    check("mxfp4_slot_relayout");
}

size_t mxfp4_experts_scratch_bytes(int T, int K) { return Scratch(nullptr, T, K).bytes; }

void mxfp4_experts(const unsigned long long* slot_ptr, const int32_t* slots, const float* weights, const float* x,
                   float* out, int T, int K, void* scratch, void* stream) {
    if (T < 1 || T > kMxMaxT || K < 1 || K > kMxMaxK) {
        std::fprintf(stderr, "mxfp4_experts: T %d K %d out of range\n", T, K);
        std::exit(1);
    }
    cudaStream_t s = (cudaStream_t) stream;
    const Scratch sc(scratch, T, K);
    prepare_kernel<<<T, kThreads, 0, s>>>(x, sc.xq);
    check("mxfp4_experts/prepare");
    const dim3 ggu(kMxFF / kGuRows, (unsigned) (T * K));
    if (T == 1) launch_pdl(gate_up_kernel<1>, ggu, s, slot_ptr, slots, T, K, sc.xq, sc.hq, sc.hd);
    else if (T == 2) launch_pdl(gate_up_kernel<2>, ggu, s, slot_ptr, slots, T, K, sc.xq, sc.hq, sc.hd);
    else if (T <= 4) launch_pdl(gate_up_kernel<4>, ggu, s, slot_ptr, slots, T, K, sc.xq, sc.hq, sc.hd);
    else launch_pdl(gate_up_kernel<8>, ggu, s, slot_ptr, slots, T, K, sc.xq, sc.hq, sc.hd);
    launch_pdl(down_kernel, dim3(kMxEmbd / kDownRows), s, slot_ptr, slots, weights, T, K, sc.hq, sc.hd, out);
}

}  // namespace strata::mimo
