// src/mimo/prefill_moe.cu - see include/strata/mimo/prefill_moe.hpp.
//
// Kernels here: the pick histogram, the stable placement of the picks in processing order, the two MXFP4 -> Q8_0
// converters (from the GGUF layout and from the decoder's slot layout) and the weighted, ordered add into `out`.  The products are
// llama.cpp's MMQ through strata::prefill::mmq (src/prefill/moe_mmq.cu, the Q8_0 instance).
#include "strata/mimo/prefill_moe.hpp"

#include "strata/mimo/expert_mxfp4.hpp"
#include "strata/prefill/moe_mmq.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace strata::mimo {
namespace {

namespace mmq = strata::prefill::mmq;

constexpr int kTypeQ8 = 8;                    // GGML_TYPE_Q8_0
constexpr int kE = kMxEmbd, kF = kMxFF;       // 4096, 2048
constexpr int kNExpert = 256;
constexpr int kGuBlocks = kE / 32, kDownBlocks = kF / 32;   // MXFP4 blocks per gate/up row, per down row
constexpr long long kGuMxBlocks = (long long) kF * kGuBlocks;          // blocks of one gate (or up) matrix
constexpr long long kDownMxBlocks = (long long) kE * kDownBlocks;
constexpr long long kAllMxBlocks = 2 * kGuMxBlocks + kDownMxBlocks;
// the slot layout (expert_mxfp4.hpp)
constexpr size_t kOffUp = kMxGuCodes, kOffDown = 2 * kMxGuCodes;
constexpr size_t kOffGateE = 2 * kMxGuCodes + kMxDownCodes, kOffUpE = kOffGateE + kMxGuScales,
                 kOffDownE = kOffUpE + kMxGuScales;

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "prefill moe: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

unsigned blocks_for(long long n, int threads) { return (unsigned) ((n + threads - 1) / threads); }

// ---------------------------------------------------------------- MXFP4 -> Q8_0
// kvalues_mxfp4 (ggml-common.h); a GGUF block is {e8m0 e; 16 bytes: value j in the low nibble of byte j, value
// j + 16 in the high nibble}.  Value = kv * 2^(e - 128) (ggml's E8M0 "half" scale).  Q8_0 = {fp16 d; int8 qs[32]}.
__device__ __forceinline__ int kv_of(int nib) {
    const int m = (int) ((0x0C08060403020100ull >> (8 * (nib & 7))) & 0xff);
    return nib & 8 ? -m : m;
}

// fp16 bits of 2^(e - 128) and how far the codes have to be shifted when that is out of fp16's range
__device__ __forceinline__ uint16_t q8_scale(int e, int& shift) {
    const int x = e - 128;
    shift = 0;
    if (x > 15) { shift = x - 15; return (uint16_t) (30 << 10); }   // 2^15; codes doubled (clamped)
    if (x >= -14) return (uint16_t) ((x + 15) << 10);                // normal
    if (x >= -24) return (uint16_t) (1u << (x + 24));                // subnormal power of two: exact
    shift = x + 24;                                                  // < 0: below fp16, codes halved (rounded)
    return 1;
}

__device__ __forceinline__ int8_t q8_code(int kv, int shift) {
    if (shift == 0) return (int8_t) kv;
    if (shift > 0) return (int8_t) max(-127, min(127, kv << min(shift, 7)));
    const float v = (float) kv * exp2f((float) shift);
    return (int8_t) __float2int_rn(v);
}

// kvalues_mxfp4 of 4 code bytes at once (llama.cpp's get_int_from_table_16 for MXFP4, as in expert_mxfp4.cu):
// .x = the 4 low-nibble values, .y = the 4 high-nibble values, as int8 lanes
__device__ __forceinline__ int2 kv4(const uint32_t q4) {
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
    return make_int2((int) __byte_perm(tmp[0], tmp[1], 0x6420), (int) __byte_perm(tmp[0], tmp[1], 0x7531));
}

__device__ __forceinline__ uint32_t rescale4(uint32_t v, int shift) {
    uint32_t r = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) r |= (uint32_t) (uint8_t) q8_code((int8_t) (v >> (8 * i)), shift) << (8 * i);
    return r;
}

// One Q8_0 block (34 bytes at `o`, 2-byte aligned: shared memory) from an E8M0 scale and the 16 code bytes
__device__ __forceinline__ void put_q8(uint8_t* o, int e, const uint4 c) {
    int shift;
    const uint16_t d = q8_scale(e, shift);
    const uint32_t w[4] = {c.x, c.y, c.z, c.w};
    uint32_t lo[4], hi[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int2 v = kv4(w[i]);
        lo[i] = (uint32_t) v.x;
        hi[i] = (uint32_t) v.y;
    }
    if (shift != 0) {   // never in MiMo's experts (docs/MIMO_PREFILL_MOE.md); kept correct-ish anyway
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            lo[i] = rescale4(lo[i], shift);
            hi[i] = rescale4(hi[i], shift);
        }
    }
    uint16_t* o16 = (uint16_t*) o;
    o16[0] = d;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        o16[1 + 2 * i] = (uint16_t) lo[i];
        o16[2 + 2 * i] = (uint16_t) (lo[i] >> 16);
        o16[9 + 2 * i] = (uint16_t) hi[i];
        o16[10 + 2 * i] = (uint16_t) (hi[i] >> 16);
    }
}

// The converters: one CUDA block = 256 MXFP4 blocks (never across the gate|up / down boundary, a multiple of 256),
// their Q8_0 images assembled in shared memory (8,704 bytes, 16-byte aligned in the output) and stored with
// coalesced 128-bit stores; the GGUF input is read the same way (4,352 bytes per CUDA block).
constexpr int kCvt = 256;
static_assert(kAllMxBlocks % kCvt == 0 && (2 * kGuMxBlocks) % kCvt == 0, "converter blocks");

__device__ __forceinline__ void store_q8(const uint4* out_s, long long n0, uint8_t* gu, uint8_t* dn) {
    uint4* dst = n0 < 2 * kGuMxBlocks ? (uint4*) (gu + n0 * 34) : (uint4*) (dn + (n0 - 2 * kGuMxBlocks) * 34);
    for (int i = threadIdx.x; i < kCvt * 34 / 16; i += kCvt) dst[i] = out_s[i];
}

__global__ void __launch_bounds__(kCvt) raw_to_q8_kernel(const uint8_t* __restrict__ raw, uint8_t* __restrict__ gu,
                                                         uint8_t* __restrict__ dn) {
    __shared__ uint4 in_s[kCvt * 17 / 16];
    __shared__ uint4 out_s[kCvt * 34 / 16];
    const long long n0 = (long long) blockIdx.x * kCvt;
    const uint4* src = (const uint4*) (raw + n0 * 17);
    for (int i = threadIdx.x; i < kCvt * 17 / 16; i += kCvt) in_s[i] = src[i];
    __syncthreads();
    const uint8_t* b = (const uint8_t*) in_s + threadIdx.x * 17;
    uint32_t w[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
        w[i] = (uint32_t) b[1 + 4 * i] | (uint32_t) b[2 + 4 * i] << 8 | (uint32_t) b[3 + 4 * i] << 16 |
               (uint32_t) b[4 + 4 * i] << 24;
    put_q8((uint8_t*) out_s + threadIdx.x * 34, b[0], make_uint4(w[0], w[1], w[2], w[3]));
    __syncthreads();
    store_q8(out_s, n0, gu, dn);
}

__global__ void __launch_bounds__(kCvt) slot_to_q8_kernel(const uint8_t* __restrict__ slot, uint8_t* __restrict__ gu,
                                                          uint8_t* __restrict__ dn) {
    __shared__ uint4 out_s[kCvt * 34 / 16];
    const long long n0 = (long long) blockIdx.x * kCvt, n = n0 + threadIdx.x;
    const uint8_t *codes, *scale;
    if (n < 2 * kGuMxBlocks) {
        const int m = (int) (n / kGuMxBlocks);
        const long long nn = n - m * kGuMxBlocks;
        const int r = (int) (nn / kGuBlocks), b = (int) (nn % kGuBlocks);
        codes = slot + (m ? kOffUp : 0) + (size_t) r * (kGuBlocks * 16) + b * 16;
        scale = slot + (m ? kOffUpE : kOffGateE) + (size_t) r * kGuBlocks + 4 * (b % 32) + b / 32;
    } else {
        const long long nn = n - 2 * kGuMxBlocks;
        const int r = (int) (nn / kDownBlocks), b = (int) (nn % kDownBlocks);
        codes = slot + kOffDown + (size_t) r * (kDownBlocks * 16) + b * 16;
        scale = slot + kOffDownE + (size_t) r * kDownBlocks + 2 * (b % 32) + b / 32;
    }
    put_q8((uint8_t*) out_s + threadIdx.x * 34, *scale, *(const uint4*) codes);
    __syncthreads();
    store_q8(out_s, n0, gu, dn);
}

// ---------------------------------------------------------------- grouping
// counts[e] += picks of expert e (counts zeroed before).  The counts do not depend on the order of the atomics.
__global__ void hist_kernel(const int32_t* __restrict__ ids, int n, int32_t* __restrict__ counts) {
    __shared__ int h[kNExpert];
    for (int i = threadIdx.x; i < kNExpert; i += blockDim.x) h[i] = 0;
    __syncthreads();
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const int e = ids[i];
        if (e >= 0 && e < kNExpert) atomicAdd(&h[e], 1);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < kNExpert; i += blockDim.x)
        if (h[i]) atomicAdd(&counts[i], h[i]);
}

// Block e: expert e's picks, in pick order, to rows bounds[rank[e]] ...: row_tok[row] = the pick's token, row_w[row]
// = its router weight.  plan = rank[256] | bounds[...].
constexpr int kPlaceThreads = 1024;
__global__ void __launch_bounds__(kPlaceThreads) place_kernel(const int32_t* __restrict__ ids, const float* __restrict__ w,
                                                              int n, int k, const int32_t* __restrict__ plan,
                                                              int32_t* __restrict__ row_tok, float* __restrict__ row_w) {
    const int e = blockIdx.x;
    const int r = plan[e];
    if (r < 0) return;
    const int base = plan[kNExpert + r];
    constexpr int kWarps = kPlaceThreads / 32;
    __shared__ int warp_n[kWarps];
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    int running = 0;
    for (int i0 = 0; i0 < n; i0 += kPlaceThreads) {
        const int i = i0 + threadIdx.x;
        const bool f = i < n && ids[i] == e;
        const unsigned m = __ballot_sync(0xffffffffu, f);
        if (lane == 0) warp_n[wid] = __popc(m);
        __syncthreads();
        int before = 0, total = 0;
#pragma unroll
        for (int q = 0; q < kWarps; ++q) {
            before += q < wid ? warp_n[q] : 0;
            total += warp_n[q];
        }
        if (f) {
            const int row = base + running + before + __popc(m & ((1u << lane) - 1));
            row_tok[row] = i / k;
            row_w[row] = w[i];
        }
        running += total;
        __syncthreads();
    }
}

// out[row_tok[i]] += row_w[i] * rows[i] for the rows of ONE expert (its tokens are distinct: no two blocks write the
// same output row).  Launched per expert in the processing order, so every output row sums in a fixed order.
__global__ void add_rows_kernel(const float* __restrict__ rows, const int32_t* __restrict__ row_tok,
                                const float* __restrict__ row_w, float* __restrict__ out) {
    const int i = blockIdx.x;
    const float wt = row_w[i];
    const float4* r = (const float4*) (rows + (size_t) i * kE);
    float4* o = (float4*) (out + (size_t) row_tok[i] * kE);
#pragma unroll
    for (int c = 0; c < kE / (4 * 256); ++c) {
        const float4 v = r[c * 256 + threadIdx.x];
        float4 a = o[c * 256 + threadIdx.x];
        a.x += wt * v.x;
        a.y += wt * v.y;
        a.z += wt * v.z;
        a.w += wt * v.w;
        o[c * 256 + threadIdx.x] = a;
    }
}

size_t align_up(size_t n, size_t a = 4096) { return (n + a - 1) / a * a; }

int max_tiles(const PrefillMoeConfig& c) {   // groups (<= 256 / smallest group + 2) + tile boundaries inside them
    return kNExpert / std::max(1, std::min(c.group, c.group_ram)) + 2 +
           (int) ((long long) c.max_tokens * c.k / std::max(1, c.tile_rows)) + 1;
}
size_t plan_ints(const PrefillMoeConfig& c) { return (size_t) (2 * kNExpert + 1) + kNExpert + 2 * (size_t) max_tiles(c); }

struct Layout {
    size_t staging, gu_q8, d_q8, xq, hq, gu, h, iota, counts, row_tok, row_w, plan, total;
    Layout(const PrefillMoeConfig& c) {
        const size_t rows = (size_t) c.max_tokens * c.k, tile = (size_t) c.tile_rows;
        const size_t g = (size_t) std::max(c.group, c.group_ram);
        size_t o = 0;
        auto take = [&](size_t bytes) { const size_t at = o; o += align_up(bytes); return at; };
        staging = take((size_t) c.staging * kMxSlotBytes);
        gu_q8 = take(g * kPfGuQ8Bytes);
        d_q8 = take(g * kPfDownQ8Bytes);
        xq = take(mmq::q8_bytes((int64_t) tile, kE));     // per tile
        hq = take(mmq::q8_bytes((int64_t) tile, kF));
        gu = take(tile * kE * sizeof(float));
        h = take(tile * kF * sizeof(float));
        iota = take(tile * 4);
        counts = take(kNExpert * 4);
        row_tok = take(rows * 4);                         // per pick: 8 bytes
        row_w = take(rows * 4);
        plan = take(plan_ints(c) * 4);
        total = o;
    }
};

}  // namespace

void mxfp4_raw_to_q8_0(const void* raw, void* gu_q8, void* d_q8, cudaStream_t stream) {
    raw_to_q8_kernel<<<(unsigned) (kAllMxBlocks / kCvt), kCvt, 0, stream>>>((const uint8_t*) raw, (uint8_t*) gu_q8,
                                                                         (uint8_t*) d_q8);
    check("raw_to_q8");
}

void mxfp4_slot_to_q8_0(const void* slot, void* gu_q8, void* d_q8, cudaStream_t stream) {
    slot_to_q8_kernel<<<(unsigned) (kAllMxBlocks / kCvt), kCvt, 0, stream>>>((const uint8_t*) slot, (uint8_t*) gu_q8,
                                                                          (uint8_t*) d_q8);
    check("slot_to_q8");
}

PrefillMoe::PrefillMoe(const PrefillMoeConfig& cfg) : cfg_(cfg) {}

PrefillMoe::~PrefillMoe() {
    if (copy_) cudaStreamSynchronize(copy_);
    for (cudaEvent_t e : {ev_start_, ev_counts_, ev_plan_, ev_copy0_, ev_copy1_, ev_end_})
        if (e) cudaEventDestroy(e);
    for (int i = 0; arrived_ && i < cfg_.staging; ++i) {
        cudaEventDestroy(arrived_[i]);
        cudaEventDestroy(freed_[i]);
    }
    delete[] arrived_;
    delete[] freed_;
    if (copy_) cudaStreamDestroy(copy_);
    if (dev_) cudaFree(dev_);
    if (h_counts_) cudaFreeHost(h_counts_);
    if (h_plan_) cudaFreeHost(h_plan_);
    delete (mmq::Context*) mmq_;
}

size_t PrefillMoe::device_bytes(const PrefillMoeConfig& c) { return Layout(c).total; }

bool PrefillMoe::init(std::string& err) {
    if (cfg_.vram_mib > 0) {   // the largest tile (a multiple of 512 rows) that fits the budget
        PrefillMoeConfig c = cfg_;
        c.tile_rows = 512;
        if (Layout(c).total > ((size_t) cfg_.vram_mib << 20)) {
            err = "prefill moe: " + std::to_string(cfg_.vram_mib) + " MiB is less than the smallest layout (" +
                  std::to_string(Layout(c).total >> 20) + " MiB)";
            return false;
        }
        while (c.tile_rows < cfg_.max_tokens * cfg_.k) {
            PrefillMoeConfig d = c;
            d.tile_rows += 512;
            if (Layout(d).total > ((size_t) cfg_.vram_mib << 20)) break;
            c = d;
        }
        cfg_.tile_rows = c.tile_rows;
    }
    if (cfg_.max_tokens < 1 || cfg_.k < 1 || cfg_.k > 64 || cfg_.group < 1 || cfg_.group_ram < 1 || cfg_.staging < 1 ||
        cfg_.tile_rows < 1) {
        err = "prefill moe: bad config";
        return false;
    }
    if (!mmq::fits(kTypeQ8, kE)) {
        err = "prefill moe: llama.cpp's Q8_0 MMQ has no tile for this GPU";
        return false;
    }
    const Layout L(cfg_);
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    if (free_b < L.total + (256ull << 20)) {
        err = "prefill moe: needs " + std::to_string(L.total >> 20) + " MiB of VRAM, " + std::to_string(free_b >> 20) +
              " MiB free";
        return false;
    }
    if (cudaMalloc(&dev_, L.total) != cudaSuccess) {
        (void) cudaGetLastError();
        err = "prefill moe: cudaMalloc of " + std::to_string(L.total >> 20) + " MiB failed";
        return false;
    }
    dev_bytes_ = L.total;
    staging_ = dev_ + L.staging;
    gu_q8_ = dev_ + L.gu_q8;
    d_q8_ = dev_ + L.d_q8;
    xq_ = dev_ + L.xq;
    hq_ = dev_ + L.hq;
    gu_ = (float*) (dev_ + L.gu);
    h_ = (float*) (dev_ + L.h);
    counts_ = (int32_t*) (dev_ + L.counts);
    row_tok_ = (int32_t*) (dev_ + L.row_tok);
    row_w_ = (float*) (dev_ + L.row_w);
    iota_ = (int32_t*) (dev_ + L.iota);
    plan_ = (int32_t*) (dev_ + L.plan);
    if (cudaMallocHost(&h_counts_, kNExpert * 4) != cudaSuccess || cudaMallocHost(&h_plan_, plan_ints(cfg_) * 4) != cudaSuccess ||
        cudaStreamCreateWithFlags(&copy_, cudaStreamNonBlocking) != cudaSuccess) {
        (void) cudaGetLastError();
        err = "prefill moe: pinned buffers / copy stream";
        return false;
    }
    arrived_ = new cudaEvent_t[(size_t) cfg_.staging];
    freed_ = new cudaEvent_t[(size_t) cfg_.staging];
    for (int i = 0; i < cfg_.staging; ++i) {
        cudaEventCreateWithFlags(&arrived_[i], cudaEventDisableTiming);
        cudaEventCreateWithFlags(&freed_[i], cudaEventDisableTiming);
    }
    for (cudaEvent_t* e : {&ev_start_, &ev_counts_, &ev_plan_, &ev_copy0_, &ev_copy1_, &ev_end_}) cudaEventCreate(e);
    mmq_ = new mmq::Context();
    mmq::iota(iota_, cfg_.tile_rows, nullptr);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        err = std::string("prefill moe: init: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    return true;
}

bool PrefillMoe::register_host(void* p, size_t bytes, std::string& err) {
    const cudaError_t e = cudaHostRegister(p, bytes, cudaHostRegisterPortable);
    if (e == cudaSuccess) return true;
    (void) cudaGetLastError();   // a refused registration is sticky until read
    err = std::string("cudaHostRegister: ") + cudaGetErrorString(e);
    return false;
}

void PrefillMoe::unregister_host(void* p) {
    if (cudaHostUnregister(p) != cudaSuccess) (void) cudaGetLastError();
}

void PrefillMoe::run_layer(const PrefillMoeLayer& L, cudaStream_t s, Mode mode) {
    const int P = L.n_tok, K = cfg_.k, n = P * K;
    if (P < 1 || P > cfg_.max_tokens) {
        std::fprintf(stderr, "prefill moe: %d tokens (1..%d)\n", P, cfg_.max_tokens);
        std::exit(1);
    }
    cudaEventRecord(ev_start_, s);
    // 1. counts -> host
    cudaMemsetAsync(counts_, 0, kNExpert * 4, s);
    hist_kernel<<<std::min(256u, blocks_for(n, 1024)), 256, 0, s>>>(L.ids, n, counts_);
    check("hist");
    cudaMemcpyAsync(h_counts_, counts_, kNExpert * 4, cudaMemcpyDeviceToHost, s);
    cudaEventRecord(ev_counts_, s);
    cudaEventSynchronize(ev_counts_);

    // the plan: resident experts first, then the copied ones, each in expert order; groups of experts (one conversion
    // each); tiles of at most tile_rows rows within a group (an expert's rows may span tiles)
    std::vector<int> order_res, order_ram;
    for (int e = 0; e < kNExpert; ++e) {
        if (h_counts_[e] == 0) continue;
        (L.resident && L.resident[e] >= 0 ? order_res : order_ram).push_back(e);
    }
    std::vector<int> order(order_res);
    order.insert(order.end(), order_ram.begin(), order_ram.end());
    const int n_used = (int) order.size(), n_res = (int) order_res.size();
    int32_t* rank = h_plan_;
    int32_t* bounds = h_plan_ + kNExpert;
    int32_t* relb = bounds + kNExpert + 1;   // per tile: its experts' row ranges relative to the tile's first row
    for (int e = 0; e < kNExpert; ++e) rank[e] = -1;
    bounds[0] = 0;
    for (int i = 0; i < n_used; ++i) {
        rank[order[(size_t) i]] = i;
        bounds[i + 1] = bounds[i] + h_counts_[order[(size_t) i]];
    }
    struct Tile { int a, b, q0, nq, rel, max_rows; };   // rows [a, b); experts order[q0 .. q0 + nq)
    struct Group { int first, n, t0, nt; };            // experts [first, first + n); tiles [t0, t0 + nt)
    std::vector<Group> groups;
    std::vector<Tile> tiles;
    int rel_n = 0;
    for (int part = 0; part < 2; ++part) {
        const int lo = part ? n_res : 0, hi = part ? n_used : n_res, g = part ? cfg_.group_ram : cfg_.group;
        for (int f = lo; f < hi; f += g) {
            Group gr{f, std::min(g, hi - f), (int) tiles.size(), 0};
            int q = f;
            for (int a = bounds[f]; a < bounds[f + gr.n]; a += cfg_.tile_rows) {
                const int b = std::min(a + cfg_.tile_rows, bounds[f + gr.n]);
                while (bounds[q + 1] <= a) ++q;
                Tile t{a, b, q, 0, rel_n, 0};
                for (int j = q; j < f + gr.n && bounds[j] < b; ++j) {
                    const int sa = std::max(bounds[j], a), sb = std::min(bounds[j + 1], b);
                    relb[rel_n++] = sa - a;
                    t.max_rows = std::max(t.max_rows, sb - sa);
                    ++t.nq;
                }
                relb[rel_n++] = b - a;
                tiles.push_back(t);
                ++gr.nt;
            }
            groups.push_back(gr);
        }
    }
    const size_t plan_bytes = (size_t) (2 * kNExpert + 1 + rel_n) * 4;
    cudaMemcpyAsync(plan_, h_plan_, plan_bytes, cudaMemcpyHostToDevice, s);
    // 2. picks in that order
    place_kernel<<<kNExpert, kPlaceThreads, 0, s>>>(L.ids, L.w, n, K, plan_, row_tok_, row_w_);
    check("place");
    if (mode != Mode::copy_only) cudaMemsetAsync(L.out, 0, (size_t) P * kE * sizeof(float), s);
    cudaEventRecord(ev_plan_, s);

    // 3. the copies (ring of staging buffers), conversions and products
    const int n_ram = n_used - n_res;
    const size_t* st = L.ram.stride;
    auto staging_of = [&](int i) { return (int) ((ring_ + (unsigned long long) i) % (unsigned long long) cfg_.staging); };
    timed_copy_ = n_ram > 0;
    auto issue_copy = [&](int i) {
        const int b = staging_of(i), e = order_ram[(size_t) i];
        cudaStreamWaitEvent(copy_, freed_[b], 0);
        if (i == 0) cudaEventRecord(ev_copy0_, copy_);
        if (mode != Mode::compute_only) {
            uint8_t* dst = staging_ + (size_t) b * kMxSlotBytes;
            cudaMemcpyAsync(dst, L.ram.matrix(0, e), st[0], cudaMemcpyHostToDevice, copy_);
            cudaMemcpyAsync(dst + st[0], L.ram.matrix(1, e), st[1], cudaMemcpyHostToDevice, copy_);
            cudaMemcpyAsync(dst + st[0] + st[1], L.ram.matrix(2, e), st[2], cudaMemcpyHostToDevice, copy_);
        }
        cudaEventRecord(arrived_[b], copy_);
        if (i == n_ram - 1) cudaEventRecord(ev_copy1_, copy_);
    };
    for (int i = 0; i < std::min(cfg_.staging, n_ram); ++i) issue_copy(i);
    // The compute stream's order: the resident groups spread over the waits for the copied ones, all of them before
    // the last copied group (so the copy ring keeps moving, and after the last copy only one group is left).
    const int n_rg = (int) (std::lower_bound(groups.begin(), groups.end(), n_res,
                                             [](const Group& g, int v) { return g.first < v; }) - groups.begin());
    const int n_cg = (int) groups.size() - n_rg;
    std::vector<int> schedule;
    for (int c = 0, r = 0; c < n_cg; ++c) {
        for (const int until = (int) ((long long) (c + 1) * n_rg / n_cg); r < until; ++r) schedule.push_back(r);
        schedule.push_back(n_rg + c);
    }
    if (n_cg == 0)
        for (int r = 0; r < n_rg; ++r) schedule.push_back(r);
    auto* ctx = (mmq::Context*) mmq_;
    for (const int gi : schedule) {
        const Group& g = groups[(size_t) gi];
        for (int j = 0; j < g.n; ++j) {
            const int q = g.first + j, e = order[(size_t) q];
            uint8_t* gq = gu_q8_ + (size_t) j * kPfGuQ8Bytes;
            uint8_t* dq = d_q8_ + (size_t) j * kPfDownQ8Bytes;
            if (q < n_res) {
                if (mode != Mode::copy_only)
                    mxfp4_slot_to_q8_0((const void*) L.slot_addr[(size_t) L.resident[e]], gq, dq, s);
            } else {
                const int i = q - n_res, b = staging_of(i);
                cudaStreamWaitEvent(s, arrived_[b], 0);
                if (mode != Mode::copy_only) mxfp4_raw_to_q8_0(staging_ + (size_t) b * kMxSlotBytes, gq, dq, s);
                cudaEventRecord(freed_[b], s);
                if (i + cfg_.staging < n_ram) issue_copy(i + cfg_.staging);
            }
        }
        if (mode == Mode::copy_only) continue;
        for (int ti = g.t0; ti < g.t0 + g.nt; ++ti) {
            const Tile& t = tiles[(size_t) ti];
            const int rows = t.b - t.a, j0 = t.q0 - g.first;
            mmq::quantize(L.x, row_tok_ + t.a, xq_, kTypeQ8, kE, kE, rows, s);   // the tile's x rows, gathered
            mmq::Product gu;
            gu.w = gu_q8_ + (size_t) j0 * kPfGuQ8Bytes;
            gu.type = kTypeQ8;
            gu.w_rows = 2 * kF;
            gu.w_cols = kE;
            gu.expert_bytes = kPfGuQ8Bytes;
            gu.n = t.nq;
            gu.xq = xq_;
            gu.bounds = plan_ + 2 * kNExpert + 1 + t.rel;
            gu.ids = iota_;
            gu.total_rows = rows;
            gu.max_rows = t.max_rows;
            gu.dst = gu_;
            gu.ld_dst = kE;
            ctx->run(gu, s);
            mmq::swiglu(gu_, h_, rows, kF, false, s);
            mmq::quantize(h_, nullptr, hq_, kTypeQ8, kF, kF, rows, s);
            mmq::Product dn = gu;
            dn.w = d_q8_ + (size_t) j0 * kPfDownQ8Bytes;
            dn.w_rows = kE;
            dn.w_cols = kF;
            dn.expert_bytes = kPfDownQ8Bytes;
            dn.xq = hq_;
            dn.dst = gu_;   // the tile's gate|up rows, which swiglu has consumed
            ctx->run(dn, s);
            // 4. weighted into `out`, one expert after the other (a fixed order for every output row)
            for (int j = 0; j < t.nq; ++j) {
                const int sa = std::max(bounds[t.q0 + j], t.a), sb = std::min(bounds[t.q0 + j + 1], t.b);
                add_rows_kernel<<<sb - sa, 256, 0, s>>>(gu_ + (size_t) (sa - t.a) * kE, row_tok_ + sa, row_w_ + sa, L.out);
            }
            check("add_rows");
        }
    }
    ring_ += (unsigned long long) n_ram;
    cudaEventRecord(ev_end_, s);
    times_ = PrefillMoeTimes{};
    times_.used = n_used;
    times_.resident = n_res;
    times_.copied = n_ram;
    times_.tiles = (int) tiles.size();
    times_.copied_bytes = mode == Mode::compute_only ? 0 : (size_t) n_ram * L.ram.expert_bytes();
}

PrefillMoeTimes PrefillMoe::times() {
    cudaEventSynchronize(ev_end_);
    cudaStreamSynchronize(copy_);
    cudaEventElapsedTime(&times_.total, ev_start_, ev_end_);
    cudaEventElapsedTime(&times_.routing, ev_start_, ev_plan_);
    if (timed_copy_) {
        cudaEventElapsedTime(&times_.copy, ev_copy0_, ev_copy1_);
        cudaEventElapsedTime(&times_.tail, ev_copy1_, ev_end_);
    } else {
        times_.copy = 0;
        times_.tail = 0;
    }
    return times_;
}

}  // namespace strata::mimo
