// mimo_prefill_moe_test: the prompt path's experts (src/mimo/prefill_moe.cu) on one real MoE layer of the GGUF.
//
// The layer's experts are copied into RAM (pinned with cudaHostRegister, as the prompt driver would), a random
// `--resident` share of them is put into VRAM slots (the decoder's slot format), and P = 1, 8, 333, 2048, 8192 tokens with
// random x (N(0, 1) times the layer's ffn_norm weights) and random routing over all 256 experts (8 distinct per
// token, random normalized weights; at 8,192 every token also picks expert 7) go through PrefillMoe::run_layer with
// 512-row tiles (experts split across tiles, expert 7 over 16 of them).  Checked against two CPU references on ggml's
// MXFP4 dequantizer (every token for P <= 333, `--check` random tokens above):
//   q8   llama.cpp MMQ's arithmetic in double: x and h quantized to q8_1 as quantize_mmq_q8_1 does (scale amax/127,
//        kept as a float), exact dot products.  Gate: max |gpu - ref| / max |ref| below 5e-3 for every token
//        (the worst cases are a q8_1 value of h rounding the other way after a sum in another order: up to 2.0e-3 on
//        layer 47's spiky inputs; the median is ~2e-7)
//   f32  plain float, no activation quantization (reported: the cost of q8_1 activations)
// Also: the two MXFP4 -> Q8_0 converters (GGUF layout and slot layout) give the same bytes, those bytes dequantize to
// exactly ggml's values, the exponent range of the layer's blocks (Q8_0's fp16 scale is exact for e in 104..143), a
// repeated call is bitwise equal, and with every expert resident or none the result moves by < 1e-3 of its max (the
// MMQ launches group the experts differently, which can change the order of a sum, not the arithmetic).
// Last, the A/B behind the Q8_0 choice: the gate|up products of 16 experts x 128 rows through llama.cpp's own MXFP4
// MMQ (on sm_120: FP4 activations, FP4 MMA) and through the Q8_0 path, both against double products of the float x,
// and their times.
//
// usage: mimo_prefill_moe_test <MiMo GGUF, any shard> [--layer 24] [--resident 0.5] [--check 96] [--seed 1]
#include "strata/mimo/expert_mxfp4.hpp"
#include "strata/mimo/model.hpp"
#include "strata/mimo/prefill_moe.hpp"
#include "strata/prefill/moe_mmq.hpp"

#include "ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace strata::mimo;

#define CK(x)                                                                                       \
    do {                                                                                            \
        cudaError_t e_ = (x);                                                                       \
        if (e_ != cudaSuccess) {                                                                    \
            std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
            std::exit(1);                                                                           \
        }                                                                                           \
    } while (0)

namespace {

constexpr int E = kMxEmbd, F = kMxFF, NE = 256, K = 8;

void parallel(int n, int threads, const auto& fn) {
    std::atomic<int> next{0};
    std::vector<std::thread> th;
    for (int t = 0; t < threads; ++t)
        th.emplace_back([&] {
            for (int i; (i = next.fetch_add(1)) < n;) fn(i);
        });
    for (auto& t : th) t.join();
}

// quantize_mmq_q8_1 (D4 layout, Q8_0 weights) as doubles: per 32 values d_inv = 127 / amax, q = round(x * d_inv),
// value = q / d_inv
void q8_mmq(const float* x, int n, double* out) {
    for (int b = 0; b < n / 32; ++b) {
        float amax = 0;
        for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(x[b * 32 + i]));
        const float d_inv = 127.0f / amax, d = 1.0f / d_inv;
        for (int i = 0; i < 32; ++i)
            out[b * 32 + i] = amax == 0.0f ? 0.0 : (double) std::round(x[b * 32 + i] * d_inv) * (double) d;
    }
}

struct Expert {
    std::vector<float> w[3];   // gate/up [F][E], down [E][F]
};

void dequant(const ExpertSlices& xs, int e, Expert& out) {
    const ggml_type_traits* tt = ggml_get_type_traits(GGML_TYPE_MXFP4);
    for (int m = 0; m < 3; ++m) {
        out.w[m].resize((size_t) E * F);
        tt->to_float(xs.matrix(m, e), out.w[m].data(), (int64_t) E * F);
    }
}

// one pick: rq (q8 arithmetic) and rf (float) of expert(x)
void ref_pick(const Expert& ex, const float* x, const double* xq, double* rq, double* rf) {
    std::vector<float> h((size_t) F);
    std::vector<double> hf((size_t) F), hq((size_t) F);
    for (int r = 0; r < F; ++r) {
        const float* gr = ex.w[0].data() + (size_t) r * E;
        const float* ur = ex.w[1].data() + (size_t) r * E;
        double g = 0, u = 0, gf = 0, uf = 0;
        for (int i = 0; i < E; ++i) {
            g += gr[i] * xq[i];
            u += ur[i] * xq[i];
            gf += (double) gr[i] * x[i];
            uf += (double) ur[i] * x[i];
        }
        h[(size_t) r] = (float) (g / (1 + std::exp(-g)) * u);
        hf[(size_t) r] = gf / (1 + std::exp(-gf)) * uf;
    }
    q8_mmq(h.data(), F, hq.data());
    for (int r = 0; r < E; ++r) {
        const float* dr = ex.w[2].data() + (size_t) r * F;
        double s = 0, sf = 0;
        for (int i = 0; i < F; ++i) {
            s += dr[i] * hq[(size_t) i];
            sf += dr[i] * hf[(size_t) i];
        }
        rq[r] = s;
        rf[r] = sf;
    }
}

struct Err { double max_rel = 0, l2 = 0; };
Err compare(const float* got, const double* ref) {
    double mref = 0, mdiff = 0, s2 = 0, r2 = 0;
    for (int i = 0; i < E; ++i) {
        mref = std::max(mref, std::fabs(ref[i]));
        mdiff = std::max(mdiff, std::fabs(got[i] - ref[i]));
        s2 += (got[i] - ref[i]) * (got[i] - ref[i]);
        r2 += ref[i] * ref[i];
    }
    return {mref > 0 ? mdiff / mref : mdiff, r2 > 0 ? std::sqrt(s2 / r2) : std::sqrt(s2)};
}

// gate|up products of experts 0..15, 128 rows each: llama.cpp's MXFP4 MMQ on the GGUF bytes vs the Q8_0 path
void fp4_ab(const ExpertSlices& xs, const ExpertSlices& rs, const float* norm, std::mt19937& rng, unsigned threads) {
    namespace mmq = strata::prefill::mmq;
    constexpr int NX = 16, R = 128, ROWS = NX * R, CHECK = 8;
    if (!mmq::supported(39) || !mmq::fits(39, 2 * F)) {
        std::printf("\nMXFP4 MMQ A/B: not in this build / no tile on this GPU\n");
        return;
    }
    uint8_t *raw = nullptr, *q8 = nullptr, *scratch = nullptr;
    void *xq8 = nullptr, *xq4 = nullptr;
    float *dx = nullptr, *d8 = nullptr, *d4 = nullptr;
    int32_t *iota = nullptr, *bounds = nullptr;
    CK(cudaMalloc(&raw, (size_t) NX * kMxSlotBytes));
    CK(cudaMalloc(&q8, (size_t) NX * kPfGuQ8Bytes));
    CK(cudaMalloc(&scratch, kPfDownQ8Bytes));
    CK(cudaMalloc(&xq8, mmq::q8_bytes(ROWS, E)));
    CK(cudaMalloc(&xq4, mmq::q8_bytes(ROWS, E)));
    CK(cudaMalloc(&dx, sizeof(float) * ROWS * E));
    CK(cudaMalloc(&d8, sizeof(float) * ROWS * 2 * F));
    CK(cudaMalloc(&d4, sizeof(float) * ROWS * 2 * F));
    CK(cudaMalloc(&iota, sizeof(int32_t) * ROWS));
    CK(cudaMalloc(&bounds, sizeof(int32_t) * (NX + 1)));
    for (int q = 0; q < NX; ++q) {
        uint8_t* dst = raw + (size_t) q * kMxSlotBytes;
        CK(cudaMemcpy(dst, rs.matrix(0, q), xs.stride[0], cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dst + xs.stride[0], rs.matrix(1, q), xs.stride[1], cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dst + xs.stride[0] + xs.stride[1], rs.matrix(2, q), xs.stride[2], cudaMemcpyHostToDevice));
        mxfp4_raw_to_q8_0(dst, q8 + (size_t) q * kPfGuQ8Bytes, scratch, nullptr);
    }
    std::vector<float> x((size_t) ROWS * E);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (int r = 0; r < ROWS; ++r)
        for (int i = 0; i < E; ++i) x[(size_t) r * E + i] = nd(rng) * (norm ? norm[i] : 1.0f);
    std::vector<int32_t> b(NX + 1);
    for (int q = 0; q <= NX; ++q) b[(size_t) q] = q * R;
    CK(cudaMemcpy(dx, x.data(), sizeof(float) * x.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(bounds, b.data(), sizeof(int32_t) * b.size(), cudaMemcpyHostToDevice));
    mmq::iota(iota, ROWS, nullptr);
    mmq::Context ctx;
    auto product = [&](int type, const void* w, size_t eb, const void* xq, float* dst) {
        mmq::Product p;
        p.w = w;
        p.type = type;
        p.w_rows = 2 * F;
        p.w_cols = E;
        p.expert_bytes = eb;
        p.n = NX;
        p.xq = xq;
        p.bounds = bounds;
        p.ids = iota;
        p.total_rows = ROWS;
        p.max_rows = R;
        p.dst = dst;
        p.ld_dst = 2 * F;
        ctx.run(p, nullptr);
    };
    auto run8 = [&] { mmq::quantize(dx, iota, xq8, 8, E, E, ROWS, nullptr); product(8, q8, kPfGuQ8Bytes, xq8, d8); };
    auto run4 = [&] { mmq::quantize(dx, iota, xq4, 39, E, E, ROWS, nullptr); product(39, raw, kMxSlotBytes, xq4, d4); };
    run8();
    run4();
    CK(cudaDeviceSynchronize());
    std::vector<float> o8((size_t) ROWS * 2 * F), o4((size_t) ROWS * 2 * F);
    CK(cudaMemcpy(o8.data(), d8, sizeof(float) * o8.size(), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(o4.data(), d4, sizeof(float) * o4.size(), cudaMemcpyDeviceToHost));
    // double products of the float x for CHECK rows per expert
    std::vector<double> ref((size_t) NX * CHECK * 2 * F);
    parallel(NX, (int) threads, [&](int q) {
        Expert ex;
        dequant(xs, q, ex);
        for (int c = 0; c < CHECK; ++c) {
            const float* xr = x.data() + (size_t) (q * R + c * (R / CHECK)) * E;
            for (int o = 0; o < 2 * F; ++o) {
                const float* wr = ex.w[o < F ? 0 : 1].data() + (size_t) (o % F) * E;
                double s = 0;
                for (int i = 0; i < E; ++i) s += (double) wr[i] * xr[i];
                ref[((size_t) q * CHECK + c) * 2 * F + o] = s;
            }
        }
    });
    std::vector<double> l8, l4;
    for (int q = 0; q < NX; ++q)
        for (int c = 0; c < CHECK; ++c) {
            const size_t row = (size_t) (q * R + c * (R / CHECK));
            const double* rf = ref.data() + ((size_t) q * CHECK + c) * 2 * F;
            l8.push_back(compare(o8.data() + row * 2 * F, rf).l2);   // compare() looks at the first E = 2F values
            l4.push_back(compare(o4.data() + row * 2 * F, rf).l2);
        }
    std::sort(l8.begin(), l8.end());
    std::sort(l4.begin(), l4.end());
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    auto time = [&](const auto& fn) {
        float best = 1e30f;
        for (int r = 0; r < 20; ++r) {
            CK(cudaEventRecord(e0, nullptr));
            fn();
            CK(cudaEventRecord(e1, nullptr));
            CK(cudaEventSynchronize(e1));
            float ms = 0;
            CK(cudaEventElapsedTime(&ms, e0, e1));
            best = std::min(best, ms);
        }
        return best;
    };
    const float t8 = time(run8), t4 = time(run4);
    std::printf("\nMXFP4 MMQ A/B, gate|up of %d experts x %d rows (x quantization included), vs double products of "
                "the float x (%d rows):\n", NX, R, NX * CHECK);
    std::printf("  Q8_0 weights, q8_1 activations (this path):  rel L2 median %.2e, worst %.2e; %.3f ms\n",
                l8[l8.size() / 2], l8.back(), t8);
    std::printf("  ggml MXFP4 MMQ (sm_120: FP4 activations):    rel L2 median %.2e, worst %.2e; %.3f ms\n",
                l4[l4.size() / 2], l4.back(), t4);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    for (void* p : {(void*) raw, (void*) q8, (void*) scratch, xq8, xq4, (void*) dx, (void*) d8, (void*) d4,
                    (void*) iota, (void*) bounds})
        cudaFree(p);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <MiMo GGUF, any shard> [--layer 24] [--resident 0.5] [--check 96] [--seed 1]\n",
                     argv[0]);
        return 2;
    }
    int layer = 24, n_check = 96;
    unsigned seed = 1;
    double resident_share = 0.5;
    for (int i = 2; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--layer") && i + 1 < argc) layer = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--resident") && i + 1 < argc) resident_share = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--check") && i + 1 < argc) n_check = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--seed") && i + 1 < argc) seed = (unsigned) std::atoi(argv[++i]);
    }
    CK(cudaSetDevice(0));
    int fails = 0;
    try {
        MimoModel model(argv[1]);
        const MimoGeometry& g = model.geometry();
        if (!g.is_moe(layer) || g.n_expert != NE || g.n_embd != E || g.n_ff_exp != F || g.n_expert_used != K) {
            std::fprintf(stderr, "layer %d is not a MiMo MoE layer of the expected shape\n", layer);
            return 1;
        }
        const ExpertSlices& xs = model.experts(layer);
        const unsigned hw = std::max(4u, std::thread::hardware_concurrency() / 2);

        // ---- the exponent range: Q8_0's fp16 scale 2^(e-128) is exact for e in 104..143
        {
            std::vector<long long> hist(256, 0);
            std::vector<std::vector<long long>> part(hw, std::vector<long long>(256, 0));
            const size_t nb[3] = {(size_t) NE * F * (E / 32), (size_t) NE * F * (E / 32), (size_t) NE * E * (F / 32)};
            parallel((int) hw, (int) hw, [&](int t) {
                for (int m = 0; m < 3; ++m)
                    for (size_t b = nb[m] / hw * t, end = t + 1 == (int) hw ? nb[m] : nb[m] / hw * (t + 1); b < end; ++b)
                        ++part[(size_t) t][xs.base[m][b * 17]];
            });
            for (auto& p : part)
                for (int i = 0; i < 256; ++i) hist[(size_t) i] += p[(size_t) i];
            long long lo = 0, hi = 0, total = 0;
            int emin = 255, emax = 0;
            for (int i = 0; i < 256; ++i) {
                total += hist[(size_t) i];
                if (!hist[(size_t) i]) continue;
                emin = std::min(emin, i);
                emax = std::max(emax, i);
                if (i < 104) lo += hist[(size_t) i];
                if (i > 143) hi += hist[(size_t) i];
            }
            std::printf("layer %d: %lld MXFP4 blocks, exponents %d..%d; %lld below 104, %lld above 143 (not exact in "
                        "Q8_0)\n", layer, total, emin, emax, lo, hi);
        }

        // ---- RAM copy (pinned) and resident slots
        const size_t layer_bytes = (size_t) NE * xs.expert_bytes();
        uint8_t* ram = (uint8_t*) std::malloc(layer_bytes);
        if (!ram) { std::fprintf(stderr, "out of RAM\n"); return 1; }
        ExpertSlices rs = xs;
        {
            uint8_t* p = ram;
            for (int m = 0; m < 3; ++m) {
                const size_t bytes = (size_t) NE * xs.stride[m];
                constexpr size_t kChunk = 64u << 20;
                const int pieces = (int) ((bytes + kChunk - 1) / kChunk);
                parallel(pieces, 16, [&](int i) {
                    const size_t o = (size_t) i * kChunk;
                    std::memcpy(p + o, xs.base[m] + o, std::min(kChunk, bytes - o));
                });
                rs.base[m] = p;
                p += bytes;
            }
        }
        std::string err;
        if (!PrefillMoe::register_host(ram, layer_bytes, err)) std::printf("RAM copy NOT pinned (%s)\n", err.c_str());

        std::mt19937 rng(seed);
        std::vector<int> perm(NE);
        for (int i = 0; i < NE; ++i) perm[(size_t) i] = i;
        std::shuffle(perm.begin(), perm.end(), rng);
        const int n_res = (int) std::lround(resident_share * NE);
        std::vector<int32_t> resident(NE, -1), all_res(NE), no_res(NE, -1);
        uint8_t* slots = nullptr;
        CK(cudaMalloc(&slots, (size_t) NE * kMxSlotBytes));   // room for every expert (the all-resident run)
        uint8_t* raw = nullptr;
        CK(cudaMalloc(&raw, kMxSlotBytes));
        std::vector<unsigned long long> slot_addr(NE);
        for (int e = 0; e < NE; ++e) {
            slot_addr[(size_t) e] = (unsigned long long) (slots + (size_t) e * kMxSlotBytes);
            all_res[(size_t) e] = e;
            CK(cudaMemcpy(raw, rs.matrix(0, e), xs.stride[0], cudaMemcpyHostToDevice));
            CK(cudaMemcpy(raw + xs.stride[0], rs.matrix(1, e), xs.stride[1], cudaMemcpyHostToDevice));
            CK(cudaMemcpy(raw + xs.stride[0] + xs.stride[1], rs.matrix(2, e), xs.stride[2], cudaMemcpyHostToDevice));
            mxfp4_slot_relayout(raw, slots + (size_t) e * kMxSlotBytes, nullptr);
        }
        for (int i = 0; i < n_res; ++i) resident[(size_t) perm[(size_t) i]] = perm[(size_t) i];   // slot e = expert e
        CK(cudaDeviceSynchronize());

        // ---- converters: GGUF layout and slot layout give the same Q8_0 bytes, equal to ggml's values
        {
            uint8_t *a = nullptr, *b = nullptr;
            const size_t q8 = kPfGuQ8Bytes + kPfDownQ8Bytes;
            CK(cudaMalloc(&a, q8));
            CK(cudaMalloc(&b, q8));
            std::vector<uint8_t> ha(q8), hb(q8);
            int bad_layout = 0;
            long long bad_value = 0;
            Expert ex;
            for (int e : {0, 77, 255}) {
                CK(cudaMemcpy(raw, rs.matrix(0, e), xs.stride[0], cudaMemcpyHostToDevice));
                CK(cudaMemcpy(raw + xs.stride[0], rs.matrix(1, e), xs.stride[1], cudaMemcpyHostToDevice));
                CK(cudaMemcpy(raw + xs.stride[0] + xs.stride[1], rs.matrix(2, e), xs.stride[2], cudaMemcpyHostToDevice));
                mxfp4_raw_to_q8_0(raw, a, a + kPfGuQ8Bytes, nullptr);
                mxfp4_slot_to_q8_0((const void*) slot_addr[(size_t) e], b, b + kPfGuQ8Bytes, nullptr);
                CK(cudaMemcpy(ha.data(), a, q8, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(hb.data(), b, q8, cudaMemcpyDeviceToHost));
                bad_layout += ha != hb;
                dequant(xs, e, ex);
                // Q8_0 blocks: gate rows, up rows, then down rows; values in ggml's dequantized order
                const float* want[3] = {ex.w[0].data(), ex.w[1].data(), ex.w[2].data()};
                const size_t nblk[3] = {(size_t) F * E / 32, (size_t) F * E / 32, (size_t) E * F / 32};
                const uint8_t* blk = ha.data();
                for (int m = 0; m < 3; ++m)
                    for (size_t i = 0; i < nblk[m]; ++i, blk += 34) {
                        uint16_t dh;
                        std::memcpy(&dh, blk, 2);
                        const float d = ggml_fp16_to_fp32(dh);
                        for (int j = 0; j < 32; ++j)
                            bad_value += (float) (int8_t) blk[2 + j] * d != want[m][i * 32 + (size_t) j];
                    }
            }
            std::printf("converters (3 experts): slot layout vs GGUF layout %s; Q8_0 values vs ggml's dequantizer: "
                        "%lld differ\n", bad_layout ? "DIFFER" : "bitwise equal", bad_value);
            fails += bad_layout != 0 || bad_value != 0;
            cudaFree(a);
            cudaFree(b);
        }

        // ---- run_layer against the references
        PrefillMoeConfig cfg;
        cfg.max_tokens = 8192;
        cfg.tile_rows = 512;   // small tiles: experts split across tiles, one expert over many
        PrefillMoe moe(cfg);
        if (!moe.init(err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
        std::printf("PrefillMoe: %.0f MiB of VRAM at P <= %d, tiles of %d rows; %d of 256 experts resident\n",
                    moe.device_bytes() / 1048576.0, cfg.max_tokens, cfg.tile_rows, n_res);
        const float* norm = (const float*) model.host_data("blk." + std::to_string(layer) + ".ffn_norm.weight");
        cudaStream_t s;
        CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
        float *dx = nullptr, *dw = nullptr, *dout = nullptr;
        int32_t* dids = nullptr;
        CK(cudaMalloc(&dx, sizeof(float) * cfg.max_tokens * E));
        CK(cudaMalloc(&dout, sizeof(float) * cfg.max_tokens * E));
        CK(cudaMalloc(&dw, sizeof(float) * cfg.max_tokens * K));
        CK(cudaMalloc(&dids, sizeof(int32_t) * cfg.max_tokens * K));
        std::printf("\n%6s %8s %6s %6s | %-24s | %-24s\n", "P", "checked", "used", "copied", "vs q8 ref: max rel, worst / median",
                    "vs f32 ref: max rel / rel L2, worst");
        for (int P : {1, 8, 333, 2048, 8192}) {
            std::normal_distribution<float> nd(0.0f, 1.0f);
            std::uniform_real_distribution<float> ud(0.05f, 1.0f);
            std::vector<float> x((size_t) P * E), w((size_t) P * K);
            std::vector<int32_t> ids((size_t) P * K);
            for (int t = 0; t < P; ++t) {
                for (int i = 0; i < E; ++i) x[(size_t) t * E + i] = nd(rng) * (norm ? norm[i] : 1.0f);
                std::shuffle(perm.begin(), perm.end(), rng);
                if (P == 8192) {   // a hot expert: every token picks expert 7 (8,192 rows, 16 tiles of its own)
                    const size_t at = (size_t) (std::find(perm.begin(), perm.end(), 7) - perm.begin());
                    if (at >= (size_t) K) std::swap(perm[at], perm[(size_t) (t % K)]);
                }
                float sum = 0;
                for (int k = 0; k < K; ++k) {
                    ids[(size_t) t * K + k] = perm[(size_t) k];
                    sum += (w[(size_t) t * K + k] = ud(rng));
                }
                for (int k = 0; k < K; ++k) w[(size_t) t * K + k] /= sum;
            }
            CK(cudaMemcpy(dx, x.data(), sizeof(float) * x.size(), cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dw, w.data(), sizeof(float) * w.size(), cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dids, ids.data(), sizeof(int32_t) * ids.size(), cudaMemcpyHostToDevice));
            PrefillMoeLayer L;
            L.n_tok = P;
            L.x = dx;
            L.ids = dids;
            L.w = dw;
            L.out = dout;
            L.resident = resident.data();
            L.slot_addr = slot_addr.data();
            L.ram = rs;
            std::vector<float> out((size_t) P * E), out2((size_t) P * E), out_all((size_t) P * E), out_none((size_t) P * E);
            moe.run_layer(L, s);
            const PrefillMoeTimes tm = moe.times();
            CK(cudaMemcpy(out.data(), dout, sizeof(float) * out.size(), cudaMemcpyDeviceToHost));
            moe.run_layer(L, s);
            CK(cudaStreamSynchronize(s));
            CK(cudaMemcpy(out2.data(), dout, sizeof(float) * out.size(), cudaMemcpyDeviceToHost));
            L.resident = all_res.data();
            moe.run_layer(L, s);
            CK(cudaStreamSynchronize(s));
            CK(cudaMemcpy(out_all.data(), dout, sizeof(float) * out.size(), cudaMemcpyDeviceToHost));
            L.resident = no_res.data();
            moe.run_layer(L, s);
            CK(cudaStreamSynchronize(s));
            CK(cudaMemcpy(out_none.data(), dout, sizeof(float) * out.size(), cudaMemcpyDeviceToHost));
            const bool repeat_ok = std::memcmp(out.data(), out2.data(), sizeof(float) * out.size()) == 0;
            double res_diff = 0, out_max = 0;
            for (size_t i = 0; i < out.size(); ++i) {
                out_max = std::max(out_max, (double) std::fabs(out[i]));
                res_diff = std::max({res_diff, (double) std::fabs(out[i] - out_all[i]), (double) std::fabs(out[i] - out_none[i])});
            }
            res_diff /= out_max > 0 ? out_max : 1.0;

            // references for the checked tokens, expert by expert
            std::vector<int> toks;
            if (P <= 333) for (int t = 0; t < P; ++t) toks.push_back(t);
            else {
                std::vector<int> all(P);
                for (int t = 0; t < P; ++t) all[(size_t) t] = t;
                std::shuffle(all.begin(), all.end(), rng);
                toks.assign(all.begin(), all.begin() + std::min(P, n_check));
                std::sort(toks.begin(), toks.end());
            }
            const int nt = (int) toks.size();
            std::vector<double> xq((size_t) nt * E);
            for (int i = 0; i < nt; ++i) q8_mmq(x.data() + (size_t) toks[(size_t) i] * E, E, xq.data() + (size_t) i * E);
            std::vector<std::vector<std::pair<int, int>>> by_expert(NE);   // (checked token index, pick)
            for (int i = 0; i < nt; ++i)
                for (int k = 0; k < K; ++k) by_expert[(size_t) ids[(size_t) toks[(size_t) i] * K + k]].push_back({i, k});
            std::vector<double> rq((size_t) nt * K * E), rf((size_t) nt * K * E);
            parallel(NE, (int) hw, [&](int e) {
                if (by_expert[(size_t) e].empty()) return;
                Expert ex;
                dequant(xs, e, ex);
                for (auto [i, k] : by_expert[(size_t) e]) {
                    const size_t o = ((size_t) i * K + k) * E;
                    ref_pick(ex, x.data() + (size_t) toks[(size_t) i] * E, xq.data() + (size_t) i * E, rq.data() + o,
                             rf.data() + o);
                }
            });
            std::vector<double> eq, ef, el2;
            std::vector<double> sq(E), sf(E);
            for (int i = 0; i < nt; ++i) {
                const int t = toks[(size_t) i];
                std::fill(sq.begin(), sq.end(), 0.0);
                std::fill(sf.begin(), sf.end(), 0.0);
                for (int k = 0; k < K; ++k) {
                    const double wt = w[(size_t) t * K + k];
                    const size_t o = ((size_t) i * K + k) * E;
                    for (int r = 0; r < E; ++r) {
                        sq[(size_t) r] += wt * rq[o + r];
                        sf[(size_t) r] += wt * rf[o + r];
                    }
                }
                eq.push_back(compare(out.data() + (size_t) t * E, sq.data()).max_rel);
                const Err f = compare(out.data() + (size_t) t * E, sf.data());
                ef.push_back(f.max_rel);
                el2.push_back(f.l2);
            }
            std::sort(eq.begin(), eq.end());
            const double wq = eq.back(), mq = eq[eq.size() / 2];
            const double wf = *std::max_element(ef.begin(), ef.end()), wl2 = *std::max_element(el2.begin(), el2.end());
            const bool ok = wq < 5e-3 && repeat_ok && res_diff < 1e-3;
            fails += !ok;
            std::printf("%6d %8d %6d %6d | %10.2e / %10.2e   | %10.2e / %10.2e   %s  (repeat %s, all / none resident: "
                        "max diff / max |out| %.1e; %.2f ms)\n",
                        P, nt, tm.used, tm.copied, wq, mq, wf, wl2, ok ? "ok" : "FAIL", repeat_ok ? "bitwise" : "DIFFERS",
                        res_diff, tm.total);
        }
        fp4_ab(xs, rs, norm, rng, hw);
        cudaFree(dx);
        cudaFree(dw);
        cudaFree(dout);
        cudaFree(dids);
        cudaFree(raw);
        cudaFree(slots);
        cudaStreamDestroy(s);
        PrefillMoe::unregister_host(ram);
        std::free(ram);
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
    std::printf("\n%s\n", fails ? "FAILED" : "all passed");
    return fails ? 1 : 0;
}
