// mimo_fused_test - the decode window's fused kernels (src/mimo/fused.cu, the q8_1 output of the attention combine)
// against the separate kernels they replace, on synthetic data (no model).  Everything must be BITWISE equal: x, h,
// the q8_1 codes and fp16 scales (not the block sums, which Q8_0 MMVQ does not read), the router's logits, picks,
// weights, slots and doorbell payload, and the attention output.
//
//   mimo_fused_test           exit 0 when all match, 1 otherwise, 77 without a GPU
//   mimo_fused_test --bench   us per call, separate kernels against fused, T = 1 and 2 (CUDA graphs of 200 calls)
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/mimo/attn.hpp"
#include "strata/mimo/expert_mxfp4.hpp"
#include "strata/mimo/fused.hpp"
#include "strata/mimo/ops.hpp"
#include "strata/mimo/router.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace strata::mimo;
namespace sk = strata::kernels;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

template <class T>
struct Dev {
    T* p = nullptr;
    size_t n = 0;
    explicit Dev(size_t count) : n(count) { ck(cudaMalloc(&p, count * sizeof(T) + 16), "cudaMalloc"); }
    ~Dev() { cudaFree(p); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    void up(const std::vector<T>& v) { ck(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "up"); }
    std::vector<T> down() const {
        std::vector<T> v(n);
        ck(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost), "down");
        return v;
    }
};

template <class T>
struct Mapped {
    T* h = nullptr;
    T* d = nullptr;
    size_t n = 0;
    explicit Mapped(size_t count) : n(count) {
        ck(cudaHostAlloc((void**) &h, count * sizeof(T) + 16, cudaHostAllocMapped), "cudaHostAlloc");
        std::memset(h, 0, count * sizeof(T) + 16);
        ck(cudaHostGetDevicePointer((void**) &d, h, 0), "mapped pointer");
    }
    ~Mapped() { cudaFreeHost(h); }
    Mapped(const Mapped&) = delete;
    Mapped& operator=(const Mapped&) = delete;
};

int g_fail = 0;

template <class T>
bool same(const char* what, const T* a, const T* b, size_t n) {
    if (std::memcmp(a, b, n * sizeof(T)) == 0) return true;
    size_t i = 0;
    while (std::memcmp(a + i, b + i, sizeof(T)) == 0) ++i;
    std::printf("  FAIL %s: first difference at %zu of %zu\n", what, i, n);
    ++g_fail;
    return false;
}

// q8_1 blocks: the fp16 scale (bytes 0-1) and the 32 codes (bytes 4-35); bytes 2-3 (the sum) are not compared
bool same_q8(const char* what, const std::vector<uint8_t>& a, const std::vector<uint8_t>& b, size_t blocks) {
    for (size_t k = 0; k < blocks; ++k) {
        const uint8_t *x = &a[k * 36], *y = &b[k * 36];
        if (std::memcmp(x, y, 2) != 0 || std::memcmp(x + 4, y + 4, 32) != 0) {
            std::printf("  FAIL %s: block %zu of %zu differs\n", what, k, blocks);
            ++g_fail;
            return false;
        }
    }
    return true;
}

std::vector<float> randv(std::mt19937& rng, size_t n, float sd) {
    std::normal_distribution<float> nd(0.f, sd);
    std::vector<float> v(n);
    for (float& x : v) x = nd(rng);
    return v;
}

// add + rmsnorm + q8_1 in its three uses: residual (scale, zero y), the MoE end (y + mapped CPU sums), no add
void test_norm(std::mt19937& rng, int T, int n) {
    std::printf("add_rmsnorm_q8_1 T=%d n=%d\n", T, n);
    const size_t N = (size_t) T * n, blocks = N / 32;
    std::vector<float> x = randv(rng, N, 3.f), y = randv(rng, N, 0.5f), c = randv(rng, N, 0.2f);
    std::vector<float> w = randv(rng, (size_t) n, 1.f);
    for (int i = 0; i < 64; ++i) x[(size_t) i] = 0.f, y[(size_t) i] = 0.f, c[(size_t) i] = 0.f;   // zero blocks
    for (int i = 64; i < 96; ++i) y[(size_t) i] *= 1e4f;                                           // a wide block
    Dev<float> dx(N), dy(N), dc(N), dw((size_t) n), dh(N), dx2(N), dy2(N), dh2(N);
    Dev<uint8_t> q1(blocks * 36), q2(blocks * 36);
    Mapped<float> mc(N), hout(N);
    const bool experts = n == kMxEmbd;   // the experts' q8_1 input too (residual mode)
    const size_t xqb = (size_t) T * kMxXqBytes, scb = mxfp4_experts_scratch_bytes(T, 1);
    Dev<uint8_t> sc1(scb), sc2(scb);
    Dev<unsigned long long> slot_ptr(1);
    Dev<int32_t> no_slot((size_t) T);
    Dev<float> wt1((size_t) T), out((size_t) T * kMxEmbd);
    no_slot.up(std::vector<int32_t>((size_t) T, -1));   // no resident expert: mxfp4_experts only quantizes x
    std::memcpy(mc.h, c.data(), N * 4);
    dw.up(w);
    cudaStream_t s = nullptr;
    ck(cudaStreamCreate(&s), "stream");
    const float eps = 1e-5f;
    for (int mode = 0; mode < 3; ++mode) {
        dx.up(x); dy.up(y); dx2.up(x); dy2.up(y);
        ck(cudaMemset(q1.p, 0, blocks * 36), "memset");
        ck(cudaMemset(q2.p, 0, blocks * 36), "memset");
        const float scale = mode == 0 ? 0.707f : 1.f;
        if (mode == 0) mimo_residual_add(dx.p, dy.p, scale, (int) N, s);
        if (mode == 1) {
            sk::copy_from_mapped(dc.p, mc.d, (int64_t) N, s);
            add_scaled(dy.p, dc.p, 1.f, (int) N, s);
            add_scaled(dx.p, dy.p, 1.f, (int) N, s);
        }
        mimo_rmsnorm(dx.p, dw.p, dh.p, n, T, eps, s);
        sk::native_quantize_q8_1(dh.p, q1.p, n, T, s);
        mimo_add_rmsnorm_q8_1(dx2.p, mode == 2 ? nullptr : dy2.p, mode == 1 ? mc.d : nullptr, scale, dw.p, dh2.p, q2.p,
                              mode == 0, n, T, eps, s, mode == 0 ? hout.d : nullptr,
                              mode == 0 && experts ? sc2.p : nullptr);
        if (mode == 0 && experts) mxfp4_experts(slot_ptr.p, no_slot.p, wt1.p, dh.p, out.p, T, 1, sc1.p, s);
        ck(cudaStreamSynchronize(s), "norm");
        const char* names[] = {"residual", "moe end", "plain"};
        std::printf("  %s\n", names[mode]);
        same("x", dx.down().data(), dx2.down().data(), N);
        same("h", dh.down().data(), dh2.down().data(), N);
        if (mode == 0) same("h (mapped copy)", dh.down().data(), hout.h, N);
        if (mode == 0 && experts) same("experts' q8_1 input", sc1.down().data(), sc2.down().data(), xqb);
        same_q8("q8_1", q1.down(), q2.down(), blocks);
        if (mode == 0) {
            const std::vector<float> z = dy2.down();
            for (size_t i = 0; i < N; ++i)
                if (z[i] != 0.f) { std::printf("  FAIL y not zeroed\n"); ++g_fail; break; }
        }
    }
    cudaStreamDestroy(s);
}

void test_router(std::mt19937& rng, int T) {
    constexpr int E = 4096, NE = 256, K = 8;
    std::printf("router_fused T=%d\n", T);
    std::vector<float> w = randv(rng, (size_t) NE * E, 0.02f), h = randv(rng, (size_t) T * E, 1.f);
    std::vector<float> bias = randv(rng, NE, 0.05f);
    for (int e = 10; e < 40; e += 3) {   // exact ties: a copied row and bias (the lower id must win)
        std::memcpy(&w[(size_t) (e + 1) * E], &w[(size_t) e * E], E * 4);
        bias[(size_t) e + 1] = bias[(size_t) e];
    }
    std::vector<int32_t> res(NE);
    for (int e = 0; e < NE; ++e) res[(size_t) e] = (rng() % 3) ? (int) (rng() % 1000) : -1;
    Dev<float> dw((size_t) NE * E), dh((size_t) T * E), db(NE), lg1((size_t) T * NE), lg2((size_t) T * NE);
    Dev<float> wt1((size_t) T * K), wt2((size_t) T * K);
    Dev<int32_t> id1((size_t) T * K), id2((size_t) T * K), dres(NE), sl1((size_t) T * K), sl2((size_t) T * K);
    dw.up(w); dh.up(h); db.up(bias); dres.up(res);
    Mapped<float> x1((size_t) T * E), mw1((size_t) T * K), mw2((size_t) T * K);
    Mapped<int32_t> mi1((size_t) T * K), mi2((size_t) T * K);
    Mapped<uint32_t> seq1(16), seq2(16);
    cudaStream_t s = nullptr;
    ck(cudaStreamCreate(&s), "stream");
    for (int rep = 0; rep < 3; ++rep) {   // the completion counter must reset between launches
        router_logits(dw.p, dh.p, lg1.p, T, E, NE, s);
        router_topk(lg1.p, db.p, id1.p, wt1.p, T, NE, K, 1.f, s);
        resident_slots(id1.p, dres.p, sl1.p, T * K, s);
        sk::doorbell_publish(dh.p, id1.p, wt1.p, (int64_t) T * E, (int64_t) T * K, x1.d, mi1.d, mw1.d, seq1.d, s);
        mimo_router_fused(dw.p, db.p, dh.p, lg2.p, id2.p, wt2.p, T, E, NE, K, 1.f, dres.p, sl2.p, mi2.d, mw2.d, seq2.d,
                          s);
        ck(cudaStreamSynchronize(s), "router");
    }
    same("logits", lg1.down().data(), lg2.down().data(), lg1.n);
    same("ids", id1.down().data(), id2.down().data(), id1.n);
    same("weights", wt1.down().data(), wt2.down().data(), wt1.n);
    same("slots", sl1.down().data(), sl2.down().data(), sl1.n);
    same("doorbell ids", mi1.h, mi2.h, mi1.n);
    same("doorbell weights", mw1.h, mw2.h, mw1.n);
    if (*seq1.h != 3 || *seq2.h != 3) { std::printf("  FAIL seq %u %u (want 3)\n", *seq1.h, *seq2.h); ++g_fail; }
    // without the doorbell and the slots (the non-graph path)
    mimo_router_fused(dw.p, db.p, dh.p, lg2.p, id2.p, wt2.p, T, E, NE, K, 1.f, nullptr, nullptr, nullptr, nullptr,
                      nullptr, s);
    ck(cudaStreamSynchronize(s), "router");
    same("ids (no doorbell)", id1.down().data(), id2.down().data(), id1.n);
    cudaStreamDestroy(s);
}

// the combine kernel's q8_1 (and the direct SWA T = 1 path's separate quantization) against native_quantize_q8_1
void test_attn(std::mt19937& rng, bool swa, int T, int pos0) {
    constexpr int H = 64, DK = 192, DV = 128;
    const int n_kv = swa ? 8 : 4, rows = DK * (H + n_kv) + DV * n_kv;
    std::printf("attn q8_1 %s T=%d pos0=%d\n", swa ? "SWA" : "global", T, pos0);
    std::string err;
    AttnKv kv;
    if (!attn_kv_alloc(kv, n_kv, swa ? 160 : 4096, swa, err)) throw std::runtime_error(err);
    Dev<float> sinks(H);
    sinks.up(randv(rng, H, 1.f));
    const AttnLayer L = attn_layer(H, n_kv, swa, 128, swa ? 10000.f : 1e7f, DK, swa ? sinks.p : nullptr);
    AttnScratch scr;
    if (!attn_scratch_alloc(scr, 0, err)) throw std::runtime_error(err);
    Dev<float> qkv((size_t) 8 * rows), q((size_t) 8 * H * DK), o1((size_t) T * H * DV), o2((size_t) T * H * DV);
    Dev<int32_t> pos(1);
    const size_t blocks = (size_t) T * H * DV / 32;
    Dev<uint8_t> q1(blocks * 36), q2(blocks * 36);
    cudaStream_t s = nullptr;
    ck(cudaStreamCreate(&s), "stream");
    for (int p = 0; p < pos0 + T; p += 8) {   // fill the cache up to the window, 8 positions at a time
        const int t = std::min(8, pos0 + T - p);
        qkv.up(randv(rng, (size_t) t * rows, 1.f));
        pos.up(std::vector<int32_t>{p});
        mimo_qkv_rope_append(qkv.p, pos.p, t, L, kv, q.p, s);
        ck(cudaStreamSynchronize(s), "append");
    }
    qkv.up(randv(rng, (size_t) T * rows, 1.f));
    pos.up(std::vector<int32_t>{pos0});
    mimo_qkv_rope_append(qkv.p, pos.p, T, L, kv, q.p, s);
    mimo_attn_decode(q.p, pos.p, T, L, kv, scr, o1.p, s);
    sk::native_quantize_q8_1(o1.p, q1.p, H * DV, T, s);
    mimo_attn_decode(q.p, pos.p, T, L, kv, scr, o2.p, s, 0, q2.p);
    ck(cudaStreamSynchronize(s), "attn");
    cudaStreamDestroy(s);
    same("out", o1.down().data(), o2.down().data(), o1.n);
    same_q8("q8_1", q1.down(), q2.down(), blocks);
    attn_scratch_free(scr);
    attn_kv_free(kv);
}

// us per call of `body`, captured 200 times into one graph (the decode window's back-to-back launches)
template <class F>
double time_graph(cudaStream_t s, F body) {
    constexpr int kReps = 200;
    cudaGraph_t g = nullptr;
    cudaGraphExec_t ge = nullptr;
    ck(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal), "capture");
    for (int i = 0; i < kReps; ++i) body();
    ck(cudaStreamEndCapture(s, &g), "end capture");
    ck(cudaGraphInstantiate(&ge, g, 0), "instantiate");
    ck(cudaGraphLaunch(ge, s), "warm-up");
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    cudaEventRecord(a, s);
    for (int i = 0; i < 5; ++i) cudaGraphLaunch(ge, s);
    cudaEventRecord(b, s);
    ck(cudaEventSynchronize(b), "time");
    float ms = 0;
    cudaEventElapsedTime(&ms, a, b);
    cudaGraphExecDestroy(ge);
    cudaGraphDestroy(g);
    return ms * 1000.0 / (5 * kReps);
}

void bench(int T) {
    constexpr int E = 4096, NE = 256, K = 8;
    std::mt19937 rng(1);
    const size_t N = (size_t) T * E;
    Dev<float> x(N), y(N), c(N), w(E), h(N), gw((size_t) NE * E), bias(NE), lg((size_t) T * NE), wt((size_t) T * K);
    Dev<int32_t> ids((size_t) T * K), res(NE), slots((size_t) T * K);
    Dev<uint8_t> q8(N / 32 * 36);
    x.up(randv(rng, N, 1.f)); y.up(randv(rng, N, 0.1f)); w.up(randv(rng, E, 1.f)); h.up(randv(rng, N, 1.f));
    gw.up(randv(rng, (size_t) NE * E, 0.02f)); bias.up(randv(rng, NE, 0.05f)); res.up(std::vector<int32_t>(NE, 3));
    Mapped<float> mc(N), mx(N), mw((size_t) T * K);
    Mapped<int32_t> mi((size_t) T * K);
    Mapped<uint32_t> seq(16);
    cudaStream_t s = nullptr;
    ck(cudaStreamCreate(&s), "stream");
    const float eps = 1e-6f;
    std::printf("T = %d\n", T);
    const double r0 = time_graph(s, [&] {
        mimo_residual_add(x.p, y.p, 0.7f, (int) N, s);
        mimo_rmsnorm(x.p, w.p, h.p, E, T, eps, s);
    });
    const double r1 = time_graph(s, [&] {
        mimo_add_rmsnorm_q8_1(x.p, y.p, nullptr, 0.7f, w.p, h.p, nullptr, true, E, T, eps, s, mx.d);
    });
    std::printf("  residual + ffn_norm (+ fill_zero, doorbell h): %6.2f -> %6.2f us\n", r0, r1);
    const double e0 = time_graph(s, [&] {
        sk::copy_from_mapped(c.p, mc.d, (int64_t) N, s);
        add_scaled(y.p, c.p, 1.f, (int) N, s);
        add_scaled(x.p, y.p, 1.f, (int) N, s);
        mimo_rmsnorm(x.p, w.p, h.p, E, T, eps, s);
        sk::native_quantize_q8_1(h.p, q8.p, E, T, s);
    });
    const double e1 = time_graph(s, [&] {
        mimo_add_rmsnorm_q8_1(x.p, y.p, mc.d, 1.f, w.p, h.p, q8.p, false, E, T, eps, s);
    });
    std::printf("  MoE end + next attn_norm + q8_1:               %6.2f -> %6.2f us\n", e0, e1);
    const double t0 = time_graph(s, [&] {
        router_logits(gw.p, h.p, lg.p, T, E, NE, s);
        router_topk(lg.p, bias.p, ids.p, wt.p, T, NE, K, 1.f, s);
        resident_slots(ids.p, res.p, slots.p, T * K, s);
        sk::doorbell_publish(h.p, ids.p, wt.p, (int64_t) N, (int64_t) T * K, mx.d, mi.d, mw.d, seq.d, s);
        fill_zero(y.p, (int) N, s);
    });
    const double t1 = time_graph(s, [&] {
        mimo_router_fused(gw.p, bias.p, h.p, lg.p, ids.p, wt.p, T, E, NE, K, 1.f, res.p, slots.p, mi.d, mw.d, seq.d,
                          s);
    });
    std::printf("  router + slots + doorbell (+ fill_zero):       %6.2f -> %6.2f us\n", t0, t1);
    const double t2 = time_graph(s, [&] {
        mimo_router_fused(gw.p, bias.p, h.p, lg.p, ids.p, wt.p, T, E, NE, K, 1.f, res.p, slots.p, nullptr, nullptr,
                          nullptr, s);
    });
    const double t3 = time_graph(s, [&] { router_logits(gw.p, h.p, lg.p, T, E, NE, s); });
    std::printf("  (fused router without the doorbell %.2f us; router_logits alone %.2f us)\n", t2, t3);
    for (int ku : {0, 1, 4}) {
        const double tk = time_graph(s, [&] {
            mimo_router_fused(gw.p, bias.p, h.p, lg.p, ids.p, wt.p, T, E, NE, ku, 1.f, res.p, slots.p, nullptr, nullptr,
                              nullptr, s);
        });
        std::printf("  (n_used %d: %.2f us)\n", ku, tk);
    }
    cudaStreamDestroy(s);
}

}  // namespace

int main(int argc, char** argv) {
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev == 0) {
        std::printf("no CUDA device: skipped\n");
        return 77;
    }
    if (argc > 1 && std::string(argv[1]) == "--bench") {
        bench(1);
        bench(2);
        return 0;
    }
    try {
        std::mt19937 rng(7);
        for (int T : {1, 2, 3, 8}) test_norm(rng, T, 4096);
        test_norm(rng, 2, 2048);
        for (int T = 1; T <= 8; ++T) test_router(rng, T);
        for (bool swa : {false, true})
            for (int T : {1, 2, 8}) test_attn(rng, swa, T, 300);
    } catch (const std::exception& e) {
        std::printf("FAIL: %s\n", e.what());
        return 1;
    }
    std::printf(g_fail ? "%d FAILED\n" : "all bitwise equal\n", g_fail);
    return g_fail ? 1 : 0;
}
