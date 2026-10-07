// expert_mxfp4_test: MiMo's GPU MXFP4 experts (src/mimo/kernels/expert_mxfp4.cu) on real experts from the GGUF.
//
// Parity: 16 experts of layers 1, 24 and 47 copied into device slots; random activations; windows of T = 1..8
// tokens, each token with K = 8 random (slot, weight) pairs of which a random subset is resident.  `out` starts
// non-zero (the call adds to it).  Compared against two CPU references built on ggml's own MXFP4 dequantizer:
//   q8   the kernel's arithmetic in double: x and h quantized to q8_1 as ggml does, exact dot products
//        (gate: max error / max |out| per token below 1e-3.  Sums run in another order, which can flip the
//        rounding of one q8_1 value of h: one flip moves the output by ~1e-4 of its max; typically ~1e-6)
//   f32  the plain float model: no activation quantization (reported; this is the q8_1 activations' own error)
// Also: the device re-layout equals the host one, and a captured CUDA graph replays bitwise equal to direct calls
// after the slot / weight / x buffers are rewritten.
//
// --bench: per-layer time of one call and effective bandwidth (distinct experts read x 13,369,344 B / time), 47
// calls in a graph over 192 slots (2.5 GB, so no window's experts are still in the 128 MB L2), repeated.
//
// usage: mimo_expert_mxfp4_test <MiMo GGUF, any shard> [--bench] [--reps N]
#include "strata/mimo/expert_mxfp4.hpp"
#include "strata/mimo/model.hpp"

#include "ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <random>
#include <set>
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

constexpr int E = kMxEmbd, F = kMxFF;

struct HostExpert {
    const uint8_t* m[3];          // GGUF gate, up, down
    std::vector<float> w[3];      // dequantized: gate/up [F][E], down [E][F]
};

// ggml quantize_q8_1 on one 32-block, as doubles: q values and the fp16-rounded scale
void q8_1(const float* x, int n, std::vector<double>& out) {
    out.resize((size_t) n);
    for (int b = 0; b < n / 32; ++b) {
        float amax = 0;
        for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(x[b * 32 + i]));
        const float d = amax / 127.0f;
        const double dh = (double) ggml_fp16_to_fp32(ggml_fp32_to_fp16(d));
        for (int i = 0; i < 32; ++i) out[(size_t) b * 32 + i] = amax == 0.0f ? 0.0 : std::round(x[b * 32 + i] / d) * dh;
    }
}

// out[e] += w * expert(x), two ways
void reference(const HostExpert& ex, const float* x, double w, double* out_q8, double* out_f32) {
    std::vector<double> xq;
    q8_1(x, E, xq);
    std::vector<float> h((size_t) F);
    std::vector<double> hf((size_t) F);
    for (int r = 0; r < F; ++r) {
        const float* gr = ex.w[0].data() + (size_t) r * E;
        const float* ur = ex.w[1].data() + (size_t) r * E;
        double g = 0, u = 0, gf = 0, uf = 0;
        for (int i = 0; i < E; ++i) {
            g += gr[i] * xq[(size_t) i];
            u += ur[i] * xq[(size_t) i];
            gf += (double) gr[i] * x[i];
            uf += (double) ur[i] * x[i];
        }
        h[(size_t) r] = (float) (g / (1 + std::exp(-g)) * u);
        hf[(size_t) r] = gf / (1 + std::exp(-gf)) * uf;
    }
    std::vector<double> hq;
    q8_1(h.data(), F, hq);
    for (int r = 0; r < E; ++r) {
        const float* dr = ex.w[2].data() + (size_t) r * F;
        double s = 0, sf = 0;
        for (int i = 0; i < F; ++i) {
            s += dr[i] * hq[(size_t) i];
            sf += dr[i] * hf[(size_t) i];
        }
        out_q8[r] += w * s;
        out_f32[r] += w * sf;
    }
}

struct Window {
    std::vector<int32_t> slots;
    std::vector<float> weights, x;
};

// T tokens x K pairs over n_slots slots; each token's slots distinct, `resident` of them >= 0 (-1: RAM)
Window make_window(std::mt19937& rng, int T, int K, int n_slots, int resident, int pool_lo = 0, int pool_n = -1) {
    if (pool_n < 0) pool_n = n_slots;
    Window w;
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_real_distribution<float> ud(0.05f, 1.0f);
    for (int t = 0; t < T; ++t) {
        std::vector<int> pick((size_t) pool_n);
        for (int i = 0; i < pool_n; ++i) pick[(size_t) i] = pool_lo + i;
        std::shuffle(pick.begin(), pick.end(), rng);
        std::vector<int> res((size_t) K, 0);
        for (int k = 0; k < resident && k < K; ++k) res[(size_t) k] = 1;
        std::shuffle(res.begin(), res.end(), rng);
        float sum = 0;
        std::vector<float> wt((size_t) K);
        for (auto& v : wt) sum += (v = ud(rng));
        for (int k = 0; k < K; ++k) {
            w.slots.push_back(res[(size_t) k] ? pick[(size_t) k] % n_slots : -1);
            w.weights.push_back(wt[(size_t) k] / sum);
        }
    }
    w.x.resize((size_t) T * E);
    for (auto& v : w.x) v = nd(rng) * 0.05f;   // ffn_norm output scale
    return w;
}

struct Dev {
    int32_t* slots = nullptr;
    float *weights = nullptr, *x = nullptr, *out = nullptr;
    void* scratch = nullptr;
    void alloc(int T, int K) {
        CK(cudaMalloc(&slots, sizeof(int32_t) * kMxMaxT * kMxMaxK));
        CK(cudaMalloc(&weights, sizeof(float) * kMxMaxT * kMxMaxK));
        CK(cudaMalloc(&x, sizeof(float) * kMxMaxT * E));
        CK(cudaMalloc(&out, sizeof(float) * kMxMaxT * E));
        CK(cudaMalloc(&scratch, mxfp4_experts_scratch_bytes(T, K)));
    }
    void upload(const Window& w) {
        CK(cudaMemcpy(slots, w.slots.data(), w.slots.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(weights, w.weights.data(), w.weights.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(x, w.x.data(), w.x.size() * 4, cudaMemcpyHostToDevice));
    }
    void free_all() {
        cudaFree(slots); cudaFree(weights); cudaFree(x); cudaFree(out); cudaFree(scratch);
    }
};

int parity(MimoModel& model, unsigned long long* d_slot_ptr, const std::vector<HostExpert>& ex, int n_slots) {
    std::mt19937 rng(1234);
    const int K = 8;
    int fails = 0;
    double worst_q8 = 0, worst_f32 = 0;
    cudaStream_t s;
    CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    std::printf("parity vs CPU (ggml dequantized weights): max |gpu - ref| / max |ref| per token\n");
    std::printf("   T  resident/token  groups   q8-ref worst   f32-ref worst\n");
    for (int T = 1; T <= kMxMaxT; ++T) {
        for (int trial = 0; trial < 2; ++trial) {
            const int resident = trial == 0 ? K : (int) (rng() % (K + 1));
            Window w = make_window(rng, T, K, n_slots, resident);
            Dev d;
            d.alloc(T, K);
            d.upload(w);
            std::vector<float> init((size_t) T * E);
            for (auto& v : init) v = (float) ((int) (rng() % 2001) - 1000) * 1e-4f;
            CK(cudaMemcpy(d.out, init.data(), init.size() * 4, cudaMemcpyHostToDevice));
            mxfp4_experts(d_slot_ptr, d.slots, d.weights, d.x, d.out, T, K, d.scratch, s);
            CK(cudaStreamSynchronize(s));
            std::vector<float> got((size_t) T * E);
            CK(cudaMemcpy(got.data(), d.out, got.size() * 4, cudaMemcpyDeviceToHost));
            std::set<int> distinct;
            double tq = 0, tf = 0;
            for (int t = 0; t < T; ++t) {
                std::vector<double> rq((size_t) E, 0.0), rf((size_t) E, 0.0);
                for (int k = 0; k < K; ++k) {
                    const int sl = w.slots[(size_t) (t * K + k)];
                    if (sl < 0) continue;
                    distinct.insert(sl);
                    reference(ex[(size_t) sl], w.x.data() + (size_t) t * E, w.weights[(size_t) (t * K + k)], rq.data(),
                              rf.data());
                }
                double mq = 0, mf = 0, eq = 0, ef = 0;
                for (int i = 0; i < E; ++i) {
                    const double g = (double) got[(size_t) t * E + i] - init[(size_t) t * E + i];
                    mq = std::max(mq, std::fabs(rq[(size_t) i]));
                    mf = std::max(mf, std::fabs(rf[(size_t) i]));
                    eq = std::max(eq, std::fabs(g - rq[(size_t) i]));
                    ef = std::max(ef, std::fabs(g - rf[(size_t) i]));
                }
                const double relq = mq > 0 ? eq / mq : eq, relf = mf > 0 ? ef / mf : ef;
                if (!(relq < 1e-3)) {
                    std::printf("  FAIL T=%d token %d: q8-ref error %.3g (max |ref| %.3g)\n", T, t, relq, mq);
                    ++fails;
                }
                tq = std::max(tq, relq);
                tf = std::max(tf, relf);
            }
            worst_q8 = std::max(worst_q8, tq);
            worst_f32 = std::max(worst_f32, tf);
            std::printf("  %2d  %14s  %6zu   %12.3g   %13.3g\n", T, trial == 0 ? "8" : "random 0..8", distinct.size(),
                        tq, tf);
            d.free_all();
        }
    }
    std::printf("worst: q8-ref %.3g, f32-ref %.3g\n", worst_q8, worst_f32);

    // a captured graph replays bitwise equal to direct calls on rewritten buffers
    {
        const int T = 4;
        Dev d;
        d.alloc(T, K);
        d.upload(make_window(rng, T, K, n_slots, 6));
        cudaGraph_t graph;
        cudaGraphExec_t exec;
        CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
        mxfp4_experts(d_slot_ptr, d.slots, d.weights, d.x, d.out, T, K, d.scratch, s);
        CK(cudaStreamEndCapture(s, &graph));
        CK(cudaGraphInstantiate(&exec, graph, 0));
        bool same = true;
        for (int i = 0; i < 3 && same; ++i) {
            d.upload(make_window(rng, T, K, n_slots, 3 + 2 * i));
            std::vector<float> a((size_t) T * E), b((size_t) T * E);
            CK(cudaMemset(d.out, 0, a.size() * 4));
            CK(cudaGraphLaunch(exec, s));
            CK(cudaStreamSynchronize(s));
            CK(cudaMemcpy(a.data(), d.out, a.size() * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemset(d.out, 0, b.size() * 4));
            mxfp4_experts(d_slot_ptr, d.slots, d.weights, d.x, d.out, T, K, d.scratch, s);
            CK(cudaStreamSynchronize(s));
            CK(cudaMemcpy(b.data(), d.out, b.size() * 4, cudaMemcpyDeviceToHost));
            same = std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
        }
        std::printf("graph replay with rewritten buffers: %s\n", same ? "bitwise equal" : "DIFFERS");
        fails += !same;
        cudaGraphExecDestroy(exec);
        cudaGraphDestroy(graph);
        d.free_all();
    }
    CK(cudaStreamDestroy(s));
    (void) model;
    return fails;
}

struct Scenario {
    const char* name;
    int T, resident;
    int pool;   // tokens pick from this many consecutive slots (0: disjoint picks per token)
};

void bench(unsigned long long* d_slot_ptr_unused, const std::vector<void*>& slot_mem, int reps) {
    (void) d_slot_ptr_unused;
    const int n_slots = (int) slot_mem.size(), K = 8, L = 47;
    std::vector<unsigned long long> ptrs(slot_mem.size());
    for (size_t i = 0; i < ptrs.size(); ++i) ptrs[i] = (unsigned long long) slot_mem[i];
    unsigned long long* d_ptr;
    CK(cudaMalloc(&d_ptr, ptrs.size() * 8));
    CK(cudaMemcpy(d_ptr, ptrs.data(), ptrs.size() * 8, cudaMemcpyHostToDevice));
    cudaStream_t s;
    CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    // T > 1: each token picks its 8 from `pool` slots, so consecutive tokens share some (pool 28: ~13.7 distinct
    // of 16 for T = 2, what two consecutive tokens pick on coding text, docs/MIMO_ROUTING.md)
    const Scenario sc[] = {{"T=1, 8 resident", 1, 8, 0},  {"T=1, 6 resident", 1, 6, 0},
                           {"T=2, 8 resident", 2, 8, 28}, {"T=2, 6 resident", 2, 6, 28},
                           {"T=4, 8 resident", 4, 8, 40}, {"T=4, 6 resident", 4, 6, 40},
                           {"T=8, 8 resident", 8, 8, 64}};
    std::printf("\nbench: one call = one layer's resident experts; %d calls per graph, %d slots cycled; %d reps\n", L,
                n_slots, reps);
    std::printf("  %-18s %8s %10s %10s %10s %12s\n", "case", "experts", "best us", "median us", "worst us", "best GB/s");
    std::mt19937 rng(99);
    for (const Scenario& c : sc) {
        std::vector<Dev> dv((size_t) L);
        double bytes = 0;
        int cursor = 0;
        for (int l = 0; l < L; ++l) {
            Window w;
            if (c.pool == 0) {   // T = 1: the next 8 slots of the cycle
                w = make_window(rng, c.T, K, n_slots, c.resident, cursor, K);
                cursor = (cursor + K) % n_slots;
            } else {
                w = make_window(rng, c.T, K, n_slots, c.resident, cursor, c.pool);
                cursor = (cursor + c.pool) % n_slots;
            }
            std::set<int> distinct;
            for (int v : w.slots)
                if (v >= 0) distinct.insert(v);
            bytes += (double) distinct.size() * kMxSlotBytes;
            dv[(size_t) l].alloc(c.T, K);
            dv[(size_t) l].upload(w);
            CK(cudaMemset(dv[(size_t) l].out, 0, sizeof(float) * c.T * E));
        }
        cudaGraph_t graph;
        cudaGraphExec_t exec;
        CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
        for (auto& d : dv) mxfp4_experts(d_ptr, d.slots, d.weights, d.x, d.out, c.T, K, d.scratch, s);
        CK(cudaStreamEndCapture(s, &graph));
        CK(cudaGraphInstantiate(&exec, graph, 0));
        cudaEvent_t e0, e1;
        CK(cudaEventCreate(&e0));
        CK(cudaEventCreate(&e1));
        for (int w = 0; w < 3; ++w) CK(cudaGraphLaunch(exec, s));   // warm-up (clocks, first touch)
        CK(cudaStreamSynchronize(s));
        std::vector<double> us;
        for (int r = 0; r < reps; ++r) {
            CK(cudaEventRecord(e0, s));
            CK(cudaGraphLaunch(exec, s));
            CK(cudaEventRecord(e1, s));
            CK(cudaEventSynchronize(e1));
            float ms = 0;
            CK(cudaEventElapsedTime(&ms, e0, e1));
            us.push_back(ms * 1000.0 / L);
        }
        std::sort(us.begin(), us.end());
        const double per_call_bytes = bytes / L;
        std::printf("  %-18s %8.1f %10.1f %10.1f %10.1f %12.0f\n", c.name, per_call_bytes / kMxSlotBytes, us.front(),
                    us[us.size() / 2], us.back(), per_call_bytes / (us.front() * 1e-6) / 1e9);
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
        cudaGraphExecDestroy(exec);
        cudaGraphDestroy(graph);
        for (auto& d : dv) d.free_all();
    }
    cudaFree(d_ptr);
    CK(cudaStreamDestroy(s));
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <MiMo GGUF, any shard> [--bench] [--reps N]\n", argv[0]);
        return 2;
    }
    bool do_bench = false;
    int reps = 20;
    for (int i = 2; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--bench")) do_bench = true;
        else if (!std::strcmp(argv[i], "--reps") && i + 1 < argc) reps = std::atoi(argv[++i]);
    }
    try {
        MimoModel model(argv[1]);
        const MimoGeometry& g = model.geometry();
        if (g.n_embd != E || g.n_ff_exp != F || model.expert_bytes() != kMxSlotBytes) {
            std::fprintf(stderr, "unexpected expert shape\n");
            return 1;
        }
        // 16 experts: layers 1, 24, 47
        const int layers[3] = {1, 24, 47};
        const int per[3] = {6, 5, 5};
        std::mt19937 rng(7);
        std::vector<HostExpert> ex;
        const auto* tr = ggml_get_type_traits(GGML_TYPE_MXFP4);
        for (int li = 0; li < 3; ++li) {
            const ExpertSlices& sl = model.experts(layers[li]);
            for (int j = 0; j < per[li]; ++j) {
                HostExpert h;
                const int e = (int) (rng() % (unsigned) g.n_expert);
                for (int m = 0; m < 3; ++m) {
                    h.m[m] = sl.matrix(m, e);
                    h.w[m].resize((size_t) E * F);
                    tr->to_float(h.m[m], h.w[m].data(), (int64_t) E * F);
                }
                ex.push_back(std::move(h));
            }
        }
        const int n_real = (int) ex.size();
        // slots: host re-layout of each, then check the device re-layout against it
        std::vector<uint8_t> blob(kMxSlotBytes), raw(kMxSlotBytes), back(kMxSlotBytes);
        void* d_raw;
        CK(cudaMalloc(&d_raw, kMxSlotBytes));
        const int n_slots = do_bench ? 192 : n_real;
        std::vector<void*> slot_mem((size_t) n_slots);
        for (auto& p : slot_mem) CK(cudaMalloc(&p, kMxSlotBytes));
        bool relayout_ok = true;
        for (int i = 0; i < n_real; ++i) {
            mxfp4_slot_from_gguf(ex[(size_t) i].m[0], ex[(size_t) i].m[1], ex[(size_t) i].m[2], blob.data());
            CK(cudaMemcpy(slot_mem[(size_t) i], blob.data(), kMxSlotBytes, cudaMemcpyHostToDevice));
            if (i < 3) {
                std::memcpy(raw.data(), ex[(size_t) i].m[0], kMxGuCodes + kMxGuScales);
                std::memcpy(raw.data() + kMxGuCodes + kMxGuScales, ex[(size_t) i].m[1], kMxGuCodes + kMxGuScales);
                std::memcpy(raw.data() + 2 * (kMxGuCodes + kMxGuScales), ex[(size_t) i].m[2],
                            kMxDownCodes + kMxDownScales);
                CK(cudaMemcpy(d_raw, raw.data(), kMxSlotBytes, cudaMemcpyHostToDevice));
                mxfp4_slot_relayout(d_raw, slot_mem[(size_t) i], nullptr);
                CK(cudaMemcpy(back.data(), slot_mem[(size_t) i], kMxSlotBytes, cudaMemcpyDeviceToHost));
                relayout_ok &= std::memcmp(back.data(), blob.data(), kMxSlotBytes) == 0;
            }
        }
        {   // what filling a slot costs on top of the copy: the host re-layout, and the device kernel
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < 4; ++i) mxfp4_slot_from_gguf(ex[0].m[0], ex[0].m[1], ex[0].m[2], blob.data());
            const double host_ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / 4;
            cudaEvent_t e0, e1;
            CK(cudaEventCreate(&e0));
            CK(cudaEventCreate(&e1));
            float best = 1e9f;
            for (int i = 0; i < 5; ++i) {
                CK(cudaEventRecord(e0));
                mxfp4_slot_relayout(d_raw, slot_mem[0], nullptr);
                CK(cudaEventRecord(e1));
                CK(cudaEventSynchronize(e1));
                float ms = 0;
                CK(cudaEventElapsedTime(&ms, e0, e1));
                best = std::min(best, ms);
            }
            std::printf("re-layout of one expert: host %.2f ms (one thread), device %.1f us\n", host_ms, best * 1000.0f);
            cudaEventDestroy(e0);
            cudaEventDestroy(e1);
            CK(cudaMemcpy(slot_mem[0], blob.data(), kMxSlotBytes, cudaMemcpyHostToDevice));   // slot 0 = expert 0 again
        }
        for (int i = n_real; i < n_slots; ++i)
            CK(cudaMemcpy(slot_mem[(size_t) i], slot_mem[(size_t) (i % n_real)], kMxSlotBytes, cudaMemcpyDeviceToDevice));
        cudaFree(d_raw);
        std::printf("device re-layout vs host re-layout (3 experts): %s\n", relayout_ok ? "identical" : "DIFFERS");
        std::vector<unsigned long long> ptrs((size_t) n_real);
        for (int i = 0; i < n_real; ++i) ptrs[(size_t) i] = (unsigned long long) slot_mem[(size_t) i];
        unsigned long long* d_slot_ptr;
        CK(cudaMalloc(&d_slot_ptr, ptrs.size() * 8));
        CK(cudaMemcpy(d_slot_ptr, ptrs.data(), ptrs.size() * 8, cudaMemcpyHostToDevice));
        int fails = relayout_ok ? 0 : 1;
        fails += parity(model, d_slot_ptr, ex, n_real);
        if (do_bench) bench(d_slot_ptr, slot_mem, reps);
        cudaFree(d_slot_ptr);
        for (auto p : slot_mem) cudaFree(p);
        std::printf("%s\n", fails ? "FAIL" : "PASS");
        return fails ? 1 : 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
