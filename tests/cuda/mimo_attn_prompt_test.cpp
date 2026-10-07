// mimo_attn_prompt_test - MiMo prompt (prefill) attention (src/mimo/attn.cu: mimo_qkv_rope_append_prompt,
// mimo_attn_prompt) against a CPU reference of llama.cpp's formula (src/models/mimo2.cpp; see mimo_attn_test.cpp),
// and against the decode path it shares the cache with.
//
// Synthetic data, no model:
//   cases     P = 1, 7, 128, 129, 1000, 2048 tokens at pos0 = 0 and pos0 = 3000 (the prior context written by the
//             decode path's append, 8 tokens at a time), global and SWA layers (sinks). The reference runs in double
//             precision on the cache contents (int8 codes x fp16 scales, read back from the GPU) and the GPU's fp16
//             roped q - what the kernel should compute up to its f16 roundings of K, P and V. Checked too: the roped
//             q against the CPU rope, the chunk's codes against the CPU quantizer, and on SWA layers the ring after
//             the chunk (its last 256 positions, the older slots untouched).
//   chunked   one sequence processed as one 2048-token chunk, as 16 chunks of 128, and as 256 decode windows of 8
//             (mimo_qkv_rope_append + mimo_attn_decode): the same outputs to rounding, the same cache bytes.
//
//   mimo_attn_prompt_test           correctness (exit 1 on failure)
//   mimo_attn_prompt_test --bench   per-layer timings, P = 512, 2048, 4096 at pos0 = 0, 32K, 128K
//   mimo_attn_prompt_test --bench P pos0 swa   one of them (for a profiler)
#include "strata/kernels/f16_bits.hpp"
#include "strata/mimo/attn.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
using namespace strata::mimo;
using strata::kernels::f16_from_f32;
using strata::kernels::f32_from_f16;

constexpr int H = 64, DK = 192, DV = 128, NSWA = 128;

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
    std::vector<T> down(size_t count = 0) const {
        std::vector<T> v(count ? count : n);
        ck(cudaMemcpy(v.data(), p, v.size() * sizeof(T), cudaMemcpyDeviceToHost), "down");
        return v;
    }
};

// ---- CPU reference pieces (as mimo_attn_test) -------------------------------------------------------------------
void rope_ref(float* x, int pos, float theta_scale) {
    float y[64];
    for (int j = 0; j < 32; ++j) {
        const float theta = (float) pos * powf(theta_scale, (float) j);
        const float cs = cosf(theta), sn = sinf(theta);
        y[j] = x[j] * cs - x[j + 32] * sn;
        y[j + 32] = x[j] * sn + x[j + 32] * cs;
    }
    std::memcpy(x, y, sizeof(y));
}

void quant_ref(const float* x, int n, int8_t* codes, uint16_t* scales) {   // kv_q8's rule
    for (int g = 0; g < n / 64; ++g) {
        float amax = 0;
        for (int i = 0; i < 64; ++i) amax = std::max(amax, std::fabs(x[g * 64 + i]));
        const uint16_t sb = f16_from_f32(amax / 127.0f);
        const float sf = f32_from_f16(sb);
        scales[g] = sb;
        for (int i = 0; i < 64; ++i) {
            int c = 0;
            if (sf > 0) c = std::clamp((int) std::nearbyint(x[g * 64 + i] / sf), -127, 127);
            codes[g * 64 + i] = (int8_t) c;
        }
    }
}

int qkv_rows(int n_kv) { return DK * (H + n_kv) + DV * n_kv; }

std::vector<float> random_qkv(int T, int n_kv, std::mt19937& rng) {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> x((size_t) T * qkv_rows(n_kv));
    for (int t = 0; t < T; ++t) {
        float* r = x.data() + (size_t) t * qkv_rows(n_kv);
        for (int hq = 0; hq < H; ++hq) {
            const float sc = 0.5f + 2.5f * (float) ((hq * 37) % 64) / 63.0f;
            for (int i = 0; i < DK; ++i) r[hq * DK + i] = sc * nd(rng);
        }
        for (int h = 0; h < n_kv; ++h)
            for (int i = 0; i < DK; ++i) r[(H + h) * DK + i] = (i < 64 ? 2.0f : 0.5f) * nd(rng);
        for (int i = 0; i < DV * n_kv; ++i) r[(H + n_kv) * DK + i] = nd(rng);
    }
    return x;
}

void split_rope_ref(const std::vector<float>& qkv, int T, int n_kv, int pos0, float theta_scale, std::vector<float>& q,
                    std::vector<float>& k, std::vector<float>& v) {
    q.assign((size_t) T * H * DK, 0);
    k.assign((size_t) T * n_kv * DK, 0);
    v.assign((size_t) T * n_kv * DV, 0);
    for (int t = 0; t < T; ++t) {
        const float* r = qkv.data() + (size_t) t * qkv_rows(n_kv);
        std::memcpy(q.data() + (size_t) t * H * DK, r, sizeof(float) * H * DK);
        std::memcpy(k.data() + (size_t) t * n_kv * DK, r + H * DK, sizeof(float) * n_kv * DK);
        std::memcpy(v.data() + (size_t) t * n_kv * DV, r + (H + n_kv) * DK, sizeof(float) * n_kv * DV);
        for (int hq = 0; hq < H; ++hq) rope_ref(q.data() + ((size_t) t * H + hq) * DK, pos0 + t, theta_scale);
        for (int h = 0; h < n_kv; ++h) rope_ref(k.data() + ((size_t) t * n_kv + h) * DK, pos0 + t, theta_scale);
    }
}

// A cache (or the staging) copied to the host, raw.
struct HostKv {
    int n_kv = 0, cap = 0;
    std::vector<int8_t> k, v;
    std::vector<uint16_t> ks, vs;
    explicit HostKv(const AttnKv& kv) : n_kv(kv.n_kv), cap(kv.cap) {
        const size_t cells = (size_t) n_kv * cap;
        k.resize(cells * DK);
        v.resize(cells * DV);
        ks.resize(cells * 4);
        vs.resize(cells * 2);
        ck(cudaMemcpy(k.data(), kv.k, k.size(), cudaMemcpyDeviceToHost), "down k");
        ck(cudaMemcpy(v.data(), kv.v, v.size(), cudaMemcpyDeviceToHost), "down v");
        ck(cudaMemcpy(ks.data(), kv.k_scale, ks.size() * 2, cudaMemcpyDeviceToHost), "down ks");
        ck(cudaMemcpy(vs.data(), kv.v_scale, vs.size() * 2, cudaMemcpyDeviceToHost), "down vs");
    }
    void deq(int row, int h, float* kk, float* vv) const {
        const size_t s = (size_t) h * cap + row;
        for (int i = 0; i < DK; ++i) kk[i] = (float) k[s * DK + i] * f32_from_f16(ks[s * 4 + i / 64]);
        for (int i = 0; i < DV; ++i) vv[i] = (float) v[s * DV + i] * f32_from_f16(vs[s * 2 + i / 64]);
    }
    bool same_row(int row, const HostKv& o, int orow, int h) const {   // codes and the used scales, bytewise
        const size_t a = (size_t) h * cap + row, b = (size_t) h * o.cap + orow;
        return std::memcmp(&k[a * DK], &o.k[b * DK], DK) == 0 && std::memcmp(&v[a * DV], &o.v[b * DV], DV) == 0 &&
               std::memcmp(&ks[a * 4], &o.ks[b * 4], 6) == 0 && std::memcmp(&vs[a * 2], &o.vs[b * 2], 4) == 0;
    }
};

struct KvRef {   // float keys and values by position
    int n_kv = 0;
    std::vector<float> k, v;
    void init(int n, int n_kv_) {
        n_kv = n_kv_;
        k.assign((size_t) n * n_kv * DK, 0.0f);
        v.assign((size_t) n * n_kv * DV, 0.0f);
    }
    float* kp(int p, int h) { return k.data() + ((size_t) p * n_kv + h) * DK; }
    float* vp(int p, int h) { return v.data() + ((size_t) p * n_kv + h) * DV; }
    const float* kp(int p, int h) const { return k.data() + ((size_t) p * n_kv + h) * DK; }
    const float* vp(int p, int h) const { return v.data() + ((size_t) p * n_kv + h) * DV; }
};

// ggml soft_max with sinks for the listed chunk tokens (rows of q), in parallel over tokens. out [tokens][H][128].
void attend_ref(const std::vector<float>& q, const KvRef& kv, int pos0, const std::vector<int>& tokens, bool swa,
                const float* sinks, std::vector<double>& out) {
    const int G = H / kv.n_kv;
    const double scale = 1.0 / std::sqrt((double) DK);
    out.assign(tokens.size() * H * DV, 0.0);
    auto work = [&](size_t i0, size_t i1) {
        std::vector<double> s;
        for (size_t i = i0; i < i1; ++i)
            for (int hq = 0; hq < H; ++hq) {
                const int t = tokens[i], h = hq / G, qp = pos0 + t;
                const float* qv = q.data() + ((size_t) t * H + hq) * DK;
                const int lo = swa ? std::max(0, qp - NSWA + 1) : 0;
                s.assign((size_t) (qp + 1 - lo), 0.0);
                double mx = -INFINITY;
                for (int p = lo; p <= qp; ++p) {
                    const float* kk = kv.kp(p, h);
                    double d = 0;
                    for (int j = 0; j < DK; ++j) d += (double) qv[j] * kk[j];
                    s[(size_t) (p - lo)] = d * scale;
                    mx = std::max(mx, d * scale);
                }
                if (sinks) mx = std::max(mx, (double) sinks[hq]);
                double sum = 0;
                for (auto& x : s) sum += (x = std::exp(x - mx));
                if (sinks) sum += std::exp((double) sinks[hq] - mx);
                double* o = out.data() + (i * H + hq) * DV;
                for (int p = lo; p <= qp; ++p) {
                    const double w = s[(size_t) (p - lo)] / sum;
                    const float* vv = kv.vp(p, h);
                    for (int j = 0; j < DV; ++j) o[j] += w * vv[j];
                }
            }
    };
    const size_t nt = std::max(1u, std::min(32u, std::thread::hardware_concurrency()));
    std::vector<std::thread> th;
    for (size_t w = 0; w < nt; ++w)
        th.emplace_back(work, tokens.size() * w / nt, tokens.size() * (w + 1) / nt);
    for (auto& t : th) t.join();
}

struct Err {
    double max_abs = 0, rms_ref = 0;
    double rel() const { return rms_ref > 0 ? max_abs / rms_ref : max_abs; }
};
template <class A, class B>
Err compare(const A* got, const B* ref, size_t n) {
    Err e;
    double ss = 0;
    for (size_t i = 0; i < n; ++i) {
        e.max_abs = std::max(e.max_abs, std::fabs((double) got[i] - (double) ref[i]));
        ss += (double) ref[i] * ref[i];
    }
    e.rms_ref = std::sqrt(ss / (double) std::max<size_t>(1, n));
    return e;
}

int g_fail = 0;
void verdict(bool ok, const char* what) {
    if (!ok) {
        ++g_fail;
        std::printf("  FAIL: %s\n", what);
    }
}

constexpr double kTolKernel = 0.02;   // as mimo_attn_test: f16 roundings are well under 1% of the rms, a bug is O(100%)

// ---- one layer: cache, sinks, constants ------------------------------------------------------------------------
struct Layer {
    bool swa;
    int n_kv, cap;
    AttnLayer L;
    AttnKv kv;
    Dev<float> sinks{H};
    std::vector<float> sinks_h;
    Layer(bool swa_, int cap_, uint32_t sink_seed) : swa(swa_), n_kv(swa_ ? 8 : 4), cap(swa_ ? kSwaRing : cap_) {
        std::string err;
        if (!attn_kv_alloc(kv, n_kv, cap, swa, err)) throw std::runtime_error(err);
        std::mt19937 rng(sink_seed);
        std::uniform_real_distribution<float> u(-2.0f, 6.0f);
        sinks_h.resize(H);
        for (auto& x : sinks_h) x = u(rng);
        sinks.up(sinks_h);
        L = attn_layer(H, n_kv, swa, NSWA, swa ? 10000.0f : 10000000.0f, DK, swa ? sinks.p : nullptr);
    }
    ~Layer() { attn_kv_free(kv); }
    int slot(int p) const { return swa ? p % cap : p; }
};

// the decode path's append for positions [from, to), windows of 8, from the given qkv rows (token t = position)
void decode_append(Layer& ly, const std::vector<float>& qkv, int from, int to) {
    Dev<float> d_qkv((size_t) kAttnMaxT * qkv_rows(ly.n_kv)), d_q((size_t) kAttnMaxT * H * DK);
    Dev<int32_t> d_pos(1);
    for (int p = from; p < to; p += kAttnMaxT) {
        const int T = std::min(kAttnMaxT, to - p);
        ck(cudaMemcpy(d_qkv.p, qkv.data() + (size_t) p * qkv_rows(ly.n_kv), sizeof(float) * T * qkv_rows(ly.n_kv),
                      cudaMemcpyHostToDevice), "up qkv");
        ck(cudaMemcpy(d_pos.p, &p, 4, cudaMemcpyHostToDevice), "up pos");
        mimo_qkv_rope_append(d_qkv.p, d_pos.p, T, ly.L, ly.kv, d_q.p, nullptr);
    }
    ck(cudaDeviceSynchronize(), "decode append");
}

std::vector<int> sample_tokens(int P) {
    std::vector<int> t;
    if (P <= 160) {
        for (int i = 0; i < P; ++i) t.push_back(i);
        return t;
    }
    for (int i = 0; i < 4; ++i) t.push_back(i);
    for (int i = 125; i < 132; ++i) t.push_back(i);
    for (int i = P - 4; i < P; ++i) t.push_back(i);
    for (int i = 0; i < 80; ++i) t.push_back(4 + (int) ((long long) i * (P - 8) / 80));
    std::sort(t.begin(), t.end());
    t.erase(std::unique(t.begin(), t.end()), t.end());
    return t;
}

// ---- a chunk of P tokens at pos0 against the reference ----------------------------------------------------------
void run_case(bool swa, int P, int pos0, const AttnPromptScratch& ps, uint32_t seed) {
    std::mt19937 rng(seed);
    Layer ly(swa, pos0 + P, seed * 7 + 1);
    const int n_kv = ly.n_kv, n = pos0 + P;
    const std::vector<float> prior = random_qkv(pos0, n_kv, rng);
    decode_append(ly, prior, 0, pos0);
    const HostKv before(ly.kv);

    const std::vector<float> qkv = random_qkv(P, n_kv, rng);
    Dev<float> d_qkv(qkv.size()), d_out((size_t) P * H * DV);
    Dev<uint16_t> d_q((size_t) P * H * DK);
    d_qkv.up(qkv);
    mimo_qkv_rope_append_prompt(d_qkv.p, pos0, P, ly.L, ly.kv, ps, d_q.p, nullptr);
    mimo_attn_prompt(d_q.p, pos0, P, ly.L, ly.kv, ps, d_out.p, nullptr);
    ck(cudaDeviceSynchronize(), "run");
    const std::vector<uint16_t> qh = d_q.down();
    std::vector<float> q_gpu(qh.size());
    for (size_t i = 0; i < qh.size(); ++i) q_gpu[i] = f32_from_f16(qh[i]);
    const std::vector<float> out = d_out.down();

    std::vector<float> q_cpu, k_cpu, v_cpu;
    split_rope_ref(qkv, P, n_kv, pos0, ly.L.theta_scale, q_cpu, k_cpu, v_cpu);
    // the fp16 q against the CPU rope: what is left after the fp16 rounding's half ulp, relative to the rms (the
    // decode test's measure and tolerance: the GPU's powf / sincosf move theta by up to pos x 6e-8)
    double q_excess = 0, q_ss = 0;
    long long q_same = 0;
    for (size_t i = 0; i < qh.size(); ++i) {
        const float r16 = f32_from_f16(f16_from_f32(q_cpu[i]));
        const float ulp = std::fabs(f32_from_f16((uint16_t) (f16_from_f32(std::fabs(q_cpu[i])) | 1u)) -
                                    f32_from_f16((uint16_t) (f16_from_f32(std::fabs(q_cpu[i])) & ~1u)));
        q_excess = std::max(q_excess, (double) std::fabs(q_gpu[i] - q_cpu[i]) - 0.5 * ulp);
        q_ss += (double) q_cpu[i] * q_cpu[i];
        q_same += q_gpu[i] == r16 ? 1 : 0;
    }
    q_excess /= std::sqrt(q_ss / (double) qh.size());

    // the keys and values as the kernel should see them: the prior context from the cache before the chunk, the
    // chunk's from where the attention reads them (global: the cache; SWA: the staging)
    const HostKv after(ly.kv);
    KvRef q8;
    q8.init(n, n_kv);
    const int prior_lo = swa ? std::max(0, pos0 - NSWA) : 0;
    for (int p = prior_lo; p < pos0; ++p)
        for (int h = 0; h < n_kv; ++h) before.deq(ly.slot(p), h, q8.kp(p, h), q8.vp(p, h));
    long long codes_off = 0, ring_bad = 0;
    {
        const HostKv stage(ps.swa);
        for (int t = 0; t < P; ++t)
            for (int h = 0; h < n_kv; ++h) {
                const int p = pos0 + t;
                if (swa) stage.deq(ps.n_swa + t, h, q8.kp(p, h), q8.vp(p, h));
                else after.deq(p, h, q8.kp(p, h), q8.vp(p, h));
                int8_t kc[DK], vc[DV];
                uint16_t ks[4], vs[2];
                quant_ref(k_cpu.data() + ((size_t) t * n_kv + h) * DK, DK, kc, ks);
                quant_ref(v_cpu.data() + ((size_t) t * n_kv + h) * DV, DV, vc, vs);
                for (int i = 0; i < DK; ++i) {
                    const float r = (float) kc[i] * f32_from_f16(ks[i / 64]), g = q8.kp(p, h)[i];
                    codes_off += std::fabs(g - r) > 1e-6f * (1 + std::fabs(g)) ? 1 : 0;
                }
                for (int i = 0; i < DV; ++i) codes_off += q8.vp(p, h)[i] != (float) vc[i] * f32_from_f16(vs[i / 64]) ? 1 : 0;
                // SWA: the ring holds the chunk's last 256 positions (bytewise the staging's rows) and, below them,
                // the older positions untouched
                if (swa && p >= n - kSwaRing) ring_bad += after.same_row(ly.slot(p), stage, ps.n_swa + t, h) ? 0 : 1;
            }
        if (swa)
            for (int p = std::max(0, n - kSwaRing); p < pos0; ++p)
                for (int h = 0; h < n_kv; ++h) ring_bad += after.same_row(ly.slot(p), before, ly.slot(p), h) ? 0 : 1;
    }
    const std::vector<int> tok = sample_tokens(P);
    std::vector<double> ref;
    attend_ref(q_gpu, q8, pos0, tok, swa, swa ? ly.sinks_h.data() : nullptr, ref);
    std::vector<float> got(tok.size() * H * DV);
    for (size_t i = 0; i < tok.size(); ++i)
        std::memcpy(&got[i * H * DV], &out[(size_t) tok[i] * H * DV], sizeof(float) * H * DV);
    const Err e = compare(got.data(), ref.data(), ref.size());
    std::printf("%-6s P=%4d pos0=%4d | fp16 q vs CPU rope: %.3f%% equal, rest %.1e | codes off %lld | ring rows off "
                "%lld | kernel vs ref (%zu tokens): max %.2e rel %.2e\n",
                swa ? "swa" : "global", P, pos0, 100.0 * (double) q_same / (double) qh.size(), q_excess, codes_off,
                ring_bad, tok.size(), e.max_abs, e.rel());
    verdict(q_excess < 2e-3, "rope");
    verdict(codes_off <= (long long) P * n_kv * 2, "cache codes");
    verdict(ring_bad == 0, "SWA ring after the chunk");
    verdict(e.rel() < kTolKernel && std::isfinite(e.max_abs), "kernel vs reference");
}

// ---- one sequence three ways: one chunk, chunks of 128, decode windows of 8 -------------------------------------
void run_chunked(bool swa, int prior_n, int N, const AttnPromptScratch& ps, uint32_t seed) {
    std::mt19937 rng(seed);
    const int n_kv = swa ? 8 : 4, total = prior_n + N;
    const std::vector<float> qkv = random_qkv(total, n_kv, rng);   // token t = position t
    Dev<float> d_qkv(qkv.size());
    d_qkv.up(qkv);
    const size_t rows = (size_t) qkv_rows(n_kv);
    std::vector<std::vector<float>> outs;
    std::vector<HostKv> caches;
    const char* names[3] = {"1 x 2048", "16 x 128", "decode 8"};
    for (int way = 0; way < 3; ++way) {
        Layer ly(swa, total, seed * 7 + 1);   // the same sinks each time
        decode_append(ly, qkv, 0, prior_n);
        Dev<float> d_out((size_t) N * H * DV);
        if (way < 2) {
            const int chunk = way == 0 ? N : 128;
            Dev<uint16_t> d_q((size_t) chunk * H * DK);
            for (int t = 0; t < N; t += chunk) {
                const int P = std::min(chunk, N - t);
                mimo_qkv_rope_append_prompt(d_qkv.p + (size_t) (prior_n + t) * rows, prior_n + t, P, ly.L, ly.kv, ps,
                                            d_q.p, nullptr);
                mimo_attn_prompt(d_q.p, prior_n + t, P, ly.L, ly.kv, ps, d_out.p + (size_t) t * H * DV, nullptr);
            }
        } else {
            AttnScratch scr;
            std::string err;
            if (!attn_scratch_alloc(scr, 0, err)) throw std::runtime_error(err);
            Dev<float> d_q((size_t) kAttnMaxT * H * DK);
            Dev<int32_t> d_pos(N / kAttnMaxT + 1);
            std::vector<int32_t> pos;
            for (int t = 0; t < N; t += kAttnMaxT) pos.push_back(prior_n + t);
            ck(cudaMemcpy(d_pos.p, pos.data(), pos.size() * 4, cudaMemcpyHostToDevice), "pos");
            for (int t = 0, w = 0; t < N; t += kAttnMaxT, ++w) {
                const int T = std::min(kAttnMaxT, N - t);
                mimo_qkv_rope_append(d_qkv.p + (size_t) (prior_n + t) * rows, d_pos.p + w, T, ly.L, ly.kv, d_q.p, nullptr);
                mimo_attn_decode(d_q.p, d_pos.p + w, T, ly.L, ly.kv, scr, d_out.p + (size_t) t * H * DV, nullptr);
            }
            ck(cudaDeviceSynchronize(), "decode");
            attn_scratch_free(scr);
        }
        ck(cudaDeviceSynchronize(), "run");
        outs.push_back(d_out.down());
        caches.emplace_back(ly.kv);
    }
    const size_t n_out = (size_t) N * H * DV;
    const Err e01 = compare(outs[1].data(), outs[0].data(), n_out), e02 = compare(outs[2].data(), outs[0].data(), n_out);
    // every position the cache still holds: global all of them, SWA the ring's 256 (positions >= total - 256)
    long long diff1 = 0, diff2 = 0;
    for (int p = swa ? total - kSwaRing : 0; p < total; ++p)
        for (int h = 0; h < n_kv; ++h) {
            const int s = swa ? p % kSwaRing : p;
            diff1 += caches[1].same_row(s, caches[0], s, h) ? 0 : 1;
            diff2 += caches[2].same_row(s, caches[0], s, h) ? 0 : 1;
        }
    std::printf("%-6s chunked, prior %d + %d tokens | outputs vs %s: %s rel %.2e, %s rel %.2e | cache rows that differ: "
                "%lld, %lld\n",
                swa ? "swa" : "global", prior_n, N, names[0], names[1], e01.rel(), names[2], e02.rel(), diff1, diff2);
    // prompt against prompt: the same roundings, only the split-K boundaries move; prompt against decode: two
    // independent sets of f16 roundings (decode keeps K exact, the prompt kernel rounds it), each up to ~kTolKernel / 2
    verdict(e01.rel() < kTolKernel && e02.rel() < 2 * kTolKernel, "chunked outputs");
    verdict(diff1 == 0 && diff2 == 0, "chunked cache contents");
}

// ---- benchmark --------------------------------------------------------------------------------------------------
void fill_random(const AttnKv& kv, std::mt19937& rng) {
    const size_t cells = (size_t) kv.n_kv * kv.cap;
    std::vector<int8_t> codes(cells * DK);
    for (auto& c : codes) c = (int8_t) ((int) (rng() % 255) - 127);
    ck(cudaMemcpy(kv.k, codes.data(), cells * DK, cudaMemcpyHostToDevice), "k");
    ck(cudaMemcpy(kv.v, codes.data(), cells * DV, cudaMemcpyHostToDevice), "v");
    std::vector<uint16_t> sc(cells * 4, f16_from_f32(0.02f));
    ck(cudaMemcpy(kv.k_scale, sc.data(), cells * 8, cudaMemcpyHostToDevice), "ks");
    ck(cudaMemcpy(kv.v_scale, sc.data(), cells * 4, cudaMemcpyHostToDevice), "vs");
}

// keys attended over a chunk, summed over its queries (the FLOPs are 2 x 64 heads x (192 + 128) x this)
double key_count(bool swa, int pos0, int P) {
    double n = 0;
    for (int t = 0; t < P; ++t) n += swa ? std::min(NSWA, pos0 + t + 1) : pos0 + t + 1;
    return n;
}

void bench(const AttnPromptScratch& ps, int only_P, int only_pos0, int only_swa) {
    cudaDeviceProp prop{};
    ck(cudaGetDeviceProperties(&prop, 0), "props");
    std::printf("device: %s, %d SMs\n", prop.name, prop.multiProcessorCount);
    constexpr int kMaxP = 4096, kMaxPos = 131072;
    std::mt19937 rng(7);
    Layer glob(false, kMaxPos + kMaxP, 1), swa(true, 0, 2);
    fill_random(glob.kv, rng);
    fill_random(swa.kv, rng);
    Dev<float> d_qkv((size_t) kMaxP * qkv_rows(8)), d_out((size_t) kMaxP * H * DV);
    Dev<uint16_t> d_q((size_t) kMaxP * H * DK);
    {
        std::vector<float> x = random_qkv(kMaxP, 8, rng);
        d_qkv.up(x);
    }
    cudaEvent_t e0, e1;
    ck(cudaEventCreate(&e0), "event");
    ck(cudaEventCreate(&e1), "event");
    auto time_us = [&](auto&& call) {   // best of 5 after a warm-up
        double best = 1e30;
        for (int rep = 0; rep < 6; ++rep) {
            ck(cudaEventRecord(e0, nullptr), "rec");
            call();
            ck(cudaEventRecord(e1, nullptr), "rec");
            ck(cudaEventSynchronize(e1), "sync");
            float ms = 0;
            ck(cudaEventElapsedTime(&ms, e0, e1), "elapsed");
            if (rep > 0) best = std::min(best, ms * 1000.0);
        }
        return best;
    };
    std::printf("\nper layer, best of 5: append = split + rope + int8 append (SWA: + ring copy), attention = "
                "mimo_attn_prompt\n");
    std::printf("%-7s %5s %7s %12s %14s %8s\n", "layer", "P", "pos0", "append us", "attention us", "TFLOPS");
    double sum_g = 0, sum_s = 0;
    for (int P : {512, 2048, 4096})
        for (int pos0 : {0, 32768, 131072})
            for (int is_swa = 0; is_swa < 2; ++is_swa) {
                if (only_P > 0 && (P != only_P || pos0 != only_pos0 || is_swa != only_swa)) continue;
                Layer& ly = is_swa ? swa : glob;
                const double ta = time_us([&] {
                    mimo_qkv_rope_append_prompt(d_qkv.p, pos0, P, ly.L, ly.kv, ps, d_q.p, nullptr);
                });
                const double tt = time_us([&] { mimo_attn_prompt(d_q.p, pos0, P, ly.L, ly.kv, ps, d_out.p, nullptr); });
                const double flop = 2.0 * H * (DK + DV) * key_count(is_swa != 0, pos0, P);
                std::printf("%-7s %5d %7d %12.1f %14.1f %8.1f\n", is_swa ? "swa" : "global", P, pos0, ta, tt,
                            flop / (tt * 1e-6) / 1e12);
                if (P == 2048 && pos0 == 0) (is_swa ? sum_s : sum_g) = ta + tt;
            }
    std::printf("\nP = 2048 at pos0 = 0, all 48 layers (9 global + 39 SWA), append + attention: %.2f ms\n",
                (9 * sum_g + 39 * sum_s) / 1000.0);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
}

}  // namespace

int main(int argc, char** argv) {
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev == 0) {
        std::printf("no CUDA device: skipped\n");
        return 77;
    }
    try {
        AttnPromptScratch ps;
        std::string err;
        if (!attn_prompt_scratch_alloc(ps, kAttnPromptMaxP, 8, NSWA, 0, err)) throw std::runtime_error(err);
        if (argc > 1 && std::strcmp(argv[1], "--bench") == 0) {
            bench(ps, argc > 4 ? std::atoi(argv[2]) : 0, argc > 4 ? std::atoi(argv[3]) : 0,
                  argc > 4 ? std::atoi(argv[4]) : 0);
            attn_prompt_scratch_free(ps);
            return 0;
        }
        uint32_t seed = 1;
        for (int pos0 : {0, 3000})
            for (int swa = 1; swa >= 0; --swa)
                for (int P : {1, 7, 128, 129, 1000, 2048}) run_case(swa != 0, P, pos0, ps, seed++);
        run_chunked(true, 300, 2048, ps, seed++);
        run_chunked(false, 300, 2048, ps, seed++);
        attn_prompt_scratch_free(ps);
    } catch (const std::exception& e) {
        std::printf("error: %s\n", e.what());
        return 1;
    }
    std::printf(g_fail ? "%d check(s) FAILED\n" : "all checks passed\n", g_fail);
    return g_fail ? 1 : 0;
}
