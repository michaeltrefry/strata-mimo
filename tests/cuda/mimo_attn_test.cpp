// mimo_attn_test - MiMo decode attention (src/mimo/attn.cu) against a CPU reference of llama.cpp's formula
// (src/models/mimo2.cpp: NEOX rope on 64 of 192 dims, scale 1/sqrt(192), GQA, causal + standard SWA mask
// pos_q - pos_k >= 128, ggml soft_max with sinks: max over scores and the sink, exp(sink - max) added to the sum).
//
// Synthetic data, no model. For every case the reference runs twice in double precision:
//   "q8"  - the keys and values exactly as the cache holds them (int8 codes x fp16 scales, read back from the GPU)
//           and the GPU's roped q: what the kernel should compute, up to its own f16 roundings of Q, P and V;
//   "f32" - the unquantized roped keys and values: the difference q8 - f32 is the cost of the int8 KV.
// Plus: the roped q against the CPU rope, the window's cache codes against the CPU quantizer, a speculative
// sequence with rejected tokens (rollback) across the SWA ring's wrap, and the sink identity
// out(sink) = out(no sink) * Z / (Z + exp(sink)), Z = sum_j exp(s_j).
//
//   mimo_attn_test            correctness (exit 1 on failure)
//   mimo_attn_test --bench    per-layer timings: global layer at 4K..256K and an SWA layer, T = 1, 2, 4, 8
//   mimo_attn_test --bench T ctx   one global-layer configuration (for a profiler)
#include "strata/kernels/f16_bits.hpp"
#include "strata/mimo/attn.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
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
    std::vector<T> down() const {
        std::vector<T> v(n);
        ck(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost), "down");
        return v;
    }
};

// ---- CPU reference pieces ------------------------------------------------------------------------------------
void rope_ref(float* x, int pos, float theta_scale) {   // ggml-cuda rope_neox's theta, precise sin/cos
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

// llama.cpp's own KV types for comparison: q8_0 (32 values per fp16 scale, quantize_row_q8_0_ref) and f16
void q8_0_roundtrip(float* x, int n) {
    for (int b = 0; b < n / 32; ++b) {
        float amax = 0;
        for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(x[b * 32 + i]));
        const float d = amax / 127.0f, id = d != 0.0f ? 1.0f / d : 0.0f, ds = f32_from_f16(f16_from_f32(d));
        for (int i = 0; i < 32; ++i) x[b * 32 + i] = roundf(x[b * 32 + i] * id) * ds;
    }
}
void f16_roundtrip(float* x, int n) {
    for (int i = 0; i < n; ++i) x[i] = f32_from_f16(f16_from_f32(x[i]));
}

// One layer's keys and values for every position the window can see, as float vectors.
struct KvRef {
    int n_kv = 0;
    std::vector<float> k, v;   // [pos][n_kv][192], [pos][n_kv][128]
    const float* kp(int p, int h) const { return k.data() + ((size_t) p * n_kv + h) * DK; }
    const float* vp(int p, int h) const { return v.data() + ((size_t) p * n_kv + h) * DV; }
};

// ggml soft_max with sinks (ggml-cpu ops.cpp): max includes the sink, exp(sink - max) joins the sum.
// out [T][H][128]; also returns Z = sum_j exp(s_j - max) and the max per (t, head) for the sink identity.
void attend_ref(const std::vector<float>& q, const KvRef& kv, int pos0, int T, bool swa, const float* sinks,
                std::vector<double>& out, std::vector<double>* zsum = nullptr, std::vector<double>* zmax = nullptr) {
    const int G = H / kv.n_kv;
    const double scale = 1.0 / std::sqrt((double) DK);
    out.assign((size_t) T * H * DV, 0.0);
    if (zsum) zsum->assign((size_t) T * H, 0.0);
    if (zmax) zmax->assign((size_t) T * H, 0.0);
    std::vector<double> s;
    for (int t = 0; t < T; ++t)
        for (int hq = 0; hq < H; ++hq) {
            const int h = hq / G, qp = pos0 + t;
            const float* qv = q.data() + ((size_t) t * H + hq) * DK;
            const int lo = swa ? std::max(0, qp - NSWA + 1) : 0;
            s.assign((size_t) (qp + 1 - lo), 0.0);
            double mx = -INFINITY;
            for (int p = lo; p <= qp; ++p) {
                const float* kk = kv.kp(p, h);
                double d = 0;
                for (int i = 0; i < DK; ++i) d += (double) qv[i] * kk[i];
                s[(size_t) (p - lo)] = d * scale;
                mx = std::max(mx, d * scale);
            }
            if (sinks) mx = std::max(mx, (double) sinks[hq]);
            double sum = 0;
            for (auto& x : s) sum += (x = std::exp(x - mx));
            if (zsum) (*zsum)[(size_t) t * H + hq] = sum;
            if (zmax) (*zmax)[(size_t) t * H + hq] = mx;
            if (sinks) sum += std::exp((double) sinks[hq] - mx);
            double* o = out.data() + ((size_t) t * H + hq) * DV;
            for (int p = lo; p <= qp; ++p) {
                const double w = s[(size_t) (p - lo)] / sum;
                const float* vv = kv.vp(p, h);
                for (int i = 0; i < DV; ++i) o[i] += w * vv[i];
            }
        }
}

struct Err {
    double max_abs = 0, rms_ref = 0;
    double rel() const { return rms_ref > 0 ? max_abs / rms_ref : max_abs; }
};
template <class A, class B>
Err compare(const std::vector<A>& got, const std::vector<B>& ref) {
    Err e;
    double ss = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        e.max_abs = std::max(e.max_abs, std::fabs((double) got[i] - (double) ref[i]));
        ss += (double) ref[i] * ref[i];
    }
    e.rms_ref = std::sqrt(ss / (double) std::max<size_t>(1, ref.size()));
    return e;
}

// ---- one layer on the GPU with a host mirror --------------------------------------------------------------------
struct Layer {
    bool swa;
    int n_kv, cap;
    AttnLayer L;
    AttnKv kv;
    Dev<float> sinks{H};
    std::vector<float> sinks_h;
    Layer(bool swa_, int cap_, std::mt19937& rng) : swa(swa_), n_kv(swa_ ? 8 : 4), cap(swa_ ? kSwaRing : cap_) {
        std::string err;
        if (!attn_kv_alloc(kv, n_kv, cap, swa, err)) throw std::runtime_error(err);
        std::uniform_real_distribution<float> u(-2.0f, 6.0f);
        sinks_h.resize(H);
        for (auto& x : sinks_h) x = u(rng);
        sinks.up(sinks_h);
        L = attn_layer(H, n_kv, swa, NSWA, swa ? 10000.0f : 10000000.0f, DK, swa ? sinks.p : nullptr);
    }
    ~Layer() { attn_kv_free(kv); }
    size_t slot(int p) const { return (size_t) (swa ? p % cap : p); }
    // write a (roped) key and value at position p of head h, quantized on the host
    void put(int p, int h, const float* k, const float* v, std::vector<int8_t>& kc, std::vector<int8_t>& vc,
             std::vector<uint16_t>& ksc, std::vector<uint16_t>& vsc) const {
        const size_t s = (size_t) h * cap + slot(p);
        quant_ref(k, DK, kc.data() + s * DK, ksc.data() + s * 4);
        quant_ref(v, DV, vc.data() + s * DV, vsc.data() + s * 2);
    }
    // the cache as the kernel sees it, dequantized, for positions [0, n)
    void dequant(int n, KvRef& out) const {
        const size_t cells = (size_t) n_kv * cap;
        std::vector<int8_t> kc(cells * DK), vc(cells * DV);
        std::vector<uint16_t> ks(cells * 4), vs(cells * 2);
        ck(cudaMemcpy(kc.data(), kv.k, kc.size(), cudaMemcpyDeviceToHost), "down k");
        ck(cudaMemcpy(vc.data(), kv.v, vc.size(), cudaMemcpyDeviceToHost), "down v");
        ck(cudaMemcpy(ks.data(), kv.k_scale, ks.size() * 2, cudaMemcpyDeviceToHost), "down ks");
        ck(cudaMemcpy(vs.data(), kv.v_scale, vs.size() * 2, cudaMemcpyDeviceToHost), "down vs");
        out.n_kv = n_kv;
        out.k.assign((size_t) n * n_kv * DK, 0.0f);
        out.v.assign((size_t) n * n_kv * DV, 0.0f);
        const int lo = swa ? std::max(0, n - cap) : 0;
        for (int p = lo; p < n; ++p)
            for (int h = 0; h < n_kv; ++h) {
                const size_t s = (size_t) h * cap + slot(p);
                float* kk = out.k.data() + ((size_t) p * n_kv + h) * DK;
                float* vv = out.v.data() + ((size_t) p * n_kv + h) * DV;
                for (int i = 0; i < DK; ++i) kk[i] = (float) kc[s * DK + i] * f32_from_f16(ks[s * 4 + i / 64]);
                for (int i = 0; i < DV; ++i) vv[i] = (float) vc[s * DV + i] * f32_from_f16(vs[s * 2 + i / 64]);
            }
    }
};

int qkv_rows(int n_kv) { return DK * (H + n_kv) + DV * n_kv; }

// random qkv rows for T tokens: q heads with varied scales, k with a larger roped part (exercises the group scales)
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

// the window's roped q, k and raw v on the CPU
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

int g_fail = 0;
void verdict(bool ok, const char* what) {
    if (!ok) {
        ++g_fail;
        std::printf("  FAIL: %s\n", what);
    }
}

// Tolerances. The kernel rounds Q, P and dequantized V to f16 (rel. 2^-11 each): against the q8 reference the max
// error stays well under 1% of the output's rms; a layout or mask bug is O(100%).
constexpr double kTolKernel = 0.02;
constexpr double kTolRope = 2e-3;   // |q_gpu - q_cpu| / rms: a 1-ulp powf difference moves theta by pos * 6e-8

// ---- a single window over a random history -------------------------------------------------------------------
void run_case(bool swa, int T, int ctx, uint32_t seed) {
    std::mt19937 rng(seed);
    const int pos0 = std::max(0, ctx - T), n = pos0 + T;
    Layer ly(swa, n, rng);
    const int n_kv = ly.n_kv;
    std::normal_distribution<float> nd(0.0f, 1.0f);
    // history: random already-roped keys and values, quantized on the host into the cache layout
    KvRef f32;
    f32.n_kv = n_kv;
    f32.k.assign((size_t) n * n_kv * DK, 0);
    f32.v.assign((size_t) n * n_kv * DV, 0);
    {
        const size_t cells = (size_t) n_kv * ly.cap;
        std::vector<int8_t> kc(cells * DK, 0), vc(cells * DV, 0);
        std::vector<uint16_t> ks(cells * 4, 0), vs(cells * 2, 0);
        for (int p = 0; p < pos0; ++p)
            for (int h = 0; h < n_kv; ++h) {
                float* kk = f32.k.data() + ((size_t) p * n_kv + h) * DK;
                float* vv = f32.v.data() + ((size_t) p * n_kv + h) * DV;
                for (int i = 0; i < DK; ++i) kk[i] = (i < 64 ? 2.0f : 0.5f) * nd(rng);
                for (int i = 0; i < DV; ++i) vv[i] = nd(rng);
                ly.put(p, h, kk, vv, kc, vc, ks, vs);
            }
        ck(cudaMemcpy(ly.kv.k, kc.data(), kc.size(), cudaMemcpyHostToDevice), "up k");
        ck(cudaMemcpy(ly.kv.v, vc.data(), vc.size(), cudaMemcpyHostToDevice), "up v");
        ck(cudaMemcpy(ly.kv.k_scale, ks.data(), ks.size() * 2, cudaMemcpyHostToDevice), "up ks");
        ck(cudaMemcpy(ly.kv.v_scale, vs.data(), vs.size() * 2, cudaMemcpyHostToDevice), "up vs");
    }
    const std::vector<float> qkv = random_qkv(T, n_kv, rng);
    Dev<float> d_qkv(qkv.size()), d_q((size_t) T * H * DK), d_out((size_t) T * H * DV);
    Dev<int32_t> d_pos(1);
    d_qkv.up(qkv);
    d_pos.up({pos0});
    AttnScratch scr;
    std::string err;
    if (!attn_scratch_alloc(scr, 0, err)) throw std::runtime_error(err);
    mimo_qkv_rope_append(d_qkv.p, d_pos.p, T, ly.L, ly.kv, d_q.p, nullptr);
    mimo_attn_decode(d_q.p, d_pos.p, T, ly.L, ly.kv, scr, d_out.p, nullptr);
    ck(cudaDeviceSynchronize(), "run");
    const std::vector<float> q_gpu = d_q.down(), out = d_out.down();

    std::vector<float> q_cpu, k_cpu, v_cpu;
    split_rope_ref(qkv, T, n_kv, pos0, ly.L.theta_scale, q_cpu, k_cpu, v_cpu);
    const Err e_rope = compare(q_gpu, q_cpu);
    for (int t = 0; t < T; ++t) {
        std::memcpy(f32.k.data() + (size_t) (pos0 + t) * n_kv * DK, k_cpu.data() + (size_t) t * n_kv * DK, sizeof(float) * n_kv * DK);
        std::memcpy(f32.v.data() + (size_t) (pos0 + t) * n_kv * DV, v_cpu.data() + (size_t) t * n_kv * DV, sizeof(float) * n_kv * DV);
    }
    // the window's cache entries against the host quantizer of the host-roped k and v
    KvRef q8;
    ly.dequant(n, q8);
    int code_mismatch = 0;
    for (int t = 0; t < T; ++t)
        for (int h = 0; h < n_kv; ++h) {
            int8_t kc[DK], vc[DV];
            uint16_t ks[4], vs[2];
            quant_ref(k_cpu.data() + ((size_t) t * n_kv + h) * DK, DK, kc, ks);
            quant_ref(v_cpu.data() + ((size_t) t * n_kv + h) * DV, DV, vc, vs);
            for (int i = 0; i < DK; ++i)
                code_mismatch += std::fabs(q8.kp(pos0 + t, h)[i] - (float) kc[i] * f32_from_f16(ks[i / 64])) > 1e-6f * (1 + std::fabs(q8.kp(pos0 + t, h)[i])) ? 1 : 0;
            for (int i = 0; i < DV; ++i)
                code_mismatch += q8.vp(pos0 + t, h)[i] != (float) vc[i] * f32_from_f16(vs[i / 64]) ? 1 : 0;
        }
    const float* sinks = swa ? ly.sinks_h.data() : nullptr;
    std::vector<double> ref_q8, ref_f32;
    attend_ref(q_gpu, q8, pos0, T, swa, sinks, ref_q8);
    attend_ref(q_cpu, f32, pos0, T, swa, sinks, ref_f32);
    // the same window with llama.cpp's q8_0 and f16 caches, for scale
    KvRef q80 = f32, f16 = f32;
    q8_0_roundtrip(q80.k.data(), (int) q80.k.size());
    q8_0_roundtrip(q80.v.data(), (int) q80.v.size());
    f16_roundtrip(f16.k.data(), (int) f16.k.size());
    f16_roundtrip(f16.v.data(), (int) f16.v.size());
    std::vector<double> ref_q80, ref_f16;
    attend_ref(q_cpu, q80, pos0, T, swa, sinks, ref_q80);
    attend_ref(q_cpu, f16, pos0, T, swa, sinks, ref_f16);
    const Err e_k = compare(out, ref_q8), e_kv = compare(ref_q8, ref_f32), e_tot = compare(out, ref_f32);
    std::printf("%-6s T=%d ctx=%6d | rope %.1e | codes off %d | kernel vs q8 ref: max %.2e rel %.2e | KV error rel "
                "to f32: this int8 %.2e, kernel %.2e, llama q8_0 %.2e, f16 %.2e\n",
                swa ? "swa" : "global", T, n, e_rope.rel(), code_mismatch, e_k.max_abs, e_k.rel(), e_kv.rel(),
                e_tot.rel(), compare(ref_q80, ref_f32).rel(), compare(ref_f16, ref_f32).rel());
    verdict(e_rope.rel() < kTolRope, "rope");
    // a code may differ by one step where the GPU's roped value sits on a rounding boundary
    verdict(code_mismatch <= T * n_kv * 2, "cache codes");
    verdict(e_k.rel() < kTolKernel, "kernel vs q8 reference");
    verdict(std::isfinite(e_k.max_abs), "finite");
    attn_scratch_free(scr);
}

// ---- speculative windows with rejected tokens: rollback and the ring's wrap ------------------------------------
void run_sequence(bool swa, int n_pos, uint32_t seed) {
    std::mt19937 rng(seed);
    Layer ly(swa, n_pos + kAttnMaxT, rng);
    const int n_kv = ly.n_kv;
    AttnScratch scr;
    std::string err;
    if (!attn_scratch_alloc(scr, 0, err)) throw std::runtime_error(err);
    Dev<float> d_qkv((size_t) kAttnMaxT * qkv_rows(n_kv)), d_q((size_t) kAttnMaxT * H * DK), d_out((size_t) kAttnMaxT * H * DV);
    Dev<int32_t> d_pos(1);
    // the accepted history, quantized by the host (independent of the GPU cache)
    KvRef hist;
    hist.n_kv = n_kv;
    hist.k.assign((size_t) (n_pos + kAttnMaxT) * n_kv * DK, 0);
    hist.v.assign((size_t) (n_pos + kAttnMaxT) * n_kv * DV, 0);
    std::uniform_int_distribution<int> ut(1, kAttnMaxT);
    double worst = 0;
    int windows = 0, rejected = 0;
    long long codes_off = 0, codes_total = 0;
    for (int pos0 = 0; pos0 < n_pos;) {
        const int T = ut(rng);
        const std::vector<float> qkv = random_qkv(T, n_kv, rng);
        d_qkv.up(qkv);
        d_pos.up({pos0});
        mimo_qkv_rope_append(d_qkv.p, d_pos.p, T, ly.L, ly.kv, d_q.p, nullptr);
        mimo_attn_decode(d_q.p, d_pos.p, T, ly.L, ly.kv, scr, d_out.p, nullptr);
        ck(cudaDeviceSynchronize(), "run");
        std::vector<float> q, k, v;
        split_rope_ref(qkv, T, n_kv, pos0, ly.L.theta_scale, q, k, v);
        // the window's own entries as the GPU just wrote them (each is snapshotted once, here, and only the
        // accepted ones enter the history): a later window that clobbers a slot still needed shows up as an error
        KvRef snap, cur = hist;
        ly.dequant(pos0 + T, snap);
        for (int t = 0; t < T; ++t)
            for (int h = 0; h < n_kv; ++h) {
                int8_t kc[DK], vc[DV];
                uint16_t ks[4], vs[2];
                quant_ref(k.data() + ((size_t) t * n_kv + h) * DK, DK, kc, ks);
                quant_ref(v.data() + ((size_t) t * n_kv + h) * DV, DV, vc, vs);
                float* kk = cur.k.data() + ((size_t) (pos0 + t) * n_kv + h) * DK;
                float* vv = cur.v.data() + ((size_t) (pos0 + t) * n_kv + h) * DV;
                std::memcpy(kk, snap.kp(pos0 + t, h), sizeof(float) * DK);
                std::memcpy(vv, snap.vp(pos0 + t, h), sizeof(float) * DV);
                for (int i = 0; i < DK; ++i) codes_off += kk[i] != (float) kc[i] * f32_from_f16(ks[i / 64]) ? 1 : 0;
                for (int i = 0; i < DV; ++i) codes_off += vv[i] != (float) vc[i] * f32_from_f16(vs[i / 64]) ? 1 : 0;
                codes_total += DK + DV;
            }
        std::vector<double> ref;
        attend_ref(d_q.down(), cur, pos0, T, swa, swa ? ly.sinks_h.data() : nullptr, ref);
        std::vector<float> out = d_out.down();
        out.resize((size_t) T * H * DV);
        const Err e = compare(out, ref);
        worst = std::max(worst, e.rel());
        // accept 1..T of the window: the rest is rolled back by simply starting the next window after the accepted
        const int A = std::uniform_int_distribution<int>(1, T)(rng);
        rejected += T - A;
        for (int t = 0; t < A; ++t) {
            std::memcpy(hist.k.data() + (size_t) (pos0 + t) * n_kv * DK, cur.k.data() + (size_t) (pos0 + t) * n_kv * DK, sizeof(float) * n_kv * DK);
            std::memcpy(hist.v.data() + (size_t) (pos0 + t) * n_kv * DV, cur.v.data() + (size_t) (pos0 + t) * n_kv * DV, sizeof(float) * n_kv * DV);
        }
        pos0 += A;
        ++windows;
    }
    std::printf("%-6s sequence to pos %d: %d windows, %d rejected tokens | cache values off the CPU quantizer: %lld of "
                "%lld | kernel vs the accepted history: rel %.2e\n",
                swa ? "swa" : "global", n_pos, windows, rejected, codes_off, codes_total, worst);
    verdict(codes_off * 1000 <= codes_total, "cache codes in the sequence");
    verdict(worst < kTolKernel, "speculative sequence (rollback / ring)");
    attn_scratch_free(scr);
}

// ---- the sink: out(sink) = out(no sink) * Z / (Z + exp(sink - max)) per head ------------------------------------
void run_sinks(int T, int ctx, uint32_t seed) {
    std::mt19937 rng(seed);
    Layer ly(true, 0, rng);
    const int pos0 = ctx - T;
    const std::vector<float> qkv = random_qkv(T, ly.n_kv, rng);
    Dev<float> d_qkv((size_t) kAttnMaxT * qkv_rows(ly.n_kv)), d_q((size_t) kAttnMaxT * H * DK), d_a((size_t) T * H * DV),
        d_b((size_t) T * H * DV);
    Dev<int32_t> d_pos(1);
    AttnScratch scr;
    std::string err;
    if (!attn_scratch_alloc(scr, 0, err)) throw std::runtime_error(err);
    // history through the append kernel itself, one window of 8 at a time
    for (int p = 0; p < pos0; p += 8) {
        const int t = std::min(8, pos0 - p);
        const std::vector<float> h = random_qkv(t, ly.n_kv, rng);
        d_qkv.up(h);
        d_pos.up({p});
        mimo_qkv_rope_append(d_qkv.p, d_pos.p, t, ly.L, ly.kv, d_q.p, nullptr);
    }
    d_qkv.up(qkv);
    d_pos.up({pos0});
    mimo_qkv_rope_append(d_qkv.p, d_pos.p, T, ly.L, ly.kv, d_q.p, nullptr);
    mimo_attn_decode(d_q.p, d_pos.p, T, ly.L, ly.kv, scr, d_a.p, nullptr);
    AttnLayer nosink = ly.L;
    nosink.sinks = nullptr;
    mimo_attn_decode(d_q.p, d_pos.p, T, nosink, ly.kv, scr, d_b.p, nullptr);
    ck(cudaDeviceSynchronize(), "run");
    KvRef q8;
    ly.dequant(ctx, q8);
    std::vector<double> ref, zsum, zmax;
    std::vector<float> q = d_q.down();
    q.resize((size_t) T * H * DK);
    const std::vector<float> a = d_a.down(), b = d_b.down();
    attend_ref(q, q8, pos0, T, true, ly.sinks_h.data(), ref, &zsum, &zmax);
    double worst = 0, fmin = 1, fmax = 0;
    for (int t = 0; t < T; ++t)
        for (int hq = 0; hq < H; ++hq) {
            const size_t i = (size_t) t * H + hq;
            const double f = zsum[i] / (zsum[i] + std::exp((double) ly.sinks_h[hq] - zmax[i]));
            fmin = std::min(fmin, f);
            fmax = std::max(fmax, f);
            double num = 0, den = 0;
            for (int d = 0; d < DV; ++d) {
                num = std::max(num, std::fabs((double) a[i * DV + d] - f * b[i * DV + d]));
                den += (double) a[i * DV + d] * a[i * DV + d];
            }
            worst = std::max(worst, num / std::sqrt(den / DV + 1e-30));
        }
    std::printf("sinks  T=%d ctx=%d: Z/(Z+e^sink) spans %.3f..%.3f | out(sink) vs out(no sink) x factor: rel %.2e | "
                "vs ggml formula: rel %.2e\n",
                T, ctx, fmin, fmax, worst, compare(a, ref).rel());
    verdict(worst < kTolKernel, "sink identity");
    verdict(compare(a, ref).rel() < kTolKernel, "sinks vs ggml formula");
    verdict(fmin < 0.9, "the sinks matter in this case");
    attn_scratch_free(scr);
}

// ---- rmsnorm and the residual add against the CPU -------------------------------------------------------------
void run_norm(uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    constexpr int N = 4096, T = 3;
    std::vector<float> x((size_t) T * N), w(N), y((size_t) T * N);
    for (auto& v : x) v = 3.0f * nd(rng);
    for (auto& v : w) v = 1.0f + 0.2f * nd(rng);
    for (auto& v : y) v = nd(rng);
    Dev<float> dx(x.size()), dw(w.size()), dy(y.size()), dn(x.size());
    dx.up(x);
    dw.up(w);
    dy.up(y);
    mimo_rmsnorm(dx.p, dw.p, dn.p, N, T, 1e-6f, nullptr);
    mimo_residual_add(dx.p, dy.p, 0.707f, T * N, nullptr);
    ck(cudaDeviceSynchronize(), "run");
    const std::vector<float> norm = dn.down(), res = dx.down();
    std::vector<double> rn(x.size()), rr(x.size());
    for (int t = 0; t < T; ++t) {
        double ss = 0;
        for (int i = 0; i < N; ++i) ss += (double) x[(size_t) t * N + i] * x[(size_t) t * N + i];
        const double r = 1.0 / std::sqrt(ss / N + 1e-6);
        for (int i = 0; i < N; ++i) rn[(size_t) t * N + i] = x[(size_t) t * N + i] * r * w[i];
    }
    for (size_t i = 0; i < x.size(); ++i) rr[i] = (double) (x[i] + y[i] * 0.707f);   // ggml: scale, then add
    const Err en = compare(norm, rn), er = compare(res, rr);
    std::printf("rmsnorm T=%d n=%d: rel %.1e | residual add x + 0.707 y: max %.1e\n", T, N, en.rel(), er.max_abs);
    verdict(en.rel() < 1e-5, "rmsnorm");
    verdict(er.max_abs == 0.0, "residual add (bit-exact)");
}

// ---- benchmark --------------------------------------------------------------------------------------------------
struct Stats {
    double best = 0, median = 0, worst = 0;
};
Stats stats(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return Stats{v.front(), v[v.size() / 2], v.back()};
}

void fill_random(const AttnKv& kv, std::mt19937& rng) {   // the kernel's speed does not depend on the values
    const size_t cells = (size_t) kv.n_kv * kv.cap;
    std::vector<int8_t> codes(cells * DK);
    for (auto& c : codes) c = (int8_t) ((int) (rng() % 255) - 127);
    ck(cudaMemcpy(kv.k, codes.data(), cells * DK, cudaMemcpyHostToDevice), "k");
    ck(cudaMemcpy(kv.v, codes.data(), cells * DV, cudaMemcpyHostToDevice), "v");
    std::vector<uint16_t> sc(cells * 4, f16_from_f32(0.02f));
    ck(cudaMemcpy(kv.k_scale, sc.data(), cells * 8, cudaMemcpyHostToDevice), "ks");
    ck(cudaMemcpy(kv.v_scale, sc.data(), cells * 4, cudaMemcpyHostToDevice), "vs");
}

// Timed the way the decode runs: N calls captured in one CUDA graph (no launch overhead), per call = graph time / N.
// Consecutive calls read different KV (other positions of the 256K global cache, other SWA layers) and every graph
// reads more than the 128 MB L2, so the KV comes from DRAM as in a real decode step, where 47 other layers' KV and
// weights pass between two reads of one layer.
void bench(int only_T, int only_ctx) {
    cudaDeviceProp prop{};
    ck(cudaGetDeviceProperties(&prop, 0), "props");
    std::printf("device: %s, %d SMs, L2 %d MB\n", prop.name, prop.multiProcessorCount, prop.l2CacheSize >> 20);
    std::mt19937 rng(7);
    constexpr int kMaxCtx = 262144, kSwaLayers = 512;   // 512 x 340 KB = 174 MB of SWA caches
    Layer glob(false, kMaxCtx, rng);
    fill_random(glob.kv, rng);
    std::vector<std::unique_ptr<Layer>> swa;
    if (only_T == 0)
        for (int i = 0; i < kSwaLayers; ++i) {
            swa.push_back(std::make_unique<Layer>(true, 0, rng));
            fill_random(swa.back()->kv, rng);
        }
    AttnScratch scr;
    std::string err;
    if (!attn_scratch_alloc(scr, 0, err)) throw std::runtime_error(err);
    Dev<float> d_q((size_t) kAttnMaxT * H * DK), d_out((size_t) kAttnMaxT * H * DV);
    {
        std::normal_distribution<float> nd(0, 1);
        std::vector<float> q((size_t) kAttnMaxT * H * DK);
        for (auto& x : q) x = nd(rng);
        d_q.up(q);
    }
    Dev<int32_t> d_pos(1);
    cudaStream_t st;
    ck(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "stream");
    cudaEvent_t e0, e1;
    ck(cudaEventCreate(&e0), "event");
    ck(cudaEventCreate(&e1), "event");
    constexpr int kRepeats = 5;
    auto graph_time = [&](int n_calls, auto&& call) {   // call(i) enqueues call i on st
        cudaGraph_t g;
        cudaGraphExec_t ge;
        ck(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal), "capture");
        for (int i = 0; i < n_calls; ++i) call(i);
        ck(cudaStreamEndCapture(st, &g), "end capture");
        ck(cudaGraphInstantiate(&ge, g, 0), "instantiate");
        std::vector<double> us;
        for (int rep = 0; rep < kRepeats + 1; ++rep) {
            ck(cudaEventRecord(e0, st), "rec");
            ck(cudaGraphLaunch(ge, st), "launch");
            ck(cudaEventRecord(e1, st), "rec");
            ck(cudaStreamSynchronize(st), "sync");
            float ms = 0;
            ck(cudaEventElapsedTime(&ms, e0, e1), "elapsed");
            if (rep > 0) us.push_back(ms * 1000.0 / n_calls);   // the first launch warms up
        }
        cudaGraphExecDestroy(ge);
        cudaGraphDestroy(g);
        return stats(us);
    };
    auto time_global = [&](int T, int ctx) {
        d_pos.up({ctx - T});
        const int regions = std::min(64, kMaxCtx / ctx);   // views of the cache at different positions
        return graph_time(std::max(32, regions), [&](int i) {
            AttnKv v = glob.kv;
            const size_t off = (size_t) (i % regions) * ctx;
            v.k += off * DK;
            v.v += off * DV;
            v.k_scale += off * 4;
            v.v_scale += off * 2;
            mimo_attn_decode(d_q.p, d_pos.p, T, glob.L, v, scr, d_out.p, st);
        });
    };
    if (only_T > 0) {   // one configuration, for a profiler
        const Stats s = time_global(only_T, only_ctx);
        std::printf("global ctx %d T %d: %.1f us\n", only_ctx, only_T, s.best);
        return;
    }
    std::printf("\nper-layer attention (split + combine kernels), in a CUDA graph, KV from DRAM; us per call: best / "
                "median / worst of %d graph launches\n",
                kRepeats);
    std::printf("%-7s %7s %2s %26s %10s %8s\n", "layer", "ctx", "T", "time us (best/med/worst)", "KV MB", "GB/s");
    for (int ctx : {4096, 32768, 131072, 262144})
        for (int T : {1, 2, 4, 8}) {
            const Stats s = time_global(T, ctx);
            const double mb = (double) attn_kv_bytes(glob.n_kv, ctx) / 1e6;
            std::printf("%-7s %7d %2d %8.1f / %7.1f / %7.1f %10.1f %8.0f\n", "global", ctx, T, s.best, s.median,
                        s.worst, mb, mb / 1e3 / (s.best * 1e-6));
        }
    for (int T : {1, 2, 4, 8}) {
        const int ctx = 4096;
        d_pos.up({ctx - T});
        const Stats s = graph_time(kSwaLayers, [&](int i) {
            mimo_attn_decode(d_q.p, d_pos.p, T, swa[(size_t) i]->L, swa[(size_t) i]->kv, scr, d_out.p, st);
        });
        const double mb = (double) attn_kv_bytes(8, NSWA + T - 1) / 1e6;
        std::printf("%-7s %7d %2d %8.1f / %7.1f / %7.1f %10.2f %8.0f\n", "swa", ctx, T, s.best, s.median, s.worst, mb,
                    mb / 1e3 / (s.best * 1e-6));
    }
    // the small kernels around it (their data stays in L2)
    {
        Dev<float> x((size_t) kAttnMaxT * 4096), w(4096), y((size_t) kAttnMaxT * 4096);
        ck(cudaMemset(x.p, 0, x.n * 4), "m");
        ck(cudaMemset(w.p, 0, w.n * 4), "m");
        Dev<float> qkv((size_t) kAttnMaxT * qkv_rows(8));
        ck(cudaMemset(qkv.p, 0, qkv.n * 4), "m");
        d_pos.up({1000});
        for (int T : {1, 8}) {
            const Stats a = graph_time(64, [&](int) { mimo_rmsnorm(x.p, w.p, y.p, 4096, T, 1e-6f, st); });
            const Stats b = graph_time(64, [&](int i) {
                mimo_qkv_rope_append(qkv.p, d_pos.p, T, swa[(size_t) i]->L, swa[(size_t) i]->kv, d_q.p, st);
            });
            const Stats c = graph_time(64, [&](int) { mimo_residual_add(x.p, y.p, 0.707f, 4096 * T, st); });
            std::printf("T=%d (graph, best): rmsnorm %.1f us, qkv_rope_append (SWA layer) %.1f us, residual add "
                        "%.1f us\n",
                        T, a.best, b.best, c.best);
        }
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cudaStreamDestroy(st);
    attn_scratch_free(scr);
}

}  // namespace

int main(int argc, char** argv) {
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev == 0) {
        std::printf("no CUDA device: skipped\n");
        return 77;
    }
    try {
        if (argc > 1 && std::strcmp(argv[1], "--bench") == 0) {
            bench(argc > 3 ? std::atoi(argv[2]) : 0, argc > 3 ? std::atoi(argv[3]) : 0);
            return 0;
        }
        uint32_t seed = 1;
        for (int T : {1, 4}) {
            for (int ctx : {1, 5, 127, 128, 129, 300}) run_case(true, T, ctx, seed++);
            for (int ctx : {1, 1000, 65536}) run_case(false, T, ctx, seed++);
        }
        run_case(false, 8, 4099, seed++);   // 8 m-tiles
        run_case(true, 3, 200, seed++);     // a partial last m-tile
        run_case(false, 2, 20000, seed++);
        run_sequence(true, 600, seed++);    // the 256-slot ring wraps twice
        run_sequence(false, 300, seed++);
        run_sinks(1, 300, seed++);
        run_sinks(4, 129, seed++);
        run_norm(seed++);
    } catch (const std::exception& e) {
        std::printf("error: %s\n", e.what());
        return 1;
    }
    std::printf(g_fail ? "%d check(s) FAILED\n" : "all checks passed\n", g_fail);
    return g_fail ? 1 : 0;
}
