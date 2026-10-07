// mimo_cpu_experts_parity: MiMo step 2's CPU experts against a float reference, on the real model's experts.
//
// For a few MoE layers and windows of T = 1..4 tokens (each token with its own random 1-8 experts and router
// weights), the pool's output (CpuExperts::run_layer: Q8_0 activations, MXFP4 x Q8_0 dots, SiLU, Q8_0 intermediate,
// down, weighted sum) is compared with:
//   1. the float reference: every weight dequantized with ggml's MXFP4 to_float, exact matmuls in double, F32 x;
//   2. the same reference fed the pool's own quantized activations (x and the intermediate through Q8_0), which
//      isolates the kernel and the pool from the activation contract (what is left is float summation order);
// and the AVX-512 MXFP4 kernel's rows are checked BITWISE against ggml's `ggml_vec_dot_mxfp4_q8_0` (1-8 tokens).
//
// usage: mimo_cpu_experts_parity <any shard of the MiMo GGUF> [layers, default 1,24,47] [trials per T, default 3]
#include "strata/kernels/cpu/mxfp4_avx512.hpp"
#include "strata/mimo/cpu_experts.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sstream>
#include <thread>
#include <vector>

using namespace strata::mimo;
namespace cpu = strata::kernels::cpu;

namespace {

// y[r] = sum_i W[r] . x for rows of a quantized matrix (dequantized row by row), in double, on all cores
void ref_matvec(const uint8_t* w, size_t row_bytes, int rows, int n, const std::vector<std::vector<double>>& xs,
                std::vector<std::vector<double>>& ys) {
    const auto* tr = ggml_get_type_traits(GGML_TYPE_MXFP4);
    ys.assign(xs.size(), std::vector<double>((size_t) rows, 0.0));
    const int nth = (int) (std::max)(1u, std::thread::hardware_concurrency() / 2);
    std::vector<std::thread> th;
    for (int t = 0; t < nth; ++t)
        th.emplace_back([&, t] {
            std::vector<float> row((size_t) n);
            for (int r = t; r < rows; r += nth) {
                tr->to_float(w + (size_t) r * row_bytes, row.data(), n);
                for (size_t k = 0; k < xs.size(); ++k) {
                    double s = 0;
                    for (int i = 0; i < n; ++i) s += (double) row[(size_t) i] * xs[k][(size_t) i];
                    ys[k][(size_t) r] = s;
                }
            }
        });
    for (auto& x : th) x.join();
}

// x -> Q8_0 -> float: the values the CPU path actually multiplies
std::vector<float> through_q8_0(const float* x, int n) {
    std::vector<uint8_t> q(ggml_row_size(GGML_TYPE_Q8_0, n));
    ggml_get_type_traits_cpu(GGML_TYPE_Q8_0)->from_float(x, q.data(), n);
    std::vector<float> back((size_t) n);
    ggml_get_type_traits(GGML_TYPE_Q8_0)->to_float(q.data(), back.data(), n);
    return back;
}

double silu(double g) { return g / (1.0 + std::exp(-g)); }

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <MiMo GGUF shard> [layers e.g. 1,24,47] [trials per T]\n", argv[0]);
        return 2;
    }
    std::vector<int> layers = {1, 24, 47};
    if (argc > 2) {
        layers.clear();
        std::stringstream ss(argv[2]);
        for (std::string tok; std::getline(ss, tok, ',');) layers.push_back(std::atoi(tok.c_str()));
    }
    const int trials = argc > 3 ? std::atoi(argv[3]) : 3;
    try {
        MimoModel model(argv[1]);
        const MimoGeometry& g = model.geometry();
        cpu::ExpertPool pool;
        CpuExperts ce(model, pool);
        const int H = g.n_embd, F = g.n_ff_exp;
        std::printf("MiMo CPU experts parity: n_embd %d, n_ff %d, %d experts; pool %d workers + host; AVX-512 MXFP4 kernel %s\n",
                    H, F, g.n_expert, pool.workers(), cpu::mxfp4_512_ok() && !std::getenv("STRATA_NO_MXFP4_512") ? "on" : "off");

        // ---- 1. the kernel's rows, bitwise against ggml's vec_dot
        std::mt19937 rng(1234);
        std::normal_distribution<float> nd(0.f, 1.f);
        int64_t rows_checked = 0, rows_differ = 0;
        if (cpu::mxfp4_512_ok()) {
            const auto* vd = ggml_get_type_traits_cpu(GGML_TYPE_MXFP4);
            for (int l : layers) {
                const ExpertSlices& s = model.experts(l);
                for (int m = 0; m < 3; ++m) {
                    const int n = m == 2 ? F : H, rows = m == 2 ? H : F;
                    const int e = (int) (rng() % (unsigned) g.n_expert);
                    for (int nt = 1; nt <= 8; ++nt) {
                        std::vector<std::vector<uint8_t>> q((size_t) nt, std::vector<uint8_t>(ggml_row_size(GGML_TYPE_Q8_0, n)));
                        std::vector<std::vector<float>> o((size_t) nt, std::vector<float>((size_t) rows));
                        const void* a[8];
                        float* op[8];
                        for (int t = 0; t < nt; ++t) {
                            std::vector<float> x((size_t) n);
                            for (auto& v : x) v = nd(rng) * 3.f;
                            ggml_get_type_traits_cpu(GGML_TYPE_Q8_0)->from_float(x.data(), q[(size_t) t].data(), n);
                            a[t] = q[(size_t) t].data();
                            op[t] = o[(size_t) t].data();
                        }
                        const int r0 = (int) (rng() % 64u), r1 = (std::min)(rows, r0 + 200);
                        cpu::mxfp4_rows(s.matrix(m, e), s.row_bytes[m], n, a, nt, op, r0, r1);
                        for (int t = 0; t < nt; ++t)
                            for (int r = r0; r < r1; ++r) {
                                float ref = 0.f;
                                vd->vec_dot(n, &ref, 0, s.matrix(m, e) + (size_t) r * s.row_bytes[m], 0, a[t], 0, 1);
                                ++rows_checked;
                                if (std::memcmp(&ref, &o[(size_t) t][(size_t) r], 4) != 0) ++rows_differ;
                            }
                    }
                }
            }
            std::printf("kernel vs ggml_vec_dot_mxfp4_q8_0: %lld rows (1-8 tokens), %lld not bitwise equal\n",
                        (long long) rows_checked, (long long) rows_differ);
        } else {
            std::printf("kernel vs ggml: skipped (no AVX-512 VNNI/VBMI on this CPU)\n");
        }

        // ---- 2. the whole layer through the pool against the float references
        double worst_rel = 0, worst_rel_q = 0, worst_max = 0, worst_x = 0;
        int cases = 0;
        for (int l : layers) {
            const ExpertSlices& s = model.experts(l);
            for (int T = 1; T <= 4; ++T)
                for (int trial = 0; trial < trials; ++trial) {
                    const int K = 8;
                    std::vector<int32_t> ids((size_t) T * K, -1);
                    std::vector<float> w((size_t) T * K, 0.f);
                    std::vector<float> x((size_t) T * H);
                    for (auto& v : x) v = nd(rng);
                    for (int t = 0; t < T; ++t) {
                        const int m = 1 + (int) (rng() % 8u);
                        std::vector<int> pick;
                        // later tokens reuse some of token 0's experts, as a verify window does
                        while ((int) pick.size() < m) {
                            int e = (t > 0 && (rng() % 3u) == 0 && ids[(size_t) pick.size()] >= 0) ? ids[(size_t) pick.size()]
                                                                                                    : (int) (rng() % (unsigned) g.n_expert);
                            if (std::find(pick.begin(), pick.end(), e) == pick.end()) pick.push_back(e);
                        }
                        float sum = 0;
                        for (int j = 0; j < m; ++j) {
                            // the ids go in random slots of the 8, the rest stay -1 ("on the GPU")
                            int slot;
                            do slot = (int) (rng() % 8u); while (ids[(size_t) t * K + slot] >= 0);
                            ids[(size_t) t * K + slot] = pick[(size_t) j];
                            w[(size_t) t * K + slot] = 0.05f + (float) (rng() % 1000u) / 1000.f;
                            sum += w[(size_t) t * K + slot];
                        }
                        for (int j = 0; j < K; ++j) w[(size_t) t * K + j] /= sum;
                    }
                    std::vector<float> out((size_t) T * H);
                    ce.run_layer(l, T, x.data(), K, ids.data(), w.data(), out.data());

                    for (int t = 0; t < T; ++t) {
                        std::vector<double> ref((size_t) H, 0.0), refq((size_t) H, 0.0), refx((size_t) H, 0.0);
                        const std::vector<float> xq = through_q8_0(&x[(size_t) t * H], H);
                        std::vector<std::vector<double>> xs(2, std::vector<double>((size_t) H));
                        for (int i = 0; i < H; ++i) {
                            xs[0][(size_t) i] = x[(size_t) t * H + i];
                            xs[1][(size_t) i] = xq[(size_t) i];
                        }
                        for (int j = 0; j < K; ++j) {
                            const int e = ids[(size_t) t * K + j];
                            if (e < 0) continue;
                            std::vector<std::vector<double>> gy, uy, dy;
                            ref_matvec(s.matrix(0, e), s.row_bytes[0], F, H, xs, gy);
                            ref_matvec(s.matrix(1, e), s.row_bytes[1], F, H, xs, uy);
                            std::vector<std::vector<double>> hs(3, std::vector<double>((size_t) F));
                            std::vector<float> hf((size_t) F);
                            for (int r = 0; r < F; ++r) {
                                hs[0][(size_t) r] = silu(gy[0][(size_t) r]) * uy[0][(size_t) r];
                                hs[2][(size_t) r] = silu(gy[1][(size_t) r]) * uy[1][(size_t) r];   // x through Q8_0 only
                                hf[(size_t) r] = (float) (silu(gy[1][(size_t) r]) * uy[1][(size_t) r]);
                            }
                            const std::vector<float> hq = through_q8_0(hf.data(), F);
                            for (int r = 0; r < F; ++r) hs[1][(size_t) r] = hq[(size_t) r];
                            ref_matvec(s.matrix(2, e), s.row_bytes[2], H, F, hs, dy);
                            const double wt = w[(size_t) t * K + j];
                            for (int i = 0; i < H; ++i) {
                                ref[(size_t) i] += wt * dy[0][(size_t) i];
                                refq[(size_t) i] += wt * dy[1][(size_t) i];
                                refx[(size_t) i] += wt * dy[2][(size_t) i];
                            }
                        }
                        double d2 = 0, r2 = 0, dq2 = 0, rq2 = 0, maxd = 0, maxr = 0, dx2 = 0;
                        for (int i = 0; i < H; ++i) {
                            const double o = out[(size_t) t * H + i];
                            d2 += (o - ref[(size_t) i]) * (o - ref[(size_t) i]);
                            r2 += ref[(size_t) i] * ref[(size_t) i];
                            dq2 += (o - refq[(size_t) i]) * (o - refq[(size_t) i]);
                            rq2 += refq[(size_t) i] * refq[(size_t) i];
                            dx2 += (refx[(size_t) i] - ref[(size_t) i]) * (refx[(size_t) i] - ref[(size_t) i]);
                            maxd = (std::max)(maxd, std::fabs(o - ref[(size_t) i]));
                            maxr = (std::max)(maxr, std::fabs(ref[(size_t) i]));
                        }
                        const double rel = r2 > 0 ? std::sqrt(d2 / r2) : std::sqrt(d2);
                        const double relq = rq2 > 0 ? std::sqrt(dq2 / rq2) : std::sqrt(dq2);
                        const double relmax = maxr > 0 ? maxd / maxr : maxd;
                        worst_rel = (std::max)(worst_rel, rel);
                        worst_x = (std::max)(worst_x, r2 > 0 ? std::sqrt(dx2 / r2) : std::sqrt(dx2));
                        worst_rel_q = (std::max)(worst_rel_q, relq);
                        worst_max = (std::max)(worst_max, relmax);
                        ++cases;
                    }
                    std::printf("layer %2d T=%d trial %d: experts per token", l, T, trial);
                    for (int t = 0; t < T; ++t) {
                        int m = 0;
                        for (int j = 0; j < K; ++j) m += ids[(size_t) t * K + j] >= 0;
                        std::printf(" %d", m);
                    }
                    std::printf("; worst so far: rel L2 %.3e (float ref), %.3e (Q8_0-activation ref), max|d|/max|ref| %.3e\n",
                                worst_rel, worst_rel_q, worst_max);
                }
        }
        std::printf("\n%d token outputs: max relative L2 error %.3e vs the float reference, %.3e vs the reference with the "
                    "pool's Q8_0 activations; max |error| / max |ref| %.3e\n", cases, worst_rel, worst_rel_q, worst_max);
        std::printf("(the float reference itself moves by up to %.3e when only x goes through Q8_0: the rest of the gap is the "
                    "intermediate's Q8_0 - ggml-cpu's contract for MXFP4 experts)\n", worst_x);
        // the kernel is held bitwise to ggml and the pool to the Q8_0-activation reference; the float gap is the
        // activation contract's, bounded loosely so that a wrong expert or weight (errors of order 1) still fails
        const bool ok = rows_differ == 0 && worst_rel_q < 2e-3 && worst_rel < 0.1;
        std::printf("%s\n", ok ? "PASS" : "FAIL");
        return ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
