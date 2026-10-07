// mxfp4-cpu-bench: how fast can the CPU compute MiMo's MXFP4 experts that are not on the GPU?
//
// Copies the real experts of the first N_LOAD MoE layers from the GGUF into ordinary RAM, then simulates decode:
// per token, 47 layers; per layer, M random experts: gate+up (2 x 2048 rows of 4096) -> SiLU*up -> down (4096 rows
// of 2048). All threads split the rows; two barriers per layer, like an engine would do.
// usage: mxfp4-cpu-bench <gguf with the expert tensors> [threads=24] [n_load=24]
#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

static const int H = 4096, FF = 2048, NE = 256, NL = 47;

struct barrier {
    std::atomic<int> count{0}, gen{0};
    int n;
    void wait() {
        int g = gen.load(std::memory_order_acquire);
        if (count.fetch_add(1, std::memory_order_acq_rel) == n - 1) {
            count.store(0, std::memory_order_relaxed);
            gen.fetch_add(1, std::memory_order_release);
        } else {
            while (gen.load(std::memory_order_acquire) == g) {
            }
        }
    }
};

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <gguf> [threads] [n_load]\n", argv[0]);
        return 1;
    }
    const int n_thr  = argc > 2 ? atoi(argv[2]) : 24;
    const int n_load = argc > 3 ? atoi(argv[3]) : 24;
    ggml_cpu_init();
    const auto * tr = ggml_get_type_traits_cpu(GGML_TYPE_MXFP4);
    const auto * q8 = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
    const size_t row_gu = ggml_row_size(GGML_TYPE_MXFP4, H), row_d = ggml_row_size(GGML_TYPE_MXFP4, FF);
    const size_t ex_gu = row_gu * FF, ex_d = row_d * H;   // bytes per expert per matrix
    const size_t ex_bytes = 2 * ex_gu + ex_d;

    gguf_init_params gp = {true, nullptr};
    gguf_context * g = gguf_init_from_file(argv[1], gp);
    if (!g) {
        fprintf(stderr, "cannot read %s\n", argv[1]);
        return 1;
    }
    FILE * f = fopen(argv[1], "rb");
    const size_t data0 = gguf_get_data_offset(g);
    // pool[l] = gate | up | down for all 256 experts of MoE layer l (blk.l+1)
    std::vector<uint8_t *> gate(n_load), up(n_load), down(n_load);
    auto t0 = std::chrono::steady_clock::now();
    for (int l = 0; l < n_load; ++l) {
        const char * names[3] = {"ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"};
        uint8_t ** dst[3] = {&gate[l], &up[l], &down[l]};
        size_t sz[3] = {ex_gu * NE, ex_gu * NE, ex_d * NE};
        for (int m = 0; m < 3; ++m) {
            char name[64];
            snprintf(name, sizeof name, "blk.%d.%s.weight", l + 1, names[m]);
            int64_t id = gguf_find_tensor(g, name);
            if (id < 0) {
                fprintf(stderr, "missing %s\n", name);
                return 1;
            }
            *dst[m] = (uint8_t *) malloc(sz[m]);
            _fseeki64(f, (long long) (data0 + gguf_get_tensor_offset(g, id)), SEEK_SET);
            if (fread(*dst[m], 1, sz[m], f) != sz[m]) {
                fprintf(stderr, "short read %s\n", name);
                return 1;
            }
        }
    }
    fclose(f);
    double load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("loaded %d layers (%.1f GiB) in %.0f s; %s\n", n_load, n_load * NE * ex_bytes / 1073741824.0, load_s,
           ggml_cpu_has_avx512() ? "AVX-512" : "AVX2");

    // activations
    const int MMAX = 8;
    std::vector<float> x(H), hbuf(MMAX * FF), gbuf(MMAX * FF), ubuf(MMAX * FF), ybuf(H);
    std::mt19937 rng(1);
    std::normal_distribution<float> nd(0, 1);
    for (auto & v : x) v = nd(rng);
    const size_t hq_row = ggml_row_size(GGML_TYPE_Q8_0, FF);
    std::vector<uint8_t> xq(ggml_row_size(GGML_TYPE_Q8_0, H)), hq(MMAX * hq_row);
    q8->from_float(x.data(), xq.data(), H);

    for (int M : {1, 2, 3, 4, 8}) {
        const int n_tok = M <= 2 ? 60 : 30;
        // pre-pick experts
        std::vector<int> pick((size_t) n_tok * NL * M);
        std::uniform_int_distribution<int> ue(0, NE - 1);
        for (auto & e : pick) e = ue(rng);
        barrier bar;
        bar.n = n_thr;
        std::atomic<bool> go{false};
        auto work = [&](int t) {
            while (!go.load()) {
            }
            for (int tok = 0; tok < n_tok; ++tok) {
                for (int L = 0; L < NL; ++L) {
                    const int l = L % n_load;
                    const int * es = &pick[((size_t) tok * NL + L) * M];
                    // phase 1: gate+up rows of all M experts, contiguous chunks per thread
                    const int n1 = M * 2 * FF, c1 = (n1 + n_thr - 1) / n_thr;
                    for (int r = t * c1; r < std::min(n1, (t + 1) * c1); ++r) {
                        const int j = r / (2 * FF), rr = r % (2 * FF);
                        const uint8_t * w = rr < FF ? gate[l] + es[j] * ex_gu + (size_t) rr * row_gu
                                                    : up[l] + es[j] * ex_gu + (size_t) (rr - FF) * row_gu;
                        float s;
                        tr->vec_dot(H, &s, 0, w, 0, xq.data(), 0, 1);
                        (rr < FF ? gbuf : ubuf)[(size_t) j * FF + rr % FF] = s;
                    }
                    bar.wait();
                    // SiLU(gate)*up and Q8_0 quantization: one expert per thread
                    for (int j = t; j < M; j += n_thr) {
                        float * hj = &hbuf[(size_t) j * FF];
                        for (int i = 0; i < FF; ++i) {
                            const float gv = gbuf[(size_t) j * FF + i];
                            hj[i] = gv / (1 + std::exp(-gv)) * ubuf[(size_t) j * FF + i];
                        }
                        q8->from_float(hj, hq.data() + j * hq_row, FF);
                    }
                    bar.wait();
                    // phase 2: down rows (output row r summed over the M experts by one thread)
                    const int c2 = (H + n_thr - 1) / n_thr;
                    for (int r = t * c2; r < std::min(H, (t + 1) * c2); ++r) {
                        float acc = 0;
                        for (int j = 0; j < M; ++j) {
                            float s;
                            tr->vec_dot(FF, &s, 0, down[l] + es[j] * ex_d + (size_t) r * row_d, 0,
                                        hq.data() + j * hq_row, 0, 1);
                            acc += s;
                        }
                        ybuf[r] = acc;
                    }
                    bar.wait();
                }
            }
        };
        std::vector<std::thread> th;
        for (int t = 0; t < n_thr; ++t) th.emplace_back(work, t);
        auto s0 = std::chrono::steady_clock::now();
        go = true;
        for (auto & x_ : th) x_.join();
        double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - s0).count();
        double bytes = (double) n_tok * NL * M * ex_bytes;
        printf("M=%d experts/layer: %.2f ms per token (47 layers), %.1f GB/s, %.1f experts/ms\n", M, s / n_tok * 1e3,
               bytes / s / 1e9, n_tok * NL * M / (s * 1e3));
    }
    return 0;
}
