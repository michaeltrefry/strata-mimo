// mimo_router_test: the GPU router against a double-precision CPU reference, on MiMo's real router weights and
// selection biases (layers 1, 24, 47), T = 1..8 random hidden states.  Prints the largest logit / weight error,
// how many picks differ (only possible at a near-tie in the selection score) and the time per layer.
//
// With a llama-mimo-dump directory as the second argument it also runs the router on llama.cpp's own router inputs
// (ffn_norm-L) for every dumped token and layer and compares the picks (as sets) and weights with llama.cpp's
// (ffn_moe_topk-L, ffn_moe_weights_norm-L).
//
// usage: mimo_router_test <any shard of the model> [dump dir]
#include "strata/mimo/model.hpp"
#include "strata/mimo/router.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <set>
#include <numeric>
#include <random>
#include <vector>

using namespace strata::mimo;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
        return 2;
    }
    MimoModel model(argv[1]);
    const MimoGeometry& g = model.geometry();
    const int E = g.n_embd, N = g.n_expert, K = g.n_expert_used;
    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.f, 1.f);
    float *d_w, *d_b, *d_h, *d_logits, *d_wt;
    int* d_ids;
    cudaMalloc(&d_w, sizeof(float) * E * N);
    cudaMalloc(&d_b, sizeof(float) * N);
    cudaMalloc(&d_h, sizeof(float) * kRouterMaxT * E);
    cudaMalloc(&d_logits, sizeof(float) * kRouterMaxT * N);
    cudaMalloc(&d_wt, sizeof(float) * kRouterMaxT * K);
    cudaMalloc(&d_ids, sizeof(int) * kRouterMaxT * K);
    double worst_logit = 0, worst_w = 0;
    int pick_diff = 0, picks = 0;
    bool ok = true;
    for (int layer : {1, 24, 47}) {
        const std::string b = "blk." + std::to_string(layer) + ".";
        const float* w = reinterpret_cast<const float*>(model.host_data(b + "ffn_gate_inp.weight"));
        const float* bias = reinterpret_cast<const float*>(model.host_data(b + "exp_probs_b.bias"));
        cudaMemcpy(d_w, w, sizeof(float) * E * N, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, bias, sizeof(float) * N, cudaMemcpyHostToDevice);
        for (int T = 1; T <= kRouterMaxT; ++T) {
            std::vector<float> h((size_t) T * E);
            for (auto& x : h) x = nd(rng);
            cudaMemcpy(d_h, h.data(), sizeof(float) * h.size(), cudaMemcpyHostToDevice);
            router_logits(d_w, d_h, d_logits, T, E, N, nullptr);
            router_topk(d_logits, d_b, d_ids, d_wt, T, N, K, g.expert_weights_scale, nullptr);
            std::vector<float> logits((size_t) T * N), wt((size_t) T * K);
            std::vector<int> ids((size_t) T * K);
            cudaMemcpy(logits.data(), d_logits, sizeof(float) * logits.size(), cudaMemcpyDeviceToHost);
            cudaMemcpy(wt.data(), d_wt, sizeof(float) * wt.size(), cudaMemcpyDeviceToHost);
            if (cudaMemcpy(ids.data(), d_ids, sizeof(int) * ids.size(), cudaMemcpyDeviceToHost) != cudaSuccess) {
                std::fprintf(stderr, "FAIL: CUDA error %s\n", cudaGetErrorString(cudaGetLastError()));
                return 1;
            }
            for (int t = 0; t < T; ++t) {
                std::vector<double> p(N), sel(N);
                for (int e = 0; e < N; ++e) {
                    double acc = 0, mag = 0;   // float rounding of a dot product scales with sum |w h|
                    for (int i = 0; i < E; ++i) {
                        acc += (double) w[(size_t) e * E + i] * h[(size_t) t * E + i];
                        mag += std::fabs((double) w[(size_t) e * E + i] * h[(size_t) t * E + i]);
                    }
                    worst_logit = std::max(worst_logit, std::fabs(acc - logits[(size_t) t * N + e]) / mag);
                    p[e] = 1.0 / (1.0 + std::exp(-acc));
                    sel[e] = p[e] + bias[e];
                }
                std::vector<int> order(N);
                std::iota(order.begin(), order.end(), 0);
                std::partial_sort(order.begin(), order.begin() + K, order.end(),
                                  [&](int a, int c) { return sel[a] > sel[c] || (sel[a] == sel[c] && a < c); });
                double sum = 0;
                for (int j = 0; j < K; ++j) sum += p[order[j]];
                for (int j = 0; j < K; ++j) {
                    ++picks;
                    const int got = ids[(size_t) t * K + j];
                    if (got != order[j]) { ++pick_diff; continue; }
                    const double ref = p[order[j]] / std::max(sum, 6.103515625e-5);
                    worst_w = std::max(worst_w, std::fabs(ref - wt[(size_t) t * K + j]) / ref);
                }
            }
        }
    }
    // time: 47 layers' routers captured in one CUDA graph (how the decode graph runs them), per layer
    cudaStream_t st;
    cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking);
    cudaEvent_t a, c;
    cudaEventCreate(&a);
    cudaEventCreate(&c);
    for (int T : {1, 2, 4, 8}) {
        cudaGraph_t graph;
        cudaGraphExec_t exec;
        cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal);
        for (int l = 0; l < 47; ++l) {
            router_logits(d_w, d_h, d_logits, T, E, N, st);
            router_topk(d_logits, d_b, d_ids, d_wt, T, N, K, g.expert_weights_scale, st);
        }
        cudaStreamEndCapture(st, &graph);
        cudaGraphInstantiate(&exec, graph, 0);
        cudaGraphLaunch(exec, st);
        float best = 1e9f;
        for (int rep = 0; rep < 20; ++rep) {
            cudaEventRecord(a, st);
            cudaGraphLaunch(exec, st);
            cudaEventRecord(c, st);
            cudaEventSynchronize(c);
            float ms = 0;
            cudaEventElapsedTime(&ms, a, c);
            best = std::min(best, ms);
        }
        std::printf("router in a CUDA graph, T=%d: %.1f us per layer (best of 20 x 47 layers)\n", T, best * 1000 / 47);
        cudaGraphExecDestroy(exec);
        cudaGraphDestroy(graph);
    }
    std::printf("largest logit error (relative to sum |w h|) %.2e, weight error %.2e; %d of %d picks differ from the reference\n",
                worst_logit, worst_w, pick_diff, picks);
    ok = worst_logit < 1e-5 && worst_w < 1e-4 && pick_diff == 0;
    if (argc > 2) {   // against llama.cpp's own routing
        auto load = [&](const std::string& name, size_t n, void* dst) {
            std::ifstream f(std::string(argv[2]) + "/" + name, std::ios::binary);
            return (bool) f.read((char*) dst, (std::streamsize) n);
        };
        std::ifstream tf(std::string(argv[2]) + "/tokens.i32", std::ios::binary | std::ios::ate);
        const int n_tok = (int) (tf.tellg() / 4);
        int set_diff = 0, total = 0, mine_exact = 0, ref_exact = 0, neither_exact = 0;
        double worst_ref_w = 0;
        std::vector<float> x((size_t) n_tok * E), rw((size_t) n_tok * K);
        std::vector<int> rid((size_t) n_tok * K);
        for (int l = 0; l < g.n_layer; ++l) {
            if (!g.is_moe(l)) continue;
            const std::string L = std::to_string(l), b = "blk." + L + ".";
            if (!load("ffn_norm-" + L + ".f32", x.size() * 4, x.data()) ||
                !load("ffn_moe_topk-" + L + ".i32", rid.size() * 4, rid.data()) ||
                !load("ffn_moe_weights_norm-" + L + ".f32", rw.size() * 4, rw.data())) {
                std::fprintf(stderr, "FAIL: dump files of layer %d missing\n", l);
                return 1;
            }
            cudaMemcpy(d_w, model.host_data(b + "ffn_gate_inp.weight"), sizeof(float) * E * N, cudaMemcpyHostToDevice);
            cudaMemcpy(d_b, model.host_data(b + "exp_probs_b.bias"), sizeof(float) * N, cudaMemcpyHostToDevice);
            for (int t0 = 0; t0 < n_tok; t0 += kRouterMaxT) {
                const int T = std::min(kRouterMaxT, n_tok - t0);
                cudaMemcpy(d_h, x.data() + (size_t) t0 * E, sizeof(float) * T * E, cudaMemcpyHostToDevice);
                router_logits(d_w, d_h, d_logits, T, E, N, nullptr);
                router_topk(d_logits, d_b, d_ids, d_wt, T, N, K, g.expert_weights_scale, nullptr);
                std::vector<int> ids((size_t) T * K);
                std::vector<float> wt((size_t) T * K);
                cudaMemcpy(ids.data(), d_ids, sizeof(int) * ids.size(), cudaMemcpyDeviceToHost);
                cudaMemcpy(wt.data(), d_wt, sizeof(float) * wt.size(), cudaMemcpyDeviceToHost);
                for (int t = 0; t < T; ++t) {
                    ++total;
                    std::set<int> mine(ids.begin() + t * K, ids.begin() + (t + 1) * K);
                    std::set<int> ref(rid.begin() + (size_t) (t0 + t) * K, rid.begin() + (size_t) (t0 + t + 1) * K);
                    if (mine != ref) {   // which side does exact arithmetic agree with?
                        ++set_diff;
                        const float* w = reinterpret_cast<const float*>(model.host_data(b + "ffn_gate_inp.weight"));
                        const float* bias = reinterpret_cast<const float*>(model.host_data(b + "exp_probs_b.bias"));
                        std::vector<double> sel(N);
                        for (int e = 0; e < N; ++e) {
                            double acc = 0;
                            for (int i = 0; i < E; ++i) acc += (double) w[(size_t) e * E + i] * x[(size_t) (t0 + t) * E + i];
                            sel[e] = 1.0 / (1.0 + std::exp(-acc)) + bias[e];
                        }
                        std::vector<int> order(N);
                        std::iota(order.begin(), order.end(), 0);
                        std::partial_sort(order.begin(), order.begin() + K, order.end(), [&](int a, int c) { return sel[a] > sel[c]; });
                        const std::set<int> exact(order.begin(), order.begin() + K);
                        (exact == mine ? mine_exact : exact == ref ? ref_exact : neither_exact)++;
                        continue;
                    }
                    for (int j = 0; j < K; ++j)
                        for (int r = 0; r < K; ++r)
                            if (rid[(size_t) (t0 + t) * K + r] == ids[(size_t) t * K + j])
                                worst_ref_w = std::max(worst_ref_w, (double) std::fabs(rw[(size_t) (t0 + t) * K + r] - wt[(size_t) t * K + j]));
                }
            }
        }
        std::printf("against llama.cpp (%d tokens x %d layers): %d of %d expert sets differ, largest weight difference %.2e\n",
                    n_tok, g.count_moe(), set_diff, total, worst_ref_w);
        // the batched router (prompt chunks) on all dumped tokens at once: its picks against the T <= 8 kernel's
        {
            cublasHandle_t cb;
            cublasCreate(&cb);
            float* d_all = nullptr;
            float* d_lg = nullptr;
            int *d_id1 = nullptr;
            float* d_w1 = nullptr;
            cudaMalloc(&d_all, sizeof(float) * x.size());
            cudaMalloc(&d_lg, sizeof(float) * (size_t) n_tok * N);
            cudaMalloc(&d_id1, sizeof(int) * (size_t) n_tok * K);
            cudaMalloc(&d_w1, sizeof(float) * (size_t) n_tok * K);
            int batched_diff = 0;
            for (int l : {1, 24, 47}) {
                const std::string L = std::to_string(l), b = "blk." + L + ".";
                load("ffn_norm-" + L + ".f32", x.size() * 4, x.data());
                cudaMemcpy(d_w, model.host_data(b + "ffn_gate_inp.weight"), sizeof(float) * E * N, cudaMemcpyHostToDevice);
                cudaMemcpy(d_b, model.host_data(b + "exp_probs_b.bias"), sizeof(float) * N, cudaMemcpyHostToDevice);
                cudaMemcpy(d_all, x.data(), sizeof(float) * x.size(), cudaMemcpyHostToDevice);
                router_logits_batched(cb, d_w, d_all, d_lg, n_tok, E, N);
                router_topk(d_lg, d_b, d_id1, d_w1, n_tok, N, K, g.expert_weights_scale, nullptr);
                std::vector<int> big((size_t) n_tok * K);
                cudaMemcpy(big.data(), d_id1, sizeof(int) * big.size(), cudaMemcpyDeviceToHost);
                for (int t0 = 0; t0 < n_tok; t0 += kRouterMaxT) {
                    const int T = std::min(kRouterMaxT, n_tok - t0);
                    cudaMemcpy(d_h, x.data() + (size_t) t0 * E, sizeof(float) * T * E, cudaMemcpyHostToDevice);
                    router_logits(d_w, d_h, d_logits, T, E, N, nullptr);
                    router_topk(d_logits, d_b, d_ids, d_wt, T, N, K, g.expert_weights_scale, nullptr);
                    std::vector<int> small((size_t) T * K);
                    cudaMemcpy(small.data(), d_ids, sizeof(int) * small.size(), cudaMemcpyDeviceToHost);
                    for (int t = 0; t < T; ++t)
                        batched_diff += std::set<int>(small.begin() + t * K, small.begin() + (t + 1) * K) !=
                                        std::set<int>(big.begin() + (size_t) (t0 + t) * K, big.begin() + (size_t) (t0 + t + 1) * K);
                }
            }
            std::printf("batched router (cuBLAS SGEMM, %d tokens at once) vs the T <= 8 kernel, layers 1/24/47: %d of %d "
                        "expert sets differ\n", n_tok, batched_diff, 3 * n_tok);
            ok = ok && batched_diff * 1000 <= 3 * n_tok;
            cudaFree(d_all); cudaFree(d_lg); cudaFree(d_id1); cudaFree(d_w1);
            cublasDestroy(cb);
        }
        std::printf("  of those, exact (double) arithmetic picks this router's set %d times, llama.cpp's %d, neither %d\n",
                    mine_exact, ref_exact, neither_exact);
        ok = ok && ref_exact == 0;   // llama.cpp may round its router (cuBLAS); this one must not be the one that is off
    }
    ok = ok && worst_w < 1e-4 && pick_diff == 0;
    std::printf("%s\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
