// mimo_router_test: the GPU router against a double-precision CPU reference, on MiMo's real router weights and
// selection biases (layers 1, 24, 47), T = 1..8 random hidden states.  Prints the largest logit / weight error,
// how many picks differ (only possible at a near-tie in the selection score) and the time per layer.
//
// usage: mimo_router_test <any shard of the model>
#include "strata/mimo/model.hpp"
#include "strata/mimo/router.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
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
    std::printf("%s\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
