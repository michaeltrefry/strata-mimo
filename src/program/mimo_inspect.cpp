// strata-mimo-inspect: open a MiMo-V2.6-Flash GGUF, print its shape, check every tensor, upload the non-expert
// weights to the GPU and verify a few of them round-trip.  The step-1 check of docs/MIMO_PORT_PLAN.md.
//
// usage: strata-mimo-inspect <any shard of the model> [--no-upload]
#include "strata/mimo/model.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <exception>
#include <vector>

using namespace strata::mimo;

static bool same_on_device(const DevTensor& d, const uint8_t* host) {
    std::vector<uint8_t> back(d.bytes);
    if (cudaMemcpy(back.data(), d.data, d.bytes, cudaMemcpyDeviceToHost) != cudaSuccess) return false;
    return std::memcmp(back.data(), host, d.bytes) == 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <model.gguf (any shard)> [--no-upload]\n", argv[0]);
        return 2;
    }
    const bool upload = !(argc > 2 && std::strcmp(argv[2], "--no-upload") == 0);
    try {
        auto t0 = std::chrono::steady_clock::now();
        MimoModel model(argv[1]);
        const MimoGeometry& g = model.geometry();
        std::printf("MiMo model: %zu shard(s), %d layers (%d sliding-window, %d global; %d MoE, %d dense)\n",
                    model.gguf().size(), g.n_layer, g.count_swa(), g.n_layer - g.count_swa(), g.count_moe(),
                    g.n_layer - g.count_moe());
        std::printf("  n_embd %d, vocab %d, heads %d, head_k %d, head_v %d, rope %d dims (base %.0f, SWA %.0f), window %d\n",
                    g.n_embd, g.n_vocab, g.n_head, g.head_k, g.head_v, g.n_rot, g.rope_base, g.rope_base_swa, g.n_swa);
        std::printf("  experts %d (top %d, ffn %d), dense ffn %d, rms eps %g, value scale %.4f, weights scale %g\n",
                    g.n_expert, g.n_expert_used, g.n_ff_exp, g.n_ff_dense, g.rms_eps, g.value_scale,
                    g.expert_weights_scale);
        std::printf("  kv heads per layer:");
        for (int l = 0; l < g.n_layer; ++l) std::printf("%s%d", l ? "," : " ", g.n_head_kv[(size_t) l]);
        std::printf("\n  qkv rows: global %d, sliding-window %d\n", g.qkv_rows(0), g.qkv_rows(1));
        const size_t eb = model.expert_bytes();
        std::printf("  one expert: %zu bytes (%.2f MiB); all experts %.1f GiB\n", eb, eb / 1048576.0,
                    (double) eb * g.n_expert * g.count_moe() / (1u << 30));
        std::printf("  opened and checked in %.2f s\n",
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());

        // expert slices agree with the tensor layout: expert e of layer 1's gate is row block e
        const ExpertSlices& s = model.experts(1);
        const strata::TensorInfo* gt = nullptr;
        const uint8_t* gate = model.host_data("blk.1.ffn_gate_exps.weight", &gt);
        if (s.matrix(0, 255) != gate + 255 * s.row_bytes[0] * g.n_ff_exp || s.row_bytes[0] != 2176 ||
            s.row_bytes[2] != 1088) {
            std::fprintf(stderr, "FAIL: expert slice arithmetic (row bytes %zu / %zu)\n", s.row_bytes[0], s.row_bytes[2]);
            return 1;
        }
        std::printf("  expert slices: gate/up rows %zu B, down rows %zu B, expert 255 at the expected offset\n",
                    s.row_bytes[0], s.row_bytes[2]);

        if (!upload) return 0;
        t0 = std::chrono::steady_clock::now();
        std::string err;
        if (!model.upload(err)) {
            std::fprintf(stderr, "FAIL: %s\n", err.c_str());
            return 1;
        }
        const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("uploaded %.2f GiB of non-expert weights in %.1f s (%.2f GB/s)\n", model.arena_bytes() / 1073741824.0,
                    secs, model.arena_bytes() / secs / 1e9);
        struct Check { const char* name; const DevTensor* d; };
        const Check checks[] = {{"output_norm.weight", &model.output_norm()},
                                {"blk.0.ffn_down.weight", &model.layer(0).ffn_down},
                                {"blk.1.attn_sinks.weight", &model.layer(1).attn_sinks},
                                {"blk.47.attn_qkv.weight", &model.layer(47).attn_qkv},
                                {"blk.47.exp_probs_b.bias", &model.layer(47).exp_probs_b}};
        for (const auto& c : checks) {
            if (!*c.d || !same_on_device(*c.d, model.host_data(c.name))) {
                std::fprintf(stderr, "FAIL: %s differs on the device\n", c.name);
                return 1;
            }
        }
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        std::printf("device copies match the file (5 tensors); GPU now %.1f GiB free of %.1f GiB\n",
                    free_b / 1073741824.0, total_b / 1073741824.0);
        std::printf("OK\n");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
