// strata-mimo-run: MiMo-V2.6-Flash greedy decode on the v0 decoder (docs/MIMO_PORT_PLAN.md step 6).
//
//   strata-mimo-run <model.gguf> --tokens <int32 file> [--generate N] [--profile P] [--budget-gib G]
//                   [--ctx N] [--window T] [--dump DIR]
// The prompt goes through in windows of T tokens (default 8), then N tokens are generated one at a time.  Prints
// the generated token ids, the speed and the share of routed experts served from resident slots.  --dump writes
// every intermediate under llama.cpp's names (compare with tools/mimo/mimo_parity.py).
#include "strata/mimo/decoder.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <vector>

using namespace strata::mimo;
using clk = std::chrono::steady_clock;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <model.gguf> --tokens FILE [--generate N] [--profile P] [--budget-gib G] "
                             "[--ctx N] [--window T] [--dump DIR]\n", argv[0]);
        return 2;
    }
    DecoderConfig cfg;
    cfg.max_ctx = 32768;
    std::string tokens_path;
    int generate = 0, window = 8;
    for (int i = 2; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "--tokens") tokens_path = argv[i + 1];
        else if (a == "--generate") generate = std::atoi(argv[i + 1]);
        else if (a == "--profile") cfg.profile = argv[i + 1];
        else if (a == "--budget-gib") cfg.expert_budget_gib = std::atof(argv[i + 1]);
        else if (a == "--ctx") cfg.max_ctx = std::atoi(argv[i + 1]);
        else if (a == "--window") window = std::atoi(argv[i + 1]);
        else if (a == "--dump") cfg.dump_dir = argv[i + 1];
        else if (a == "--cpu-experts") cfg.cpu_experts = std::atoi(argv[i + 1]) != 0;
        else if (a == "--ram-copy") cfg.ram_copy = std::atoi(argv[i + 1]) != 0;
        else if (a == "--graph") cfg.graph = std::atoi(argv[i + 1]) != 0;
        else {
            std::fprintf(stderr, "unknown option %s\n", a.c_str());
            return 2;
        }
    }
    std::vector<int32_t> prompt;
    {
        std::ifstream f(tokens_path, std::ios::binary);
        int32_t v;
        while (f.read((char*) &v, 4)) prompt.push_back(v);
    }
    if (prompt.empty()) {
        std::fprintf(stderr, "no prompt tokens in '%s'\n", tokens_path.c_str());
        return 2;
    }
    if (!cfg.dump_dir.empty()) {   // the dump appends: start from an empty directory
        std::filesystem::create_directories(cfg.dump_dir);
        for (auto& e : std::filesystem::directory_iterator(cfg.dump_dir))
            if (e.path().extension() == ".f32" || e.path().extension() == ".i32") std::filesystem::remove(e.path());
    }
    try {
        auto t0 = clk::now();
        MimoModel model(argv[1]);
        MimoDecoder dec(model, cfg);
        std::string err;
        if (!dec.init(err)) {
            std::fprintf(stderr, "FAIL: %s\n", err.c_str());
            return 1;
        }
        std::printf("ready in %.1f s: %d resident experts, context %d\n",
                    std::chrono::duration<double>(clk::now() - t0).count(), dec.resident_experts(), cfg.max_ctx);
        int32_t next[8];
        int pos = 0, last = 0;   // next[last]: the prediction after the prompt's final token
        t0 = clk::now();
        while (pos < (int) prompt.size()) {
            const int T = std::min(window, (int) prompt.size() - pos);
            if (!dec.forward(prompt.data() + pos, T, pos, next, nullptr, err)) {
                std::fprintf(stderr, "FAIL at %d: %s\n", pos, err.c_str());
                return 1;
            }
            pos += T;
            last = T - 1;
        }
        const double pf = std::chrono::duration<double>(clk::now() - t0).count();
        std::printf("prompt: %d tokens in %.2f s (%.1f tok/s), first next token %d\n", pos, pf, pos / pf, next[last]);
        const DecoderStats before = dec.stats();
        std::vector<int32_t> out;
        int32_t tok = next[last];
        t0 = clk::now();
        for (int i = 0; i < generate; ++i) {
            out.push_back(tok);
            if (!dec.forward(&tok, 1, pos, next, nullptr, err)) {
                std::fprintf(stderr, "FAIL at %d: %s\n", pos, err.c_str());
                return 1;
            }
            ++pos;
            tok = next[0];
        }
        if (generate > 0) {
            const double gs = std::chrono::duration<double>(clk::now() - t0).count();
            const DecoderStats& st = dec.stats();
            const double uses = (double) (st.expert_uses - before.expert_uses);
            std::printf("generated %d tokens in %.2f s (%.2f tok/s); resident hit rate %.1f%%, %.1f experts staged per "
                        "token, CPU experts %.2f ms per token\n",
                        generate, gs, generate / gs, 100.0 * (st.expert_hits - before.expert_hits) / uses,
                        (double) (st.staged - before.staged) / generate, (st.cpu_ms - before.cpu_ms) / generate);
            std::printf("tokens:");
            for (int32_t t : out) std::printf(" %d", t);
            std::printf("\n");
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
