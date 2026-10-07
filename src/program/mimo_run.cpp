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
        else if (a == "--profile-gpu") cfg.profile_gpu = std::atoi(argv[i + 1]) != 0;
        else if (a == "--mtp") cfg.mtp_path = argv[i + 1];
        else if (a == "--mtp-heads") cfg.mtp_heads = std::atoi(argv[i + 1]);
        else if (a == "--batched") cfg.batched_prefill = std::atoi(argv[i + 1]) != 0;
        else if (a == "--batched-min") cfg.batched_min_prompt = std::atoi(argv[i + 1]);
        else if (a == "--adapt") cfg.adapt_weight = std::strtof(argv[i + 1], nullptr);
        else if (a == "--adapt-decode") cfg.adapt_decode = std::atoi(argv[i + 1]) != 0;
        else if (a == "--adapt-swaps") cfg.adapt_swaps = std::atoi(argv[i + 1]);
        else if (a == "--adapt-interval") cfg.adapt_interval = std::atoi(argv[i + 1]);
        else if (a == "--adapt-margin") cfg.adapt_margin = std::strtof(argv[i + 1], nullptr);
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
        std::printf("ready in %.1f s: %d resident experts, context %d, %.2f GiB of VRAM left\n",
                    std::chrono::duration<double>(clk::now() - t0).count(), dec.resident_experts(), cfg.max_ctx,
                    dec.vram_free_after_init() / 1073741824.0);
        int32_t next[8], draft[8] = {0};
        int pos = 0, last = 0;   // next[last]: the prediction after the prompt's final token
        const bool mtp = dec.has_mtp();
        t0 = clk::now();
        {   // the prompt (batched for long prompts unless --batched 0), then the first token greedily
            auto greedy = [](const float*, int32_t g) { return g; };
            if (!dec.prompt(prompt.data(), (int) prompt.size(), 0, greedy, &next[0], &draft[0], nullptr, err)) {
                std::fprintf(stderr, "FAIL in the prompt: %s\n", err.c_str());
                return 1;
            }
            pos = (int) prompt.size();
        }
        const double pf = std::chrono::duration<double>(clk::now() - t0).count();
        std::printf("prompt: %d tokens in %.2f s (%.1f tok/s), first next token %d; %llu experts swapped in after it (%.0f ms)\n",
                    pos, pf, pos / pf, next[last], (unsigned long long) dec.stats().swapped, dec.stats().swap_ms);
        const DecoderStats before = dec.stats();
        dec.reset_profile();   // the generation's windows only
        std::vector<int32_t> out;
        int32_t tok = next[last];
        int windows = 0, accepted = 0;
        double ms_main = 0, ms_mtp = 0;
        t0 = clk::now();
        if (mtp) {
            // verify [tok, d] at pos: the main model's prediction after tok decides whether d stands; the MTP layer
            // then takes the accepted positions (and drafts the next d from the last of them)
            int32_t d = draft[last];
            const bool two = dec.has_mtp2();
            while ((int) out.size() < generate) {
                int32_t d2 = 0;
                if (two) {
                    const auto m1 = clk::now();
                    const bool ok2 = dec.mtp_draft2(d, pos + 1, &d2, err);
                    ms_mtp += std::chrono::duration<double, std::milli>(clk::now() - m1).count();
                    if (!ok2) {
                        std::fprintf(stderr, "FAIL (draft 2) at %d: %s\n", pos, err.c_str());
                        return 1;
                    }
                }
                int32_t win[3] = {tok, d, d2};
                const int W = two ? 3 : 2;
                const auto m0 = clk::now();
                const bool ok_main = dec.forward(win, W, pos, next, nullptr, err);
                ms_main += std::chrono::duration<double, std::milli>(clk::now() - m0).count();
                if (!ok_main) {
                    std::fprintf(stderr, "FAIL at %d: %s\n", pos, err.c_str());
                    return 1;
                }
                ++windows;
                // the main model's picks decide: keep drafts while they agree, then its own next token
                int A = 1;
                out.push_back(tok);
                while (A < W && next[A - 1] == win[A]) {
                    out.push_back(win[A]);
                    ++accepted;
                    ++A;
                }
                tok = next[A - 1];
                int32_t mt[3];
                for (int j = 1; j < A; ++j) mt[j - 1] = win[j];
                mt[A - 1] = tok;
                const auto m1 = clk::now();
                const bool ok_mtp = dec.mtp_forward(mt, A, pos + 1, draft, err);
                ms_mtp += std::chrono::duration<double, std::milli>(clk::now() - m1).count();
                if (!ok_mtp) {
                    std::fprintf(stderr, "FAIL (MTP) at %d: %s\n", pos, err.c_str());
                    return 1;
                }
                d = draft[A - 1];
                pos += A;
            }
            out.resize((size_t) generate);
        } else {
            for (int i = 0; i < generate; ++i) {
                out.push_back(tok);
                if (!dec.forward(&tok, 1, pos, next, nullptr, err)) {
                    std::fprintf(stderr, "FAIL at %d: %s\n", pos, err.c_str());
                    return 1;
                }
                ++pos;
                tok = next[0];
            }
        }
        if (generate > 0) {
            const double gs = std::chrono::duration<double>(clk::now() - t0).count();
            const DecoderStats& st = dec.stats();
            const double uses = (double) (st.expert_uses - before.expert_uses);
            std::printf("generated %d tokens in %.2f s (%.2f tok/s); resident hit rate %.1f%%, %.1f experts staged per "
                        "token, CPU experts %.2f ms per token, waiting for the GPU %.2f ms per token\n",
                        generate, gs, generate / gs, 100.0 * (st.expert_hits - before.expert_hits) / uses,
                        (double) (st.staged - before.staged) / generate, (st.cpu_ms - before.cpu_ms) / generate,
                        (st.wait_ms - before.wait_ms) / generate);
            if (mtp)
                std::printf("MTP: %d windows, %d drafts accepted (%.1f%% of windows' drafts), %.2f tokens per window\n", windows, accepted,
                            100.0 * accepted / std::max(windows * (dec.has_mtp2() ? 2 : 1), 1), (double) generate / std::max(windows, 1));
            if (mtp)
                std::printf("per window: main %.2f ms, MTP %.2f ms\n", ms_main / std::max(windows, 1),
                            ms_mtp / std::max(windows, 1));
            dec.print_profile();
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
