// Model-required assertion test, never a default CTest: it occupies a full MiMo model's RAM/VRAM.
// mimo_decoder_slots_test MODEL.gguf MTP.gguf [PROFILE [EXPERT_BUDGET_GIB [MTP_HEADS]]]
// No profile: decode-window prompt chunks. With a profile: also exercises borrowed batched prompt buffers.
#include "strata/mimo/decoder.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

using strata::mimo::DecoderConfig;
using strata::mimo::MimoDecoder;
using strata::mimo::MimoModel;

namespace {
void require(bool value, const std::string& message) {
    if (!value) throw std::runtime_error(message);
}
struct Result {
    std::vector<float> logits;
    std::vector<int32_t> ids;
};
void compare(const Result& a, const Result& b, const std::string& label) {
    require(a.ids == b.ids, label + ": greedy/MTP predictions differ");
    require(a.logits.size() == b.logits.size(), label + ": logits shape differs");
    for (size_t i = 0; i < a.logits.size(); ++i) {
        const float x = a.logits[i], y = b.logits[i];
        require(std::isfinite(x) && std::isfinite(y), label + ": non-finite logits");
        if (std::abs(x - y) > 1e-4f + 1e-4f * std::max(std::abs(x), std::abs(y)))
            throw std::runtime_error(label + ": logit " + std::to_string(i) + " differs: " +
                                     std::to_string(x) + " vs " + std::to_string(y));
    }
}

struct Sequence {
    std::vector<int32_t> prompt;
    MimoDecoder::PromptState state;
    Result result;
    int cursor = 0, pos = 0, step = 0;
    int32_t first = 0, draft = 0;
    int seed;
    explicit Sequence(int n, int seed) : prompt((size_t) n), seed(seed) {
        // Fixed distinct in-vocabulary inputs, not generated text or a parity script that only prints differences.
        for (int i = 0; i < n; ++i) prompt[(size_t) i] = 32 + (seed * 17 + i * 13) % 200;
    }
    bool prompted() const { return cursor == (int) prompt.size(); }
    void prompt_step(MimoDecoder& dec, int slot) {
        std::string err;
        const int n = std::min(137, (int) prompt.size() - cursor);
        const bool final = cursor + n == (int) prompt.size();
        auto choose = [&](const float* logits, int32_t greedy) {
            if (!final) return prompt[(size_t) cursor + n];  // real next prompt token, no sampling
            result.logits.insert(result.logits.end(), logits, logits + dec.vocab());
            return greedy;
        };
        require(dec.prompt_chunk(slot, prompt.data() + cursor, n, cursor, choose, &first, &draft, nullptr,
                                 state, final, err), "prompt: " + err);
        cursor += n;
        pos = cursor;
        if (final) {
            result.ids.push_back(first);
            result.ids.push_back(draft);
            if (dec.has_mtp2()) {
                int32_t d2 = 0;
                require(dec.mtp_draft2(slot, draft, pos + 1, &d2, err), "prompt draft2: " + err);
                result.ids.push_back(d2);
            }
        }
    }
    void decode_step(MimoDecoder& dec, int slot) {
        std::string err;
        int32_t tok[3], next[3], mt[3], drafts[3];
        for (int t = 0; t < 3; ++t) tok[t] = 32 + (seed * 19 + step * 23 + t * 11) % 200;
        std::vector<float> logits;
        require(dec.forward(slot, tok, 3, pos, next, &logits, err), "forward: " + err);
        result.logits.insert(result.logits.end(), logits.begin(), logits.end());
        result.ids.insert(result.ids.end(), next, next + 3);
        // Alternating acceptance/rejection rewrites the speculative tail on subsequent windows.
        const int accepted = step % 2 ? 3 : 1;
        for (int t = 0; t < accepted; ++t) mt[t] = t + 1 < accepted ? tok[t + 1] : next[t];
        require(dec.mtp_forward(slot, mt, accepted, pos + 1, drafts, err), "MTP: " + err);
        result.ids.insert(result.ids.end(), drafts, drafts + accepted);
        if (dec.has_mtp2()) {
            int32_t d2 = 0;
            require(dec.mtp_draft2(slot, drafts[accepted - 1], pos + accepted + 1, &d2, err), "draft2: " + err);
            result.ids.push_back(d2);
        }
        pos += accepted;
        ++step;
    }
};

// Small chunks keep every reference/reuse case on the same decode-window prompt path, even in the optional
// batched exercise. Otherwise a GEMM/GEMV rounding difference could hide the one-position-ahead MTP bug.
void prompt_range(MimoDecoder& dec, int slot, const std::vector<int32_t>& tokens, int begin, int end,
                  bool final, MimoDecoder::PromptState& state, Result& result) {
    std::string err;
    for (int cursor = begin; cursor < end;) {
        const int n = std::min(31, end - cursor);
        const bool last = final && cursor + n == end;
        int32_t first = 0, draft = 0;
        auto choose = [&](const float* logits, int32_t greedy) {
            if (!last) return tokens[(size_t) cursor + n];
            result.logits.insert(result.logits.end(), logits, logits + dec.vocab());
            return greedy;
        };
        const bool ok = dec.prompt_chunk(slot, tokens.data() + cursor, n, cursor, choose, &first, &draft,
                                         nullptr, state, last, err);
        require(ok, "re-prompt chunk: " + err);
        cursor += n;
        if (last) {
            result.ids.push_back(first);
            result.ids.push_back(draft);
            if (dec.has_mtp2()) {
                int32_t d2 = 0;
                const bool ok2 = dec.mtp_draft2(slot, draft, end + 1, &d2, err);
                require(ok2, "re-prompt draft2: " + err);
                result.ids.push_back(d2);
            }
        }
    }
}

void compare_diagnostic(const Result& a, const Result& b, const std::string& label) {
    float maximum = 0;
    if (a.logits.size() == b.logits.size())
        for (size_t i = 0; i < a.logits.size(); ++i) maximum = std::max(maximum, std::abs(a.logits[i] - b.logits[i]));
    std::fprintf(stderr, "CHECK %s: %zu logits, max absolute error %.9g\n", label.c_str(),
                 a.logits.size(), maximum);
    compare(a, b, label);
}

void reprompt_regressions(MimoDecoder& dec, bool graph) {
    enum Kind { divergent, accepted_extension, canceled_chunk };
    for (Kind kind : {divergent, accepted_extension, canceled_chunk}) {
        std::string err;
        const std::string label = "graph=" + std::to_string(graph) + " " +
            (kind == divergent ? "divergent suffix" :
             kind == accepted_extension ? "accepted context extension" : "canceled chunk");
        std::vector<int32_t> old(217), target;
        for (size_t i = 0; i < old.size(); ++i) old[i] = 32 + (int) ((i * 37 + 29) % 200);
        int common = 186;
        if (kind == accepted_extension) common = 189;  // 186 prompt positions + three accepted positions
        target.assign(old.begin(), old.begin() + common);
        for (int i = common; i < 233; ++i) target.push_back(32 + (i * 17 + 113) % 200);
        require(target[(size_t) common] != old[(size_t) common], "fixture must replace old MTP lookahead");
        const int start = common - 1;  // replay the predecessor to rebuild MTP(common) with the NEW token

        // A genuinely fresh target history, with the same final suffix boundaries as the retained run.
        require(dec.reset_sequence(1, err), "reset fresh re-prompt: " + err);
        MimoDecoder::PromptState fresh_state;
        Result fresh, fresh_prefix;
        if (kind == accepted_extension) {
            // Match the retained side's T=3 main history exactly. A prompt through188 would use T=2 at186
            // instead, confounding window-width arithmetic with the MTP lookahead being tested here.
            prompt_range(dec, 1, target, 0, 186, false, fresh_state, fresh);
            int32_t next[3], drafts[3];
            const bool main_ok = dec.forward(1, target.data() + 186, 3, 186, next, &fresh_prefix.logits, err);
            require(main_ok, "fresh accepted main window: " + err);
            fresh_prefix.ids.assign(next, next + 3);
            const int32_t mt[3] = {target[187], target[188], target[189]};
            const bool mtp_ok = dec.mtp_forward(1, mt, 3, 187, drafts, err);
            require(mtp_ok, "fresh accepted MTP window: " + err);
            if (dec.has_mtp2()) {
                int32_t d2 = 0;
                const bool ok2 = dec.mtp_draft2(1, drafts[2], 190, &d2, err);
                require(ok2, "fresh accepted second draft: " + err);
            }
        } else {
            prompt_range(dec, 1, target, 0, start, false, fresh_state, fresh);
        }
        MimoDecoder::PromptState fresh_replay;
        prompt_range(dec, 1, target, start, (int) target.size(), true, fresh_replay, fresh);
        Sequence fresh_tail(0, 19);
        fresh_tail.pos = (int) target.size();
        fresh_tail.result = fresh;
        fresh_tail.decode_step(dec, 1);
        fresh_tail.decode_step(dec, 1);

        require(dec.reset_sequence(0, err), "reset old re-prompt: " + err);
        MimoDecoder::PromptState old_state;
        Result ignored;
        if (kind == divergent) {
            // Old suffix and its sampled lookahead are present, but remain inside the ring's safe rewind range.
            prompt_range(dec, 0, old, 0, (int) old.size(), true, old_state, ignored);
        } else {
            // Cancel after a complete nonfinal chunk: main [0,186), MTP through position 186. Its state is
            // intentionally abandoned. For the extension case, accept a complete fixed three-token window next.
            prompt_range(dec, 0, old, 0, 186, false, old_state, ignored);
            if (kind == accepted_extension) {
                int32_t next[3], drafts[3];
                Result old_prefix;
                const bool main_ok = dec.forward(0, old.data() + 186, 3, 186, next, &old_prefix.logits, err);
                require(main_ok, "accepted main window: " + err);
                old_prefix.ids.assign(next, next + 3);
                compare_diagnostic(fresh_prefix, old_prefix, label + " pre-suffix T3 main history");
                require(target[(size_t) common] != next[2], "extension must replace accepted window's MTP lookahead");
                const int32_t mt[3] = {old[187], old[188], next[2]};
                const bool mtp_ok = dec.mtp_forward(0, mt, 3, 187, drafts, err);
                require(mtp_ok, "accepted MTP window: " + err);
                if (dec.has_mtp2()) {
                    int32_t d2 = 0;
                    const bool ok2 = dec.mtp_draft2(0, drafts[2], 190, &d2, err);
                    require(ok2, "accepted second draft: " + err);
                }
            }
        }
        // No reset: preserve actual main/MTP prefix contents, discard old request routing state, then rewind one.
        MimoDecoder::PromptState reused_state;
        Result reused;
        prompt_range(dec, 0, target, start, (int) target.size(), true, reused_state, reused);
        compare_diagnostic(fresh, reused, label + " fresh vs retained logits/MTP");
        Sequence reused_tail(0, 19);
        reused_tail.pos = (int) target.size();
        reused_tail.result = reused;
        reused_tail.decode_step(dec, 0);
        reused_tail.decode_step(dec, 0);
        compare_diagnostic(fresh_tail.result, reused_tail.result, label + " continuation");
    }
}

Result isolated(MimoDecoder& dec, int seed, int n) {
    std::string err;
    require(dec.reset_sequence(0, err), "reset baseline: " + err);
    Sequence seq(n, seed);
    while (!seq.prompted()) seq.prompt_step(dec, 0);
    for (int i = 0; i < 6; ++i) seq.decode_step(dec, 0);
    return seq.result;
}

std::vector<Result> exercise(MimoModel& model, DecoderConfig cfg) {
    MimoDecoder dec(model, cfg);  // one decoder only, shared model and scratch
    std::string err;
    require(dec.init(err), "init: " + err);
    const Result a = isolated(dec, 1, 281), b = isolated(dec, 7, 337);
    require(dec.reset_sequence(0, err), "reset A: " + err);
    require(dec.reset_sequence(1, err), "reset B: " + err);
    Sequence sa(281, 1), sb(337, 7);
    // Main/MTP graphs bind different KV pointers, and unequal prompt lengths cross the SWA ring boundary.
    while (!sa.prompted() || !sb.prompted()) {
        if (!sa.prompted()) sa.prompt_step(dec, 0);
        if (!sb.prompted()) sb.prompt_step(dec, 1);
    }
    for (int i = 0; i < 6; ++i) {
        sa.decode_step(dec, 0);
        sb.decode_step(dec, 1);
    }
    compare(a, sa.result, "isolated vs interleaved A");
    compare(b, sb.result, "isolated vs interleaved B");

    // A retained prefix stays untouched while B is recycled and rebuilt; A continues without reset/re-prefill.
    sa.decode_step(dec, 0);
    const Result extended_a = sa.result;
    require(dec.reset_sequence(1, err), "recycle B: " + err);
    Sequence recycled(337, 7);
    while (!recycled.prompted()) recycled.prompt_step(dec, 1);
    for (int i = 0; i < 6; ++i) recycled.decode_step(dec, 1);
    compare(b, recycled.result, "recycled slot B");
    sa.decode_step(dec, 0);
    const Result continued_a = sa.result;
    require(dec.reset_sequence(0, err), "rebuild A: " + err);
    Sequence rebuilt(281, 1);
    while (!rebuilt.prompted()) rebuilt.prompt_step(dec, 0);
    for (int i = 0; i < 7; ++i) rebuilt.decode_step(dec, 0);
    compare(extended_a, rebuilt.result, "peer reset preserves A prefix");
    rebuilt.decode_step(dec, 0);
    compare(continued_a, rebuilt.result, "peer prompt/decode preserves A continuation");
    require(dec.kv_bytes() > 0, "missing cache byte accounting");
    reprompt_regressions(dec, cfg.graph);
    std::printf("PASS graph=%d, batched=%d, two slots, main/MTP, rewind, chunking, retained prefix and recycling\n",
                cfg.graph, cfg.batched_prefill);
    return {a, b};
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 3 || argc > 6) {
        std::fprintf(stderr, "usage: %s MODEL.gguf MTP.gguf [PROFILE [EXPERT_BUDGET_GIB [MTP_HEADS]]]\n"
                             "Full-model test: do not run alongside a live service.\n", argv[0]);
        return 2;
    }
    try {
        DecoderConfig cfg;
        cfg.slots = 2;
        cfg.max_ctx = 1024;
        cfg.mtp_path = argv[2];
        cfg.mtp_heads = argc == 6 ? std::atoi(argv[5]) : 1;
        require(cfg.mtp_heads == 1 || cfg.mtp_heads == 2, "MTP_HEADS must be 1 or 2");
        cfg.ram_copy = false;     // identical expert bytes from the mapping; do not duplicate ~150 GiB for a test
        cfg.adapt_decode = false;  // fixed residency so interleaving does not change kernels/expert placement
        cfg.adapt_min_prompt = 2048;
        cfg.batched_prefill = false;
        if (argc >= 4) cfg.profile = argv[3];
        if (argc >= 5) cfg.expert_budget_gib = std::strtod(argv[4], nullptr);
        MimoModel model(argv[1]);
        cfg.graph = false;
        const auto reference = exercise(model, cfg);
        cfg.graph = true;
        const auto graphed = exercise(model, cfg);
        compare(reference[0], graphed[0], "graph off/on A");
        compare(reference[1], graphed[1], "graph off/on B");
        if (!cfg.profile.empty()) {
            cfg.batched_prefill = true;
            cfg.prefill_slice = 512;
            cfg.prefill_tile = 128;
            exercise(model, cfg);  // compare identical batched kernels; GEMM vs GEMV rounding is not conflated
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
