// CPU-only include override for the ACTUAL src/program/mimo_serve.cpp.
// Build with tests/mimo/legacy_fake before include/. No model files or CUDA headers are used.
#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace strata::mimo {
constexpr int kSwaRing = 256;
constexpr int kMaxT = 8;
constexpr int kFakeVocab = 32749;

// Unsigned arithmetic makes the Python oracle bit-exact across compilers.
inline uint32_t fake_mix(uint32_t a, uint32_t b) {
    uint32_t x = a ^ (b + 0x9e3779b9u + (a << 6) + (a >> 2));
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    return x;
}
inline int32_t fake_token(uint32_t h) { return 1 + h % (kFakeVocab - 1); }

struct DecoderConfig {
    int slots = 1, max_ctx = 262144, batched_min_prompt = 128;
    double expert_budget_gib = 0;
    bool ram_copy = true;
    std::string mtp_path, profile;
};
struct DecoderStats { uint64_t expert_hits = 0, expert_uses = 0; };
struct MimoModel { explicit MimoModel(const std::string&) {} };

class MimoDecoder {
    using Ring = std::array<uint32_t, kSwaRing>;
    struct Sequence {
        Ring key[2]{}, mtp{};
        std::array<uint32_t, kMaxT> hidden{};
    };
    DecoderConfig cfg_;
    Sequence seq_[2];
    DecoderStats stats_;

    // Deliberately no position tags, repair, or stale-read exception. Rewinding beyond
    // live history consumes whatever a later position left at the same physical row.
    static uint32_t attend(const Ring& keys, int p, uint32_t self) {
        uint32_t h = fake_mix(0x12345678u, uint32_t(p));
        for (int j = std::max(0, p - 127); j <= p; ++j)
            h = fake_mix(h, j == p ? self : keys[size_t(j) % kSwaRing]);
        return h;
    }
    static uint32_t embedding(int32_t t, int p) {
        return fake_mix(uint32_t(t) + 17u, uint32_t(p));
    }
    static uint32_t main_hidden(const Sequence& s, int32_t t, int p) {
        const uint32_t h0 = attend(s.key[0], p, embedding(t, p));
        return attend(s.key[1], p, fake_mix(h0, 0xabcdef01u));
    }
    bool window(int slot, const int32_t* t, int n, int p, std::string& err) const {
        if (slot < 0 || slot >= cfg_.slots || !t || n < 1 || n > kMaxT || p < 0 || p + n > cfg_.max_ctx + 1) {
            err = "fake decoder: invalid window"; return false;
        }
        for (int i = 0; i < n; ++i) if (t[i] < 0 || t[i] >= vocab()) {
            err = "fake decoder: invalid token"; return false;
        }
        return true;
    }
public:
    using ChooseFn = std::function<int32_t(const float*, int32_t)>;
    struct PromptState { int slot = -1, next_pos = -1; bool finished = false; };
    MimoDecoder(MimoModel&, const DecoderConfig& cfg) : cfg_(cfg) {}
    bool init(std::string&) { return true; }
    int vocab() const { return kFakeVocab; }
    bool has_mtp() const { return !cfg_.mtp_path.empty(); }
    int resident_experts() const { return 0; }
    size_t vram_free_after_init() const { return 0; }
    size_t kv_bytes() const { return sizeof(seq_); }
    const DecoderStats& stats() const { return stats_; }
    bool save_profile(const std::string&, std::string&) const { return true; }
    bool reset_sequence(int slot, std::string& err) {
        if (slot < 0 || slot >= cfg_.slots) { err = "fake decoder: invalid slot"; return false; }
        // Includes the normally unwritten MTP position-zero sentinel. prompt() must
        // NOT clear this implicitly: the regression exercises the caller's reset.
        seq_[slot] = Sequence{};
        return true;
    }
    bool forward(int slot, const int32_t* t, int n, int p, int32_t* next,
                 std::vector<float>* logits, std::string& err) {
        if (!window(slot, t, n, p, err)) return false;
        auto& s = seq_[slot];
        std::array<uint32_t, kMaxT> h;
        for (int i = 0; i < n; ++i) h[i] = embedding(t[i], p + i);
        // Two causal 128-wide layers, layer-major like the real fixed-window path.
        // Layer two keys depend on layer one's hidden, not merely on token IDs:
        // a missing prefix cannot be reconstructed by replaying only 128 tokens.
        for (int l = 0; l < 2; ++l) {
            for (int i = 0; i < n; ++i) s.key[l][size_t(p + i) % kSwaRing] = h[i];
            for (int i = 0; i < n; ++i) {
                h[i] = attend(s.key[l], p + i, s.key[l][size_t(p + i) % kSwaRing]);
                if (l == 0) h[i] = fake_mix(h[i], 0xabcdef01u);
            }
        }
        if (logits) logits->assign(size_t(n) * vocab(), -1000.f);
        for (int i = 0; i < n; ++i) {
            s.hidden[i] = h[i];
            next[i] = fake_token(h[i]);
            if (logits) (*logits)[size_t(i) * vocab() + next[i]] = 0.f;
        }
        stats_.expert_uses += n;
        return true;
    }
    bool forward(const int32_t* t, int n, int p, int32_t* next, std::vector<float>* logits, std::string& err) {
        return forward(0, t, n, p, next, logits, err);
    }
    bool mtp_forward(int slot, const int32_t* t, int n, int p, int32_t* draft, std::string& err) {
        if (!window(slot, t, n, p, err)) return false;
        auto& s = seq_[slot];
        for (int i = 0; i < n; ++i)
            s.mtp[size_t(p + i) % kSwaRing] = fake_mix(embedding(t[i], p + i), s.hidden[i]);
        for (int i = 0; i < n; ++i) {
            const uint32_t h = attend(s.mtp, p + i, s.mtp[size_t(p + i) % kSwaRing]);
            // A toy draft head: correct on roughly a third of positions, otherwise
            // a hash token. The gate depends on ALL MTP attention history, so stale
            // predecessor/lookahead rows change acceptance, not just diagnostics.
            draft[i] = fake_token(h % 3 == 0 ? main_hidden(s, t[i], p + i) : h);
        }
        return true;
    }
    bool mtp_forward(const int32_t* t, int n, int p, int32_t* draft, std::string& err) {
        return mtp_forward(0, t, n, p, draft, err);
    }
    bool prompt_chunk(int slot, const int32_t* t, int n, int p, const ChooseFn& choose,
                      int32_t* first, int32_t* draft, const std::function<void(int)>& progress,
                      PromptState& state, bool final, std::string& err) {
        if (n < 1 || state.finished || (state.slot >= 0 && (state.slot != slot || state.next_pos != p))) {
            err = "fake decoder: invalid prompt chunk"; return false;
        }
        int32_t next[kMaxT], dr[kMaxT], mt[kMaxT];
        std::vector<float> logits;
        for (int done = 0; done < n;) {
            const int count = std::min(kMaxT, n - done);
            const bool last = done + count == n;
            if (!forward(slot, t + done, count, p + done, next, last ? &logits : nullptr, err)) return false;
            if (last) *first = choose(logits.data() + size_t(count - 1) * vocab(), next[count - 1]);
            if (has_mtp()) {
                for (int i = 0; i < count; ++i) mt[i] = done + i + 1 < n ? t[done + i + 1] : *first;
                // Shift by one, including the chosen-but-not-emitted lookahead.
                // In particular do not rewrite MTP row p at a retained-prefix start.
                if (!mtp_forward(slot, mt, count, p + done + 1, dr, err)) return false;
                if (last && draft) *draft = dr[count - 1];
            }
            done += count;
            if (progress) progress(done);
        }
        state.slot = slot; state.next_pos = p + n; state.finished = final;
        return true;
    }
    bool prompt(int slot, const int32_t* t, int n, int p, const ChooseFn& choose,
                int32_t* first, int32_t* draft, const std::function<void(int)>& progress, std::string& err) {
        PromptState state;
        return prompt_chunk(slot, t, n, p, choose, first, draft, progress, state, true, err);
    }
    bool prompt(const int32_t* t, int n, int p, const ChooseFn& choose, int32_t* first,
                int32_t* draft, const std::function<void(int)>& progress, std::string& err) {
        return prompt(0, t, n, p, choose, first, draft, progress, err);
    }
};
} // namespace strata::mimo
