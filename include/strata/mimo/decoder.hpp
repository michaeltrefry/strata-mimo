// include/strata/mimo/decoder.hpp - MiMo-V2.6-Flash decode: windows of T = 1..8 consecutive tokens through all
// 48 layers, the logits of every token out.
//
// v0 (this file) is the correctness path of docs/MIMO_PORT_PLAN.md step 6: synchronous, no CUDA graph, every
// expert computed on the GPU.  The resident experts (the profile's most used, as many as the VRAM budget holds)
// stay in their slots; a routed expert that is not resident is copied from the mapped GGUF into a staging slot for
// that window.  Its job is to be right (checked against llama.cpp with tools/mimo/mimo_parity.py) and to give the
// hit rate; the fast path computes the misses on the CPU instead and runs as one graph.
#pragma once

#include "strata/mimo/attn.hpp"
#include "strata/mimo/model.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace strata::mimo {

struct DecoderConfig {
    int max_ctx = 262144;           // positions of the global layers' KV caches
    double expert_budget_gib = 0;   // VRAM for resident experts; 0 = everything left after the rest, minus a margin
    std::string profile;            // MIMOPROF file (tools/mimo/make_profile.py); empty = no resident experts
    std::string dump_dir;           // when set: every intermediate, named as llama.cpp names it, appended per window
};

struct DecoderStats {
    uint64_t expert_uses = 0, expert_hits = 0;   // routed (token, expert) pairs, and those served from a resident slot
    uint64_t staged = 0;                         // experts copied into a staging slot (distinct per window and layer)
};

class MimoDecoder {
public:
    MimoDecoder(MimoModel& model, const DecoderConfig& cfg);
    ~MimoDecoder();
    MimoDecoder(const MimoDecoder&) = delete;
    MimoDecoder& operator=(const MimoDecoder&) = delete;

    /// Allocates the KV caches and buffers and fills the resident expert slots. False with `err` on failure.
    bool init(std::string& err);

    /// Runs tokens[0..T) at positions pos0..pos0+T-1 (T <= 8; positions >= pos0 + T from earlier windows are
    /// forgotten).  Writes the greedy next token of each position to `next` (T values) and, when `logits` is not
    /// null, the full logits [T][n_vocab].
    bool forward(const int32_t* tokens, int T, int pos0, int32_t* next, std::vector<float>* logits, std::string& err);

    const DecoderStats& stats() const { return stats_; }
    int resident_experts() const { return n_resident_; }

private:
    void dump(const char* name, int layer, const void* dev, size_t bytes_per_token, int T, bool is_int = false);

    MimoModel& model_;
    DecoderConfig cfg_;
    DecoderStats stats_;
    cudaStream_t stream_ = nullptr;
    std::vector<void*> allocs_;     // device allocations owned here
    void* pinned_ = nullptr;        // host staging for expert uploads
    std::vector<AttnKv> kv_;
    std::vector<AttnLayer> attn_;
    AttnScratch attn_scratch_{};
    // activations ([T][...] each)
    float *x_ = nullptr, *h_ = nullptr, *qkv_ = nullptr, *q_ = nullptr, *a_ = nullptr, *y_ = nullptr;
    float *g_ = nullptr, *u_ = nullptr, *m_ = nullptr, *logits_ = nullptr, *router_logits_ = nullptr, *wt_ = nullptr;
    int32_t *ids_ = nullptr, *tok_ = nullptr, *pos0_ = nullptr, *slots_ = nullptr, *next_ = nullptr;
    void* q8_ = nullptr;            // Q8_1 scratch of the dense GEMVs
    void* mx_scratch_ = nullptr;    // expert kernel scratch
    void* raw_ = nullptr;           // one expert in GGUF layout, before relayout into a staging slot
    unsigned long long* slot_ptr_ = nullptr;   // device address of every slot: resident first, then staging
    int n_resident_ = 0, n_staging_ = 0;
    std::vector<std::vector<int>> resident_;   // [layer][expert] -> slot, -1 = not resident
    std::vector<unsigned long long> slot_addr_;
};

}  // namespace strata::mimo
