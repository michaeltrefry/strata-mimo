// include/strata/mimo/decoder.hpp - MiMo-V2.6-Flash decode: windows of T = 1..8 consecutive tokens through all
// 48 layers, the logits of every token out.
//
// v1: the routed experts that are not resident are computed by the CPU (CpuExperts, step 2) while the GPU computes
// the resident ones; one host round trip per MoE layer, no CUDA graph yet.  With cpu_experts off it is v0:
// v0 is the correctness path of docs/MIMO_PORT_PLAN.md step 6: synchronous, no CUDA graph, every
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
#include <memory>
#include <string>
#include <vector>

namespace strata::kernels::cpu { class ExpertPool; }

namespace strata::mimo {

class CpuExperts;

struct DecoderConfig {
    int max_ctx = 262144;           // positions of the global layers' KV caches
    double expert_budget_gib = 0;   // VRAM for resident experts; 0 = everything left after the rest, minus a margin
    std::string profile;            // MIMOPROF file (tools/mimo/make_profile.py); empty = no resident experts
    std::string dump_dir;           // when set: every intermediate, named as llama.cpp names it, appended per window
    bool cpu_experts = true;        // non-resident experts on the CPU (v1); false: copied into GPU staging slots (v0)
    bool ram_copy = true;           // CPU experts read from a RAM copy of every MoE layer (else the GGUF mapping)
    bool graph = true;              // v2: one CUDA graph per window size, the CPU handoff inside it (no dump)
    bool profile_gpu = false;       // v2: GPU timestamps per layer (print_profile)
};

struct DecoderStats {
    uint64_t expert_uses = 0, expert_hits = 0;   // routed (token, expert) pairs, and those served from a resident slot
    uint64_t staged = 0;                         // experts copied into a staging slot (distinct per window and layer)
    double cpu_ms = 0;                           // time spent in CpuExperts::run_layer
    double wait_ms = 0;                          // v2: host waiting for the GPU to reach the next MoE layer
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
    /// With profile_gpu: per MoE layer, the averages since init of (layer start -> doorbell, doorbell -> resident
    /// experts done, -> CPU flag seen, -> layer end) in microseconds, printed as one table.
    void print_profile() const;
    void reset_profile() { prof_us_.assign(prof_us_.size(), 0.0); prof_windows_ = 0; }
    int resident_experts() const { return n_resident_; }

private:
    bool forward_graph(int T, int pos0, int32_t* next, std::string& err);
    bool capture(int T, std::string& err);
    void release_gpu_waits();
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
    // v1: the CPU's share
    std::unique_ptr<kernels::cpu::ExpertPool> pool_;
    std::unique_ptr<CpuExperts> cpu_;
    std::vector<void*> ram_;                   // RAM copies of the MoE layers' experts (VirtualAlloc / malloc)
    float *h_host_ = nullptr, *wt_host_ = nullptr, *cpu_out_host_ = nullptr;   // pinned
    int32_t* ids_host_ = nullptr;              // pinned
    float* cpu_out_ = nullptr;                 // device copy of the CPU's sums
    long long prev_affinity_ = -1;
    // v2: the window graphs and the mapped handoff (host pointer h_*, device alias m_*)
    cudaGraphExec_t graphs_[9] = {};
    int32_t* d_res_ = nullptr;                 // [layer][expert] -> resident slot or -1, on the device
    float *h_x_ = nullptr, *m_x_ = nullptr, *h_w_ = nullptr, *m_w_ = nullptr, *h_out_ = nullptr, *m_out_ = nullptr;
    int32_t *h_ids_ = nullptr, *m_ids_ = nullptr;
    uint32_t *h_seq_ = nullptr, *m_seq_ = nullptr, *h_flag_ = nullptr, *m_flag_ = nullptr;
    unsigned long long* stamps_ = nullptr;     // device [48 * 5] GPU %globaltimer stamps of the last window
    std::vector<double> prof_us_;              // accumulated [48 * 4] intervals
    int prof_windows_ = 0;
};

}  // namespace strata::mimo
