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
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace strata::kernels::cpu { class ExpertPool; }
namespace strata::prefill { class Gemm; }

namespace strata::mimo {

class CpuExperts;
class PrefillMoe;

struct DecoderConfig {
    int slots = 1;                  // independent sequences (1 or 2); all compute and weights stay shared
    int max_ctx = 262144;           // positions of the global layers' KV caches
    double expert_budget_gib = 0;   // VRAM for resident experts; 0 = everything left after the rest, minus a margin
    std::string profile;            // MIMOPROF file (tools/mimo/make_profile.py); empty = no resident experts
    std::string dump_dir;           // when set: every intermediate, named as llama.cpp names it, appended per window
    bool cpu_experts = true;        // non-resident experts on the CPU (v1); false: copied into GPU staging slots (v0)
    bool ram_copy = true;           // CPU experts read from a RAM copy of every MoE layer (else the GGUF mapping)
    bool graph = true;              // v2: one CUDA graph per window size, the CPU handoff inside it (no dump)
    bool profile_gpu = false;       // v2: GPU timestamps per layer (print_profile)
    std::string mtp_path;           // the MTP GGUF (mtp-*.gguf): enables mtp_forward (draft head = its first layer)
    int mtp_heads = 1;              // 2: also the second nextn layer, for a second draft token (mtp_draft2)
    bool batched_prefill = true;    // prompts of more than 64 tokens through the batched path (src/mimo/prefill.cpp)
    int batched_min_prompt = 128;   // shorter prompts run through decode windows (faster below ~128 tokens)
    int prefill_slice = 16384;       // tokens per layer-major slice (each non-resident expert copied once per slice)
    int prefill_tile = 2048;        // tokens per attention / dense GEMM tile within a slice
    int adapt_min_prompt = 2048;    // after a batched prompt this long, the resident set follows its routing too
    float adapt_weight = 0.5f;      // the prompt's share in the blended score (the profile has the rest)
    // residency that follows the generation (v2 only): recent picks decay with this half-life; every
    // adapt_interval windows up to adapt_swaps hot non-resident experts are copied into spare slots in the
    // background and replace resident experts scoring at least adapt_margin times lower.  Trickle, not bursts: the
    // copies share PCIe with the per-layer CPU handoff - 8 swaps every 16 windows cost 30% on a short prompt, 2 per
    // window cost nothing and gave the most (docs/MIMO_DECODER.md)
    bool adapt_decode = true;
    int adapt_spares = 8, adapt_interval = 1, adapt_swaps = 2;
    float adapt_half_life = 256.f, adapt_recent_weight = 0.5f, adapt_margin = 1.2f;
};

struct DecoderStats {
    uint64_t expert_uses = 0, expert_hits = 0;   // routed (token, expert) pairs, and those served from a resident slot
    uint64_t staged = 0;                         // experts copied into a staging slot (distinct per window and layer)
    double cpu_ms = 0;                           // time spent in CpuExperts::run_layer
    double wait_ms = 0;                          // v2: host waiting for the GPU to reach the next MoE layer
    uint64_t swapped = 0;                        // experts moved into resident slots after prompts
    double swap_ms = 0;
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

    /// MTP draft (needs cfg.mtp_path).  Runs the first nextn layer over positions pos0 .. pos0+T-1 with tokens[t] and
    /// the main model's final hidden state of the previous position, which must be row t of the last forward()'s
    /// window (so call this right after forward(), with pos0 = that window's pos0 + 1).  Writes the MTP layer's
    /// greedy prediction for each position + 1 to draft[t]; it also fills the layer's own KV (sliding window).
    bool mtp_forward(const int32_t* tokens, int T, int pos0, int32_t* draft, std::string& err);
    bool has_mtp() const { return mtp_arena_ != nullptr; }
    bool has_mtp2() const { return mtp2_arena_ != nullptr; }
    /// The second draft: the second nextn layer at position `pos` with token d1 (the first layer's draft from the last
    /// mtp_forward) and that layer's state at its last row.  Its KV entry is in the draft region and is rewritten by
    /// the next mtp_forward over accepted positions.
    bool mtp_draft2(int32_t d1, int pos, int32_t* d2, std::string& err);

    /// The prompt tokens[0..n) at positions pos0.. (the KV before pos0 must hold the same tokens), then, with MTP,
    /// the MTP layer's KV over the same positions.  `choose(logits of the last position, its greedy id)` picks the
    /// first generated token (sampling, or just the greedy id); it is returned in *first, and with MTP the draft for
    /// the position after it in *draft.  `progress(done)` is called after each window of the prompt.
    using ChooseFn = std::function<int32_t(const float* logits, int32_t greedy)>;
    bool prompt(const int32_t* tokens, int n, int pos0, const ChooseFn& choose, int32_t* first, int32_t* draft,
                const std::function<void(int)>& progress, std::string& err);
    // All execution calls are serialized by the caller. Finish main/MTP/draft2 before switching slots.
    bool forward(int slot, const int32_t* tokens, int T, int pos0, int32_t* next, std::vector<float>* logits,
                 std::string& err);
    bool mtp_forward(int slot, const int32_t* tokens, int T, int pos0, int32_t* draft, std::string& err);
    bool mtp_draft2(int slot, int32_t d1, int pos, int32_t* d2, std::string& err);
    bool prompt(int slot, const int32_t* tokens, int n, int pos0, const ChooseFn& choose, int32_t* first,
                int32_t* draft, const std::function<void(int)>& progress, std::string& err);
    // Fresh invalidation only, NOT retained-prefix continuation. Graph addresses remain valid.
    bool reset_sequence(int slot, std::string& err);
    struct PromptState {
        std::vector<std::vector<uint32_t>> counts;
        uint64_t tokens = 0;
        int slot = -1, next_pos = -1;
        bool finished = false;
    };
    // Nonfinal choose must return the next actual prompt token, without consuming sampling RNG.
    // Each return is a safe scheduling boundary: the borrowed expert slots have been refilled.
    bool prompt_chunk(int slot, const int32_t* tokens, int n, int pos0, const ChooseFn& choose, int32_t* first,
                      int32_t* draft, const std::function<void(int)>& progress, PromptState& state, bool final,
                      std::string& err);
    size_t kv_bytes() const {
        size_t bytes = 0;
        for (const auto& seq : sequences_) {
            for (const auto& kv : seq->kv) bytes += attn_kv_bytes(kv.n_kv, kv.cap);
            if (seq->mtp_kv.k) bytes += attn_kv_bytes(seq->mtp_kv.n_kv, seq->mtp_kv.cap);
            if (seq->mtp2_kv.k) bytes += attn_kv_bytes(seq->mtp2_kv.n_kv, seq->mtp2_kv.cap);
        }
        return bytes;
    }
    int vocab() const { return model_.geometry().n_vocab; }
    size_t vram_free_after_init() const { return vram_free_after_init_; }

    const DecoderStats& stats() const { return stats_; }
    /// The profile blended with every pick counted since start (each side weighted by the tokens behind it), as a
    /// MIMOPROF file: what this session's usage taught, for the next start's --expert-profile.
    bool save_profile(const std::string& path, std::string& err) const;
    /// With profile_gpu: per MoE layer, the averages since init of (layer start -> doorbell, doorbell -> resident
    /// experts done, -> CPU flag seen, -> layer end) in microseconds, printed as one table.
    void print_profile() const;
    void reset_profile() { prof_us_.assign(prof_us_.size(), 0.0); prof_windows_ = 0; }
    int resident_experts() const { return n_resident_; }

private:
    bool forward_graph(int slot, int T, int pos0, int32_t* next, std::string& err);
    // the batched prompt path (src/mimo/prefill.cpp)
    bool prefill_alloc(std::string& err);
    bool prefill_place(std::string& err);
    bool commit_swaps(bool wait, std::string& err);
    bool borrow_slots(std::string& err);
    bool refill_slots(const std::vector<std::vector<uint32_t>>* counts, std::string& err);
    bool prompt_batched(int slot, const int32_t* tokens, int n, int pos0, const ChooseFn& choose, int32_t* first, int32_t* draft,
                        const std::function<void(int)>& progress, PromptState& state, bool final, std::string& err);
    void gemm_q8(const DevTensor& w, uint16_t* scratch, bool dequant, const uint16_t* x16, float* y, int T);
    void prefill_attention(const DevTensor& attn_norm, const DevTensor& qkv, const DevTensor& out, const AttnLayer& L,
                           const AttnKv& kv, float* x, int T, int pos, bool dq_qkv, bool dq_out);
    void prefill_dense_ffn(const DevTensor& gate, const DevTensor& up, const DevTensor& down, const float* hn, float* x,
                           int T);
    void prefill_dump(const char* name, int layer, const float* dev, size_t floats_per_row, int rows);
    /// Re-ranks the experts by the profile blended with `counts` (a prompt's picks per layer and expert) and moves
    /// the best non-resident ones into the slots of the worst resident ones.
    bool adapt_residency(const std::vector<std::vector<uint32_t>>& counts, std::string& err);
    // decode-time residency (v2): the picks of each window, commits of finished background copies, new copies
    void adapt_count(int l, const int32_t* ids, int n);
    bool adapt_window(int T, std::string& err);
    struct MtpLayer;
    bool load_mtp(std::string& err);
    bool load_mtp_layer(int k, MtpLayer& m, void*& arena, AttnLayer& al, std::string& err);
    void mtp_layer_enqueue(const MtpLayer& m, const AttnKv& kv, const AttnLayer& al, const int32_t* tok,
                           const int32_t* pos, const float* hsrc, float* mres, int T, bool head);
    void mtp_enqueue(int slot, int T);
    bool capture(int slot, int T, std::string& err);
    void release_gpu_waits();
    void dump(const char* name, int layer, const void* dev, size_t bytes_per_token, int T, bool is_int = false);

    MimoModel& model_;
    DecoderConfig cfg_;
    DecoderStats stats_;
    cudaStream_t stream_ = nullptr;
    std::vector<void*> allocs_;     // device allocations owned here
    void* pinned_ = nullptr;        // host staging for expert uploads
    struct Sequence {
        std::vector<AttnKv> kv;
        AttnKv mtp_kv{}, mtp2_kv{};
        cudaGraphExec_t graphs[9] = {}, mtp_graphs[9] = {};
        cudaGraphExec_t draft2_graph = nullptr;
    };
    std::vector<std::unique_ptr<Sequence>> sequences_;   // stable KV addresses until destruction
    bool valid_slot(int slot, std::string& err) const;
    bool initialized_ = false;
    bool failed_ = false;                       // unsafe GPU/borrow failure: no further execution
    int hidden_slot_ = -1, mtp_hidden_slot_ = -1;
    std::vector<std::vector<uint32_t>>* prompt_counts_ = nullptr;
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
    size_t vram_free_after_init_ = 0;
    std::vector<std::vector<int>> resident_;   // [layer][expert] -> slot, -1 = not resident
    std::vector<float> profile_score_;         // [layer][expert] the profile's share of its layer's picks
    uint64_t profile_tokens_ = 0;              // the tokens behind it (its header)
    std::vector<double> usage_;                // [layer][expert] every pick counted since start
    uint64_t usage_tokens_ = 0;
    std::vector<float> recent_;                // [layer][expert] decayed recent picks (in units of 1 / recent_scale_)
    float recent_scale_ = 1.f;
    std::vector<int> spares_;                  // slots holding no resident expert (copy targets)
    struct Swap { int layer, expert, slot; cudaEvent_t done; };
    std::vector<Swap> in_flight_;
    cudaStream_t copy_stream_ = nullptr;
    void* raw_copy_ = nullptr;                 // the copy stream's GGUF-layout buffer
    int windows_since_adapt_ = 0;
    std::vector<unsigned long long> slot_addr_;
    // v1: the CPU's share
    std::unique_ptr<kernels::cpu::ExpertPool> pool_;
    std::unique_ptr<CpuExperts> cpu_;
    std::vector<void*> ram_;                   // RAM copies of the MoE layers' experts (cudaHostAlloc or malloc)
    bool ram_pinned_ = false;
    float *h_host_ = nullptr, *wt_host_ = nullptr, *cpu_out_host_ = nullptr;   // pinned
    int32_t* ids_host_ = nullptr;              // pinned
    float* cpu_out_ = nullptr;                 // device copy of the CPU's sums
    long long prev_affinity_ = -1;
    // v2: the window graphs and the mapped handoff (host pointer h_*, device alias m_*)
    int32_t* d_res_ = nullptr;                 // [layer][expert] -> resident slot or -1, on the device
    float *h_x_ = nullptr, *m_x_ = nullptr, *h_w_ = nullptr, *m_w_ = nullptr, *h_out_ = nullptr, *m_out_ = nullptr;
    int32_t *h_ids_ = nullptr, *m_ids_ = nullptr;
    uint32_t *h_seq_ = nullptr, *m_seq_ = nullptr, *h_flag_ = nullptr, *m_flag_ = nullptr;
    unsigned long long* stamps_ = nullptr;     // device [48 * 5] GPU %globaltimer stamps of the last window
    // MTP: the first nextn layer (blk.48 of the MTP GGUF); the embedding and the head are the main model's
    struct MtpLayer {
        DevTensor eh_proj, enorm, hnorm, attn_norm, attn_qkv, attn_output, attn_sinks, ffn_norm, ffn_gate, ffn_up,
            ffn_down, out_norm;
    } mtp_, mtp2_;
    void* mtp_arena_ = nullptr;
    void* mtp2_arena_ = nullptr;
    AttnLayer mtp_attn_{}, mtp2_attn_{};
    float *mres2_ = nullptr, *hd_ = nullptr;   // the second layer's residual; the first layer's last row
    int32_t *d2_tok_ = nullptr, *d2_pos_ = nullptr;
    float *e_ = nullptr, *cat_ = nullptr, *mres_ = nullptr;
    int32_t *mtp_tok_ = nullptr, *mtp_pos0_ = nullptr;
    // the batched prompt path's buffers (slice S, tile P)
    std::unique_ptr<prefill::Gemm> gemm_;
    std::unique_ptr<PrefillMoe> pmoe_;
    AttnPromptScratch aps_{};
    float *pX_ = nullptr, *pHn_ = nullptr, *pY_ = nullptr, *pW_ = nullptr, *pLogits_ = nullptr;   // [S][..]
    float *pQkv_ = nullptr, *pA_ = nullptr, *pG_ = nullptr, *pU_ = nullptr, *pE_ = nullptr;        // [P][..]
    int32_t *pIds_ = nullptr, *pTok_ = nullptr;
    uint16_t *p16_ = nullptr, *pQ16_ = nullptr, *pW1_ = nullptr, *pW2_ = nullptr;                // FP16
    // The prompt buffers live in the expert slots of the last slot chunks ("borrowed"): those slots hold resident
    // experts between prompts, are emptied for a batched prompt, and are filled again after it.
    struct PfBuf { void** ptr; size_t bytes; };
    std::vector<PfBuf> pf_bufs_;
    int borrow_lo_ = -1;                       // first borrowed slot (all slots from it on), -1 = not placed
    int lost_spares_ = 0;                      // spare slots that were in the borrowed range during a prompt
    std::vector<double> prof_us_;              // accumulated [48 * 4] intervals
    int prof_windows_ = 0;
};

}  // namespace strata::mimo
