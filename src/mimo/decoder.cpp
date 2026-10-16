#include "strata/mimo/decoder.hpp"

#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/mimo/cpu_experts.hpp"
#include "strata/mimo/expert_mxfp4.hpp"
#include "strata/mimo/fused.hpp"
#include "strata/mimo/ops.hpp"
#include "strata/mimo/prefill_moe.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/mimo/router.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <climits>
#include <cmath>
#include <immintrin.h>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <numeric>

namespace strata::mimo {
namespace cpu = strata::kernels::cpu;
namespace {
constexpr int kMaxT = 8;
constexpr size_t kMiB = 1u << 20;

// After a CUDA or borrowed-workspace failure, no peer sequence may continue on uncertain shared state.
struct RunGuard {
    bool& failed;
    bool complete = false;
    ~RunGuard() { if (!complete) failed = true; }
};

bool cuda_ok(cudaError_t st, const char* what, std::string& err) {
    if (st == cudaSuccess) return true;
    err = std::string(what) + ": " + cudaGetErrorString(st);
    return false;
}

// y = W x for T tokens, W in its GGUF blocks (Q8_0, or a K-quant the attention may be re-quantized to)
void q8_0_gemv(const DevTensor& w, const float* x, void* q8, float* y, int T, cudaStream_t s) {
    if (w.type == kTypeQ8_0) {
        strata::kernels::native_q8_0_f32(w.data, x, q8, y, (int) w.ne0, (int) w.ne1, T, s);
        return;
    }
    strata::kernels::native_quantize_q8_1(x, q8, (int) w.ne0, T, s);
    strata::kernels::native_mmvq((int) w.type, w.data, q8, y, (int) w.ne0, (int) w.ne1, T, s);
}
// the same with x already quantized (mimo_add_rmsnorm_q8_1, mimo_attn_decode's q8_1 output)
void q8_0_mmvq(const DevTensor& w, const void* q8, float* y, int T, cudaStream_t s) {
    strata::kernels::native_q8_0_mmvq(w.data, q8, y, (int) w.ne0, (int) w.ne1, T, s);
}

// MIMOPROF v1 (tools/mimo/make_profile.py): score[layer][expert]
bool read_profile(const std::string& path, int n_layer, int n_expert, std::vector<float>& score, std::string& err,
                  uint64_t* tokens_out = nullptr) {
    std::ifstream f(path, std::ios::binary);
    char magic[8];
    uint32_t hdr[3];
    uint64_t tokens = 0;
    if (!f.read(magic, 8) || std::memcmp(magic, "MIMOPROF", 8) != 0 || !f.read((char*) hdr, 12) ||
        !f.read((char*) &tokens, 8) || hdr[0] != 1) {
        err = "not a MIMOPROF v1 profile: " + path;
        return false;
    }
    if ((int) hdr[1] != n_layer || (int) hdr[2] != n_expert) {
        err = "profile " + path + " is for " + std::to_string(hdr[1]) + " layers x " + std::to_string(hdr[2]) + " experts";
        return false;
    }
    if (tokens_out) *tokens_out = tokens;
    score.resize((size_t) n_layer * n_expert);
    if (!f.read((char*) score.data(), (std::streamsize) (score.size() * 4))) {
        err = "profile " + path + " is truncated";
        return false;
    }
    return true;
}
}  // namespace

MimoDecoder::MimoDecoder(MimoModel& model, const DecoderConfig& cfg) : model_(model), cfg_(cfg) {}

MimoDecoder::~MimoDecoder() {
    release_gpu_waits();   // no spin kernel may outlive this object (Strata #267)
    if (stream_) cudaStreamSynchronize(stream_);
    for (auto& seq : sequences_) {
        for (auto ge : seq->graphs) if (ge) cudaGraphExecDestroy(ge);
        for (auto ge : seq->mtp_graphs) if (ge) cudaGraphExecDestroy(ge);
        if (seq->draft2_graph) cudaGraphExecDestroy(seq->draft2_graph);
        for (auto& kv : seq->kv) attn_kv_free(kv);
        attn_kv_free(seq->mtp_kv);
        attn_kv_free(seq->mtp2_kv);
    }
    if (copy_stream_) cudaStreamSynchronize(copy_stream_);
    for (auto& sw : in_flight_) cudaEventDestroy(sw.done);
    if (copy_stream_) cudaStreamDestroy(copy_stream_);
    if (mtp2_arena_) cudaFree(mtp2_arena_);
    attn_prompt_scratch_free(aps_);
    pmoe_.reset();
    gemm_.reset();
    if (mtp_arena_) cudaFree(mtp_arena_);
    for (void* p : {(void*) h_x_, (void*) h_w_, (void*) h_out_, (void*) h_ids_, (void*) h_seq_, (void*) h_flag_})
        if (p) cudaFreeHost(p);
    attn_scratch_free(attn_scratch_);
    for (void* p : allocs_) cudaFree(p);
    if (pinned_) cudaFreeHost(pinned_);
    cpu_.reset();
    pool_.reset();
    for (void* p : ram_) {
        if (ram_pinned_) cudaFreeHost(p);
        else std::free(p);
    }
    for (void* p : {(void*) h_host_, (void*) wt_host_, (void*) cpu_out_host_, (void*) ids_host_})
        if (p) cudaFreeHost(p);
    if (prev_affinity_ >= 0) cpu::restore_thread_affinity(prev_affinity_);
    if (stream_) cudaStreamDestroy(stream_);
}

bool MimoDecoder::forward(const int32_t* t, int n, int p, int32_t* next, std::vector<float>* logits, std::string& err) {
    return forward(0, t, n, p, next, logits, err);
}
bool MimoDecoder::mtp_forward(const int32_t* t, int n, int p, int32_t* draft, std::string& err) {
    return mtp_forward(0, t, n, p, draft, err);
}
bool MimoDecoder::mtp_draft2(int32_t d1, int p, int32_t* d2, std::string& err) {
    return mtp_draft2(0, d1, p, d2, err);
}

bool MimoDecoder::valid_slot(int slot, std::string& err) const {
    if (!initialized_) { err = "decoder not initialized"; return false; }
    if (failed_) { err = "decoder failed; restart required"; return false; }
    if (slot < 0 || slot >= (int) sequences_.size()) { err = "sequence slot out of range"; return false; }
    return true;
}

bool MimoDecoder::reset_sequence(int slot, std::string& err) {
    if (!valid_slot(slot, err)) return false;
    auto clear = [&](const AttnKv& kv) {
        return !kv.k || cuda_ok(cudaMemsetAsync(kv.k, 0, attn_kv_bytes(kv.n_kv, kv.cap), stream_), "reset KV", err);
    };
    if (!cuda_ok(cudaStreamSynchronize(stream_), "reset boundary", err)) { failed_ = true; return false; }
    Sequence& seq = *sequences_[(size_t) slot];
    seq.checkpoints.reset();
    for (const auto& kv : seq.kv) if (!clear(kv)) { failed_ = true; return false; }
    if (!clear(seq.mtp_kv) || !clear(seq.mtp2_kv) ||
        !cuda_ok(cudaStreamSynchronize(stream_), "reset sequence", err)) { failed_ = true; return false; }
    hidden_slot_ = mtp_hidden_slot_ = -1;
    return true;
}

MimoDecoder::PrefixCheckpointStats MimoDecoder::prefix_checkpoint_stats(int slot) const {
    if (slot < 0 || slot >= (int) sequences_.size()) return {};
    return sequences_[(size_t) slot]->checkpoints.stats();
}

std::vector<int> MimoDecoder::prefix_checkpoint_boundaries(int slot) const {
    std::vector<int> result;
    if (slot < 0 || slot >= (int) sequences_.size()) return result;
    for (const auto& e : sequences_[(size_t) slot]->checkpoints.entries)
        if (e) result.push_back(e->info.committed_end);
    std::sort(result.begin(), result.end());
    return result;
}

bool MimoDecoder::find_prefix_checkpoint(int slot, const int32_t* target, int n, PrefixCheckpointInfo& out,
                                         std::string& miss_reason, std::string& err) const {
    out = {};
    miss_reason.clear();
    if (!valid_slot(slot, err)) return false;
    if (!target || n < 1 || n > cfg_.max_ctx) { err = "checkpoint target out of range"; return false; }
    const auto& policy = sequences_[(size_t) slot]->checkpoints;
    if (const auto* e = policy.find(target, n)) { out = e->info; return true; }
    miss_reason = policy.stats().count ? "prefix_mismatch" : "no_checkpoint";
    return true;
}

bool MimoDecoder::capture_prefix_checkpoint(int slot, int committed_end, bool prompt_end, bool& captured,
                                            std::string& err) {
    captured = false;
    if (!valid_slot(slot, err)) return false;
    auto& seq = *sequences_[(size_t) slot];
    auto& policy = seq.checkpoints;
    auto skip = [&](const char* reason) {
        ++policy.counters.skipped;
        policy.counters.last_skip_reason = reason;
        return true;
    };
    if (!policy.journal_enabled) return skip("disabled");
    std::string reason;
    if (!policy.eligible(committed_end, prompt_end, reason)) return skip("not_needed");
    if (!next_checkpoint_id_) return skip("disabled");  // never recycle opaque handles on overflow
    size_t bytes = 0;
    const auto& g = model_.geometry();
    for (size_t l = 0; l < seq.kv.size(); ++l)
        if (g.is_swa((int) l)) bytes += attn_kv_bytes(seq.kv[l].n_kv, seq.kv[l].cap);
    for (const auto* kv : {&seq.mtp_kv, &seq.mtp2_kv})
        if (kv->k) bytes += attn_kv_bytes(kv->n_kv, kv->cap);
    size_t retained = 0;
    for (const auto& s : sequences_) {
        const auto usage = s->checkpoints.stats();
        retained += usage.payload_bytes + usage.metadata_bytes;
    }
    using Entry = detail::PrefixCheckpointPolicy::Entry;
    const size_t minimum_metadata = sizeof(Entry) + (size_t) committed_end * sizeof(int32_t);
    // Count the old entry AND the in-flight replacement: allocation skips never destroy a working fallback.
    if (!detail::PrefixCheckpointPolicy::fits_budget(retained, bytes, minimum_metadata)) return skip("budget");
    std::unique_ptr<Entry> entry;
    try {
        entry = std::make_unique<Entry>();
        entry->prefix.assign(policy.journal.begin(), policy.journal.end());
        if (!detail::PrefixCheckpointPolicy::fits_budget(retained, bytes, entry->metadata_bytes())) return skip("budget");
        entry->payload.reset(new unsigned char[bytes]);
    } catch (const std::bad_alloc&) { return skip("host_allocation"); }
    entry->info.id = next_checkpoint_id_++;
    entry->info.committed_end = committed_end;
    entry->info.replay_pos = committed_end - 1;
    entry->info.main_written_end = policy.main_written_end;
    entry->info.mtp_written_end = policy.mtp_written_end;
    entry->info.mtp2_written_end = policy.mtp2_written_end;
    entry->info.host_bytes = bytes;
    entry->info.metadata_bytes = entry->metadata_bytes();
    entry->milestone = detail::PrefixCheckpointPolicy::milestone(committed_end);
    entry->prompt_end = prompt_end;
    const auto start = std::chrono::steady_clock::now();
    RunGuard guard{failed_};
    size_t offset = 0;
    auto copy = [&](const AttnKv& kv) {
        if (!kv.k) return true;
        const size_t size = attn_kv_bytes(kv.n_kv, kv.cap);
        const bool result = cuda_ok(cudaMemcpyAsync(entry->payload.get() + offset, kv.k, size,
                                                    cudaMemcpyDeviceToHost, stream_), "checkpoint capture", err);
        offset += size;
        return result;
    };
    bool result = cuda_ok(cudaStreamSynchronize(stream_), "checkpoint capture boundary", err);
    for (size_t l = 0; result && l < seq.kv.size(); ++l) if (g.is_swa((int) l)) result = copy(seq.kv[l]);
    if (result) result = copy(seq.mtp_kv) && copy(seq.mtp2_kv);
    // Drain even a partially enqueued capture before freeing its pageable destination.
    const bool synced = cuda_ok(cudaStreamSynchronize(stream_), "checkpoint capture completion", err);
    policy.counters.capture_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    if (!result || !synced) { release_gpu_waits(); return false; }
    policy.publish(std::move(entry));
    ++policy.counters.captures;
    policy.counters.last_skip_reason = "none";
    captured = guard.complete = true;
    return true;
}

bool MimoDecoder::restore_prefix_checkpoint(int slot, uint64_t id, const int32_t* target, int n,
                                            PrefixCheckpointInfo& restored, std::string& err) {
    restored = {};
    if (!valid_slot(slot, err)) return false;
    if (!target || n < 1 || n > cfg_.max_ctx) { err = "checkpoint target out of range"; return false; }
    auto& seq = *sequences_[(size_t) slot];
    auto& policy = seq.checkpoints;
    const auto* entry = policy.by_id(id);
    // Revalidate at restore, not just lookup: no GPU operation on unknown/stale/wrong-slot/mismatching handles.
    if (!entry || !entry->matches(target, n)) { err = "checkpoint unavailable or prefix mismatch"; return false; }
    const auto start = std::chrono::steady_clock::now();
    RunGuard guard{failed_};
    hidden_slot_ = mtp_hidden_slot_ = -1;
    policy.clear_boundary();
    size_t offset = 0;
    auto copy = [&](const AttnKv& kv) {
        if (!kv.k) return true;
        const size_t size = attn_kv_bytes(kv.n_kv, kv.cap);
        const bool result = cuda_ok(cudaMemcpyAsync(kv.k, entry->payload.get() + offset, size,
                                                    cudaMemcpyHostToDevice, stream_), "checkpoint restore", err);
        offset += size;
        return result;
    };
    bool result = cuda_ok(cudaStreamSynchronize(stream_), "checkpoint restore boundary", err);
    const auto& g = model_.geometry();
    for (size_t l = 0; result && l < seq.kv.size(); ++l) if (g.is_swa((int) l)) result = copy(seq.kv[l]);
    if (result) result = copy(seq.mtp_kv) && copy(seq.mtp2_kv);
    const bool synced = cuda_ok(cudaStreamSynchronize(stream_), "checkpoint restore completion", err);
    policy.counters.restore_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    if (!result || !synced) { release_gpu_waits(); return false; }  // partial restore is fatal, never a cache miss
    policy.restored(*entry);
    restored = entry->info;
    guard.complete = true;
    return true;
}

bool MimoDecoder::init(std::string& err) {
    if (stream_) { err = "decoder init already attempted"; return false; }
    if (cfg_.slots != 1 && cfg_.slots != 2) { err = "decoder slots must be 1 or 2"; return false; }
    if (cfg_.max_ctx < 1 || !std::isfinite(cfg_.expert_budget_gib) || cfg_.expert_budget_gib < 0) {
        err = "invalid context or expert budget"; return false;
    }
    const MimoGeometry& g = model_.geometry();
    if (!model_.token_embd() && !model_.upload(err)) return false;
    if (!cuda_ok(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "stream", err)) return false;
    auto dev = [&](size_t bytes, auto** out) {
        void* p = nullptr;
        if (!cuda_ok(cudaMalloc(&p, std::max<size_t>(bytes, 256)), "cudaMalloc", err)) return false;
        allocs_.push_back(p);
        *out = static_cast<std::remove_pointer_t<decltype(out)>>(p);
        return true;
    };
    const size_t E = (size_t) g.n_embd, maxq = (size_t) g.qkv_rows(1) > (size_t) g.qkv_rows(0) ? g.qkv_rows(1) : g.qkv_rows(0);
    const size_t ffd = (size_t) g.n_ff_dense, V = (size_t) g.n_vocab;
    bool ok = dev(kMaxT * E * 4, &x_) && dev(kMaxT * E * 4, &h_) && dev(kMaxT * maxq * 4, &qkv_) &&
              dev(kMaxT * (size_t) g.q_rows() * 4, &q_) && dev(kMaxT * (size_t) g.attn_out_in() * 4, &a_) &&
              dev(kMaxT * E * 4, &y_) && dev(kMaxT * ffd * 4, &g_) && dev(kMaxT * ffd * 4, &u_) &&
              dev(kMaxT * ffd * 4, &m_) && dev(kMaxT * V * 4, &logits_) &&
              dev(kMaxT * (size_t) g.n_expert * 4, &router_logits_) && dev(kMaxT * kRouterMaxUsed * 4, &wt_) &&
              dev(kMaxT * kRouterMaxUsed * 4, &ids_) && dev(kMaxT * 4, &tok_) && dev(4, &pos0_) &&
              dev(kMaxT * kRouterMaxUsed * 4, &slots_) && dev(kMaxT * 4, &next_) &&
              dev(strata::kernels::native_q8_1_bytes((int) ffd, kMaxT), &q8_) &&
              dev(mxfp4_experts_scratch_bytes(kMaxT, g.n_expert_used), &mx_scratch_) && dev(kMxSlotBytes, &raw_);
    if (!ok) return false;
    // KV caches and attention constants. Reject cache oversubscription before WDDM can spill an allocation.
    size_t cache_bytes = 0, free_cache = 0, total_cache = 0;
    for (int l = 0; l < g.n_layer; ++l)
        cache_bytes += attn_kv_bytes(g.n_head_kv[(size_t) l], g.is_swa(l) ? kSwaRing : cfg_.max_ctx);
    if (!cfg_.mtp_path.empty()) cache_bytes += (cfg_.mtp_heads >= 2 ? 2 : 1) * attn_kv_bytes(g.n_head_kv[1], kSwaRing);
    cache_bytes *= (size_t) cfg_.slots;
    if (!cuda_ok(cudaMemGetInfo(&free_cache, &total_cache), "cache budget", err)) return false;
    if (cache_bytes > free_cache || free_cache - cache_bytes < 1536 * kMiB) {
        err = "not enough VRAM for all sequence caches and 1536 MiB margin"; return false;
    }
    attn_.resize((size_t) g.n_layer);
    for (int slot = 0; slot < cfg_.slots; ++slot) {
        sequences_.push_back(std::make_unique<Sequence>());
        Sequence& seq = *sequences_.back();
        seq.checkpoints.reserve_journal(cfg_.max_ctx);
        seq.kv.resize((size_t) g.n_layer);
        for (int l = 0; l < g.n_layer; ++l) {
            const bool swa = g.is_swa(l);
            if (!attn_kv_alloc(seq.kv[(size_t) l], g.n_head_kv[(size_t) l], swa ? kSwaRing : cfg_.max_ctx, swa, err))
                return false;
            attn_[(size_t) l] = attn_layer(g.n_head, g.n_head_kv[(size_t) l], swa, g.n_swa,
                swa ? g.rope_base_swa : g.rope_base, g.head_k,
                swa ? (const float*) model_.layer(l).attn_sinks.data : nullptr);
        }
    }
    if (!attn_scratch_alloc(attn_scratch_, 0, err)) return false;
    if (!cuda_ok(cudaMallocHost(&pinned_, kMxSlotBytes), "pinned staging", err)) return false;

    // the MTP layer before the experts: everything else must be in VRAM before the expert budget takes what is left
    // (on Windows an allocation past the VRAM does not fail, it lands in system memory and runs at PCIe speed)
    if (!cfg_.mtp_path.empty()) {
        if (!dev(kMaxT * E * 4, &e_) || !dev(kMaxT * E * 8, &cat_) || !dev(kMaxT * E * 4, &mres_) ||
            !dev(kMaxT * 4, &mtp_tok_) || !dev(4, &mtp_pos0_) || !dev(kMaxT * E * 4, &mres2_) ||
            !dev(E * 4, &hd_) || !dev(4, &d2_tok_) || !dev(4, &d2_pos_) || !load_mtp(err))
            return false;
    }
    if (cfg_.batched_prefill && cfg_.cpu_experts && cfg_.graph && !prefill_alloc(err)) return false;
    // v1: the CPU computes the non-resident experts, from a RAM copy of every MoE layer
    if (cfg_.cpu_experts) {
        // MIMO_HOST_CORE=last: the host loop on the last physical core and one worker fewer, away from core 0
        // (where Windows services interrupts); default: Strata's layout, the host on the first physical core
        const char* hc = std::getenv("MIMO_HOST_CORE");
        const std::vector<int> cores = cpu::physical_cores(false);
        if (hc && std::string(hc) == "last" && cores.size() > 2) {
            pool_ = std::make_unique<cpu::ExpertPool>((int) cores.size() - 2, true, true);   // workers on cores 1..n-2
            prev_affinity_ = cpu::pin_current_thread(cores.back());
        } else {
            pool_ = std::make_unique<cpu::ExpertPool>(0, true, true);   // one worker per physical core but the first
            const cpu::CpuTopology topo = cpu::detect_cpu_topology(true);
            if (topo.host_core >= 0) prev_affinity_ = cpu::pin_current_thread(topo.host_core);   // the host works too
        }
        cpu::raise_thread_priority();   // the host loop works in the pool too
        cpu_ = std::make_unique<CpuExperts>(model_, *pool_);   // the RAM copy is made at the end of init
        const size_t act = (size_t) kMaxT * E * 4, small = (size_t) kMaxT * kRouterMaxUsed * 4;
        if (!cuda_ok(cudaMallocHost(&h_host_, act), "pinned", err) || !cuda_ok(cudaMallocHost(&cpu_out_host_, act), "pinned", err) ||
            !cuda_ok(cudaMallocHost(&wt_host_, small), "pinned", err) || !cuda_ok(cudaMallocHost(&ids_host_, small), "pinned", err) ||
            !dev(act, &cpu_out_))
            return false;
    }
    // v2: the mapped handoff between the window graph and the host loop
    if (cfg_.cpu_experts && cfg_.graph) {
        auto mapped = [&](size_t bytes, auto** h, auto** m) {
            void* p = nullptr;
            if (!cuda_ok(cudaHostAlloc(&p, bytes, cudaHostAllocMapped), "mapped memory", err)) return false;
            std::memset(p, 0, bytes);
            void* d = nullptr;
            if (!cuda_ok(cudaHostGetDevicePointer(&d, p, 0), "mapped pointer", err)) return false;
            *h = static_cast<std::remove_pointer_t<decltype(h)>>(p);
            *m = static_cast<std::remove_pointer_t<decltype(m)>>(d);
            return true;
        };
        const size_t act = (size_t) kMaxT * E * 4, small = (size_t) kMaxT * kRouterMaxUsed * 4;
        if (!mapped(act, &h_x_, &m_x_) || !mapped(act, &h_out_, &m_out_) || !mapped(small, &h_w_, &m_w_) ||
            !mapped(small, &h_ids_, &m_ids_) || !mapped(64, &h_seq_, &m_seq_) || !mapped(64, &h_flag_, &m_flag_))
            return false;
    }
    // staging slots (v0): one per (token, expert) a window can route
    n_staging_ = cfg_.cpu_experts ? 0 : kMaxT * g.n_expert_used;
    // resident slots: the profile's best experts, as many as the budget holds
    size_t free_b = 0, total_b = 0;
    if (!cuda_ok(cudaMemGetInfo(&free_b, &total_b), "free VRAM", err)) return false;
    const size_t staging_bytes = (size_t) n_staging_ * kMxSlotBytes;
    const size_t late_bytes = kMxSlotBytes + sizeof(int32_t) * (size_t) g.n_layer * g.n_expert +
                              sizeof(unsigned long long) * (size_t) g.n_layer * g.n_expert +
                              (cfg_.profile_gpu ? sizeof(unsigned long long) * (size_t) g.n_layer * 5 : 0);
    const size_t reserve = staging_bytes + late_bytes + 1536 * kMiB;
    if (free_b < reserve) { err = "not enough VRAM for sequence caches, staging and 1536 MiB margin"; return false; }
    const size_t available = free_b - reserve;
    if (cfg_.expert_budget_gib > (double) available / 1073741824.0) {
        err = "explicit expert budget exceeds VRAM remaining after sequence caches and 1536 MiB margin";
        return false;
    }
    size_t budget = cfg_.expert_budget_gib > 0 ? (size_t) (cfg_.expert_budget_gib * 1073741824.0) : available;
    int want = cfg_.profile.empty() ? 0 : (int) (budget / kMxSlotBytes);
    resident_.assign((size_t) g.n_layer, std::vector<int>((size_t) g.n_expert, -1));
    std::vector<std::pair<int, int>> pick;   // (layer, expert), most used first
    if (want > 0) {
        std::vector<float> score;
        if (!read_profile(cfg_.profile, g.n_layer, g.n_expert, score, err, &profile_tokens_)) return false;
        profile_score_ = score;
        std::vector<int> order(score.size());
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return score[(size_t) a] > score[(size_t) b]; });
        for (int i : order) {
            if ((int) pick.size() >= want) break;
            const int l = i / g.n_expert, e = i % g.n_expert;
            if (g.is_moe(l)) pick.emplace_back(l, e);
        }
    }
    // spare slots for decode-time residency: taken from the budget, filled as experts move in
    const int n_spare = cfg_.adapt_decode && cfg_.cpu_experts && cfg_.graph && !pick.empty()
                            ? std::min(cfg_.adapt_spares, (int) pick.size() / 2) : 0;
    pick.resize(pick.size() - (size_t) n_spare);
    n_resident_ = (int) pick.size();
    const int n_slots = n_resident_ + n_spare + n_staging_;
    slot_addr_.resize((size_t) n_slots);
    // slots in chunks of 64 (816 MiB): a slot on its own would be rounded up to whole 2 MiB pages (14 MiB for 12.75),
    // and on Windows the overshoot does not fail - it lands in system memory and those experts run at PCIe speed
    constexpr int kChunk = 64;
    for (int i = 0; i < n_slots; i += kChunk) {
        const int n = std::min(kChunk, n_slots - i);
        void* p = nullptr;
        if (!cuda_ok(cudaMalloc(&p, (size_t) n * kMxSlotBytes), "expert slots", err)) return false;
        allocs_.push_back(p);
        for (int j = 0; j < n; ++j) slot_addr_[(size_t) (i + j)] = (unsigned long long) p + (size_t) j * kMxSlotBytes;
    }
    if (!dev(sizeof(unsigned long long) * n_slots, &slot_ptr_)) return false;
    cudaMemcpy(slot_ptr_, slot_addr_.data(), sizeof(unsigned long long) * n_slots, cudaMemcpyHostToDevice);
    for (int i = 0; i < n_resident_; ++i) {
        const auto [l, e] = pick[(size_t) i];
        const ExpertSlices& s = model_.experts(l);
        uint8_t* p = (uint8_t*) pinned_;
        std::memcpy(p, s.matrix(0, e), s.stride[0]);
        std::memcpy(p + s.stride[0], s.matrix(1, e), s.stride[1]);
        std::memcpy(p + s.stride[0] + s.stride[1], s.matrix(2, e), s.stride[2]);
        if (!cuda_ok(cudaMemcpyAsync(raw_, pinned_, kMxSlotBytes, cudaMemcpyHostToDevice, stream_), "expert upload", err)) return false;
        mxfp4_slot_relayout(raw_, (void*) slot_addr_[(size_t) (n_spare + i)], stream_);
        if (!cuda_ok(cudaStreamSynchronize(stream_), "expert relayout", err)) return false;
        resident_[(size_t) l][(size_t) e] = n_spare + i;   // the spares first: the last slots are the least used
    }
    if (pmoe_ && !prefill_place(err)) return false;   // the prompt buffers in the last (least used) slots
    // decode-time residency: the spare slots, a copy stream and its buffer, the recent counts
    for (int i = 0; i < n_spare; ++i) spares_.push_back(i);
    if (n_spare > 0 && !profile_score_.empty()) {
        if (!cuda_ok(cudaStreamCreateWithFlags(&copy_stream_, cudaStreamNonBlocking), "copy stream", err) ||
            !dev(kMxSlotBytes, &raw_copy_))
            return false;
        recent_.assign((size_t) g.n_layer * g.n_expert, 0.f);
    }
    // the residency table on the device (v2 builds each layer's slot list there)
    if (!dev(sizeof(int32_t) * (size_t) g.n_layer * g.n_expert, &d_res_)) return false;
    if (cfg_.profile_gpu && !dev(sizeof(unsigned long long) * (size_t) g.n_layer * 5, &stamps_)) return false;
    std::vector<int32_t> flat;
    for (const auto& r : resident_) flat.insert(flat.end(), r.begin(), r.end());
    if (!cuda_ok(cudaMemcpy(d_res_, flat.data(), sizeof(int32_t) * flat.size(), cudaMemcpyHostToDevice), "residency", err))
        return false;
    if (!cuda_ok(cudaMemGetInfo(&free_b, &total_b), "remaining VRAM", err)) return false;
    if (free_b < 1536 * kMiB) { err = "expert allocations consumed the 1536 MiB VRAM margin"; return false; }
    vram_free_after_init_ = free_b;
    // The CPU's copy of every MoE layer's experts - LAST, after the expert budget: on Windows, pinning ~150 GB of host
    // memory lowers the free VRAM cudaMemGetInfo reports by tens of GiB.  With the prompt path it is pinned memory
    // from cudaHostAlloc: its copies of non-resident experts run at 56-57 GB/s (44-46 with cudaHostRegister on a
    // malloc), it starts no slower (the fill takes no page faults), and the CPU reads it as fast
    // (docs/MIMO_PREFILL_MOE.md).
    if (cpu_ && cfg_.ram_copy) {
        ram_pinned_ = (cfg_.batched_prefill && pmoe_ != nullptr) || !recent_.empty();   // prompt copies, decode swaps
        for (int l = 0; l < g.n_layer; ++l) {
            if (!g.is_moe(l)) continue;
            void* p = nullptr;
            if (ram_pinned_) {
                if (!cuda_ok(cudaHostAlloc(&p, cpu_->layer_bytes(l), cudaHostAllocPortable), "pinned RAM for the experts", err))
                    return false;
            } else if (!(p = std::malloc(cpu_->layer_bytes(l)))) {
                err = "out of RAM for the experts' copy (layer " + std::to_string(l) + ")";
                return false;
            }
            ram_.push_back(p);
            cpu_->copy_layer(l, (uint8_t*) p, 16);
        }
    }
    initialized_ = true;
    return true;
}

// The first nextn layer of the MTP GGUF into VRAM (its own copies of the embedding and head are not needed: llama.cpp
// also uses the main model's when nextn.embed_tokens / shared_head_head are absent, as they are here).
bool MimoDecoder::load_mtp_layer(int k, MtpLayer& m, void*& arena, AttnLayer& al, std::string& err) {
    const MimoGeometry& g = model_.geometry();
    try {
        GgufModel mtp = GgufModel::open(cfg_.mtp_path);
        const MetaValue* arch = mtp.meta().get("general.architecture");
        const MetaValue* nn = mtp.meta().get("mimo2.nextn_predict_layers");
        if (!arch || arch->s != "mimo2" || !nn || nn->u < (uint64_t) k + 1) {
            err = cfg_.mtp_path + " is not a MiMo MTP file (mimo2 with nextn_predict_layers)";
            return false;
        }
        const int L = g.n_layer + k;   // nextn layer k: blk.48 + k
        const std::string b = "blk." + std::to_string(L) + ".";
        const uint64_t E = (uint64_t) g.n_embd, F = (uint64_t) g.n_ff_dense, Q = (uint64_t) g.qkv_rows(1);
        struct Want { const char* name; DevTensor* dst; std::vector<uint64_t> shape; uint32_t type; };
        const Want want[] = {
            {"nextn.eh_proj.weight", &m.eh_proj, {2 * E, E}, kTypeQ8_0},
            {"nextn.enorm.weight", &m.enorm, {E}, kTypeF32},
            {"nextn.hnorm.weight", &m.hnorm, {E}, kTypeF32},
            {"attn_norm.weight", &m.attn_norm, {E}, kTypeF32},
            {"attn_qkv.weight", &m.attn_qkv, {E, Q}, kTypeQ8_0},
            {"attn_output.weight", &m.attn_output, {(uint64_t) g.attn_out_in(), E}, kTypeQ8_0},
            {"attn_sinks.weight", &m.attn_sinks, {(uint64_t) g.n_head}, kTypeF32},
            {"ffn_norm.weight", &m.ffn_norm, {E}, kTypeF32},
            {"ffn_gate.weight", &m.ffn_gate, {E, F}, kTypeQ8_0},
            {"ffn_up.weight", &m.ffn_up, {E, F}, kTypeQ8_0},
            {"ffn_down.weight", &m.ffn_down, {F, E}, kTypeQ8_0},
            {"layer_output_norm.weight", &m.out_norm, {E}, kTypeF32},
        };
        size_t total = 0;
        for (const Want& w : want) {
            const TensorInfo* t = mtp.find(b + w.name);
            if (!t || t->shape != w.shape || t->type != w.type) {
                err = "MTP tensor " + b + w.name + (t ? " has an unexpected shape or type" : " is missing");
                return false;
            }
            total += (tensor_payload_bytes(*t) + 255) / 256 * 256;
        }
        if (!cuda_ok(cudaMalloc(&arena, total), "MTP weights", err)) return false;
        size_t off = 0;
        for (const Want& w : want) {
            size_t shard = 0;
            const TensorInfo* t = mtp.find(b + w.name, &shard);
            const size_t bytes = (size_t) tensor_payload_bytes(*t);
            uint8_t* dst = (uint8_t*) arena + off;
            if (!cuda_ok(cudaMemcpy(dst, mtp.shard(shard).tensor_data(*t), bytes, cudaMemcpyHostToDevice), "MTP upload", err))
                return false;
            *w.dst = DevTensor{dst, t->type, (int64_t) t->shape[0], t->shape.size() > 1 ? (int64_t) t->shape[1] : 1, bytes};
            off += (bytes + 255) / 256 * 256;
        }
    } catch (const std::exception& e) {
        err = std::string("MTP: ") + e.what();
        return false;
    }
    al = attn_layer(g.n_head, g.n_head_kv[1], true, g.n_swa, g.rope_base_swa, g.head_k,
                           (const float*) m.attn_sinks.data);
    return true;
}

bool MimoDecoder::load_mtp(std::string& err) {
    if (!load_mtp_layer(0, mtp_, mtp_arena_, mtp_attn_, err)) return false;
    if (cfg_.mtp_heads >= 2 && !load_mtp_layer(1, mtp2_, mtp2_arena_, mtp2_attn_, err)) return false;
    for (auto& seq : sequences_) {
        if (!attn_kv_alloc(seq->mtp_kv, model_.geometry().n_head_kv[1], kSwaRing, true, err)) return false;
        if (cfg_.mtp_heads >= 2 && !attn_kv_alloc(seq->mtp2_kv, model_.geometry().n_head_kv[1], kSwaRing, true, err))
            return false;
    }
    return true;
}

// llama.cpp graph_mtp: cur = eh_proj([enorm(embed(tok)) | hnorm(h_prev)]); an SWA attention block with sinks and
// value_scale; a dense SiLU FFN; then the head on layer_output_norm(cur).  h_prev = x_, the main window's final rows.
void MimoDecoder::mtp_layer_enqueue(const MtpLayer& m, const AttnKv& kv, const AttnLayer& al, const int32_t* tok,
                                    const int32_t* pos, const float* hsrc, float* mres, int T, bool head) {
    const MimoGeometry& g = model_.geometry();
    const int E = g.n_embd;
    cudaStream_t s = stream_;
    embed_q8_0(model_.token_embd().data, tok, e_, T, E, s);
    mimo_rmsnorm(e_, (const float*) m.enorm.data, e_, E, T, g.rms_eps, s);
    mimo_rmsnorm(hsrc, (const float*) m.hnorm.data, h_, E, T, g.rms_eps, s);
    concat_rows(e_, h_, cat_, T, E, s);
    q8_0_gemv(m.eh_proj, cat_, q8_, mres, T, s);
    mimo_add_rmsnorm_q8_1(mres, nullptr, nullptr, 1.f, (const float*) m.attn_norm.data, h_, q8_, false, E, T,
                          g.rms_eps, s);
    q8_0_mmvq(m.attn_qkv, q8_, qkv_, T, s);
    mimo_qkv_rope_append(qkv_, pos, T, al, kv, q_, s);
    mimo_attn_decode(q_, pos, T, al, kv, attn_scratch_, a_, s, 0, q8_);
    q8_0_mmvq(m.attn_output, q8_, y_, T, s);
    mimo_add_rmsnorm_q8_1(mres, y_, nullptr, g.value_scale, (const float*) m.ffn_norm.data, h_, q8_, false, E, T,
                          g.rms_eps, s);
    q8_0_mmvq(m.ffn_gate, q8_, g_, T, s);
    q8_0_mmvq(m.ffn_up, q8_, u_, T, s);
    silu_mul(g_, u_, m_, T * g.n_ff_dense, s);
    q8_0_gemv(m.ffn_down, m_, q8_, y_, T, s);
    if (!head) {
        add_scaled(mres, y_, 1.f, E * T, s);
        return;
    }
    mimo_add_rmsnorm_q8_1(mres, y_, nullptr, 1.f, (const float*) m.out_norm.data, h_, q8_, false, E, T, g.rms_eps, s);
    q8_0_mmvq(model_.output(), q8_, logits_, T, s);
    argmax_rows(logits_, next_, T, g.n_vocab, s);
}

// The first nextn layer over the window (its draft per row in next_); its last row kept for mtp_draft2; then the
// second layer over the same inputs, for its KV only (llama.cpp fills every head's KV from the main model's states).
void MimoDecoder::mtp_enqueue(int slot, int T) {
    const int E = model_.geometry().n_embd;
    mtp_layer_enqueue(mtp_, sequences_[(size_t) slot]->mtp_kv, mtp_attn_, mtp_tok_, mtp_pos0_, x_, mres_, T, true);
    if (!mtp2_arena_) return;
    cudaMemcpyAsync(hd_, mres_ + (size_t) (T - 1) * E, sizeof(float) * E, cudaMemcpyDeviceToDevice, stream_);
    mtp_layer_enqueue(mtp2_, sequences_[(size_t) slot]->mtp2_kv, mtp2_attn_, mtp_tok_, mtp_pos0_, x_, mres2_, T, false);
}

bool MimoDecoder::mtp_draft2(int slot, int32_t d1, int pos, int32_t* d2, std::string& err) {
    if (!valid_slot(slot, err)) return false;
    if (mtp_hidden_slot_ != slot) { err = "MTP hidden state belongs to another slot"; return false; }
    if (!mtp2_arena_) { err = "no second MTP layer loaded"; return false; }
    if (!d2 || d1 < 0 || d1 >= model_.geometry().n_vocab || pos < 0 || pos > cfg_.max_ctx + (int64_t) 1) { err = "second draft out of range"; return false; }
    sequences_[(size_t) slot]->checkpoints.before_mtp(pos + 1, true, true);
    RunGuard guard{failed_};
    const bool result = [&]() -> bool {
        cudaStream_t s = stream_;
        if (!cuda_ok(cudaMemcpyAsync(d2_tok_, &d1, sizeof(int32_t), cudaMemcpyHostToDevice, s), "device transfer", err)) return false;
        if (!cuda_ok(cudaMemcpyAsync(d2_pos_, &pos, sizeof(int32_t), cudaMemcpyHostToDevice, s), "device transfer", err)) return false;
        if (!cfg_.graph) {
            mtp_layer_enqueue(mtp2_, sequences_[(size_t) slot]->mtp2_kv, mtp2_attn_, d2_tok_, d2_pos_, hd_, mres2_, 1, true);
        } else {
            if (!sequences_[(size_t) slot]->draft2_graph) {
                cudaGraph_t graph = nullptr;
                if (!cuda_ok(cudaStreamSynchronize(s), "draft2 inputs", err) ||
                    !cuda_ok(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal), "draft2 capture", err))
                    return false;
                mtp_layer_enqueue(mtp2_, sequences_[(size_t) slot]->mtp2_kv, mtp2_attn_, d2_tok_, d2_pos_, hd_, mres2_, 1, true);
                if (!cuda_ok(cudaStreamEndCapture(s, &graph), "draft2 end capture", err)) return false;
                const cudaError_t st = cudaGraphInstantiate(&sequences_[(size_t) slot]->draft2_graph, graph, 0);
                cudaGraphDestroy(graph);
                if (!cuda_ok(st, "draft2 instantiate", err)) return false;
            }
            if (!cuda_ok(cudaGraphLaunch(sequences_[(size_t) slot]->draft2_graph, s), "draft2 launch", err)) return false;
        }
        if (!cuda_ok(cudaMemcpyAsync(d2, next_, sizeof(int32_t), cudaMemcpyDeviceToHost, s), "device transfer", err)) return false;
        return cuda_ok(cudaStreamSynchronize(s), "draft2", err);   // d1, pos and d2 are the caller's
    }();
    guard.complete = result;
    if (!result) release_gpu_waits();
    return result;
}

bool MimoDecoder::mtp_forward(int slot, const int32_t* tokens, int T, int pos0, int32_t* draft, std::string& err) {
    if (!valid_slot(slot, err)) return false;
    if (hidden_slot_ != slot) { err = "main hidden state belongs to another slot"; return false; }
    if (!mtp_arena_) { err = "no MTP layer loaded"; return false; }
    if (!tokens || !draft || T < 1 || T > kMaxT || pos0 < 0 || pos0 > cfg_.max_ctx - T + 1) { err = "MTP window out of range"; return false; }
    for (int t = 0; t < T; ++t)
        if (tokens[t] < 0 || tokens[t] >= model_.geometry().n_vocab) { err = "MTP token out of range"; return false; }
    sequences_[(size_t) slot]->checkpoints.before_mtp(pos0 + T, has_mtp2());
    RunGuard guard{failed_};
    const bool result = [&]() -> bool {
        mtp_hidden_slot_ = slot;
        cudaStream_t s = stream_;
        if (!cuda_ok(cudaMemcpyAsync(mtp_tok_, tokens, sizeof(int32_t) * T, cudaMemcpyHostToDevice, s), "device transfer", err)) return false;
        if (!cuda_ok(cudaMemcpyAsync(mtp_pos0_, &pos0, sizeof(int32_t), cudaMemcpyHostToDevice, s), "device transfer", err)) return false;
        if (cfg_.graph) {
            if (!sequences_[(size_t) slot]->mtp_graphs[T]) {
                cudaGraph_t graph = nullptr;
                if (!cuda_ok(cudaStreamSynchronize(s), "MTP inputs", err) ||
                    !cuda_ok(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal), "MTP capture", err))
                    return false;
                mtp_enqueue(slot, T);
                if (!cuda_ok(cudaStreamEndCapture(s, &graph), "MTP end capture", err)) return false;
                const cudaError_t st = cudaGraphInstantiate(&sequences_[(size_t) slot]->mtp_graphs[T], graph, 0);
                cudaGraphDestroy(graph);
                if (!cuda_ok(st, "MTP instantiate", err)) return false;
            }
            if (!cuda_ok(cudaGraphLaunch(sequences_[(size_t) slot]->mtp_graphs[T], s), "MTP launch", err)) return false;
        } else {
            mtp_enqueue(slot, T);
        }
        if (!cuda_ok(cudaMemcpyAsync(draft, next_, sizeof(int32_t) * T, cudaMemcpyDeviceToHost, s), "device transfer", err)) return false;
        return cuda_ok(cudaStreamSynchronize(s), "MTP", err);   // `tokens`, `pos0` and `draft` are the caller's
    }();
    guard.complete = result;
    if (!result) release_gpu_waits();
    return result;
}

bool MimoDecoder::prompt(const int32_t* tokens, int n, int pos0, const ChooseFn& choose, int32_t* first,
                         int32_t* draft, const std::function<void(int)>& progress, std::string& err) {
    return prompt(0, tokens, n, pos0, choose, first, draft, progress, err);
}

bool MimoDecoder::prompt(int slot, const int32_t* tokens, int n, int pos0, const ChooseFn& choose, int32_t* first,
                         int32_t* draft, const std::function<void(int)>& progress, std::string& err) {
    PromptState state;
    return prompt_chunk(slot, tokens, n, pos0, choose, first, draft, progress, state, true, err);
}

bool MimoDecoder::prompt_chunk(int slot, const int32_t* tokens, int n, int pos0, const ChooseFn& choose,
                               int32_t* first, int32_t* draft, const std::function<void(int)>& progress,
                               PromptState& state, bool final, std::string& err) {
    if (!valid_slot(slot, err)) return false;
    if (n < 1 || pos0 < 0 || n > cfg_.max_ctx || pos0 > cfg_.max_ctx - n || !choose || !first || !tokens) {
        err = "prompt chunk out of range or missing input/chooser"; return false;
    }
    if (state.finished || (state.slot >= 0 && (state.slot != slot || state.next_pos != pos0))) {
        err = "prompt state finished, wrong slot or non-contiguous chunk"; return false;
    }
    const auto& g = model_.geometry();
    for (int t = 0; t < n; ++t)
        if (tokens[t] < 0 || tokens[t] >= g.n_vocab) { err = "prompt token out of range"; return false; }
    if (state.slot < 0) {
        state.slot = slot;
        state.counts.assign((size_t) g.n_layer, std::vector<uint32_t>((size_t) g.n_expert));
    }
    sequences_[(size_t) slot]->checkpoints.clear_boundary();
    // A failed compute/borrow cannot safely be retried. Also restore the temporary routing-count pointer on throws.
    bool complete = false;
    struct Guard {
        MimoDecoder& decoder;
        bool& complete;
        ~Guard() { decoder.prompt_counts_ = nullptr; if (!complete) decoder.failed_ = true; }
    } guard{*this, complete};
    bool result = true;
    if (pmoe_ && n >= cfg_.batched_min_prompt) {
        result = prompt_batched(slot, tokens, n, pos0, choose, first, draft, progress, state, final, err);
    } else {
        prompt_counts_ = &state.counts;
        constexpr int W = kMaxT;
        const int V = g.n_vocab;
        int32_t next[kMaxT], dr[kMaxT];
        std::vector<float> logits;
        for (int done = 0; done < n;) {
            const int T = std::min(W, n - done);
            const bool last = done + T == n;
            if (!forward(slot, tokens + done, T, pos0 + done, next, last ? &logits : nullptr, err)) return false;
            if (last) {
                *first = choose(logits.data() + (size_t) (T - 1) * V, next[T - 1]);
                if (*first < 0 || *first >= V) { err = "prompt chooser returned invalid token"; return false; }
            }
            if (has_mtp()) {
                int32_t mt[kMaxT];
                for (int t = 0; t < T; ++t) mt[t] = done + t + 1 < n ? tokens[done + t + 1] : *first;
                if (!mtp_forward(slot, mt, T, pos0 + done + 1, dr, err)) return false;
                if (last && draft) *draft = dr[T - 1];
            }
            done += T;
            if (progress) progress(done);
        }
        prompt_counts_ = nullptr;
        if (final && state.tokens + (uint64_t) n >= (uint64_t) cfg_.adapt_min_prompt && !profile_score_.empty()) {
            if (!commit_swaps(true, err)) return false;
            result = adapt_residency(state.counts, err);
        }
    }
    if (!result) return false;
    state.tokens += (uint64_t) n;
    state.next_pos = pos0 + n;
    state.finished = final;
    sequences_[(size_t) slot]->checkpoints.finish_prompt(pos0 + n, final);
    hidden_slot_ = slot;
    mtp_hidden_slot_ = has_mtp2() ? slot : -1;
    complete = true;
    return true;
}

bool MimoDecoder::save_profile(const std::string& path, std::string& err) const {
    const MimoGeometry& g = model_.geometry();
    const int ne = g.n_expert;
    const double wb = profile_score_.empty() ? 0.0 : (double) std::max<uint64_t>(profile_tokens_, 1);
    const double wo = (double) usage_tokens_;
    std::vector<float> out((size_t) g.n_layer * ne, 0.f);
    for (int l = 0; l < g.n_layer; ++l) {
        double tot = 0;
        for (int e = 0; e < ne; ++e) tot += usage_.empty() ? 0.0 : usage_[(size_t) l * ne + e];
        for (int e = 0; e < ne; ++e) {
            const double b = profile_score_.empty() ? 0.0 : profile_score_[(size_t) l * ne + e];
            const double o = tot > 0 ? usage_[(size_t) l * ne + e] / tot : 0.0;
            out[(size_t) l * ne + e] = (wb + (tot > 0 ? wo : 0.0)) > 0 ? (float) ((wb * b + (tot > 0 ? wo : 0.0) * o) /
                                                                                 (wb + (tot > 0 ? wo : 0.0))) : 0.f;
        }
    }
    const std::string tmp = path + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary);
        const uint32_t hdr[3] = {1, (uint32_t) g.n_layer, (uint32_t) ne};
        const uint64_t tokens = (uint64_t) (wb + wo);
        f.write("MIMOPROF", 8);
        f.write((const char*) hdr, 12);
        f.write((const char*) &tokens, 8);
        f.write((const char*) out.data(), (std::streamsize) (out.size() * 4));
        if (!f) { err = "cannot write " + tmp; return false; }
    }
    std::remove(path.c_str());   // replaced whole: a crash mid-write leaves the old file
    if (std::rename(tmp.c_str(), path.c_str()) != 0) { err = "cannot rename " + tmp + " to " + path; return false; }
    return true;
}

void MimoDecoder::adapt_count(int l, const int32_t* ids, int n) {
    const int ne = model_.geometry().n_expert;
    if (prompt_counts_)
        for (int i = 0; i < n; ++i)
            if (ids[i] >= 0 && ids[i] < ne) ++(*prompt_counts_)[(size_t) l][(size_t) ids[i]];
    if (recent_.empty()) return;
    const float add = 1.f / recent_scale_;   // lazy decay: counts are stored divided by the running scale
    for (int i = 0; i < n; ++i)
        if (ids[i] >= 0 && ids[i] < ne) recent_[(size_t) l * ne + ids[i]] += add;
}

// After a window (no graph in flight): commit the background copies that have finished, decay the recent counts,
// and every adapt_interval windows start copies of the best non-resident experts into the spare slots.
bool MimoDecoder::adapt_window(int T, std::string& err) {
    const MimoGeometry& g = model_.geometry();
    const int ne = g.n_expert;
    // commits: the copy is in its slot; the expert it replaces gives up its slot as the next spare
    for (size_t i = 0; i < in_flight_.size();) {
        Swap& sw = in_flight_[i];
        const cudaError_t query = cudaEventQuery(sw.done);
        if (query == cudaErrorNotReady) { ++i; continue; }
        if (!cuda_ok(query, "residency copy", err)) return false;
        // the victim: the lowest-scoring resident expert now (scores may have moved since the copy started)
        const float a = cfg_.adapt_recent_weight;
        double tot_v = 0;
        int vl = -1, ve = -1;
        float vs = 1e30f;
        std::vector<double> tot((size_t) g.n_layer, 0.0);
        for (int l = 0; l < g.n_layer; ++l)
            for (int e = 0; e < ne; ++e) tot[(size_t) l] += recent_[(size_t) l * ne + e];
        for (int l = 0; l < g.n_layer; ++l) {
            if (!g.is_moe(l)) continue;
            for (int e = 0; e < ne; ++e) {
                if (resident_[(size_t) l][(size_t) e] < 0) continue;
                const float sc = (1.f - a) * profile_score_[(size_t) l * ne + e] +
                                 a * (tot[(size_t) l] > 0 ? (float) (recent_[(size_t) l * ne + e] / tot[(size_t) l]) : 0.f);
                if (sc < vs) { vs = sc; vl = l; ve = e; }
            }
        }
        (void) tot_v;
        int32_t upd[2] = {sw.slot, -1};
        resident_[(size_t) sw.layer][(size_t) sw.expert] = sw.slot;
        cudaMemcpyAsync(d_res_ + (size_t) sw.layer * ne + sw.expert, &upd[0], 4, cudaMemcpyHostToDevice, stream_);
        if (vl >= 0) {
            spares_.push_back(resident_[(size_t) vl][(size_t) ve]);
            resident_[(size_t) vl][(size_t) ve] = -1;
            cudaMemcpyAsync(d_res_ + (size_t) vl * ne + ve, &upd[1], 4, cudaMemcpyHostToDevice, stream_);
        }
        if (!cuda_ok(cudaStreamSynchronize(stream_), "residency commit", err)) return false;   // upd is on the stack
        cudaEventDestroy(sw.done);
        in_flight_.erase(in_flight_.begin() + (std::ptrdiff_t) i);
        ++stats_.swapped;
    }
    // decay: half-life adapt_half_life tokens
    if (T > 0) recent_scale_ *= std::pow(0.5f, (float) T / cfg_.adapt_half_life);
    if (recent_scale_ < 1e-12f) {   // fold the scale into the counts before it underflows
        for (float& v : recent_) v *= recent_scale_;
        recent_scale_ = 1.f;
    }
    if (T == 0 || ++windows_since_adapt_ < cfg_.adapt_interval || spares_.empty()) return true;
    windows_since_adapt_ = 0;
    // scores of every MoE expert; the best non-resident ones that beat the worst resident ones by adapt_margin
    const float a = cfg_.adapt_recent_weight;
    std::vector<std::pair<float, int>> cand, res;
    for (int l = 0; l < g.n_layer; ++l) {
        if (!g.is_moe(l)) continue;
        double tot = 0;
        for (int e = 0; e < ne; ++e) tot += recent_[(size_t) l * ne + e];
        for (int e = 0; e < ne; ++e) {
            const float sc = (1.f - a) * profile_score_[(size_t) l * ne + e] +
                             a * (tot > 0 ? (float) (recent_[(size_t) l * ne + e] / tot) : 0.f);
            bool moving = false;
            for (const Swap& sw : in_flight_) moving |= sw.layer == l && sw.expert == e;
            if (moving) continue;
            (resident_[(size_t) l][(size_t) e] >= 0 ? res : cand).emplace_back(sc, l * ne + e);
        }
    }
    const int n = std::min({(int) spares_.size(), cfg_.adapt_swaps, (int) cand.size(), (int) res.size()});
    std::partial_sort(cand.begin(), cand.begin() + n, cand.end(), [](auto& x, auto& y) { return x.first > y.first; });
    std::partial_sort(res.begin(), res.begin() + n, res.end(), [](auto& x, auto& y) { return x.first < y.first; });
    for (int i = 0; i < n; ++i) {
        if (cand[(size_t) i].first <= res[(size_t) i].first * cfg_.adapt_margin) break;
        const int l = cand[(size_t) i].second / ne, e = cand[(size_t) i].second % ne;
        const int slot = spares_.back();
        spares_.pop_back();
        const ExpertSlices& src = cpu_->source(l);   // the pinned RAM copy
        uint8_t* raw = (uint8_t*) raw_copy_;
        cudaMemcpyAsync(raw, src.matrix(0, e), src.stride[0], cudaMemcpyHostToDevice, copy_stream_);
        cudaMemcpyAsync(raw + src.stride[0], src.matrix(1, e), src.stride[1], cudaMemcpyHostToDevice, copy_stream_);
        cudaMemcpyAsync(raw + src.stride[0] + src.stride[1], src.matrix(2, e), src.stride[2], cudaMemcpyHostToDevice,
                        copy_stream_);
        mxfp4_slot_relayout(raw, (void*) slot_addr_[(size_t) slot], copy_stream_);
        Swap sw{l, e, slot, nullptr};
        if (!cuda_ok(cudaEventCreateWithFlags(&sw.done, cudaEventDisableTiming), "event", err) ||
            !cuda_ok(cudaEventRecord(sw.done, copy_stream_), "event", err))
            return false;
        in_flight_.push_back(sw);
    }
    return true;
}

void MimoDecoder::release_gpu_waits() {
    if (h_flag_) {
        *(volatile uint32_t*) h_flag_ = UINT32_MAX;   // past every ring: a waiting kernel returns
        std::atomic_thread_fence(std::memory_order_seq_cst);
        _mm_sfence();
    }
}

// One window of T tokens, all 48 layers, as one graph: per MoE layer the router's picks go to the host through
// doorbell_publish, the GPU runs its resident experts while the host loop runs the rest (forward_graph), and the
// GPU waits for the host's flag before adding the CPU's sums.  Rings are 1..n_moe in layer order.
bool MimoDecoder::capture(int slot, int T, std::string& err) {
    const MimoGeometry& g = model_.geometry();
    const int E = g.n_embd, K = g.n_expert_used;
    cudaStream_t s = stream_;
    cudaGraph_t graph = nullptr;
    if (!cuda_ok(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal), "capture", err)) return false;
    embed_q8_0(model_.token_embd().data, tok_, x_, T, E, s);
    // Each norm also writes its output as q8_1 for the GEMV after it; the end of a layer (its residual add, and for
    // a MoE layer the CPU's sums) is one kernel with the next layer's attn_norm (docs/MIMO_DECODER.md).
    mimo_add_rmsnorm_q8_1(x_, nullptr, nullptr, 1.f, (const float*) model_.layer(0).attn_norm.data, h_, q8_, false, E,
                          T, g.rms_eps, s);
    uint32_t ring = 0;
    auto stamp = [&](int l, int i) {
        if (cfg_.profile_gpu) strata::kernels::gpu_stamp(stamps_, l * 5 + i, s);
    };
    for (int l = 0; l < g.n_layer; ++l) {
        const LayerWeights& w = model_.layer(l);
        const AttnLayer& L = attn_[(size_t) l];
        stamp(l, 0);
        q8_0_mmvq(w.attn_qkv, q8_, qkv_, T, s);
        mimo_qkv_rope_append(qkv_, pos0_, T, L, sequences_[(size_t) slot]->kv[(size_t) l], q_, s);
        mimo_attn_decode(q_, pos0_, T, L, sequences_[(size_t) slot]->kv[(size_t) l], attn_scratch_, a_, s, 0, q8_);
        q8_0_mmvq(w.attn_output, q8_, y_, T, s);
        // residual + ffn_norm; for a MoE layer also the doorbell's copy of h_ and the experts' q8_1 input, and y_
        // becomes the experts' accumulator (zeroed)
        const bool moe = g.is_moe(l);
        mimo_add_rmsnorm_q8_1(x_, y_, nullptr, g.value_scale, (const float*) w.ffn_norm.data, h_, moe ? nullptr : q8_,
                              moe, E, T, g.rms_eps, s, moe ? m_x_ : nullptr, moe ? mx_scratch_ : nullptr);
        if (!moe) {
            q8_0_mmvq(w.ffn_gate, q8_, g_, T, s);
            q8_0_mmvq(w.ffn_up, q8_, u_, T, s);
            silu_mul(g_, u_, m_, T * g.n_ff_dense, s);
            q8_0_gemv(w.ffn_down, m_, q8_, y_, T, s);
        } else {
            mimo_router_fused((const float*) w.gate_inp.data, (const float*) w.exp_probs_b.data, h_, router_logits_,
                              ids_, wt_, T, E, g.n_expert, K, g.expert_weights_scale, d_res_ + (size_t) l * g.n_expert,
                              slots_, m_ids_, m_w_, m_seq_, s);
            stamp(l, 1);
            mxfp4_experts(slot_ptr_, slots_, wt_, nullptr, y_, T, K, mx_scratch_, s);   // x: in mx_scratch_
            stamp(l, 2);
            strata::kernels::wait_flag_ge(m_flag_, ++ring, s);
            stamp(l, 3);
        }
        const DevTensor& next_norm = l + 1 < g.n_layer ? model_.layer(l + 1).attn_norm : model_.output_norm();
        mimo_add_rmsnorm_q8_1(x_, y_, moe ? m_out_ : nullptr, 1.f, (const float*) next_norm.data, h_, q8_, false, E, T,
                              g.rms_eps, s);
        stamp(l, 4);
    }
    q8_0_mmvq(model_.output(), q8_, logits_, T, s);
    argmax_rows(logits_, next_, T, g.n_vocab, s);
    if (!cuda_ok(cudaStreamEndCapture(s, &graph), "end capture", err)) return false;
    const cudaError_t st = cudaGraphInstantiate(&sequences_[(size_t) slot]->graphs[T], graph, 0);
    cudaGraphDestroy(graph);
    return cuda_ok(st, "graph instantiate", err);
}

bool MimoDecoder::forward_graph(int slot, int T, int pos0, int32_t* next, std::string& err) {
    (void) pos0;   // already in pos0_ (forward wrote it)
    const MimoGeometry& g = model_.geometry();
    const int K = g.n_expert_used;
    if (!sequences_[(size_t) slot]->graphs[T] && !capture(slot, T, err)) return false;
    cudaStream_t s = stream_;
    *(volatile uint32_t*) h_seq_ = 0;
    *(volatile uint32_t*) h_flag_ = 0;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    if (!cuda_ok(cudaGraphLaunch(sequences_[(size_t) slot]->graphs[T], s), "graph launch", err)) return false;
    volatile uint32_t* const seq = h_seq_;
    volatile uint32_t* const flag = h_flag_;
    uint32_t ring = 0;
    int32_t ids[kMaxT * kRouterMaxUsed];
    for (int l = 0; l < g.n_layer; ++l) {
        if (!g.is_moe(l)) continue;
        const uint32_t want = ++ring;
        const auto a = std::chrono::steady_clock::now();
        auto last_query = a;
        uint32_t spins = 0;
        while (*seq < want) {
            _mm_pause();
            if ((++spins & 1023u) != 0) continue;
            // a driver call costs tens of microseconds, during which the doorbell goes unseen: ask whether the
            // graph died only every 2 ms (Strata's Verifier does the same)
            const auto now = std::chrono::steady_clock::now();
            if (now - last_query < std::chrono::milliseconds(2)) continue;
            last_query = now;
            const cudaError_t q = cudaStreamQuery(s);
            if (q != cudaErrorNotReady && *seq < want) {
                release_gpu_waits();
                err = "layer " + std::to_string(l) + " never rang (" +
                      std::string(q == cudaSuccess ? "graph finished" : cudaGetErrorString(q)) + ")";
                return false;
            }
            if (std::chrono::steady_clock::now() - a > std::chrono::seconds(20)) {
                release_gpu_waits();
                cudaStreamSynchronize(s);
                err = "timed out waiting for layer " + std::to_string(l);
                return false;
            }
        }
        std::atomic_thread_fence(std::memory_order_acquire);
        stats_.wait_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count();
        for (int i = 0; i < T * K; ++i) {
            const int e = ((volatile int32_t*) h_ids_)[i];
            ids[i] = e;
            ++stats_.expert_uses;
            const bool hit = resident_[(size_t) l][(size_t) e] >= 0;
            stats_.expert_hits += hit;
            ids[i] = hit ? -1 : e;
        }
        {   // the window's picks of this layer: decode-time residency and the learned profile
            int32_t picks[kMaxT * kRouterMaxUsed];
            for (int i = 0; i < T * K; ++i) picks[i] = ((volatile int32_t*) h_ids_)[i];
            adapt_count(l, picks, T * K);
            if (usage_.empty()) usage_.assign((size_t) g.n_layer * g.n_expert, 0.0);
            for (int i = 0; i < T * K; ++i)
                if (picks[i] >= 0 && picks[i] < g.n_expert) usage_[(size_t) l * g.n_expert + picks[i]] += 1.0;
        }
        const auto c0 = std::chrono::steady_clock::now();
        static const bool skip_cpu = std::getenv("MIMO_SKIP_CPU") != nullptr;   // timing diagnostic: wrong output
        static const double fake_us = std::getenv("MIMO_FAKE_CPU_US") ? std::atof(std::getenv("MIMO_FAKE_CPU_US")) : 0;
        if (skip_cpu) {
            std::memset(h_out_, 0, sizeof(float) * T * g.n_embd);
            while (std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - c0).count() < fake_us)
                _mm_pause();
        }
        else cpu_->run_layer(l, T, h_x_, K, ids, h_w_, h_out_);
        stats_.cpu_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - c0).count();
        std::atomic_thread_fence(std::memory_order_seq_cst);
        _mm_sfence();
        *flag = want;
    }
    if (!cuda_ok(cudaMemcpyAsync(next, next_, sizeof(int32_t) * T, cudaMemcpyDeviceToHost, s), "device transfer", err)) return false;
    if (!cuda_ok(cudaStreamSynchronize(s), "window", err)) return false;
    usage_tokens_ += (uint64_t) T;
    if (!prompt_counts_ && !recent_.empty() && !adapt_window(T, err)) return false;
    if (cfg_.profile_gpu) {
        std::vector<unsigned long long> st((size_t) g.n_layer * 5);
        cudaMemcpy(st.data(), stamps_, st.size() * 8, cudaMemcpyDeviceToHost);
        prof_us_.resize((size_t) g.n_layer * 4, 0.0);
        for (int l = 0; l < g.n_layer; ++l) {
            if (!g.is_moe(l)) continue;
            const unsigned long long* p = &st[(size_t) l * 5];
            for (int i = 0; i < 4; ++i) prof_us_[(size_t) l * 4 + i] += (double) (p[i + 1] - p[i]) / 1000.0;
        }
        ++prof_windows_;
    }
    return true;
}

void MimoDecoder::print_profile() const {
    if (!prof_windows_) return;
    const MimoGeometry& g = model_.geometry();
    double sum[4] = {0, 0, 0, 0};
    int n = 0;
    std::printf("GPU per MoE layer, us (avg of %d windows): attention+router | resident experts | wait for CPU | combine\n",
                prof_windows_);
    for (int l = 0; l < g.n_layer; ++l) {
        if (!g.is_moe(l)) continue;
        ++n;
        for (int i = 0; i < 4; ++i) sum[i] += prof_us_[(size_t) l * 4 + i] / prof_windows_;
        if (l == 1 || l == 5 || l == 24 || l == 47)
            std::printf("  layer %2d (%s): %7.1f %7.1f %7.1f %7.1f\n", l, g.is_swa(l) ? "SWA   " : "global",
                        prof_us_[(size_t) l * 4] / prof_windows_, prof_us_[(size_t) l * 4 + 1] / prof_windows_,
                        prof_us_[(size_t) l * 4 + 2] / prof_windows_, prof_us_[(size_t) l * 4 + 3] / prof_windows_);
    }
    std::printf("  mean          : %7.1f %7.1f %7.1f %7.1f   (x %d layers = %.2f ms)\n", sum[0] / n, sum[1] / n,
                sum[2] / n, sum[3] / n, n, (sum[0] + sum[1] + sum[2] + sum[3]) / 1000.0);
}

void MimoDecoder::dump(const char* name, int layer, const void* dev, size_t bytes_per_token, int T, bool is_int) {
    if (cfg_.dump_dir.empty()) return;
    std::vector<char> host(bytes_per_token * T);
    cudaStreamSynchronize(stream_);
    cudaMemcpy(host.data(), dev, host.size(), cudaMemcpyDeviceToHost);
    std::string file = cfg_.dump_dir + "/" + name + (layer >= 0 ? "-" + std::to_string(layer) : std::string()) +
                       (is_int ? ".i32" : ".f32");
    std::ofstream(file, std::ios::binary | std::ios::app).write(host.data(), (std::streamsize) host.size());
}

bool MimoDecoder::forward(int slot, const int32_t* tokens, int T, int pos0, int32_t* next, std::vector<float>* logits,
                          std::string& err) {
    const MimoGeometry& g = model_.geometry();
    if (!valid_slot(slot, err)) return false;
    if (!tokens || !next || T < 1 || T > kMaxT || pos0 < 0 || pos0 > cfg_.max_ctx - T) {
        err = "window out of range";
        return false;
    }
    for (int t = 0; t < T; ++t)
        if (tokens[t] < 0 || tokens[t] >= g.n_vocab) { err = "token out of range"; return false; }
    auto& checkpoint_policy = sequences_[(size_t) slot]->checkpoints;
    checkpoint_policy.before_main(pos0, T);
    RunGuard guard{failed_};
    const bool result = [&]() -> bool {
        hidden_slot_ = slot;
        mtp_hidden_slot_ = -1;
        const int E = g.n_embd, K = g.n_expert_used;
        cudaStream_t s = stream_;
        if (!cuda_ok(cudaMemcpyAsync(tok_, tokens, sizeof(int32_t) * T, cudaMemcpyHostToDevice, s), "device transfer", err)) return false;
        if (!cuda_ok(cudaMemcpyAsync(pos0_, &pos0, sizeof(int32_t), cudaMemcpyHostToDevice, s), "device transfer", err)) return false;
        if (cfg_.graph && cfg_.cpu_experts && cfg_.dump_dir.empty()) {
            if (!cuda_ok(cudaStreamSynchronize(s), "inputs", err)) return false;   // `tokens` / `pos0` live on the stack
            if (!forward_graph(slot, T, pos0, next, err)) return false;
            if (logits) {
                logits->resize((size_t) T * g.n_vocab);
                return cuda_ok(cudaMemcpy(logits->data(), logits_, sizeof(float) * logits->size(), cudaMemcpyDeviceToHost),
                               "logits", err);
            }
            return true;
        }
        if (!cfg_.dump_dir.empty()) {
            if (!cuda_ok(cudaStreamSynchronize(s), "dump boundary", err)) return false;
            std::ofstream(cfg_.dump_dir + "/tokens.i32", std::ios::binary | std::ios::app).write((const char*) tokens, 4 * T);
        }
        embed_q8_0(model_.token_embd().data, tok_, x_, T, E, s);
        // the graph's kernels (capture): norms with their q8_1, the end of a layer with the next layer's attn_norm
        mimo_add_rmsnorm_q8_1(x_, nullptr, nullptr, 1.f, (const float*) model_.layer(0).attn_norm.data, h_, q8_, false, E,
                              T, g.rms_eps, s);
        for (int l = 0; l < g.n_layer; ++l) {
            const LayerWeights& w = model_.layer(l);
            const AttnLayer& L = attn_[(size_t) l];
            dump("attn_norm", l, h_, (size_t) E * 4, T);
            q8_0_mmvq(w.attn_qkv, q8_, qkv_, T, s);
            dump("wqkv", l, qkv_, (size_t) g.qkv_rows(l) * 4, T);
            mimo_qkv_rope_append(qkv_, pos0_, T, L, sequences_[(size_t) slot]->kv[(size_t) l], q_, s);
            dump("Qcur", l, q_, (size_t) g.q_rows() * 4, T);
            mimo_attn_decode(q_, pos0_, T, L, sequences_[(size_t) slot]->kv[(size_t) l], attn_scratch_, a_, s, 0, q8_);
            q8_0_mmvq(w.attn_output, q8_, y_, T, s);
            dump("attn_out", l, y_, (size_t) E * 4, T);
            mimo_add_rmsnorm_q8_1(x_, y_, nullptr, g.value_scale, (const float*) w.ffn_norm.data, h_,
                                  g.is_moe(l) ? nullptr : q8_, g.is_moe(l), E, T, g.rms_eps, s, nullptr,
                                  g.is_moe(l) ? mx_scratch_ : nullptr);
            dump("ffn_inp", l, x_, (size_t) E * 4, T);
            dump("ffn_norm", l, h_, (size_t) E * 4, T);
            if (!g.is_moe(l)) {
                q8_0_mmvq(w.ffn_gate, q8_, g_, T, s);
                q8_0_mmvq(w.ffn_up, q8_, u_, T, s);
                silu_mul(g_, u_, m_, T * g.n_ff_dense, s);
                q8_0_gemv(w.ffn_down, m_, q8_, y_, T, s);
                dump("ffn_out", l, y_, (size_t) E * 4, T);
            } else {
                mimo_router_fused((const float*) w.gate_inp.data, (const float*) w.exp_probs_b.data, h_, router_logits_,
                                  ids_, wt_, T, E, g.n_expert, K, g.expert_weights_scale, nullptr, nullptr, nullptr,
                                  nullptr, nullptr, s);
                dump("ffn_moe_topk", l, ids_, (size_t) K * 4, T, true);
                if (cfg_.cpu_experts) {   // v1: GPU hits and CPU misses at the same time
                    if (!cuda_ok(cudaMemcpyAsync(ids_host_, ids_, sizeof(int32_t) * T * K, cudaMemcpyDeviceToHost, s), "device transfer", err)) return false;
                    if (!cuda_ok(cudaMemcpyAsync(wt_host_, wt_, sizeof(float) * T * K, cudaMemcpyDeviceToHost, s), "device transfer", err)) return false;
                    if (!cuda_ok(cudaMemcpyAsync(h_host_, h_, sizeof(float) * T * E, cudaMemcpyDeviceToHost, s), "device transfer", err)) return false;
                    if (!cuda_ok(cudaStreamSynchronize(s), "router", err)) return false;
                    adapt_count(l, ids_host_, T * K);
                    if (usage_.empty()) usage_.assign((size_t) g.n_layer * g.n_expert, 0.0);
                    for (int i = 0; i < T * K; ++i)
                        if (ids_host_[i] >= 0 && ids_host_[i] < g.n_expert) usage_[(size_t) l * g.n_expert + ids_host_[i]] += 1.0;
                    int32_t slots[kMaxT * kRouterMaxUsed];
                    for (int i = 0; i < T * K; ++i) {
                        const int expert_slot = resident_[(size_t) l][(size_t) ids_host_[i]];
                        ++stats_.expert_uses;
                        if (expert_slot >= 0) ++stats_.expert_hits;
                        slots[i] = expert_slot;
                        ids_host_[i] = expert_slot >= 0 ? -1 : ids_host_[i];   // the CPU's share: the misses
                    }
                    if (!cuda_ok(cudaMemcpyAsync(slots_, slots, sizeof(int32_t) * T * K, cudaMemcpyHostToDevice, s), "device transfer", err)) return false;
                    mxfp4_experts(slot_ptr_, slots_, wt_, nullptr, y_, T, K, mx_scratch_, s);   // x, y_: by the norm
                    const auto c0 = std::chrono::steady_clock::now();
                    cpu_->run_layer(l, T, h_host_, K, ids_host_, wt_host_, cpu_out_host_);
                    stats_.cpu_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - c0).count();
                    if (!cuda_ok(cudaMemcpyAsync(cpu_out_, cpu_out_host_, sizeof(float) * T * E, cudaMemcpyHostToDevice, s), "device transfer", err)) return false;
                    add_scaled(y_, cpu_out_, 1.f, E * T, s);
                    if (!cuda_ok(cudaStreamSynchronize(s), "experts", err)) return false;   // `slots` and the pinned rows
                    dump("ffn_moe_out", l, y_, (size_t) E * 4, T);
                    const DevTensor& next_norm = l + 1 < g.n_layer ? model_.layer(l + 1).attn_norm : model_.output_norm();
                    mimo_add_rmsnorm_q8_1(x_, y_, nullptr, 1.f, (const float*) next_norm.data, h_, q8_, false, E, T,
                                          g.rms_eps, s);
                    dump("l_out", l, x_, (size_t) E * 4, T);
                    continue;
                }
                // v0: route on the host; stage the non-resident experts into staging slots
                int32_t ids[kMaxT * kRouterMaxUsed], slots[kMaxT * kRouterMaxUsed];
                if (!cuda_ok(cudaMemcpyAsync(ids, ids_, sizeof(int32_t) * T * K, cudaMemcpyDeviceToHost, s), "device transfer", err)) return false;
                if (!cuda_ok(cudaStreamSynchronize(s), "router", err)) return false;
                adapt_count(l, ids, T * K);
                if (usage_.empty()) usage_.assign((size_t) g.n_layer * g.n_expert, 0.0);
                for (int i = 0; i < T * K; ++i)
                    if (ids[i] >= 0 && ids[i] < g.n_expert) usage_[(size_t) l * g.n_expert + ids[i]] += 1.0;
                std::map<int, int> staged;   // expert -> staging slot, this window and layer
                for (int i = 0; i < T * K; ++i) {
                    const int e = ids[i];
                    ++stats_.expert_uses;
                    int expert_slot = resident_[(size_t) l][(size_t) e];
                    if (expert_slot >= 0) {
                        ++stats_.expert_hits;
                    } else if (auto it = staged.find(e); it != staged.end()) {
                        expert_slot = it->second;
                    } else {
                        expert_slot = n_resident_ + (int) staged.size();   // v0 (no spares): staging after the resident slots
                        staged.emplace(e, expert_slot);
                        ++stats_.staged;
                        const ExpertSlices& es = model_.experts(l);
                        uint8_t* p = (uint8_t*) pinned_;
                        std::memcpy(p, es.matrix(0, e), es.stride[0]);
                        std::memcpy(p + es.stride[0], es.matrix(1, e), es.stride[1]);
                        std::memcpy(p + es.stride[0] + es.stride[1], es.matrix(2, e), es.stride[2]);
                        if (!cuda_ok(cudaMemcpyAsync(raw_, pinned_, kMxSlotBytes, cudaMemcpyHostToDevice, s), "device transfer", err)) return false;
                        mxfp4_slot_relayout(raw_, (void*) slot_addr_[(size_t) expert_slot], s);
                        if (!cuda_ok(cudaStreamSynchronize(s), "expert staging", err)) return false;
                    }
                    slots[i] = expert_slot;
                }
                if (!cuda_ok(cudaMemcpyAsync(slots_, slots, sizeof(int32_t) * T * K, cudaMemcpyHostToDevice, s), "device transfer", err)) return false;
                mxfp4_experts(slot_ptr_, slots_, wt_, nullptr, y_, T, K, mx_scratch_, s);
                if (!cuda_ok(cudaStreamSynchronize(s), "experts", err)) return false;   // `slots` is a stack array
                dump("ffn_moe_out", l, y_, (size_t) E * 4, T);
            }
            const DevTensor& next_norm = l + 1 < g.n_layer ? model_.layer(l + 1).attn_norm : model_.output_norm();
            mimo_add_rmsnorm_q8_1(x_, y_, nullptr, 1.f, (const float*) next_norm.data, h_, q8_, false, E, T, g.rms_eps, s);
            dump("l_out", l, x_, (size_t) E * 4, T);
        }
        dump("result_norm", -1, h_, (size_t) E * 4, T);
        q8_0_mmvq(model_.output(), q8_, logits_, T, s);
        dump("result_output", -1, logits_, (size_t) g.n_vocab * 4, T);
        argmax_rows(logits_, next_, T, g.n_vocab, s);
        if (!cuda_ok(cudaMemcpyAsync(next, next_, sizeof(int32_t) * T, cudaMemcpyDeviceToHost, s), "device transfer", err)) return false;
        if (logits) {
            logits->resize((size_t) T * g.n_vocab);
            if (!cuda_ok(cudaMemcpyAsync(logits->data(), logits_, sizeof(float) * logits->size(), cudaMemcpyDeviceToHost, s), "device transfer", err)) return false;
        }
        if (!cuda_ok(cudaStreamSynchronize(s), "forward", err)) return false;
        usage_tokens_ += (uint64_t) T;
        return true;
    }();
    if (result) checkpoint_policy.after_main(tokens, pos0, T);
    guard.complete = result;
    if (!result) release_gpu_waits();
    return result;
}

}  // namespace strata::mimo
