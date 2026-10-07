#include "strata/mimo/decoder.hpp"

#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/mimo/cpu_experts.hpp"
#include "strata/mimo/expert_mxfp4.hpp"
#include "strata/mimo/ops.hpp"
#include "strata/mimo/router.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <climits>
#include <immintrin.h>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <numeric>

namespace strata::mimo {
namespace cpu = strata::kernels::cpu;
namespace {
constexpr int kMaxT = 8;
constexpr size_t kMiB = 1u << 20;

bool cuda_ok(cudaError_t st, const char* what, std::string& err) {
    if (st == cudaSuccess) return true;
    err = std::string(what) + ": " + cudaGetErrorString(st);
    return false;
}

void q8_0_gemv(const DevTensor& w, const float* x, void* q8, float* y, int T, cudaStream_t s) {
    strata::kernels::native_q8_0_f32(w.data, x, q8, y, (int) w.ne0, (int) w.ne1, T, s);
}

// MIMOPROF v1 (tools/mimo/make_profile.py): score[layer][expert]
bool read_profile(const std::string& path, int n_layer, int n_expert, std::vector<float>& score, std::string& err) {
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
    for (auto& ge : graphs_)
        if (ge) cudaGraphExecDestroy(ge);
    for (void* p : {(void*) h_x_, (void*) h_w_, (void*) h_out_, (void*) h_ids_, (void*) h_seq_, (void*) h_flag_})
        if (p) cudaFreeHost(p);
    for (auto& kv : kv_) attn_kv_free(kv);
    attn_scratch_free(attn_scratch_);
    for (void* p : allocs_) cudaFree(p);
    if (pinned_) cudaFreeHost(pinned_);
    cpu_.reset();
    pool_.reset();
    for (void* p : ram_) std::free(p);
    for (void* p : {(void*) h_host_, (void*) wt_host_, (void*) cpu_out_host_, (void*) ids_host_})
        if (p) cudaFreeHost(p);
    if (prev_affinity_ >= 0) cpu::restore_thread_affinity(prev_affinity_);
    if (stream_) cudaStreamDestroy(stream_);
}

bool MimoDecoder::init(std::string& err) {
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
    // KV caches and attention constants
    kv_.resize((size_t) g.n_layer);
    attn_.resize((size_t) g.n_layer);
    for (int l = 0; l < g.n_layer; ++l) {
        const bool swa = g.is_swa(l);
        if (!attn_kv_alloc(kv_[(size_t) l], g.n_head_kv[(size_t) l], swa ? kSwaRing : cfg_.max_ctx, swa, err)) return false;
        attn_[(size_t) l] = attn_layer(g.n_head, g.n_head_kv[(size_t) l], swa, g.n_swa, swa ? g.rope_base_swa : g.rope_base,
                                       g.head_k, swa ? (const float*) model_.layer(l).attn_sinks.data : nullptr);
    }
    if (!attn_scratch_alloc(attn_scratch_, 0, err)) return false;
    if (!cuda_ok(cudaMallocHost(&pinned_, kMxSlotBytes), "pinned staging", err)) return false;

    // v1: the CPU computes the non-resident experts, from a RAM copy of every MoE layer
    if (cfg_.cpu_experts) {
        pool_ = std::make_unique<cpu::ExpertPool>(0, true, true);   // one worker per physical core but the first
        const cpu::CpuTopology topo = cpu::detect_cpu_topology(true);
        if (topo.host_core >= 0) prev_affinity_ = cpu::pin_current_thread(topo.host_core);   // the host works in the pool
        cpu_ = std::make_unique<CpuExperts>(model_, *pool_);
        if (cfg_.ram_copy)
            for (int l = 0; l < g.n_layer; ++l) {
                if (!g.is_moe(l)) continue;
                void* p = std::malloc(cpu_->layer_bytes(l));
                if (!p) { err = "out of RAM for the experts' copy (layer " + std::to_string(l) + ")"; return false; }
                ram_.push_back(p);
                cpu_->copy_layer(l, (uint8_t*) p, 16);
            }
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
    cudaMemGetInfo(&free_b, &total_b);
    const size_t staging_bytes = (size_t) n_staging_ * kMxSlotBytes;
    size_t budget = cfg_.expert_budget_gib > 0 ? (size_t) (cfg_.expert_budget_gib * 1073741824.0)
                                               : (free_b > staging_bytes + 1536 * kMiB ? free_b - staging_bytes - 1536 * kMiB : 0);
    int want = cfg_.profile.empty() ? 0 : (int) (budget / kMxSlotBytes);
    resident_.assign((size_t) g.n_layer, std::vector<int>((size_t) g.n_expert, -1));
    std::vector<std::pair<int, int>> pick;   // (layer, expert), most used first
    if (want > 0) {
        std::vector<float> score;
        if (!read_profile(cfg_.profile, g.n_layer, g.n_expert, score, err)) return false;
        std::vector<int> order(score.size());
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return score[(size_t) a] > score[(size_t) b]; });
        for (int i : order) {
            if ((int) pick.size() >= want) break;
            const int l = i / g.n_expert, e = i % g.n_expert;
            if (g.is_moe(l)) pick.emplace_back(l, e);
        }
    }
    n_resident_ = (int) pick.size();
    const int n_slots = n_resident_ + n_staging_;
    slot_addr_.resize((size_t) n_slots);
    for (int i = 0; i < n_slots; ++i) {   // one allocation per slot keeps a failed big cudaMalloc from costing it all
        void* p = nullptr;
        if (!cuda_ok(cudaMalloc(&p, kMxSlotBytes), "expert slot", err)) return false;
        allocs_.push_back(p);
        slot_addr_[(size_t) i] = (unsigned long long) p;
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
        mxfp4_slot_relayout(raw_, (void*) slot_addr_[(size_t) i], stream_);
        if (!cuda_ok(cudaStreamSynchronize(stream_), "expert relayout", err)) return false;
        resident_[(size_t) l][(size_t) e] = i;
    }
    // the residency table on the device (v2 builds each layer's slot list there)
    if (!dev(sizeof(int32_t) * (size_t) g.n_layer * g.n_expert, &d_res_)) return false;
    std::vector<int32_t> flat;
    for (const auto& r : resident_) flat.insert(flat.end(), r.begin(), r.end());
    return cuda_ok(cudaMemcpy(d_res_, flat.data(), sizeof(int32_t) * flat.size(), cudaMemcpyHostToDevice), "residency", err);
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
bool MimoDecoder::capture(int T, std::string& err) {
    const MimoGeometry& g = model_.geometry();
    const int E = g.n_embd, K = g.n_expert_used;
    cudaStream_t s = stream_;
    cudaGraph_t graph = nullptr;
    if (!cuda_ok(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal), "capture", err)) return false;
    embed_q8_0(model_.token_embd().data, tok_, x_, T, E, s);
    uint32_t ring = 0;
    for (int l = 0; l < g.n_layer; ++l) {
        const LayerWeights& w = model_.layer(l);
        const AttnLayer& L = attn_[(size_t) l];
        mimo_rmsnorm(x_, (const float*) w.attn_norm.data, h_, E, T, g.rms_eps, s);
        q8_0_gemv(w.attn_qkv, h_, q8_, qkv_, T, s);
        mimo_qkv_rope_append(qkv_, pos0_, T, L, kv_[(size_t) l], q_, s);
        mimo_attn_decode(q_, pos0_, T, L, kv_[(size_t) l], attn_scratch_, a_, s);
        q8_0_gemv(w.attn_output, a_, q8_, y_, T, s);
        mimo_residual_add(x_, y_, g.value_scale, E * T, s);
        mimo_rmsnorm(x_, (const float*) w.ffn_norm.data, h_, E, T, g.rms_eps, s);
        if (!g.is_moe(l)) {
            q8_0_gemv(w.ffn_gate, h_, q8_, g_, T, s);
            q8_0_gemv(w.ffn_up, h_, q8_, u_, T, s);
            silu_mul(g_, u_, m_, T * g.n_ff_dense, s);
            q8_0_gemv(w.ffn_down, m_, q8_, y_, T, s);
        } else {
            router_logits((const float*) w.gate_inp.data, h_, router_logits_, T, E, g.n_expert, s);
            router_topk(router_logits_, (const float*) w.exp_probs_b.data, ids_, wt_, T, g.n_expert, K,
                        g.expert_weights_scale, s);
            resident_slots(ids_, d_res_ + (size_t) l * g.n_expert, slots_, T * K, s);
            strata::kernels::doorbell_publish(h_, ids_, wt_, (int64_t) T * E, (int64_t) T * K, m_x_, m_ids_, m_w_,
                                              m_seq_, s);
            fill_zero(y_, T * E, s);
            mxfp4_experts(slot_ptr_, slots_, wt_, h_, y_, T, K, mx_scratch_, s);
            strata::kernels::wait_flag_ge(m_flag_, ++ring, s);
            strata::kernels::copy_from_mapped(cpu_out_, m_out_, (int64_t) T * E, s);
            add_scaled(y_, cpu_out_, 1.f, E * T, s);
        }
        add_scaled(x_, y_, 1.f, E * T, s);
    }
    mimo_rmsnorm(x_, (const float*) model_.output_norm().data, h_, E, T, g.rms_eps, s);
    q8_0_gemv(model_.output(), h_, q8_, logits_, T, s);
    argmax_rows(logits_, next_, T, g.n_vocab, s);
    if (!cuda_ok(cudaStreamEndCapture(s, &graph), "end capture", err)) return false;
    const cudaError_t st = cudaGraphInstantiate(&graphs_[T], graph, 0);
    cudaGraphDestroy(graph);
    return cuda_ok(st, "graph instantiate", err);
}

bool MimoDecoder::forward_graph(int T, int pos0, int32_t* next, std::string& err) {
    (void) pos0;   // already in pos0_ (forward wrote it)
    const MimoGeometry& g = model_.geometry();
    const int K = g.n_expert_used;
    if (!graphs_[T] && !capture(T, err)) return false;
    cudaStream_t s = stream_;
    *(volatile uint32_t*) h_seq_ = 0;
    *(volatile uint32_t*) h_flag_ = 0;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    if (!cuda_ok(cudaGraphLaunch(graphs_[T], s), "graph launch", err)) return false;
    volatile uint32_t* const seq = h_seq_;
    volatile uint32_t* const flag = h_flag_;
    uint32_t ring = 0;
    int32_t ids[kMaxT * kRouterMaxUsed];
    for (int l = 0; l < g.n_layer; ++l) {
        if (!g.is_moe(l)) continue;
        const uint32_t want = ++ring;
        const auto a = std::chrono::steady_clock::now();
        uint32_t spins = 0;
        while (*seq < want) {
            _mm_pause();
            if ((++spins & 4095u) != 0) continue;
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
        for (int i = 0; i < T * K; ++i) {
            const int e = ((volatile int32_t*) h_ids_)[i];
            ++stats_.expert_uses;
            const bool hit = resident_[(size_t) l][(size_t) e] >= 0;
            stats_.expert_hits += hit;
            ids[i] = hit ? -1 : e;
        }
        const auto c0 = std::chrono::steady_clock::now();
        cpu_->run_layer(l, T, h_x_, K, ids, h_w_, h_out_);
        stats_.cpu_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - c0).count();
        std::atomic_thread_fence(std::memory_order_seq_cst);
        _mm_sfence();
        *flag = want;
    }
    cudaMemcpyAsync(next, next_, sizeof(int32_t) * T, cudaMemcpyDeviceToHost, s);
    return cuda_ok(cudaStreamSynchronize(s), "window", err);
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

bool MimoDecoder::forward(const int32_t* tokens, int T, int pos0, int32_t* next, std::vector<float>* logits,
                          std::string& err) {
    const MimoGeometry& g = model_.geometry();
    if (T < 1 || T > kMaxT || pos0 < 0 || pos0 + T > cfg_.max_ctx) {
        err = "window out of range";
        return false;
    }
    const int E = g.n_embd, K = g.n_expert_used;
    cudaStream_t s = stream_;
    cudaMemcpyAsync(tok_, tokens, sizeof(int32_t) * T, cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(pos0_, &pos0, sizeof(int32_t), cudaMemcpyHostToDevice, s);
    if (cfg_.graph && cfg_.cpu_experts && cfg_.dump_dir.empty() && !logits) {
        if (!cuda_ok(cudaStreamSynchronize(s), "inputs", err)) return false;   // `tokens` / `pos0` live on the stack
        return forward_graph(T, pos0, next, err);
    }
    if (!cfg_.dump_dir.empty()) {
        cudaStreamSynchronize(s);
        std::ofstream(cfg_.dump_dir + "/tokens.i32", std::ios::binary | std::ios::app).write((const char*) tokens, 4 * T);
    }
    embed_q8_0(model_.token_embd().data, tok_, x_, T, E, s);
    for (int l = 0; l < g.n_layer; ++l) {
        const LayerWeights& w = model_.layer(l);
        const AttnLayer& L = attn_[(size_t) l];
        mimo_rmsnorm(x_, (const float*) w.attn_norm.data, h_, E, T, g.rms_eps, s);
        dump("attn_norm", l, h_, (size_t) E * 4, T);
        q8_0_gemv(w.attn_qkv, h_, q8_, qkv_, T, s);
        dump("wqkv", l, qkv_, (size_t) g.qkv_rows(l) * 4, T);
        mimo_qkv_rope_append(qkv_, pos0_, T, L, kv_[(size_t) l], q_, s);
        dump("Qcur", l, q_, (size_t) g.q_rows() * 4, T);
        mimo_attn_decode(q_, pos0_, T, L, kv_[(size_t) l], attn_scratch_, a_, s);
        q8_0_gemv(w.attn_output, a_, q8_, y_, T, s);
        dump("attn_out", l, y_, (size_t) E * 4, T);
        mimo_residual_add(x_, y_, g.value_scale, E * T, s);
        dump("ffn_inp", l, x_, (size_t) E * 4, T);
        mimo_rmsnorm(x_, (const float*) w.ffn_norm.data, h_, E, T, g.rms_eps, s);
        dump("ffn_norm", l, h_, (size_t) E * 4, T);
        if (!g.is_moe(l)) {
            q8_0_gemv(w.ffn_gate, h_, q8_, g_, T, s);
            q8_0_gemv(w.ffn_up, h_, q8_, u_, T, s);
            silu_mul(g_, u_, m_, T * g.n_ff_dense, s);
            q8_0_gemv(w.ffn_down, m_, q8_, y_, T, s);
            dump("ffn_out", l, y_, (size_t) E * 4, T);
        } else {
            router_logits((const float*) w.gate_inp.data, h_, router_logits_, T, E, g.n_expert, s);
            router_topk(router_logits_, (const float*) w.exp_probs_b.data, ids_, wt_, T, g.n_expert, K,
                        g.expert_weights_scale, s);
            dump("ffn_moe_topk", l, ids_, (size_t) K * 4, T, true);
            if (cfg_.cpu_experts) {   // v1: GPU hits and CPU misses at the same time
                cudaMemcpyAsync(ids_host_, ids_, sizeof(int32_t) * T * K, cudaMemcpyDeviceToHost, s);
                cudaMemcpyAsync(wt_host_, wt_, sizeof(float) * T * K, cudaMemcpyDeviceToHost, s);
                cudaMemcpyAsync(h_host_, h_, sizeof(float) * T * E, cudaMemcpyDeviceToHost, s);
                if (!cuda_ok(cudaStreamSynchronize(s), "router", err)) return false;
                int32_t slots[kMaxT * kRouterMaxUsed];
                for (int i = 0; i < T * K; ++i) {
                    const int slot = resident_[(size_t) l][(size_t) ids_host_[i]];
                    ++stats_.expert_uses;
                    if (slot >= 0) ++stats_.expert_hits;
                    slots[i] = slot;
                    ids_host_[i] = slot >= 0 ? -1 : ids_host_[i];   // the CPU's share: the misses
                }
                cudaMemcpyAsync(slots_, slots, sizeof(int32_t) * T * K, cudaMemcpyHostToDevice, s);
                cudaMemsetAsync(y_, 0, sizeof(float) * T * E, s);
                mxfp4_experts(slot_ptr_, slots_, wt_, h_, y_, T, K, mx_scratch_, s);
                const auto c0 = std::chrono::steady_clock::now();
                cpu_->run_layer(l, T, h_host_, K, ids_host_, wt_host_, cpu_out_host_);
                stats_.cpu_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - c0).count();
                cudaMemcpyAsync(cpu_out_, cpu_out_host_, sizeof(float) * T * E, cudaMemcpyHostToDevice, s);
                add_scaled(y_, cpu_out_, 1.f, E * T, s);
                if (!cuda_ok(cudaStreamSynchronize(s), "experts", err)) return false;   // `slots` and the pinned rows
                dump("ffn_moe_out", l, y_, (size_t) E * 4, T);
                add_scaled(x_, y_, 1.f, E * T, s);
                dump("l_out", l, x_, (size_t) E * 4, T);
                continue;
            }
            // v0: route on the host; stage the non-resident experts into staging slots
            int32_t ids[kMaxT * kRouterMaxUsed], slots[kMaxT * kRouterMaxUsed];
            cudaMemcpyAsync(ids, ids_, sizeof(int32_t) * T * K, cudaMemcpyDeviceToHost, s);
            if (!cuda_ok(cudaStreamSynchronize(s), "router", err)) return false;
            std::map<int, int> staged;   // expert -> staging slot, this window and layer
            for (int i = 0; i < T * K; ++i) {
                const int e = ids[i];
                ++stats_.expert_uses;
                int slot = resident_[(size_t) l][(size_t) e];
                if (slot >= 0) {
                    ++stats_.expert_hits;
                } else if (auto it = staged.find(e); it != staged.end()) {
                    slot = it->second;
                } else {
                    slot = n_resident_ + (int) staged.size();
                    staged.emplace(e, slot);
                    ++stats_.staged;
                    const ExpertSlices& es = model_.experts(l);
                    uint8_t* p = (uint8_t*) pinned_;
                    std::memcpy(p, es.matrix(0, e), es.stride[0]);
                    std::memcpy(p + es.stride[0], es.matrix(1, e), es.stride[1]);
                    std::memcpy(p + es.stride[0] + es.stride[1], es.matrix(2, e), es.stride[2]);
                    cudaMemcpyAsync(raw_, pinned_, kMxSlotBytes, cudaMemcpyHostToDevice, s);
                    mxfp4_slot_relayout(raw_, (void*) slot_addr_[(size_t) slot], s);
                    if (!cuda_ok(cudaStreamSynchronize(s), "expert staging", err)) return false;
                }
                slots[i] = slot;
            }
            cudaMemcpyAsync(slots_, slots, sizeof(int32_t) * T * K, cudaMemcpyHostToDevice, s);
            cudaMemsetAsync(y_, 0, sizeof(float) * T * E, s);
            mxfp4_experts(slot_ptr_, slots_, wt_, h_, y_, T, K, mx_scratch_, s);
            if (!cuda_ok(cudaStreamSynchronize(s), "experts", err)) return false;   // `slots` is a stack array
            dump("ffn_moe_out", l, y_, (size_t) E * 4, T);
        }
        add_scaled(x_, y_, 1.f, E * T, s);
        dump("l_out", l, x_, (size_t) E * 4, T);
    }
    mimo_rmsnorm(x_, (const float*) model_.output_norm().data, h_, E, T, g.rms_eps, s);
    dump("result_norm", -1, h_, (size_t) E * 4, T);
    q8_0_gemv(model_.output(), h_, q8_, logits_, T, s);
    dump("result_output", -1, logits_, (size_t) g.n_vocab * 4, T);
    argmax_rows(logits_, next_, T, g.n_vocab, s);
    cudaMemcpyAsync(next, next_, sizeof(int32_t) * T, cudaMemcpyDeviceToHost, s);
    if (logits) {
        logits->resize((size_t) T * g.n_vocab);
        cudaMemcpyAsync(logits->data(), logits_, sizeof(float) * logits->size(), cudaMemcpyDeviceToHost, s);
    }
    return cuda_ok(cudaStreamSynchronize(s), "forward", err);
}

}  // namespace strata::mimo
