// MimoDecoder's batched prompt path (docs/MIMO_PREFILL_MOE.md, docs/MIMO_ATTN.md): the prompt in slices of up to
// `prefill_slice` tokens, each slice LAYER-MAJOR - layer l over the whole slice, then layer l + 1 - so a non-resident
// expert crosses PCIe once per layer and slice rather than once per chunk.  Within a layer the attention part runs
// in tiles of `prefill_tile` tokens (FP16 GEMMs on the dequantized Q8_0 weights, dequantized once per layer and
// slice; the prompt attention kernels), then the routed experts of the whole slice in one PrefillMoe call.  After
// the 48 layers: the head for the slice's last position only, then the MTP layer over the same positions.
#include "strata/mimo/decoder.hpp"

#include "strata/kernels/dequant_bf16.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/mimo/cpu_experts.hpp"
#include "strata/mimo/ops.hpp"
#include "strata/mimo/prefill_moe.hpp"
#include "strata/mimo/expert_mxfp4.hpp"
#include "strata/mimo/router.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>

namespace strata::mimo {
namespace {
bool ok(cudaError_t st, const char* what, std::string& err) {
    if (st == cudaSuccess) return true;
    err = std::string(what) + ": " + cudaGetErrorString(st);
    return false;
}
}  // namespace

// The prompt path's device memory, allocated in init() BEFORE the expert budget is taken.
// The prompt path's objects and the SIZES of its buffers (init, before the expert budget).  The buffers themselves
// are placed in borrowed expert slots by prefill_place() once the slots exist.
bool MimoDecoder::prefill_alloc(std::string& err) {
    const MimoGeometry& g = model_.geometry();
    const size_t S = (size_t) cfg_.prefill_slice, P = (size_t) cfg_.prefill_tile, E = (size_t) g.n_embd;
    const size_t Q = (size_t) std::max(g.qkv_rows(0), g.qkv_rows(1)), F = (size_t) g.n_ff_dense;
    const size_t wmax = std::max({Q * E, F * E, E * (size_t) g.attn_out_in(), 2 * E * E});   // dequantized weights
    pf_bufs_ = {{(void**) &pX_, S * E * 4}, {(void**) &pHn_, S * E * 4}, {(void**) &pY_, S * E * 4},
                {(void**) &pIds_, S * 8 * 4}, {(void**) &pW_, S * 8 * 4}, {(void**) &pTok_, S * 4},
                {(void**) &pLogits_, P * (size_t) g.n_expert * 4}, {(void**) &p16_, P * F * 2},
                {(void**) &pQkv_, P * Q * 4}, {(void**) &pQ16_, P * (size_t) g.q_rows() * 2},
                {(void**) &pA_, P * (size_t) g.attn_out_in() * 4}, {(void**) &pG_, P * F * 4},
                {(void**) &pU_, P * F * 4}, {(void**) &pW1_, wmax * 2}, {(void**) &pW2_, wmax * 2},
                {(void**) &pE_, P * E * 4}};
    if (!attn_prompt_scratch_alloc(aps_, (int) P, g.n_head_kv[1], g.n_swa, 0, err)) return false;
    gemm_ = std::make_unique<prefill::Gemm>();
    if (!gemm_->init(stream_, 1 << 20, err)) return false;
    PrefillMoeConfig mc;
    mc.max_tokens = (int) S;
    mc.k = g.n_expert_used;
    pmoe_ = std::make_unique<PrefillMoe>(mc);
    pf_bufs_.push_back({nullptr, pmoe_->layout_bytes()});   // the experts' buffers (ptr null: init_external)
    return true;
}

// Places the prompt buffers in the last slot chunks (each buffer inside one chunk: slots are 64 to an allocation),
// taking as few chunks from the end as fit them; the slots from the first of those chunks on are "borrowed".
bool MimoDecoder::prefill_place(std::string& err) {
    constexpr int kChunk = 64;
    const int n_slots = (int) slot_addr_.size();
    std::vector<PfBuf> bufs = pf_bufs_;
    std::sort(bufs.begin(), bufs.end(), [](const PfBuf& x, const PfBuf& y) { return x.bytes > y.bytes; });
    const int n_chunks = (n_slots + kChunk - 1) / kChunk;
    for (int take = 1; take <= n_chunks; ++take) {
        // chunks [n_chunks - take, n_chunks): free bytes per chunk, first fit, largest buffer first
        std::vector<size_t> used((size_t) take, 0), cap((size_t) take);
        for (int c = 0; c < take; ++c) {
            const int c0 = (n_chunks - take + c) * kChunk;
            cap[(size_t) c] = (size_t) (std::min(c0 + kChunk, n_slots) - c0) * kMxSlotBytes;
        }
        std::vector<std::pair<int, size_t>> where;   // (chunk, offset) per buffer
        bool fits = true;
        for (const PfBuf& b : bufs) {
            const size_t bytes = (b.bytes + 255) / 256 * 256;
            int c = 0;
            while (c < take && used[(size_t) c] + bytes > cap[(size_t) c]) ++c;
            if (c == take) { fits = false; break; }
            where.emplace_back(c, used[(size_t) c]);
            used[(size_t) c] += bytes;
        }
        if (!fits) continue;
        borrow_lo_ = (n_chunks - take) * kChunk;
        for (size_t i = 0; i < bufs.size(); ++i) {
            const int c0 = (n_chunks - take + where[i].first) * kChunk;
            uint8_t* p = (uint8_t*) slot_addr_[(size_t) c0] + where[i].second;
            if (bufs[i].ptr) *bufs[i].ptr = p;
            else if (!pmoe_->init_external(p, err)) return false;
        }
        size_t total = 0;
        for (const PfBuf& b : pf_bufs_) total += b.bytes;
        std::fprintf(stderr, "prompt path: slice %d, tile %d; its %.2f GiB of buffers borrow the last %d expert slots\n",
                     cfg_.prefill_slice, cfg_.prefill_tile, total / 1073741824.0, n_slots - borrow_lo_);
        return true;
    }
    err = "the prompt path's buffers do not fit in the expert slots (too little VRAM for experts)";
    return false;
}

// Finished background copies become resident (wait: all of them, after the copy stream drains).
bool MimoDecoder::commit_swaps(bool wait, std::string& err) {
    if (!copy_stream_ || in_flight_.empty()) return true;
    if (wait && !ok(cudaStreamSynchronize(copy_stream_), "copy boundary", err)) return false;
    windows_since_adapt_ = 0;
    return adapt_window(0, err);   // T = 0: commits only, no decay; the interval reset above keeps it from starting copies
}

// Before a batched prompt: the experts in the borrowed slots stop being resident (their slots hold the prompt's
// buffers until refill_slots), and no background copy may still target them.
bool MimoDecoder::borrow_slots(std::string& err) {
    const MimoGeometry& g = model_.geometry();
    if (!commit_swaps(true, err)) return false;
    for (int l = 0; l < g.n_layer; ++l)
        for (int e = 0; e < g.n_expert; ++e)
            if (resident_[(size_t) l][(size_t) e] >= borrow_lo_) resident_[(size_t) l][(size_t) e] = -1;
    const size_t before = spares_.size();
    spares_.erase(std::remove_if(spares_.begin(), spares_.end(), [&](int s) { return s >= borrow_lo_; }), spares_.end());
    lost_spares_ = (int) (before - spares_.size());
    std::vector<int32_t> flat;
    for (const auto& r : resident_) flat.insert(flat.end(), r.begin(), r.end());
    pmoe_->refresh(stream_);
    return ok(cudaMemcpyAsync(d_res_, flat.data(), sizeof(int32_t) * flat.size(), cudaMemcpyHostToDevice, stream_),
              "residency", err) && ok(cudaStreamSynchronize(stream_), "borrow", err);
}

// After a batched prompt: the borrowed slots get the best non-resident experts again (scores: the profile, blended
// with the recent picks and, when given, the prompt's own), except the spares the range held, which stay empty.
bool MimoDecoder::refill_slots(const std::vector<std::vector<uint32_t>>* counts, std::string& err) {
    const MimoGeometry& g = model_.geometry();
    const int ne = g.n_expert, n_slots = (int) slot_addr_.size();
    std::vector<int> free_slots;
    for (int s = borrow_lo_; s < n_slots; ++s) free_slots.push_back(s);
    for (int i = 0; i < lost_spares_ && !free_slots.empty(); ++i) {
        spares_.push_back(free_slots.back());
        free_slots.pop_back();
    }
    lost_spares_ = 0;
    std::vector<std::pair<float, int>> cand;
    for (int l = 0; l < g.n_layer; ++l) {
        if (!g.is_moe(l)) continue;
        double tr = 0, tp = 0;
        for (int e = 0; e < ne; ++e) {
            if (!recent_.empty()) tr += recent_[(size_t) l * ne + e];
            if (counts) tp += (*counts)[(size_t) l][(size_t) e];
        }
        for (int e = 0; e < ne; ++e) {
            if (resident_[(size_t) l][(size_t) e] >= 0) continue;
            float sc = profile_score_.empty() ? 0.f : profile_score_[(size_t) l * ne + e];
            if (tr > 0) sc = 0.5f * sc + 0.5f * (float) (recent_[(size_t) l * ne + e] / tr);
            if (counts && tp > 0) sc = (1.f - cfg_.adapt_weight) * sc + cfg_.adapt_weight * (float) ((*counts)[(size_t) l][(size_t) e] / tp);
            cand.emplace_back(sc, l * ne + e);
        }
    }
    const size_t n = std::min(free_slots.size(), cand.size());
    std::partial_sort(cand.begin(), cand.begin() + (std::ptrdiff_t) n, cand.end(),
                      [](const auto& x, const auto& y) { return x.first > y.first; });
    cudaStream_t s = stream_;
    for (size_t i = 0; i < n; ++i) {
        const int l = cand[i].second / ne, e = cand[i].second % ne, slot = free_slots[i];
        const ExpertSlices& src = cpu_ ? cpu_->source(l) : model_.experts(l);
        uint8_t* raw = (uint8_t*) raw_;
        if (!ok(cudaMemcpyAsync(raw, src.matrix(0, e), src.stride[0], cudaMemcpyHostToDevice, s), "expert transfer", err)) return false;
        if (!ok(cudaMemcpyAsync(raw + src.stride[0], src.matrix(1, e), src.stride[1], cudaMemcpyHostToDevice, s), "expert transfer", err)) return false;
        if (!ok(cudaMemcpyAsync(raw + src.stride[0] + src.stride[1], src.matrix(2, e), src.stride[2], cudaMemcpyHostToDevice, s), "expert transfer", err)) return false;
        mxfp4_slot_relayout(raw, (void*) slot_addr_[(size_t) slot], s);
        resident_[(size_t) l][(size_t) e] = slot;
    }
    std::vector<int32_t> flat;
    for (const auto& r : resident_) flat.insert(flat.end(), r.begin(), r.end());
    return ok(cudaMemcpyAsync(d_res_, flat.data(), sizeof(int32_t) * flat.size(), cudaMemcpyHostToDevice, s),
              "residency", err) && ok(cudaStreamSynchronize(s), "refill", err);
}

void MimoDecoder::gemm_q8(const DevTensor& w, uint16_t* scratch, bool dequant, const uint16_t* x16, float* y, int T) {
    // y[T][ne1] = x16[T][ne0] . W^T, W dequantized to FP16 in `scratch` unless it is already there
    if (dequant) strata::kernels::dequant_f16((int) w.type, w.data, 0, w.ne1, w.ne0, scratch, stream_);
    gemm_->f16(x16, scratch, y, T, w.ne1, w.ne0);
}

// The attention half of a layer (or the MTP layer) for rows [r0, r0 + T) of the slice at positions pos0 + r0 ...:
// x += value_scale * attn_output(attention(rope(qkv(rmsnorm(x) * attn_norm))))
void MimoDecoder::prefill_attention(const DevTensor& attn_norm, const DevTensor& qkv, const DevTensor& out,
                                    const AttnLayer& L, const AttnKv& kv, float* x, int T, int pos, bool dq_qkv,
                                    bool dq_out) {
    const MimoGeometry& g = model_.geometry();
    const int E = g.n_embd;
    cudaStream_t s = stream_;
    mimo_rmsnorm(x, (const float*) attn_norm.data, pE_, E, T, g.rms_eps, s);
    prefill::to_f16(pE_, p16_, (int64_t) T * E, s);
    gemm_q8(qkv, pW1_, dq_qkv, p16_, pQkv_, T);
    mimo_qkv_rope_append_prompt(pQkv_, pos, T, L, kv, aps_, pQ16_, s);
    mimo_attn_prompt(pQ16_, pos, T, L, kv, aps_, pA_, s);
    prefill::to_f16(pA_, p16_, (int64_t) T * g.attn_out_in(), s);
    gemm_q8(out, pW2_, dq_out, p16_, pE_, T);
    mimo_residual_add(x, pE_, g.value_scale, E * T, s);
}

// A dense SiLU FFN on rows of `hn` (already normed), added to x: layer 0 and the MTP layer.
void MimoDecoder::prefill_dense_ffn(const DevTensor& gate, const DevTensor& up, const DevTensor& down, const float* hn,
                                    float* x, int T) {
    const MimoGeometry& g = model_.geometry();
    const int E = g.n_embd, F = g.n_ff_dense;
    cudaStream_t s = stream_;
    prefill::to_f16(hn, p16_, (int64_t) T * E, s);
    gemm_q8(gate, pW1_, true, p16_, pG_, T);
    gemm_q8(up, pW1_, true, p16_, pU_, T);
    silu_mul(pG_, pU_, pG_, T * F, s);
    prefill::to_f16(pG_, p16_, (int64_t) T * F, s);
    gemm_q8(down, pW1_, true, p16_, pE_, T);
    add_scaled(x, pE_, 1.f, E * T, s);
}

void MimoDecoder::prefill_dump(const char* name, int layer, const float* dev, size_t floats_per_row, int rows) {
    if (cfg_.dump_dir.empty()) return;
    std::vector<float> host(floats_per_row * rows);
    cudaStreamSynchronize(stream_);
    cudaMemcpy(host.data(), dev, host.size() * 4, cudaMemcpyDeviceToHost);
    const std::string f = cfg_.dump_dir + "/" + name + (layer >= 0 ? "-" + std::to_string(layer) : std::string()) + ".f32";
    std::ofstream(f, std::ios::binary | std::ios::app).write((const char*) host.data(), (std::streamsize) host.size() * 4);
}

bool MimoDecoder::prompt_batched(int slot, const int32_t* tokens, int n, int pos0, const ChooseFn& choose, int32_t* first,
                                 int32_t* draft, const std::function<void(int)>& progress, PromptState& state, bool final, std::string& err) {
    const MimoGeometry& g = model_.geometry();
    const int E = g.n_embd, K = g.n_expert_used, V = g.n_vocab;
    cudaStream_t s = stream_;
    if (!cfg_.dump_dir.empty())
        std::ofstream(cfg_.dump_dir + "/tokens.i32", std::ios::binary | std::ios::app).write((const char*) tokens, 4 * (size_t) n);
    if (borrow_lo_ >= 0 && !borrow_slots(err)) return false;
    std::vector<int32_t> res(g.n_expert);
    // the prompt's routing, for adapt_residency: picks per layer and expert
    const bool adapt = final && state.tokens + (uint64_t) n >= (uint64_t) cfg_.adapt_min_prompt && !profile_score_.empty();
    std::vector<std::vector<uint32_t>> counts((size_t) g.n_layer, std::vector<uint32_t>((size_t) g.n_expert));
    std::vector<int32_t> picks;
    std::vector<float> last_logits(V);
    for (int done = 0; done < n;) {
        const int S = std::min(cfg_.prefill_slice, n - done), base = pos0 + done;
        const bool last_slice = done + S == n;
        if (!ok(cudaMemcpyAsync(pTok_, tokens + done, sizeof(int32_t) * S, cudaMemcpyHostToDevice, s), "tokens", err))
            return false;
        embed_q8_0(model_.token_embd().data, pTok_, pX_, S, E, s);
        for (int l = 0; l < g.n_layer; ++l) {
            const LayerWeights& w = model_.layer(l);
            for (int r = 0; r < S; r += cfg_.prefill_tile) {
                const int T = std::min(cfg_.prefill_tile, S - r);
                float* x = pX_ + (size_t) r * E;
                prefill_attention(w.attn_norm, w.attn_qkv, w.attn_output, attn_[(size_t) l], sequences_[(size_t) slot]->kv[(size_t) l], x, T,
                                  base + r, r == 0, r == 0);
                float* hn = pHn_ + (size_t) r * E;
                mimo_rmsnorm(x, (const float*) w.ffn_norm.data, hn, E, T, g.rms_eps, s);
                if (!g.is_moe(l)) {
                    prefill_dense_ffn(w.ffn_gate, w.ffn_up, w.ffn_down, hn, x, T);
                    // the dense FFN reused pW1_: the next tile's qkv must be dequantized again
                    if (r + cfg_.prefill_tile < S)
                        strata::kernels::dequant_f16((int) w.attn_qkv.type, w.attn_qkv.data, 0, w.attn_qkv.ne1,
                                                     w.attn_qkv.ne0, pW1_, s);
                } else {
                    router_logits_batched(gemm_->cublas(), (const float*) w.gate_inp.data, hn, pLogits_, T, E, g.n_expert);
                    router_topk(pLogits_, (const float*) w.exp_probs_b.data, pIds_ + (size_t) r * K, pW_ + (size_t) r * K,
                                T, g.n_expert, K, g.expert_weights_scale, s);
                }
            }
            if (g.is_moe(l)) {   // the slice's experts in one call: each non-resident expert copied once
                for (int e = 0; e < g.n_expert; ++e) res[(size_t) e] = resident_[(size_t) l][(size_t) e];
                PrefillMoeLayer ml;
                ml.n_tok = S;
                ml.x = pHn_;
                ml.ids = pIds_;
                ml.w = pW_;
                ml.out = pY_;
                ml.resident = res.data();
                ml.slot_addr = slot_addr_.data();
                ml.ram = cpu_ ? cpu_->source(l) : model_.experts(l);
                pmoe_->run_layer(ml, s);
                {   // the slice's picks: prompt-driven residency and the learned profile
                    picks.resize((size_t) S * K);
                    if (!ok(cudaMemcpyAsync(picks.data(), pIds_, sizeof(int32_t) * picks.size(), cudaMemcpyDeviceToHost, s),
                            "prompt routing", err) || !ok(cudaStreamSynchronize(s), "prompt routing", err)) return false;
                    for (int32_t e : picks)
                        if (e >= 0 && e < g.n_expert) {
                            ++counts[(size_t) l][(size_t) e];
                            ++stats_.expert_uses;
                            stats_.expert_hits += res[(size_t) e] >= 0;
                        }
                }
                add_scaled(pX_, pY_, 1.f, E * S, s);
            }
            prefill_dump("l_out", l, pX_, (size_t) E, S);
            if (progress && l + 1 == g.n_layer) progress(done + S);
        }
        // the head for the slice's last position (the next token) - only on the final slice
        const float* xl = pX_ + (size_t) (S - 1) * E;
        if (last_slice) {
            mimo_rmsnorm(xl, (const float*) model_.output_norm().data, h_, E, 1, g.rms_eps, s);
            strata::kernels::native_q8_0_f32(model_.output().data, h_, q8_, logits_, E, V, 1, s);
            argmax_rows(logits_, next_, 1, V, s);
            int32_t greedy = 0;
            if (!ok(cudaMemcpyAsync(&greedy, next_, 4, cudaMemcpyDeviceToHost, s), "prompt transfer", err)) return false;
            if (!ok(cudaMemcpyAsync(last_logits.data(), logits_, sizeof(float) * V, cudaMemcpyDeviceToHost, s), "logits", err) ||
                !ok(cudaStreamSynchronize(s), "prompt slice", err))
                return false;
            *first = choose(last_logits.data(), greedy);
            if (*first < 0 || *first >= V) { err = "prompt chooser returned invalid token"; return false; }
        }
        // decode reads the window's final hidden state from x_: leave the slice's last position there
        if (!ok(cudaMemcpyAsync(x_, xl, sizeof(float) * E, cudaMemcpyDeviceToDevice, s), "prompt transfer", err)) return false;
        if (has_mtp()) {   // the MTP layers over positions base + 1 .. base + S: token p with the state at p - 1
            std::vector<int32_t> mt((size_t) S);
            for (int j = 0; j < S; ++j) mt[(size_t) j] = done + j + 1 < n ? tokens[done + j + 1] : *first;
            if (!ok(cudaMemcpyAsync(pTok_, mt.data(), sizeof(int32_t) * S, cudaMemcpyHostToDevice, s), "prompt transfer", err)) return false;
            auto run_layer = [&](const MtpLayer& m, const AttnKv& kv, const AttnLayer& al) -> bool {
                embed_q8_0(model_.token_embd().data, pTok_, pY_, S, E, s);   // pY_: the MTP layer's token rows
                bool first_tile = true;
                for (int r = 0; r < S; r += cfg_.prefill_tile) {
                    const int T = std::min(cfg_.prefill_tile, S - r);
                    float* er = pY_ + (size_t) r * E;
                    mimo_rmsnorm(er, (const float*) m.enorm.data, er, E, T, g.rms_eps, s);
                    mimo_rmsnorm(pX_ + (size_t) r * E, (const float*) m.hnorm.data, pHn_ + (size_t) r * E, E, T, g.rms_eps, s);
                    concat_rows(er, pHn_ + (size_t) r * E, pG_, T, E, s);   // [T][8192] in pG_ (n_ff >= 2 n_embd)
                    prefill::to_f16(pG_, p16_, (int64_t) T * 2 * E, s);
                    float* mr = pHn_ + (size_t) r * E;                        // the layer's residual stream, in place
                    gemm_q8(m.eh_proj, pW1_, true, p16_, mr, T);
                    prefill_attention(m.attn_norm, m.attn_qkv, m.attn_output, al, kv, mr, T, base + 1 + r, true, first_tile);
                    mimo_rmsnorm(mr, (const float*) m.ffn_norm.data, pE_, E, T, g.rms_eps, s);
                    if (!ok(cudaMemcpyAsync(pA_, pE_, sizeof(float) * T * E, cudaMemcpyDeviceToDevice, s), "prompt transfer", err)) return false;
                    prefill_dense_ffn(m.ffn_gate, m.ffn_up, m.ffn_down, pA_, mr, T);
                    first_tile = false;
                }
                return ok(cudaGetLastError(), "MTP prompt kernels", err);
            };
            if (!run_layer(mtp_, sequences_[(size_t) slot]->mtp_kv, mtp_attn_)) return false;
            const float* ml = pHn_ + (size_t) (S - 1) * E;
            if (last_slice && draft) {
                mimo_rmsnorm(ml, (const float*) mtp_.out_norm.data, h_, E, 1, g.rms_eps, s);
                strata::kernels::native_q8_0_f32(model_.output().data, h_, q8_, logits_, E, V, 1, s);
                argmax_rows(logits_, next_, 1, V, s);
                if (!ok(cudaMemcpyAsync(draft, next_, 4, cudaMemcpyDeviceToHost, s), "prompt transfer", err)) return false;
            }
            if (has_mtp2()) {   // the second layer: its KV over the same positions; the first one's last row for draft 2
                if (!ok(cudaMemcpyAsync(hd_, ml, sizeof(float) * E, cudaMemcpyDeviceToDevice, s), "prompt transfer", err)) return false;
                if (!run_layer(mtp2_, sequences_[(size_t) slot]->mtp2_kv, mtp2_attn_)) return false;
            }
        }
        if (!ok(cudaGetLastError(), "prompt kernels", err) ||
            !ok(cudaStreamSynchronize(s), "prompt slice", err)) return false;
        done += S;
    }
    if (usage_.empty()) usage_.assign((size_t) g.n_layer * g.n_expert, 0.0);
    for (int l = 0; l < g.n_layer; ++l)
        for (int e = 0; e < g.n_expert; ++e) {
            usage_[(size_t) l * g.n_expert + e] += counts[(size_t) l][(size_t) e];
            state.counts[(size_t) l][(size_t) e] += counts[(size_t) l][(size_t) e];
        }
    usage_tokens_ += (uint64_t) n;
    if (borrow_lo_ >= 0 && !refill_slots(adapt ? &state.counts : nullptr, err)) return false;
    return adapt ? adapt_residency(state.counts, err) : true;
}

bool MimoDecoder::adapt_residency(const std::vector<std::vector<uint32_t>>& counts, std::string& err) {
    const MimoGeometry& g = model_.geometry();
    const auto t0 = std::chrono::steady_clock::now();
    const float a = cfg_.adapt_weight;
    // the blended score of every MoE expert: shares within its layer, so layers weigh the same
    std::vector<std::pair<float, int>> rank;   // (score, layer * n_expert + e)
    for (int l = 0; l < g.n_layer; ++l) {
        if (!g.is_moe(l)) continue;
        double tot = 0;
        for (uint32_t c : counts[(size_t) l]) tot += c;
        for (int e = 0; e < g.n_expert; ++e) {
            const float prompt = tot > 0 ? (float) (counts[(size_t) l][(size_t) e] / tot) : 0.f;
            const float prof = profile_score_[(size_t) l * g.n_expert + e];
            rank.emplace_back((1.f - a) * prof + a * prompt, l * g.n_expert + e);
        }
    }
    std::stable_sort(rank.begin(), rank.end(), [](const auto& x, const auto& y) { return x.first > y.first; });
    std::vector<char> keep((size_t) g.n_layer * g.n_expert, 0);
    for (int i = 0; i < n_resident_ && i < (int) rank.size(); ++i) keep[(size_t) rank[(size_t) i].second] = 1;
    std::vector<int> in, out;   // experts to load, resident experts to drop (their slots are reused)
    for (int i = 0; i < n_resident_ && i < (int) rank.size(); ++i) {
        const int id = rank[(size_t) i].second;
        if (resident_[(size_t) (id / g.n_expert)][(size_t) (id % g.n_expert)] < 0) in.push_back(id);
    }
    for (int l = 0; l < g.n_layer; ++l)
        for (int e = 0; e < g.n_expert; ++e)
            if (resident_[(size_t) l][(size_t) e] >= 0 && !keep[(size_t) l * g.n_expert + e]) out.push_back(l * g.n_expert + e);
    const size_t n = std::min(in.size(), out.size());
    cudaStream_t s = stream_;
    for (size_t i = 0; i < n; ++i) {
        const int li = in[i] / g.n_expert, ei = in[i] % g.n_expert, lo = out[i] / g.n_expert, eo = out[i] % g.n_expert;
        const int slot = resident_[(size_t) lo][(size_t) eo];
        const ExpertSlices& src = cpu_ ? cpu_->source(li) : model_.experts(li);   // the pinned RAM copy
        uint8_t* raw = (uint8_t*) raw_;
        if (!ok(cudaMemcpyAsync(raw, src.matrix(0, ei), src.stride[0], cudaMemcpyHostToDevice, s), "expert transfer", err)) return false;
        if (!ok(cudaMemcpyAsync(raw + src.stride[0], src.matrix(1, ei), src.stride[1], cudaMemcpyHostToDevice, s), "expert transfer", err)) return false;
        if (!ok(cudaMemcpyAsync(raw + src.stride[0] + src.stride[1], src.matrix(2, ei), src.stride[2], cudaMemcpyHostToDevice, s), "expert transfer", err)) return false;
        mxfp4_slot_relayout(raw, (void*) slot_addr_[(size_t) slot], s);
        resident_[(size_t) lo][(size_t) eo] = -1;
        resident_[(size_t) li][(size_t) ei] = slot;
    }
    std::vector<int32_t> flat;
    for (const auto& r : resident_) flat.insert(flat.end(), r.begin(), r.end());
    if (!ok(cudaMemcpyAsync(d_res_, flat.data(), sizeof(int32_t) * flat.size(), cudaMemcpyHostToDevice, s), "residency", err) ||
        !ok(cudaStreamSynchronize(s), "residency swap", err))
        return false;
    stats_.swapped += n;
    stats_.swap_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return true;
}

}  // namespace strata::mimo
