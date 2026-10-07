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
bool MimoDecoder::prefill_alloc(std::string& err) {
    const MimoGeometry& g = model_.geometry();
    const size_t S = (size_t) cfg_.prefill_slice, P = (size_t) cfg_.prefill_tile, E = (size_t) g.n_embd;
    const size_t Q = (size_t) std::max(g.qkv_rows(0), g.qkv_rows(1)), F = (size_t) g.n_ff_dense;
    const size_t wmax = std::max({Q * E, F * E, E * (size_t) g.attn_out_in(), 2 * E * E});   // dequantized weights
    auto dev = [&](size_t bytes, auto** out) {
        void* p = nullptr;
        if (!ok(cudaMalloc(&p, std::max<size_t>(bytes, 256)), "prompt buffers", err)) return false;
        allocs_.push_back(p);
        *out = static_cast<std::remove_pointer_t<decltype(out)>>(p);
        return true;
    };
    if (!dev(S * E * 4, &pX_) || !dev(S * E * 4, &pHn_) || !dev(S * E * 4, &pY_) || !dev(S * 8 * 4, &pIds_) ||
        !dev(S * 8 * 4, &pW_) || !dev(S * 4, &pTok_) || !dev(P * (size_t) g.n_expert * 4, &pLogits_) ||
        !dev(P * F * 2, &p16_) || !dev(P * Q * 4, &pQkv_) || !dev(P * (size_t) g.q_rows() * 2, &pQ16_) ||
        !dev(P * (size_t) g.attn_out_in() * 4, &pA_) || !dev(P * F * 4, &pG_) || !dev(P * F * 4, &pU_) ||
        !dev(wmax * 2, &pW1_) || !dev(wmax * 2, &pW2_) || !dev(P * E * 4, &pE_))
        return false;
    if (!attn_prompt_scratch_alloc(aps_, (int) P, g.n_head_kv[1], g.n_swa, 0, err)) return false;
    gemm_ = std::make_unique<prefill::Gemm>();
    if (!gemm_->init(stream_, 1 << 20, err)) return false;
    PrefillMoeConfig mc;
    mc.max_tokens = (int) S;
    mc.k = g.n_expert_used;
    pmoe_ = std::make_unique<PrefillMoe>(mc);
    if (!pmoe_->init(err)) return false;
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    std::fprintf(stderr, "prompt path: slice %d, tile %d, experts' buffers %.2f GiB; %.2f GiB of VRAM free after it\n",
                 cfg_.prefill_slice, cfg_.prefill_tile, pmoe_->device_bytes() / 1073741824.0, free_b / 1073741824.0);
    return true;
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

bool MimoDecoder::prompt_batched(const int32_t* tokens, int n, int pos0, const ChooseFn& choose, int32_t* first,
                                 int32_t* draft, const std::function<void(int)>& progress, std::string& err) {
    const MimoGeometry& g = model_.geometry();
    const int E = g.n_embd, K = g.n_expert_used, V = g.n_vocab;
    cudaStream_t s = stream_;
    if (!cfg_.dump_dir.empty())
        std::ofstream(cfg_.dump_dir + "/tokens.i32", std::ios::binary | std::ios::app).write((const char*) tokens, 4 * (size_t) n);
    std::vector<int32_t> res(g.n_expert);
    // the prompt's routing, for adapt_residency: picks per layer and expert
    const bool adapt = n >= cfg_.adapt_min_prompt && !profile_score_.empty();
    std::vector<std::vector<uint32_t>> counts(adapt ? (size_t) g.n_layer : 0, std::vector<uint32_t>((size_t) g.n_expert));
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
                prefill_attention(w.attn_norm, w.attn_qkv, w.attn_output, attn_[(size_t) l], kv_[(size_t) l], x, T,
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
                if (adapt) {
                    picks.resize((size_t) S * K);
                    cudaMemcpyAsync(picks.data(), pIds_, sizeof(int32_t) * picks.size(), cudaMemcpyDeviceToHost, s);
                    cudaStreamSynchronize(s);
                    for (int32_t e : picks)
                        if (e >= 0 && e < g.n_expert) ++counts[(size_t) l][(size_t) e];
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
            cudaMemcpyAsync(&greedy, next_, 4, cudaMemcpyDeviceToHost, s);
            if (!ok(cudaMemcpyAsync(last_logits.data(), logits_, sizeof(float) * V, cudaMemcpyDeviceToHost, s), "logits", err) ||
                !ok(cudaStreamSynchronize(s), "prompt slice", err))
                return false;
            *first = choose(last_logits.data(), greedy);
        }
        // decode reads the window's final hidden state from x_: leave the slice's last position there
        cudaMemcpyAsync(x_, xl, sizeof(float) * E, cudaMemcpyDeviceToDevice, s);
        if (has_mtp()) {   // the MTP layers over positions base + 1 .. base + S: token p with the state at p - 1
            std::vector<int32_t> mt((size_t) S);
            for (int j = 0; j < S; ++j) mt[(size_t) j] = done + j + 1 < n ? tokens[done + j + 1] : *first;
            cudaMemcpyAsync(pTok_, mt.data(), sizeof(int32_t) * S, cudaMemcpyHostToDevice, s);
            auto run_layer = [&](const MtpLayer& m, const AttnKv& kv, const AttnLayer& al) {
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
                    cudaMemcpyAsync(pA_, pE_, sizeof(float) * T * E, cudaMemcpyDeviceToDevice, s);
                    prefill_dense_ffn(m.ffn_gate, m.ffn_up, m.ffn_down, pA_, mr, T);
                    first_tile = false;
                }
            };
            run_layer(mtp_, mtp_kv_, mtp_attn_);
            const float* ml = pHn_ + (size_t) (S - 1) * E;
            if (last_slice && draft) {
                mimo_rmsnorm(ml, (const float*) mtp_.out_norm.data, h_, E, 1, g.rms_eps, s);
                strata::kernels::native_q8_0_f32(model_.output().data, h_, q8_, logits_, E, V, 1, s);
                argmax_rows(logits_, next_, 1, V, s);
                cudaMemcpyAsync(draft, next_, 4, cudaMemcpyDeviceToHost, s);
            }
            if (has_mtp2()) {   // the second layer: its KV over the same positions; the first one's last row for draft 2
                cudaMemcpyAsync(hd_, ml, sizeof(float) * E, cudaMemcpyDeviceToDevice, s);
                run_layer(mtp2_, mtp2_kv_, mtp2_attn_);
            }
        }
        if (!ok(cudaStreamSynchronize(s), "prompt slice", err)) return false;
        done += S;
    }
    return adapt ? adapt_residency(counts, err) : true;
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
        cudaMemcpyAsync(raw, src.matrix(0, ei), src.stride[0], cudaMemcpyHostToDevice, s);
        cudaMemcpyAsync(raw + src.stride[0], src.matrix(1, ei), src.stride[1], cudaMemcpyHostToDevice, s);
        cudaMemcpyAsync(raw + src.stride[0] + src.stride[1], src.matrix(2, ei), src.stride[2], cudaMemcpyHostToDevice, s);
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
