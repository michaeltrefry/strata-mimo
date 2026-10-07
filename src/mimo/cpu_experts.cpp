// src/mimo/cpu_experts.cpp - MiMo step 2: the CPU's share of a MoE layer.  See the header.
#include "strata/mimo/cpu_experts.hpp"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>

namespace strata::mimo {

namespace cpu = strata::kernels::cpu;

CpuExperts::CpuExperts(const MimoModel& model, cpu::ExpertPool& pool) : model_(model), pool_(pool) {
    const MimoGeometry& g = model.geometry();
    n_embd_ = g.n_embd;
    n_expert_ = g.n_expert;
    fmt_.resize((size_t) g.n_layer);
    src_.resize((size_t) g.n_layer);
    for (int l = 0; l < g.n_layer; ++l) {
        if (!g.is_moe(l)) continue;
        const std::string b = "blk." + std::to_string(l) + ".";
        const TensorInfo* gate = nullptr;
        const TensorInfo* down = nullptr;
        model.host_data(b + "ffn_gate_exps.weight", &gate);
        model.host_data(b + "ffn_down_exps.weight", &down);
        std::string err;
        if (!cpu::native_fmt((int) gate->type, (int) down->type, g.n_embd, g.n_ff_exp, fmt_[(size_t) l], err))
            throw std::runtime_error("MiMo CPU experts, layer " + std::to_string(l) + ": " + err);
        src_[(size_t) l] = model.experts(l);
        const ExpertSlices& s = src_[(size_t) l];
        const cpu::NativeFmt& f = fmt_[(size_t) l];
        if (s.row_bytes[0] != f.gu_row || s.row_bytes[1] != f.gu_row || s.row_bytes[2] != f.d_row)
            throw std::runtime_error("MiMo CPU experts, layer " + std::to_string(l) + ": row sizes do not match the GGUF");
        act_bytes_ = f.act_bytes;
    }
    if (g.n_expert > 32767) throw std::runtime_error("MiMo CPU experts: too many experts");
    act_.resize((size_t) cpu::MAXT * act_bytes_);
    slot_.assign((size_t) n_expert_, -1);
    jobs_.resize((size_t) cpu::MAXT * (size_t) (std::max)(1, g.n_expert_used));
}

size_t CpuExperts::layer_bytes(int layer) const {
    const ExpertSlices& s = model_.experts(layer);
    return (size_t) n_expert_ * s.expert_bytes();
}

void CpuExperts::copy_layer(int layer, uint8_t* dst, int threads) {
    const ExpertSlices& from = model_.experts(layer);
    ExpertSlices to = from;
    uint8_t* p = dst;
    struct Piece { const uint8_t* src; uint8_t* dst; size_t bytes; };
    std::vector<Piece> pieces;
    for (int m = 0; m < 3; ++m) {
        const size_t bytes = (size_t) n_expert_ * from.stride[m];
        to.base[m] = p;
        constexpr size_t kChunk = 64u << 20;
        for (size_t o = 0; o < bytes; o += kChunk) pieces.push_back({from.base[m] + o, p + o, (std::min)(kChunk, bytes - o)});
        p += bytes;
    }
    std::atomic<size_t> next{0};
    std::vector<std::thread> th;
    for (int t = 0; t < (std::max)(1, threads); ++t)
        th.emplace_back([&] {
            for (size_t i; (i = next.fetch_add(1)) < pieces.size();) std::memcpy(pieces[i].dst, pieces[i].src, pieces[i].bytes);
        });
    for (auto& t : th) t.join();
    src_[(size_t) layer] = to;
}

void CpuExperts::quantize(int layer, int n_tok, const float* x, uint8_t* act) const {
    for (int t = 0; t < n_tok; ++t)
        cpu::native_quant_act(fmt_[(size_t) layer], x + (size_t) t * n_embd_, act + (size_t) t * act_bytes_);
}

void CpuExperts::run_layer(int layer, int n_tok, const float* x, int k, const int32_t* ids, const float* w, float* out) {
    if (n_tok < 1 || n_tok > cpu::MAXT) throw std::runtime_error("MiMo CPU experts: 1..8 tokens per call");
    quantize(layer, n_tok, x, act_.data());
    run_layer_q(layer, n_tok, act_.data(), act_bytes_, k, ids, w, out);
}

void CpuExperts::run_layer_q(int layer, int n_tok, const uint8_t* act, size_t act_stride, int k, const int32_t* ids,
                             const float* w, float* out) {
    if (n_tok < 1 || n_tok > cpu::MAXT) throw std::runtime_error("MiMo CPU experts: 1..8 tokens per call");
    const ExpertSlices& s = src_[(size_t) layer];
    if (!s.base[0]) throw std::runtime_error("MiMo CPU experts: layer " + std::to_string(layer) + " has no experts");
    if ((size_t) n_tok * (size_t) k > jobs_.size()) jobs_.resize((size_t) n_tok * (size_t) k);
    // group by expert: a token pair that shares an expert reads its rows once
    int n = 0;
    int32_t used[cpu::MAXT * 64];
    if (k > 64) throw std::runtime_error("MiMo CPU experts: at most 64 experts per token");
    for (int t = 0; t < n_tok; ++t)
        for (int j = 0; j < k; ++j) {
            const int32_t e = ids[(size_t) t * k + j];
            if (e < 0) continue;
            if (e >= n_expert_) throw std::runtime_error("MiMo CPU experts: expert id out of range");
            int16_t& sl = slot_[(size_t) e];
            if (sl < 0) {
                used[n] = e;
                sl = (int16_t) n++;
                cpu::MoeExpert& x = jobs_[(size_t) sl];
                x.gate = s.matrix(0, e);
                x.up = s.matrix(1, e);
                x.down = s.matrix(2, e);
                x.nt = 0;
            }
            cpu::MoeExpert& x = jobs_[(size_t) sl];
            x.tok[x.nt] = t;
            x.w[x.nt] = w[(size_t) t * k + j];
            ++x.nt;
        }
    for (int i = 0; i < n; ++i) slot_[(size_t) used[i]] = -1;   // the table is all -1 again for the next call
    const void* a[cpu::MAXT];
    float* o[cpu::MAXT];
    for (int t = 0; t < n_tok; ++t) {
        a[t] = act + (size_t) t * act_stride;
        o[t] = out + (size_t) t * n_embd_;
    }
    pool_.run_moe(fmt_[(size_t) layer], jobs_.data(), n, a, n_tok, o);
    experts_run_ += n;
    bytes_read_ += (int64_t) n * (int64_t) s.expert_bytes();
}

}  // namespace strata::mimo
