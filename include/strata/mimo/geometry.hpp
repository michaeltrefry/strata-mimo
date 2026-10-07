// include/strata/mimo/geometry.hpp - MiMo-V2.6-Flash ("mimo2" in llama.cpp) model shape, read from the GGUF.
//
// Every number comes from the file's `mimo2.*` metadata; `check_mimo_tensors` then holds each tensor the engine
// reads to the shape and type those numbers imply, so a different model (or a different conversion) is refused
// with the tensor and the expected shape named instead of being run wrong.
//
// The forward pass this describes (llama.cpp src/models/mimo2.cpp, the reference):
//   x = token_embd[tok]
//   per layer l:   h = rmsnorm(x) * attn_norm
//                  q|k|v = attn_qkv(h)            (q: n_head x head_k, k: n_head_kv(l) x head_k, v: n_head_kv(l) x head_v)
//                  NEOX rope on the first n_rot dims of q and k (base: rope_base, or rope_base_swa on SWA layers)
//                  a = softmax(q.k / sqrt(head_k), + one sink logit per head on SWA layers) . v
//                      (SWA layers: keys with pos_q - pos_k < n_swa)
//                  x = x + attn_output(a) * value_scale
//                  h = rmsnorm(x) * ffn_norm
//                  layer 0: x = x + down(silu(gate(h)) * up(h))                       (dense, n_ff_dense)
//                  else:    p = sigmoid(gate_inp(h)); pick top n_expert_used of p + exp_probs_b;
//                           w = p[picked] / sum(p[picked]);  x = x + sum w_i * expert_i(h)  (n_ff_exp)
//   logits = output(rmsnorm(x) * output_norm)
#pragma once

#include "strata/artifact/gguf_reader.hpp"

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::mimo {

inline constexpr uint32_t kTypeF32 = 0, kTypeQ8_0 = 8, kTypeMXFP4 = 39;
// attention / dense weights may also be re-quantized K-quants (served by native_mmvq and dequant_f16)
inline constexpr uint32_t kTypeQ4_K = 12, kTypeQ5_K = 13, kTypeQ6_K = 14;

struct MimoGeometry {
    int n_layer = 0, n_embd = 0, n_vocab = 0, n_head = 0;
    int head_k = 0, head_v = 0, n_rot = 0, n_swa = 0;
    int n_ff_dense = 0, n_expert = 0, n_expert_used = 0, n_ff_exp = 0;
    int gating_func = 0;   // llama.cpp LLAMA_EXPERT_GATING_FUNC_TYPE: 2 = sigmoid (the only one supported)
    float rope_base = 0, rope_base_swa = 0, rms_eps = 0, value_scale = 1, expert_weights_scale = 0;
    uint64_t n_ctx_train = 0;
    std::vector<int> n_head_kv;    // per layer
    std::vector<uint8_t> swa;      // per layer: 1 = sliding window, 0 = global
    std::vector<uint8_t> moe;      // per layer: 1 = routed experts, 0 = dense FFN (from the tensors present)

    bool is_swa(int l) const { return swa[(size_t) l] != 0; }
    bool is_moe(int l) const { return moe[(size_t) l] != 0; }
    /// Rows of the fused attn_qkv output of layer l: q, then k, then v.
    int qkv_rows(int l) const { return head_k * (n_head + n_head_kv[(size_t) l]) + head_v * n_head_kv[(size_t) l]; }
    int q_rows() const { return head_k * n_head; }
    int attn_out_in() const { return head_v * n_head; }
    int count_swa() const { int n = 0; for (auto s : swa) n += s; return n; }
    int count_moe() const { int n = 0; for (auto m : moe) n += m; return n; }
    float attn_scale() const { return 1.0f / std::sqrt((float) head_k); }
};

namespace detail {
inline const MetaValue* key(const GgufFile& g, const std::string& k, std::string& err) {
    const MetaValue* v = g.get("mimo2." + k);
    if (!v && err.empty()) err = "missing mimo2." + k;
    return v;
}
inline bool int_key(const GgufFile& g, const std::string& k, int& out, std::string& err) {
    const MetaValue* v = key(g, k, err);
    if (!v) return false;
    if (!v->is_num()) { err = "mimo2." + k + " is not a number"; return false; }
    out = (int) v->num();
    return true;
}
inline bool float_key(const GgufFile& g, const std::string& k, float& out, std::string& err, bool required = true) {
    const MetaValue* v = g.get("mimo2." + k);
    if (!v) {
        if (required && err.empty()) err = "missing mimo2." + k;
        return !required;
    }
    if (!v->is_num()) { err = "mimo2." + k + " is not a number"; return false; }
    out = (float) v->num();
    return true;
}
// a per-layer array, or one number meaning "every layer"
inline bool layer_array(const GgufFile& g, const std::string& k, int n_layer, std::vector<int>& out, std::string& err) {
    const MetaValue* v = key(g, k, err);
    if (!v) return false;
    out.assign((size_t) n_layer, 0);
    if (v->type == MetaType::ARRAY) {
        if (v->count != (uint64_t) n_layer || v->items.size() != (size_t) n_layer) {
            err = "mimo2." + k + " has " + std::to_string(v->count) + " entries, expected " + std::to_string(n_layer);
            return false;
        }
        for (int l = 0; l < n_layer; ++l) out[(size_t) l] = (int) v->items[(size_t) l].num();
    } else {
        for (auto& x : out) x = (int) v->num();
    }
    return true;
}
inline std::string shape_str(const std::vector<uint64_t>& s) {
    std::string r = "[";
    for (size_t i = 0; i < s.size(); ++i) r += (i ? ", " : "") + std::to_string(s[i]);
    return r + "]";
}
}  // namespace detail

/// Reads the shape from the metadata shard and the tensor directory (dense vs MoE layers). Empty string = ok.
inline std::string read_mimo_geometry(const GgufModel& model, MimoGeometry& g) {
    const GgufFile& m = model.meta();
    const MetaValue* arch = m.get("general.architecture");
    if (!arch) return "missing general.architecture";
    if (arch->s != "mimo2") return "architecture is '" + arch->s + "', this engine requires 'mimo2' (MiMo-V2.6-Flash)";
    std::string err;
    bool ok = detail::int_key(m, "block_count", g.n_layer, err) && detail::int_key(m, "embedding_length", g.n_embd, err) &&
              detail::int_key(m, "attention.head_count", g.n_head, err) &&
              detail::int_key(m, "attention.key_length", g.head_k, err) &&
              detail::int_key(m, "attention.value_length", g.head_v, err) &&
              detail::int_key(m, "rope.dimension_count", g.n_rot, err) &&
              detail::int_key(m, "attention.sliding_window", g.n_swa, err) &&
              detail::int_key(m, "feed_forward_length", g.n_ff_dense, err) &&
              detail::int_key(m, "expert_count", g.n_expert, err) &&
              detail::int_key(m, "expert_used_count", g.n_expert_used, err) &&
              detail::int_key(m, "expert_feed_forward_length", g.n_ff_exp, err) &&
              detail::int_key(m, "expert_gating_func", g.gating_func, err) &&
              detail::float_key(m, "rope.freq_base", g.rope_base, err) &&
              detail::float_key(m, "rope.freq_base_swa", g.rope_base_swa, err) &&
              detail::float_key(m, "attention.layer_norm_rms_epsilon", g.rms_eps, err) &&
              detail::float_key(m, "attention.value_scale", g.value_scale, err, false) &&
              detail::float_key(m, "expert_weights_scale", g.expert_weights_scale, err, false);
    if (!ok) return err;
    if (g.n_layer <= 0 || g.n_layer > 256) return "implausible mimo2.block_count " + std::to_string(g.n_layer);
    std::vector<int> kv, swa;
    if (!detail::layer_array(m, "attention.head_count_kv", g.n_layer, kv, err) ||
        !detail::layer_array(m, "attention.sliding_window_pattern", g.n_layer, swa, err))
        return err;
    g.n_head_kv = kv;
    g.swa.assign(swa.begin(), swa.end());
    if (const MetaValue* c = m.get("mimo2.context_length")) g.n_ctx_train = c->u;
    if (g.gating_func != 2) return "mimo2.expert_gating_func = " + std::to_string(g.gating_func) + ", only 2 (sigmoid) is supported";
    if (g.n_rot <= 0 || g.n_rot > g.head_k || g.n_rot % 2) return "mimo2.rope.dimension_count " + std::to_string(g.n_rot) + " does not fit head_k";
    if (g.n_expert_used <= 0 || g.n_expert_used > g.n_expert || g.n_expert > 1024) return "implausible expert counts";
    for (int l = 0; l < g.n_layer; ++l)
        if (kv[(size_t) l] <= 0 || g.n_head % kv[(size_t) l]) return "layer " + std::to_string(l) + ": n_head not a multiple of n_head_kv";
    const TensorInfo* emb = model.find("token_embd.weight");
    if (!emb || emb->shape.size() != 2) return "missing token_embd.weight";
    g.n_vocab = (int) emb->shape[1];
    g.moe.assign((size_t) g.n_layer, 0);
    for (int l = 0; l < g.n_layer; ++l)
        g.moe[(size_t) l] = model.find("blk." + std::to_string(l) + ".ffn_gate_inp.weight") ? 1 : 0;
    return {};
}

/// Holds every tensor the engine reads to the shape and type the geometry implies. Empty string = ok.
inline std::string check_mimo_tensors(const GgufModel& model, const MimoGeometry& g) {
    using Shape = std::vector<uint64_t>;
    auto want = [&](const std::string& name, const Shape& shape, std::vector<uint32_t> types) -> std::string {
        size_t shard = 0;
        const TensorInfo* t = model.find(name, &shard);
        if (!t) return "missing tensor " + name;
        if (t->shape != shape) return name + " is " + detail::shape_str(t->shape) + ", expected " + detail::shape_str(shape);
        bool type_ok = false;
        for (uint32_t ty : types) type_ok |= t->type == ty;
        if (!type_ok) return name + " has type " + t->type_name() + ", which this engine does not read";
        if (!model.in_bounds(*t, shard)) return name + " runs past the end of its file (truncated download?)";
        return {};
    };
    const uint64_t E = (uint64_t) g.n_embd, V = (uint64_t) g.n_vocab;
    std::string e;
    if (!(e = want("token_embd.weight", {E, V}, {kTypeQ8_0})).empty()) return e;
    if (!(e = want("output.weight", {E, V}, {kTypeQ8_0})).empty()) return e;
    if (!(e = want("output_norm.weight", {E}, {kTypeF32})).empty()) return e;
    for (int l = 0; l < g.n_layer; ++l) {
        const std::string b = "blk." + std::to_string(l) + ".";
        const uint64_t qkv = (uint64_t) g.qkv_rows(l);
        if (!(e = want(b + "attn_norm.weight", {E}, {kTypeF32})).empty()) return e;
        if (!(e = want(b + "ffn_norm.weight", {E}, {kTypeF32})).empty()) return e;
        const std::vector<uint32_t> dense = {kTypeQ8_0, kTypeQ6_K, kTypeQ5_K, kTypeQ4_K};
        if (!(e = want(b + "attn_qkv.weight", {E, qkv}, dense)).empty()) return e;
        if (!(e = want(b + "attn_output.weight", {(uint64_t) g.attn_out_in(), E}, dense)).empty()) return e;
        const bool has_sinks = model.find(b + "attn_sinks.weight") != nullptr;
        if (has_sinks != g.is_swa(l))
            return b + "attn_sinks.weight " + (has_sinks ? "present on a global" : "missing on a sliding-window") + " layer";
        if (has_sinks && !(e = want(b + "attn_sinks.weight", {(uint64_t) g.n_head}, {kTypeF32})).empty()) return e;
        if (g.is_moe(l)) {
            const uint64_t N = (uint64_t) g.n_expert, F = (uint64_t) g.n_ff_exp;
            if (!(e = want(b + "ffn_gate_inp.weight", {E, N}, {kTypeF32})).empty()) return e;
            if (!(e = want(b + "exp_probs_b.bias", {N}, {kTypeF32})).empty()) return e;
            if (!(e = want(b + "ffn_gate_exps.weight", {E, F, N}, {kTypeMXFP4})).empty()) return e;
            if (!(e = want(b + "ffn_up_exps.weight", {E, F, N}, {kTypeMXFP4})).empty()) return e;
            if (!(e = want(b + "ffn_down_exps.weight", {F, E, N}, {kTypeMXFP4})).empty()) return e;
        } else {
            const uint64_t F = (uint64_t) g.n_ff_dense;
            if (!(e = want(b + "ffn_gate.weight", {E, F}, {kTypeQ8_0})).empty()) return e;
            if (!(e = want(b + "ffn_up.weight", {E, F}, {kTypeQ8_0})).empty()) return e;
            if (!(e = want(b + "ffn_down.weight", {F, E}, {kTypeQ8_0})).empty()) return e;
        }
    }
    return {};
}

}  // namespace strata::mimo
