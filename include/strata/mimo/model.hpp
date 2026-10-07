// include/strata/mimo/model.hpp - MiMo-V2.6-Flash weights: the GGUF opened (mmap), its shape checked, every
// non-expert tensor uploaded unchanged into one VRAM arena, and the routed experts located in the mapped file.
//
// Nothing is repacked: a device tensor holds the GGUF's own blocks (Q8_0 matrices, F32 norms/routers/sinks), so a
// kernel reads exactly what llama.cpp reads.  The experts stay in the file mapping here; where each one lives at
// run time (VRAM slot or RAM) is the expert cache's decision, not the loader's.
#pragma once

#include "strata/mimo/geometry.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::mimo {

/// One tensor in the VRAM arena: ne0 = elements per row (GGUF dim 0), ne1 = rows.
struct DevTensor {
    const void* data = nullptr;
    uint32_t type = 0;
    int64_t ne0 = 0, ne1 = 1;
    size_t bytes = 0;
    explicit operator bool() const { return data != nullptr; }
};

struct LayerWeights {
    DevTensor attn_norm, attn_qkv, attn_output, attn_sinks, ffn_norm;
    DevTensor ffn_gate, ffn_up, ffn_down;     // layer 0 (dense)
    DevTensor gate_inp, exp_probs_b;          // MoE layers: router [n_embd x n_expert] F32, selection bias
};

/// Where expert e of one MoE layer is in the mapped file: matrix m (0 gate, 1 up, 2 down) at base[m] + e * stride[m].
struct ExpertSlices {
    const uint8_t* base[3] = {nullptr, nullptr, nullptr};
    size_t stride[3] = {0, 0, 0};   // bytes of one expert's matrix
    size_t row_bytes[3] = {0, 0, 0};
    const uint8_t* matrix(int m, int e) const { return base[m] + (size_t) e * stride[m]; }
    size_t expert_bytes() const { return stride[0] + stride[1] + stride[2]; }
};

class MimoModel {
public:
    /// Opens every shard of the model `any_shard` belongs to and checks its shape and tensors. Throws on error.
    explicit MimoModel(const std::string& any_shard);
    ~MimoModel();
    MimoModel(const MimoModel&) = delete;
    MimoModel& operator=(const MimoModel&) = delete;

    /// Copies every non-expert tensor into one VRAM allocation. False with `err` on failure.
    bool upload(std::string& err);

    const MimoGeometry& geometry() const { return geo_; }
    const GgufModel& gguf() const { return *gguf_; }
    const LayerWeights& layer(int l) const { return layers_[(size_t) l]; }
    const DevTensor& token_embd() const { return token_embd_; }
    const DevTensor& output() const { return output_; }
    const DevTensor& output_norm() const { return output_norm_; }
    /// Experts of layer l (empty for the dense layer).
    const ExpertSlices& experts(int l) const { return experts_[(size_t) l]; }
    size_t expert_bytes() const;    // one expert, gate + up + down
    size_t arena_bytes() const { return arena_bytes_; }
    /// The host bytes of a tensor in the mapping (nullptr when absent).
    const uint8_t* host_data(const std::string& name, const TensorInfo** info = nullptr) const;

private:
    std::unique_ptr<GgufModel> gguf_;
    MimoGeometry geo_;
    std::vector<LayerWeights> layers_;
    std::vector<ExpertSlices> experts_;
    DevTensor token_embd_, output_, output_norm_;
    void* arena_ = nullptr;
    size_t arena_bytes_ = 0;
};

}  // namespace strata::mimo
