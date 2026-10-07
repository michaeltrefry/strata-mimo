// include/strata/mimo/cpu_experts.hpp - MiMo step 2: the routed experts the GPU does not hold, computed on the CPU.
//
// One call per MoE layer: the layer's normalized hidden state for each token of the window (or its Q8_0 image), the
// experts each token picked that are NOT resident on the GPU (an id < 0 means "not mine", so the driver can pass the
// router's top-8 with the hits masked out), and their router weights.  The result is one F32 row per token: the
// weighted sum of those experts' outputs, which the GPU adds to its own share.
//
// Nothing is repacked: an expert's gate, up and down rows are read where `ExpertSlices` says they are - the GGUF
// mapping by default, or a RAM copy of a layer (`copy_layer`) with the same layout.  The arithmetic is ggml-cpu's
// (MXFP4 x Q8_0, bitwise `ggml_vec_dot_mxfp4_q8_0` through the AVX-512 kernel of kernels/cpu/mxfp4_avx512.hpp), on
// the engine's `ExpertPool`: docs/MIMO_CPU_EXPERTS.md has the measurements and the protocol the decode driver uses.
#pragma once

#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "strata/mimo/model.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace strata::mimo {

class CpuExperts {
public:
    /// The per-layer formats from the model's geometry and tensor types.  Throws when ggml-cpu cannot run them or
    /// the shape does not fit the pool's buffers.  `pool` and `model` must outlive this object.
    CpuExperts(const MimoModel& model, kernels::cpu::ExpertPool& pool);

    const kernels::cpu::NativeFmt& fmt(int layer) const { return fmt_[(size_t) layer]; }
    /// Bytes of one Q8_0 activation row (n_embd values): the stride of `run_layer_q`'s input.
    size_t act_bytes() const { return act_bytes_; }
    /// Where the CPU reads layer l's experts (the GGUF mapping unless changed).
    const ExpertSlices& source(int layer) const { return src_[(size_t) layer]; }
    void set_source(int layer, const ExpertSlices& s) { src_[(size_t) layer] = s; }
    /// Bytes of all experts of one MoE layer (gate + up + down tensors).
    size_t layer_bytes(int layer) const;
    /// Copies layer l's three expert tensors from the mapping to `dst` (layer_bytes(l) bytes, gate | up | down,
    /// each in its GGUF layout) on `threads` threads, and points layer l at the copy.
    void copy_layer(int layer, uint8_t* dst, int threads = 8);

    /// x (n_tok rows of n_embd F32) -> act (n_tok rows of act_bytes(), Q8_0).  Exactly what run_layer does first.
    void quantize(int layer, int n_tok, const float* x, uint8_t* act) const;

    /// THE STEP-6 ENTRY POINT.  out[t * n_embd + i] = sum over j < k with ids[t * k + j] >= 0 of
    ///   w[t * k + j] * expert_{ids[t * k + j]}(x_t)[i]
    /// for t < n_tok <= MAXT (8).  `out` is overwritten (zeros for a token with no CPU expert).  An expert two tokens
    /// share is read once for both.  Blocks until done; the calling thread works in the pool too.
    void run_layer(int layer, int n_tok, const float* x, int k, const int32_t* ids, const float* w, float* out);
    /// The same with the activations already quantized (n_tok rows, `act_stride` bytes apart, Q8_0).
    void run_layer_q(int layer, int n_tok, const uint8_t* act, size_t act_stride, int k, const int32_t* ids,
                     const float* w, float* out);

    /// Distinct experts and bytes read by run_layer* so far (the benchmark's GB/s).
    int64_t experts_run() const { return experts_run_; }
    int64_t bytes_read() const { return bytes_read_; }

private:
    const MimoModel& model_;
    kernels::cpu::ExpertPool& pool_;
    std::vector<kernels::cpu::NativeFmt> fmt_;
    std::vector<ExpertSlices> src_;
    size_t act_bytes_ = 0;
    int n_embd_ = 0, n_expert_ = 0;
    std::vector<uint8_t> act_;                    // run_layer's quantized activations
    std::vector<int16_t> slot_;                   // expert id -> its MoeExpert this call (-1 = none)
    std::vector<kernels::cpu::MoeExpert> jobs_;
    int64_t experts_run_ = 0, bytes_read_ = 0;
};

}  // namespace strata::mimo
