// include/strata/kernels/cpu/native_expert.hpp - plan v0.3 P6: one routed expert in its GGUF form on the CPU.
//
// The IQ2_XS / IQ3_XXS model files keep their experts in i-quant formats (IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS,
// IQ3_S gate/up; Q2_0 or IQ4_NL down) whose values cannot be re-expressed in the Q2_0 pack form.  A native expert
// blob is the three GGUF slices back to back, [gate rows | up rows | down rows], and the arithmetic is ggml-cpu's
// own (`ggml_get_type_traits_cpu`): the activation is quantized with the weight type's `vec_dot_type` and each row
// is one `vec_dot`, exactly what llama.cpp's CPU backend computes for the same tensor.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace strata::kernels::cpu {

/// Bytes of the largest quantized activation any native layer uses: MiMo's 4096 values as Q8_0 are 128 x 34 = 4352,
/// as Q8_K 16 x 292 = 4672 (Qwen's 2560 as Q8_K: 10 x 292).
inline constexpr size_t kNativeActBytes = 4672;
/// Bytes of the largest quantized down activation: MiMo's 2048 values as Q8_0 are 64 x 34 = 2176, as Q8_K
/// 8 x 292 = 2336 (Qwen's 640 as Q8_0: 20 x 34, or Q8_K 3 x 292).
inline constexpr size_t kNativeHBytes = 2336;
/// The widest expert intermediate (n_ff) the pool's buffers hold: MiMo's 2048 (Qwen: 640).
inline constexpr int kNativeMaxFF = 2048;
/// The widest n_embd the pool's buffers hold: MiMo's 4096 (Qwen: 2560).
inline constexpr int kNativeMaxEmbd = 4096;

/// One layer's native expert geometry.
struct NativeFmt {
    int gu_type = -1, d_type = -1;      ///< ggml types of gate/up and of down
    int gu_act = -1, d_act = -1;        ///< their vec_dot_type (the activation formats)
    int64_t n_embd = 0, n_ff = 0;
    size_t gu_row = 0, d_row = 0;       ///< bytes per weight row
    size_t up_off = 0, down_off = 0;    ///< inside the blob
    size_t bytes = 0;                   ///< the whole blob
    size_t act_bytes = 0, h_bytes = 0;  ///< quantized activation sizes (n_embd of gu_act, n_ff of d_act)
    int h_block = 1;                    ///< values per block of d_act (32 for Q8_0, 256 for Q8_K)
};

/// Whether this build has the ggml-cpu path.
bool native_experts_available() noexcept;
/// Fills `f` for a layer; false (with a reason) when ggml-cpu has no dot product for a type.
bool native_fmt(int gu_type, int d_type, int64_t n_embd, int64_t n_ff, NativeFmt& f, std::string& err);

/// x (n_embd floats) -> the gate/up activation (act_bytes).
void native_quant_act(const NativeFmt& f, const float* x, void* dst);
/// h (n_ff floats) -> the down activation (h_bytes).
void native_quant_h(const NativeFmt& f, const float* h, void* dst);

/// ff[t][r] = silu(gate_r . a[t]) * (up_r . a[t]) for rows r in [r0, r1), `nt` tokens.
void native_gu_rows(const NativeFmt& f, const uint8_t* blob, const void* const* act, int nt, float* const* ff,
                    int r0, int r1);
/// out[t][r] = down_r . hq[t] for rows r in [r0, r1).
void native_down_rows(const NativeFmt& f, const uint8_t* blob, const void* const* hq, int nt, float* const* out,
                      int r0, int r1);

/// The same two with each matrix given by its own first row instead of one blob: MiMo's experts are read in place
/// from the GGUF, where an expert's gate, up and down rows sit in three different tensors.  `gate`/`up`/`down` point
/// at row 0 of that expert's matrix; `native_gu_rows(f, blob, ...)` is `native_gu_rows_at(f, blob, blob + up_off,
/// ...)`.  MXFP4 (ggml type 39) runs on the AVX-512 kernel of mxfp4_avx512.hpp when the CPU has it (bitwise
/// ggml's `ggml_vec_dot_mxfp4_q8_0`; STRATA_NO_MXFP4_512=1 takes ggml's own dot instead).
void native_gu_rows_at(const NativeFmt& f, const uint8_t* gate, const uint8_t* up, const void* const* act, int nt,
                       float* const* ff, int r0, int r1);
void native_down_rows_at(const NativeFmt& f, const uint8_t* down, const void* const* hq, int nt, float* const* out,
                         int r0, int r1);
/// Quantizes h[r0, r1) into the down activation's blocks [r0, r1) (both ends multiples of `h_block`), so the
/// thread that computed those rows of the intermediate can quantize them without a pass over the whole vector.
/// Whether MXFP4 takes the AVX-512 kernel: 1 yes (when the CPU has it), 0 ggml's dot, -1 the default
/// (STRATA_NO_MXFP4_512 unset).  A bench/test switch; set it while no expert is running.
void native_set_mxfp4_512(int mode);
bool native_mxfp4_512();
void native_quant_h_part(const NativeFmt& f, const float* h, int r0, int r1, void* dst);

}  // namespace strata::kernels::cpu
