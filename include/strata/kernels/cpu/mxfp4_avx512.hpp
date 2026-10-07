// include/strata/kernels/cpu/mxfp4_avx512.hpp - MiMo step 2: MXFP4 expert rows (ggml type 39) in 512-bit lanes,
// several tokens per weight row.
//
// ggml-cpu's `ggml_vec_dot_mxfp4_q8_0` is AVX2 and single-token: every token re-decodes the 4-bit codes.  Here two
// 32-value blocks are decoded once (one masked load, one `vpermb`, one table lookup, one sign mask) and every token
// applies them with a masked subtract, a `vpdpbusd`, a convert and an FMA.  The arithmetic, including the order of
// every float operation, is ggml's AVX2 path: per block the int32 lanes of four products, times
// fp16(d_y) * e8m0_half(e), accumulated even blocks / odd blocks in two 8-lane sums, then ggml's `hsum_float_8` of
// their sum.  So each token's row is BITWISE ggml's dot (mimo_cpu_experts_parity checks it).
//
// The activations are ggml's `block_q8_0` rows (34 bytes per 32 values, what `native_quant_act` writes); each call
// copies them once into a planar form (the int8 values contiguous, the scales pre-broadcast) that it then reads
// with aligned 64-byte loads.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels::cpu {

/// Whether this CPU runs the kernel (AVX-512 F/BW/VL/DQ/VNNI/VBMI; probed without AVX-512, so safe to ask anywhere).
bool mxfp4_512_ok();
/// out[t][r] = w_r . a[t] for rows [r0, r1); `w` is row 0 (row_bytes = n / 32 * 17), `act[t]` n values of Q8_0.
/// At most 8 tokens; n a multiple of 64 and at most 8192.
void mxfp4_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out, int r0,
                int r1);
/// g[t][r] = gate_r . a[t] and u[t][r] = up_r . a[t] for rows [r0, r1), row by row (one copy of the activations
/// for both matrices).  The SwiGLU is the caller's (native_gu_rows_at), so it is the same code as ggml's path.
void mxfp4_rows2(const uint8_t* gate, const uint8_t* up, size_t row_bytes, int n, const void* const* act, int nt,
                 float* const* g, float* const* u, int r0, int r1);
/// Software prefetch distance in bytes ahead of the row being read (STRATA_MXFP4_PREFETCH; 0 = off).
int mxfp4_prefetch_bytes();
void mxfp4_set_prefetch_bytes(int bytes);

}  // namespace strata::kernels::cpu
