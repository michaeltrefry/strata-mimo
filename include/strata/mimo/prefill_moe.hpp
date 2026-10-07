// include/strata/mimo/prefill_moe.hpp - MiMo's routed experts for a prompt chunk: P tokens (1..4096) x top-8 of one
// MoE layer, out[t] = sum over the 8 picks of w * down(silu(gate x) * up x).
//
// Per layer (run_layer):
//   1. group by expert on the device: a histogram of the P*8 picks; the counts go to the host (the one host sync per
//      layer), which orders the used experts (resident ones first, then the ones in RAM), and a stable placement
//      kernel lays the picks out in that order (deterministic: no atomics decide an order)
//   2. the used experts that are not resident are copied from RAM (GGUF layout, gate | up | down, 13.4 MB) into a
//      ring of device staging buffers on a second stream, while the GPU computes the resident ones (events between
//      the two streams; the ring carries over from one layer to the next)
//   3. every expert is converted on the device into Q8_0 (from its VRAM slot or from the staging buffer).  MXFP4 to
//      Q8_0 is exact: a block's 32 values are kvalues_mxfp4 (integers -12..12) times 2^(e - 128), which is a Q8_0
//      block with d = 2^(e - 128) as fp16 (exact for e in 104..143; outside that the block is rounded, see
//      docs/MIMO_PREFILL_MOE.md for how often that happens: never in the model's experts)
//   4. llama.cpp's Q8_0 MMQ (int8 tensor cores, activations as q8_1) for gate|up of a group of experts in one
//      launch, SiLU * up, h to q8_1, down, into one F32 row per pick; then a combine kernel sums each token's 8
//      rows times their weights in pick order (no atomics: the same input gives the same bits)
//
// Why Q8_0 and not ggml's MXFP4 MMQ: on sm_120 that kernel quantizes the activations to FP4 (block_fp4_mmq) for the
// native FP4 MMA, and FP4 activations are far coarser than q8_1 (docs/MIMO_PREFILL_MOE.md).  The Q8_0 conversion
// keeps llama.cpp's non-Blackwell arithmetic (int8 weights x q8_1 activations) at twice the weight bytes, which the
// MMQ reads from VRAM, not over PCIe.
#pragma once

#include "strata/mimo/model.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <string>

namespace strata::mimo {

struct PrefillMoeConfig {
    int max_tokens = 4096;   // P per call
    int k = 8;               // picks per token (n_expert_used)
    int group = 16;          // resident experts per MMQ launch
    int group_ram = 8;       // copied experts per MMQ launch (smaller: the last group after the last copy is the tail)
    int staging = 16;        // device buffers for experts in flight from RAM (13.4 MB each)
};

/// What one layer's call needs.  Device pointers unless said otherwise.
struct PrefillMoeLayer {
    int n_tok = 0;                               // P, 1..max_tokens
    const float* x = nullptr;                    // [P][4096] the ffn_norm output
    const int32_t* ids = nullptr;                // [P][k] the router's picks (expert ids; < 0 = none)
    const float* w = nullptr;                    // [P][k] their weights
    float* out = nullptr;                        // [P][4096] overwritten with the weighted sums
    const int32_t* resident = nullptr;           // HOST [256]: expert -> VRAM slot, -1 = only in RAM
    const unsigned long long* slot_addr = nullptr;   // HOST: slot -> device address of the slot (expert_mxfp4.hpp format)
    ExpertSlices ram;                            // HOST: the layer's experts in GGUF layout (CpuExperts::source(l) or
                                                 // the model's mapping); pinned (register_host) for full PCIe speed
};

/// Timings of the last call (milliseconds, from CUDA events; times() waits for the call to finish).
struct PrefillMoeTimes {
    float total = 0;      // start of the call -> `out` written
    float routing = 0;    // start -> picks grouped (includes the host's plan)
    float copy = 0;       // first copy started -> last copy done (the copy stream's span; 0 when nothing was copied)
    float tail = 0;       // last copy done -> `out` written
    int used = 0, resident = 0, copied = 0;      // experts with picks; of them resident; copied from RAM
    size_t copied_bytes = 0;
};

class PrefillMoe {
public:
    /// Benchmark modes: `compute_only` skips the copies (staging holds stale bytes: wrong results, compute time);
    /// `copy_only` skips the conversions and products (PCIe time).
    enum class Mode { full, compute_only, copy_only };

    explicit PrefillMoe(const PrefillMoeConfig& cfg = {});
    ~PrefillMoe();
    PrefillMoe(const PrefillMoe&) = delete;
    PrefillMoe& operator=(const PrefillMoe&) = delete;

    /// Allocates the device buffers (device_bytes(), a few large allocations) and the copy stream.  Refuses when the
    /// GPU has less free memory than that plus 256 MiB (on Windows an allocation past the free VRAM lands in system
    /// memory instead of failing).  False with `err` on failure.
    bool init(std::string& err);
    size_t device_bytes() const;

    /// One MoE layer.  Enqueued on `stream` (and the internal copy stream); the host waits once, for the routing
    /// counts, then returns with the work in flight: `out` is ready when `stream` reaches it.  Inputs must stay
    /// valid until then; the host tables (`resident`, `slot_addr`, `ram`) are read before it returns.
    void run_layer(const PrefillMoeLayer& layer, cudaStream_t stream, Mode mode = Mode::full);
    PrefillMoeTimes times();

    /// Pins host memory for DMA (cudaHostRegister).  False with `err` when the driver refuses.
    static bool register_host(void* p, size_t bytes, std::string& err);
    static void unregister_host(void* p);

private:
    PrefillMoeConfig cfg_;
    uint8_t* dev_ = nullptr;          // one allocation: the buffers below
    size_t dev_bytes_ = 0;
    uint8_t* staging_ = nullptr;      // [staging][13,369,344] raw experts
    uint8_t* gu_q8_ = nullptr;        // [max(group, group_ram)][4096 rows x 4096] Q8_0 gate|up
    uint8_t* d_q8_ = nullptr;         // [..][4096 rows x 2048] Q8_0 down
    void* xq_ = nullptr;              // x rows in pick order, q8_1 (MMQ layout)
    void* hq_ = nullptr;              // a group's h, q8_1
    float* gu_ = nullptr;             // [P*k][4096] gate|up products, then the down products (same rows)
    float* h_ = nullptr;              // [rows of a group][2048]
    int32_t *counts_ = nullptr, *row_tok_ = nullptr, *pos_ = nullptr, *iota_ = nullptr, *plan_ = nullptr;
    int32_t* h_counts_ = nullptr;     // pinned
    int32_t* h_plan_ = nullptr;       // pinned: rank[256] | bounds[n+1] | per group: relative bounds
    cudaStream_t copy_ = nullptr;
    cudaEvent_t *arrived_ = nullptr, *freed_ = nullptr;
    cudaEvent_t ev_start_ = nullptr, ev_counts_ = nullptr, ev_plan_ = nullptr, ev_copy0_ = nullptr,
                ev_copy1_ = nullptr, ev_end_ = nullptr;
    unsigned long long ring_ = 0;     // copies issued so far (the staging ring's position)
    void* mmq_ = nullptr;             // prefill::mmq::Context
    PrefillMoeTimes times_;
    bool timed_copy_ = false;
};

/// The device converters, exposed for the test: one expert into Q8_0 gate|up ([4096 rows][128 blocks]) and down
/// ([4096][64]).  `raw` is gate | up | down in GGUF layout; `slot` the expert_mxfp4.hpp slot format.
void mxfp4_raw_to_q8_0(const void* raw, void* gu_q8, void* d_q8, cudaStream_t stream);
void mxfp4_slot_to_q8_0(const void* slot, void* gu_q8, void* d_q8, cudaStream_t stream);
inline constexpr size_t kPfGuQ8Bytes = 4096ull * 128 * 34, kPfDownQ8Bytes = 4096ull * 64 * 34;

}  // namespace strata::mimo
