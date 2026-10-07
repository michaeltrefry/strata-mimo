// mimo_dense_q8_0_bench: Strata's Q8_0 MMVQ (native_q8_0_mmvq, include/strata/kernels/native_mmvq.hpp) on the dense
// matrices of one MiMo-V2.6-Flash decode step, T = 1..4 tokens.  Synthetic Q8_0 weights (random codes, small fp16
// scales); timing only.  Each shape is cycled over enough copies (>= 512 MB per cycle) that the 128 MB L2 does not
// serve a repeat, 32 calls per CUDA graph, repeated; GB/s = weight bytes / time.
//
// First line: what a plain streaming read reaches on this card (128-bit loads, grid-stride, 1 GiB), the ceiling
// the GB/s columns compare against.
//
// usage: mimo_dense_q8_0_bench [--reps N]
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CK(x)                                                                                       \
    do {                                                                                            \
        cudaError_t e_ = (x);                                                                       \
        if (e_ != cudaSuccess) {                                                                    \
            std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
            std::exit(1);                                                                           \
        }                                                                                           \
    } while (0)

namespace {
struct Shape {
    const char* name;
    int n_in, n_out;
};

__global__ void read_kernel(const uint4* __restrict__ p, size_t n, unsigned* __restrict__ sink) {
    unsigned acc = 0;
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        const uint4 v = __ldcs(p + i);
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x12345678u) *sink = acc;   // keeps the loads
}

// random Q8_0 blocks: fp16 scale in [1e-3, 2e-3), int8 codes
void fill(std::vector<uint8_t>& h, std::mt19937& rng) {
    for (size_t b = 0; b + 34 <= h.size(); b += 34) {
        const __half d = __float2half(1e-3f * (1.0f + (rng() % 1000) / 1000.0f));
        std::memcpy(&h[b], &d, 2);
        for (int i = 0; i < 32; ++i) h[b + 2 + i] = (uint8_t) rng();
    }
}
}  // namespace

int main(int argc, char** argv) {
    int reps = 20;
    for (int i = 1; i < argc; ++i)
        if (!std::strcmp(argv[i], "--reps") && i + 1 < argc) reps = std::atoi(argv[++i]);
    const Shape shapes[] = {{"attn_qkv global 4096->13568", 4096, 13568}, {"attn_qkv SWA 4096->14848", 4096, 14848},
                            {"attn_output 8192->4096", 8192, 4096},       {"ffn gate/up 4096->16384", 4096, 16384},
                            {"ffn down 16384->4096", 16384, 4096},        {"output 4096->152576", 4096, 152576}};
    std::mt19937 rng(5);
    cudaStream_t s;
    CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    float* x;
    void* xq;
    float* y;
    CK(cudaMalloc(&x, sizeof(float) * 8 * 16384));
    CK(cudaMalloc(&xq, strata::kernels::native_q8_1_bytes(16384, 8)));
    CK(cudaMalloc(&y, sizeof(float) * 8 * 152576));
    {
        std::vector<float> hx(8 * 16384);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (auto& v : hx) v = nd(rng);
        CK(cudaMemcpy(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice));
    }
    {
        const size_t n = (size_t) 1 << 30;
        void* buf;
        unsigned* sink;
        CK(cudaMalloc(&buf, n));
        CK(cudaMalloc(&sink, 4));
        CK(cudaMemset(buf, 1, n));
        int sms = 0;
        CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
        cudaEvent_t e0, e1;
        CK(cudaEventCreate(&e0));
        CK(cudaEventCreate(&e1));
        double best = 1e30;
        for (int r = 0; r < reps; ++r) {
            CK(cudaEventRecord(e0, s));
            read_kernel<<<sms * 8, 256, 0, s>>>((const uint4*) buf, n / 16, sink);
            CK(cudaEventRecord(e1, s));
            CK(cudaEventSynchronize(e1));
            float ms = 0;
            CK(cudaEventElapsedTime(&ms, e0, e1));
            best = std::min(best, (double) ms);
        }
        std::printf("streaming read of 1 GiB: best %.0f GB/s\n", n / (best * 1e-3) / 1e9);
        cudaFree(buf);
        cudaFree(sink);
    }
    std::printf("native_q8_0_mmvq, %d reps of a %d-call graph; us per call (best / median / worst), best GB/s\n", reps, 32);
    std::printf("  %-30s %10s", "shape", "MB");
    for (int T = 1; T <= 4; ++T) std::printf("   T=%d us (best/med/worst)  GB/s", T);
    std::printf("\n");
    for (const Shape& sh : shapes) {
        const size_t bytes = (size_t) sh.n_out * (sh.n_in / 32) * 34;
        const int copies = std::max<int>(2, (int) ((512u << 20) / bytes + 1));
        std::vector<void*> w((size_t) copies);
        std::vector<uint8_t> h(bytes);
        fill(h, rng);
        for (auto& p : w) {
            CK(cudaMalloc(&p, bytes));
            CK(cudaMemcpy(p, h.data(), bytes, cudaMemcpyHostToDevice));
        }
        std::printf("  %-30s %10.1f", sh.name, bytes / 1e6);
        for (int T = 1; T <= 4; ++T) {
            strata::kernels::native_quantize_q8_1(x, xq, sh.n_in, T, s);
            const int calls = 32;
            cudaGraph_t graph;
            cudaGraphExec_t exec;
            CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
            for (int c = 0; c < calls; ++c)
                strata::kernels::native_q8_0_mmvq(w[(size_t) (c % copies)], xq, y, sh.n_in, sh.n_out, T, s);
            CK(cudaStreamEndCapture(s, &graph));
            CK(cudaGraphInstantiate(&exec, graph, 0));
            cudaEvent_t e0, e1;
            CK(cudaEventCreate(&e0));
            CK(cudaEventCreate(&e1));
            CK(cudaGraphLaunch(exec, s));
            CK(cudaStreamSynchronize(s));
            std::vector<double> us;
            for (int r = 0; r < reps; ++r) {
                CK(cudaEventRecord(e0, s));
                CK(cudaGraphLaunch(exec, s));
                CK(cudaEventRecord(e1, s));
                CK(cudaEventSynchronize(e1));
                float ms = 0;
                CK(cudaEventElapsedTime(&ms, e0, e1));
                us.push_back(ms * 1000.0 / calls);
            }
            std::sort(us.begin(), us.end());
            std::printf("   %7.1f/%7.1f/%7.1f %7.0f", us.front(), us[us.size() / 2], us.back(),
                        bytes / (us.front() * 1e-6) / 1e9);
            cudaEventDestroy(e0);
            cudaEventDestroy(e1);
            cudaGraphExecDestroy(exec);
            cudaGraphDestroy(graph);
        }
        std::printf("\n");
        for (auto p : w) cudaFree(p);
    }
    return 0;
}
