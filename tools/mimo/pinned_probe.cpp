// mimo_pinned_probe: can the prompt path DMA MiMo's experts straight out of the decoder's RAM copy?
//
// The decoder keeps one malloc'd copy per MoE layer (3,422,552,064 B: gate | up | down of 256 experts, GGUF layout).
// This allocates `--layers` such buffers the same way, fills them (as copy_layer does), then pins them one by one
// with cudaHostRegister and reports how long each takes and where it stops.  Then the H2D rate of 13.4 MB expert
// copies (3 memcpys each, as the prompt path issues them) from: the registered copy, the same memory unregistered
// (pageable), and a cudaMallocHost buffer.  `--register-first` registers each buffer before filling it instead;
// `--host-alloc` allocates each layer with cudaHostAlloc (pinned from the start) instead of malloc + register.
//
// usage: mimo_pinned_probe [--layers 47] [--register-first | --host-alloc]
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

namespace {

constexpr size_t kExpert = 13369344, kGate = 2048ull * 2176, kDown = 4096ull * 1088;
constexpr size_t kLayer = 256 * kExpert;

double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

void fill(uint8_t* p, size_t n, int threads = 16) {
    std::vector<std::thread> th;
    for (int t = 0; t < threads; ++t)
        th.emplace_back([=] {
            const size_t a = n / threads * t, b = t + 1 == threads ? n : n / threads * (t + 1);
            std::memset(p + a, 0x5A + t, b - a);
        });
    for (auto& t : th) t.join();
}

// n experts of one buffer to the device, 3 copies each (gate, up, down live in three tensors of the layer)
double h2d_rate(const uint8_t* base, size_t span, uint8_t* dev, cudaStream_t s, int n) {
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    cudaEventRecord(a, s);
    for (int i = 0; i < n; ++i) {
        const int e = (i * 37) % 256;
        const uint8_t* g = base + (size_t) e * kGate;
        const uint8_t* u = base + 256 * kGate + (size_t) e * kGate;
        const uint8_t* d = base + 512 * kGate + (size_t) e * kDown;
        if (span < kLayer) {   // a small buffer: experts stored whole, one after the other
            g = base + (size_t) (i % (span / kExpert)) * kExpert;
            u = g + kGate;
            d = u + kGate;
        }
        uint8_t* dst = dev + (size_t) (i % 4) * kExpert;
        cudaMemcpyAsync(dst, g, kGate, cudaMemcpyHostToDevice, s);
        cudaMemcpyAsync(dst + kGate, u, kGate, cudaMemcpyHostToDevice, s);
        cudaMemcpyAsync(dst + 2 * kGate, d, kDown, cudaMemcpyHostToDevice, s);
    }
    cudaEventRecord(b, s);
    cudaEventSynchronize(b);
    float ms = 0;
    cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    return (double) n * kExpert / (ms * 1e-3) / 1e9;
}

}  // namespace

int main(int argc, char** argv) {
    int layers = 47;
    bool reg_first = false, host_alloc = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--layers") && i + 1 < argc) layers = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--register-first")) reg_first = true;
        else if (!std::strcmp(argv[i], "--host-alloc")) host_alloc = true;
    }
    cudaSetDevice(0);
    cudaFree(nullptr);
    cudaStream_t s;
    cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    uint8_t* dev = nullptr;
    if (cudaMalloc(&dev, 4 * kExpert) != cudaSuccess) { std::fprintf(stderr, "cudaMalloc failed\n"); return 1; }

    std::vector<uint8_t*> buf;
    std::vector<bool> pinned;
    double t_reg = 0;
    size_t reg_bytes = 0;
    std::printf("layer  alloc+fill s  register s  GB/s registered  total pinned GiB\n");
    for (int l = 0; l < layers; ++l) {
        double t0 = now_s();
        uint8_t* p = nullptr;
        double t_fill = 0, t_r = 0;
        cudaError_t e = cudaSuccess;
        if (host_alloc) {
            e = cudaHostAlloc((void**) &p, kLayer, cudaHostAllocPortable);
            t_r = now_s() - t0;
            if (e == cudaSuccess) {
                const double f0 = now_s();
                fill(p, kLayer);
                t_fill = now_s() - f0;
            } else {
                p = nullptr;
            }
        } else if (!(p = (uint8_t*) std::malloc(kLayer))) {
            std::printf("malloc failed at layer %d\n", l);
            break;
        } else if (reg_first) {
            const double r0 = now_s();
            e = cudaHostRegister(p, kLayer, cudaHostRegisterPortable);
            t_r = now_s() - r0;
            const double f0 = now_s();
            fill(p, kLayer);
            t_fill = now_s() - f0 + (r0 - t0);
        } else {
            fill(p, kLayer);
            t_fill = now_s() - t0;
            const double r0 = now_s();
            e = cudaHostRegister(p, kLayer, cudaHostRegisterPortable);
            t_r = now_s() - r0;
        }
        if (p) {
            buf.push_back(p);
            pinned.push_back(e == cudaSuccess);
        }
        if (e != cudaSuccess) {
            std::printf("%5d  %10.2f  %10.2f  REFUSED: %s\n", l, t_fill, t_r, cudaGetErrorString(e));
            (void) cudaGetLastError();
            break;
        }
        t_reg += t_r;
        reg_bytes += kLayer;
        const double gbs = h2d_rate(p, kLayer, dev, s, 64);
        std::printf("%5d  %10.2f  %10.2f  %15.1f  %16.1f\n", l, t_fill, t_r, gbs, reg_bytes / 1073741824.0);
        std::fflush(stdout);
    }
    std::printf("registered %.1f GiB in %.1f s (%.2f s per 3.19 GiB layer)\n", reg_bytes / 1073741824.0, t_reg,
                buf.empty() ? 0.0 : t_reg / std::max<size_t>(1, reg_bytes / kLayer));

    // rates: registered (first buffer, all buffers round robin), pageable, cudaMallocHost
    if (!buf.empty() && pinned[0]) {
        double best = 0;
        for (int r = 0; r < 3; ++r) best = std::max(best, h2d_rate(buf[0], kLayer, dev, s, 128));
        std::printf("H2D from the registered copy: %.1f GB/s (128 experts, 3 copies each, best of 3)\n", best);
        // the same bytes as one 13.4 MB copy per expert, and 3 copies each on two streams taking turns
        cudaStream_t s2;
        cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking);
        cudaEvent_t a, b, b2;
        cudaEventCreate(&a);
        cudaEventCreate(&b);
        cudaEventCreate(&b2);
        double one = 0, two = 0;
        for (int r = 0; r < 3; ++r) {
            cudaEventRecord(a, s);
            for (int i = 0; i < 128; ++i)
                cudaMemcpyAsync(dev + (size_t) (i % 4) * kExpert, buf[0] + (size_t) ((i * 37) % 255) * kExpert, kExpert,
                                cudaMemcpyHostToDevice, s);
            cudaEventRecord(b, s);
            cudaEventSynchronize(b);
            float ms = 0;
            cudaEventElapsedTime(&ms, a, b);
            one = std::max(one, 128.0 * kExpert / (ms * 1e-3) / 1e9);
            cudaEventRecord(a, s);
            cudaStreamWaitEvent(s2, a, 0);
            for (int i = 0; i < 128; ++i) {
                cudaStream_t q = i % 2 ? s2 : s;
                const int e = (i * 37) % 256;
                uint8_t* dst = dev + (size_t) (i % 4) * kExpert;
                cudaMemcpyAsync(dst, buf[0] + (size_t) e * kGate, kGate, cudaMemcpyHostToDevice, q);
                cudaMemcpyAsync(dst + kGate, buf[0] + 256 * kGate + (size_t) e * kGate, kGate, cudaMemcpyHostToDevice, q);
                cudaMemcpyAsync(dst + 2 * kGate, buf[0] + 512 * kGate + (size_t) e * kDown, kDown, cudaMemcpyHostToDevice, q);
            }
            cudaEventRecord(b2, s2);
            cudaStreamWaitEvent(s, b2, 0);
            cudaEventRecord(b, s);
            cudaEventSynchronize(b);
            cudaEventElapsedTime(&ms, a, b);
            two = std::max(two, 128.0 * kExpert / (ms * 1e-3) / 1e9);
        }
        std::printf("H2D from the registered copy: %.1f GB/s (one 13.4 MB copy per expert), %.1f GB/s (3 copies, two "
                    "streams)\n", one, two);
        if (buf.size() > 1) {
            double all = 0;
            int n = 0;
            for (size_t i = 0; i < buf.size(); ++i)
                if (pinned[i]) all += h2d_rate(buf[i], kLayer, dev, s, 32), ++n;
            std::printf("H2D from every registered layer, mean of 32 experts each: %.1f GB/s\n", all / std::max(1, n));
        }
        cudaStreamDestroy(s2);
    }
    {
        uint8_t* p = (uint8_t*) std::malloc(kLayer);
        fill(p, kLayer);
        double best = 0;
        for (int r = 0; r < 3; ++r) best = std::max(best, h2d_rate(p, kLayer, dev, s, 64));
        std::printf("H2D from pageable memory:     %.1f GB/s (64 experts, best of 3)\n", best);
        std::free(p);
    }
    {
        uint8_t* p = nullptr;
        if (cudaMallocHost(&p, 32 * kExpert) == cudaSuccess) {
            fill(p, 32 * kExpert);
            double best = 0;
            cudaEvent_t a, b;
            cudaEventCreate(&a);
            cudaEventCreate(&b);
            for (int r = 0; r < 3; ++r) {
                cudaEventRecord(a, s);
                for (int i = 0; i < 128; ++i)
                    cudaMemcpyAsync(dev + (size_t) (i % 4) * kExpert, p + (size_t) (i % 32) * kExpert, kExpert,
                                    cudaMemcpyHostToDevice, s);
                cudaEventRecord(b, s);
                cudaEventSynchronize(b);
                float ms = 0;
                cudaEventElapsedTime(&ms, a, b);
                best = std::max(best, 128.0 * kExpert / (ms * 1e-3) / 1e9);
            }
            std::printf("H2D from cudaMallocHost:      %.1f GB/s (one 13.4 MB copy per expert, best of 3)\n", best);
            best = 0;
            for (int r = 0; r < 3; ++r) best = std::max(best, h2d_rate(p, 32 * kExpert, dev, s, 128));
            std::printf("H2D from cudaMallocHost:      %.1f GB/s (3 copies per expert, best of 3)\n", best);
            cudaFreeHost(p);
        }
    }
    if (!buf.empty()) {   // what the CPU experts see: 24 threads reading the first layer
        double best = 0;
        for (int r = 0; r < 3; ++r) {
            std::vector<std::thread> th;
            std::vector<unsigned long long> sums(24, 0);
            const double r0 = now_s();
            for (int t = 0; t < 24; ++t)
                th.emplace_back([&, t] {
                    const unsigned long long* q = (const unsigned long long*) buf[0];
                    const size_t n = kLayer / 8, a = n / 24 * t, b = t == 23 ? n : n / 24 * (t + 1);
                    unsigned long long acc = 0;
                    for (size_t i = a; i < b; ++i) acc += q[i];
                    sums[(size_t) t] = acc;
                });
            for (auto& t : th) t.join();
            best = std::max(best, kLayer / (now_s() - r0) / 1e9);
        }
        std::printf("CPU read of the first layer (24 threads): %.1f GB/s\n", best);
    }
    const double u0 = now_s();
    for (size_t i = 0; i < buf.size(); ++i) {
        if (host_alloc) cudaFreeHost(buf[i]);
        else if (pinned[i]) cudaHostUnregister(buf[i]);
    }
    std::printf("%s all: %.2f s\n", host_alloc ? "cudaFreeHost" : "unregister", now_s() - u0);
    if (!host_alloc)
        for (auto* p : buf) std::free(p);
    cudaFree(dev);
    return 0;
}
