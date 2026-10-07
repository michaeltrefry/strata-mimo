// mimo_prefill_moe_bench: how long do the routed experts of a prompt chunk take, per layer and for the whole model,
// with a realistic residency and routing - and is it the PCIe copies or the MMQ products?
//
// Residency as the decoder chooses it: the `--resident` share of all MoE experts (default 0.55, ~6,600 of 12,032),
// the highest scores of a MIMOPROF profile first.  Routing: per token 8 distinct experts drawn with probability
// proportional to the profile's score of each expert in that layer (`--routing uniform`: equal), weights random and
// normalized; x ~ N(0, 1).  Tokens are drawn independently (real text repeats experts between neighbouring tokens,
// which only matters below ~512 tokens, where not every expert is used).
//
// Per MoE layer: its experts copied into one pinned RAM buffer (`--ram hostalloc`: cudaHostAlloc, the default;
// `register`: malloc + cudaHostRegister, as the decoder's copy would be pinned after the fact), its resident experts
// put into slots (the decoder's format), then for each P: PrefillMoe::run_layer in three modes, best of `--reps`:
//   full          what the prompt driver gets
//   compute_only  the copies skipped (the conversions and products of every expert: the GPU's time)
//   copy_only     the products skipped (the copies of the non-resident experts: the PCIe time)
// overlap = (copy_only + compute_only - full) / min(copy_only, compute_only): 1 = the shorter one is fully hidden.
// The times exclude attention and the router; tok/s is P / (sum of the 47 layers' full time).
//
// --startup hostalloc|register: only the time to allocate, fill and pin the decoder's RAM copy of all 47 layers.
//
// usage: mimo_prefill_moe_bench <MiMo GGUF shard> --profile <MIMOPROF> [--p 4096,8192,16384,32768] [--resident 0.55]
//        [--routing profile|uniform] [--layers N] [--reps 2] [--ram hostalloc|register] [--group 16]
//        [--group-ram 8] [--staging 16] [--tile-rows 4096] [--vram-mib N] [--verbose]
//        mimo_prefill_moe_bench <MiMo GGUF shard> --startup hostalloc|register
#include "strata/mimo/expert_mxfp4.hpp"
#include "strata/mimo/model.hpp"
#include "strata/mimo/prefill_moe.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>\n#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <functional>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace strata::mimo;

#define CK(x)                                                                                       \
    do {                                                                                            \
        cudaError_t e_ = (x);                                                                       \
        if (e_ != cudaSuccess) {                                                                    \
            std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
            std::exit(1);                                                                           \
        }                                                                                           \
    } while (0)

namespace {

constexpr int E = kMxEmbd, NE = 256, K = 8;

std::vector<int> parse_list(const char* s) {
    std::vector<int> v;
    std::stringstream ss(s);
    for (std::string t; std::getline(ss, t, ',');) v.push_back(std::atoi(t.c_str()));
    return v;
}

bool read_profile(const std::string& path, int n_layer, std::vector<float>& score) {
    std::ifstream f(path, std::ios::binary);
    char magic[8];
    uint32_t hdr[3];
    uint64_t tokens = 0;
    if (!f.read(magic, 8) || std::memcmp(magic, "MIMOPROF", 8) != 0 || !f.read((char*) hdr, 12) ||
        !f.read((char*) &tokens, 8) || hdr[0] != 1 || (int) hdr[1] != n_layer || hdr[2] != NE)
        return false;
    score.resize((size_t) n_layer * NE);
    return (bool) f.read((char*) score.data(), (std::streamsize) (score.size() * 4));
}

void parallel(int n, int threads, const auto& fn) {
    std::atomic<int> next{0};
    std::vector<std::thread> th;
    for (int t = 0; t < threads; ++t)
        th.emplace_back([&] {
            for (int i; (i = next.fetch_add(1)) < n;) fn(i);
        });
    for (auto& t : th) t.join();
}

// The decoder's RAM copy of all MoE layers, pinned two ways: cudaHostAlloc per layer then filled, or malloc, filled,
// cudaHostRegister.  The fill is CpuExperts::copy_layer's: 64 MiB pieces from the GGUF mapping on 16 threads.
int startup_time(const MimoModel& model, bool reg) {
    const MimoGeometry& g = model.geometry();
    auto now = [] { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); };
    double t_alloc = 0, t_fill = 0, t_reg = 0;
    std::vector<uint8_t*> bufs;
    size_t total = 0;
    for (int l = 0; l < g.n_layer; ++l) {
        if (!g.is_moe(l)) continue;
        const ExpertSlices& xs = model.experts(l);
        const size_t bytes = (size_t) NE * xs.expert_bytes();
        double t0 = now();
        uint8_t* p = nullptr;
        if (reg) p = (uint8_t*) std::malloc(bytes);
        else if (cudaHostAlloc((void**) &p, bytes, cudaHostAllocPortable) != cudaSuccess) p = nullptr;
        if (!p) { std::fprintf(stderr, "allocation failed at layer %d\n", l); return 1; }
        t_alloc += now() - t0;
        t0 = now();
        uint8_t* d = p;
        for (int m = 0; m < 3; ++m) {
            const size_t mb = (size_t) NE * xs.stride[m];
            constexpr size_t kChunk = 64u << 20;
            parallel((int) ((mb + kChunk - 1) / kChunk), 16, [&](int i) {
                const size_t o = (size_t) i * kChunk;
                std::memcpy(d + o, xs.base[m] + o, std::min(kChunk, mb - o));
            });
            d += mb;
        }
        t_fill += now() - t0;
        if (reg) {
            t0 = now();
            std::string err;
            if (!PrefillMoe::register_host(p, bytes, err)) { std::fprintf(stderr, "layer %d: %s\n", l, err.c_str()); return 1; }
            t_reg += now() - t0;
        }
        bufs.push_back(p);
        total += bytes;
    }
    std::printf("%s: %.1f GiB in %zu layers: allocate %.1f s, fill (copy_layer, 16 threads) %.1f s, register %.1f s; "
                "total %.1f s\n", reg ? "malloc + cudaHostRegister" : "cudaHostAlloc", total / 1073741824.0, bufs.size(),
                t_alloc, t_fill, t_reg, t_alloc + t_fill + t_reg);
    const double t0 = now();
    for (uint8_t* p : bufs) {
        if (reg) { PrefillMoe::unregister_host(p); std::free(p); }
        else cudaFreeHost(p);
    }
    std::printf("release: %.1f s\n", now() - t0);
    return 0;
}

struct Acc {   // sums over layers for one P
    double full = 0, compute = 0, copy = 0, copy_span = 0, routing = 0, tail = 0;
    double bytes = 0;
    long long used = 0, copied = 0;
    int layers = 0;
};

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <MiMo GGUF shard> --profile <MIMOPROF> [--p 512,2048,4096] [--resident 0.55] "
                             "[--routing profile|uniform] [--layers N] [--reps 2] [--ram hostalloc|register] "
                             "[--group 16] [--group-ram 8] [--staging 16] [--tile-rows 4096] [--vram-mib N] [--verbose] | "
                             "--startup hostalloc|register\n", argv[0]);
        return 2;
    }
    std::string profile, routing = "profile", ram_mode = "hostalloc", startup;
    std::vector<int> Ps = {4096, 8192, 16384, 32768};
    double resident_share = 0.55;
    int max_layers = 1000, reps = 2;
    bool verbose = false;
    PrefillMoeConfig cfg;
    for (int i = 2; i < argc; ++i) {
        auto arg = [&](const char* n) { return !std::strcmp(argv[i], n) && i + 1 < argc; };
        if (arg("--profile")) profile = argv[++i];
        else if (arg("--p")) Ps = parse_list(argv[++i]);
        else if (arg("--resident")) resident_share = std::atof(argv[++i]);
        else if (arg("--routing")) routing = argv[++i];
        else if (arg("--layers")) max_layers = std::atoi(argv[++i]);
        else if (arg("--reps")) reps = std::atoi(argv[++i]);
        else if (arg("--ram")) ram_mode = argv[++i];
        else if (arg("--group")) cfg.group = std::atoi(argv[++i]);
        else if (arg("--group-ram")) cfg.group_ram = std::atoi(argv[++i]);
        else if (arg("--staging")) cfg.staging = std::atoi(argv[++i]);
        else if (arg("--tile-rows")) cfg.tile_rows = std::atoi(argv[++i]);
        else if (arg("--vram-mib")) cfg.vram_mib = std::atoi(argv[++i]);
        else if (arg("--startup")) startup = argv[++i];
        else if (!std::strcmp(argv[i], "--verbose")) verbose = true;
    }
    CK(cudaSetDevice(0));
    try {
        MimoModel model(argv[1]);
        const MimoGeometry& g = model.geometry();
        if (!startup.empty()) return startup_time(model, startup == "register");
        std::vector<float> score;
        if (profile.empty() || !read_profile(profile, g.n_layer, score)) {
            std::fprintf(stderr, "need a MIMOPROF v1 profile for %d layers x %d experts (--profile)\n", g.n_layer, NE);
            return 1;
        }
        // residency: the profile's best (layer, expert) pairs over all MoE layers
        const int n_moe = g.count_moe();
        const int want = (int) std::lround(resident_share * n_moe * NE);
        std::vector<int> order(score.size());
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return score[(size_t) a] > score[(size_t) b]; });
        std::vector<std::vector<int32_t>> resident((size_t) g.n_layer, std::vector<int32_t>(NE, -1));
        std::vector<int> n_res((size_t) g.n_layer, 0);
        int picked = 0;
        for (int i : order) {
            if (picked >= want) break;
            const int l = i / NE, e = i % NE;
            if (!g.is_moe(l)) continue;
            resident[(size_t) l][(size_t) e] = n_res[(size_t) l]++;   // slot index within the layer's slot pool
            ++picked;
        }
        int min_res = NE, max_res = 0;
        for (int l = 0; l < g.n_layer; ++l)
            if (g.is_moe(l)) {
                min_res = std::min(min_res, n_res[(size_t) l]);
                max_res = std::max(max_res, n_res[(size_t) l]);
            }
        std::printf("residency: %d of %d MoE experts (%.1f%%), per layer %d..%d; routing: %s; RAM: %s\n", picked,
                    n_moe * NE, 100.0 * picked / (n_moe * NE), min_res, max_res, routing.c_str(), ram_mode.c_str());

        // device: slots for one layer's residents, the prompt path's buffers
        cfg.max_tokens = *std::max_element(Ps.begin(), Ps.end());
        size_t free0 = 0, total0 = 0;
        CK(cudaMemGetInfo(&free0, &total0));
        uint8_t* slots = nullptr;
        uint8_t* raw = nullptr;
        CK(cudaMalloc(&slots, (size_t) std::max(1, max_res) * kMxSlotBytes));
        CK(cudaMalloc(&raw, kMxSlotBytes));
        std::vector<unsigned long long> slot_addr((size_t) std::max(1, max_res));
        for (int i = 0; i < max_res; ++i) slot_addr[(size_t) i] = (unsigned long long) (slots + (size_t) i * kMxSlotBytes);
        PrefillMoe moe(cfg);
        std::string err;
        if (!moe.init(err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
        size_t free1 = 0;
        CK(cudaMemGetInfo(&free1, &total0));
        std::printf("VRAM: %.2f GiB slots + %.2f GiB prompt buffers (S <= %d, tiles of %d rows, group %d / %d, staging "
                    "%d); %.1f GiB free before, %.1f after\n", (double) max_res * kMxSlotBytes / 1073741824.0,
                    moe.device_bytes() / 1073741824.0, cfg.max_tokens, moe.config().tile_rows, cfg.group, cfg.group_ram,
                    cfg.staging, free0 / 1073741824.0, free1 / 1073741824.0);

        // RAM: one layer at a time
        int first_moe = 0;
        while (!g.is_moe(first_moe)) ++first_moe;
        const ExpertSlices& x0 = model.experts(first_moe);
        const size_t layer_bytes = (size_t) NE * x0.expert_bytes();
        uint8_t* ram = nullptr;
        if (ram_mode == "register") {
            ram = (uint8_t*) std::malloc(layer_bytes);
            if (!ram) { std::fprintf(stderr, "out of RAM\n"); return 1; }
            std::memset(ram, 0, layer_bytes);
            if (!PrefillMoe::register_host(ram, layer_bytes, err)) {
                std::fprintf(stderr, "RAM: %s\n", err.c_str());
                return 1;
            }
        } else {
            CK(cudaHostAlloc((void**) &ram, layer_bytes, cudaHostAllocPortable));
        }

        cudaStream_t s;
        CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
        const int Pmax = cfg.max_tokens;
        float *dx = nullptr, *dw = nullptr, *dout = nullptr;
        int32_t* dids = nullptr;
        CK(cudaMalloc(&dx, sizeof(float) * Pmax * E));
        CK(cudaMalloc(&dout, sizeof(float) * Pmax * E));
        CK(cudaMalloc(&dw, sizeof(float) * Pmax * K));
        CK(cudaMalloc(&dids, sizeof(int32_t) * Pmax * K));
        std::mt19937 rng(7);
        {
            std::vector<float> x((size_t) Pmax * E);
            std::normal_distribution<float> nd(0.0f, 1.0f);
            for (auto& v : x) v = nd(rng);
            CK(cudaMemcpy(dx, x.data(), sizeof(float) * x.size(), cudaMemcpyHostToDevice));
        }

        std::vector<Acc> acc(Ps.size());
        long long blocks_out_of_range = 0;
        int done = 0, warm = 0;
        const PrefillMoe::Mode modes[3] = {PrefillMoe::Mode::full, PrefillMoe::Mode::compute_only,
                                           PrefillMoe::Mode::copy_only};
        if (verbose) std::printf("\nlayer      S  used copied  full ms  compute ms  copy ms  span ms  GB/s\n");
        for (int l = 0; l < g.n_layer && done < max_layers; ++l) {
            if (!g.is_moe(l)) continue;
            ++done;
            // the layer's experts into the pinned RAM buffer (gate | up | down tensors), exponent check on the way
            const ExpertSlices& xs = model.experts(l);
            ExpertSlices rs = xs;
            uint8_t* p = ram;
            std::atomic<long long> oor{0};
            for (int m = 0; m < 3; ++m) {
                const size_t bytes = (size_t) NE * xs.stride[m];
                constexpr size_t kChunk = 64u << 20;
                const int pieces = (int) ((bytes + kChunk - 1) / kChunk);
                parallel(pieces, 16, [&](int i) {
                    const size_t o = (size_t) i * kChunk, n = std::min(kChunk, bytes - o);
                    std::memcpy(p + o, xs.base[m] + o, n);
                    long long bad = 0;
                    for (size_t b = (o + 16) / 17 * 17; b < o + n; b += 17) bad += p[b] < 104 || p[b] > 143;
                    oor += bad;
                });
                rs.base[m] = p;
                p += bytes;
            }
            blocks_out_of_range += oor;
            // its resident experts into slots
            for (int e = 0; e < NE; ++e) {
                const int sl = resident[(size_t) l][(size_t) e];
                if (sl < 0) continue;
                CK(cudaMemcpy(raw, rs.matrix(0, e), xs.stride[0], cudaMemcpyHostToDevice));
                CK(cudaMemcpy(raw + xs.stride[0], rs.matrix(1, e), xs.stride[1], cudaMemcpyHostToDevice));
                CK(cudaMemcpy(raw + xs.stride[0] + xs.stride[1], rs.matrix(2, e), xs.stride[2], cudaMemcpyHostToDevice));
                mxfp4_slot_relayout(raw, slots + (size_t) sl * kMxSlotBytes, nullptr);
            }
            CK(cudaDeviceSynchronize());
            // routing weights for this layer
            std::vector<double> pw(NE);
            for (int e = 0; e < NE; ++e)
                pw[(size_t) e] = routing == "uniform" ? 1.0 : std::max(1e-6, (double) score[(size_t) l * NE + e]);
            for (size_t pi = 0; pi < Ps.size(); ++pi) {
                const int P = Ps[pi];
                std::vector<int32_t> ids((size_t) P * K);
                std::vector<float> w((size_t) P * K);
                std::uniform_real_distribution<double> ud(0.0, 1.0);
                std::vector<std::pair<double, int>> key(NE);
                for (int t = 0; t < P; ++t) {
                    // Efraimidis-Spirakis: the 8 largest log(u) / weight are a weighted sample without replacement
                    for (int e = 0; e < NE; ++e) key[(size_t) e] = {std::log(ud(rng) + 1e-300) / pw[(size_t) e], e};
                    std::partial_sort(key.begin(), key.begin() + K, key.end(), std::greater<>());
                    float sum = 0;
                    for (int k = 0; k < K; ++k) {
                        ids[(size_t) t * K + k] = key[(size_t) k].second;
                        sum += (w[(size_t) t * K + k] = (float) (0.05 + ud(rng)));
                    }
                    for (int k = 0; k < K; ++k) w[(size_t) t * K + k] /= sum;
                }
                CK(cudaMemcpy(dids, ids.data(), sizeof(int32_t) * ids.size(), cudaMemcpyHostToDevice));
                CK(cudaMemcpy(dw, w.data(), sizeof(float) * w.size(), cudaMemcpyHostToDevice));
                PrefillMoeLayer L;
                L.n_tok = P;
                L.x = dx;
                L.ids = dids;
                L.w = dw;
                L.out = dout;
                L.resident = resident[(size_t) l].data();
                L.slot_addr = slot_addr.data();
                L.ram = rs;
                if (!warm) {   // first launches (MMQ's pools, clocks)
                    for (int r = 0; r < 2; ++r) moe.run_layer(L, s);
                    moe.times();
                    warm = 1;
                }
                PrefillMoeTimes best[3];
                for (int m = 0; m < 3; ++m) {
                    for (int r = 0; r < reps; ++r) {
                        moe.run_layer(L, s, modes[m]);
                        const PrefillMoeTimes t = moe.times();
                        if (r == 0 || t.total < best[m].total) best[m] = t;
                    }
                }
                Acc& a = acc[pi];
                a.full += best[0].total;
                a.compute += best[1].total;
                a.copy += best[2].total;
                a.copy_span += best[0].copy;
                a.routing += best[0].routing;
                a.tail += best[0].tail;
                a.bytes += (double) best[0].copied_bytes;
                a.used += best[0].used;
                a.copied += best[0].copied;
                ++a.layers;
                if (verbose)
                    std::printf("%5d %6d %5d %6d %8.2f %11.2f %8.2f %8.2f %5.1f\n", l, P, best[0].used, best[0].copied,
                                best[0].total, best[1].total, best[2].total, best[0].copy,
                                best[0].copy > 0 ? best[0].copied_bytes / (best[0].copy * 1e-3) / 1e9 : 0.0);
            }
            if (!verbose) { std::printf("."); std::fflush(stdout); }
        }
        std::printf("\n\n%d MoE layers; MXFP4 blocks with an exponent outside 104..143 (Q8_0 not exact): %lld\n\n", done,
                    blocks_out_of_range);
        std::printf("    S | used/layer copied/layer  GB copied | full ms  compute ms  copy ms | routing ms  tail ms | "
                    "H2D GB/s (full / copy only) | overlap | MoE tok/s\n");
        for (size_t pi = 0; pi < Ps.size(); ++pi) {
            const Acc& a = acc[pi];
            const double ov = (a.copy + a.compute - a.full) / std::max(1e-9, std::min(a.copy, a.compute));
            std::printf("%5d | %10.1f %12.1f %10.2f | %7.1f %11.1f %8.1f | %10.1f %8.1f | %10.1f / %-14.1f | %7.2f | "
                        "%9.0f\n", Ps[pi], (double) a.used / a.layers, (double) a.copied / a.layers, a.bytes / 1e9,
                        a.full, a.compute, a.copy, a.routing, a.tail,
                        a.copy_span > 0 ? a.bytes / (a.copy_span * 1e-3) / 1e9 : 0.0,
                        a.copy > 0 ? a.bytes / (a.copy * 1e-3) / 1e9 : 0.0, ov, Ps[pi] / (a.full * 1e-3));
        }
        cudaFree(dx);
        cudaFree(dw);
        cudaFree(dout);
        cudaFree(dids);
        cudaFree(raw);
        cudaFree(slots);
        cudaStreamDestroy(s);
        if (ram_mode == "register") {
            PrefillMoe::unregister_host(ram);
            std::free(ram);
        } else {
            cudaFreeHost(ram);
        }
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
    return 0;
}
