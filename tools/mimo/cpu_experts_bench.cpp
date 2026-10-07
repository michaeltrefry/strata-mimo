// mimo_cpu_experts_bench: how fast does the engine's CPU expert path (CpuExperts on ExpertPool) compute MiMo's MXFP4
// experts that are not on the GPU?
//
// Copies the real experts of the first `--load` MoE layers (24 = 76.5 GiB) from the GGUF into RAM, then simulates
// decode: per window, 47 MoE layers (layer l reads the copy of layer (l - 1) % load + 1); per layer one
// CpuExperts::run_layer call with T tokens x M random experts (F32 hidden state in, weighted F32 sums out, the
// activation quantization included).  T = 2 is a verify window: the second token shares round(M * 2.2 / 8) of the
// first token's experts (MIMO_ROUTING.md: two consecutive tokens pick 13.6-13.9 distinct of their 16).
// Every measurement is repeated (--reps) and reported as best and worst; GB/s counts each distinct expert once.
//
// usage: mimo_cpu_experts_bench <MiMo GGUF shard> [--load 24] [--large-pages] [--reps 3] [--m 1,2,3,4,8] [--t 1,2]
//        [--config SPEC]...  [--sweep]
// SPEC = name,key=value,...  keys: kernel=512|ggml, park=0|1 (the pool waits for its workers to park around each
// phase), tasks=N (tasks per thread and phase), pf=BYTES (software prefetch distance, 0 = off), workers=N (0: one per
// physical core but the first), pin=0|1, host=0|1 (the calling thread drains too).  Each config gets its own pool
// over the same RAM copy.  --sweep runs the sequence measured in docs/MIMO_CPU_EXPERTS.md.
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/mxfp4_avx512.hpp"
#include "strata/mimo/cpu_experts.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <immintrin.h>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <sys/mman.h>
#endif

using namespace strata::mimo;
namespace cpu = strata::kernels::cpu;

namespace {

std::vector<int> parse_list(const char* s) {
    std::vector<int> v;
    std::stringstream ss(s);
    for (std::string t; std::getline(ss, t, ',');) v.push_back(std::atoi(t.c_str()));
    return v;
}

// RAM for one layer's copy: large pages when asked and allowed (Windows: SeLockMemoryPrivilege), else normal pages
uint8_t* alloc_layer(size_t bytes, bool large, std::string& how) {
#if defined(_WIN32)
    if (large) {
        static bool tried = false, have = false;
        if (!tried) {
            tried = true;
            HANDLE tok = nullptr;
            if (OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &tok)) {
                TOKEN_PRIVILEGES tp{};
                tp.PrivilegeCount = 1;
                tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
                if (LookupPrivilegeValueW(nullptr, L"SeLockMemoryPrivilege", &tp.Privileges[0].Luid))
                    have = AdjustTokenPrivileges(tok, FALSE, &tp, 0, nullptr, nullptr) && GetLastError() == ERROR_SUCCESS;
                CloseHandle(tok);
            }
            if (!have) std::printf("large pages: SeLockMemoryPrivilege is not held by this account - normal pages\n");
        }
        const size_t lp = GetLargePageMinimum();
        if (have && lp) {
            const size_t lb = (bytes + lp - 1) / lp * lp;
            if (void* p = VirtualAlloc(nullptr, lb, MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES, PAGE_READWRITE)) {
                how = "large pages (" + std::to_string(lp >> 20) + " MiB)";
                return (uint8_t*) p;
            }
            std::printf("large pages: VirtualAlloc(MEM_LARGE_PAGES) of %zu MiB failed (error %lu) - normal pages\n", lb >> 20,
                        GetLastError());
        }
    }
    how = "VirtualAlloc, 4 KiB pages, not locked";
    return (uint8_t*) VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
    void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | (large ? MAP_HUGETLB : 0), -1, 0);
    if (p == MAP_FAILED && large) {
        how = "mmap, 4 KiB pages (MAP_HUGETLB refused)";
        p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    } else {
        how = large ? "mmap MAP_HUGETLB" : "mmap, 4 KiB pages";
    }
    return p == MAP_FAILED ? nullptr : (uint8_t*) p;
#endif
}

// The ceiling: every physical core streams a disjoint slice of the copy with 256-bit loads and nothing else.
double read_bw(const std::vector<std::pair<const uint8_t*, size_t>>& regions, int threads) {
    const std::vector<int> cores = cpu::physical_cores(false);
    size_t total = 0;
    for (const auto& r : regions) total += r.second;
    std::vector<std::thread> th;
    std::vector<long long> sink((size_t) threads * 8);
    const auto s0 = std::chrono::steady_clock::now();
    for (int t = 0; t < threads; ++t)
        th.emplace_back([&, t] {
            if (t < (int) cores.size()) cpu::pin_current_thread(cores[(size_t) t]);
            __m256i acc = _mm256_setzero_si256();
            for (const auto& r : regions) {
                const size_t per = r.second / (size_t) threads / 64 * 64;
                const uint8_t* p = r.first + per * (size_t) t;
                for (size_t o = 0; o < per; o += 64)
                    acc = _mm256_add_epi64(acc, _mm256_add_epi64(_mm256_loadu_si256((const __m256i*) (p + o)),
                                                                 _mm256_loadu_si256((const __m256i*) (p + o + 32))));
            }
            _mm256_storeu_si256((__m256i*) &sink[(size_t) t * 8], acc);
        });
    for (auto& x : th) x.join();
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - s0).count();
    return (double) (total / 64 * 64) / s / 1e9;
}

struct Config {
    std::string name = "default";
    int kernel512 = 1, park = 1, tasks = 3, pf = 0, workers = 0, pin = 1, host = 1;   // the pool's defaults
};

Config parse_config(const std::string& spec) {
    Config c;
    std::stringstream ss(spec);
    std::string tok;
    std::getline(ss, c.name, ',');
    while (std::getline(ss, tok, ',')) {
        const size_t eq = tok.find('=');
        const std::string k = tok.substr(0, eq), v = eq == std::string::npos ? "1" : tok.substr(eq + 1);
        if (k == "kernel") c.kernel512 = v == "512";
        else if (k == "park") c.park = std::atoi(v.c_str());
        else if (k == "tasks") c.tasks = std::atoi(v.c_str());
        else if (k == "pf") c.pf = std::atoi(v.c_str());
        else if (k == "workers") c.workers = std::atoi(v.c_str());
        else if (k == "pin") c.pin = std::atoi(v.c_str());
        else if (k == "host") c.host = std::atoi(v.c_str());
        else throw std::runtime_error("unknown config key " + k);
    }
    return c;
}

const char* kSweep[] = {
    "before: ggml dot + pool protocol,kernel=ggml,park=1,tasks=3",
    "AVX-512 kernel,kernel=512,park=1,tasks=3",
    "+ no park waits,kernel=512,park=0,tasks=3",
    "+ 2 tasks/thread,kernel=512,park=0,tasks=2",
    "+ 1 task/thread,kernel=512,park=0,tasks=1",
    "+ 4 tasks/thread,kernel=512,park=0,tasks=4",
    "+ prefetch 1 KiB,kernel=512,park=0,tasks=2,pf=1024",
    "+ prefetch 4 KiB,kernel=512,park=0,tasks=2,pf=4096",
    "unpinned workers,kernel=512,park=0,tasks=2,pin=0",
    "47 workers (SMT),kernel=512,park=0,tasks=2,workers=47",
};

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <MiMo GGUF shard> [--load 24] [--large-pages] [--reps 3] [--m 1,2,3,4,8] [--t 1,2] "
                             "[--config name,key=value,...]... [--sweep]\n", argv[0]);
        return 2;
    }
    int n_load = 24, reps = 3;
    bool large = false, readbw = false;
    double share = 2.2 / 8.0;   // of token 0's experts each later token of a window repeats
    std::vector<int> ms = {1, 2, 3, 4, 8}, ts = {1, 2};
    std::vector<Config> configs;
    try {
        for (int i = 2; i < argc; ++i) {
            const std::string a = argv[i];
            auto next = [&]() -> const char* { return i + 1 < argc ? argv[++i] : "0"; };
            if (a == "--load") n_load = std::atoi(next());
            else if (a == "--large-pages") large = true;
            else if (a == "--readbw") readbw = true;
            else if (a == "--share") share = std::atof(next());
            else if (a == "--reps") reps = std::atoi(next());
            else if (a == "--m") ms = parse_list(next());
            else if (a == "--t") ts = parse_list(next());
            else if (a == "--config") configs.push_back(parse_config(next()));
            else if (a == "--sweep") { for (const char* sp : kSweep) configs.push_back(parse_config(sp)); }
            else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
        }
        if (configs.empty()) configs.push_back(Config{});

        MimoModel model(argv[1]);
        const MimoGeometry& g = model.geometry();
        std::vector<int> moe;
        for (int l = 0; l < g.n_layer; ++l)
            if (g.is_moe(l)) moe.push_back(l);
        n_load = (std::min)(n_load, (int) moe.size());
        // the RAM copy, once (copy_layer does not use the pool, so a one-thread pool carries it)
        std::vector<ExpertSlices> ram((size_t) g.n_layer);
        const auto t0 = std::chrono::steady_clock::now();
        std::string how;
        double gib = 0;
        {
            cpu::ExpertPool loader(1, false, false);
            CpuExperts ce(model, loader);
            for (int i = 0; i < n_load; ++i) {
                const size_t bytes = ce.layer_bytes(moe[(size_t) i]);
                uint8_t* p = alloc_layer(bytes, large, how);
                if (!p) { std::fprintf(stderr, "out of memory copying layer %d\n", moe[(size_t) i]); return 1; }
                ce.copy_layer(moe[(size_t) i], p, 16);
                ram[(size_t) moe[(size_t) i]] = ce.source(moe[(size_t) i]);
                gib += bytes / 1073741824.0;
            }
            for (size_t i = (size_t) n_load; i < moe.size(); ++i) ram[(size_t) moe[i]] = ram[(size_t) moe[i % (size_t) n_load]];
        }
        std::printf("copied %d MoE layers' experts (%.1f GiB) into RAM in %.0f s: %s; %s; the %d MoE layers read them in turn\n",
                    n_load, gib, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), how.c_str(),
                    cpu::cpu_name().c_str(), (int) moe.size());

        if (readbw) {
            std::vector<std::pair<const uint8_t*, size_t>> regions;
            for (int i = 0; i < n_load; ++i) {
                const ExpertSlices& s = ram[(size_t) moe[(size_t) i]];
                regions.emplace_back(s.base[0], (size_t) g.n_expert * s.expert_bytes());
            }
            for (int th : {24, 16, 8}) {
                double best = 0, worst = 1e30;
                for (int rep = 0; rep < reps; ++rep) {
                    const double v = read_bw(regions, th);
                    best = (std::max)(best, v);
                    worst = (std::min)(worst, v);
                }
                std::printf("plain read of the copy, %d threads (one per physical core): %.1f GB/s best, %.1f worst\n", th, best, worst);
            }
        }

        std::mt19937 rng(7);
        std::normal_distribution<float> nd(0.f, 1.f);
        const int H = g.n_embd, NE = g.n_expert, NL = (int) moe.size();
        std::vector<float> x((size_t) cpu::MAXT * H), out((size_t) cpu::MAXT * H);
        for (auto& v : x) v = nd(rng);
        for (const Config& c : configs) {
            cpu::native_set_mxfp4_512(c.kernel512);
            cpu::mxfp4_set_prefetch_bytes(c.pf);
            cpu::ExpertPool pool(c.workers, c.pin != 0, c.host != 0);
            pool.moe_tasks_per_thread = c.tasks;
            pool.moe_park_waits = c.park != 0;
            const cpu::CpuTopology topo = cpu::detect_cpu_topology(true);
            const long long prev = c.pin && topo.host_core >= 0 ? cpu::pin_current_thread(topo.host_core) : -1;
            CpuExperts ce(model, pool);
            for (int l : moe) ce.set_source(l, ram[(size_t) l]);
            std::printf("\n### %s\n\npool: %d workers%s + host%s; MXFP4 dot: %s; prefetch %d B; %d tasks/thread; park waits %s\n\n",
                        c.name.c_str(), pool.workers(), c.pin ? " (one per physical core)" : " (unpinned)",
                        c.host ? "" : " (idle)", cpu::native_mxfp4_512() ? "AVX-512 kernel" : "ggml vec_dot",
                        cpu::mxfp4_prefetch_bytes(), pool.moe_tasks_per_thread, pool.moe_park_waits ? "on" : "off");
            std::printf("| experts/layer | T | distinct/layer | ms per window (best) | spread | ms per token | GB/s (best) |\n"
                        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n");
            for (int T : ts)
                for (int M : ms) {
                    const int n_win = (std::max)(6, 240 / (M * T)), warm = 2;
                    const int shared = (int) std::lround(M * share);
                    // every window's experts up front: distinct within a token; token t > 0 repeats `shared` of
                    // token 0's and the rest are new to the window
                    std::vector<int32_t> ids((size_t) (n_win + warm) * NL * T * M);
                    std::vector<float> w(ids.size(), 1.f / M);
                    std::uniform_int_distribution<int> ue(0, NE - 1);
                    for (size_t base = 0; base < ids.size(); base += (size_t) T * M) {
                        int32_t* p = &ids[base];
                        for (int t = 0; t < T; ++t)
                            for (int j = 0; j < M; ++j) {
                                if (t > 0 && j < shared) { p[t * M + j] = p[j]; continue; }
                                int e;
                                bool dup;
                                do {
                                    e = ue(rng);
                                    dup = false;
                                    for (int q = 0; q < t * M + j; ++q) dup |= p[q] == e;
                                } while (dup);
                                p[t * M + j] = e;
                            }
                    }
                    double best = 1e30, worst = 0, best_gbs = 0, distinct = 0;
                    for (int rep = 0; rep < reps; ++rep) {
                        int64_t b0 = 0, e0 = 0;
                        auto s0 = std::chrono::steady_clock::now();
                        for (int win = 0; win < n_win + warm; ++win) {
                            if (win == warm) {
                                s0 = std::chrono::steady_clock::now();
                                b0 = ce.bytes_read();
                                e0 = ce.experts_run();
                            }
                            for (int L = 0; L < NL; ++L) {
                                const size_t base = ((size_t) win * NL + (size_t) L) * (size_t) T * M;
                                ce.run_layer(moe[(size_t) L], T, x.data(), M, &ids[base], &w[base], out.data());
                            }
                        }
                        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - s0).count();
                        const double ms_win = s / n_win * 1e3;
                        distinct = (double) (ce.experts_run() - e0) / ((double) n_win * NL);
                        best = (std::min)(best, ms_win);
                        worst = (std::max)(worst, ms_win);
                        best_gbs = (std::max)(best_gbs, (double) (ce.bytes_read() - b0) / s / 1e9);
                    }
                    std::printf("| %d | %d | %.2f | %.2f | %.2f-%.2f | %.2f | %.1f |\n", M, T, distinct, best, best, worst,
                                best / T, best_gbs);
                    std::fflush(stdout);
                }
            std::printf("\nrun_moe phases over this config: gate/up %.0f ms, down %.0f ms\n", pool.ms_moe_gu, pool.ms_moe_down);
            cpu::restore_thread_affinity(prev);
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
