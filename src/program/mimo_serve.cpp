// strata-mimo: the MiMo-V2.6-Flash engine behind Strata's server (serve/server.py), speaking the same line protocol
// as `strata --serve` on stdin/stdout:
//   in : GEN <max_new>[ key=value ...] <id,id,...>   |   STOP   |   QUIT
//   out: INFO k=v ...  READY <max_context> stop          (once, after loading)
//        RESUME <reused>  PP <pos> <total> <ms> <tok/s> ...  T <id> ...  DONE <14 fields>   |   ERR <message>
// Sampling keys: temperature, top_p, top_k, min_p, penalty_repeat, penalty_freq, penalty_present, penalty_last_n,
// seed; others are ignored.  The KV of the last request is kept: a prompt that starts with it reuses it (backing up
// 128 tokens when it diverges far enough back that the sliding-window rings no longer hold the keys it needs).
// With --slots 2, READY adds multiplex=1 slots=2. GEN has <command-id> <slot-id> before max_new; STOP has a
// command-id, and every request event has that id after its kind. The two VRAM caches share one compute owner;
// generation steps and complete prompt chunks alternate. Context is per slot. Old one-slot clients are unchanged.
//
//   strata-mimo --serve --native <model.gguf> [--mtp <mtp.gguf>] [--expert-profile <MIMOPROF>]
//               [--max-context N] [--eos-ids a,b] [--budget-gib G]          (unknown flags are ignored)
#include "strata/mimo/decoder.hpp"
#include "strata/mimo/serve.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <iostream>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace strata::mimo;
using clk = std::chrono::steady_clock;

namespace {

// stdin on its own thread: STOP must reach a running request, everything else is queued
struct Input {
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::string> lines;
    bool eof = false;
    bool multiplex = false;
    std::atomic<bool> busy{false}, stop{false};

    void run() {
        std::string line;
        while (std::getline(std::cin, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (!multiplex && line == "STOP") {
                if (busy) stop = true;   // a STOP between requests is stale
                continue;
            }
            std::unique_lock<std::mutex> lk(mu);
            if (multiplex) cv.wait(lk, [&] { return lines.size() < 64; });
            lines.push_back(line);
            cv.notify_one();
        }
        std::lock_guard<std::mutex> lk(mu);
        eof = true;
        cv.notify_one();
    }
    bool next(std::string& out) {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&] { return !lines.empty() || eof; });
        if (lines.empty()) return false;
        out = std::move(lines.front());
        lines.pop_front();
        cv.notify_one();
        return true;
    }
    bool poll(std::string& out) {
        std::lock_guard<std::mutex> lk(mu);
        if (lines.empty()) return false;
        out = std::move(lines.front());
        lines.pop_front();
        cv.notify_one();
        return true;
    }
    bool closed() {
        std::lock_guard<std::mutex> lk(mu);
        return eof && lines.empty();
    }
};

void say(const std::string& s) {
    std::fwrite(s.data(), 1, s.size(), stdout);
    std::fputc('\n', stdout);
    std::fflush(stdout);
}

bool parse_gen(const std::string& line, int& max_new, SamplingParams& sp, std::vector<int32_t>& ids, std::string& err) {
    return serve_parse_gen(line, max_new, sp, ids, err);
}

}  // namespace

int main(int argc, char** argv) {
    DecoderConfig cfg;
    cfg.max_ctx = 262144;
    std::string model_path;
    std::set<int32_t> eos = {151645, 151643};   // <|im_end|>, <|endoftext|>
    bool serve = false;
    int prefill_quantum = 2048;
    std::string save_path;          // the learned profile (Strata's #477 protocol): on QUIT and every N minutes
    double save_every_min = 10;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&](std::string& out) {
            if (i + 1 < argc) out = argv[++i];
        };
        std::string v;
        if (a == "--serve") serve = true;
        else if (a == "--native" || a == "--model") val(model_path);
        else if (a == "--mtp") val(cfg.mtp_path);
        else if (a == "--expert-profile") val(cfg.profile);
        else if (a == "--max-context") { val(v); cfg.max_ctx = std::atoi(v.c_str()); }
        else if (a == "--slots" || a == "--prefill-quantum") {
            val(v);
            uint64_t count = 0;
            if (!serve_uint(v, count) || (a == "--slots" ? count < 1 || count > 2 : count < 1 || count > 16384)) {
                std::fprintf(stderr, "strata-mimo: %s needs %s\n", a.c_str(), a == "--slots" ? "1 or 2" : "1..16384 tokens");
                return 2;
            }
            if (a == "--slots") cfg.slots = (int) count;
            else prefill_quantum = (int) count;
        }
        else if (a == "--budget-gib") { val(v); cfg.expert_budget_gib = std::atof(v.c_str()); }
        else if (a == "--ram-copy") {
            val(v);
            if (v != "0" && v != "1") {
                std::fprintf(stderr, "strata-mimo: --ram-copy needs 0 or 1\n");
                return 2;
            }
            cfg.ram_copy = v == "1";
        }
        else if (a == "--batched-min") { val(v); cfg.batched_min_prompt = std::atoi(v.c_str()); }
        else if (a == "--expert-profile-save") val(save_path);
        else if (a == "--expert-profile-save-every") { val(v); save_every_min = std::atof(v.c_str()); }
        else if (a == "--eos-ids") {
            val(v);
            eos.clear();
            std::stringstream ls(v);
            std::string id;
            while (std::getline(ls, id, ',')) eos.insert(std::atoi(id.c_str()));
        } else if (a.rfind("--", 0) == 0 && i + 1 < argc && std::strncmp(argv[i + 1], "--", 2) != 0) {
            ++i;   // an unknown flag with a value (Strata's setup may pass flags this engine does not use)
        }
    }
    if (!serve || model_path.empty()) {
        std::fprintf(stderr, "usage: %s --serve --native <model.gguf> [--mtp <mtp.gguf>] [--expert-profile <file>] "
                             "[--max-context N] [--eos-ids a,b] [--slots 1|2] [--prefill-quantum N] [--ram-copy 0|1]\n", argv[0]);
        return 2;
    }
    try {
        const auto t_load = clk::now();
        MimoModel model(model_path);
        MimoDecoder dec(model, cfg);
        std::string err;
        if (!dec.init(err)) {
            std::fprintf(stderr, "strata-mimo: %s\n", err.c_str());
            return 1;
        }
        std::fprintf(stderr, "strata-mimo: ready in %.1f s, %d resident experts, MTP %s\n",
                     std::chrono::duration<double>(clk::now() - t_load).count(), dec.resident_experts(),
                     dec.has_mtp() ? "on" : "off");
        say("INFO engine=strata-mimo resident_experts=" + std::to_string(dec.resident_experts()) +
            " vram_free_mib=" + std::to_string(dec.vram_free_after_init() >> 20) + " mtp=" + (dec.has_mtp() ? "1" : "0") +
            " ram_copy=" + (cfg.ram_copy ? "1" : "0"));
        if (cfg.slots == 2)
            say("INFO slots=2 context_per_slot=" + std::to_string(cfg.max_ctx) +
                " kv_bytes=" + std::to_string(dec.kv_bytes()) + " prefill_quantum=" + std::to_string(prefill_quantum));
        say("READY " + std::to_string(cfg.max_ctx) + " stop" + (cfg.slots == 2 ? " multiplex=1 slots=2" : ""));

        auto input_owner = std::make_shared<Input>();
        Input& input = *input_owner;
        input.multiplex = cfg.slots == 2;
        std::thread reader([input_owner] { input_owner->run(); });
        reader.detach();   // blocked in getline at exit; the process ends with it

        const int V = dec.vocab();
        std::vector<int32_t> session;   // tokens whose KV is valid, in position order
        Sampler sampler;
        std::string line;
        auto last_save = clk::now();
        auto save_learned = [&](bool force) {
            if (save_path.empty()) return;
            const double mins = std::chrono::duration<double>(clk::now() - last_save).count() / 60.0;
            if (!force && (save_every_min <= 0 || mins < save_every_min)) return;
            std::string e;
            if (!dec.save_profile(save_path, e)) std::fprintf(stderr, "strata-mimo: %s\n", e.c_str());
            last_save = clk::now();
        };
        if (cfg.slots == 2) {
            ServeOptions options;
            options.max_context = cfg.max_ctx;
            options.prefill_quantum = prefill_quantum;
            options.ring_capacity = kSwaRing;
            options.eos = eos;
            MimoServeScheduler<MimoDecoder> scheduler(dec, options, say);
            try {
                while (true) {
                    if (!scheduler.active()) {
                        if (scheduler.quitting() || !input.next(line)) break;
                        scheduler.command(line);
                    }
                    // Bounded input draining prevents a stream of commands from starving admitted compute.
                    for (int i = 0; i < 64 && input.poll(line); ++i) scheduler.command(line);
                    if (input.closed()) scheduler.shutdown();
                    if (!scheduler.step(err)) throw std::runtime_error(err);
                    save_learned(false);
                }
            } catch (const std::exception& e) {
                scheduler.fail_all(e.what());
                throw;
            }
            save_learned(true);
            return 0;
        }
        while (input.next(line)) {
            if (line == "QUIT") break;
            save_learned(false);
            if (line.rfind("GEN ", 0) != 0) {
                if (!line.empty()) say("ERR unknown command");
                continue;
            }
            int max_new = 0;
            SamplingParams sp;
            std::vector<int32_t> ids;
            if (!parse_gen(line, max_new, sp, ids, err)) { say("ERR " + err); continue; }
            const int n = (int) ids.size();
            bool bad_id = false;
            for (int32_t id : ids) bad_id |= id < 0 || id >= V;
            if (bad_id) { say("ERR a token id is outside the vocabulary"); continue; }
            if ((long long) n + max_new + 8 > cfg.max_ctx) {
                say("ERR prompt (" + std::to_string(n) + " tokens) + max_new (" + std::to_string(max_new) +
                    ") exceeds the context (" + std::to_string(cfg.max_ctx) + ")");
                continue;
            }
            input.stop = false;
            input.busy = true;
            // reuse: the longest common prefix with the kept KV, at most n - 1 (the last prompt token always runs)
            int keep = 0;
            while (keep < (int) session.size() && keep < n - 1 && session[(size_t) keep] == ids[(size_t) keep]) ++keep;
            // the sliding-window layers keep a ring of the last kSwaRing positions of the session; the positions run
            // from `keep` on need the n_swa - 1 keys before them, which the ring still holds unless keep lies more
            // than kSwaRing - n_swa + 1 positions back - then back up n_swa positions and rebuild them
            const bool ring_ok = keep >= (int) session.size() - (kSwaRing - 127);
            const int start = ring_ok ? keep : std::max(0, keep - 128);
            say("RESUME " + std::to_string(start));
            sampler.begin(sp);
            sampler.history(ids.data(), n);
            const bool host_choice = sp.needs_logits();
            auto choose = [&](const float* logits, int32_t greedy) { return host_choice ? sampler.sample(logits, V) : greedy; };
            const auto t0 = clk::now();
            int32_t tok = 0, draft = 0;
            bool cancelled = false;
            auto progress = [&](int done) {
                const double ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
                if (done == n - start || done % 256 < 8)
                    say("PP " + std::to_string(start + done) + " " + std::to_string(n) + " " + std::to_string(ms) + " " +
                        std::to_string(ms > 0 ? done / ms * 1000.0 : 0.0));
            };
            session.assign(ids.begin(), ids.begin() + start);
            if (!dec.prompt(ids.data() + start, n - start, start, choose, &tok, dec.has_mtp() ? &draft : nullptr, progress, err)) {
                session.clear();
                input.busy = false;
                say("ERR " + err);
                continue;
            }
            session.assign(ids.begin(), ids.end());
            const double prompt_ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
            const auto t1 = clk::now();
            int generated = 0, pos = n;
            long long offered = 0, accepted = 0;
            std::string finish = "length";
            std::vector<float> logits;
            int32_t next[8];
            auto emit = [&](int32_t id) {
                say("T " + std::to_string(id));
                ++generated;
                if (eos.count(id)) finish = "stop";
            };
            while (generated < max_new && finish == "length") {
                if (input.stop) { finish = "cancel"; cancelled = true; break; }
                if (dec.has_mtp() && generated + 1 < max_new) {
                    // verify [tok, draft]; the main model's distribution after tok decides whether draft stands
                    int32_t win[2] = {tok, draft};
                    if (!dec.forward(win, 2, pos, next, host_choice ? &logits : nullptr, err)) break;
                    ++offered;
                    int32_t other = next[0];
                    session.push_back(tok);   // the position the draft is judged at sees tok
                    bool keep_draft;
                    if (host_choice) {
                        sampler.history(session.data(), (int) session.size());
                        keep_draft = sampler.accept_draft(logits.data(), V, draft, &other);
                    } else {
                        keep_draft = next[0] == draft;
                    }
                    int32_t mt[2];
                    int A;
                    emit(tok);
                    if (keep_draft && finish == "length") {
                        ++accepted;
                        emit(draft);
                        session.push_back(draft);
                        if (host_choice) sampler.history(session.data(), (int) session.size());
                        tok = host_choice ? sampler.sample(logits.data() + V, V) : next[1];
                        mt[0] = draft;
                        mt[1] = tok;
                        A = 2;
                    } else {
                        tok = other;
                        mt[0] = tok;
                        A = 1;
                    }
                    if (!dec.mtp_forward(mt, A, pos + 1, next, err)) break;
                    draft = next[A - 1];
                    pos += A;
                } else {
                    if (!dec.forward(&tok, 1, pos, next, host_choice ? &logits : nullptr, err)) break;
                    emit(tok);
                    session.push_back(tok);
                    if (host_choice) sampler.history(session.data(), (int) session.size());
                    tok = host_choice ? sampler.sample(logits.data(), V) : next[0];
                    ++pos;
                }
            }
            input.busy = false;
            if (!err.empty() && finish == "length" && generated < max_new) {
                session.clear();
                say("ERR " + err);
                err.clear();
                continue;
            }
            (void) cancelled;
            const double decode_ms = std::chrono::duration<double, std::milli>(clk::now() - t1).count();
            const DecoderStats& st = dec.stats();
            char done[512];
            std::snprintf(done, sizeof done, "DONE %d %d %.1f %.1f %s %lld %lld %d %llu %llu 0 0 0.0 %d", generated, n,
                          prompt_ms, decode_ms, finish.c_str(), accepted, offered, start,
                          (unsigned long long) st.expert_hits, (unsigned long long) st.expert_uses, n - start);
            say(done);
        }
        save_learned(true);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata-mimo: %s\n", e.what());
        return 1;
    }
}
