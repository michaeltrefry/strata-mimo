// GPU-free tests of the production two-slot scheduler: tagged ownership, fair prompt/decode turns, MTP boundaries,
// cancellation, prefix reuse and request-local sampling/counters. The fake asserts that main/MTP remain atomic.
#include "strata/mimo/serve.hpp"

#include <array>
#include <iostream>
#include <map>
#include <stdexcept>
#include <tuple>

using namespace strata::mimo;

namespace {
void require(bool value, const char* what) {
    if (!value) throw std::runtime_error(what);
}

struct FakeDecoder {
    struct PromptState {};
    struct Stats { uint64_t expert_hits = 0, expert_uses = 0; } counters;
    struct PrefixCheckpointInfo {
        uint64_t id = 0;
        int committed_end = 0, replay_pos = 0, main_written_end = 0, mtp_written_end = 0, mtp2_written_end = 0;
        size_t host_bytes = 0, metadata_bytes = 0;
    };
    struct PrefixCheckpointStats {
        size_t count = 0, payload_bytes = 0, metadata_bytes = 0;
        uint64_t captures = 0, restores = 0, skipped = 0, invalidations = 0;
        double capture_ms = 0, restore_ms = 0;
        std::string last_skip_reason;
    };
    struct Cell { int position = -1; int32_t token = 0; };
    using Ring = std::array<Cell, 256>;
    struct Checkpoint { PrefixCheckpointInfo info; std::vector<int32_t> prefix; Ring main, mtp; bool milestone; };
    struct PromptCall { int slot, pos, count, boundary; bool final; };
    std::array<std::vector<int32_t>, 2> history;
    std::array<int, 2> resets{}, main_end{}, mtp_end{}, capture_end{{-1, -1}};
    std::array<Ring, 2> main_ring, mtp_ring;
    std::array<std::vector<Checkpoint>, 2> checkpoints;
    std::array<PrefixCheckpointStats, 2> checkpoint_counters;
    std::vector<PromptCall> prompts;
    std::function<void(const char*, int, int)> work;
    bool mtp = true, reject = false, fail = false, checkpoints_enabled = true;
    int awaiting_mtp = -1;
    uint64_t next_checkpoint = 1;
    int vocab() const { return 32; }
    bool has_mtp() const { return mtp; }
    const Stats& stats() const { return counters; }
    void charge(const char* kind, int slot, int rows = 0) { if (work) work(kind, slot, rows); }
    bool reset_sequence(int slot, std::string&) {
        require(awaiting_mtp == -1, "reset between main and MTP");
        ++resets[(size_t) slot];
        history[(size_t) slot].clear();
        checkpoints[(size_t) slot].clear();
        main_ring[(size_t) slot] = {};
        mtp_ring[(size_t) slot] = {};
        main_end[(size_t) slot] = mtp_end[(size_t) slot] = 0;
        capture_end[(size_t) slot] = -1;
        charge("reset", slot);
        return true;
    }
    void invalidate(int slot, int pos) {
        auto& entries = checkpoints[(size_t) slot];
        const size_t count = entries.size();
        entries.erase(std::remove_if(entries.begin(), entries.end(),
            [&](const auto& entry) { return entry.info.replay_pos > pos; }), entries.end());
        checkpoint_counters[(size_t) slot].invalidations += count - entries.size();
        capture_end[(size_t) slot] = -1;
    }
    void check_prior(int slot, const Ring& ring, int pos, bool mtp_cache) {
        const auto& h = history[(size_t) slot];
        for (int p = std::max(mtp_cache ? 1 : 0, pos - 127); p < pos; ++p) {
            const auto& cell = ring[(size_t) p % ring.size()];
            require(cell.position == p, "bounded ring history was lost without restoration");
            if (p < (int) h.size()) require(cell.token == h[(size_t) p], "stale MTP/prefix token in bounded history");
        }
    }
    void main_write(int slot, const int32_t* tokens, int n, int pos) {
        auto& h = history[(size_t) slot];
        require(pos <= (int) h.size(), "main write needs missing global prefix");
        invalidate(slot, pos);
        h.resize((size_t) pos);
        for (int t = 0; t < n; ++t) {
            check_prior(slot, main_ring[(size_t) slot], pos + t, false);
            h.push_back(tokens[t]);
            main_ring[(size_t) slot][(size_t) (pos + t) % 256] = {pos + t, tokens[t]};
        }
        main_end[(size_t) slot] = std::max(main_end[(size_t) slot], pos + n);
    }
    void mtp_write(int slot, const int32_t* tokens, int n, int pos) {
        capture_end[(size_t) slot] = -1;
        for (int t = 0; t < n; ++t) {
            check_prior(slot, mtp_ring[(size_t) slot], pos + t, true);
            mtp_ring[(size_t) slot][(size_t) (pos + t) % 256] = {pos + t, tokens[t]};
        }
        mtp_end[(size_t) slot] = std::max(mtp_end[(size_t) slot], pos + n);
    }
    void rows(int32_t value, float* logits) {
        for (int i = 0; i < vocab(); ++i) logits[i] = i == value ? 2.f : (i == (value + 1) % vocab() ? 1.f : -10.f);
    }
    bool prompt_chunk(int slot, const int32_t* tokens, int n, int pos, const std::function<int32_t(const float*, int32_t)>& choose,
                      int32_t* first, int32_t* draft, const std::function<void(int)>&,
                      PromptState&, bool final, std::string& err) {
        require(awaiting_mtp == -1, "prefill between main and MTP");
        if (fail) { err = "fake compute failure"; return false; }
        main_write(slot, tokens, n, pos);
        float logits[32];
        const int32_t greedy = (tokens[n - 1] + 1) % vocab();
        rows(greedy, logits);
        *first = choose(logits, greedy);
        if (draft) *draft = (*first + (reject ? 2 : 1)) % vocab();
        if (mtp) {
            std::vector<int32_t> shifted(tokens + 1, tokens + n);
            shifted.push_back(*first);
            mtp_write(slot, shifted.data(), n, pos + 1);
        }
        capture_end[(size_t) slot] = pos + n;
        prompts.push_back({slot, pos, n, *first, final});
        counters.expert_hits += (uint64_t) n;
        counters.expert_uses += (uint64_t) n * 2;
        charge("prompt", slot, n);
        return true;
    }
    bool forward(int slot, const int32_t* tokens, int n, int pos, int32_t* next, std::vector<float>* logits, std::string& err) {
        require(awaiting_mtp == -1, "main forward interrupted MTP");
        if (fail) { err = "fake compute failure"; return false; }
        main_write(slot, tokens, n, pos);
        if (logits) logits->resize((size_t) n * vocab());
        for (int t = 0; t < n; ++t) {
            next[t] = (tokens[t] + 1) % vocab();
            if (logits) rows(next[t], logits->data() + (size_t) t * vocab());
        }
        counters.expert_hits += (uint64_t) n;
        counters.expert_uses += (uint64_t) n * 2;
        if (mtp) awaiting_mtp = slot;
        charge("forward", slot, n);
        return true;
    }
    bool mtp_forward(int slot, const int32_t* tokens, int n, int pos, int32_t* draft, std::string&) {
        require(awaiting_mtp == slot, "MTP consumed another slot's hidden rows");
        mtp_write(slot, tokens, n, pos);
        for (int t = 0; t < n; ++t) draft[t] = (tokens[t] + (reject ? 2 : 1)) % vocab();
        awaiting_mtp = -1;
        charge("mtp", slot, n);
        return true;
    }
    PrefixCheckpointStats prefix_checkpoint_stats(int slot) const {
        auto stats = checkpoint_counters[(size_t) slot];
        stats.count = checkpoints[(size_t) slot].size();
        for (const auto& entry : checkpoints[(size_t) slot]) {
            stats.payload_bytes += entry.info.host_bytes;
            stats.metadata_bytes += entry.info.metadata_bytes;
        }
        return stats;
    }
    std::vector<int> prefix_checkpoint_boundaries(int slot) const {
        std::vector<int> out;
        for (const auto& entry : checkpoints[(size_t) slot]) out.push_back(entry.info.committed_end);
        return out;
    }
    bool capture_prefix_checkpoint(int slot, int end, bool prompt_end, bool& captured, std::string&) {
        captured = false;
        if (!checkpoints_enabled || capture_end[(size_t) slot] != end) return true;
        auto& entries = checkpoints[(size_t) slot];
        const bool milestone = end >= 4096 && (end & (end - 1)) == 0;
        entries.erase(std::remove_if(entries.begin(), entries.end(), [&](const auto& entry) {
            return entry.info.committed_end == end || (prompt_end && !entry.milestone);
        }), entries.end());
        if (entries.size() >= 8) return true;
        PrefixCheckpointInfo info;
        info.id = next_checkpoint++;
        info.committed_end = end;
        info.replay_pos = end - 1;
        info.main_written_end = main_end[(size_t) slot];
        info.mtp_written_end = mtp_end[(size_t) slot];
        info.host_bytes = sizeof(Ring) * (mtp ? 2 : 1);
        info.metadata_bytes = sizeof(int32_t) * (size_t) end;
        entries.push_back({info, history[(size_t) slot], main_ring[(size_t) slot], mtp_ring[(size_t) slot], milestone});
        ++checkpoint_counters[(size_t) slot].captures;
        checkpoint_counters[(size_t) slot].last_skip_reason = "none";  // production capture success sentinel
        captured = true;
        charge("capture", slot);
        return true;
    }
    bool find_prefix_checkpoint(int slot, const int32_t* target, int n, PrefixCheckpointInfo& out,
                                std::string& reason, std::string&) const {
        out = {};
        const auto& entries = checkpoints[(size_t) slot];
        reason = entries.empty() ? "no_checkpoint" : "prefix_mismatch";
        for (const auto& entry : entries) {
            const int count = entry.info.committed_end;
            if (count <= n && std::equal(entry.prefix.begin(), entry.prefix.begin() + count, target) &&
                entry.info.replay_pos > out.replay_pos) out = entry.info;
        }
        return true;
    }
    bool restore_prefix_checkpoint(int slot, uint64_t id, const int32_t* target, int n,
                                   PrefixCheckpointInfo& restored, std::string& err) {
        require(awaiting_mtp == -1, "restore between main and MTP");
        for (const auto& entry : checkpoints[(size_t) slot]) {
            if (entry.info.id != id) continue;
            if (entry.info.committed_end > n || !std::equal(entry.prefix.begin(), entry.prefix.end(), target)) break;
            restored = entry.info;
            history[(size_t) slot].assign(entry.prefix.begin(), entry.prefix.begin() + restored.replay_pos);
            main_ring[(size_t) slot] = entry.main;
            mtp_ring[(size_t) slot] = entry.mtp;
            main_end[(size_t) slot] = restored.main_written_end;
            mtp_end[(size_t) slot] = restored.mtp_written_end;
            capture_end[(size_t) slot] = -1;
            ++checkpoint_counters[(size_t) slot].restores;
            charge("restore", slot);
            return true;
        }
        err = "invalid fake checkpoint";
        return false;
    }
};

struct FakeClock {
    using Clock = std::chrono::steady_clock;
    Clock::time_point value{};
    void advance(double ms) {
        value += std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double, std::milli>(ms));
    }
    double ms() const { return std::chrono::duration<double, std::milli>(value.time_since_epoch()).count(); }
};

struct Fixture {
    FakeDecoder decoder;
    std::vector<std::string> output;
    ServeOptions options;
    std::unique_ptr<MimoServeScheduler<FakeDecoder>> scheduler;
    FakeClock* clock = nullptr;
    double output_delay_ms = 0;
    explicit Fixture(int quantum = 2, int context = 2048, FakeClock* time = nullptr,
                     int decode_percent = 80, int prefill_wait = 250) : clock(time) {
        options.max_context = context;
        options.prefill_quantum = quantum;
        options.decode_service_percent = decode_percent;
        options.prefill_max_wait_ms = prefill_wait;
        options.eos.clear();
        MimoServeScheduler<FakeDecoder>::Now now = [time] {
            return time ? time->value : std::chrono::steady_clock::now();
        };
        scheduler = std::make_unique<MimoServeScheduler<FakeDecoder>>(decoder, options,
            [&](const std::string& line) {
                output.push_back(line);
                if (clock) clock->advance(output_delay_ms);
            }, now);
    }
    void drain() {
        std::string err;
        for (int i = 0; scheduler->active(); ++i) {
            require(i < 4096, "scheduler did not settle");
            require(scheduler->step(err), err.c_str());
        }
    }
    std::vector<int> tokens(uint64_t id) const {
        std::vector<int> result;
        const std::string prefix = "T " + std::to_string(id) + " ";
        for (const auto& line : output) if (line.rfind(prefix, 0) == 0) result.push_back(std::stoi(line.substr(prefix.size())));
        return result;
    }
    std::vector<std::string> events(const std::string& kind, uint64_t id) const {
        std::vector<std::string> result;
        const std::string prefix = kind + " " + std::to_string(id) + " ";
        for (const auto& line : output) if (line.rfind(prefix, 0) == 0) result.push_back(line.substr(prefix.size()));
        return result;
    }
    std::map<std::string, std::string> diagnostic(const std::string& kind, uint64_t id) const {
        std::map<std::string, std::string> result;
        for (const auto& line : output) {
            if (line.rfind("INFO request_diag_v=1 ", 0) != 0) continue;
            std::istringstream words(line);
            std::map<std::string, std::string> values;
            std::string word;
            while (words >> word) {
                const auto eq = word.find('=');
                if (eq != std::string::npos) values[word.substr(0, eq)] = word.substr(eq + 1);
            }
            if (values["diag_kind"] == kind && values["command_id"] == std::to_string(id)) result = std::move(values);
        }
        return result;
    }
};

void parse() {
    SamplingParams params;
    std::vector<int32_t> ids;
    std::string err;
    int count;
    require(serve_parse_gen("GEN 4 temperature=0.5 seed=19 1,2,3", count, params, ids, err), "valid GEN rejected");
    require(count == 4 && params.seed == 19 && ids.size() == 3, "GEN fields differ");
    for (const char* bad : {"GEN -1 2", "GEN 9999999999999999999 2", "GEN 2 4294967297", "GEN 2 1,,2",
                            "GEN 2 1,", "GEN 2 temperature=nan 1", "GEN 2 seed=-1 1", "GEN 2 1 extra"})
        require(!serve_parse_gen(bad, count, params, ids, err), "bad GEN accepted");
}

void interleave(bool reject) {
    Fixture f;
    f.decoder.reject = reject;
    f.scheduler->command("GEN 1 0 8 1,2,3,4,5");
    f.scheduler->command("GEN 2 1 8 10,11,12,13,14");
    f.drain();
    require(f.tokens(1).size() == 8 && f.tokens(2).size() == 8, "wrong token ownership/count");
    require(f.tokens(1).front() == 6 && f.tokens(2).front() == 15, "crossed prompt history");
    require(f.events("DONE", 1).size() == 1 && f.events("DONE", 2).size() == 1, "terminal count differs");
    bool a = false, b = false;
    for (const auto& line : f.output) {
        if (line.rfind("T 1 ", 0) == 0) a = true;
        if (line.rfind("T 2 ", 0) == 0) b = true;
        if (line.rfind("DONE ", 0) == 0) { require(a && b, "one request finished before both made progress"); break; }
    }
    require(f.decoder.prompts.size() == 6, "prompt did not yield at quantum");
    require(f.decoder.prompts[0].slot == 0 && f.decoder.prompts[1].slot == 1, "prompt turns not alternating");
    require(f.decoder.prompts[0].boundary == 3 && f.decoder.prompts[2].boundary == 5, "wrong MTP prompt boundary");
    for (uint64_t id : {1, 2}) {
        std::istringstream done(f.events("DONE", id).front());
        int generated, prompt, reused, read;
        double prompt_ms, decode_ms;
        std::string finish;
        long long accepted, offered;
        uint64_t hits, uses;
        int ram, file;
        double mb;
        done >> generated >> prompt >> prompt_ms >> decode_ms >> finish >> accepted >> offered >> reused >> hits >> uses
             >> ram >> file >> mb >> read;
        require(hits * 2 == uses && hits > 0 && read == 5 && generated == 8, "request counter attribution failed");
        require(reject ? accepted == 0 : accepted > 0, "speculative acceptance differs");
    }
}

void cancellation() {
    Fixture f;
    f.scheduler->command("GEN 1 0 12 1,2,3,4,5,6");
    f.scheduler->command("GEN 2 1 12 10,11,12");
    std::string err;
    require(f.scheduler->step(err), "first chunk failed");
    f.scheduler->command("STOP 1");
    f.drain();
    require(f.tokens(1).empty() && f.tokens(2).size() == 12, "STOP harmed peer");
    require(f.events("DONE", 1).front().find(" cancel ") != std::string::npos, "cancel has no terminal");
    f.scheduler->command("GEN 3 0 4 20,21,22");
    f.scheduler->command("STOP 1");
    f.drain();
    require(f.tokens(3).size() == 4, "late STOP harmed new occupant");

    Fixture queued;
    queued.scheduler->command("GEN 1 0 5 1,2");
    queued.scheduler->command("STOP 1");
    queued.drain();
    require(queued.decoder.prompts.empty() && queued.decoder.resets[0] == 0, "pre-dispatch STOP ran compute");
}

void late_prompt() {
    Fixture f;
    std::string err;
    f.scheduler->command("GEN 1 0 32 1,2");
    require(f.scheduler->step(err) && f.scheduler->step(err), "established decoder failed");
    const size_t before = f.output.size();
    f.scheduler->command("GEN 2 1 8 10,11,12,13,14,15,16,17");
    f.drain();
    bool reading = false, peer_progress = false, a_done = false;
    for (size_t i = before; i < f.output.size(); ++i) {
        const auto& line = f.output[i];
        if (line.rfind("DONE 1 ", 0) == 0) a_done = true;
        if (line.rfind("PP 2 ", 0) == 0) {
            if (reading && !a_done) require(peer_progress, "long prompt starved established decoder");
            reading = true;
            peer_progress = false;
        }
        if (reading && line.rfind("T 1 ", 0) == 0) peer_progress = true;
    }
    require(reading, "late prompt made no progress");
}

void interactive_prefill() {
    Fixture f(256);
    std::string err;
    f.scheduler->command("GEN 1 0 16 1,2");
    require(f.scheduler->step(err) && f.scheduler->step(err), "established decoder failed");
    std::string prompt;
    for (int i = 0; i < 100; ++i) prompt += (i ? "," : "") + std::to_string(i % 32);
    f.scheduler->command("GEN 2 1 4 " + prompt);
    bool a_done = false, small = false, bulk = false;
    for (int turns = 0; f.scheduler->active(); ++turns) {
        require(turns < 256, "interactive prefill did not settle");
        const size_t before = f.decoder.prompts.size();
        require(f.scheduler->step(err), "interactive prefill failed");
        for (size_t i = before; i < f.decoder.prompts.size(); ++i) {
            const auto& call = f.decoder.prompts[i];
            if (call.slot != 1) continue;
            if (!a_done) {
                require(call.count <= 8, "bulk prefill paused a live decoder");
                small = true;
            } else {
                bulk |= call.count > 8;
                require(call.count <= 256, "bulk prefill exceeded configured ceiling");
            }
        }
        a_done = !f.events("DONE", 1).empty();
    }
    require(small && bulk, "prefill did not switch from interactive windows back to bulk");
    require(f.tokens(1).size() == 16 && f.tokens(2).size() == 4, "interactive prefill changed token counts");
}

void prefix() {
    Fixture f;
    f.scheduler->command("GEN 1 0 4 1,2,3");
    f.drain();
    const auto a = f.tokens(1);
    std::string prompt = "1,2,3";
    for (int t : a) prompt += "," + std::to_string(t);
    f.scheduler->command("GEN 2 0 2 " + prompt);
    f.drain();
    require(f.events("RESUME", 2).front() == "6", "exact matching prompt did not replay its final token");
    require(f.decoder.resets[0] == 1, "retained prefix reset MTP state");
    f.scheduler->command("GEN 3 1 2 " + prompt);
    f.drain();
    require(f.events("RESUME", 3).front() == "0", "prefix crossed slots");

    Fixture divergent;
    divergent.scheduler->command("GEN 1 0 1 1,2,3,4");
    divergent.drain();
    divergent.scheduler->command("GEN 2 0 2 1,2,20,21");
    divergent.drain();
    require(divergent.events("RESUME", 2).front() == "1", "divergent MTP boundary was not recomputed");
    require(divergent.decoder.resets[0] == 1, "safe divergent reuse discarded the whole prefix");

    Fixture interrupted;
    interrupted.scheduler->command("GEN 1 0 8 1,2,3,4,5,6");
    std::string err;
    require(interrupted.scheduler->step(err), "partial prefill failed");
    interrupted.scheduler->command("STOP 1");
    interrupted.drain();
    interrupted.scheduler->command("GEN 2 0 2 1,2,20,21");
    interrupted.drain();
    require(interrupted.events("RESUME", 2).front() == "1", "cancelled prefill retained stale MTP lookahead");

    Fixture extended;
    extended.scheduler->command("GEN 1 0 1 1,2,3,4");
    extended.drain();
    extended.scheduler->command("GEN 2 0 2 1,2,3,4,5,20,21");
    extended.drain();
    require(extended.events("RESUME", 2).front() == "4", "extension retained sampled MTP lookahead");

    Fixture wrapped(64);
    std::string long_prompt;
    for (int i = 0; i < 600; ++i) long_prompt += (i ? "," : "") + std::to_string(i % 32);
    wrapped.scheduler->command("GEN 1 0 2 " + long_prompt);
    wrapped.drain();
    wrapped.scheduler->command("GEN 2 0 2 0,1,2,3,4");
    wrapped.drain();
    require(wrapped.events("RESUME", 2).front() == "0", "lost SWA history used an unsafe retained prefix");
}

std::vector<int> sampled(bool peer, int quantum) {
    Fixture f(quantum);
    f.scheduler->command("GEN 1 0 24 temperature=0.7 seed=1234 1,2,3,4,5,6,7");
    if (peer) f.scheduler->command("GEN 2 1 24 temperature=0.7 seed=4321 11,12,13,14,15,16,17");
    f.drain();
    return f.tokens(1);
}

std::string token_list(int n, int seed = 0) {
    std::string out;
    for (int i = 0; i < n; ++i) out += (i ? "," : "") + std::to_string((i + seed) % 32);
    return out;
}

void checkpoint_restore() {
    Fixture f(64, 4096);
    const std::string prefix = token_list(600);
    f.scheduler->command("GEN 1 0 400 " + prefix);
    f.drain();
    require(f.tokens(1).size() > 256, "checkpoint fixture did not age the live rings");
    f.scheduler->command("GEN 2 0 8 " + prefix + ",17,18,19,20,21,22,23,24,25,26");
    f.drain();
    require(f.events("RESUME", 2).front() == "599", "lost prompt prefix was not restored from its checkpoint");
    auto decision = f.diagnostic("prefix", 2), timing = f.diagnostic("timing", 2);
    require(decision["prefix_reason"] == "restored_checkpoint" && decision["checkpoint_reason"] == "restored",
            "checkpoint restoration was not explained");
    require(decision["raw_lcp"] == "600" && decision["checkpoint_boundary"] == "600", "wrong checkpoint identity");
    require(timing["fresh_prompt_tokens"] == "11" && timing["checkpoint_restores"] == "1", "whole prefix replayed");
    require(f.decoder.resets[0] == 1, "checkpoint restore destructively reset the cache");
    // The restored watermark must replace the old far-ahead value, not retain it and reject all future live reuse.
    f.scheduler->command("GEN 3 1 4 11,12,13");
    f.drain();
    f.scheduler->command("GEN 4 0 4 " + prefix + ",31,30");
    f.drain();
    require(f.diagnostic("prefix", 4)["prefix_reason"] == "retained_live", "restore kept the stale write watermark");
    require(f.events("RESUME", 4).front() == "599", "peer reset or MTP lookahead broke restored continuation");
}

void checkpoint_branch_invalidation() {
    Fixture f(2048, 16384);
    const std::string original = token_list(9000);
    f.scheduler->command("GEN 1 0 400 " + original);
    f.drain();
    require(f.decoder.prefix_checkpoint_boundaries(0).size() == 3, "missing geometric checkpoint anchors");
    f.scheduler->command("GEN 2 0 4 " + token_list(5000) + "," + token_list(1000, 17));
    f.drain();
    require(f.events("RESUME", 2).front() == "4095", "did not choose the largest matching earlier anchor");
    const auto anchors = f.decoder.prefix_checkpoint_boundaries(0);
    require(std::find(anchors.begin(), anchors.end(), 8192) == anchors.end() &&
            std::find(anchors.begin(), anchors.end(), 9000) == anchors.end(), "overwritten global branch stayed reusable");
    f.scheduler->command("GEN 3 0 2 " + original);
    f.drain();
    require(f.events("RESUME", 3).front() == "4095", "returned to a stale future checkpoint without global KV");
    require(f.decoder.prefix_checkpoint_stats(0).count <= 8, "checkpoint count grew without a bound");

    Fixture missing(64, 4096);
    missing.decoder.checkpoints_enabled = false;
    missing.scheduler->command("GEN 1 0 400 " + token_list(600));
    missing.drain();
    missing.scheduler->command("GEN 2 0 2 " + token_list(600));
    missing.drain();
    require(missing.events("RESUME", 2).front() == "0" &&
            missing.diagnostic("prefix", 2)["prefix_reason"] == "ring_history_missing", "missing snapshot bypassed safe replay");
}

void weighted_service() {
    for (double prompt_cost : {2.0, 80.0}) {
        FakeClock clock;
        Fixture f(256, 2048, &clock, 80, 10000);
        bool mixed = false;
        double decode_ms = 0, prefill_ms = 0;
        int decode_turns = 0, prefill_turns = 0;
        f.decoder.work = [&](const char* operation, int slot, int) {
            const std::string kind(operation);
            if (kind == "prompt") {
                clock.advance(prompt_cost);
                if (mixed && slot == 1) { prefill_ms += prompt_cost; ++prefill_turns; }
            } else if (kind == "forward" || kind == "mtp") {
                const double ms = kind == "forward" ? 10 : 2;
                clock.advance(ms);
                if (mixed && slot == 0) {
                    decode_ms += ms;
                    if (kind == "forward") ++decode_turns;
                }
            }
        };
        std::string err;
        f.scheduler->command("GEN 1 0 1000 1,2");
        require(f.scheduler->step(err) && f.scheduler->step(err), "weighted fixture could not establish decoding");
        f.scheduler->command("GEN 2 1 8 " + token_list(1600));
        mixed = true;
        for (int i = 0; i < 200; ++i) require(f.scheduler->step(err), "weighted service failed");
        require(prefill_turns > 0 && decode_turns > 0, "weighted service starved a phase");
        require(std::fabs(decode_ms / (decode_ms + prefill_ms) - 0.8) < 0.05, "service share followed turn counts, not time");
        require(prompt_cost > 12 ? decode_turns > prefill_turns * 4 : decode_turns < prefill_turns,
                "unequal operation costs did not change the dispatch ratio");
        mixed = false;
        f.scheduler->shutdown();
        f.drain();
        auto a = f.diagnostic("timing", 1), b = f.diagnostic("timing", 2);
        require(std::stod(a["prompt_service_ms"]) == prompt_cost, "final prefill charged to decode");
        require(std::fabs(std::stod(a["decode_service_ms"]) - (decode_ms + 12)) < 1e-6, "peer service charged to decoder");
        require(std::fabs(std::stod(b["prompt_service_ms"]) - prefill_ms) < 1e-6 && b["decode_service_ms"] == "0.000000",
                "phase service accounting differs");
        require(a["prompt_expert_hits"] == "2" && a["prompt_expert_uses"] == "4", "phase routing counters mixed");
        std::istringstream done(f.events("DONE", 1).front());
        int generated, tokens;
        double prompt_wall, decode_wall;
        done >> generated >> tokens >> prompt_wall >> decode_wall;
        require(std::fabs(decode_wall - (decode_ms + prefill_ms + 12)) < 0.11, "decode wall excluded peer work");
    }
}

void service_deadline_and_cancellation() {
    FakeClock clock;
    Fixture f(256, 2048, &clock, 99, 250);
    bool mixed = false;
    double last_prefill_done = -1, largest_wait = 0;
    int prefill_turns = 0;
    std::vector<int> forward_slots;
    f.decoder.work = [&](const char* operation, int slot, int) {
        const std::string kind(operation);
        if (kind == "prompt") {
            if (mixed && slot == 1) {
                if (last_prefill_done >= 0) {
                    const double waited = clock.ms() - last_prefill_done;
                    largest_wait = std::max(largest_wait, waited);
                    require(waited <= 250 + 40 + 0.001, "prefill deadline exceeded more than one atomic unit");
                }
                ++prefill_turns;
            }
            clock.advance(80);
            if (mixed && slot == 1) last_prefill_done = clock.ms();
        } else if (kind == "forward" || kind == "mtp") {
            if (kind == "forward") forward_slots.push_back(slot);
            clock.advance(kind == "forward" ? 30 : 10);
        }
    };
    std::string err;
    f.scheduler->command("GEN 1 0 1000 1,2");
    require(f.scheduler->step(err) && f.scheduler->step(err), "deadline fixture failed");
    f.scheduler->command("GEN 2 1 8 " + token_list(1600));
    mixed = true;
    for (int i = 0; i < 150; ++i) require(f.scheduler->step(err), "deadline scheduling failed");
    require(prefill_turns > 4 && largest_wait >= 250, "high decode weight suppressed deadline overrides");
    const double stopped_at = clock.ms();
    f.scheduler->command("STOP 2");
    require(f.scheduler->step(err), "deadline cancellation failed");
    require(clock.ms() == stopped_at && f.events("DONE", 2).size() == 1, "weighted debt delayed cancellation");
    mixed = false;
    forward_slots.clear();
    f.scheduler->command("GEN 3 1 4 3,4");
    require(f.scheduler->step(err), "replacement mixed epoch failed");
    require(!forward_slots.empty() && forward_slots.front() == 0, "new command inherited old service debt");
    f.scheduler->shutdown();
    f.drain();
}

void same_phase_round_robin() {
    FakeClock clock;
    Fixture prefill(2, 2048, &clock);
    prefill.decoder.work = [&](const char*, int, int) { clock.advance(1); };
    prefill.scheduler->command("GEN 1 0 100 " + token_list(40));
    prefill.scheduler->command("GEN 2 1 100 " + token_list(40, 7));
    std::string err;
    for (int i = 0; i < 10; ++i) require(prefill.scheduler->step(err), "same-phase prefill failed");
    for (size_t i = 0; i < prefill.decoder.prompts.size(); ++i)
        require(prefill.decoder.prompts[i].slot == (int) (i % 2), "two prefills lost round-robin fairness");
    prefill.scheduler->shutdown();
    prefill.drain();

    FakeClock decode_clock;
    Fixture decode(2, 2048, &decode_clock);
    std::vector<int> order;
    decode.decoder.work = [&](const char* op, int slot, int) {
        if (std::string(op) == "forward") order.push_back(slot);
        decode_clock.advance(1);
    };
    decode.scheduler->command("GEN 1 0 100 1");
    decode.scheduler->command("GEN 2 1 100 7");
    for (int i = 0; i < 3; ++i) require(decode.scheduler->step(err), "same-phase decode setup failed");
    order.clear();
    for (int i = 0; i < 8; ++i) require(decode.scheduler->step(err), "same-phase decode failed");
    for (size_t i = 0; i < order.size(); ++i) require(order[i] == (int) (i % 2), "two decoders lost round-robin fairness");
    decode.scheduler->shutdown();
    decode.drain();
}

void service_excludes_output() {
    FakeClock clock;
    Fixture f(8, 2048, &clock);
    f.output_delay_ms = 5;
    f.decoder.work = [&](const char* op, int slot, int) {
        const std::string kind(op);
        double ms = kind == "reset" ? 2 : kind == "prompt" ? 7 : kind == "capture" ? 3 :
                    kind == "forward" ? 10 : kind == "mtp" ? 3 : 0;
        clock.advance(ms);
        if (kind == "capture") f.decoder.checkpoint_counters[(size_t) slot].capture_ms += ms;
    };
    f.scheduler->command("GEN 1 0 4 1,2");
    f.drain();
    auto timing = f.diagnostic("timing", 1);
    require(timing["prompt_service_ms"] == "12.000000" && timing["decode_service_ms"] == "26.000000",
            "protocol output was charged as compute service");
    require(timing["checkpoint_capture_ms"] == "3.000000", "checkpoint copy overhead not separated");
    std::istringstream done(f.events("DONE", 1).front());
    int generated, tokens;
    double prompt_ms, decode_ms;
    done >> generated >> tokens >> prompt_ms >> decode_ms;
    require(decode_ms > 26 + 20, "delivered wall time incorrectly excluded output delays");
}

void errors_and_shutdown() {
    Fixture f;
    f.scheduler->command("GEN 1 0 8 1,2");
    f.scheduler->command("GEN 2 0 8 3,4");
    f.scheduler->command("GEN 3 9 8 3,4");
    f.scheduler->command("GEN 4 1 8 99");
    require(f.events("ERR", 2).size() == 1 && f.events("ERR", 3).size() == 1 && f.events("ERR", 4).size() == 1,
            "bad command error attribution failed");
    f.scheduler->shutdown();
    f.drain();
    require(f.events("DONE", 1).size() == 1, "shutdown lost terminal");
    bool duplicate = false;
    try { f.scheduler->command("GEN 1 0 2 1,2"); } catch (const std::runtime_error&) { duplicate = true; }
    require(duplicate, "reused id accepted");

    Fixture failed;
    failed.scheduler->command("GEN 1 0 8 1,2");
    failed.scheduler->command("GEN 2 1 8 3,4");
    failed.decoder.fail = true;
    std::string err;
    require(!failed.scheduler->step(err), "shared compute error was ignored");
    failed.scheduler->fail_all(err);
    require(!failed.scheduler->active() && failed.events("ERR", 1).size() == 1 && failed.events("ERR", 2).size() == 1,
            "shared error did not settle both requests");

    Fixture zero;
    zero.decoder.mtp = false;
    zero.scheduler->command("GEN 1 0 0 1,2");
    zero.drain();
    require(zero.tokens(1).empty() && zero.events("DONE", 1).size() == 1, "zero-token command did not finish");
}
}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--protocol-trace") {
            FakeClock clock;
            Fixture f(64, 4096, &clock);
            f.decoder.work = [&](const char*, int, int) { clock.advance(1); };
            f.scheduler->command("GEN 1 0 400 " + token_list(600));
            f.drain();
            f.scheduler->command("GEN 2 0 8 " + token_list(600) + ",17,18,19,20,21,22,23,24,25,26");
            f.drain();
            for (const auto& line : f.output) std::cout << line << '\n';
            return 0;
        }
        parse();
        interleave(false);
        interleave(true);
        cancellation();
        late_prompt();
        interactive_prefill();
        prefix();
        checkpoint_restore();
        checkpoint_branch_invalidation();
        weighted_service();
        service_deadline_and_cancellation();
        same_phase_round_robin();
        service_excludes_output();
        require(sampled(false, 2) == sampled(true, 2), "peer consumed request RNG");
        require(sampled(false, 2) == sampled(false, 64), "intermediate prompt chunks consumed RNG");
        errors_and_shutdown();
        std::cout << "MiMo two-slot scheduler: all tests passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "MiMo two-slot scheduler: " << e.what() << '\n';
        return 1;
    }
}
