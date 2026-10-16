// GPU-free tests of the production two-slot scheduler: tagged ownership, fair prompt/decode turns, MTP boundaries,
// cancellation, prefix reuse and request-local sampling/counters. The fake asserts that main/MTP remain atomic.
#include "strata/mimo/serve.hpp"

#include <array>
#include <iostream>
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
    struct PromptCall { int slot, pos, count, boundary; bool final; };
    std::array<std::vector<int32_t>, 2> history;
    std::array<int, 2> resets{};
    std::vector<PromptCall> prompts;
    bool mtp = true, reject = false, fail = false;
    int awaiting_mtp = -1;
    int vocab() const { return 32; }
    bool has_mtp() const { return mtp; }
    const Stats& stats() const { return counters; }
    bool reset_sequence(int slot, std::string&) {
        require(awaiting_mtp == -1, "reset between main and MTP");
        ++resets[(size_t) slot];
        history[(size_t) slot].clear();
        return true;
    }
    void rows(int32_t value, float* logits) {
        for (int i = 0; i < vocab(); ++i) logits[i] = i == value ? 2.f : (i == (value + 1) % vocab() ? 1.f : -10.f);
    }
    bool prompt_chunk(int slot, const int32_t* tokens, int n, int pos, const std::function<int32_t(const float*, int32_t)>& choose,
                      int32_t* first, int32_t* draft, const std::function<void(int)>&,
                      PromptState&, bool final, std::string& err) {
        require(awaiting_mtp == -1, "prefill between main and MTP");
        if (fail) { err = "fake compute failure"; return false; }
        auto& h = history[(size_t) slot];
        require(pos <= (int) h.size(), "prefill needs missing prefix");
        h.resize((size_t) pos);
        h.insert(h.end(), tokens, tokens + n);
        float logits[32];
        const int32_t greedy = (tokens[n - 1] + 1) % vocab();
        rows(greedy, logits);
        *first = choose(logits, greedy);
        if (draft) *draft = (*first + (reject ? 2 : 1)) % vocab();
        prompts.push_back({slot, pos, n, *first, final});
        counters.expert_hits += (uint64_t) n;
        counters.expert_uses += (uint64_t) n * 2;
        return true;
    }
    bool forward(int slot, const int32_t* tokens, int n, int pos, int32_t* next, std::vector<float>* logits, std::string& err) {
        require(awaiting_mtp == -1, "main forward interrupted MTP");
        if (fail) { err = "fake compute failure"; return false; }
        auto& h = history[(size_t) slot];
        require(pos <= (int) h.size(), "decode needs missing prefix");
        h.resize((size_t) pos);
        h.insert(h.end(), tokens, tokens + n);
        if (logits) logits->resize((size_t) n * vocab());
        for (int t = 0; t < n; ++t) {
            next[t] = (tokens[t] + 1) % vocab();
            if (logits) rows(next[t], logits->data() + (size_t) t * vocab());
        }
        counters.expert_hits += (uint64_t) n;
        counters.expert_uses += (uint64_t) n * 2;
        if (mtp) awaiting_mtp = slot;
        return true;
    }
    bool mtp_forward(int slot, const int32_t* tokens, int n, int, int32_t* draft, std::string&) {
        require(awaiting_mtp == slot, "MTP consumed another slot's hidden rows");
        for (int t = 0; t < n; ++t) draft[t] = (tokens[t] + (reject ? 2 : 1)) % vocab();
        awaiting_mtp = -1;
        return true;
    }
};

struct Fixture {
    FakeDecoder decoder;
    std::vector<std::string> output;
    ServeOptions options;
    std::unique_ptr<MimoServeScheduler<FakeDecoder>> scheduler;
    explicit Fixture(int quantum = 2) {
        options.max_context = 2048;
        options.prefill_quantum = quantum;
        options.eos.clear();
        scheduler = std::make_unique<MimoServeScheduler<FakeDecoder>>(decoder, options,
            [&](const std::string& line) { output.push_back(line); });
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
    bool reading = false, peer_progress = false;
    for (size_t i = before; i < f.output.size(); ++i) {
        const auto& line = f.output[i];
        if (line.rfind("PP 2 ", 0) == 0) {
            if (reading) require(peer_progress, "long prompt starved established decoder");
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
    require(f.events("RESUME", 2).front() == "5", "retained prefix did not replay MTP boundary predecessor");
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

int main() {
    try {
        parse();
        interleave(false);
        interleave(true);
        cancellation();
        late_prompt();
        interactive_prefill();
        prefix();
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
