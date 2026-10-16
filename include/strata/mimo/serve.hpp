// Two live MiMo requests, one compute owner. The decoder is a template parameter so the protocol and scheduling
// invariants can be tested with a small CPU fake; production uses MimoDecoder, with one KV/graph set per slot.
#pragma once

#include "strata/mimo/sampling.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace strata::mimo {

inline bool serve_uint(const std::string& text, uint64_t& value) {
    if (text.empty()) return false;
    const auto r = std::from_chars(text.data(), text.data() + text.size(), value);
    return r.ec == std::errc{} && r.ptr == text.data() + text.size();
}

// Shared with the legacy parser: everything after GEN is the existing max_new / sampling keys / token list.
inline bool serve_parse_gen(const std::string& line, int& max_new, SamplingParams& sp,
                            std::vector<int32_t>& ids, std::string& err) {
    std::istringstream in(line);
    std::string word, amount;
    in >> word >> amount;
    uint64_t count = 0;
    if (!serve_uint(amount, count) || count > (uint64_t) std::numeric_limits<int>::max()) {
        err = "bad request: GEN needs a nonnegative <max_new>";
        return false;
    }
    max_new = (int) count;
    sp = SamplingParams{};
    ids.clear();
    while (in >> word) {
        const size_t eq = word.find('=');
        if (eq == std::string::npos) {
            std::stringstream ls(word);
            std::string value;
            while (std::getline(ls, value, ',')) {
                uint64_t token = 0;
                if (!serve_uint(value, token) || token > (uint64_t) std::numeric_limits<int32_t>::max()) {
                    err = "bad request: token id '" + value + "'";
                    return false;
                }
                ids.push_back((int32_t) token);
            }
            if (!word.empty() && word.back() == ',') {
                err = "bad request: empty token id";
                return false;
            }
            if (in >> word) { err = "bad request: text after token list"; return false; }
            break;
        }
        const std::string key = word.substr(0, eq), value = word.substr(eq + 1);
        auto number = [&](float& out) {
            char* end = nullptr;
            out = std::strtof(value.c_str(), &end);
            return !value.empty() && end == value.c_str() + value.size() && std::isfinite(out);
        };
        auto integer = [&](int& out) {
            const auto r = std::from_chars(value.data(), value.data() + value.size(), out);
            return !value.empty() && r.ec == std::errc{} && r.ptr == value.data() + value.size();
        };
        bool good = true;
        if (key == "temperature") good = number(sp.temperature);
        else if (key == "top_p") good = number(sp.top_p);
        else if (key == "top_k") good = integer(sp.top_k);
        else if (key == "min_p") good = number(sp.min_p);
        else if (key == "penalty_repeat") good = number(sp.penalty_repeat);
        else if (key == "penalty_freq") good = number(sp.penalty_freq);
        else if (key == "penalty_present") good = number(sp.penalty_present);
        else if (key == "penalty_last_n") good = integer(sp.penalty_last_n);
        else if (key == "seed") good = serve_uint(value, sp.seed);
        if (!good) { err = "bad request: sampling value for " + key; return false; }
    }
    if (ids.empty()) { err = "bad request: no prompt tokens"; return false; }
    return true;
}

struct ServeOptions {
    int max_context = 262144;
    int prefill_quantum = 2048;
    int ring_capacity = 256;
    std::set<int32_t> eos = {151645, 151643};
};

template <class Decoder>
class MimoServeScheduler {
    using Clock = std::chrono::steady_clock;
    struct Request {
        uint64_t id = 0;
        int max_new = 0, start = 0, cursor = 0, pos = 0, generated = 0;
        bool cancel = false, prepared = false, host_choice = false;
        int32_t token = 0, draft = 0;
        uint64_t hits = 0, uses = 0;
        long long offered = 0, accepted = 0;
        double prompt_ms = 0;
        Clock::time_point began = Clock::now(), decode_began{};
        std::string finish = "length";
        std::vector<int32_t> prompt;
        Sampler sampler;
        typename Decoder::PromptState prompt_state;
    };
    struct Slot {
        std::vector<int32_t> session;
        // A rejected draft or MTP may have written beyond the accepted session. Include that watermark when
        // deciding whether the old SWA rings still contain all keys required by a retained prefix.
        int written_end = 0;
        std::unique_ptr<Request> request;
    };

public:
    using Output = std::function<void(const std::string&)>;
    MimoServeScheduler(Decoder& decoder, ServeOptions options, Output output)
        : decoder_(decoder), options_(std::move(options)), output_(std::move(output)), slots_(2) {}

    bool active() const {
        for (const auto& slot : slots_) if (slot.request) return true;
        return false;
    }
    bool quitting() const { return quitting_; }

    // Called only by the compute owner, between complete decoder calls. IDs increase in GEN send order;
    // this permits stale/duplicate rejection without an ever-growing set of completed command ids.
    void command(const std::string& line) {
        if (line.empty()) return;
        if (line == "QUIT") { shutdown(); return; }
        std::istringstream in(line);
        std::string kind, name;
        in >> kind >> name;
        uint64_t id = 0;
        if (!serve_uint(name, id) || id == 0) { output_("ERR 0 bad request: command needs a positive id"); return; }
        if (kind == "STOP") {
            for (auto& slot : slots_)
                if (slot.request && slot.request->id == id) slot.request->cancel = true;
            return;
        }
        if (kind != "GEN") { event("ERR", id, "unknown command"); return; }
        if (id <= last_id_) {
            // A duplicate live id cannot receive another terminal without ambiguously terminating its owner.
            // Treat a broken command-id stream as a process protocol failure instead.
            throw std::runtime_error("GEN command ids must increase and must not be reused");
        }
        last_id_ = id;
        if (quitting_) { event("ERR", id, "engine is closing"); return; }
        std::string slot_text;
        in >> slot_text;
        uint64_t slot_id = 0;
        if (!serve_uint(slot_text, slot_id) || slot_id >= slots_.size()) {
            event("ERR", id, "bad request: slot must be 0 or 1");
            return;
        }
        auto& slot = slots_[(size_t) slot_id];
        if (slot.request) { event("ERR", id, "slot is occupied"); return; }
        std::string rest;
        std::getline(in, rest);
        auto request = std::make_unique<Request>();
        SamplingParams params;
        std::string err;
        if (!serve_parse_gen("GEN" + rest, request->max_new, params, request->prompt, err)) {
            event("ERR", id, err);
            return;
        }
        const int n = (int) request->prompt.size();
        for (int32_t token : request->prompt) {
            if (token < 0 || token >= decoder_.vocab()) { event("ERR", id, "a token id is outside the vocabulary"); return; }
        }
        if ((int64_t) n + request->max_new + 8 > options_.max_context) {
            event("ERR", id, "prompt + max_new exceeds the per-slot context (" + std::to_string(options_.max_context) + ")");
            return;
        }
        int keep = 0;
        while (keep < (int) slot.session.size() && keep < n - 1 && slot.session[(size_t) keep] == request->prompt[(size_t) keep])
            ++keep;
        // MTP row p combines token p with main hidden(p - 1). The first unvalidated token therefore invalidates
        // MTP at keep itself, while prompt(main keep) would only rewrite MTP starting at keep + 1. Replay its
        // predecessor too, including when extending an old session past a sampled-but-unemitted lookahead.
        if (decoder_.has_mtp()) keep = std::max(0, keep - 1);
        // If the ring cannot serve that earlier start, replay from zero. Replaying just 128 tokens would itself
        // need older SWA keys at its first position; it is not an exact reconstruction of the missing history.
        if (keep < slot.written_end - (options_.ring_capacity - 127)) keep = 0;
        request->id = id;
        request->start = request->cursor = keep;
        request->host_choice = params.needs_logits();
        request->sampler.begin(params);
        request->sampler.history(request->prompt.data(), n);
        slot.request = std::move(request);
        event("RESUME", id, std::to_string(keep));
    }

    void shutdown() {
        quitting_ = true;
        for (auto& slot : slots_) if (slot.request) slot.request->cancel = true;
    }

    void fail_all(const std::string& reason) {
        for (auto& slot : slots_) {
            if (slot.request) event("ERR", slot.request->id, reason);
            slot.request.reset();
            slot.session.clear();
        }
    }

    // False means shared compute state cannot safely serve another request. The caller must terminate the engine.
    bool step(std::string& err) {
        for (size_t i = 0; i < slots_.size(); ++i) {
            const size_t index = next_;
            next_ = (next_ + 1) % slots_.size();
            auto& slot = slots_[index];
            if (!slot.request) continue;
            auto& request = *slot.request;
            if (request.cancel) {
                request.finish = "cancel";
                finish(slot);
                return true;
            }
            const auto before = decoder_.stats();
            if (!quantum((int) index, slot, err)) return false;
            const auto after = decoder_.stats();
            request.hits += after.expert_hits - before.expert_hits;
            request.uses += after.expert_uses - before.expert_uses;
            if (request.decode_began != Clock::time_point{} &&
                (request.generated >= request.max_new || request.finish != "length")) finish(slot);
            return true;
        }
        return true;
    }

private:
    static double elapsed(Clock::time_point began) {
        return std::chrono::duration<double, std::milli>(Clock::now() - began).count();
    }
    void event(const char* kind, uint64_t id, const std::string& payload) {
        output_(std::string(kind) + " " + std::to_string(id) + " " + payload);
    }
    void token(Request& request, int32_t id) {
        event("T", request.id, std::to_string(id));
        ++request.generated;
        if (options_.eos.count(id)) request.finish = "stop";
    }
    void finish(Slot& slot) {
        auto& r = *slot.request;
        const bool decoding = r.decode_began != Clock::time_point{};
        if (!decoding) r.prompt_ms = elapsed(r.began);
        char payload[512];
        std::snprintf(payload, sizeof payload, "%d %d %.1f %.1f %s %lld %lld %d %llu %llu 0 0 0.0 %d",
                      r.generated, (int) r.prompt.size(), r.prompt_ms, decoding ? elapsed(r.decode_began) : 0.0,
                      r.finish.c_str(), r.accepted, r.offered, r.start, (unsigned long long) r.hits,
                      (unsigned long long) r.uses, r.cursor - r.start);
        event("DONE", r.id, payload);
        slot.request.reset();
    }
    bool quantum(int index, Slot& slot, std::string& err) {
        auto& r = *slot.request;
        const int n = (int) r.prompt.size(), vocab = decoder_.vocab();
        if (!r.prepared) {
            if (r.start == 0) {
                if (!decoder_.reset_sequence(index, err)) return false;
                slot.written_end = 0;
            }
            slot.session.resize((size_t) r.start);
            r.prepared = true;
        }
        if (r.cursor < n) {
            bool peer_decoding = false;
            for (const auto& peer : slots_)
                if (&peer != &slot && peer.request && peer.request->decode_began != Clock::time_point{})
                    peer_decoding = true;
            // Bulk prefill repeatedly streams missing experts and borrows/refills their slots. Even a modest
            // bulk slice can pause a live response for seconds. One decode-sized prompt window avoids that
            // transition while a peer streams; return to the configured bulk ceiling once it finishes.
            const int quantum = peer_decoding ? std::min(options_.prefill_quantum, 8) : options_.prefill_quantum;
            const int count = std::min(quantum, n - r.cursor);
            const bool final = r.cursor + count == n;
            auto choose = [&](const float* logits, int32_t greedy) {
                if (!final) return r.prompt[(size_t) (r.cursor + count)];
                return r.host_choice ? r.sampler.sample(logits, vocab) : greedy;
            };
            if (!decoder_.prompt_chunk(index, r.prompt.data() + r.cursor, count, r.cursor, choose, &r.token,
                                       decoder_.has_mtp() ? &r.draft : nullptr, {}, r.prompt_state, final, err)) return false;
            r.cursor += count;
            slot.session.assign(r.prompt.begin(), r.prompt.begin() + r.cursor);
            slot.written_end = std::max(slot.written_end, r.cursor + (decoder_.has_mtp() ? 1 : 0));
            const double ms = elapsed(r.began);
            event("PP", r.id, std::to_string(r.cursor) + " " + std::to_string(n) + " " + std::to_string(ms) + " " +
                  std::to_string(ms > 0 ? (r.cursor - r.start) / ms * 1000.0 : 0.0));
            if (final) {
                r.prompt_ms = ms;
                r.decode_began = Clock::now();
                r.pos = n;
            }
            return true;
        }
        std::vector<float> logits;
        int32_t next[8];
        if (decoder_.has_mtp() && r.generated + 1 < r.max_new) {
            const int32_t window[2] = {r.token, r.draft};
            if (!decoder_.forward(index, window, 2, r.pos, next, r.host_choice ? &logits : nullptr, err)) return false;
            slot.written_end = std::max(slot.written_end, r.pos + 2);
            ++r.offered;
            int32_t other = next[0];
            slot.session.push_back(r.token);
            bool keep;
            if (r.host_choice) {
                r.sampler.history(slot.session.data(), (int) slot.session.size());
                keep = r.sampler.accept_draft(logits.data(), vocab, r.draft, &other);
            } else {
                keep = next[0] == r.draft;
            }
            int32_t mt[2];
            int accepted;
            token(r, r.token);
            if (keep && r.finish == "length") {
                ++r.accepted;
                token(r, r.draft);
                slot.session.push_back(r.draft);
                if (r.host_choice) r.sampler.history(slot.session.data(), (int) slot.session.size());
                r.token = r.host_choice ? r.sampler.sample(logits.data() + vocab, vocab) : next[1];
                mt[0] = r.draft;
                mt[1] = r.token;
                accepted = 2;
            } else {
                r.token = other;
                mt[0] = r.token;
                accepted = 1;
            }
            if (!decoder_.mtp_forward(index, mt, accepted, r.pos + 1, next, err)) return false;
            r.draft = next[accepted - 1];
            r.pos += accepted;
            slot.written_end = std::max(slot.written_end, r.pos + 1);
        } else {
            if (!decoder_.forward(index, &r.token, 1, r.pos, next, r.host_choice ? &logits : nullptr, err)) return false;
            token(r, r.token);
            slot.session.push_back(r.token);
            if (r.host_choice) r.sampler.history(slot.session.data(), (int) slot.session.size());
            r.token = r.host_choice ? r.sampler.sample(logits.data(), vocab) : next[0];
            ++r.pos;
            // Complete the MTP history even on the final single-token turn, so prefix reuse has no missing row.
            if (decoder_.has_mtp() && !decoder_.mtp_forward(index, &r.token, 1, r.pos, &r.draft, err)) return false;
            slot.written_end = std::max(slot.written_end, r.pos + (decoder_.has_mtp() ? 1 : 0));
        }
        return true;
    }

    Decoder& decoder_;
    ServeOptions options_;
    Output output_;
    std::vector<Slot> slots_;
    size_t next_ = 0;
    uint64_t last_id_ = 0;
    bool quitting_ = false;
};

}  // namespace strata::mimo
