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
            if (!word.empty() && word.back() == ',') { err = "bad request: empty token id"; return false; }
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
    int decode_service_percent = 80;
    int prefill_max_wait_ms = 250;
    std::set<int32_t> eos = {151645, 151643};
};

template <class Decoder>
class MimoServeScheduler {
public:
    using Clock = std::chrono::steady_clock;
    using Now = std::function<Clock::time_point()>;
    using Output = std::function<void(const std::string&)>;

private:
    struct Request {
        uint64_t id = 0;
        int slot = -1, max_new = 0, start = 0, cursor = 0, pos = 0, generated = 0;
        bool cancel = false, prepared = false, decoding = false, host_choice = false;
        int32_t token = 0, draft = 0;
        uint64_t hits = 0, uses = 0, prompt_hits = 0, prompt_uses = 0, decode_hits = 0, decode_uses = 0;
        long long offered = 0, accepted = 0;
        double prompt_ms = 0, prompt_service_ms = 0, decode_service_ms = 0;
        double checkpoint_capture_ms = 0, checkpoint_restore_ms = 0;
        uint64_t checkpoint_captures = 0, checkpoint_restores = 0;
        Clock::time_point began{}, decode_began{};
        std::string finish = "length";
        std::vector<int32_t> prompt;
        Sampler sampler;
        typename Decoder::PromptState prompt_state;
    };
    struct Slot {
        std::vector<int32_t> session;
        // Include speculative and MTP lookahead writes when checking physical ring coverage.
        int written_end = 0;
        std::unique_ptr<Request> request;
    };
    struct MixedEpoch {
        uint64_t prefill_id = 0, decode_id = 0;
        double prefill_ms = 0, decode_ms = 0;
        Clock::time_point last_prefill_done{};
    };

public:
    MimoServeScheduler(Decoder& decoder, ServeOptions options, Output output,
                      Now now = [] { return Clock::now(); })
        : decoder_(decoder), options_(std::move(options)), output_(std::move(output)), slots_(2), now_(std::move(now)) {
        if (options_.prefill_quantum < 1 || options_.decode_service_percent < 1 || options_.decode_service_percent > 99 ||
            options_.prefill_max_wait_ms < 1 || options_.prefill_max_wait_ms > 60000)
            throw std::invalid_argument("invalid prefill/service scheduling options");
    }

    bool active() const {
        for (const auto& slot : slots_) if (slot.request) return true;
        return false;
    }
    bool quitting() const { return quitting_; }

    // Only the compute owner calls this, between complete main/MTP units. IDs increase in GEN send order.
    void command(const std::string& line) {
        if (line.empty()) return;
        if (line == "QUIT") { shutdown(); return; }
        std::istringstream in(line);
        std::string kind, name;
        in >> kind >> name;
        uint64_t id = 0;
        if (!serve_uint(name, id) || id == 0) { output_("ERR 0 bad request: command needs a positive id"); return; }
        if (kind == "STOP") {
            for (auto& slot : slots_) if (slot.request && slot.request->id == id) slot.request->cancel = true;
            return;
        }
        if (kind != "GEN") { event("ERR", id, "unknown command"); return; }
        if (id <= last_id_) throw std::runtime_error("GEN command ids must increase and must not be reused");
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
        request->began = now_();
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
        request->id = id;
        request->slot = (int) slot_id;
        request->host_choice = params.needs_logits();
        request->sampler.begin(params);
        request->sampler.history(request->prompt.data(), n);
        slot.request = std::move(request);
    }

    void shutdown() {
        quitting_ = true;
        for (auto& slot : slots_) if (slot.request) slot.request->cancel = true;
    }
    void fail_all(const std::string& reason) {
        deferred_.clear();
        collecting_ = false;
        for (auto& slot : slots_) {
            if (slot.request) event("ERR", slot.request->id, reason);
            slot.request.reset();
            slot.session.clear();
        }
        mixed_ = {};
    }

    // False means shared compute state is unsafe. The caller must terminate the engine, not serve another slot.
    bool step(std::string& err) {
        for (size_t i = 0; i < slots_.size(); ++i) {
            const size_t index = (next_ + i) % slots_.size();
            auto& slot = slots_[index];
            if (slot.request && slot.request->cancel) {
                slot.request->finish = "cancel";
                finish(slot);
                next_ = (index + 1) % slots_.size();
                return true;
            }
        }
        const int selected = select();
        if (selected < 0) return true;
        auto& slot = slots_[(size_t) selected];
        auto& request = *slot.request;
        const bool prompt = !request.decoding;
        const auto before = decoder_.stats();
        const auto cache_before = decoder_.prefix_checkpoint_stats(selected);
        const auto began = now_();
        collecting_ = true;
        deferred_.clear();
        try {
            if (!quantum(selected, slot, err)) {
                collecting_ = false;
                deferred_.clear();
                return false;
            }
        } catch (...) {
            collecting_ = false;
            deferred_.clear();
            throw;
        }
        const auto ended = now_();
        const double service = milliseconds(began, ended);
        const auto after = decoder_.stats();
        const auto cache_after = decoder_.prefix_checkpoint_stats(selected);
        const uint64_t hits = after.expert_hits - before.expert_hits, uses = after.expert_uses - before.expert_uses;
        request.hits += hits;
        request.uses += uses;
        if (prompt) {
            request.prompt_service_ms += service;
            request.prompt_hits += hits;
            request.prompt_uses += uses;
        } else {
            request.decode_service_ms += service;
            request.decode_hits += hits;
            request.decode_uses += uses;
        }
        request.checkpoint_capture_ms += cache_after.capture_ms - cache_before.capture_ms;
        request.checkpoint_restore_ms += cache_after.restore_ms - cache_before.restore_ms;
        request.checkpoint_captures += cache_after.captures - cache_before.captures;
        request.checkpoint_restores += cache_after.restores - cache_before.restores;
        if (request.id == mixed_.prefill_id) {
            mixed_.prefill_ms += service;
            mixed_.last_prefill_done = ended;
        } else if (request.id == mixed_.decode_id) {
            mixed_.decode_ms += service;
        }
        next_ = ((size_t) selected + 1) % slots_.size();
        collecting_ = false;
        // No stdout or client delay is charged as compute-owner service. Keep each command's event order.
        for (const auto& line : deferred_) output_(line);
        deferred_.clear();
        if (request.decoding && (request.generated >= request.max_new || request.finish != "length")) finish(slot);
        return true;
    }

private:
    static double milliseconds(Clock::time_point began, Clock::time_point ended) {
        return std::max(0.0, std::chrono::duration<double, std::milli>(ended - began).count());
    }
    double elapsed(Clock::time_point began) const { return milliseconds(began, now_()); }

    int select() {
        int prefill = -1, decode = -1, count = 0;
        for (int i = 0; i < (int) slots_.size(); ++i) {
            if (!slots_[(size_t) i].request) continue;
            ++count;
            if (slots_[(size_t) i].request->decoding) decode = i;
            else prefill = i;
        }
        if (count == 2 && prefill >= 0 && decode >= 0) {
            const auto p = slots_[(size_t) prefill].request->id, d = slots_[(size_t) decode].request->id;
            if (mixed_.prefill_id != p || mixed_.decode_id != d) mixed_ = {p, d, 0, 0, now_()};
            if (elapsed(mixed_.last_prefill_done) >= options_.prefill_max_wait_ms) return prefill;
            const double dw = options_.decode_service_percent, pw = 100 - dw;
            return mixed_.decode_ms / dw <= mixed_.prefill_ms / pw ? decode : prefill;
        }
        mixed_ = {};
        for (size_t i = 0; i < slots_.size(); ++i) {
            const size_t index = (next_ + i) % slots_.size();
            if (slots_[index].request) return (int) index;
        }
        return -1;
    }
    void emit(std::string line) {
        if (collecting_) deferred_.push_back(std::move(line));
        else output_(line);
    }
    void event(const char* kind, uint64_t id, const std::string& payload) {
        emit(std::string(kind) + " " + std::to_string(id) + " " + payload);
    }
    void diagnostic(const Request& r, const char* kind, const std::string& fields) {
        emit("INFO request_diag_v=1 command_id=" + std::to_string(r.id) + " diag_kind=" + kind +
             " slot=" + std::to_string(r.slot) + " " + fields);
    }
    std::string cache_fields(int slot) const {
        const auto stats = decoder_.prefix_checkpoint_stats(slot);
        auto boundaries = decoder_.prefix_checkpoint_boundaries(slot);
        std::sort(boundaries.begin(), boundaries.end());
        boundaries.erase(std::unique(boundaries.begin(), boundaries.end()), boundaries.end());
        std::string encoded;
        for (int boundary : boundaries) encoded += (encoded.empty() ? "" : ",") + std::to_string(boundary);
        std::string out = "checkpoint_count=" + std::to_string(stats.count) +
                          " checkpoint_bytes=" + std::to_string(stats.payload_bytes) +
                          " checkpoint_metadata_bytes=" + std::to_string(stats.metadata_bytes) +
                          " checkpoint_boundaries=" + (encoded.empty() ? "none" : encoded);
        out += " checkpoint_skip_reason=" + (stats.last_skip_reason.empty() ? std::string("not_needed") : stats.last_skip_reason);
        return out;
    }
    void token(Request& request, int32_t id) {
        event("T", request.id, std::to_string(id));
        ++request.generated;
        if (options_.eos.count(id)) request.finish = "stop";
    }
    void finish(Slot& slot) {
        auto& r = *slot.request;
        if (!r.decoding) r.prompt_ms = elapsed(r.began);
        diagnostic(r, "timing", "native_prepared=" + std::to_string(r.prepared ? 1 : 0) +
            " live_written_end=" + std::to_string(slot.written_end) +
            " prompt_service_ms=" + std::to_string(r.prompt_service_ms) +
            " decode_service_ms=" + std::to_string(r.decode_service_ms) +
            " prompt_expert_hits=" + std::to_string(r.prompt_hits) + " prompt_expert_uses=" + std::to_string(r.prompt_uses) +
            " decode_expert_hits=" + std::to_string(r.decode_hits) + " decode_expert_uses=" + std::to_string(r.decode_uses) +
            " checkpoint_capture_ms=" + std::to_string(r.checkpoint_capture_ms) +
            " checkpoint_restore_ms=" + std::to_string(r.checkpoint_restore_ms) +
            " checkpoint_captures=" + std::to_string(r.checkpoint_captures) +
            " checkpoint_restores=" + std::to_string(r.checkpoint_restores) +
            " fresh_prompt_tokens=" + std::to_string(r.cursor - r.start) + " " + cache_fields(r.slot));
        char payload[512];
        std::snprintf(payload, sizeof payload, "%d %d %.1f %.1f %s %lld %lld %d %llu %llu 0 0 0.0 %d",
                      r.generated, (int) r.prompt.size(), r.prompt_ms, r.decoding ? elapsed(r.decode_began) : 0.0,
                      r.finish.c_str(), r.accepted, r.offered, r.start, (unsigned long long) r.hits,
                      (unsigned long long) r.uses, r.cursor - r.start);
        event("DONE", r.id, payload);
        slot.request.reset();
    }
    bool prepare(int index, Slot& slot, std::string& err) {
        auto& r = *slot.request;
        const int n = (int) r.prompt.size(), previous_written = slot.written_end;
        int raw_lcp = 0;
        while (raw_lcp < (int) slot.session.size() && raw_lcp < n &&
               slot.session[(size_t) raw_lcp] == r.prompt[(size_t) raw_lcp]) ++raw_lcp;
        // All matching prompt tokens may remain valid, but the last must execute to recover logits. MTP also
        // needs the predecessor of the first differing token to repair its one-ahead row.
        const int boundary = std::max(0, std::min(n - 1, raw_lcp - (decoder_.has_mtp() ? 1 : 0)));
        const bool ring_ok = boundary >= slot.written_end - (options_.ring_capacity - 127);
        int resume = ring_ok ? boundary : 0;
        std::string reason = slot.session.empty() ? "empty_slot" : raw_lcp == 0 ? "no_common_prefix" :
                             boundary == 0 ? "boundary_replay_from_zero" : !ring_ok ? "ring_history_missing" : "retained_live";
        typename Decoder::PrefixCheckpointInfo checkpoint;
        std::string checkpoint_reason;
        if (!decoder_.find_prefix_checkpoint(index, r.prompt.data(), n, checkpoint, checkpoint_reason, err)) return false;
        if (checkpoint.id && checkpoint.replay_pos > resume) {
            typename Decoder::PrefixCheckpointInfo restored;
            if (!decoder_.restore_prefix_checkpoint(index, checkpoint.id, r.prompt.data(), n, restored, err)) return false;
            checkpoint = restored;
            resume = restored.replay_pos;
            slot.written_end = std::max({restored.main_written_end, restored.mtp_written_end, restored.mtp2_written_end});
            reason = "restored_checkpoint";
            checkpoint_reason = "restored";
        } else if (resume > 0 || checkpoint.id) {
            checkpoint_reason = "not_needed";
        } else {
            if (checkpoint_reason.empty()) checkpoint_reason = "no_checkpoint";
        }
        if (resume == 0) {
            if (!decoder_.reset_sequence(index, err)) return false;
            slot.written_end = 0;
        }
        r.start = r.cursor = resume;
        slot.session.assign(r.prompt.begin(), r.prompt.begin() + resume);
        r.prepared = true;
        diagnostic(r, "prefix", "raw_lcp=" + std::to_string(raw_lcp) + " boundary_keep=" + std::to_string(boundary) +
            " written_end=" + std::to_string(previous_written) + " resume=" + std::to_string(resume) +
            " prefix_reason=" + reason + " checkpoint_boundary=" + std::to_string(checkpoint.committed_end) +
            " checkpoint_id=" + std::to_string(checkpoint.id) + " checkpoint_reason=" + checkpoint_reason);
        event("RESUME", r.id, std::to_string(resume));
        return true;
    }
    bool quantum(int index, Slot& slot, std::string& err) {
        auto& r = *slot.request;
        const int n = (int) r.prompt.size(), vocab = decoder_.vocab();
        if (!r.prepared && !prepare(index, slot, err)) return false;
        if (r.cursor < n) {
            bool peer_decoding = false;
            for (const auto& peer : slots_)
                if (&peer != &slot && peer.request && peer.request->decoding) peer_decoding = true;
            const int quantum = peer_decoding ? std::min(options_.prefill_quantum, 8) : options_.prefill_quantum;
            int count = std::min(quantum, n - r.cursor);
            // A progress callback is not a safe checkpoint boundary. End a whole prompt call at each milestone.
            for (int boundary = 4096; boundary <= 262144; boundary *= 2) {
                if (boundary > r.cursor) { count = std::min(count, boundary - r.cursor); break; }
            }
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
            const bool milestone = r.cursor >= 4096 && r.cursor <= 262144 && (r.cursor & (r.cursor - 1)) == 0;
            if (final || milestone) {
                bool captured = false;
                if (!decoder_.capture_prefix_checkpoint(index, r.cursor, final, captured, err)) return false;
                if (captured) diagnostic(r, "cache", cache_fields(index));
            }
            const double ms = elapsed(r.began);
            event("PP", r.id, std::to_string(r.cursor) + " " + std::to_string(n) + " " + std::to_string(ms) + " " +
                  std::to_string(ms > 0 ? (r.cursor - r.start) / ms * 1000.0 : 0.0));
            if (final) {
                r.prompt_ms = ms;
                r.decode_began = now_();
                r.decoding = true;
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
            if (decoder_.has_mtp() && !decoder_.mtp_forward(index, &r.token, 1, r.pos, &r.draft, err)) return false;
            slot.written_end = std::max(slot.written_end, r.pos + (decoder_.has_mtp() ? 1 : 0));
        }
        return true;
    }

    Decoder& decoder_;
    ServeOptions options_;
    Output output_;
    std::vector<Slot> slots_;
    Now now_;
    MixedEpoch mixed_;
    std::vector<std::string> deferred_;
    size_t next_ = 0;
    uint64_t last_id_ = 0;
    bool collecting_ = false, quitting_ = false;
};

}  // namespace strata::mimo
