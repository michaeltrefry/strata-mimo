// CPU-only ownership/validity policy for MiMo's pageable SWA/MTP checkpoints.
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::mimo {

struct PrefixCheckpointInfo {
    uint64_t id = 0;
    int committed_end = 0, replay_pos = 0;
    int main_written_end = 0, mtp_written_end = 0, mtp2_written_end = 0;
    size_t host_bytes = 0, metadata_bytes = 0;
};

struct PrefixCheckpointStats {
    size_t count = 0, payload_bytes = 0, metadata_bytes = 0;
    uint64_t captures = 0, restores = 0, skipped = 0, invalidations = 0;
    double capture_ms = 0, restore_ms = 0;
    std::string last_skip_reason;
};

namespace detail {
struct PrefixCheckpointPolicy {
    static constexpr size_t kMaxEntries = 8;
    static constexpr size_t kHostBudget = size_t(512) << 20;
    struct Entry {
        PrefixCheckpointInfo info;
        std::vector<int32_t> prefix;
        std::unique_ptr<unsigned char[]> payload;  // ordinary pageable new[], never cudaHostAlloc
        bool milestone = false, prompt_end = false;
        size_t metadata_bytes() const { return sizeof(Entry) + prefix.capacity() * sizeof(int32_t); }
        bool matches(const int32_t* target, int n) const {
            return target && n >= info.committed_end &&
                   prefix.size() == (size_t) info.committed_end &&
                   std::equal(prefix.begin(), prefix.end(), target);
        }
    };
    std::array<std::unique_ptr<Entry>, kMaxEntries> entries{};
    // Only [0,journal.size()) is known to have executed contiguously; never fill uninitialized holes.
    std::vector<int32_t> journal;
    int prompt_boundary = -1;
    bool prompt_final = false, journal_enabled = true;
    int main_written_end = 0, mtp_written_end = 0, mtp2_written_end = 0;
    PrefixCheckpointStats counters;

    static bool fits_budget(size_t retained, size_t payload, size_t metadata) {
        return retained <= kHostBudget && payload <= kHostBudget - retained &&
               metadata <= kHostBudget - retained - payload;
    }
    static bool milestone(int end) {
        return end >= 4096 && end <= 262144 && (end & (end - 1)) == 0;
    }
    void reserve_journal(int max_ctx) {
        // Two physical slots share the budget. Oversized contexts may run without checkpoint journaling.
        if (max_ctx < 1 || (size_t) max_ctx > kHostBudget / 4 / sizeof(int32_t)) {
            journal_enabled = false; return;
        }
        try { journal.reserve((size_t) max_ctx); }
        catch (const std::bad_alloc&) { journal_enabled = false; }
    }
    void clear_boundary() { prompt_boundary = -1; prompt_final = false; }
    // This must precede EVERY main enqueue, including identical-token rewrites and batched paths.
    void before_main(int pos, int n) {
        clear_boundary();
        for (auto& e : entries)
            if (e && e->info.replay_pos > pos) { e.reset(); ++counters.invalidations; }
        if ((size_t) pos < journal.size()) journal.resize((size_t) pos);
        main_written_end = std::max(main_written_end, pos + n);
    }
    void after_main(const int32_t* tokens, int pos, int n) {
        if (journal_enabled && (size_t) pos == journal.size())
            journal.insert(journal.end(), tokens, tokens + n);  // capacity reserved before execution
    }
    void before_mtp(int end, bool second, bool second_only = false) {
        clear_boundary();
        if (!second_only) mtp_written_end = std::max(mtp_written_end, end);
        if (second) mtp2_written_end = std::max(mtp2_written_end, end);
    }
    void finish_prompt(int end, bool final) {
        if (journal_enabled && journal.size() == (size_t) end) {
            prompt_boundary = end;
            prompt_final = final;
        }
    }
    void reset() {
        for (auto& e : entries) { if (e) ++counters.invalidations; e.reset(); }
        journal.clear();  // bounded reserved capacity, reused by the next generation
        clear_boundary();
        main_written_end = mtp_written_end = mtp2_written_end = 0;
    }
    bool eligible(int end, bool final, std::string& reason) const {
        if (end < 1 || prompt_boundary != end || journal.size() != (size_t) end) {
            reason = "not_prompt_boundary"; return false;
        }
        if (final && !prompt_final) { reason = "not_final_prompt_boundary"; return false; }
        if (!final && !milestone(end)) { reason = "not_milestone"; return false; }
        return true;
    }
    const Entry* find(const int32_t* target, int n) const {
        const Entry* best = nullptr;
        for (const auto& e : entries)
            if (e && e->matches(target, n) && (!best || e->info.replay_pos > best->info.replay_pos)) best = e.get();
        return best;
    }
    Entry* by_id(uint64_t id) {
        for (auto& e : entries) if (e && e->info.id == id) return e.get();
        return nullptr;
    }
    PrefixCheckpointStats stats() const {
        PrefixCheckpointStats s = counters;
        s.count = s.payload_bytes = 0;
        s.metadata_bytes = sizeof(*this) + journal.capacity() * sizeof(int32_t) + counters.last_skip_reason.capacity();
        for (const auto& e : entries) if (e) {
            ++s.count;
            s.payload_bytes += e->info.host_bytes;
            s.metadata_bytes += e->metadata_bytes();
        }
        return s;
    }
    // Publish only after all copies and synchronization succeed. Seven possible milestones plus one latest end.
    void publish(std::unique_ptr<Entry> entry) {
        for (auto& e : entries) if (e) {
            if (e->info.committed_end == entry->info.committed_end) e.reset();
            else if (entry->prompt_end && e->prompt_end) {
                e->prompt_end = false;
                if (!e->milestone) e.reset();
            }
        }
        for (auto& e : entries) if (!e) { e = std::move(entry); return; }
    }
    void restored(const Entry& e) {
        // The protected GLOBAL prefix belongs to this checkpoint even if an earlier restore truncated the journal.
        journal.assign(e.prefix.begin(), e.prefix.begin() + e.info.replay_pos);
        main_written_end = e.info.main_written_end;
        mtp_written_end = e.info.mtp_written_end;
        mtp2_written_end = e.info.mtp2_written_end;
        clear_boundary();
        ++counters.restores;
    }
};
}  // namespace detail
}  // namespace strata::mimo
