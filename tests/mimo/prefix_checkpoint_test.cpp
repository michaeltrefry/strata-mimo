// CPU-only tests: no CUDA headers, devices, model files or timing assumptions.
#include "strata/mimo/prefix_checkpoint.hpp"

#include <cstdio>
#include <limits>
#include <stdexcept>

using Policy = strata::mimo::detail::PrefixCheckpointPolicy;
namespace {
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
std::unique_ptr<Policy::Entry> entry(uint64_t id, int end, bool final = false, size_t bytes = 17) {
    auto e = std::make_unique<Policy::Entry>();
    e->info.id = id;
    e->info.committed_end = end;
    e->info.replay_pos = end - 1;
    e->info.main_written_end = end + 19;  // older speculative writes are NOT the logical committed end
    e->info.mtp_written_end = end + 23;
    e->info.mtp2_written_end = end + 24;
    e->info.host_bytes = bytes;
    e->prefix.resize((size_t) end);
    for (int i = 0; i < end; ++i) e->prefix[(size_t) i] = 32 + i % 200;
    e->payload.reset(new unsigned char[bytes]);
    e->info.metadata_bytes = e->metadata_bytes();
    e->milestone = Policy::milestone(end);
    e->prompt_end = final;
    return e;
}
void journal_and_eligibility() {
    Policy p;
    p.reserve_journal(1024);
    const int32_t tokens[] = {1, 2, 3, 4};
    std::string reason;
    p.before_main(2, 2);
    p.after_main(tokens, 2, 2);
    p.finish_prompt(4, true);
    require(p.journal.empty() && !p.eligible(4, true, reason), "a hole cannot authorize capture");
    p.before_main(0, 4);
    p.after_main(tokens, 0, 4);
    p.finish_prompt(4, false);
    require(!p.eligible(4, true, reason), "caller cannot claim a final boundary");
    require(!p.eligible(4, false, reason), "nonmilestone chunk must not create a decode cadence");
    p.finish_prompt(4, true);
    require(p.eligible(4, true, reason), "successful final prompt boundary missing");
    p.before_mtp(5, true);
    require(!p.eligible(4, true, reason), "manual MTP must remove capture eligibility");
    p.finish_prompt(4, true);
    p.before_mtp(6, true, true);
    require(!p.eligible(4, true, reason), "manual second draft must remove capture eligibility");
    p.finish_prompt(4, true);
    p.before_main(3, 1);
    p.after_main(tokens + 3, 3, 1);
    require(!p.eligible(4, true, reason), "identical manual main write must remove eligibility");
    p.finish_prompt(4, true);
    require(p.eligible(4, true, reason), "complete self replay can establish a new prompt boundary");
}
void identity_and_invalidation() {
    Policy p, peer;
    p.reserve_journal(1024);
    auto e = entry(1, 300, true);
    std::vector<int32_t> target = e->prefix;
    target.push_back(987);  // old MTP lookahead C is deliberately not part of authorization
    p.publish(std::move(e));
    require(p.find(target.data(), (int) target.size())->info.id == 1, "exact prefix including A must match");
    target[299] ^= 1;
    require(!p.find(target.data(), (int) target.size()), "anchor token mismatch cannot be authorized by hash/LCP");
    target[299] ^= 1;
    require(!p.find(target.data(), 299), "short target missing replay token must miss");
    require(!peer.by_id(1), "wrong slot must not resolve an opaque handle");
    p.before_main(300, 300);  // ring age >256 is NOT global-prefix invalidation
    require(p.by_id(1), "append must preserve a checkpoint after its live ring ages out");
    p.before_main(299, 1);
    require(p.by_id(1), "replay at the checkpoint's own anchor must preserve it");
    p.before_main(298, 1);  // even the SAME token changes window arithmetic; global prefix is no longer protected
    require(!p.by_id(1) && p.counters.invalidations == 1, "earlier identical rewrite must invalidate BEFORE enqueue");
    peer.publish(entry(2, 100, true));
    p.reset();
    require(peer.by_id(2), "peer reset/write must not change another slot's ownership");
}
void restore_watermarks() {
    Policy p;
    p.reserve_journal(1024);
    p.publish(entry(3, 300, true));
    const auto* e = p.by_id(3);
    p.main_written_end = 900;
    p.mtp_written_end = 901;
    p.mtp2_written_end = 902;
    p.finish_prompt(900, true);
    p.restored(*e);
    require(p.journal.size() == 299 && p.journal.back() == e->prefix[298], "restore journal must stop at protected A");
    require(p.main_written_end == 319 && p.mtp_written_end == 323 && p.mtp2_written_end == 324,
            "restore must recover actual snapshot watermarks, not C or current live maxima");
    require(p.prompt_boundary < 0, "restore cannot authorize capture or reuse hidden rows");
    p.before_main(299, 1);
    p.after_main(e->prefix.data() + 299, 299, 1);
    p.finish_prompt(300, true);
    require(p.journal.size() == 300 && p.by_id(3), "self-anchor replay must rebuild journal while preserving snapshot");
    p.counters.captures = 7;
    p.counters.capture_ms = 1.25;
    p.reset();
    require(!p.by_id(3) && p.journal.empty() && !p.main_written_end && !p.mtp_written_end && !p.mtp2_written_end,
            "reset must destroy the generation's authorization and watermarks");
    require(p.counters.captures == 7 && p.counters.restores == 1 && p.counters.capture_ms == 1.25,
            "request diagnostic counters must remain monotonic across reset");
}
void bounded_retention_and_budget() {
    Policy p;
    p.reserve_journal(262144);
    uint64_t id = 1;
    for (int end = 4096; end <= 262144; end *= 2) p.publish(entry(id++, end));
    p.publish(entry(id++, 260001, true));
    require(p.stats().count == 8, "seven milestones plus latest prompt end must fit exactly eight entries");
    for (int i = 0; i < 100; ++i) {
        p.publish(entry(id++, 260002 + i, true));
        require(p.stats().count == 8, "replacement of latest prompt end must not grow retained count");
    }
    const size_t before = p.stats().metadata_bytes;
    p.publish(entry(id++, 262144, true));
    require(p.stats().count == 7, "coincident latest/milestone must deduplicate");
    require(p.stats().metadata_bytes < before, "removed token vectors must free retained capacity");
    auto best = p.find(p.by_id(id - 1)->prefix.data(), 262144);
    require(best && best->info.committed_end == 262144, "lookup must select largest matching anchor");
    auto stats = p.stats();
    require(stats.payload_bytes == 7 * 17 && stats.metadata_bytes >= p.journal.capacity() * sizeof(int32_t),
            "usage must include actual retained payload and journal/token capacity");
    constexpr size_t budget = Policy::kHostBudget;
    require(Policy::fits_budget(budget - 20, 17, 3), "exact budget bound must fit");
    require(!Policy::fits_budget(budget - 20, 17, 4), "metadata cannot escape the aggregate budget");
    require(!Policy::fits_budget(budget, 1, 0), "old+in-flight payload must fit without evicting a fallback");
    require(!Policy::fits_budget(std::numeric_limits<size_t>::max(), 1, 1), "budget arithmetic must not overflow");
    require(!Policy::milestone(2048) && !Policy::milestone(524288) && !Policy::milestone(4097), "milestones must be 4K..256K powers of two");
    Policy oversized;
    oversized.reserve_journal(std::numeric_limits<int>::max());
    require(!oversized.journal_enabled && oversized.journal.capacity() == 0, "oversized journal must not bypass host budget");
}
}  // namespace
int main() {
    try {
        journal_and_eligibility();
        identity_and_invalidation();
        restore_watermarks();
        bounded_retention_and_budget();
        std::puts("PASS MiMo checkpoint policy: exact identity, bounded ownership, protected global prefix and journal");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
