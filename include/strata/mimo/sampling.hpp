// include/strata/mimo/sampling.hpp - token choice on the host for strata-mimo --serve: the sampling keys of Strata's
// server (serve/server.py sampling_keys: temperature, top_p, top_k <= 64, min_p, repeat / frequency / presence
// penalties over the last N tokens, seed), and the speculative step's exact acceptance rule.
//
// Order, as llama.cpp's sampler chain applies them: penalties on the raw logits -> the top_k best -> temperature
// -> softmax -> top_p (smallest set holding top_p of the mass) -> min_p (drop p < min_p * p_max) -> renormalize.
// temperature <= 0 is greedy (the argmax after penalties).
//
// Speculative decoding with a deterministic draft d (the MTP layer's argmax): accept d with probability p(d); on a
// rejection draw from p with d removed and renormalized.  The output then has exactly the distribution p of plain
// sampling (the standard speculative-sampling rule with q = a point mass on d).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <random>
#include <unordered_map>
#include <utility>
#include <vector>

namespace strata::mimo {

struct SamplingParams {
    float temperature = 0.f, top_p = 1.f, min_p = 0.f;
    int top_k = 64;
    float penalty_repeat = 1.f, penalty_freq = 0.f, penalty_present = 0.f;
    int penalty_last_n = 64;
    uint64_t seed = 0;
    bool greedy() const { return temperature <= 0.f; }
    bool penalties() const { return penalty_repeat != 1.f || penalty_freq != 0.f || penalty_present != 0.f; }
    /// The logits are needed on the host (sampling or penalties); greedy without penalties uses the GPU's argmax.
    bool needs_logits() const { return !greedy() || penalties(); }
};

class Sampler {
public:
    /// A new request: its parameters, and the random stream (seeded by `seed`, or randomly when it is 0).
    void begin(const SamplingParams& p) {
        p_ = p;
        rng_.seed(p.seed ? p.seed : std::random_device{}());
    }
    /// The tokens before the position being sampled (the penalties look at the last penalty_last_n of them).
    void history(const int32_t* h, int n) {
        hist_.clear();
        if (!p_.penalties()) return;
        for (int i = std::max(0, n - std::max(0, p_.penalty_last_n)); i < n; ++i) hist_.push_back(h[i]);
    }

    /// The candidate distribution of one position (ids and probabilities, summing to 1; greedy: one candidate).
    void dist(const float* logits, int V, std::vector<std::pair<int32_t, float>>& out) const {
        constexpr int kMaxK = 64;
        const int k = std::clamp(p_.top_k <= 0 ? kMaxK : p_.top_k, 1, kMaxK);
        std::unordered_map<int32_t, int> counts;
        if (p_.penalties())
            for (int32_t id : hist_) ++counts[id];
        auto value = [&](int32_t id) {
            float l = logits[id];
            if (!counts.empty()) {
                const auto it = counts.find(id);
                if (it != counts.end()) {
                    if (p_.penalty_repeat != 1.f) l = l > 0 ? l / p_.penalty_repeat : l * p_.penalty_repeat;
                    l -= (float) it->second * p_.penalty_freq + p_.penalty_present;
                }
            }
            return l;
        };
        // the k largest in one pass (a min-heap of k)
        std::vector<std::pair<float, int32_t>> heap;
        heap.reserve(k + 1);
        auto cmp = [](const std::pair<float, int32_t>& a, const std::pair<float, int32_t>& b) { return a.first > b.first; };
        for (int32_t id = 0; id < V; ++id) {
            const float l = counts.empty() ? logits[id] : value(id);
            if ((int) heap.size() < k) {
                heap.emplace_back(l, id);
                std::push_heap(heap.begin(), heap.end(), cmp);
            } else if (l > heap.front().first) {
                std::pop_heap(heap.begin(), heap.end(), cmp);
                heap.back() = {l, id};
                std::push_heap(heap.begin(), heap.end(), cmp);
            }
        }
        std::sort(heap.begin(), heap.end(), [](const auto& a, const auto& b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
        out.clear();
        if (p_.greedy()) {
            out.emplace_back(heap.front().second, 1.f);
            return;
        }
        const float t = p_.temperature, top = heap.front().first;
        double sum = 0;
        for (auto& [l, id] : heap) {
            const double e = std::exp((double) (l - top) / t);
            out.emplace_back(id, (float) e);
            sum += e;
        }
        for (auto& c : out) c.second = (float) (c.second / sum);
        if (p_.top_p < 1.f) {
            double cum = 0;
            size_t keep = 0;
            while (keep < out.size()) {
                cum += out[keep++].second;
                if (cum >= p_.top_p) break;
            }
            out.resize(std::max<size_t>(1, keep));
        }
        if (p_.min_p > 0.f) {
            const float cut = p_.min_p * out.front().second;
            size_t keep = 1;
            while (keep < out.size() && out[keep].second >= cut) ++keep;
            out.resize(keep);
        }
        normalize(out);
    }

    int32_t sample(const float* logits, int V) {
        std::vector<std::pair<int32_t, float>> d;
        dist(logits, V, d);
        return draw(d);
    }

    /// The speculative rule for draft `d` at a position with these logits: true = keep d; false = *other is the
    /// token drawn instead (from p without d).
    bool accept_draft(const float* logits, int V, int32_t d, int32_t* other) {
        std::vector<std::pair<int32_t, float>> p;
        dist(logits, V, p);
        float pd = 0.f;
        for (auto& c : p)
            if (c.first == d) pd = c.second;
        if (pd > 0.f && std::uniform_real_distribution<float>(0.f, 1.f)(rng_) < pd) return true;
        p.erase(std::remove_if(p.begin(), p.end(), [&](const auto& c) { return c.first == d; }), p.end());
        if (p.empty()) {   // greedy, and the draft was the argmax: cannot be rejected
            return true;
        }
        normalize(p);
        *other = draw(p);
        return false;
    }

private:
    static void normalize(std::vector<std::pair<int32_t, float>>& v) {
        double s = 0;
        for (auto& c : v) s += c.second;
        for (auto& c : v) c.second = (float) (c.second / s);
    }
    int32_t draw(const std::vector<std::pair<int32_t, float>>& d) {
        float u = std::uniform_real_distribution<float>(0.f, 1.f)(rng_);
        for (auto& c : d) {
            if (u < c.second) return c.first;
            u -= c.second;
        }
        return d.back().first;
    }

    SamplingParams p_;
    std::mt19937_64 rng_;
    std::deque<int32_t> hist_;
};

}  // namespace strata::mimo
