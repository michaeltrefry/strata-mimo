#include "strata/mimo/model.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace strata::mimo {
namespace {
constexpr size_t kAlign = 256;
size_t align_up(size_t n) { return (n + kAlign - 1) / kAlign * kAlign; }
}  // namespace

MimoModel::MimoModel(const std::string& any_shard) : gguf_(std::make_unique<GgufModel>(GgufModel::open(any_shard))) {
    std::string err = read_mimo_geometry(*gguf_, geo_);
    if (err.empty()) err = check_mimo_tensors(*gguf_, geo_);
    if (!err.empty()) throw std::runtime_error("MiMo model " + any_shard + ": " + err);
    layers_.resize((size_t) geo_.n_layer);
    experts_.resize((size_t) geo_.n_layer);
    static const char* names[3] = {"ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"};
    for (int l = 0; l < geo_.n_layer; ++l) {
        if (!geo_.is_moe(l)) continue;
        ExpertSlices& s = experts_[(size_t) l];
        for (int m = 0; m < 3; ++m) {
            const TensorInfo* t = nullptr;
            s.base[m] = host_data("blk." + std::to_string(l) + "." + names[m] + ".weight", &t);
            const uint64_t bytes = tensor_payload_bytes(*t);
            s.stride[m] = (size_t) (bytes / t->shape[2]);
            s.row_bytes[m] = (size_t) (s.stride[m] / t->shape[1]);
        }
    }
}

MimoModel::~MimoModel() {
    if (arena_) cudaFree(arena_);
}

const uint8_t* MimoModel::host_data(const std::string& name, const TensorInfo** info) const {
    size_t shard = 0;
    const TensorInfo* t = gguf_->find(name, &shard);
    if (info) *info = t;
    return t ? gguf_->shard(shard).tensor_data(*t) : nullptr;
}

size_t MimoModel::expert_bytes() const {
    for (const auto& s : experts_)
        if (s.base[0]) return s.expert_bytes();
    return 0;
}

bool MimoModel::upload(std::string& err) {
    if (arena_) { err = "MiMo weights already uploaded"; return false; }
    // every tensor the decode path reads from VRAM, in arena order
    std::vector<std::pair<std::string, DevTensor*>> plan;
    plan.emplace_back("token_embd.weight", &token_embd_);
    plan.emplace_back("output.weight", &output_);
    plan.emplace_back("output_norm.weight", &output_norm_);
    for (int l = 0; l < geo_.n_layer; ++l) {
        const std::string b = "blk." + std::to_string(l) + ".";
        LayerWeights& w = layers_[(size_t) l];
        plan.emplace_back(b + "attn_norm.weight", &w.attn_norm);
        plan.emplace_back(b + "attn_qkv.weight", &w.attn_qkv);
        plan.emplace_back(b + "attn_output.weight", &w.attn_output);
        plan.emplace_back(b + "ffn_norm.weight", &w.ffn_norm);
        if (geo_.is_swa(l)) plan.emplace_back(b + "attn_sinks.weight", &w.attn_sinks);
        if (geo_.is_moe(l)) {
            plan.emplace_back(b + "ffn_gate_inp.weight", &w.gate_inp);
            plan.emplace_back(b + "exp_probs_b.bias", &w.exp_probs_b);
        } else {
            plan.emplace_back(b + "ffn_gate.weight", &w.ffn_gate);
            plan.emplace_back(b + "ffn_up.weight", &w.ffn_up);
            plan.emplace_back(b + "ffn_down.weight", &w.ffn_down);
        }
    }
    size_t total = 0;
    std::vector<std::pair<const TensorInfo*, const uint8_t*>> src(plan.size());
    for (size_t i = 0; i < plan.size(); ++i) {
        src[i].second = host_data(plan[i].first, &src[i].first);
        if (!src[i].second) { err = "missing tensor " + plan[i].first; return false; }
        total += align_up((size_t) tensor_payload_bytes(*src[i].first));
    }
    void* arena = nullptr;
    cudaError_t st = cudaMalloc(&arena, total);
    if (st != cudaSuccess) {
        err = "MiMo weights: cudaMalloc of " + std::to_string(total >> 20) + " MiB failed: " + cudaGetErrorString(st);
        return false;
    }
    // copy through a pinned staging buffer: the mapping is pageable and a direct cudaMemcpy from it is slow
    constexpr size_t kStage = 64u << 20;
    void* stage[2] = {nullptr, nullptr};
    cudaStream_t stream = nullptr;
    cudaEvent_t done[2] = {nullptr, nullptr};
    auto cleanup = [&] {
        for (int i = 0; i < 2; ++i) {
            if (done[i]) cudaEventDestroy(done[i]);
            if (stage[i]) cudaFreeHost(stage[i]);
        }
        if (stream) cudaStreamDestroy(stream);
    };
    st = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    for (int i = 0; i < 2 && st == cudaSuccess; ++i) {
        st = cudaMallocHost(&stage[i], kStage);
        if (st == cudaSuccess) st = cudaEventCreateWithFlags(&done[i], cudaEventDisableTiming);
    }
    size_t off = 0;
    int buf = 0;
    bool used[2] = {false, false};
    for (size_t i = 0; i < plan.size() && st == cudaSuccess; ++i) {
        const TensorInfo& t = *src[i].first;
        const size_t bytes = (size_t) tensor_payload_bytes(t);
        uint8_t* dst = (uint8_t*) arena + off;
        for (size_t done_bytes = 0; done_bytes < bytes && st == cudaSuccess;) {
            const size_t n = (std::min)(kStage, bytes - done_bytes);
            if (used[buf]) st = cudaEventSynchronize(done[buf]);
            if (st != cudaSuccess) break;
            std::memcpy(stage[buf], src[i].second + done_bytes, n);
            st = cudaMemcpyAsync(dst + done_bytes, stage[buf], n, cudaMemcpyHostToDevice, stream);
            if (st == cudaSuccess) st = cudaEventRecord(done[buf], stream);
            used[buf] = true;
            buf ^= 1;
            done_bytes += n;
        }
        DevTensor& d = *plan[i].second;
        d.data = dst;
        d.type = t.type;
        d.ne0 = (int64_t) t.shape[0];
        d.ne1 = t.shape.size() > 1 ? (int64_t) t.shape[1] : 1;
        d.bytes = bytes;
        off += align_up(bytes);
    }
    if (st == cudaSuccess) st = cudaStreamSynchronize(stream);
    cleanup();
    if (st != cudaSuccess) {
        cudaFree(arena);
        for (auto& p : plan) *p.second = DevTensor{};
        err = std::string("MiMo weights upload: ") + cudaGetErrorString(st);
        return false;
    }
    arena_ = arena;
    arena_bytes_ = total;
    return true;
}

}  // namespace strata::mimo
