// moe-trace: record which experts each MoE layer selects for every token.
//
// Input (-f): a text file of documents separated by lines containing only "<<<DOC>>>".
// Each document is evaluated from an empty KV cache (teacher-forced), in n_ubatch chunks up to n_ctx tokens.
// Output (MOE_TRACE_OUT env var): binary records, little-endian:
//   header: "MOETRACE" u32 version=1, u32 n_layer, u32 n_expert_used
//   per token: u32 doc, u32 pos, then for each layer: n_expert_used x i16 expert id (-1 = dense layer),
//              n_expert_used x f32 router weight (before normalization)
#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml-backend.h"

#include <clocale>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

struct trace_state {
    int n_used = 0;
    std::map<int, std::vector<int32_t>> ids;   // layer -> [n_tokens * n_used]
    std::map<int, std::vector<float>>   wts;
};

static int layer_of(const char * name, const char * prefix) {
    size_t n = strlen(prefix);
    if (strncmp(name, prefix, n) != 0 || name[n] != '-') {
        return -1;
    }
    return atoi(name + n + 1);
}

// copy a possibly strided (view) tensor of 4-byte elements into a dense [ne1*ne2][ne0] buffer
template <typename T>
static std::vector<T> read_rows(const struct ggml_tensor * t) {
    std::vector<T> buf(ggml_nelements(t));
    const size_t row = (size_t) t->ne[0] * sizeof(T);
    size_t r = 0;
    for (int64_t i2 = 0; i2 < t->ne[2]; ++i2) {
        for (int64_t i1 = 0; i1 < t->ne[1]; ++i1, ++r) {
            ggml_backend_tensor_get(t, buf.data() + r * t->ne[0], i1 * t->nb[1] + i2 * t->nb[2], row);
        }
    }
    return buf;
}

static bool trace_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * st = (trace_state *) user_data;
    int il_ids = layer_of(t->name, "ffn_moe_topk");
    int il_wts = layer_of(t->name, "ffn_moe_weights");
    if (ask) {
        return il_ids >= 0 || il_wts >= 0;
    }
    if (il_ids >= 0) {
        st->n_used = (int) t->ne[0];
        st->ids[il_ids] = read_rows<int32_t>(t);
    } else if (il_wts >= 0) {
        st->wts[il_wts] = read_rows<float>(t);  // [1, n_used, n_tokens] -> dense n_used per token
    }
    return true;
}

static std::vector<std::string> read_docs(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    std::vector<std::string> docs;
    std::string line, cur;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line == "<<<DOC>>>") {
            if (!cur.empty()) {
                docs.push_back(cur);
            }
            cur.clear();
        } else {
            cur += line;
            cur += '\n';
        }
    }
    if (!cur.empty()) {
        docs.push_back(cur);
    }
    return docs;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    const char * out_path = getenv("MOE_TRACE_OUT");
    if (!out_path || params.prompt_file.empty()) {
        LOG_ERR("set MOE_TRACE_OUT and pass the documents with -f\n");
        return 1;
    }

    llama_backend_init();
    trace_state st;
    params.cb_eval = trace_cb;
    params.cb_eval_user_data = &st;
    params.warmup = false;

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();
    if (!model || !ctx) {
        LOG_ERR("failed to init\n");
        return 1;
    }
    const int n_layer = llama_model_n_layer(model);
    const int n_ctx   = llama_n_ctx(ctx);
    const int n_ub    = (int) llama_n_ubatch(ctx);
    const bool add_bos = llama_vocab_get_add_bos(llama_model_get_vocab(model));

    std::vector<std::string> docs = read_docs(params.prompt_file);
    FILE * out = fopen(out_path, "wb");
    if (!out) {
        LOG_ERR("cannot open %s\n", out_path);
        return 1;
    }
    bool header_done = false;
    size_t total = 0;
    const int64_t t0 = ggml_time_us();

    for (size_t d = 0; d < docs.size(); ++d) {
        std::vector<llama_token> toks = common_tokenize(ctx, docs[d], add_bos, true);
        if ((int) toks.size() > n_ctx) {
            toks.resize(n_ctx);
        }
        llama_memory_clear(llama_get_memory(ctx), true);
        for (int p0 = 0; p0 < (int) toks.size(); p0 += n_ub) {
            const int n = std::min(n_ub, (int) toks.size() - p0);
            st.ids.clear();
            st.wts.clear();
            // every token is an output, otherwise the last layer only runs for the final token
            common_batch batch(ctx);
            for (int i = 0; i < n; ++i) {
                batch.add(toks[p0 + i], p0 + i, 0, true);
            }
            if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
                LOG_ERR("decode failed (doc %zu pos %d)\n", d, p0);
                return 1;
            }
            const int k = st.n_used;
            if (!header_done) {
                uint32_t v = 1, nl = (uint32_t) n_layer, nk = (uint32_t) k;
                fwrite("MOETRACE", 1, 8, out);
                fwrite(&v, 4, 1, out);
                fwrite(&nl, 4, 1, out);
                fwrite(&nk, 4, 1, out);
                header_done = true;
            }
            for (auto & [il, v] : st.ids) {
                if (v.size() != (size_t) n * k || st.wts[il].size() != (size_t) n * k) {
                    LOG_ERR("layer %d traced %zu ids / %zu weights for %d tokens\n", il, v.size(), st.wts[il].size(), n);
                    return 1;
                }
            }
            std::vector<int16_t> ids(k);
            std::vector<float> w(k);
            for (int i = 0; i < n; ++i) {
                uint32_t dd = (uint32_t) d, pos = (uint32_t) (p0 + i);
                fwrite(&dd, 4, 1, out);
                fwrite(&pos, 4, 1, out);
                for (int il = 0; il < n_layer; ++il) {
                    auto it = st.ids.find(il);
                    auto iw = st.wts.find(il);
                    for (int j = 0; j < k; ++j) {
                        ids[j] = it == st.ids.end() ? -1 : (int16_t) it->second[(size_t) i * k + j];
                        w[j]   = iw == st.wts.end() ? 0.0f : iw->second[(size_t) i * k + j];
                    }
                    fwrite(ids.data(), 2, k, out);
                    fwrite(w.data(), 4, k, out);
                }
            }
            total += n;
        }
        const double s = (ggml_time_us() - t0) / 1e6;
        LOG_INF("doc %zu/%zu: %zu tokens traced, %.1f tok/s\n", d + 1, docs.size(), total, total / s);
    }
    fclose(out);
    llama_backend_free();
    return 0;
}
