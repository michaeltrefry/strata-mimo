// mimo-dump: write llama.cpp's intermediate tensors for MiMo (arch mimo2) on a fixed token sequence, as the
// reference for Strata-MiMo's parity checks (Strata-MiMo/tools/mimo/mimo_parity.py).
//
// usage: MIMO_DUMP_DIR=<dir> llama-mimo-dump -m <model> -p "<prompt>" [-ngl 99 -ncmoe N ...]
//        (or MIMO_DUMP_TOKENS=<file of int32 token ids> instead of -p)
// writes <dir>/tokens.i32 and one <name>.f32 / .i32 per captured tensor (<name> = llama.cpp's, e.g. "l_out-17"),
// each a raw little-endian array of ne0 x ne1 (x ne2) values, plus <dir>/index.txt: "name type ne0 ne1 ne2".
// Every token is an output, so the last layer runs for all of them.  Use a small -ub (e.g. 8): each ubatch appends
// its rows, and llama.cpp's large-batch CPU MoE path for MXFP4 is not accurate (11-30% off on 311-token batches,
// 0.9% at 4 tokens, measured against an exact float computation).
#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml-backend.h"

#include <clocale>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <set>
#include <string>
#include <vector>

static const char * kCapture[] = {
    "attn_norm", "wqkv", "Qcur", "Kcur", "Vcur", "attn_out", "attn_out_scaled", "ffn_inp", "ffn_norm",
    "ffn_moe_logits", "ffn_moe_topk", "ffn_moe_weights_norm", "ffn_moe_out", "ffn_out", "l_out",
    "result_norm", "result_output", "inp_embd", "h_nextn",
};

struct dump_state {
    std::string dir;
    FILE * index = nullptr;
    std::set<std::string> written;   // first tensor of a name per ubatch wins (llama.cpp reuses some names)
    std::set<std::string> indexed;
};

static bool wanted(const char * name) {
    for (const char * c : kCapture) {
        const size_t n = strlen(c);
        if (strncmp(name, c, n) == 0 && (name[n] == '\0' || name[n] == '-')) return true;
    }
    return false;
}

static bool dump_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * st = (dump_state *) user_data;
    if (!wanted(t->name) || strchr(t->name, ' ')) return ask ? false : true;   // no "(view)" / "(reshaped)" copies
    if (ask) return true;
    if (strcmp(t->name, "attn_norm-0") == 0) st->written.clear();   // a new ubatch: its tensors append to the files
    if (!st->written.insert(t->name).second) return true;
    if (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_I32) return true;
    // dense copy of a possibly strided tensor (up to 3 dims)
    std::vector<uint8_t> buf((size_t) ggml_nelements(t) * 4);
    const size_t row = (size_t) t->ne[0] * 4;
    size_t r = 0;
    for (int64_t i2 = 0; i2 < t->ne[2]; ++i2)
        for (int64_t i1 = 0; i1 < t->ne[1]; ++i1, ++r)
            ggml_backend_tensor_get(t, buf.data() + r * row, i1 * t->nb[1] + i2 * t->nb[2], row);
    const std::string path = st->dir + "/" + t->name + (t->type == GGML_TYPE_I32 ? ".i32" : ".f32");
    FILE * f = fopen(path.c_str(), "ab");
    if (f) {
        fwrite(buf.data(), 1, buf.size(), f);
        fclose(f);
    }
    if (!st->indexed.insert(t->name).second) return true;
    fprintf(st->index, "%s %s %lld %lld %lld\n", t->name, ggml_type_name(t->type), (long long) t->ne[0],
            (long long) t->ne[1], (long long) t->ne[2]);
    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) return 1;
    const char * dir = getenv("MIMO_DUMP_DIR");
    if (!dir) {
        LOG_ERR("set MIMO_DUMP_DIR\n");
        return 1;
    }
    llama_backend_init();
    dump_state st;
    st.dir = dir;
    st.index = fopen((st.dir + "/index.txt").c_str(), "w");
    if (!st.index) {
        LOG_ERR("cannot write %s/index.txt\n", dir);
        return 1;
    }
    params.cb_eval = dump_cb;
    params.cb_eval_user_data = &st;
    params.warmup = false;
    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx = llama_init->context();
    if (!model || !ctx) return 1;

    std::vector<llama_token> toks;
    if (const char * tf = getenv("MIMO_DUMP_TOKENS")) {
        std::ifstream in(tf, std::ios::binary);
        int32_t v;
        while (in.read((char *) &v, 4)) toks.push_back(v);
    } else {
        toks = common_tokenize(ctx, params.prompt, llama_vocab_get_add_bos(llama_model_get_vocab(model)), true);
    }
    if (toks.empty() || (int) toks.size() > (int) llama_n_batch(ctx)) {
        LOG_ERR("need 1..n_batch tokens, got %zu\n", toks.size());
        return 1;
    }
    FILE * tf = fopen((st.dir + "/tokens.i32").c_str(), "wb");
    fwrite(toks.data(), 4, toks.size(), tf);
    fclose(tf);
    common_batch batch(ctx);
    for (size_t i = 0; i < toks.size(); ++i) batch.add(toks[i], (llama_pos) i, 0, true);
    if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
        LOG_ERR("decode failed\n");
        return 1;
    }
    fclose(st.index);
    LOG_INF("dumped %zu tokens to %s\n", toks.size(), dir);
    llama_backend_free();
    return 0;
}
