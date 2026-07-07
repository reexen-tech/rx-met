/**
 * dump_moe_prefill — MoE GPU 采数工具 (golden reference for RTL 专家重排验证).
 *
 * Loads a fixed MoE GGUF model, runs an 8192-token Prefill (single llama_decode,
 * no decode/generation loop) and dumps, for every MoE layer, the router / expert
 * reorder / expert result / un-reorder tensors emitted by build_moe_ffn():
 *
 *   ffn_moe_topk-<L>           top-k selected expert ids (i32)         [topk, token]
 *   ffn_moe_weights-<L>        router weights (pre-norm)               [1, topk, token]
 *   ffn_moe_weights_norm-<L>   router weights (normalized)             [1, topk, token]
 *   ffn_moe_down-<L>           per-token per-expert output (unweighted)[hidden, topk, token]
 *   ffn_moe_weighted-<L>       per-token per-expert output (weighted)  [hidden, topk, token]
 *   ffn_moe_out-<L>            final summed MoE output (sanity check)  [hidden, token]
 *
 * The dump is performed by common_debug_cb_eval() (common/debug.cpp), configured
 * per case via common_debug_moe_dump_set(). C++ writes raw `.bin` + `.json`
 * sidecars; the Python post-processor (moe_postprocess.py) converts them to
 * normalized `.npy` plus expert maps / load ratios.
 *
 * Output layout (default root: <repo>/output):
 *   output/run_metadata.json
 *   output/case_XX/input_token_ids.bin (+ .json)
 *   output/case_XX/raw/<tensor>.bin (+ .json)        <- consumed by post-process
 *
 * Usage:
 *   dump_moe_prefill [-m model.gguf] [-o out_dir] [--cases N] [--layers a,b,c]
 *                    [--n-tokens N] [-ngl N] [-t threads]
 *
 * Defaults: full run = 6 cases, all MoE layers, 8192 tokens, ngl=999.
 */
#include "llama.h"
#include "ggml.h"
#include "debug.h"

#include <sys/stat.h>
#include <sys/types.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static const char * DEFAULT_MODEL =
    "/mnt/data8t/share/models/Qwen/Qwen3.5-35B-A3B/Qwen3.5-35B-A3B-Q4_0_64.gguf";
static const char * DEFAULT_OUT = "/mnt/data8t/zcx/aimet_rx/llama.cpp/output";

// 6 fixed seed texts; each is tokenized and cycled to exactly n_tokens.
static const char * SEED_TEXTS[6] = {
    "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump! Sphinx of black quartz, judge my vow.",

    "In the beginning the universe was created. This has made a lot of people very angry and "
    "been widely regarded as a bad move. Space is big. Really big. You just won't believe how "
    "vastly, hugely, mind-bogglingly big it is.",

    "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    "
    "return a\n\nclass Matrix:\n    def __init__(self, rows, cols):\n        self.rows = rows\n",

    "深度学习是机器学习的一个分支，它基于人工神经网络进行表示学习。大语言模型通过自注意力机制"
    "捕捉序列中的长程依赖关系，并在海量文本上进行预训练，从而获得强大的语言理解与生成能力。",

    "The mitochondria is the powerhouse of the cell. Photosynthesis converts light energy into "
    "chemical energy stored in glucose. DNA carries the genetic instructions used in growth, "
    "development, functioning and reproduction of all known living organisms.",

    "Once upon a time in a distant kingdom there lived a wise old king who loved riddles. "
    "Every traveller who solved his riddle was rewarded with gold, but those who failed were "
    "sent to tend the royal gardens for a year and a day.",
};

static void usage(const char * prog) {
    fprintf(stderr,
        "Usage: %s [-m model.gguf] [-o out_dir] [--cases N] [--layers a,b,c]\n"
        "          [--n-tokens N] [-ngl N] [-t threads]\n"
        "  -m         model path (default: fixed Q4_0_64 model)\n"
        "  -o         output root (default: <repo>/output)\n"
        "  --cases    number of cases 1..6 (default 6)\n"
        "  --layers   comma-separated MoE layer indices (default: all MoE layers)\n"
        "  --n-tokens prefill length (default 8192)\n"
        "  -ngl       GPU layers to offload (default 999)\n"
        "  -t         threads (default 8)\n"
        "  --override-kv KEY=TYPE:VALUE  override model metadata (repeatable),\n"
        "             e.g. --override-kv qwen35moe.expert_used_count=int:4 for top-4\n", prog);
}

static bool make_dir(const std::string & path) {
    if (mkdir(path.c_str(), 0777) == 0) {
        return true;
    }
    return errno == EEXIST;
}

static std::vector<int> parse_int_list(const char * s) {
    std::vector<int> out;
    const char * p = s;
    while (*p) {
        while (*p == ',' || *p == ' ') ++p;
        if (!*p) break;
        char * end = nullptr;
        long v = strtol(p, &end, 10);
        if (end == p) break;
        out.push_back((int) v);
        p = end;
    }
    return out;
}

// Find a gguf meta value whose key ends with `suffix` (e.g. ".expert_count").
// Returns -1 if not found / not an integer.
static int64_t meta_int_by_suffix(const llama_model * model, const char * suffix) {
    const int32_t n = llama_model_meta_count(model);
    char key[256];
    char val[256];
    const size_t slen = strlen(suffix);
    for (int32_t i = 0; i < n; i++) {
        if (llama_model_meta_key_by_index(model, i, key, sizeof(key)) < 0) continue;
        const size_t klen = strlen(key);
        if (klen < slen) continue;
        if (strcmp(key + klen - slen, suffix) != 0) continue;
        if (llama_model_meta_val_str_by_index(model, i, val, sizeof(val)) < 0) continue;
        return (int64_t) strtoll(val, nullptr, 10);
    }
    return -1;
}

// Write an int32 array as raw `.bin` + `.json` sidecar (same convention as the
// MoE tensor dump, so the Python tooling can read it uniformly).
static void write_token_ids(const std::string & dir, const std::vector<int32_t> & ids) {
    std::string bin = dir + "/input_token_ids.bin";
    if (FILE * f = fopen(bin.c_str(), "wb")) {
        if (!ids.empty()) fwrite(ids.data(), sizeof(int32_t), ids.size(), f);
        fclose(f);
    }
    std::string js = dir + "/input_token_ids.json";
    if (FILE * j = fopen(js.c_str(), "w")) {
        fprintf(j, "{\n");
        fprintf(j, "  \"name\": \"input_token_ids\",\n");
        fprintf(j, "  \"ggml_type\": \"i32\",\n");
        fprintf(j, "  \"type_size\": 4,\n");
        fprintf(j, "  \"ne\": [%zu, 1, 1, 1],\n", ids.size());
        fprintf(j, "  \"layout\": \"logical_contiguous\"\n");
        fprintf(j, "}\n");
        fclose(j);
    }
}

static std::string json_escape(const std::string & s) {
    std::string o;
    for (char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += c; }
        else if (c == '\n') { o += "\\n"; }
        else { o += c; }
    }
    return o;
}

int main(int argc, char ** argv) {
    const char * model_path = DEFAULT_MODEL;
    std::string  out_root   = DEFAULT_OUT;
    int n_cases   = 6;
    int n_tokens  = 8192;
    int n_gpu_layers = 999;
    int n_threads = 8;
    std::vector<int> layer_filter; // empty = all
    std::vector<llama_model_kv_override> kv_overrides; // empty = none

    for (int i = 1; i < argc; i++) {
        if      (strcmp(argv[i], "-m") == 0 && i + 1 < argc)        { model_path = argv[++i]; }
        else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc)        { out_root = argv[++i]; }
        else if (strcmp(argv[i], "--cases") == 0 && i + 1 < argc)   { n_cases = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--layers") == 0 && i + 1 < argc)  { layer_filter = parse_int_list(argv[++i]); }
        else if (strcmp(argv[i], "--n-tokens") == 0 && i + 1 < argc){ n_tokens = atoi(argv[++i]); }
        else if (strcmp(argv[i], "-ngl") == 0 && i + 1 < argc)      { n_gpu_layers = atoi(argv[++i]); }
        else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc)        { n_threads = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--override-kv") == 0 && i + 1 < argc) {
            if (!string_parse_kv_override(argv[++i], kv_overrides)) {
                fprintf(stderr, "ERROR: invalid --override-kv: %s\n", argv[i]);
                return 1;
            }
        }
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) { usage(argv[0]); return 0; }
        else { fprintf(stderr, "Unknown arg: %s\n", argv[i]); usage(argv[0]); return 1; }
    }

    if (n_cases < 1) n_cases = 1;
    if (n_cases > 6) n_cases = 6;
    if (n_tokens < 1) n_tokens = 1;

    if (!make_dir(out_root)) {
        fprintf(stderr, "ERROR: cannot create output root: %s\n", out_root.c_str());
        return 1;
    }

    llama_backend_init();

    // Load model.
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;
    if (!kv_overrides.empty()) {
        // kv_overrides must be terminated with an empty-key sentinel.
        kv_overrides.emplace_back();
        kv_overrides.back().key[0] = 0;
        mparams.kv_overrides = kv_overrides.data();
        for (const auto & o : kv_overrides) {
            if (o.key[0]) fprintf(stderr, "  override-kv: %s\n", o.key);
        }
    }
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        fprintf(stderr, "ERROR: failed to load model: %s\n", model_path);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int64_t n_embd   = llama_model_n_embd(model);
    const int64_t n_layer  = llama_model_n_layer(model);
    const int64_t n_expert = meta_int_by_suffix(model, ".expert_count");
    int64_t       n_topk   = meta_int_by_suffix(model, ".expert_used_count");
    // The GGUF meta API returns the on-disk value, not a runtime override; reflect
    // a --override-kv <arch>.expert_used_count here so logs / run_metadata are accurate.
    for (const auto & o : kv_overrides) {
        const char * suf = ".expert_used_count";
        const size_t klen = strlen(o.key), slen = strlen(suf);
        if (o.tag == LLAMA_KV_OVERRIDE_TYPE_INT && klen >= slen &&
            strcmp(o.key + klen - slen, suf) == 0) {
            n_topk = o.val_i64;
        }
    }

    fprintf(stderr, "Model: %s\n", model_path);
    fprintf(stderr, "  n_embd=%lld n_layer=%lld n_expert=%lld n_expert_used=%lld\n",
            (long long) n_embd, (long long) n_layer, (long long) n_expert, (long long) n_topk);
    fprintf(stderr, "  cases=%d n_tokens=%d ngl=%d\n", n_cases, n_tokens, n_gpu_layers);

    // Context: single-shot prefill, n_ctx == n_batch == n_ubatch == n_tokens.
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx          = (uint32_t) n_tokens;
    cparams.n_batch        = (uint32_t) n_tokens;
    cparams.n_ubatch       = (uint32_t) n_tokens;
    cparams.n_threads      = n_threads;
    cparams.n_threads_batch= n_threads;

    // Suppress the verbose per-tensor printing in common_debug_cb_eval by using a
    // filter that matches nothing; the MoE dump is independent of this filter.
    static base_callback_data cb_data;
    cb_data.tensor_filters.clear();
    try {
        cb_data.tensor_filters.emplace_back("^__moe_dump_no_print__$");
    } catch (...) {}
    cparams.cb_eval           = common_debug_cb_eval<false>;
    cparams.cb_eval_user_data = &cb_data;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "ERROR: failed to create context\n");
        llama_model_free(model);
        return 1;
    }

    const int * layers_ptr = layer_filter.empty() ? nullptr : layer_filter.data();
    const int   n_layers_f = (int) layer_filter.size();

    for (int c = 0; c < n_cases; c++) {
        // Build token ids: tokenize seed, then cycle to exactly n_tokens.
        const std::string seed = SEED_TEXTS[c % 6];
        std::vector<llama_token> seed_tok(seed.size() + 16);
        int nt = llama_tokenize(vocab, seed.c_str(), (int) seed.size(),
                                seed_tok.data(), (int) seed_tok.size(), true, false);
        if (nt < 0) {
            seed_tok.resize(-nt);
            nt = llama_tokenize(vocab, seed.c_str(), (int) seed.size(),
                                seed_tok.data(), (int) seed_tok.size(), true, false);
        }
        seed_tok.resize(nt > 0 ? nt : 1);
        if (nt <= 0) seed_tok[0] = 0;

        std::vector<int32_t> tokens((size_t) n_tokens);
        for (int i = 0; i < n_tokens; i++) {
            tokens[i] = (int32_t) seed_tok[i % seed_tok.size()];
        }

        char case_dir[1024];
        snprintf(case_dir, sizeof(case_dir), "%s/case_%02d", out_root.c_str(), c);
        std::string raw_dir = std::string(case_dir) + "/raw";
        if (!make_dir(case_dir) || !make_dir(raw_dir)) {
            fprintf(stderr, "ERROR: cannot create case dir %s\n", case_dir);
            llama_free(ctx); llama_model_free(model); return 1;
        }

        // Fresh KV cache for each independent prefill case.
        llama_memory_clear(llama_get_memory(ctx), true);

        common_debug_moe_dump_set(raw_dir.c_str(), layers_ptr, n_layers_f);

        fprintf(stderr, "[case %02d] prefill %d tokens -> %s\n", c, n_tokens, raw_dir.c_str());
        llama_batch batch = llama_batch_get_one(tokens.data(), n_tokens);
        int rc = llama_decode(ctx, batch);

        common_debug_moe_dump_clear();

        if (rc != 0) {
            fprintf(stderr, "ERROR: llama_decode failed for case %d: %d\n", c, rc);
            llama_free(ctx); llama_model_free(model); return 1;
        }

        write_token_ids(case_dir, tokens);
        fprintf(stderr, "[case %02d] done\n", c);
    }

    // Run-level metadata.
    {
        const char * container = getenv("DUMP_MOE_CONTAINER");
        const char * commit    = getenv("DUMP_MOE_GIT_COMMIT");
        std::string meta = out_root + "/run_metadata.json";
        if (FILE * j = fopen(meta.c_str(), "w")) {
            fprintf(j, "{\n");
            fprintf(j, "  \"model_path\": \"%s\",\n", json_escape(model_path).c_str());
            fprintf(j, "  \"docker_container\": \"%s\",\n", container ? json_escape(container).c_str() : "quant-gru-cuda128");
            fprintf(j, "  \"git_commit\": \"%s\",\n", commit ? json_escape(commit).c_str() : "unknown");
            fprintf(j, "  \"n_layer\": %lld,\n", (long long) n_layer);
            fprintf(j, "  \"n_embd\": %lld,\n", (long long) n_embd);
            fprintf(j, "  \"n_expert\": %lld,\n", (long long) n_expert);
            fprintf(j, "  \"n_expert_used\": %lld,\n", (long long) n_topk);
            fprintf(j, "  \"n_cases\": %d,\n", n_cases);
            fprintf(j, "  \"n_tokens\": %d,\n", n_tokens);
            fprintf(j, "  \"n_gpu_layers\": %d,\n", n_gpu_layers);
            fprintf(j, "  \"layer_filter\": [");
            for (size_t i = 0; i < layer_filter.size(); i++) {
                fprintf(j, "%s%d", i ? ", " : "", layer_filter[i]);
            }
            fprintf(j, "],\n");
            fprintf(j, "  \"run_command\": \"");
            for (int i = 0; i < argc; i++) {
                fprintf(j, "%s%s", i ? " " : "", json_escape(argv[i]).c_str());
            }
            fprintf(j, "\"\n}\n");
            fclose(j);
        }
    }

    fprintf(stderr, "All cases done. Raw dumps under %s/case_*/raw\n", out_root.c_str());

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
