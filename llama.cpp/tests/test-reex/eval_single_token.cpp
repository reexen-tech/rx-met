/**
 * Minimal single-token evaluation for layer dump comparison.
 *
 * Usage:
 *   REEX_DUMP_DIR=<dir> REEX_DUMP_LAYER=0 \
 *       ./eval_single_token -m <model.gguf> [-t <threads>] [-p "prompt text"] [-f prompt.txt] \
 *       [-c ctx] [-b batch] [--n-tokens N] [-ctk f16|q8_0|q4_0] [-ctv f16|q8_0|q4_0] [-ngl N] [-fa]
 *
 * Loads model, tokenizes prompt, evaluates first batch, dumps target layer tensors
 * and final logits (final_logits.bin) for end-to-end comparison.
 *
 * -ctk / -ctv: KV cache quantization type for K / V (default: f16).
 * REEX_TARGET_LAYER: -1 (default, all layers) or N (only layer N uses REEX).
 *
 * ─── Qwen3Moe 单 block 三组对比（金标 float×float）───
 * 金标为 float×float；3 组：金标 vs W4×float、vs W4×INT16、vs W4×INT8。
 * 需要 4 次运行（4 种编译/运行路径），每次写不同 REEX_DUMP_DIR，再统一用 compare_layer_dumps.py 对比。
 *
 * 1) 金标 float×float：不启用 REX 或使用全 float 路径，dump 到 dump_f32f32
 *    REEX_DUMP_DIR=dump_f32f32 REEX_DUMP_LAYER=0 ./eval_single_token -m model.gguf -p "Hello"
 *
 * REEX_PRINT_CUDA_TRACE=1 且 REEX_DUMP_DIR 已设时，decode 成功后写入 <REEX_DUMP_DIR>/reex_cuda_trace.txt
 * （需 GGML_USE_CUDA 构建；供 run_reex_cuda_trace_sweep.sh 解析）。
 *
 * 2) W4×float：编译 REX FLOAT，dump 到 dump_w4f32
 *    REEX_DUMP_DIR=dump_w4f32 REEX_DUMP_LAYER=0 REEX_TARGET_LAYER=0 ./eval_single_token -m model.gguf -p "Hello"
 *
 * 3) W4×INT16：编译 REX Q16，dump 到 dump_w4q16
 *    REEX_DUMP_DIR=dump_w4q16 ... 同上
 *
 * 4) W4×INT8：编译 REX Q8，dump 到 dump_w4q8
 *    REEX_DUMP_DIR=dump_w4q8 ... 同上
 *
 * 5) 三组对比（金标 vs 三种路径）：
 *    python compare_layer_dumps.py --gold dump_f32f32 \\
 *        --test dump_w4f32 "W4×float" --test dump_w4q16 "W4×INT16" --test dump_w4q8 "W4×INT8"
 */
#include "llama.h"
#include "ggml.h"
#include "debug.h"

#ifdef GGML_USE_REEX_Q64
#include "reex/ggml-reex-q64-hw-dump.h"
#endif

#include <sys/stat.h>
#include <sys/types.h>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

static bool make_dir(const char * path) {
    if (!path || !path[0]) {
        return false;
    }
    if (mkdir(path, 0777) == 0) {
        return true;
    }
    return errno == EEXIST;
}

static ggml_type parse_kv_type(const char * s) {
    if (strcmp(s, "f32")  == 0) return GGML_TYPE_F32;
    if (strcmp(s, "f16")  == 0) return GGML_TYPE_F16;
    if (strcmp(s, "q8_0") == 0) return GGML_TYPE_Q8_0;
    if (strcmp(s, "q4_0") == 0) return GGML_TYPE_Q4_0;
    if (strcmp(s, "q4_1") == 0) return GGML_TYPE_Q4_1;
    if (strcmp(s, "q5_0") == 0) return GGML_TYPE_Q5_0;
    if (strcmp(s, "q5_1") == 0) return GGML_TYPE_Q5_1;
    if (strcmp(s, "q6_k") == 0) return GGML_TYPE_Q6_K;
    fprintf(stderr, "ERROR: unknown KV cache type: %s\n", s);
    exit(1);
}

static const char * kv_type_name(ggml_type t) {
    switch (t) {
        case GGML_TYPE_F32:  return "f32";
        case GGML_TYPE_F16:  return "f16";
        case GGML_TYPE_Q8_0: return "q8_0";
        case GGML_TYPE_Q4_0: return "q4_0";
        case GGML_TYPE_Q4_1: return "q4_1";
        case GGML_TYPE_Q5_0: return "q5_0";
        case GGML_TYPE_Q5_1: return "q5_1";
        case GGML_TYPE_Q6_K: return "q6_k";
        default:             return "?";
    }
}

static void usage(const char * prog) {
    fprintf(stderr, "Usage: %s -m <model.gguf> [-t threads] [-p \"prompt\"] [-f prompt.txt]\n", prog);
    fprintf(stderr, "       [-c ctx] [-b batch] [--n-tokens N] [-ctk type] [-ctv type] [-ngl N] [-fa]\n");
    fprintf(stderr, "  -f: read prompt text from file (overrides -p)\n");
    fprintf(stderr, "  -c: context size (default 512)\n");
    fprintf(stderr, "  -b: max batch / ubatch for chunked prefill (default min(512, ctx))\n");
    fprintf(stderr, "  --n-tokens: cycle tokenized prompt to exactly N tokens (Prefill length)\n");
    fprintf(stderr, "  -ctk / -ctv: KV cache type (f16, q8_0, q4_0, ...). Default: f16\n");
    fprintf(stderr, "  -ngl N: number of layers to offload to GPU (default: 999 = all)\n");
    fprintf(stderr, "  -fa: force enable Flash Attention (required for quantized V cache)\n");
    fprintf(stderr, "  env: REEX_DUMP_DIR, REEX_DUMP_LAYER, REEX_TARGET_LAYER, REEX_PRINT_CUDA_TRACE\n");
}

static std::string read_file(const char * path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        fprintf(stderr, "ERROR: cannot read prompt file: %s\n", path);
        exit(1);
    }
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

static void cycle_tokens(std::vector<llama_token> & tokens, int n_target) {
    if (n_target <= 0 || tokens.empty()) {
        return;
    }
    const llama_token seed0 = tokens[0];
    std::vector<llama_token> seed = tokens;
    tokens.resize((size_t) n_target);
    for (int i = 0; i < n_target; i++) {
        tokens[i] = seed.empty() ? seed0 : seed[i % (int) seed.size()];
    }
}

#ifdef GGML_USE_CUDA
// CUDA REEX 命中计数（定义在 libggml-cuda），供 run_reex_cuda_trace_sweep.sh 解析。
// Q8 / Q16 trace 计数器在 ggml-cuda 中无条件编译，因此可直接引用。
extern "C" void ggml_reex_cuda_reset_q8_stats(void);
extern "C" int  ggml_reex_cuda_get_q8_mul_mat_hits(void);
extern "C" int  ggml_reex_cuda_get_q8_mul_mat_id_hits(void);

extern "C" void ggml_reex_cuda_reset_stats(void);
extern "C" int  ggml_reex_cuda_get_q16_mul_mat_hits(void);
extern "C" int  ggml_reex_cuda_get_q16_batch_fallbacks(void);
extern "C" int  ggml_reex_cuda_get_q16_mul_mat_id_hits(void);
extern "C" int  ggml_reex_cuda_get_q16_mul_mat_id_unsupported(void);

static bool env_truthy(const char * v) {
    if (!v || !v[0] || strcmp(v, "0") == 0) {
        return false;
    }
    return true;
}

static void reex_reset_cuda_trace_if_requested(void) {
    if (!env_truthy(getenv("REEX_PRINT_CUDA_TRACE"))) {
        return;
    }
    ggml_reex_cuda_reset_stats();
    ggml_reex_cuda_reset_q8_stats();
}

static void reex_write_cuda_trace_file(const char * dump_dir) {
    if (!env_truthy(getenv("REEX_PRINT_CUDA_TRACE"))) {
        return;
    }
    if (!dump_dir || !dump_dir[0]) {
        fprintf(stderr, "WARNING: REEX_PRINT_CUDA_TRACE set but REEX_DUMP_DIR empty; skip reex_cuda_trace.txt\n");
        return;
    }
    std::string path = std::string(dump_dir) + "/reex_cuda_trace.txt";
    FILE * fp = fopen(path.c_str(), "w");
    if (!fp) {
        fprintf(stderr, "WARNING: could not write %s\n", path.c_str());
        return;
    }
    fprintf(fp, "q8_mul_mat_hits=%d\n",              ggml_reex_cuda_get_q8_mul_mat_hits());
    fprintf(fp, "q8_mul_mat_id_hits=%d\n",           ggml_reex_cuda_get_q8_mul_mat_id_hits());
    fprintf(fp, "q16_mul_mat_hits=%d\n",             ggml_reex_cuda_get_q16_mul_mat_hits());
    fprintf(fp, "q16_batch_fallbacks=%d\n",          ggml_reex_cuda_get_q16_batch_fallbacks());
    fprintf(fp, "q16_mul_mat_id_hits=%d\n",          ggml_reex_cuda_get_q16_mul_mat_id_hits());
    fprintf(fp, "q16_mul_mat_id_unsupported=%d\n",   ggml_reex_cuda_get_q16_mul_mat_id_unsupported());
    fclose(fp);
    fprintf(stderr, "Wrote REEX CUDA trace stats to %s\n", path.c_str());
}
#endif

int main(int argc, char ** argv) {
    const char * model_path = nullptr;
    int n_threads = 4;
    int n_gpu_layers = 999;
    int n_ctx = 512;
    int n_batch = 0;
    int n_tokens_target = 0;
    bool force_fa = false;
    std::string prompt = "Hello";
    const char * prompt_file = nullptr;
    ggml_type type_k = GGML_TYPE_F16;
    ggml_type type_v = GGML_TYPE_F16;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) { model_path = argv[++i]; }
        else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) { n_threads = atoi(argv[++i]); }
        else if (strcmp(argv[i], "-ngl") == 0 && i + 1 < argc) { n_gpu_layers = atoi(argv[++i]); }
        else if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) { n_ctx = atoi(argv[++i]); }
        else if (strcmp(argv[i], "-b") == 0 && i + 1 < argc) { n_batch = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--n-tokens") == 0 && i + 1 < argc) { n_tokens_target = atoi(argv[++i]); }
        else if (strcmp(argv[i], "-fa") == 0) { force_fa = true; }
        else if (strcmp(argv[i], "-f") == 0 && i + 1 < argc) { prompt_file = argv[++i]; }
        else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) { prompt = argv[++i]; }
        else if (strcmp(argv[i], "-ctk") == 0 && i + 1 < argc) { type_k = parse_kv_type(argv[++i]); }
        else if (strcmp(argv[i], "-ctv") == 0 && i + 1 < argc) { type_v = parse_kv_type(argv[++i]); }
        else { usage(argv[0]); return 1; }
    }
    if (!model_path) { usage(argv[0]); return 1; }
    if (prompt_file) {
        prompt = read_file(prompt_file);
    }
    if (n_ctx < 1) { n_ctx = 512; }
    if (n_batch <= 0) { n_batch = std::min(n_ctx, 512); }
    if (n_batch > n_ctx) { n_batch = n_ctx; }

    const char * dump_dir = getenv("REEX_DUMP_DIR");
    if (!dump_dir || !dump_dir[0]) {
        fprintf(stderr, "WARNING: REEX_DUMP_DIR not set, no tensors will be dumped\n");
    } else {
        make_dir(dump_dir);
    }

    // Load model
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        fprintf(stderr, "ERROR: failed to load model: %s\n", model_path);
        return 1;
    }

    // Create context with cb_eval for tensor dumping
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx    = (uint32_t) n_ctx;
    cparams.n_batch  = (uint32_t) n_batch;
    cparams.n_ubatch = (uint32_t) n_batch;
    cparams.n_threads = n_threads;
    cparams.n_threads_batch = n_threads;
    cparams.type_k = type_k;
    cparams.type_v = type_v;

    if (force_fa) {
        cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    }

    // Set up debug callback for tensor dumping
    static base_callback_data cb_data;
    cparams.cb_eval           = common_debug_cb_eval<false>;
    cparams.cb_eval_user_data = &cb_data;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "ERROR: failed to create context\n");
        llama_model_free(model);
        return 1;
    }

    // Tokenize
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokens(prompt.size() + 8);
    int n_tokens = llama_tokenize(vocab, prompt.c_str(), (int)prompt.size(),
                                  tokens.data(), (int)tokens.size(), true, false);
    if (n_tokens < 0) {
        tokens.resize(-n_tokens);
        n_tokens = llama_tokenize(vocab, prompt.c_str(), (int)prompt.size(),
                                  tokens.data(), (int)tokens.size(), true, false);
    }
    tokens.resize(n_tokens);
    if (n_tokens_target > 0) {
        cycle_tokens(tokens, n_tokens_target);
        n_tokens = n_tokens_target;
    }
    if (n_tokens > n_ctx) {
        fprintf(stderr, "ERROR: prompt has %d tokens but context is %d; increase -c\n", n_tokens, n_ctx);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    fprintf(stderr, "Model: %s\n", model_path);
    fprintf(stderr, "GPU layers: %d\n", n_gpu_layers);
    fprintf(stderr, "Context: n_ctx=%d n_batch=%d\n", n_ctx, n_batch);
    fprintf(stderr, "KV cache: K=%s  V=%s\n", kv_type_name(type_k), kv_type_name(type_v));
    if (prompt.size() <= 120) {
        fprintf(stderr, "Prompt: \"%s\" (%d tokens)\n", prompt.c_str(), n_tokens);
    } else {
        fprintf(stderr, "Prompt: \"%.*s...\" (%d tokens)\n", 80, prompt.c_str(), n_tokens);
    }
    if (dump_dir) {
        fprintf(stderr, "Dump dir: %s\n", dump_dir);
    }

    fprintf(stderr, "Evaluating %d tokens...\n", n_tokens);
#ifdef GGML_USE_CUDA
    reex_reset_cuda_trace_if_requested();
#endif
    int rc = 0;
    int last_chunk = 0;
    for (int pos = 0; pos < n_tokens && rc == 0; pos += n_batch) {
        last_chunk = std::min(n_batch, n_tokens - pos);
        llama_batch batch = llama_batch_get_one(tokens.data() + pos, last_chunk);
        rc = llama_decode(ctx, batch);
        if (rc != 0) {
            fprintf(stderr, "ERROR: llama_decode failed at pos=%d: %d\n", pos, rc);
        }
    }
    if (rc == 0) {
        fprintf(stderr, "Decode OK. Tensor dumps saved to %s\n", dump_dir ? dump_dir : "(none)");

        // Dump final logits (last token) — the most important end-to-end metric
        if (dump_dir && dump_dir[0]) {
            const float * logits = llama_get_logits_ith(ctx, last_chunk - 1);
            const int32_t n_vocab = llama_vocab_n_tokens(vocab);
            if (logits && n_vocab > 0) {
                std::string path = std::string(dump_dir) + "/final_logits.bin";
                FILE * fp = fopen(path.c_str(), "wb");
                if (fp) {
                    fwrite(logits, sizeof(float), n_vocab, fp);
                    fclose(fp);
                    fprintf(stderr, "Saved final logits (%d values) to %s\n", n_vocab, path.c_str());
                }

                // Top-5 next-token predictions for quick visual comparison
                std::vector<std::pair<float, int>> scores(n_vocab);
                for (int32_t i = 0; i < n_vocab; i++) scores[i] = {logits[i], i};
                int top_n = std::min(5, (int)n_vocab);
                std::partial_sort(scores.begin(), scores.begin() + top_n, scores.end(),
                    [](const std::pair<float,int> & a, const std::pair<float,int> & b) {
                        return a.first > b.first;
                    });
                fprintf(stderr, "Top-5 next-token predictions:\n");
                for (int i = 0; i < top_n; i++) {
                    const char * text = llama_vocab_get_text(vocab, scores[i].second);
                    fprintf(stderr, "  #%d  token=%-6d logit=%9.4f  \"%s\"\n",
                        i + 1, scores[i].second, scores[i].first, text ? text : "?");
                }
            }
        }
#ifdef GGML_USE_CUDA
        if (dump_dir && dump_dir[0]) {
            reex_write_cuda_trace_file(dump_dir);
        }
#endif
    }
    llama_free(ctx);
    llama_model_free(model);
#ifdef GGML_USE_REEX_Q64
    reex_q64_hw_dump_flush();
#endif
    return rc;
}
