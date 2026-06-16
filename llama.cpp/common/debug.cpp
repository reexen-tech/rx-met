#include "debug.h"

#include "log.h"
#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// MoE golden-reference dump (runtime-gated, compiled unconditionally).
// ---------------------------------------------------------------------------
namespace {

struct moe_dump_cfg {
    bool          active = false;  // set via common_debug_moe_dump_set()
    std::string   dir;
    std::set<int> layers;          // empty => all layers
};

moe_dump_cfg g_moe_dump_cfg;

bool moe_env_truthy(const char * v) {
    return v && v[0] && strcmp(v, "0") != 0;
}

// Parse "ffn_moe_down-12" -> base="ffn_moe_down", layer=12.
// Returns false if the name has no trailing "-<digits>" suffix.
bool moe_parse_name(const char * name, std::string & base, int & layer) {
    const char * dash = strrchr(name, '-');
    if (!dash || dash == name || dash[1] == '\0') {
        return false;
    }
    for (const char * p = dash + 1; *p; ++p) {
        if (*p < '0' || *p > '9') {
            return false;
        }
    }
    layer = atoi(dash + 1);
    base.assign(name, (size_t) (dash - name));
    return true;
}

bool moe_is_target_base(const std::string & base) {
    return base == "ffn_moe_topk"          ||
           base == "ffn_moe_weights"       ||
           base == "ffn_moe_weights_norm"  ||
           base == "ffn_moe_down"          ||
           base == "ffn_moe_weighted"      ||
           base == "ffn_moe_out";
}

// Write one tensor as raw `.bin` (contiguous logical order, dtype preserved)
// plus a `.json` sidecar with shape/stride/dtype metadata.
void moe_dump_tensor(const char * dir, const ggml_tensor * t, const uint8_t * data,
                     int layer, const std::string & base) {
    const size_t  ts  = ggml_type_size(t->type);
    const int64_t ne0 = t->ne[0], ne1 = t->ne[1], ne2 = t->ne[2], ne3 = t->ne[3];
    const int64_t n   = ne0 * ne1 * ne2 * ne3;

    std::vector<uint8_t> buf((size_t) n * ts);
    size_t out = 0;
    for (int64_t i3 = 0; i3 < ne3; i3++) {
        for (int64_t i2 = 0; i2 < ne2; i2++) {
            for (int64_t i1 = 0; i1 < ne1; i1++) {
                for (int64_t i0 = 0; i0 < ne0; i0++) {
                    const size_t off = (size_t) i0 * t->nb[0] + (size_t) i1 * t->nb[1] +
                                       (size_t) i2 * t->nb[2] + (size_t) i3 * t->nb[3];
                    memcpy(buf.data() + out, data + off, ts);
                    out += ts;
                }
            }
        }
    }

    char path[1200];
    snprintf(path, sizeof(path), "%s/%s.bin", dir, t->name);
    if (FILE * f = fopen(path, "wb")) {
        if (!buf.empty()) {
            fwrite(buf.data(), 1, buf.size(), f);
        }
        fclose(f);
    } else {
        LOG_ERR("[moe-dump] failed to open %s for writing\n", path);
        return;
    }

    snprintf(path, sizeof(path), "%s/%s.json", dir, t->name);
    if (FILE * j = fopen(path, "w")) {
        fprintf(j, "{\n");
        fprintf(j, "  \"name\": \"%s\",\n", t->name);
        fprintf(j, "  \"base\": \"%s\",\n", base.c_str());
        fprintf(j, "  \"layer\": %d,\n", layer);
        fprintf(j, "  \"ggml_type\": \"%s\",\n", ggml_type_name(t->type));
        fprintf(j, "  \"ggml_type_enum\": %d,\n", (int) t->type);
        fprintf(j, "  \"type_size\": %zu,\n", ts);
        fprintf(j, "  \"ne\": [%lld, %lld, %lld, %lld],\n",
                (long long) ne0, (long long) ne1, (long long) ne2, (long long) ne3);
        fprintf(j, "  \"nb\": [%zu, %zu, %zu, %zu],\n",
                t->nb[0], t->nb[1], t->nb[2], t->nb[3]);
        fprintf(j, "  \"layout\": \"logical_contiguous\",\n");
        fprintf(j, "  \"note\": \"bin is contiguous in ggml logical order (i0 fastest); numpy: frombuffer then reshape to (ne3, ne2, ne1, ne0)\"\n");
        fprintf(j, "}\n");
        fclose(j);
    }
}

} // namespace

void common_debug_moe_dump_set(const char * dir, const int * layers, int n_layers) {
    g_moe_dump_cfg.active = (dir && dir[0]);
    g_moe_dump_cfg.dir    = dir ? dir : "";
    g_moe_dump_cfg.layers.clear();
    if (layers && n_layers > 0) {
        for (int i = 0; i < n_layers; i++) {
            g_moe_dump_cfg.layers.insert(layers[i]);
        }
    }
}

void common_debug_moe_dump_clear() {
    g_moe_dump_cfg.active = false;
    g_moe_dump_cfg.dir.clear();
    g_moe_dump_cfg.layers.clear();
}

static std::string common_ggml_ne_string(const ggml_tensor * t) {
    std::string str;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        str += std::to_string(t->ne[i]);
        if (i + 1 < GGML_MAX_DIMS) {
            str += ", ";
        }
    }
    return str;
}

static float common_ggml_get_float_value(const uint8_t * data,
                           ggml_type       type,
                           const size_t *  nb,
                           size_t          i0,
                           size_t          i1,
                           size_t          i2,
                           size_t          i3) {
    size_t i = i3 * nb[3] + i2 * nb[2] + i1 * nb[1] + i0 * nb[0];
    float  v;
    if (type == GGML_TYPE_F16) {
        v = ggml_fp16_to_fp32(*(const ggml_fp16_t *) &data[i]);
    } else if (type == GGML_TYPE_F32) {
        v = *(const float *) &data[i];
    } else if (type == GGML_TYPE_I64) {
        v = (float) *(const int64_t *) &data[i];
    } else if (type == GGML_TYPE_I32) {
        v = (float) *(const int32_t *) &data[i];
    } else if (type == GGML_TYPE_I16) {
        v = (float) *(const int16_t *) &data[i];
    } else if (type == GGML_TYPE_I8) {
        v = (float) *(const int8_t *) &data[i];
    } else if (type == GGML_TYPE_BF16) {
        v = ggml_bf16_to_fp32(*(const ggml_bf16_t *) &data[i]);
    } else {
        GGML_ABORT("fatal error");
    }
    return v;
}

#define INDENT "    "

template <bool abort>
void common_debug_print_tensor(uint8_t * data, ggml_type type, const int64_t * ne, const size_t * nb, int64_t n) {
    GGML_ASSERT(n > 0);
    float sum = 0;
    for (int64_t i3 = 0; i3 < ne[3]; i3++) {
        for (int64_t i2 = 0; i2 < ne[2]; i2++) {
            for (int64_t i1 = 0; i1 < ne[1]; i1++) {
                for (int64_t i0 = 0; i0 < ne[0]; i0++) {
                    const float v = common_ggml_get_float_value(data, type, nb, i0, i1, i2, i3);
                    sum += v;
                }
            }
        }
    }
    for (int64_t i3 = 0; i3 < ne[3]; i3++) {
        LOG(INDENT "[\n");
        for (int64_t i2 = 0; i2 < ne[2]; i2++) {
            if (i2 == n && ne[2] > 2 * n) {
                LOG(INDENT INDENT "..., \n");
                i2 = ne[2] - n;
            }
            LOG(INDENT INDENT "[\n");
            for (int64_t i1 = 0; i1 < ne[1]; i1++) {
                if (i1 == n && ne[1] > 2 * n) {
                    LOG(INDENT INDENT INDENT "..., \n");
                    i1 = ne[1] - n;
                }
                LOG(INDENT INDENT INDENT "[");
                for (int64_t i0 = 0; i0 < ne[0]; i0++) {
                    if (i0 == n && ne[0] > 2 * n) {
                        LOG("   ..., ");
                        i0 = ne[0] - n;
                    }
                    const float v = common_ggml_get_float_value(data, type, nb, i0, i1, i2, i3);
                    LOG("%12.4f", v);
                    if (i0 < ne[0] - 1) {
                        LOG(", ");
                    }
                }
                LOG("  ],\n");
            }
            LOG(INDENT INDENT "],\n");
        }
        LOG(INDENT "]\n");
        LOG(INDENT "sum = %f\n", sum);
    }

    if constexpr (abort) {
        if (std::isnan(sum)) {
            LOG("encountered NaN - aborting\n");
            exit(0);
        }
    }
}

/**
 * GGML operations callback during the graph execution.
 *
 * @param t current tensor
 * @param ask when ask is true, the scheduler wants to know if we are interested in data from this tensor
 *            if we return true, a follow-up call will be made with ask=false in which we can do the actual collection.
 *            see ggml_backend_sched_eval_callback
 * @param user_data user data to pass at each call back
 * @return true to receive data or continue the graph, false otherwise
 */
template <bool abort_on_nan> bool common_debug_cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * cb_data = (base_callback_data *) user_data;

    const struct ggml_tensor * src0 = t->src[0];
    const struct ggml_tensor * src1 = t->src[1];

    if (ask) {
        return true;  // Always retrieve data
    }

    bool matches_filter = cb_data->tensor_filters.empty();

    if (!matches_filter) {
        for (const auto & filter : cb_data->tensor_filters) {
            if (std::regex_search(t->name, filter)) {
                matches_filter = true;
                break;
            }
        }
    }

    char src1_str[128] = { 0 };
    if (src1) {
        snprintf(src1_str, sizeof(src1_str), "%s{%s}", src1->name, common_ggml_ne_string(src1).c_str());
    }

    if (matches_filter) {
        LOG("%s: %24s = (%s) %10s(%s{%s}, %s}) = {%s}\n", __func__, t->name, ggml_type_name(t->type),
            ggml_op_desc(t), src0->name, common_ggml_ne_string(src0).c_str(), src1 ? src1_str : "",
            common_ggml_ne_string(t).c_str());
    }

    const bool is_host = ggml_backend_buffer_is_host(t->buffer);

    if (!is_host) {
        auto n_bytes = ggml_nbytes(t);
        cb_data->data.resize(n_bytes);
        ggml_backend_tensor_get(t, cb_data->data.data(), 0, n_bytes);
    }

    // --- MoE golden-reference dump (unconditional compile, runtime-gated) ---
    // Active either via common_debug_moe_dump_set() (preferred) or via env vars
    // REEX_DUMP_MOE_ONLY=1 + REEX_DUMP_DIR. Only MoE tensors are written here;
    // the existing general dump below (GGML_USE_REEX) is untouched.
    {
        bool         moe_active = g_moe_dump_cfg.active;
        const char * moe_dir    = moe_active ? g_moe_dump_cfg.dir.c_str() : nullptr;
        bool         use_cfg_layers = moe_active && !g_moe_dump_cfg.layers.empty();

        if (!moe_active) {
            const char * env_only = getenv("REEX_DUMP_MOE_ONLY");
            const char * env_dir  = getenv("REEX_DUMP_DIR");
            if (moe_env_truthy(env_only) && env_dir && env_dir[0]) {
                moe_active = true;
                moe_dir    = env_dir;
            }
        }

        if (moe_active && moe_dir && moe_dir[0] && t->name[0] && !ggml_is_quantized(t->type)) {
            std::string base;
            int         layer = -1;
            if (moe_parse_name(t->name, base, layer) && moe_is_target_base(base)) {
                bool layer_ok = true;
                if (use_cfg_layers) {
                    layer_ok = g_moe_dump_cfg.layers.count(layer) > 0;
                } else {
                    // env single-layer filter: REEX_DUMP_LAYER>=0 restricts to that layer.
                    const char * env_layer = getenv("REEX_DUMP_LAYER");
                    if (env_layer && env_layer[0]) {
                        const int dl = atoi(env_layer);
                        if (dl >= 0) {
                            layer_ok = (dl == layer);
                        }
                    }
                }
                if (layer_ok) {
                    const uint8_t * dptr = is_host ? (const uint8_t *) t->data : cb_data->data.data();
                    moe_dump_tensor(moe_dir, t, dptr, layer, base);
                }
            }
        }
    }

#ifdef GGML_USE_REEX
    // Reex tensor dump: REEX_DUMP_DIR=<dir> [REEX_DUMP_LAYER=<n>]
    // REEX_DUMP_LAYER=-1 dumps ALL layers; >=0 dumps only that layer (default 0).
    static const char * dump_dir   = getenv("REEX_DUMP_DIR");
    static const int    dump_layer = []{ const char *v = getenv("REEX_DUMP_LAYER"); return v ? atoi(v) : 0; }();
    if (dump_dir && dump_dir[0] && !ggml_is_quantized(t->type)) {
        const char * name = t->name && t->name[0] ? t->name : nullptr;
        if (name) {
            bool should_dump = false;

            if (dump_layer < 0) {
                const char * p = name + strlen(name) - 1;
                while (p > name && *p >= '0' && *p <= '9') p--;
                if (*p == '-' && p > name && p[1] != '\0') {
                    should_dump = true;
                }
                if (strncmp(name, "inp_embd", 8) == 0 ||
                    strncmp(name, "result_", 7) == 0 ||
                    strcmp(name, "output") == 0) {
                    should_dump = true;
                }
            } else {
                char suffix[32];
                snprintf(suffix, sizeof(suffix), "-%d", dump_layer);
                size_t slen = strlen(suffix);

                if (strcmp(name, "inp_embd") == 0 && dump_layer == 0) {
                    should_dump = true;
                } else {
                    size_t nlen = strlen(name);
                    if (nlen > slen && strcmp(name + nlen - slen, suffix) == 0) {
                        should_dump = true;
                    }
                }
            }

            if (should_dump) {
                const int64_t n = ggml_nelements(t);
                std::vector<float> buf((size_t)n);
                uint8_t * data_ptr = is_host ? (uint8_t *) t->data : cb_data->data.data();
                for (int64_t idx = 0; idx < n; idx++) {
                    int64_t i0 = idx % t->ne[0];
                    int64_t i1 = (idx / t->ne[0]) % t->ne[1];
                    int64_t i2 = (idx / (t->ne[0] * t->ne[1])) % t->ne[2];
                    int64_t i3 = idx / (t->ne[0] * t->ne[1] * t->ne[2]);
                    buf[(size_t)idx] = common_ggml_get_float_value(data_ptr, t->type, t->nb, i0, i1, i2, i3);
                }
                char path[512];
                snprintf(path, sizeof(path), "%s/%s.bin", dump_dir, name);
                FILE * f = fopen(path, "wb");
                if (f) {
                    fwrite(buf.data(), sizeof(float), (size_t)n, f);
                    fclose(f);
                    LOG_ERR("[reex-dump] saved %s (%lld floats) -> %s\n", name, (long long)n, path);
                }
            }
        }
    }
#endif // GGML_USE_REEX

    if (!ggml_is_quantized(t->type) && matches_filter) {
        uint8_t * data = is_host ? (uint8_t *) t->data : cb_data->data.data();
        common_debug_print_tensor<abort_on_nan>(data, t->type, t->ne, t->nb, 3);
    }

    return true;
}

// Explicit template instantiations
template bool common_debug_cb_eval<false>(ggml_tensor *, bool, void *);
template bool common_debug_cb_eval<true>(ggml_tensor *, bool, void *);
template void common_debug_print_tensor<false>(uint8_t *, ggml_type, const int64_t *, const size_t *, int64_t);
template void common_debug_print_tensor<true>(uint8_t *, ggml_type, const int64_t *, const size_t *, int64_t);
