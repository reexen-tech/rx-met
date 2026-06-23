#ifdef GGML_USE_REEX_Q64

#include "reex/ggml-reex-q64-hw-dump.h"
#include "reex/ggml-reex-q64.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#include <errno.h>

/* Implemented in libggml-cuda when CUDA is enabled. */
extern void ggml_reex_q64_hw_dump_arm(int max_m);
extern void ggml_reex_q64_hw_dump_disarm(void);
extern int  ggml_reex_q64_hw_dump_download(reex_q64_hw_psum_record * out, int max_out);

static reex_q64_hw_dump_cfg g_cfg;
static int                  g_cfg_init = 0;
static int                  g_total_records = 0;
static char                 g_case_name[160] = { 0 };

static void reex_q64_hw_dump_init(void) {
    if (g_cfg_init) {
        return;
    }
    memset(&g_cfg, 0, sizeof(g_cfg));
    g_cfg.layer = -1;
    g_cfg.dump_pre  = true;
    g_cfg.dump_post = true;

    const char * dir = getenv("REEX_HW_DUMP_DIR");
    if (dir && dir[0]) {
        g_cfg.active = true;
        strncpy(g_cfg.dir, dir, sizeof(g_cfg.dir) - 1);
    }
    const char * tname = getenv("REEX_HW_DUMP_TENSOR");
    if (tname && tname[0]) {
        strncpy(g_cfg.tensor, tname, sizeof(g_cfg.tensor) - 1);
    }
    const char * layer = getenv("REEX_HW_DUMP_LAYER");
    if (layer && layer[0]) {
        g_cfg.layer = atoi(layer);
    }
    const char * token = getenv("REEX_HW_DUMP_TOKEN");
    if (token && token[0]) {
        g_cfg.token = atoi(token);
    }
    const char * max_m = getenv("REEX_HW_DUMP_MAX_M");
    if (max_m && max_m[0]) {
        g_cfg.max_m = atoi(max_m);
    }
    const char * psum = getenv("REEX_HW_DUMP_PSUM");
    if (psum && psum[0]) {
        g_cfg.dump_pre  = strstr(psum, "pre")  != NULL;
        g_cfg.dump_post = strstr(psum, "post") != NULL;
    }
    g_cfg.psum_bits = reex_q64_psum_bits();
    g_cfg_init = 1;
}

static int reex_q64_hw_dump_mkdir_p(const char * path) {
    char tmp[512];
    strncpy(tmp, path, sizeof(tmp) - 1);
    tmp[sizeof(tmp) - 1] = '\0';
    for (char * p = tmp + 1; *p; ++p) {
        if (*p == '/') {
            *p = '\0';
            mkdir(tmp, 0777);
            *p = '/';
        }
    }
    return mkdir(tmp, 0777);
}

static void reex_q64_hw_dump_write_bin(const char * path, const void * data, size_t n) {
    FILE * f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "[reex-hw-dump] cannot write %s\n", path);
        return;
    }
    if (data && n) {
        fwrite(data, 1, n, f);
    }
    fclose(f);
}

const reex_q64_hw_dump_cfg * reex_q64_hw_dump_get_cfg(void) {
    reex_q64_hw_dump_init();
    return &g_cfg;
}

bool reex_q64_hw_dump_should_record(const char * tensor, int layer, int token) {
    reex_q64_hw_dump_init();
    if (!g_cfg.active || !tensor || !tensor[0]) {
        return false;
    }
    if (g_cfg.layer >= 0 && g_cfg.layer != layer) {
        return false;
    }
    if (g_cfg.token >= 0 && g_cfg.token != token) {
        return false;
    }
    if (g_cfg.tensor[0]) {
        const char * dash = strrchr(tensor, '-');
        size_t blen = dash ? (size_t)(dash - tensor) : strlen(tensor);
        size_t tlen = strlen(g_cfg.tensor);
        if (tlen != blen || strncmp(tensor, g_cfg.tensor, blen) != 0) {
            return false;
        }
    }
    return true;
}

void reex_q64_hw_dump_matmul_begin(const char * name, int layer) {
    reex_q64_hw_dump_init();
    if (!g_cfg.active || !name) {
        return;
    }
    snprintf(g_case_name, sizeof(g_case_name), "%s", name);
    (void) layer;
    ggml_reex_q64_hw_dump_arm(g_cfg.max_m > 0 ? g_cfg.max_m : 64);
    fprintf(stderr, "[reex-hw-dump] arm matmul %s (max_m=%d)\n", name, g_cfg.max_m);
}

void reex_q64_hw_dump_matmul_end(
    const char * name, int layer,
    const void * weight_data, int64_t weight_bytes,
    const void * act_data, int64_t act_bytes,
    const float * output_data, int64_t output_count) {

    reex_q64_hw_dump_init();
    if (!g_cfg.active || !name || g_case_name[0] == '\0') {
        return;
    }
    if (strcmp(name, g_case_name) != 0) {
        return;
    }

    enum { kMaxRec = 65536 };
    static reex_q64_hw_psum_record recs[kMaxRec];
    const int nrec = ggml_reex_q64_hw_dump_download(recs, kMaxRec);
    ggml_reex_q64_hw_dump_disarm();
    g_total_records += nrec;

    char case_dir[640];
    snprintf(case_dir, sizeof(case_dir), "%s/%s", g_cfg.dir, name);
    reex_q64_hw_dump_mkdir_p(case_dir);

    if (g_cfg.dump_pre) {
        char path[704];
        snprintf(path, sizeof(path), "%s/psum_pre_trunc_i32.bin", case_dir);
        FILE * f = fopen(path, "wb");
        if (f) {
            for (int i = 0; i < nrec; ++i) {
                int32_t v = recs[i].psum_pre;
                fwrite(&v, sizeof(v), 1, f);
            }
            fclose(f);
        }
    }
    if (g_cfg.dump_post) {
        char path[704];
        snprintf(path, sizeof(path), "%s/psum_post_trunc_i32.bin", case_dir);
        FILE * f = fopen(path, "wb");
        if (f) {
            for (int i = 0; i < nrec; ++i) {
                int32_t v = recs[i].psum_post;
                fwrite(&v, sizeof(v), 1, f);
            }
            fclose(f);
        }
    }

    char path[704];
    snprintf(path, sizeof(path), "%s/weight_block.bin", case_dir);
    reex_q64_hw_dump_write_bin(path, weight_data, (size_t) weight_bytes);

    snprintf(path, sizeof(path), "%s/act_q8_block.bin", case_dir);
    reex_q64_hw_dump_write_bin(path, act_data, (size_t) act_bytes);

    snprintf(path, sizeof(path), "%s/output_f32.bin", case_dir);
    reex_q64_hw_dump_write_bin(path, output_data, (size_t) output_count * sizeof(float));

    snprintf(path, sizeof(path), "%s/meta.json", case_dir);
    FILE * mf = fopen(path, "w");
    if (mf) {
        fprintf(mf,
            "{\n"
            "  \"phase\": 1,\n"
            "  \"tensor\": \"%s\",\n"
            "  \"layer\": %d,\n"
            "  \"psum_bits\": %d,\n"
            "  \"record_count\": %d,\n"
            "  \"weight_bytes\": %lld,\n"
            "  \"act_bytes\": %lld,\n"
            "  \"output_floats\": %lld\n"
            "}\n",
            name, layer, g_cfg.psum_bits, nrec,
            (long long) weight_bytes, (long long) act_bytes,
            (long long) output_count);
        fclose(mf);
    }

    fprintf(stderr, "[reex-hw-dump] end matmul %s: %d psum records -> %s\n",
            name, nrec, case_dir);
    g_case_name[0] = '\0';
}

void reex_q64_hw_dump_flush(void) {
    reex_q64_hw_dump_init();
    if (!g_cfg.active) {
        return;
    }
    char path[576];
    snprintf(path, sizeof(path), "%s/meta.json", g_cfg.dir);
    FILE * f = fopen(path, "w");
    if (!f) {
        fprintf(stderr, "[reex-hw-dump] cannot write %s\n", path);
        return;
    }
    time_t now = time(NULL);
    const char * status = g_total_records > 0 ? "partial" : "skeleton";
    fprintf(f,
        "{\n"
        "  \"phase\": 1,\n"
        "  \"status\": \"%s\",\n"
        "  \"timestamp\": %ld,\n"
        "  \"dir\": \"%s\",\n"
        "  \"filter_tensor\": \"%s\",\n"
        "  \"filter_layer\": %d,\n"
        "  \"filter_token\": %d,\n"
        "  \"max_m\": %d,\n"
        "  \"psum_bits\": %d,\n"
        "  \"record_count\": %d\n"
        "}\n",
        status, (long) now, g_cfg.dir, g_cfg.tensor, g_cfg.layer, g_cfg.token,
        g_cfg.max_m, g_cfg.psum_bits, g_total_records);
    fclose(f);
    fprintf(stderr, "[reex-hw-dump] flushed meta.json (%d total records) -> %s\n",
            g_total_records, path);
}

#endif /* GGML_USE_REEX_Q64 */
