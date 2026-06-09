// SPDX-License-Identifier: MIT
//
// reex_turboquant_wht.c — REEX_TURBOQUANT only.
//
// CPU implementation of the structured Walsh-Hadamard rotation; see
// reex_turboquant_wht.h for the algorithm and conventions.
//
// Implementation choices
// ----------------------
//   * The butterfly is the textbook in-place FWHT on a power-of-two array
//     (Sylvester construction).  No SIMD intrinsics in Phase 1 — keeping the
//     scalar reference clean lets vs-oracle pin the algorithm exactly; SIMD
//     specialisation can plug in later for `dim==128` once the algorithmic
//     path is set in stone.
//   * The (1/sqrt(d)) normalisation is folded into a single multiplication
//     after the butterfly (rather than a `1/sqrt(2)` per stage) so the result
//     is bit-equal regardless of `dim`.
//   * `scale_inv` is applied to the INPUT row before the inner sign step —
//     i.e. `effective_input[i] = x[i] * scale_inv[i]`.  This is the position
//     required by InnerQ EMA's invariance property
//     `<Q ⊙ s_inv, K ⊙ s> = <Q, K>`: orthogonal H_d preserves inner products
//     of its inputs, but H_d does NOT commute with diagonal scaling, so
//     post-WHT scaling would break the invariant.  See reex_turboquant_wht.h
//     for the full math.  P4 A2.4 Stage 2a moved this from post- to pre-WHT
//     application; the previous post-WHT location was a P2-era placeholder
//     never exercised on the hot path.
//   * `sq_accum` (per-channel F64 accumulator) is updated AFTER the rotation
//     (post-`1/sqrt(d)` normalisation), so it captures the variance of the
//     *post-WHT* output — i.e. the values that would be quantised by TQ_K3.
//     This is what InnerQ EMA's "calibrate-then-freeze" actually wants to
//     measure.  When both `scale_inv` and `sq_accum` are non-NULL the
//     accumulator sees the post-equalisation values.

#include "reex_turboquant_wht.h"

#ifdef REEX_TURBOQUANT

#include "reex_turboquant_wht_signs.inc"

#include "ggml-impl.h"

#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// =============================================================================
// V_WHT marginal dump hook (root-cause analysis A1).
//
// When env `REEX_TQ_VWHT_DUMP_PATH` is set, every forward (direction==0)
// dim==128 WHT row is appended to that file in binary form, up to a cap of
// `REEX_TQ_VWHT_DUMP_MAX_ROWS` (default 10000).  Used to empirically test
// whether the post-WHT marginal matches the Beta_d(x) assumption baked into
// `tests/oracle/reex_turboquant_oracle_core.cpp::build_lloyd_max_codebook` or
// the N(0, 1/d) approximation used by TheTom's `llama-cpp-turboquant`.
//
// File layout (little-endian, host):
//   uint32_t magic    = 0x52454558  ('REEX')
//   uint32_t dim      = 128
//   uint64_t max_rows = N
//   float[]  rows     = up to (count, dim) actually written
// The actual row count is just (file_size - 16) / (dim * 4).
//
// This is debug-only instrumentation gated by REEX_TURBOQUANT and env var:
// zero overhead when the env is unset (single TLS check + early return).
// =============================================================================

static pthread_mutex_t  g_vwht_dump_mtx  = PTHREAD_MUTEX_INITIALIZER;
static pthread_once_t   g_vwht_dump_once = PTHREAD_ONCE_INIT;
static FILE *           g_vwht_dump_fp   = NULL;
static uint64_t         g_vwht_dump_remaining = 0;

static void vwht_dump_init_once_fn(void) {
    const char * path = getenv("REEX_TQ_VWHT_DUMP_PATH");
    if (path == NULL || path[0] == '\0') {
        return;
    }
    const char * max_s = getenv("REEX_TQ_VWHT_DUMP_MAX_ROWS");
    uint64_t max_rows = 10000;
    if (max_s != NULL && max_s[0] != '\0') {
        max_rows = strtoull(max_s, NULL, 10);
    }
    if (max_rows == 0) {
        return;
    }
    FILE * fp = fopen(path, "wb");
    if (fp == NULL) {
        fprintf(stderr, "[reex-tq-vwht-dump] failed to open '%s' for writing\n", path);
        return;
    }
    const uint32_t magic = 0x52454558u;
    const uint32_t dim   = 128u;
    fwrite(&magic,    sizeof(magic),    1, fp);
    fwrite(&dim,      sizeof(dim),      1, fp);
    fwrite(&max_rows, sizeof(max_rows), 1, fp);
    fflush(fp);
    g_vwht_dump_fp = fp;
    g_vwht_dump_remaining = max_rows;
    fprintf(stderr, "[reex-tq-vwht-dump] initialised: path='%s' max_rows=%llu\n",
            path, (unsigned long long) max_rows);
}

static inline void vwht_dump_maybe_init(void) {
    pthread_once(&g_vwht_dump_once, vwht_dump_init_once_fn);
}

static void vwht_dump_row_locked(const float * row, int64_t dim) {
    if (g_vwht_dump_fp == NULL || g_vwht_dump_remaining == 0) return;
    if (dim != 128) return;
    fwrite(row, sizeof(float), (size_t) dim, g_vwht_dump_fp);
    g_vwht_dump_remaining--;
    if (g_vwht_dump_remaining == 0) {
        fflush(g_vwht_dump_fp);
        fclose(g_vwht_dump_fp);
        g_vwht_dump_fp = NULL;
        fprintf(stderr, "[reex-tq-vwht-dump] cap reached, closed dump file\n");
    }
}

static const int8_t * pick_signs(int64_t dim, int s1_or_s2) {
    if (dim == 128) {
        return s1_or_s2 == 0
            ? reex_turboquant_wht_signs_s1_d128
            : reex_turboquant_wht_signs_s2_d128;
    }
    if (dim == 64) {
        return s1_or_s2 == 0
            ? reex_turboquant_wht_signs_s1_d64
            : reex_turboquant_wht_signs_s2_d64;
    }
    return NULL;
}

static int is_power_of_two(int64_t n) {
    return n > 0 && (n & (n - 1)) == 0;
}

// In-place FWHT (unscaled Sylvester Hadamard).
//   After the call, sum-of-squares is multiplied by `dim` (because H_d^T H_d =
//   d * I).  Caller is responsible for the (1/sqrt(d)) normalisation.
static void fwht_inplace(float * GGML_RESTRICT a, int64_t dim) {
    for (int64_t h = 1; h < dim; h <<= 1) {
        for (int64_t i = 0; i < dim; i += (h << 1)) {
            for (int64_t j = i; j < i + h; ++j) {
                const float x = a[j];
                const float y = a[j + h];
                a[j]     = x + y;
                a[j + h] = x - y;
            }
        }
    }
}

void reex_turboquant_cpu_wht_apply_row(
        const float * GGML_RESTRICT x,
        float       * GGML_RESTRICT out,
        int64_t                      dim,
        int                          direction,
        const float * GGML_RESTRICT scale_inv,
        double      *               sq_accum) {
    if (!is_power_of_two(dim) || (dim != 128 && dim != 64)) {
        GGML_ABORT("reex_turboquant_cpu_wht_apply_row: unsupported dim=%lld "
                   "(only 64 / 128 are supported in Phase 1)", (long long) dim);
    }

    // Determine the inner / outer sign tables based on direction.  Forward
    // applies (s1, fwht, s2); inverse applies (s2, fwht, s1).  Both compose
    // to the identity since H_d^2 = d * I and Diag(s)^2 = I.
    const int8_t * inner = direction == 0
        ? pick_signs(dim, /*s1_or_s2=*/0)   // s1
        : pick_signs(dim, /*s1_or_s2=*/1);  // s2
    const int8_t * outer = direction == 0
        ? pick_signs(dim, /*s1_or_s2=*/1)   // s2
        : pick_signs(dim, /*s1_or_s2=*/0);  // s1
    if (!inner || !outer) {
        GGML_ABORT("reex_turboquant_cpu_wht_apply_row: missing sign table for dim=%lld",
                   (long long) dim);
    }

    // Stage 0 (NEW, P4 A2.4 Stage 2a): pre-multiply by scale_inv.  This is the
    // InnerQ EMA per-channel rescaling — applying it BEFORE the rotation is
    // what makes the invariant <Q ⊙ s_inv, K ⊙ s> = <Q, K> hold (H_d does
    // not commute with Diag(scale_inv)).  Folded with the inner sign step
    // below to avoid an extra dim-sized pass.
    //
    // Stage 1: y = Diag(inner) * (x ⊙ scale_inv) — or just Diag(inner) * x
    // when scale_inv == NULL.  Sign extraction is branch-free per element.
    if (scale_inv != NULL) {
        for (int64_t i = 0; i < dim; ++i) {
            const float v = x[i] * scale_inv[i];
            out[i] = inner[i] >= 0 ? v : -v;
        }
    } else {
        for (int64_t i = 0; i < dim; ++i) {
            out[i] = inner[i] >= 0 ? x[i] : -x[i];
        }
    }

    // Stage 2: y <- H_d * y  (unscaled).
    fwht_inplace(out, dim);

    // Stage 3: y <- Diag(outer) * y.
    for (int64_t i = 0; i < dim; ++i) {
        if (outer[i] < 0) {
            out[i] = -out[i];
        }
    }

    // Stage 4: y <- y / sqrt(dim).
    const float inv_sqrt_d = 1.0f / sqrtf((float) dim);
    for (int64_t i = 0; i < dim; ++i) {
        out[i] = out[i] * inv_sqrt_d;
    }

    // Stage 5 (NEW, P4 A2.4 Stage 2a): per-channel variance accumulator.  We
    // accumulate the SQUARE of the post-rotation output (not its absolute
    // value) so the caller can recover RMS via sqrt(sq_accum / count).
    // Caller is responsible for thread-safety: the GGML_OP_REEX_WHT
    // dispatcher pins to a single thread when sq_accum is non-NULL.
    if (sq_accum != NULL) {
        for (int64_t i = 0; i < dim; ++i) {
            const double v = (double) out[i];
            sq_accum[i] += v * v;
        }
    }
}

void reex_turboquant_cpu_wht_apply_rows(
        const float * GGML_RESTRICT x,
        float       * GGML_RESTRICT out,
        int64_t                      rows,
        int64_t                      dim,
        int                          direction,
        const float * GGML_RESTRICT scale_inv,
        double      *               sq_accum) {
    for (int64_t r = 0; r < rows; ++r) {
        reex_turboquant_cpu_wht_apply_row(
            x   + r * dim,
            out + r * dim,
            dim, direction, scale_inv, sq_accum);
    }

    if (direction == 0 && dim == 128) {
        vwht_dump_maybe_init();
        if (g_vwht_dump_fp != NULL && g_vwht_dump_remaining > 0) {
            pthread_mutex_lock(&g_vwht_dump_mtx);
            for (int64_t r = 0; r < rows && g_vwht_dump_remaining > 0; ++r) {
                vwht_dump_row_locked(out + r * dim, dim);
            }
            pthread_mutex_unlock(&g_vwht_dump_mtx);
        }
    }
}

#endif  // REEX_TURBOQUANT
