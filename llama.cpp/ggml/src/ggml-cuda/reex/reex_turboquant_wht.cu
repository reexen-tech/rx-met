// SPDX-License-Identifier: MIT
//
// reex_turboquant_wht.cu — REEX_TURBOQUANT only.
//
// CUDA implementation of GGML_OP_REEX_WHT (see header for the algorithm).
// Algorithm and bit-level conventions exactly mirror
// `ggml/src/ggml-cpu/reex/reex_turboquant_wht.c`; cross-checked by
// `tests/reex/test-reex-turboquant-cuda-wht-parity.cpp`.

#include "reex_turboquant_wht.cuh"

#ifdef REEX_TURBOQUANT

#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include <cmath>

// --------------------------------------------------------------------------
// atomicAdd(double*, double) compat shim (P4 A2.4 Stage 4)
//
// Native `atomicAdd(double *, double)` requires sm_60+.  Some build trees
// in this repo are configured for older arch lists (no explicit
// CMAKE_CUDA_ARCHITECTURES → cmake falls back to a wider compatibility
// list).  Provide a CAS-based emulation when targeting < sm_60 so the
// kernel template instantiates regardless of arch; the InnerQ EMA path
// is gated by `ggml_backend_cuda_supports_op` which is sm-agnostic but
// the actual atomicAdd hot-path only runs when sq_accum != nullptr (i.e.
// during the InnerQ calibration window — first ~1024 tokens per chunk).
// --------------------------------------------------------------------------

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
__device__ static inline double reex_atomicAddDouble(double * address, double val) {
    unsigned long long * address_as_ull = (unsigned long long *) address;
    unsigned long long old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#else
__device__ static inline double reex_atomicAddDouble(double * address, double val) {
    return atomicAdd(address, val);
}
#endif

// --------------------------------------------------------------------------
// Sign tables (mirror of ggml-cpu/reex/reex_turboquant_wht_signs.inc).
//
// We store +1/-1 as int8_t to match the CPU reference.  The CPU `.inc` lives
// in a different TU and is `static`, so we re-include it here directly: the
// auto-generated file is small and re-inclusion guarantees the two backends
// are byte-identical (any change to the generator will recompile both TUs
// against the new signs).
// --------------------------------------------------------------------------

#include "../../ggml-cpu/reex/reex_turboquant_wht_signs.inc"

// Mirror s1/s2 sign tables into __constant__ device memory.
// We copy them from the host arrays at first launch.

static __constant__ int8_t reex_wht_s1_d128_d[128];
static __constant__ int8_t reex_wht_s2_d128_d[128];
static __constant__ int8_t reex_wht_s1_d64_d [64];
static __constant__ int8_t reex_wht_s2_d64_d [64];

static bool g_reex_wht_constants_loaded = false;

static void reex_wht_ensure_constants_loaded() {
    if (g_reex_wht_constants_loaded) return;
    CUDA_CHECK(cudaMemcpyToSymbol(reex_wht_s1_d128_d, reex_turboquant_wht_signs_s1_d128, 128 * sizeof(int8_t)));
    CUDA_CHECK(cudaMemcpyToSymbol(reex_wht_s2_d128_d, reex_turboquant_wht_signs_s2_d128, 128 * sizeof(int8_t)));
    CUDA_CHECK(cudaMemcpyToSymbol(reex_wht_s1_d64_d,  reex_turboquant_wht_signs_s1_d64,   64 * sizeof(int8_t)));
    CUDA_CHECK(cudaMemcpyToSymbol(reex_wht_s2_d64_d,  reex_turboquant_wht_signs_s2_d64,   64 * sizeof(int8_t)));
    g_reex_wht_constants_loaded = true;
}

// --------------------------------------------------------------------------
// FWHT row kernel.
//
// One block per row, blockDim.x == DIM threads.  Each thread holds one
// element in its register `v`.  The butterfly stages walk through pair
// distances h = 1, 2, 4, ..., DIM/2; at every stage thread `tid` cooperates
// with `tid ^ h` via shared memory:
//
//     a, b = v[tid], v[tid ^ h]
//     v[tid]      = (tid < tid^h) ? a + b : (b ... wait we need ordering)
//
// To stay exactly bit-equal with the CPU (which iterates strictly
//   for j in [i, i+h):  a[j] = a[j] + a[j+h];  a[j+h] = a[j] - a[j+h];
//   ^^ note: read a[j+h] BEFORE the first store on a[j], otherwise wrong)
// we use a clean shared-mem two-step:
//   1) load `a, b = sm[tid], sm[tid ^ h]`,
//   2) sync, write back: if (tid & h) -> a - b   else -> a + b.
// This is symmetric across all threads and avoids the read-after-write
// hazard while producing the SAME numerical result as the CPU loop, which
// for floats holds because (x + y) and (x - y) are the only two ops; their
// rounding is order-independent in IEEE-754 single precision since both
// inputs come from the same level.
// --------------------------------------------------------------------------

template <int DIM>
__device__ static __forceinline__ const int8_t * reex_wht_pick_sign_d(int s1_or_s2) {
    if constexpr (DIM == 128) {
        return s1_or_s2 == 0 ? reex_wht_s1_d128_d : reex_wht_s2_d128_d;
    } else {
        return s1_or_s2 == 0 ? reex_wht_s1_d64_d : reex_wht_s2_d64_d;
    }
}

template <int DIM>
__global__ static void reex_kernel_wht_row(
        const float * __restrict__ x,
        float       * __restrict__ y,
        int                          direction,
        const float * __restrict__ scale_inv,
        double      * __restrict__ sq_accum) {
    static_assert(DIM == 64 || DIM == 128, "REEX_WHT CUDA: DIM must be 64 or 128");

    const int   tid = threadIdx.x;
    const int64_t row = blockIdx.x;
    const float * x_row = x + row * DIM;
    float       * y_row = y + row * DIM;

    __shared__ float sm[DIM];

    // Pick inner / outer sign tables exactly like the CPU path:
    //   forward (direction == 0):  inner = s1,  outer = s2
    //   inverse (direction == 1):  inner = s2,  outer = s1
    const int8_t * inner = direction == 0 ? reex_wht_pick_sign_d<DIM>(0) : reex_wht_pick_sign_d<DIM>(1);
    const int8_t * outer = direction == 0 ? reex_wht_pick_sign_d<DIM>(1) : reex_wht_pick_sign_d<DIM>(0);

    // Stage 0 (NEW, P4 A2.4 Stage 2a): pre-multiply by scale_inv for InnerQ
    // EMA's per-channel rescaling.  Pre-rotation placement preserves the
    // inner-product invariant — see CPU comments and reex_turboquant_wht.h.
    // Stage 1: y = Diag(inner) * (x ⊙ scale_inv).  Load to shared memory.
    const float xi = (scale_inv != nullptr) ? (x_row[tid] * scale_inv[tid]) : x_row[tid];
    float v = inner[tid] >= 0 ? xi : -xi;
    sm[tid] = v;
    __syncthreads();

    // Stage 2: in-place FWHT (Sylvester) using shared memory.
    //
    // For h = 1, 2, 4, ..., DIM/2:
    //   pair = tid ^ h
    //   if (tid & h) == 0  -> v_new = sm[tid] + sm[pair]
    //   else               -> v_new = sm[pair] - sm[tid]   (i.e. -(sm[tid]-sm[pair]))
    // which matches the CPU butterfly:
    //   a[j]      = a[j] + a[j+h]
    //   a[j+h]    = a[j_old] - a[j+h]   <-- but j_old still equals a[j] before the new store
    //
    // To keep both legs of the pair using the SAME pair of pre-stage values
    // (just like the CPU loop reads a[j] and a[j+h] up-front), we read into
    // local registers `a, b` BEFORE writing.
#pragma unroll
    for (int h = 1; h < DIM; h <<= 1) {
        const int pair = tid ^ h;
        const float a  = sm[tid];
        const float b  = sm[pair];
        __syncthreads();
        // (tid & h) == 0 means we're the "low" half-pair; +.  Else -.
        v = (tid & h) == 0 ? (a + b) : (b - a);
        sm[tid] = v;
        __syncthreads();
    }

    // Stage 3: y <- Diag(outer) * y.
    if (outer[tid] < 0) {
        v = -v;
    }

    // Stage 4: y <- y / sqrt(DIM).  scale_inv was already applied at Stage 0
    // (pre-rotation), so this is a plain normalisation.  We mirror the CPU
    // code's exact ordering: `inv_sqrt_d = 1.0f / sqrtf(d)` (NOT `rsqrtf`,
    // which would invoke the CUDA fast-path that is not always bit-equal to
    // the IEEE-754-correct sqrt+divide).  This keeps cross-backend numerical
    // agreement on dim ∈ {64, 128} where 64 yields an exact 1/8 and 128
    // yields a single-precision rounded 1/sqrt(128).
    const float inv_sqrt_d = 1.0f / sqrtf((float) DIM);
    v = v * inv_sqrt_d;

    y_row[tid] = v;

    // Stage 5 (P4 A2.4 Stage 4): per-channel variance accumulator.
    // Mirrors the CPU code's order: square the post-rotation, post-1/sqrt(d)
    // output and atomically add into `sq_accum[tid]`.  Each block accumulates
    // its own row's contribution into the shared per-channel slot.
    //
    // Numerical contract (matches CPU `reex_turboquant_cpu_wht_apply_row`):
    //   sq_accum[j] += (double)(y[j]) * (double)(y[j])
    // Atomicity is provided by `atomicAdd(double *, double)` (sm_60+).  RTX
    // 30/40/50/60 series and Ampere/Ada/Hopper/Blackwell all support this.
    if (sq_accum != nullptr) {
        const double vd = (double) v;
        reex_atomicAddDouble(&sq_accum[tid], vd * vd);
    }
}

void reex_turboquant_cuda_wht_apply_rows(
        const float * x_d,
        float       * y_d,
        int64_t       rows,
        int64_t       dim,
        int           direction,
        const float * scale_inv_d,
        double      * sq_accum_d,
        cudaStream_t  stream) {
    GGML_ASSERT(direction == 0 || direction == 1);
    GGML_ASSERT(dim == 64 || dim == 128);
    GGML_ASSERT(rows >= 0);
    if (rows == 0) {
        return;
    }

    reex_wht_ensure_constants_loaded();

    if (dim == 128) {
        reex_kernel_wht_row<128><<<(unsigned int) rows, 128, 0, stream>>>(
            x_d, y_d, direction, scale_inv_d, sq_accum_d);
    } else {
        reex_kernel_wht_row<64><<<(unsigned int) rows, 64, 0, stream>>>(
            x_d, y_d, direction, scale_inv_d, sq_accum_d);
    }
}

// --------------------------------------------------------------------------
// GGML compute-forward bridge.
//
// Mirrors the CPU `case GGML_OP_REEX_WHT` in ggml-cpu.c: reads op_params
// (direction, group_size, has_scale), validates, and dispatches.
// --------------------------------------------------------------------------

void ggml_cuda_op_reex_wht(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(src0 != nullptr);
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int direction    = ggml_get_op_params_i32(dst, 0);
    const int group_size   = ggml_get_op_params_i32(dst, 1);
    const int has_scale    = ggml_get_op_params_i32(dst, 2);
    const int has_sq_accum = ggml_get_op_params_i32(dst, 3);

    GGML_ASSERT(direction == 0 || direction == 1);
    GGML_ASSERT(group_size == 64 || group_size == 128);
    GGML_ASSERT(src0->ne[0] % group_size == 0);

    const int64_t total = ggml_nelements(src0);
    const int64_t rows  = total / group_size;

    cudaStream_t stream = ctx.stream();

    const float * src_d = (const float *) src0->data;
    float       * dst_d = (float       *) dst->data;

    const float * scale_d = nullptr;
    // If scale is F32 on device we can use it directly. F16 fallback is rare
    // (the WHT op is only emitted on F32 paths in build_attn) — assert it.
    if (has_scale) {
        const ggml_tensor * src1 = dst->src[1];
        GGML_ASSERT(src1 != nullptr);
        GGML_ASSERT(src1->type == GGML_TYPE_F32);
        GGML_ASSERT(src1->ne[0] == group_size);
        scale_d = (const float *) src1->data;
    }

    // P4 A2.4 Stage 4: optional InnerQ EMA calibration accumulator.  When
    // src[2] is present it MUST be F64, ne[0] == group_size, and
    // ggml_backend_cuda_supports_op already verified that the scheduler may
    // legally place this op on CUDA (i.e. src[2] lives in CUDA memory so
    // atomicAdd into it is well-defined).  See ggml-cuda.cu's REEX_WHT case
    // for the supports_op contract and reex_kernel_wht_row above for the
    // numerical contract (matches the CPU `+= (double)y * (double)y`).
    double * sq_accum_d = nullptr;
    if (has_sq_accum) {
        const ggml_tensor * src2 = dst->src[2];
        GGML_ASSERT(src2 != nullptr);
        GGML_ASSERT(src2->type == GGML_TYPE_F64);
        GGML_ASSERT(src2->ne[0] == group_size);
        sq_accum_d = (double *) src2->data;
    }

    reex_turboquant_cuda_wht_apply_rows(src_d, dst_d, rows, group_size, direction, scale_d, sq_accum_d, stream);
}

#endif  // REEX_TURBOQUANT
