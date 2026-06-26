#include "intgemm.cuh"
#include "dumper.h"   // make_dirs
#include "convert.cuh" // shared OutConv (rgd_out_write_i32)

#include "ggml.h"
#include "reex/ggml-reex-q64-common.h"   // reex_q64_psum_trunc_b (psum_bits>0)

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <random>
#include <vector>
#include <cuda_runtime.h>

namespace rgd {

#define ICUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "[rgd] CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
    abort(); } } while (0)

// ---- GPU: pure integer GEMM, C[m,n] = Σ_k A[m,k]*W[n,k] -----------------------
// Accumulate in int32 with STEP-WISE SATURATION: after every MAC the running sum
// is clamped to [INT32_MIN, INT32_MAX]. GPU and CPU golden walk k in the same
// order (0..K-1) so the saturated results are bit-exact. A/W are row-major raw
// integer values (signed or unsigned, already in their bit-width range).
__global__ void kernel_int_gemm(
    const int32_t * __restrict__ A, const int32_t * __restrict__ W,
    int32_t * __restrict__ C,
    int64_t M, int64_t N, int64_t K, TilingSpec ts, int psum_bits) {

    const int64_t n = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t m = (int64_t) blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || n >= N) return;

    long long acc = 0;
    const long long HI = 2147483647LL, LO = -2147483648LL;
    const int32_t * arow = A + (size_t) m * K;
    const int32_t * wrow = W + (size_t) n * K;
    for (int64_t k = 0; k < K; ++k) {
        acc += (long long) arow[k] * (long long) wrow[k];
        if (acc > HI) acc = HI;
        if (acc < LO) acc = LO;
    }
    int32_t r = (int32_t) acc;
    if (psum_bits > 0) r = reex_q64_psum_trunc_b(r, psum_bits);
    C[result_tiled_index(m, n, M, ts)] = r;
}

// ---- helpers ---------------------------------------------------------------
static void iwrite_bin(const std::string & path, const void * data, size_t bytes) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) { fprintf(stderr, "[rgd] cannot write %s\n", path.c_str()); return; }
    if (data && bytes) fwrite(data, 1, bytes, f);
    fclose(f);
}

// LSB-first bit writer into a pre-zeroed byte buffer (absolute bit offset).
static inline void put_bits(uint8_t * buf, int64_t bitpos, uint32_t val, int nbits) {
    for (int b = 0; b < nbits; ++b, ++bitpos)
        if ((val >> b) & 1u) buf[bitpos >> 3] |= (uint8_t) (1u << (bitpos & 7));
}

std::string run_intblock_case(const std::string & out_root, const GemmCase & c,
                              const TilingSpec & ts, const WQuantType & wt) {
    (void) wt;
    const int64_t M = c.M, N = c.N, K = c.K;
    const int  ab = c.A_bits, wb = c.w_bits;
    const bool a_uns = c.a_unsigned, w_uns = c.w_unsigned;

    // value ranges: signed -> two's complement [-2^(b-1), 2^(b-1)-1]; unsigned -> [0, 2^b-1]
    const int a_lo = a_uns ? 0 : -(1 << (ab - 1));
    const int a_hi = a_uns ? ((1 << ab) - 1) : ((1 << (ab - 1)) - 1);
    const int w_lo = w_uns ? 0 : -(1 << (wb - 1));
    const int w_hi = w_uns ? ((1 << wb) - 1) : ((1 << (wb - 1)) - 1);

    // 1) random integer inputs (no scale), row-major A[M,K], W[N,K]
    std::vector<int32_t> A((size_t) (M * K)), W((size_t) (N * K));
    {
        std::mt19937_64 rng(c.seed);
        std::uniform_int_distribution<int> da(a_lo, a_hi);
        for (auto & x : A) x = (int32_t) da(rng);
        std::mt19937_64 rngw(c.seed ^ 0x9E3779B97F4A7C15ULL);
        std::uniform_int_distribution<int> dw(w_lo, w_hi);
        for (auto & x : W) x = (int32_t) dw(rngw);
    }

    // 2) GPU integer GEMM (row-major inputs) -> C_gpu (int32, result tile order)
    std::vector<int32_t> C_gpu((size_t) (M * N));
    {
        int32_t * d_a = nullptr; int32_t * d_w = nullptr; int32_t * d_C = nullptr;
        ICUDA_CHECK(cudaMalloc(&d_a, A.size() * sizeof(int32_t)));
        ICUDA_CHECK(cudaMalloc(&d_w, W.size() * sizeof(int32_t)));
        ICUDA_CHECK(cudaMalloc(&d_C, C_gpu.size() * sizeof(int32_t)));
        ICUDA_CHECK(cudaMemcpy(d_a, A.data(), A.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
        ICUDA_CHECK(cudaMemcpy(d_w, W.data(), W.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
        dim3 blk(16, 16);
        dim3 grd((unsigned) ((N + blk.x - 1) / blk.x), (unsigned) ((M + blk.y - 1) / blk.y));
        kernel_int_gemm<<<grd, blk>>>(d_a, d_w, d_C, M, N, K, ts, c.psum_bits);
        ICUDA_CHECK(cudaGetLastError());
        ICUDA_CHECK(cudaDeviceSynchronize());
        ICUDA_CHECK(cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
        cudaFree(d_a); cudaFree(d_w); cudaFree(d_C);
    }

    // 3) CPU integer golden (same step-wise saturating accumulation) + exact compare
    std::vector<int32_t> C_ref((size_t) (M * N));
    int64_t max_abs = 0;
    for (int64_t m = 0; m < M; ++m)
        for (int64_t n = 0; n < N; ++n) {
            long long acc = 0;
            const long long HI = 2147483647LL, LO = -2147483648LL;
            const int32_t * arow = A.data() + (size_t) m * K;
            const int32_t * wrow = W.data() + (size_t) n * K;
            for (int64_t k = 0; k < K; ++k) {
                acc += (long long) arow[k] * (long long) wrow[k];
                if (acc > HI) acc = HI;
                if (acc < LO) acc = LO;
            }
            int32_t r = (int32_t) acc;
            if (c.psum_bits > 0) r = reex_q64_psum_trunc_b(r, c.psum_bits);
            const int64_t idx = result_tiled_index(m, n, M, ts);
            C_ref[(size_t) idx] = r;
            const int64_t d = std::llabs((int64_t) r - (int64_t) C_gpu[(size_t) idx]);
            if (d > max_abs) max_abs = d;
        }
    fprintf(stderr, "[rgd] verify (int golden, exact): max_abs=%lld\n", (long long) max_abs);

    // 4) bit-pack inputs into [16x16] block tile order at exact bit-width.
    //    activation: intra-block ROW-major (act_elem_slot); weight: intra-block
    //    COLUMN-major (weight_elem_slot). Continuous LSB-first stream (each block
    //    is 256*bits bits = byte-aligned). signed -> two's complement low bits.
    const uint32_t a_mask = (ab >= 32) ? 0xFFFFFFFFu : ((1u << ab) - 1u);
    const uint32_t w_mask = (wb >= 32) ? 0xFFFFFFFFu : ((1u << wb) - 1u);
    std::vector<uint8_t> a_packed((size_t) ((M * K * ab + 7) / 8), 0);
    std::vector<uint8_t> w_packed((size_t) ((N * K * wb + 7) / 8), 0);
    for (int64_t m = 0; m < M; ++m)
        for (int64_t k = 0; k < K; ++k) {
            const int64_t slot = act_elem_slot(m, k, K, ts);
            put_bits(a_packed.data(), slot * ab, (uint32_t) A[(size_t) (m * K + k)] & a_mask, ab);
        }
    for (int64_t n = 0; n < N; ++n)
        for (int64_t k = 0; k < K; ++k) {
            const int64_t slot = weight_elem_slot(n, k, N, ts);
            put_bits(w_packed.data(), slot * wb, (uint32_t) W[(size_t) (n * K + k)] & w_mask, wb);
        }

    // 5) dump (timestamped dir)
    char ts_buf[24];
    { std::time_t t = std::time(nullptr); std::tm tmv; localtime_r(&t, &tmv);
      std::strftime(ts_buf, sizeof(ts_buf), "%Y%m%d-%H%M%S", &tmv); }
    const std::string dir = out_root + "/" + c.name + "-" + ts_buf;
    make_dirs(dir);

    iwrite_bin(dir + "/weight_blocks.bin", w_packed.data(), w_packed.size());
    iwrite_bin(dir + "/act_blocks.bin",    a_packed.data(), a_packed.size());

    // exact saturated INT32 output (the canonical "输出INT数值", no per-dtype clamp)
    iwrite_bin(dir + "/output_i32.bin", C_gpu.data(), C_gpu.size() * sizeof(int32_t));

    // output_<DT>.bin: the saturated int32 cast/clamped to each output dtype (NO scale),
    //   via the shared OutConv (convert.cuh). float = exact cast (fp16/bf16 saturate);
    //   int = clamp/saturate into the container range.
    {
        int nsp; const OutSpec * sp = out_specs(nsp);
        for (int s = 0; s < nsp; ++s) {
            const OutSpec & o = sp[s];
            std::vector<uint8_t> obuf((size_t) (M * N) * o.bytes);
            for (int64_t i = 0; i < M * N; ++i) {
                rgd_out_write_i32(obuf.data() + (size_t) i * o.bytes,
                                  o.dt, o.is_float, o.bytes, o.qmax, C_gpu[(size_t) i]);
            }
            iwrite_bin(dir + "/output_" + o.name + ".bin", obuf.data(), obuf.size());
        }
    }

    // saturated integer golden as f32 (exact only for |C| < 2^24; output_i32 is canonical)
    std::vector<float> gold((size_t) (M * N));
    for (int64_t i = 0; i < M * N; ++i) gold[(size_t) i] = (float) C_ref[(size_t) i];
    iwrite_bin(dir + "/golden_f32.bin", gold.data(), gold.size() * sizeof(float));

    // meta.json (self-describing)
    {
        const int64_t Mtiles = M / ts.Mt, Ntiles = N / ts.Nt, Ktiles = K / ts.Kt;
        const int64_t blk_elems = (int64_t) ts.Nt * ts.Kt;     // 16*16 = 256
        const int64_t n_wblk = N * K / blk_elems;
        const int64_t n_ablk = M * K / ((int64_t) ts.Mt * ts.Kt);
        const int     w_blk_bytes = (int) (blk_elems * wb / 8);                 // 256*wb/8
        const int     a_blk_bytes = (int) (((int64_t) ts.Mt * ts.Kt) * ab / 8); // 256*ab/8
        const char *  a_sgn = a_uns ? "uint" : "int(two's complement)";
        const char *  w_sgn = w_uns ? "uint" : "int(two's complement)";
        FILE * j = fopen((dir + "/meta.json").c_str(), "w");
        if (j) {
            fprintf(j,
                "{\n"
                "  \"name\": \"%s\",\n"
                "  \"gemm\": {\n"
                "    \"_doc\": \"PURE INTEGER matmul, no scale; C = A @ W^T, int32 accumulator with STEP-WISE saturation\",\n"
                "    \"formula\": \"C[M,N] = sat_i32( A[M,K] @ W[N,K]^T )  (clamp to int32 after every MAC)\",\n"
                "    \"M\": %lld, \"N\": %lld, \"K\": %lld\n"
                "  },\n"
                "  \"quant\": {\n"
                "    \"family\": \"IntBlock\", \"scale\": \"none\",\n"
                "    \"act_dtype\": \"%s %d-bit\",  \"act_range\": [%d, %d],\n"
                "    \"weight_dtype\": \"%s %d-bit\", \"weight_range\": [%d, %d],\n"
                "    \"output_dtypes\": \"i32(saturated, canonical) + F16,BF16,I16,I8,I6,I4 (cast/clamp from the saturated i32, no scale)\",\n"
                "    \"psum_trunc_bits\": %d, \"_psum_doc\": \"0 = int32 accumulator only (step-wise saturated)\", \"seed\": %llu\n"
                "  },\n"
                "  \"tiling\": {\n"
                "    \"_doc\": \"block-wise [16x16], bit-packed at exact bit-width (LSB-first, each block byte-aligned)\",\n"
                "    \"Mt\": %d, \"Nt\": %d, \"Kt\": %d,\n"
                "    \"Mtiles\": %lld, \"Ntiles\": %lld, \"Ktiles\": %lld,\n"
                "    \"act_order\":    \"intra-block ROW-major (M-row major, K contiguous); inter-block (mt,kt) row-major\",\n"
                "    \"weight_order\": \"intra-block COLUMN-major (K-col major, N contiguous); inter-block (kt,nt) row-major\",\n"
                "    \"result_order\": \"intra-block row-major; inter-block (nt,mt) column-major\"\n"
                "  },\n"
                "  \"files\": {\n"
                "    \"weight_blocks.bin\": {\n"
                "      \"_doc\": \"raw INT weights W[N,K], no scale, bit-packed [16 N-rows x 16 K-cols] blocks, intra-block COLUMN(=N)-major\",\n"
                "      \"dtype\": \"%s %d-bit (bit-packed)\", \"total_bytes\": %lld, \"block_bytes\": %d, \"num_blocks\": %lld,\n"
                "      \"block_covers\": \"16 weight-rows(N) x 16 K-elems\",\n"
                "      \"bit_index\": \"bitpos(n,k) = weight_elem_slot(n,k)*Wbits;  slot = ((k/16*Ntiles + n/16)*16 + k%%16)*16 + n%%16\"\n"
                "    },\n"
                "    \"act_blocks.bin\": {\n"
                "      \"_doc\": \"raw INT activations A[M,K], no scale, bit-packed [16 M-rows x 16 K-cols] blocks, intra-block ROW(=M)-major\",\n"
                "      \"dtype\": \"%s %d-bit (bit-packed)\", \"total_bytes\": %lld, \"block_bytes\": %d, \"num_blocks\": %lld,\n"
                "      \"block_covers\": \"16 act-rows(M) x 16 K-elems\",\n"
                "      \"bit_index\": \"bitpos(m,k) = act_elem_slot(m,k)*Abits;  slot = ((m/16*Ktiles + k/16)*16 + m%%16)*16 + k%%16\"\n"
                "    },\n"
                "    \"output_i32.bin\": {\n"
                "      \"_doc\": \"saturated INT32 accumulator (canonical integer output)\",\n"
                "      \"dtype\": \"int32\", \"total_bytes\": %lld, \"shape\": [%lld, %lld],\n"
                "      \"elem_index\": \"idx(m,n) = (n/Nt*Mtiles + m/Mt)*(Mt*Nt) + (m%%Mt)*Nt + n%%Nt\"\n"
                "    },\n"
                "    \"output_<DT>.bin\": {\n"
                "      \"_doc\": \"saturated i32 cast/clamped to each DT in [F16,BF16,I16,I8,I6,I4] (NO scale); F16/BF16 saturate beyond range, int clamps. I6/I4 in int8 container\",\n"
                "      \"dtypes\": [\"F16\",\"BF16\",\"I16\",\"I8\",\"I6\",\"I4\"], \"int_saturate\": {\"I16\":[-32768,32767],\"I8\":[-128,127],\"I6\":[-32,31],\"I4\":[-8,7]},\n"
                "      \"shape\": [%lld, %lld], \"elem_index\": \"same as output_i32.bin\"\n"
                "    },\n"
                "    \"golden_f32.bin\": {\n"
                "      \"_doc\": \"saturated int accumulator as f32 (exact for |C| < 2^24), same order as output; prefer output_i32.bin\",\n"
                "      \"dtype\": \"f32\", \"total_bytes\": %lld, \"shape\": [%lld, %lld],\n"
                "      \"elem_index\": \"same as output_i32.bin\"\n"
                "    }\n"
                "  },\n"
                "  \"verify\": { \"_doc\": \"GPU vs int golden, exact integer compare\", \"max_abs\": %lld }\n"
                "}\n",
                c.name.c_str(),
                (long long) M, (long long) N, (long long) K,
                a_sgn, ab, a_lo, a_hi,
                w_sgn, wb, w_lo, w_hi,
                c.psum_bits, (unsigned long long) c.seed,
                ts.Mt, ts.Nt, ts.Kt,
                (long long) Mtiles, (long long) Ntiles, (long long) Ktiles,
                w_sgn, wb, (long long) w_packed.size(), w_blk_bytes, (long long) n_wblk,
                a_sgn, ab, (long long) a_packed.size(), a_blk_bytes, (long long) n_ablk,
                (long long) (C_gpu.size() * sizeof(int32_t)), (long long) M, (long long) N,
                (long long) M, (long long) N,
                (long long) (gold.size() * sizeof(float)), (long long) M, (long long) N,
                (long long) max_abs);
            fclose(j);
        }
    }

    return dir;
}

} // namespace rgd
