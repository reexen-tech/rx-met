#include "intgemm.cuh"
#include "dumper.h"   // make_dirs
#include "convert.cuh" // shared OutConv (rgd_out_write_i32)

#include "ggml.h"
#include "reex/ggml-reex-q64-common.h"   // reex_q64_psum_trunc_b (psum_bits>0)

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

// ---- GPU: pure integer GEMM, C[m,n] = Σ_k A[m,k]*W[n,k] (int32 acc) ----------
__global__ void kernel_int_gemm(
    const int8_t * __restrict__ A, const int8_t * __restrict__ W,
    int32_t * __restrict__ C,
    int64_t M, int64_t N, int64_t K, TilingSpec ts, int psum_bits) {

    const int64_t n = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t m = (int64_t) blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || n >= N) return;

    int32_t acc = 0;
    for (int64_t k = 0; k < K; ++k)
        acc += (int32_t) A[act_elem_slot(m, k, K, ts)] * (int32_t) W[weight_elem_slot(n, k, N, ts)];
    if (psum_bits > 0) acc = reex_q64_psum_trunc_b(acc, psum_bits);
    C[result_tiled_index(m, n, M, ts)] = acc;
}

// ---- helpers ---------------------------------------------------------------
static void iwrite_bin(const std::string & path, const void * data, size_t bytes) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) { fprintf(stderr, "[rgd] cannot write %s\n", path.c_str()); return; }
    if (data && bytes) fwrite(data, 1, bytes, f);
    fclose(f);
}

std::string run_intblock_case(const std::string & out_root, const GemmCase & c,
                              const TilingSpec & ts, const WQuantType & wt) {
    const int64_t M = c.M, N = c.N, K = c.K;
    const int     a_max = (1 << (c.A_bits - 1)) - 1;   // A8 -> 127
    const int     w_max = (1 << (wt.W_bits - 1)) - 1;  // W8 -> 127

    // 1) integer inputs (no scale): A[M,K] in [-a_max,a_max], W[N,K] in [-w_max,w_max]
    std::vector<int8_t> A((size_t) (M * K)), W((size_t) (N * K));
    {
        std::mt19937_64 rng(c.seed);
        std::uniform_int_distribution<int> da(-a_max, a_max);
        for (auto & x : A) x = (int8_t) da(rng);
        std::mt19937_64 rngw(c.seed ^ 0x9E3779B97F4A7C15ULL);
        std::uniform_int_distribution<int> dw(-w_max, w_max);
        for (auto & x : W) x = (int8_t) dw(rngw);
    }

    // 2) pack into [16x16] tile order (act_elem_slot / weight_elem_slot)
    std::vector<int8_t> a_tiled((size_t) (M * K)), w_tiled((size_t) (N * K));
    for (int64_t m = 0; m < M; ++m)
        for (int64_t k = 0; k < K; ++k)
            a_tiled[(size_t) act_elem_slot(m, k, K, ts)] = A[(size_t) (m * K + k)];
    for (int64_t n = 0; n < N; ++n)
        for (int64_t k = 0; k < K; ++k)
            w_tiled[(size_t) weight_elem_slot(n, k, N, ts)] = W[(size_t) (n * K + k)];

    // 3) GPU integer GEMM -> C_gpu (int32, result tile order)
    std::vector<int32_t> C_gpu((size_t) (M * N));
    {
        int8_t * d_a = nullptr; int8_t * d_w = nullptr; int32_t * d_C = nullptr;
        ICUDA_CHECK(cudaMalloc(&d_a, a_tiled.size()));
        ICUDA_CHECK(cudaMalloc(&d_w, w_tiled.size()));
        ICUDA_CHECK(cudaMalloc(&d_C, C_gpu.size() * sizeof(int32_t)));
        ICUDA_CHECK(cudaMemcpy(d_a, a_tiled.data(), a_tiled.size(), cudaMemcpyHostToDevice));
        ICUDA_CHECK(cudaMemcpy(d_w, w_tiled.data(), w_tiled.size(), cudaMemcpyHostToDevice));
        dim3 blk(16, 16);
        dim3 grd((unsigned) ((N + blk.x - 1) / blk.x), (unsigned) ((M + blk.y - 1) / blk.y));
        kernel_int_gemm<<<grd, blk>>>(d_a, d_w, d_C, M, N, K, ts, c.psum_bits);
        ICUDA_CHECK(cudaGetLastError());
        ICUDA_CHECK(cudaDeviceSynchronize());
        ICUDA_CHECK(cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
        cudaFree(d_a); cudaFree(d_w); cudaFree(d_C);
    }

    // 4) CPU integer golden + exact compare
    std::vector<int32_t> C_ref((size_t) (M * N));
    int64_t max_abs = 0;
    for (int64_t m = 0; m < M; ++m)
        for (int64_t n = 0; n < N; ++n) {
            int32_t acc = 0;
            for (int64_t k = 0; k < K; ++k)
                acc += (int32_t) A[(size_t) (m * K + k)] * (int32_t) W[(size_t) (n * K + k)];
            if (c.psum_bits > 0) acc = reex_q64_psum_trunc_b(acc, c.psum_bits);
            const int64_t idx = result_tiled_index(m, n, M, ts);
            C_ref[(size_t) idx] = acc;
            const int64_t d = std::llabs((int64_t) acc - (int64_t) C_gpu[(size_t) idx]);
            if (d > max_abs) max_abs = d;
        }
    fprintf(stderr, "[rgd] verify (int golden, exact): max_abs=%lld\n", (long long) max_abs);

    // 5) dump (timestamped dir)
    char ts_buf[24];
    { std::time_t t = std::time(nullptr); std::tm tmv; localtime_r(&t, &tmv);
      std::strftime(ts_buf, sizeof(ts_buf), "%Y%m%d-%H%M%S", &tmv); }
    const std::string dir = out_root + "/" + c.name + "-" + ts_buf;
    make_dirs(dir);

    iwrite_bin(dir + "/weight_blocks.bin", w_tiled.data(), w_tiled.size());
    iwrite_bin(dir + "/act_blocks.bin",    a_tiled.data(), a_tiled.size());

    // exact raw INT output (the canonical "输出INT数值(不带scale)"; no saturation)
    iwrite_bin(dir + "/output_i32.bin", C_gpu.data(), C_gpu.size() * sizeof(int32_t));

    // output_<DT>.bin: DIRECT cast of the int accumulator to every dtype (NO scale),
    //   via the shared OutConv black box (convert.cuh). float=exact cast; int=clamp.
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

    // exact integer golden as f32 (int values < 2^24 are exact in f32)
    std::vector<float> gold((size_t) (M * N));
    for (int64_t i = 0; i < M * N; ++i) gold[(size_t) i] = (float) C_ref[(size_t) i];
    iwrite_bin(dir + "/golden_f32.bin", gold.data(), gold.size() * sizeof(float));

    // meta.json (self-describing)
    {
        const int64_t Mtiles = M / ts.Mt, Ntiles = N / ts.Nt, Ktiles = K / ts.Kt;
        const int64_t blk_elems = (int64_t) ts.Nt * ts.Kt;     // 16*16 = 256
        const int64_t n_wblk = N * K / blk_elems;
        const int64_t n_ablk = M * K / ((int64_t) ts.Mt * ts.Kt);
        FILE * j = fopen((dir + "/meta.json").c_str(), "w");
        if (j) {
            fprintf(j,
                "{\n"
                "  \"name\": \"%s\",\n"
                "  \"gemm\": {\n"
                "    \"_doc\": \"PURE INTEGER matmul, no scale; C = A @ W^T accumulated in int32\",\n"
                "    \"formula\": \"C[M,N] = A[M,K] @ W[N,K]^T  (integer)\",\n"
                "    \"M\": %lld, \"N\": %lld, \"K\": %lld\n"
                "  },\n"
                "  \"quant\": {\n"
                "    \"weight_type\": \"%s\", \"weight_family\": \"IntBlock\", \"weight_bits\": %d, \"scale\": \"none\",\n"
                "    \"act_dtype\": \"int%d (raw, no scale)\", \"output_dtypes\": \"i32(exact) + F16,BF16,I16,I8,I6,I4 (direct cast/clamp, no scale)\",\n"
                "    \"psum_trunc_bits\": %d, \"_psum_doc\": \"0 = full int32 accumulator, no truncation\", \"seed\": %llu\n"
                "  },\n"
                "  \"tiling\": {\n"
                "    \"_doc\": \"block-wise [16x16]; act [Mt(M) x Kt(K)], weight [Nt(N) x Kt(K)], result element\",\n"
                "    \"Mt\": %d, \"Nt\": %d, \"Kt\": %d,\n"
                "    \"Mtiles\": %lld, \"Ntiles\": %lld, \"Ktiles\": %lld\n"
                "  },\n"
                "  \"files\": {\n"
                "    \"weight_blocks.bin\": {\n"
                "      \"_doc\": \"raw INT weights W[N,K], no scale, block-wise [16 N-rows x 16 K-cols]\",\n"
                "      \"dtype\": \"int8\", \"total_bytes\": %lld, \"block_bytes\": %lld, \"num_blocks\": %lld,\n"
                "      \"block_covers\": \"16 weight-rows(N) x 16 K-elems\", \"byte_layout\": \"i8[256] = [16 N-rows x 16 K-cols] row-major\",\n"
                "      \"elem_index\": \"slot(n,k) = ((k/Kt*Ntiles + n/Nt)*Nt + n%%Nt)*Kt + k%%Kt\"\n"
                "    },\n"
                "    \"act_blocks.bin\": {\n"
                "      \"_doc\": \"raw INT activations A[M,K], no scale, block-wise [16 M-rows x 16 K-cols]\",\n"
                "      \"dtype\": \"int8\", \"total_bytes\": %lld, \"block_bytes\": %lld, \"num_blocks\": %lld,\n"
                "      \"block_covers\": \"16 act-rows(M) x 16 K-elems\", \"byte_layout\": \"i8[256] = [16 M-rows x 16 K-cols] row-major\",\n"
                "      \"elem_index\": \"slot(m,k) = ((m/Mt*Ktiles + k/Kt)*Mt + m%%Mt)*Kt + k%%Kt\"\n"
                "    },\n"
                "    \"output_i32.bin\": {\n"
                "      \"_doc\": \"EXACT raw INT accumulator (no scale, no saturation) - the canonical integer output\",\n"
                "      \"dtype\": \"int32\", \"total_bytes\": %lld, \"shape\": [%lld, %lld],\n"
                "      \"elem_index\": \"idx(m,n) = (n/Nt*Mtiles + m/Mt)*(Mt*Nt) + (m%%Mt)*Nt + n%%Nt\"\n"
                "    },\n"
                "    \"output_<DT>.bin\": {\n"
                "      \"_doc\": \"GPU int accumulator cast/clamped directly to each DT in [F16,BF16,I16,I8,I6,I4] (NO scale); F16/BF16 saturate beyond range, int clamps. I6/I4 in int8 container. Prefer output_i32.bin\",\n"
                "      \"dtypes\": [\"F16\",\"BF16\",\"I16\",\"I8\",\"I6\",\"I4\"], \"int_saturate\": {\"I16\":[-32768,32767],\"I8\":[-128,127],\"I6\":[-32,31],\"I4\":[-8,7]},\n"
                "      \"shape\": [%lld, %lld], \"elem_index\": \"same as output_i32.bin\"\n"
                "    },\n"
                "    \"golden_f32.bin\": {\n"
                "      \"_doc\": \"EXACT int accumulator as f32 (exact for |C| < 2^24), same order as output\",\n"
                "      \"dtype\": \"f32\", \"total_bytes\": %lld, \"shape\": [%lld, %lld],\n"
                "      \"elem_index\": \"same as output_i32.bin\"\n"
                "    }\n"
                "  },\n"
                "  \"verify\": { \"_doc\": \"GPU vs int golden, exact integer compare\", \"max_abs\": %lld }\n"
                "}\n",
                c.name.c_str(),
                (long long) M, (long long) N, (long long) K,
                wt.name, wt.W_bits, c.A_bits,
                c.psum_bits, (unsigned long long) c.seed,
                ts.Mt, ts.Nt, ts.Kt,
                (long long) Mtiles, (long long) Ntiles, (long long) Ktiles,
                (long long) w_tiled.size(), (long long) blk_elems, (long long) n_wblk,
                (long long) a_tiled.size(), (long long) ((int64_t) ts.Mt * ts.Kt), (long long) n_ablk,
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
