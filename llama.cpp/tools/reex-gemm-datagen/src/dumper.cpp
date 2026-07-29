#include "dumper.h"
#include "fp8.h"

#include "ggml.h"

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <sys/stat.h>
#include <sys/types.h>

namespace rgd {

bool make_dirs(const std::string & path) {
    std::string tmp = path;
    for (size_t i = 1; i < tmp.size(); ++i) {
        if (tmp[i] == '/') {
            tmp[i] = '\0';
            mkdir(tmp.c_str(), 0777);
            tmp[i] = '/';
        }
    }
    return mkdir(tmp.c_str(), 0777) == 0 || errno == EEXIST;
}

static void write_bin(const std::string & path, const void * data, size_t bytes) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) { fprintf(stderr, "[rgd] cannot write %s\n", path.c_str()); return; }
    if (data && bytes) fwrite(data, 1, bytes, f);
    fclose(f);
}

std::string dump_case(
    const std::string & out_root, const GemmCase & c,
    const TilingSpec & ts, const WQuantType & wt,
    const void * w_blocks, size_t w_blocks_bytes,
    const char * w_layout_desc,
    const std::vector<uint8_t> & a_blocks,
    const std::vector<float> & A_src,
    const std::vector<float> & W_src,
    const std::vector<uint8_t> * out_bufs,
    const std::vector<float> & C_ref_tiled,
    const CaseError & err) {

    const int64_t M = c.M, N = c.N, K = c.K;

    // timestamped artifact dir: <case_name>-YYYYMMDD-HHMMSS
    char ts_buf[24];
    {
        std::time_t t = std::time(nullptr);
        std::tm     tmv;
        localtime_r(&t, &tmv);
        std::strftime(ts_buf, sizeof(ts_buf), "%Y%m%d-%H%M%S", &tmv);
    }
    const std::string dir = out_root + "/" + c.name + "-" + ts_buf;
    make_dirs(dir);

    // ---- inputs: packed blocks (scales embedded), already §4.1 order ----
    write_bin(dir + "/weight_blocks.bin", w_blocks, w_blocks_bytes);
    write_bin(dir + "/act_blocks.bin", a_blocks.data(), a_blocks.size());

    // ---- source inputs (pre-quant), §4.1 tile order (same tiling as the quantized
    //      blocks). Activation is stored in its NATIVE act_in dtype; weight as fp16.
    //        act:    slot = act_group_slot(m, k/Kt)*Kt + k%Kt     tile [Mt x Kt]
    //        weight: slot = weight_block_slot(n, k/Kt)*Kt + k%Kt  tile [Nt x Kt]
    const int    asrc_esz = act_src_elem_bytes(c.act_in);
    const char * asrc_nm  = actdtype_name(c.act_in);
    {
        const int agroup = ts.agroup;            // act quant group == Kt
        const int wK     = wt.elems_per_block;   // weight quant block K-size == Kt

        std::vector<uint8_t> asrc((size_t) (M * K) * asrc_esz);
        for (int64_t m = 0; m < M; ++m)
            for (int64_t k = 0; k < K; ++k) {
                const int64_t slot = act_group_slot(m, k / agroup, M, K, ts) * agroup + (k % agroup);
                const float   v    = A_src[(size_t) (m * K + k)];
                uint8_t * dst = asrc.data() + (size_t) slot * asrc_esz;
                switch (c.act_in) {
                    case ActDType::F32:  { std::memcpy(dst, &v, 4); } break;
                    case ActDType::F16:  { ggml_fp16_t h = ggml_fp32_to_fp16(v); std::memcpy(dst, &h, 2); } break;
                    case ActDType::BF16: { ggml_bf16_t h = ggml_fp32_to_bf16(v); std::memcpy(dst, &h, 2); } break;
                    case ActDType::E5M2: { dst[0] = fp8_encode(v, fp8_e5m2()); } break;
                    case ActDType::E4M3: { dst[0] = fp8_encode(v, fp8_e4m3()); } break;
                }
            }
        write_bin(dir + "/act_src_" + asrc_nm + ".bin", asrc.data(), asrc.size());

        std::vector<ggml_fp16_t> w16((size_t) (N * K));
        for (int64_t n = 0; n < N; ++n)
            for (int64_t k = 0; k < K; ++k) {
                const int64_t slot = weight_block_slot(n, k / wK, N, ts) * wK + (k % wK);
                w16[(size_t) slot] = ggml_fp32_to_fp16(W_src[(size_t) (n * K + k)]);
            }
        write_bin(dir + "/weight_src_f16.bin", w16.data(), w16.size() * sizeof(ggml_fp16_t));
    }

    // ---- outputs: produced IN-CHIP by the GEMM kernel's OutConv (all 6 dtypes,
    //      result tile order, NO scale). Dumper just writes the buffers verbatim.
    {
        int nsp; const OutSpec * sp = out_specs(nsp);
        for (int s = 0; s < nsp; ++s)
            write_bin(dir + "/output_" + sp[s].name + ".bin", out_bufs[s].data(), out_bufs[s].size());
    }

    // ---- golden (§4.1 result tile order, real-valued reference) ----
    write_bin(dir + "/golden_f32.bin", C_ref_tiled.data(), C_ref_tiled.size() * sizeof(float));

    // ---- meta.json (self-describing: every entry carries a one-line _doc,
    //      every .bin carries total_bytes + block layout + de-tile index) ----
    {
        const int64_t Mtiles = M / ts.Mt, Ntiles = N / ts.Nt, Ktiles = K / ts.Kt;
        const int64_t n_wblk = N * K / wt.elems_per_block;
        const int     agroup = ts.agroup;
        const size_t  act_stride = (size_t) act_block_bytes(ts, c.A_bits); // 2 + agroup*elem
        const int64_t n_ablk = M * K / agroup;
        const size_t  w_bb   = n_wblk ? (size_t) (w_blocks_bytes / n_wblk) : 0;
        const size_t  gold_bytes = (size_t) (M * N) * sizeof(float);
        const size_t  act_bytes  = a_blocks.size();
        const size_t  asrc_bytes = (size_t) (M * K) * asrc_esz;
        const size_t  wsrc_bytes = (size_t) (N * K) * sizeof(ggml_fp16_t);
        const char *  aqs_ct = c.A_bits > 8 ? "i16" : "i8";       // act qs container
        FILE * j = fopen((dir + "/meta.json").c_str(), "w");
        if (j) {
            fprintf(j,
                "{\n"
                "  \"name\": \"%s\",\n"
                "  \"gemm\": {\n"
                "    \"_doc\": \"row-major matmul; A and C are [M,K]/[M,N], W is [N,K] used transposed\",\n"
                "    \"formula\": \"C[M,N] = A[M,K] @ W[N,K]^T\",\n"
                "    \"M\": %lld, \"N\": %lld, \"K\": %lld\n"
                "  },\n"
                "  \"quant\": {\n"
                "    \"weight_type\": \"%s\", \"weight_family\": \"%s\", \"weight_bits\": %d, \"weight_has_min\": %s,\n"
                "    \"act_input_dtype\": \"%s\", \"act_compute_bits\": %d, \"output_dtypes\": \"F16,BF16,E4M3,I16,I8,I6,I4,U16,U8,U6,U4\",\n"
                "    \"psum_trunc_bits\": %d, \"_psum_doc\": \"0 = no integer-Psum truncation\", \"seed\": %llu\n"
                "  },\n"
                "  \"tiling\": {\n"
                "    \"_doc\": \"tile = Mt rows x Nt cols(N) x Kt cols(K); weight/act tiled at quant-block granularity, result at element\",\n"
                "    \"Mt\": %d, \"Nt\": %d, \"Kt\": %d,\n"
                "    \"act_group_elems\": %d, \"weight_group_elems\": %d,\n"
                "    \"Mtiles\": %lld, \"Ntiles\": %lld, \"Ktiles\": %lld\n"
                "  },\n"
                "  \"files\": {\n"
                "    \"act_src_%s.bin\": {\n"
                "      \"_doc\": \"native-dtype source activation A[M,K] BEFORE int quant, §4.1 tile order [Mt x Kt] (same tiling as act_blocks)\",\n"
                "      \"dtype\": \"%s\", \"elem_bytes\": %d, \"total_bytes\": %zu, \"shape\": [%lld, %lld], \"tile\": [%d, %d],\n"
                "      \"elem_index\": \"idx(m,k) = ((m/Mt*Ktiles + k/Kt)*Mt + m%%Mt)*Kt + k%%Kt\"\n"
                "    },\n"
                "    \"weight_src_f16.bin\": {\n"
                "      \"_doc\": \"FP16 source weight W[N,K] BEFORE quant, §4.1 tile order [Nt x Kt] (same tiling as weight_blocks)\",\n"
                "      \"dtype\": \"f16\", \"total_bytes\": %zu, \"shape\": [%lld, %lld], \"tile\": [%d, %d],\n"
                "      \"elem_index\": \"idx(n,k) = ((k/Kt*Ntiles + n/Nt)*Nt + n%%Nt)*Kt + k%%Kt\"\n"
                "    },\n"
                "    \"weight_blocks.bin\": {\n"
                "      \"_doc\": \"quantized weights W[N,K], packed blocks in tile order\",\n"
                "      \"qtype\": \"%s\", \"total_bytes\": %zu, \"block_bytes\": %zu, \"num_blocks\": %lld,\n"
                "      \"block_covers\": \"n=1 weight-row x %d K-elems (one quant block)\",\n"
                "      \"byte_layout\": \"%s\",\n"
                "      \"block_index\": \"slot(n,sb) = (sb*Ntiles + n/Nt)*Nt + n%%Nt   for sb in [0,Ktiles), n in [0,N)\"\n"
                "    },\n"
                "    \"act_blocks.bin\": {\n"
                "      \"_doc\": \"int activations A[M,K], one contiguous quant group per block (1 fp16 scale + agroup intX, X=A_bits container)\",\n"
                "      \"struct\": \"{ f16 d; %s qs[agroup] }\", \"act_bits\": %d, \"total_bytes\": %zu, \"block_bytes\": %zu, \"num_blocks\": %lld,\n"
                "      \"block_covers\": \"m=1 act-row x %d K-elems (one quant group)\",\n"
                "      \"byte_layout\": \"d:f16; qs:%s[%d]\",\n"
                "      \"block_index\": \"slot(m,kg) = (m/Mt*Ktiles + kg)*Mt + m%%Mt   for kg in [0,K/%d), agroup==Kt\"\n"
                "    },\n"
                "    \"output_<DT>.bin\": {\n"
                "      \"_doc\": \"GPU result C[M,N] converted to each DT; element tile order. NO scale (saturation only). float=fp cast (round-to-nearest, FP16 overflow->Inf); E4M3=RNE+saturate +/-448 (no Inf); int=round-half-to-even then saturate. I6/I4 in int8 container, U6/U4 in uint8 container\",\n"
                "      \"dtypes\": [\"F16\",\"BF16\",\"E4M3\",\"I16\",\"I8\",\"I6\",\"I4\",\"U16\",\"U8\",\"U6\",\"U4\"],\n"
                "      \"fp8_e4m3\": {\"max\":448,\"inf\":false}, \"int_saturate\": {\"I16\":[-32768,32767],\"I8\":[-128,127],\"I6\":[-32,31],\"I4\":[-8,7]}, \"uint_saturate\": {\"U16\":[0,65535],\"U8\":[0,255],\"U6\":[0,63],\"U4\":[0,15]},\n"
                "      \"shape\": [%lld, %lld],\n"
                "      \"elem_index\": \"idx(m,n) = (n/Nt*Mtiles + m/Mt)*(Mt*Nt) + (m%%Mt)*Nt + n%%Nt\"\n"
                "    },\n"
                "    \"golden_f32.bin\": {\n"
                "      \"_doc\": \"CPU reference C[M,N] (real-valued), same order as output\",\n"
                "      \"dtype\": \"f32\", \"total_bytes\": %zu, \"shape\": [%lld, %lld],\n"
                "      \"elem_index\": \"same as output_<DT>.bin\"\n"
                "    }\n"
                "  },\n"
                "  \"verify\": {\n"
                "    \"_doc\": \"GPU vs integer golden, element-wise over all M*N\",\n"
                "    \"max_abs\": %.6g, \"max_rel\": %.6g, \"mse\": %.6g\n"
                "  }\n"
                "}\n",
                c.name.c_str(),
                (long long) M, (long long) N, (long long) K,
                wt.name, family_name(wt.family), wt.W_bits, wt.has_min ? "true" : "false",
                actdtype_name(c.act_in), c.A_bits,
                c.psum_bits, (unsigned long long) c.seed,
                ts.Mt, ts.Nt, ts.Kt, ts.agroup, ts.wgroup,
                (long long) Mtiles, (long long) Ntiles, (long long) Ktiles,
                // act_src_<dt>.bin / weight_src_f16.bin
                asrc_nm, asrc_nm, asrc_esz, asrc_bytes, (long long) M, (long long) K, ts.Mt, ts.Kt,
                wsrc_bytes, (long long) N, (long long) K, ts.Nt, ts.Kt,
                // weight_blocks.bin
                wt.name, (size_t) w_blocks_bytes, w_bb, (long long) n_wblk,
                wt.elems_per_block, w_layout_desc,
                // act_blocks.bin
                aqs_ct, c.A_bits, act_bytes, act_stride, (long long) n_ablk,
                agroup, aqs_ct, agroup, agroup,
                // output_<DT> / golden
                (long long) M, (long long) N,
                gold_bytes, (long long) M, (long long) N,
                err.max_abs, err.max_rel, err.mse);
            fclose(j);
        }
    }

    return dir;
}

// ---- qkt chain (docs/chain-qkt.md) ----------------------------------------

// Serialize a float matrix src[rows*cols] (row-major) into §4.1 act tile order
// in the native act_in dtype.
static void write_act_src_tiled(const std::string & path,
                                const std::vector<float> & src,
                                int64_t M, int64_t K, const TilingSpec & ts,
                                ActDType act_in) {
    const int esz    = act_src_elem_bytes(act_in);
    const int agroup = ts.agroup;
    std::vector<uint8_t> buf((size_t) (M * K) * esz);
    for (int64_t m = 0; m < M; ++m)
        for (int64_t k = 0; k < K; ++k) {
            const int64_t slot = act_group_slot(m, k / agroup, M, K, ts) * agroup + (k % agroup);
            const float   v    = src[(size_t) (m * K + k)];
            uint8_t * dst = buf.data() + (size_t) slot * esz;
            switch (act_in) {
                case ActDType::F32:  { std::memcpy(dst, &v, 4); } break;
                case ActDType::F16:  { ggml_fp16_t h = ggml_fp32_to_fp16(v); std::memcpy(dst, &h, 2); } break;
                case ActDType::BF16: { ggml_bf16_t h = ggml_fp32_to_bf16(v); std::memcpy(dst, &h, 2); } break;
                case ActDType::E5M2: { dst[0] = fp8_encode(v, fp8_e5m2()); } break;
                case ActDType::E4M3: { dst[0] = fp8_encode(v, fp8_e4m3()); } break;
            }
        }
    write_bin(path, buf.data(), buf.size());
}

std::string dump_chain_case(
    const std::string & out_root, const std::string & case_name,
    const GemmCase & c1, const TilingSpec & ts1, const WQuantType & wt1,
    const void * w1_blocks, size_t w1_blocks_bytes,
    const char * w1_layout_desc,
    const std::vector<uint8_t> & a1_blocks,
    const std::vector<float> & X_src,
    const std::vector<float> & Wk_src,
    const std::vector<float> & C1_tiled,
    int kbits, const WQuantType & wtk,
    const std::vector<uint8_t> & kblocks_pre,
    const std::vector<uint8_t> & kblocks_post,
    const TilingSpec & ts2, int64_t M2,
    const std::vector<uint8_t> & q_blocks,
    const std::vector<float> & Q_src,
    const std::vector<uint8_t> * mma2_out_bufs,
    const std::vector<float> & C2_ref_tiled,
    const ChainVerify & v) {

    const int64_t M1 = c1.M, N1 = c1.N, K1 = c1.K;
    const int64_t num_keys = M1, head_dim = N1;    // MMA2: N2 = num_keys, K2 = head_dim
    const int64_t N2 = num_keys, K2 = head_dim;
    const int64_t n_hd_groups = head_dim / 64;
    const int64_t n_kblocks   = num_keys * n_hd_groups;
    const int     kbb         = (int) wtk.block_bytes;

    char ts_buf[24];
    {
        std::time_t t = std::time(nullptr);
        std::tm     tmv;
        localtime_r(&t, &tmv);
        std::strftime(ts_buf, sizeof(ts_buf), "%Y%m%d-%H%M%S", &tmv);
    }
    const std::string dir = out_root + "/" + case_name + "-" + ts_buf;
    make_dirs(dir);

    const char * asrc_nm  = actdtype_name(c1.act_in);
    const int    asrc_esz = act_src_elem_bytes(c1.act_in);

    // ---- MMA1 inputs ----
    write_act_src_tiled(dir + "/mma1_act_src_" + asrc_nm + ".bin", X_src, M1, K1, ts1, c1.act_in);
    {
        const int wK = wt1.elems_per_block;
        std::vector<ggml_fp16_t> w16((size_t) (N1 * K1));
        for (int64_t n = 0; n < N1; ++n)
            for (int64_t k = 0; k < K1; ++k) {
                const int64_t slot = weight_block_slot(n, k / wK, N1, ts1) * wK + (k % wK);
                w16[(size_t) slot] = ggml_fp32_to_fp16(Wk_src[(size_t) (n * K1 + k)]);
            }
        write_bin(dir + "/mma1_weight_src_f16.bin", w16.data(), w16.size() * sizeof(ggml_fp16_t));
    }
    write_bin(dir + "/mma1_weight_blocks.bin", w1_blocks, w1_blocks_bytes);
    write_bin(dir + "/mma1_act_blocks.bin", a1_blocks.data(), a1_blocks.size());

    // ---- MMA1 result (pre-quant real values) + fused-quant K blocks ----
    write_bin(dir + "/mma1_rslt_f32.bin", C1_tiled.data(), C1_tiled.size() * sizeof(float));
    write_bin(dir + "/kblocks_pretrans.bin",  kblocks_pre.data(),  kblocks_pre.size());
    write_bin(dir + "/kblocks_posttrans.bin", kblocks_post.data(), kblocks_post.size());

    // ---- MMA2 inputs ----
    write_act_src_tiled(dir + "/mma2_act_src_" + asrc_nm + ".bin", Q_src, M2, K2, ts2, c1.act_in);
    write_bin(dir + "/mma2_act_blocks.bin", q_blocks.data(), q_blocks.size());

    // ---- MMA2 outputs + golden ----
    {
        int nsp; const OutSpec * sp = out_specs(nsp);
        for (int s = 0; s < nsp; ++s)
            write_bin(dir + "/mma2_output_" + sp[s].name + ".bin",
                      mma2_out_bufs[s].data(), mma2_out_bufs[s].size());
    }
    write_bin(dir + "/mma2_golden_f32.bin", C2_ref_tiled.data(), C2_ref_tiled.size() * sizeof(float));

    // ---- meta.json ----
    {
        const int64_t n_w1blk = N1 * K1 / wt1.elems_per_block;
        const size_t  w1_bb   = n_w1blk ? w1_blocks_bytes / (size_t) n_w1blk : 0;
        const size_t  a1_bb   = (size_t) act_block_bytes(ts1, c1.A_bits);
        const size_t  q_bb    = (size_t) act_block_bytes(ts2, c1.A_bits);
        const char *  aqs_ct  = c1.A_bits > 8 ? "i16" : "i8";
        const char *  kqs_ct  = kbits == 4 ? "i4(nibble)" : "i8";
        FILE * j = fopen((dir + "/meta.json").c_str(), "w");
        if (j) {
            fprintf(j,
                "{\n"
                "  \"name\": \"%s\",\n"
                "  \"chain\": {\n"
                "    \"_doc\": \"two-stage fused MMA (docs/chain-qkt.md): MMA1 builds K and its output stage emits quant blocks directly (NO dequant/requant round-trip); a block-atomic transpose re-lays them; MMA2 consumes them verbatim as its weight operand\",\n"
                "    \"semantics\": \"K[key,head_dim] = X @ Wk^T ; S[query,key] = Q @ K^T (K = MMA1 result)\",\n"
                "    \"head_dim\": %lld, \"num_keys\": %lld, \"num_queries\": %lld\n"
                "  },\n"
                "  \"mma1\": {\n"
                "    \"formula\": \"Rslt[M1,N1] = X[M1,K1] @ Wk[N1,K1]^T   (m1=key/token, n1=head_dim)\",\n"
                "    \"M\": %lld, \"N\": %lld, \"K\": %lld,\n"
                "    \"weight_type\": \"%s\", \"weight_bits\": %d, \"act_bits\": %d, \"act_input_dtype\": \"%s\",\n"
                "    \"psum_trunc_bits\": %d, \"seed\": %llu,\n"
                "    \"tiling\": { \"Mt\": %d, \"Nt\": %d, \"Kt\": %d, \"agroup\": %d, \"wgroup\": %d }\n"
                "  },\n"
                "  \"kblocks\": {\n"
                "    \"_doc\": \"fused MMA1 output-stage quantization: ONE group per output row segment [1,64] over head_dim; amax -> scale=amax/qmax (fp16) -> round-half-to-even -> clamp(+/-qmax); source = raw fp32 accumulator (no fp16 round-trip). Same byte format the GEMM reads as weight blocks -> MMA2 consumes without requant\",\n"
                "    \"ktype\": \"%s\", \"kbits\": %d, \"qmax\": %d, \"block_bytes\": %d, \"num_blocks\": %lld,\n"
                "    \"byte_layout\": \"{ d: f16; qs: %s[64] }%s\",\n"
                "    \"pretrans_index\": \"slot(key,g) = key*%lld + g   (token-major; g = head_dim group)\",\n"
                "    \"posttrans_index\": \"slot(key,g) = g*%lld + key   (== weight_block_slot(n2=key, sb=g) for Nt=64)\",\n"
                "    \"transpose\": \"block-atomic key<->head_dim grid transpose; whole {scale+64 codes} blocks move, group contents untouched\"\n"
                "  },\n"
                "  \"mma2\": {\n"
                "    \"formula\": \"S[M2,N2] = Q[M2,K2] @ K[N2,K2]^T ; weight = kblocks_posttrans (NOT re-quantized)\",\n"
                "    \"M\": %lld, \"N\": %lld, \"K\": %lld,\n"
                "    \"weight_type\": \"%s\", \"act_bits\": %d, \"psum_trunc_bits\": %d,\n"
                "    \"tiling\": { \"Mt\": %d, \"Nt\": %d, \"Kt\": %d, \"agroup\": %d, \"wgroup\": %d }\n"
                "  },\n"
                "  \"files\": {\n"
                "    \"mma1_act_src_%s.bin\":   { \"_doc\": \"X[M1,K1] pre-quant, native dtype, act tile order (mma1 tiling)\", \"dtype\": \"%s\", \"elem_bytes\": %d, \"total_bytes\": %zu,\n"
                "      \"elem_index\": \"idx(m,k) = ((m/Mt*Ktiles + k/Kt)*Mt + m%%Mt)*Kt + k%%Kt\" },\n"
                "    \"mma1_weight_src_f16.bin\": { \"_doc\": \"Wk[N1,K1] pre-quant fp16, weight tile order (mma1 tiling)\", \"total_bytes\": %zu,\n"
                "      \"elem_index\": \"idx(n,k) = ((k/Kt*Ntiles + n/Nt)*Nt + n%%Nt)*Kt + k%%Kt\" },\n"
                "    \"mma1_weight_blocks.bin\": { \"_doc\": \"Wk quant blocks, weight_block_slot order\", \"qtype\": \"%s\", \"total_bytes\": %zu, \"block_bytes\": %zu, \"num_blocks\": %lld,\n"
                "      \"byte_layout\": \"%s\" },\n"
                "    \"mma1_act_blocks.bin\":  { \"_doc\": \"X quant blocks { f16 d; %s qs[64] }, act_group_slot order\", \"total_bytes\": %zu, \"block_bytes\": %zu },\n"
                "    \"mma1_rslt_f32.bin\":    { \"_doc\": \"MMA1 fp32 accumulator (real values BEFORE fused quant), result tile order (mma1 tiling)\", \"total_bytes\": %zu, \"shape\": [%lld, %lld],\n"
                "      \"elem_index\": \"idx(m,n) = (n/Nt*Mtiles + m/Mt)*(Mt*Nt) + (m%%Mt)*Nt + n%%Nt\" },\n"
                "    \"kblocks_pretrans.bin\":  { \"_doc\": \"fused MMA1 output quant blocks BEFORE block transpose (token-major)\", \"total_bytes\": %zu, \"block_bytes\": %d, \"num_blocks\": %lld },\n"
                "    \"kblocks_posttrans.bin\": { \"_doc\": \"AFTER block transpose == mma2 weight_blocks (weight_block_slot order, mma2 tiling). Identical bytes serve both roles\", \"total_bytes\": %zu, \"block_bytes\": %d, \"num_blocks\": %lld },\n"
                "    \"mma2_act_src_%s.bin\":  { \"_doc\": \"Q[M2,K2] pre-quant, native dtype, act tile order (mma2 tiling)\", \"dtype\": \"%s\", \"total_bytes\": %zu },\n"
                "    \"mma2_act_blocks.bin\":  { \"_doc\": \"Q quant blocks { f16 d; %s qs[64] }, act_group_slot order (mma2 tiling)\", \"total_bytes\": %zu, \"block_bytes\": %zu },\n"
                "    \"mma2_output_<DT>.bin\": { \"_doc\": \"S[M2,N2] converted to each DT, result tile order (mma2 tiling); NO scale, saturation only (same rules as single-GEMM datagen)\",\n"
                "      \"dtypes\": [\"F16\",\"BF16\",\"E4M3\",\"I16\",\"I8\",\"I6\",\"I4\",\"U16\",\"U8\",\"U6\",\"U4\"], \"shape\": [%lld, %lld] },\n"
                "    \"mma2_golden_f32.bin\":  { \"_doc\": \"CPU integer golden of MMA2 (consumes the SAME GPU kblocks_posttrans -> chain-consistent)\", \"total_bytes\": %zu, \"shape\": [%lld, %lld] }\n"
                "  },\n"
                "  \"verify\": {\n"
                "    \"mma1\": { \"max_abs\": %.6g, \"max_rel\": %.6g, \"mse\": %.6g, \"dequant_max_abs\": %.6g },\n"
                "    \"kblocks_gpu_cpu_bitexact\": %s,\n"
                "    \"mma2\": { \"max_abs\": %.6g, \"max_rel\": %.6g, \"mse\": %.6g, \"dequant_max_abs\": %.6g },\n"
                "    \"_doc\": \"mma1/mma2: GPU fp32 acc vs CPU double integer golden (~1e-5). kblocks: GPU output-stage kernel vs CPU recompute of the SAME fp32 accumulator, byte-for-byte. dequant_max_abs < 0 means skipped (psum != 0)\"\n"
                "  }\n"
                "}\n",
                case_name.c_str(),
                (long long) head_dim, (long long) num_keys, (long long) M2,
                (long long) M1, (long long) N1, (long long) K1,
                wt1.name, wt1.W_bits, c1.A_bits, asrc_nm,
                c1.psum_bits, (unsigned long long) c1.seed,
                ts1.Mt, ts1.Nt, ts1.Kt, ts1.agroup, ts1.wgroup,
                wtk.name, kbits, (1 << (kbits - 1)) - 1, kbb, (long long) n_kblocks,
                kqs_ct, kbits == 4 ? " (nibble interleave: e<32 low, e>=32 high of qs[e-32])" : "",
                (long long) n_hd_groups,
                (long long) num_keys,
                (long long) M2, (long long) N2, (long long) K2,
                wtk.name, c1.A_bits, c1.psum_bits,
                ts2.Mt, ts2.Nt, ts2.Kt, ts2.agroup, ts2.wgroup,
                // files
                asrc_nm, asrc_nm, asrc_esz, (size_t) (M1 * K1) * asrc_esz,
                (size_t) (N1 * K1) * sizeof(ggml_fp16_t),
                wt1.name, w1_blocks_bytes, w1_bb, (long long) n_w1blk,
                w1_layout_desc,
                aqs_ct, a1_blocks.size(), a1_bb,
                C1_tiled.size() * sizeof(float), (long long) M1, (long long) N1,
                kblocks_pre.size(), kbb, (long long) n_kblocks,
                kblocks_post.size(), kbb, (long long) n_kblocks,
                asrc_nm, asrc_nm, (size_t) (M2 * K2) * asrc_esz,
                aqs_ct, q_blocks.size(), q_bb,
                (long long) M2, (long long) N2,
                C2_ref_tiled.size() * sizeof(float), (long long) M2, (long long) N2,
                v.err1.max_abs, v.err1.max_rel, v.err1.mse, v.dq1_max_abs,
                v.k_bitexact ? "true" : "false",
                v.err2.max_abs, v.err2.max_rel, v.err2.mse, v.dq2_max_abs);
            fclose(j);
        }
    }

    return dir;
}

} // namespace rgd
