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

} // namespace rgd
