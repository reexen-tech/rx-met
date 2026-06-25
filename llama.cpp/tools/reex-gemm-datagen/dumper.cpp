#include "dumper.h"

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

// element size of the output dtype (bytes)
static size_t out_elem_size(OutDType d) {
    switch (d) {
        case OutDType::F16:  return 2;
        case OutDType::BF16: return 2;
        case OutDType::I16:  return 2;
        case OutDType::I8:   return 1;
    }
    return 4;
}

std::string dump_case(
    const std::string & out_root, const GemmCase & c,
    const TilingSpec & ts, const WQuantType & wt,
    const void * w_blocks, size_t w_blocks_bytes,
    const char * w_layout_desc,
    const std::vector<uint8_t> & a_blocks,
    const std::vector<float> & A_src,
    const std::vector<float> & W_src,
    const std::vector<float> & C_gpu_tiled,
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

    // ---- FP16 source inputs (pre-quant), §4.1 tile order (same tiling as the
    //      quantized blocks): each quant block expands to its Kt fp16 elements.
    //        act:    slot = act_group_slot(m, k/Kt)*Kt + k%Kt     tile [Mt x Kt]
    //        weight: slot = weight_block_slot(n, k/Kt)*Kt + k%Kt  tile [Nt x Kt]
    {
        const int agroup = ts.agroup;            // act quant group == Kt
        const int wK     = wt.elems_per_block;   // weight quant block K-size == Kt

        std::vector<ggml_fp16_t> a16((size_t) (M * K));
        for (int64_t m = 0; m < M; ++m)
            for (int64_t k = 0; k < K; ++k) {
                const int64_t slot = act_group_slot(m, k / agroup, M, K, ts) * agroup + (k % agroup);
                a16[(size_t) slot] = ggml_fp32_to_fp16(A_src[(size_t) (m * K + k)]);
            }
        write_bin(dir + "/act_src_f16.bin", a16.data(), a16.size() * sizeof(ggml_fp16_t));

        std::vector<ggml_fp16_t> w16((size_t) (N * K));
        for (int64_t n = 0; n < N; ++n)
            for (int64_t k = 0; k < K; ++k) {
                const int64_t slot = weight_block_slot(n, k / wK, N, ts) * wK + (k % wK);
                w16[(size_t) slot] = ggml_fp32_to_fp16(W_src[(size_t) (n * K + k)]);
            }
        write_bin(dir + "/weight_src_f16.bin", w16.data(), w16.size() * sizeof(ggml_fp16_t));
    }

    // ---- output_<dtype>.bin: cast C_gpu in place (already result tile order) ----
    const size_t esz = out_elem_size(c.out);
    std::vector<uint8_t> obuf((size_t) (M * N) * esz);
    std::vector<float>   out_scale; // only for I16/I8 (per-row)

    auto put = [&](int64_t idx, const void * src) {
        std::memcpy(obuf.data() + (size_t) idx * esz, src, esz);
    };

    if (c.out == OutDType::F16 || c.out == OutDType::BF16) {
        for (int64_t i = 0; i < M * N; ++i) {
            const float v = C_gpu_tiled[i];
            if (c.out == OutDType::F16) { ggml_fp16_t h = ggml_fp32_to_fp16(v); put(i, &h); }
            else                        { ggml_bf16_t h = ggml_fp32_to_bf16(v); put(i, &h); }
        }
    } else { // I16 / I8 per-row symmetric (gather row m from result tile order)
        const int qmax = (c.out == OutDType::I16) ? 32767 : 127;
        out_scale.resize((size_t) M);
        for (int64_t m = 0; m < M; ++m) {
            float amax = 0.0f;
            for (int64_t n = 0; n < N; ++n)
                amax = std::fmax(amax, std::fabs(C_gpu_tiled[result_tiled_index(m, n, M, ts)]));
            const float scale = amax > 0.0f ? amax / (float) qmax : 1.0f;
            const float inv   = amax > 0.0f ? (float) qmax / amax : 0.0f;
            out_scale[m] = scale;
            for (int64_t n = 0; n < N; ++n) {
                const int64_t idx = result_tiled_index(m, n, M, ts);
                int q = (int) std::lrintf(C_gpu_tiled[idx] * inv);
                if (q >  qmax) q =  qmax;
                if (q < -qmax) q = -qmax;
                if (c.out == OutDType::I16) { int16_t v16 = (int16_t) q; put(idx, &v16); }
                else                        { int8_t  v8  = (int8_t)  q; put(idx, &v8); }
            }
        }
    }

    char outname[64];
    snprintf(outname, sizeof(outname), "/output_%s.bin", outdtype_name(c.out));
    write_bin(dir + outname, obuf.data(), obuf.size());
    if (!out_scale.empty())
        write_bin(dir + "/out_scale_f32.bin", out_scale.data(), out_scale.size() * sizeof(float));

    // ---- golden (§4.1 result tile order, same as output) ----
    write_bin(dir + "/golden_f32.bin", C_ref_tiled.data(), C_ref_tiled.size() * sizeof(float));

    // ---- meta.json (self-describing: every entry carries a one-line _doc,
    //      every .bin carries total_bytes + block layout + de-tile index) ----
    {
        const int64_t Mtiles = M / ts.Mt, Ntiles = N / ts.Nt, Ktiles = K / ts.Kt;
        const int64_t n_wblk = N * K / wt.elems_per_block;
        const int     agroup = ts.agroup;
        const size_t  act_stride = (size_t) act_block_bytes(ts);     // 2 + agroup
        const int64_t n_ablk = M * K / agroup;
        const size_t  w_bb   = n_wblk ? (size_t) (w_blocks_bytes / n_wblk) : 0;
        const size_t  out_bytes  = (size_t) (M * N) * esz;
        const size_t  gold_bytes = (size_t) (M * N) * sizeof(float);
        const size_t  act_bytes  = a_blocks.size();
        const size_t  asrc_bytes = (size_t) (M * K) * sizeof(ggml_fp16_t);
        const size_t  wsrc_bytes = (size_t) (N * K) * sizeof(ggml_fp16_t);
        const char *  out_nm     = outdtype_name(c.out);
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
                "    \"act_input_dtype\": \"%s\", \"act_compute_bits\": %d, \"output_dtype\": \"%s\",\n"
                "    \"psum_trunc_bits\": %d, \"_psum_doc\": \"0 = no integer-Psum truncation\", \"seed\": %llu\n"
                "  },\n"
                "  \"tiling\": {\n"
                "    \"_doc\": \"tile = Mt rows x Nt cols(N) x Kt cols(K); weight/act tiled at quant-block granularity, result at element\",\n"
                "    \"Mt\": %d, \"Nt\": %d, \"Kt\": %d,\n"
                "    \"act_group_elems\": %d, \"weight_group_elems\": %d,\n"
                "    \"Mtiles\": %lld, \"Ntiles\": %lld, \"Ktiles\": %lld\n"
                "  },\n"
                "  \"files\": {\n"
                "    \"act_src_f16.bin\": {\n"
                "      \"_doc\": \"FP16 source activation A[M,K] BEFORE int8 quant, §4.1 tile order [Mt x Kt] (same tiling as act_blocks)\",\n"
                "      \"dtype\": \"f16\", \"total_bytes\": %zu, \"shape\": [%lld, %lld], \"tile\": [%d, %d],\n"
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
                "      \"_doc\": \"int8 activations A[M,K], one contiguous quant group per block (1 scale + agroup int8)\",\n"
                "      \"struct\": \"{ f16 d; i8 qs[agroup] }\", \"total_bytes\": %zu, \"block_bytes\": %zu, \"num_blocks\": %lld,\n"
                "      \"block_covers\": \"m=1 act-row x %d K-elems (one quant group)\",\n"
                "      \"byte_layout\": \"d:f16; qs:i8[%d]\",\n"
                "      \"block_index\": \"slot(m,kg) = (m/Mt*Ktiles + kg)*Mt + m%%Mt   for kg in [0,K/%d), agroup==Kt\"\n"
                "    },\n"
                "    \"output_%s.bin\": {\n"
                "      \"_doc\": \"GPU result C[M,N], element tile order (de-tile with elem_index)\",\n"
                "      \"dtype\": \"%s\", \"total_bytes\": %zu, \"shape\": [%lld, %lld],\n"
                "      \"elem_index\": \"idx(m,n) = (n/Nt*Mtiles + m/Mt)*(Mt*Nt) + (m%%Mt)*Nt + n%%Nt\"\n"
                "    },\n"
                "    \"golden_f32.bin\": {\n"
                "      \"_doc\": \"CPU integer-MAC reference C[M,N], same order as output\",\n"
                "      \"dtype\": \"f32\", \"total_bytes\": %zu, \"shape\": [%lld, %lld],\n"
                "      \"elem_index\": \"same as output_%s.bin\"\n"
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
                actdtype_name(c.act_in), c.A_bits, out_nm,
                c.psum_bits, (unsigned long long) c.seed,
                ts.Mt, ts.Nt, ts.Kt, ts.agroup, ts.wgroup,
                (long long) Mtiles, (long long) Ntiles, (long long) Ktiles,
                // act_src_f16.bin / weight_src_f16.bin
                asrc_bytes, (long long) M, (long long) K, ts.Mt, ts.Kt,
                wsrc_bytes, (long long) N, (long long) K, ts.Nt, ts.Kt,
                // weight_blocks.bin
                wt.name, (size_t) w_blocks_bytes, w_bb, (long long) n_wblk,
                wt.elems_per_block, w_layout_desc,
                // act_blocks.bin
                act_bytes, act_stride, (long long) n_ablk,
                agroup, agroup, agroup,
                // output / golden
                out_nm, out_nm, out_bytes, (long long) M, (long long) N,
                gold_bytes, (long long) M, (long long) N, out_nm,
                err.max_abs, err.max_rel, err.mse);
            fclose(j);
        }
    }

    return dir;
}

} // namespace rgd
