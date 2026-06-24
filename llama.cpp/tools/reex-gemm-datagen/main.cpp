// REEX GEMM datagen — single symmetric case end-to-end driver.
//
// Pipeline (struct-faithful, mirrors llama.cpp/reex storage + MAC):
//   datagen -> weight encode(reex blocks) -> reorder to §4.1 block order
//           -> act quantize (block_q8_1, per-group scale, §4.1)
//           -> GPU int GEMM (reads packed blocks) -> CPU golden -> compare
//           -> dump (packed weight/act blocks + result + meta.json)
#include "case.h"
#include "datagen.h"
#include "dumper.h"
#include "gemm.cuh"
#include "reference.h"
#include "wquant.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace rgd;

static void build_name(GemmCase & c, const WQuantType & wt) {
    char buf[192];
    snprintf(buf, sizeof(buf), "%s-A%d-%sin-%sout-psum%d",
             wt.name, c.A_bits, actdtype_name(c.act_in),
             outdtype_name(c.out), c.psum_bits);
    c.name = buf;
}

int main(int argc, char ** argv) {
    std::string out_root = "output/datagen";
    std::string wtype    = "q8_0_64";  // default: symmetric Legacy W8
    int64_t check_rows = 256;          // rows validated by the dequant golden
    GemmCase c;

    for (int i = 1; i < argc; ++i) {
        if      (!strcmp(argv[i], "--out")   && i + 1 < argc) out_root = argv[++i];
        else if (!strcmp(argv[i], "--wtype") && i + 1 < argc) wtype = argv[++i];
        else if (!strcmp(argv[i], "--seed")  && i + 1 < argc) c.seed = strtoull(argv[++i], nullptr, 10);
        else if (!strcmp(argv[i], "--M")     && i + 1 < argc) c.M = atoll(argv[++i]);
        else if (!strcmp(argv[i], "--N")     && i + 1 < argc) c.N = atoll(argv[++i]);
        else if (!strcmp(argv[i], "--K")     && i + 1 < argc) c.K = atoll(argv[++i]);
        else if (!strcmp(argv[i], "--check-rows") && i + 1 < argc) check_rows = atoll(argv[++i]);
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            printf("Usage: %s [--out DIR] [--wtype NAME] [--seed S] [--M m --N n --K k] [--check-rows R]\n", argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown arg: %s\n", argv[i]);
            return 1;
        }
    }

    const int wid = wquant_find(wtype.c_str());
    if (wid < 0) { fprintf(stderr, "wtype '%s' not in registry\n", wtype.c_str()); return 1; }
    c.wtype_id = wid;
    c.act_in   = ActDType::F16;
    c.A_bits   = 8;
    c.out      = OutDType::F16;
    c.psum_bits = 0;

    const WQuantType & wt = wquant_get(wid);
    const TilingSpec   ts = tiling_for(wt.family);
    build_name(c, wt);

    if (c.K % ts.Kt || c.N % ts.Nt || c.M % ts.Mt || c.K % ts.agroup) {
        fprintf(stderr, "shape not divisible by tiling (Mt=%d Nt=%d Kt=%d agroup=%d)\n",
                ts.Mt, ts.Nt, ts.Kt, ts.agroup);
        return 1;
    }

    fprintf(stderr, "[rgd] case %s  M=%lld N=%lld K=%lld\n",
            c.name.c_str(), (long long) c.M, (long long) c.N, (long long) c.K);

    // 1) inputs
    std::vector<float> A, W;
    datagen_fill(A, W, c);

    // 2) weight encode (reex packed blocks, native row-major) -> §4.1 block order
    const size_t wbytes = wquant_blocks_bytes(wid, c.N, c.K);
    std::vector<uint8_t> w_native(wbytes);
    wt.encode(W.data(), w_native.data(), c.N, c.K);
    std::vector<uint8_t> w_blocks(wbytes);          // §4.1 block-tile order
    wquant_reorder_to_tiled(wid, w_native.data(), w_blocks.data(), c.N, c.K, ts);

    // 3) activation quantize -> contiguous group blocks {f16 d; i8 qs[agroup]}, §4.1 order
    std::vector<uint8_t> a_blocks;
    quantize_act(A, c.M, c.K, ts, c.A_bits, c.act_in, a_blocks);

    // 4) GPU integer GEMM (reads packed blocks, writes result tile order)
    std::vector<float> C_gpu((size_t) (c.M * c.N));
    gemm_run_host(wid, w_blocks.data(), a_blocks.data(), a_blocks.size(), C_gpu.data(),
                  c.M, c.N, c.K, ts, c.psum_bits);

    // 5) CPU golden (same shared MAC) + element-wise compare
    std::vector<float> C_ref;
    golden_cpu_symmetric(wid, w_blocks.data(), a_blocks, c.M, c.N, c.K, ts, c.psum_bits, C_ref);
    const CaseError err = compare(C_gpu.data(), C_ref.data(), c.M * c.N);
    fprintf(stderr, "[rgd] verify (integer golden): max_abs=%.6g max_rel=%.6g mse=%.6g\n",
            err.max_abs, err.max_rel, err.mse);

    // Independent cross-check via reex dequantize_row_* (psum=0 only).
    if (c.psum_bits == 0) {
        const DequantCheck dq = golden_dequant_check(
            wid, w_blocks.data(), a_blocks, C_gpu.data(), c.M, c.N, c.K, ts, check_rows);
        fprintf(stderr, "[rgd] verify (reex dequant golden, %lld rows): max_abs=%.6g\n",
                (long long) dq.rows, dq.max_abs);
    }

    // 6) dump — K-quant gets re-packed into the HW layout
    //    glb_scale + 4*[sub_scale + 64*data]; Legacy keeps the reex struct {d; qs}.
    const size_t hw_bb = wquant_hw_block_bytes(wid);
    std::vector<uint8_t> w_hw;
    const void * w_dump      = w_blocks.data();
    size_t       w_dump_size = wbytes;
    const char * w_desc      = "reex native struct {d(fp16); qs} (scale-first)";
    if (hw_bb) {
        const int64_t nblk = c.N * c.K / wt.elems_per_block;
        w_hw.resize((size_t) nblk * hw_bb);
        wquant_repack_hw(wid, w_blocks.data(), w_hw.data(), c.N, c.K);
        w_dump      = w_hw.data();
        w_dump_size = w_hw.size();
        w_desc      = "LSB-first bitstream: glb_scale(fp16,16b) + 4*[sub_scale(scale_bits, signed two's-comp) + 64 codes @W bits]";
    }
    const std::string dir = dump_case(out_root, c, ts, wt, w_dump, w_dump_size, w_desc,
                                      a_blocks, A, W, C_gpu, C_ref, err);
    fprintf(stderr, "[rgd] dumped -> %s\n", dir.c_str());

    return 0;
}
