// REEX GEMM datagen — single symmetric case end-to-end driver.
//
// Pipeline (struct-faithful, mirrors llama.cpp/reex storage + MAC):
//   datagen -> weight encode(reex blocks) -> reorder to §4.1 block order
//           -> act quantize (block_q8_1, per-group scale, §4.1)
//           -> GPU int GEMM (reads packed blocks) -> CPU golden -> compare
//           -> dump (packed weight/act blocks + result + meta.json)
#include "reex_layout.h"
#include "actquant.cuh"
#include "datagen.h"
#include "dumper.h"
#include "gemm.cuh"
#include "intgemm.cuh"
#include "outquant.cuh"
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
    snprintf(buf, sizeof(buf), "%s-A%dW%d-%sin-psum%d",
             wt.name, c.A_bits, wt.W_bits, actdtype_name(c.act_in), c.psum_bits);
    c.name = buf;
}

// --chain qkt: two-stage fused MMA (docs/chain-qkt.md).
//   MMA1 [64,64,64] builds K = X @ Wk^T; its output stage emits Legacy quant
//   blocks directly (per output row [1,64]: fp16 scale + 64 INTx codes) — no
//   dequant/requant round-trip. A block-atomic transpose re-lays them, then
//   MMA2 [1,64,64] computes S = Q @ K^T consuming those blocks verbatim.
static int run_chain_qkt(const std::string & out_root, GemmCase & c,
                         const WQuantType & wt1, int kbits, int64_t check_rows) {
    if (wt1.family != Family::Legacy) {
        fprintf(stderr, "--chain qkt requires a Legacy block-64 wtype (q8_0_64/q8_1_64/q4_0_64), got %s\n", wt1.name);
        return 1;
    }
    if (kbits != 8 && kbits != 4) { fprintf(stderr, "--kbits must be 8 or 4\n"); return 1; }

    // Fixed shapes (chain-qkt.md §2): MMA1 [64,64,64], MMA2 [1,64,64].
    c.M = 64; c.N = 64; c.K = 64;
    const int64_t M1 = c.M, N1 = c.N, K1 = c.K;
    const int64_t M2 = 1, N2 = M1 /*num_keys*/, K2 = N1 /*head_dim*/;
    const TilingSpec ts1 = tiling_for(Family::Legacy);          // {4,64,64,64,64}
    const TilingSpec ts2 = { 1, 64, 64, 64, 64 };               // MMA2: M2=1 -> Mt=1

    const int kwid = wquant_find(kbits == 8 ? "q8_0_64" : "q4_0_64");
    const WQuantType & wtk = wquant_get(kwid);

    char nbuf[192];
    snprintf(nbuf, sizeof(nbuf), "qkt-A%dW%d-K%d-%sin-psum%d",
             c.A_bits, wt1.W_bits, kbits, actdtype_name(c.act_in), c.psum_bits);
    c.name = nbuf;
    fprintf(stderr, "[rgd] chain %s  MMA1[%lld,%lld,%lld] -> K blocks (%s) -> MMA2[%lld,%lld,%lld]\n",
            c.name.c_str(), (long long) M1, (long long) N1, (long long) K1,
            wtk.name, (long long) M2, (long long) N2, (long long) K2);

    // ---- MMA1: X @ Wk^T -------------------------------------------------
    std::vector<float> X, Wk;
    datagen_fill(X, Wk, c);

    const int wid1 = wquant_find(wt1.name);
    const size_t w1bytes = wquant_blocks_bytes(wid1, N1, K1);
    std::vector<uint8_t> w1_native(w1bytes), w1_blocks(w1bytes);
    wt1.encode(Wk.data(), w1_native.data(), N1, K1);
    wquant_reorder_to_tiled(wid1, w1_native.data(), w1_blocks.data(), N1, K1, ts1);

    std::vector<uint8_t> a1_blocks;
    act_quant_run_host(X.data(), M1, K1, ts1, c.A_bits, a1_blocks);

    std::vector<float> C1((size_t) (M1 * N1));
    gemm_run_host(wid1, w1_blocks.data(), a1_blocks.data(), a1_blocks.size(), C1.data(),
                  nullptr, M1, N1, K1, ts1, c.A_bits, c.psum_bits);

    ChainVerify v{};
    std::vector<float> C1_ref;
    golden_cpu_symmetric(wid1, w1_blocks.data(), a1_blocks, M1, N1, K1, ts1, c.A_bits, c.psum_bits, C1_ref);
    v.err1 = compare(C1.data(), C1_ref.data(), M1 * N1);
    fprintf(stderr, "[rgd] verify MMA1 (integer golden): max_abs=%.6g max_rel=%.6g mse=%.6g\n",
            v.err1.max_abs, v.err1.max_rel, v.err1.mse);
    v.dq1_max_abs = -1.0;
    if (c.psum_bits == 0) {
        const DequantCheck dq = golden_dequant_check(
            wid1, w1_blocks.data(), a1_blocks, C1.data(), M1, N1, K1, ts1, c.A_bits, check_rows);
        v.dq1_max_abs = dq.max_abs;
        fprintf(stderr, "[rgd] verify MMA1 (reex dequant golden, %lld rows): max_abs=%.6g\n",
                (long long) dq.rows, dq.max_abs);
    }

    // ---- fused output-stage quantization (GPU authoritative) ------------
    std::vector<uint8_t> kblocks_pre, kblocks_pre_cpu;
    out_quant_run_host(C1.data(), M1, N1, ts1, kbits, kblocks_pre);
    out_quant_cpu(C1.data(), M1, N1, ts1, kbits, kblocks_pre_cpu);
    v.k_bitexact = (kblocks_pre == kblocks_pre_cpu);
    fprintf(stderr, "[rgd] verify K blocks (GPU vs CPU recompute, byte-for-byte): %s\n",
            v.k_bitexact ? "MATCH" : "MISMATCH");

    // ---- block-atomic transpose (HW block-transpose unit model) ---------
    std::vector<uint8_t> kblocks_post;
    kblocks_grid_transpose(kblocks_pre, kblocks_post, N2 /*num_keys*/, K2 / 64, kblock_bytes(kbits));

    // ---- MMA2: Q @ K^T (K blocks consumed verbatim, NO requant) ---------
    std::vector<float> Q;
    datagen_fill_act(Q, M2 * K2, c.act_in, c.seed);
    std::vector<uint8_t> q_blocks;
    act_quant_run_host(Q.data(), M2, K2, ts2, c.A_bits, q_blocks);

    std::vector<float> C2((size_t) (M2 * N2));
    int nsp; const OutSpec * osp = out_specs(nsp);
    std::vector<std::vector<uint8_t>> obufs(nsp);
    uint8_t * out_ptrs[RGD_MAX_OUT] = {nullptr};
    for (int s = 0; s < nsp; ++s) {
        obufs[s].assign((size_t) (M2 * N2) * osp[s].bytes, 0);
        out_ptrs[s] = obufs[s].data();
    }
    gemm_run_host(kwid, kblocks_post.data(), q_blocks.data(), q_blocks.size(), C2.data(),
                  out_ptrs, M2, N2, K2, ts2, c.A_bits, c.psum_bits);

    std::vector<float> C2_ref;
    golden_cpu_symmetric(kwid, kblocks_post.data(), q_blocks, M2, N2, K2, ts2, c.A_bits, c.psum_bits, C2_ref);
    v.err2 = compare(C2.data(), C2_ref.data(), M2 * N2);
    fprintf(stderr, "[rgd] verify MMA2 (integer golden): max_abs=%.6g max_rel=%.6g mse=%.6g\n",
            v.err2.max_abs, v.err2.max_rel, v.err2.mse);
    v.dq2_max_abs = -1.0;
    if (c.psum_bits == 0) {
        const DequantCheck dq = golden_dequant_check(
            kwid, kblocks_post.data(), q_blocks, C2.data(), M2, N2, K2, ts2, c.A_bits, M2);
        v.dq2_max_abs = dq.max_abs;
        fprintf(stderr, "[rgd] verify MMA2 (reex dequant golden, %lld rows): max_abs=%.6g\n",
                (long long) dq.rows, dq.max_abs);
    }

    // ---- dump ------------------------------------------------------------
    const char * w1_desc = "reex native struct {d(fp16); qs} (scale-first)";
    const std::string dir = dump_chain_case(out_root, c.name, c, ts1, wt1,
                                            w1_blocks.data(), w1bytes, w1_desc,
                                            a1_blocks, X, Wk, C1,
                                            kbits, wtk, kblocks_pre, kblocks_post,
                                            ts2, M2, q_blocks, Q,
                                            obufs.data(), C2_ref, v);
    fprintf(stderr, "[rgd] dumped -> %s\n", dir.c_str());
    return v.k_bitexact ? 0 : 2;
}

static bool parse_actdtype(const char * s, ActDType & out) {
    if      (!strcmp(s, "F32"))  out = ActDType::F32;
    else if (!strcmp(s, "F16"))  out = ActDType::F16;
    else if (!strcmp(s, "BF16")) out = ActDType::BF16;
    else if (!strcmp(s, "E5M2")) out = ActDType::E5M2;
    else if (!strcmp(s, "E4M3")) out = ActDType::E4M3;
    else return false;
    return true;
}

int main(int argc, char ** argv) {
    std::string out_root = "output/datagen";
    std::string wtype    = "q8_0_64";  // default: symmetric Legacy W8
    std::string chain;                 // "qkt" -> two-stage fused MMA (docs/chain-qkt.md)
    int         kbits    = 8;          // chain: fused K-block precision (8|4)
    int64_t check_rows = 256;          // rows validated by the dequant golden
    GemmCase c;
    c.act_in   = ActDType::F16;
    c.A_bits   = 8;
    c.out      = OutDType::F16;
    c.psum_bits = 0;

    for (int i = 1; i < argc; ++i) {
        if      (!strcmp(argv[i], "--out")   && i + 1 < argc) out_root = argv[++i];
        else if (!strcmp(argv[i], "--wtype") && i + 1 < argc) wtype = argv[++i];
        else if (!strcmp(argv[i], "--abits") && i + 1 < argc) c.A_bits = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--actin") && i + 1 < argc) {
            if (!parse_actdtype(argv[++i], c.act_in)) { fprintf(stderr, "bad --actin (F32/F16/BF16/E5M2/E4M3)\n"); return 1; }
        }
        else if (!strcmp(argv[i], "--psum")  && i + 1 < argc) c.psum_bits = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--chain") && i + 1 < argc) chain = argv[++i];
        else if (!strcmp(argv[i], "--kbits") && i + 1 < argc) kbits = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--wbits") && i + 1 < argc) c.w_bits = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--asign") && i + 1 < argc) {
            const char c0 = argv[++i][0];
            c.a_unsigned = (c0 == 'u' || c0 == 'U');
        } else if (!strcmp(argv[i], "--wsign") && i + 1 < argc) {
            const char c0 = argv[++i][0];
            c.w_unsigned = (c0 == 'u' || c0 == 'U');
        }
        else if (!strcmp(argv[i], "--seed")  && i + 1 < argc) c.seed = strtoull(argv[++i], nullptr, 10);
        else if (!strcmp(argv[i], "--M")     && i + 1 < argc) c.M = atoll(argv[++i]);
        else if (!strcmp(argv[i], "--N")     && i + 1 < argc) c.N = atoll(argv[++i]);
        else if (!strcmp(argv[i], "--K")     && i + 1 < argc) c.K = atoll(argv[++i]);
        else if (!strcmp(argv[i], "--check-rows") && i + 1 < argc) check_rows = atoll(argv[++i]);
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            printf("Usage: %s [--out DIR] [--wtype NAME] [--abits 16|8|4] [--actin F32|F16|BF16|E5M2|E4M3]\n"
                   "          [--psum B] [--seed S] [--M m --N n --K k] [--check-rows R]\n"
                   "          [--wbits 8|6|5|4|3|2] [--asign i|u] [--wsign i|u]   (INT path only)\n"
                   "          [--chain qkt] [--kbits 8|4]   (two-stage fused MMA, shapes fixed 64; docs/chain-qkt.md)\n", argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown arg: %s\n", argv[i]);
            return 1;
        }
    }

    const int wid = wquant_find(wtype.c_str());
    if (wid < 0) { fprintf(stderr, "wtype '%s' not in registry\n", wtype.c_str()); return 1; }
    c.wtype_id = wid;

    const WQuantType & wt = wquant_get(wid);

    if (!chain.empty()) {
        if (chain != "qkt") { fprintf(stderr, "unknown --chain '%s' (only: qkt)\n", chain.c_str()); return 1; }
        return run_chain_qkt(out_root, c, wt, kbits, check_rows);
    }

    const TilingSpec   ts = tiling_for(wt.family);
    build_name(c, wt);

    if (c.K % ts.Kt || c.N % ts.Nt || c.M % ts.Mt || c.K % ts.agroup) {
        fprintf(stderr, "shape not divisible by tiling (Mt=%d Nt=%d Kt=%d agroup=%d)\n",
                ts.Mt, ts.Nt, ts.Kt, ts.agroup);
        return 1;
    }

    // IntBlock: pure-integer GEMM (no scale) takes a dedicated path.
    if (wt.family == Family::IntBlock) {
        char nbuf[192];
        snprintf(nbuf, sizeof(nbuf), "int-A%c%d-W%c%d-psum%d",
                 c.a_unsigned ? 'U' : 'I', c.A_bits,
                 c.w_unsigned ? 'U' : 'I', c.w_bits, c.psum_bits);
        c.name = nbuf;
        fprintf(stderr, "[rgd] case %s  M=%lld N=%lld K=%lld (pure int)\n",
                c.name.c_str(), (long long) c.M, (long long) c.N, (long long) c.K);
        const std::string dir = run_intblock_case(out_root, c, ts, wt);
        fprintf(stderr, "[rgd] dumped -> %s\n", dir.c_str());
        return 0;
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

    // 3) stage-1 in-chip activation quantization (device): A -> act_blocks
    //    {f16 d; intX qs[agroup]}, §4.1 order. A is already rounded to act_in.
    std::vector<uint8_t> a_blocks;
    act_quant_run_host(A.data(), c.M, c.K, ts, c.A_bits, a_blocks);

    // 4) GPU integer GEMM (reads packed blocks); writes fp32 acc (C_gpu) AND, via
    //    in-chip OutConv, the 6 dtype outputs (result tile order).
    std::vector<float> C_gpu((size_t) (c.M * c.N));
    int nsp; const OutSpec * osp = out_specs(nsp);
    std::vector<std::vector<uint8_t>> obufs(nsp);
    uint8_t * out_ptrs[RGD_MAX_OUT] = {nullptr};
    for (int s = 0; s < nsp; ++s) {
        obufs[s].assign((size_t) (c.M * c.N) * osp[s].bytes, 0);
        out_ptrs[s] = obufs[s].data();
    }
    gemm_run_host(wid, w_blocks.data(), a_blocks.data(), a_blocks.size(), C_gpu.data(),
                  out_ptrs, c.M, c.N, c.K, ts, c.A_bits, c.psum_bits);

    // 5) CPU golden (same shared MAC) + element-wise compare
    std::vector<float> C_ref;
    golden_cpu_symmetric(wid, w_blocks.data(), a_blocks, c.M, c.N, c.K, ts, c.A_bits, c.psum_bits, C_ref);
    const CaseError err = compare(C_gpu.data(), C_ref.data(), c.M * c.N);
    fprintf(stderr, "[rgd] verify (integer golden): max_abs=%.6g max_rel=%.6g mse=%.6g\n",
            err.max_abs, err.max_rel, err.mse);

    // Independent cross-check via reex dequantize_row_* (psum=0 only).
    if (c.psum_bits == 0) {
        const DequantCheck dq = golden_dequant_check(
            wid, w_blocks.data(), a_blocks, C_gpu.data(), c.M, c.N, c.K, ts, c.A_bits, check_rows);
        fprintf(stderr, "[rgd] verify (reex dequant golden, %lld rows): max_abs=%.6g\n",
                (long long) dq.rows, dq.max_abs);
    }

    // 6) dump — the reex block IS the HW layout, so w_blocks is dumped directly:
    //    K-quant  = LSB-first bitstream glb_scale + 4*[sub_scale + 64*data];
    //    Legacy   = reex struct {d(fp16); qs} (scale-first).
    const size_t hw_bb = wquant_hw_block_bytes(wid);
    const void * w_dump      = w_blocks.data();
    size_t       w_dump_size = wbytes;
    const char * w_desc      = hw_bb
        ? "LSB-first bitstream: glb_scale(fp16,16b) + 4*[sub_scale(scale_bits, signed two's-comp) + 64 codes @W bits]"
        : "reex native struct {d(fp16); qs} (scale-first)";
    const std::string dir = dump_case(out_root, c, ts, wt, w_dump, w_dump_size, w_desc,
                                      a_blocks, A, W, obufs.data(), C_ref, err);
    fprintf(stderr, "[rgd] dumped -> %s\n", dir.c_str());

    return 0;
}
