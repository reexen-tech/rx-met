// Artifact dumper — writes packed inputs, output, golden and a self-describing
// meta.json into out_root/<case_name>/.
#pragma once

#include "reex_layout.h"
#include "reference.h"   // CaseError
#include "wquant.h"      // WQuantType

#include <cstdint>
#include <string>
#include <vector>

namespace rgd {

bool make_dirs(const std::string & path);

// Writes all artifacts into out_root/<case_name>-<YYYYMMDD-HHMMSS>/ and returns
// that directory path.
std::string dump_case(
    const std::string & out_root, const GemmCase & c,
    const TilingSpec & ts, const WQuantType & wt,
    const void * w_blocks, size_t w_blocks_bytes,        // dump buffer, §4.1 order
    const char * w_layout_desc,                          // weight block byte layout
    const std::vector<uint8_t> & a_blocks,               // §4.1 act group blocks (raw)
    const std::vector<float> & A_src,                    // [M*K] native-dtype source act
    const std::vector<float> & W_src,                    // [N*K] fp16-source weight
    const std::vector<uint8_t> * out_bufs,               // chip dtype outputs (out_specs order, nsp entries)
    const std::vector<float> & C_ref_tiled,              // [M*N] golden, result tile order
    const CaseError & err);

// ---- qkt chain (docs/chain-qkt.md) ----------------------------------------

struct ChainVerify {
    CaseError err1;              // MMA1: GPU vs integer golden
    double    dq1_max_abs;       // MMA1 dequant cross-check (psum=0), <0 if skipped
    bool      k_bitexact;        // fused-quant blocks: GPU vs CPU byte equality
    CaseError err2;              // MMA2: GPU vs integer golden
    double    dq2_max_abs;       // MMA2 dequant cross-check, <0 if skipped
};

// Writes the full two-stage chain artifact set into
// out_root/<case_name>-<YYYYMMDD-HHMMSS>/ and returns that directory path.
// Shapes: MMA1 = c1.{M,N,K}; MMA2 = [M2, num_keys=c1.M, head_dim=c1.N].
std::string dump_chain_case(
    const std::string & out_root, const std::string & case_name,
    const GemmCase & c1, const TilingSpec & ts1, const WQuantType & wt1,
    const void * w1_blocks, size_t w1_blocks_bytes,
    const char * w1_layout_desc,
    const std::vector<uint8_t> & a1_blocks,              // X quant blocks (ts1)
    const std::vector<float> & X_src,                    // [M1*K1]
    const std::vector<float> & Wk_src,                   // [N1*K1]
    const std::vector<float> & C1_tiled,                 // [M1*N1] fp32 rslt (result order ts1)
    int kbits, const WQuantType & wtk,                   // K blocks type (q8_0_64 / q4_0_64)
    const std::vector<uint8_t> & kblocks_pre,            // pre-transpose (token-major)
    const std::vector<uint8_t> & kblocks_post,           // post-transpose (weight order ts2)
    const TilingSpec & ts2, int64_t M2,
    const std::vector<uint8_t> & q_blocks,               // Q quant blocks (ts2)
    const std::vector<float> & Q_src,                    // [M2*head_dim]
    const std::vector<uint8_t> * mma2_out_bufs,          // out_specs order (nsp entries)
    const std::vector<float> & C2_ref_tiled,             // [M2*num_keys] golden
    const ChainVerify & v);

} // namespace rgd
