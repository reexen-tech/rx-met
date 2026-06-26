// Artifact dumper — writes packed inputs, output, golden and a self-describing
// meta.json into out_root/<case_name>/.
#pragma once

#include "case.h"
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
    const std::vector<uint8_t> * out_bufs,               // 6 chip dtype outputs (out_specs order)
    const std::vector<float> & C_ref_tiled,              // [M*N] golden, result tile order
    const CaseError & err);

} // namespace rgd
