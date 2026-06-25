// IntBlock pure-integer GEMM path (NO scale): INT activations @ INT weights ->
// INT accumulator -> direct cast to the requested output dtype. Parallel to the
// quantized path; self-contained (datagen + pack + GPU gemm + golden + dump).
#pragma once

#include "case.h"
#include "wquant.h"

#include <string>

namespace rgd {

// Run one IntBlock case end-to-end and dump artifacts into
// out_root/<case_name>-<timestamp>/. Returns that directory path.
std::string run_intblock_case(const std::string & out_root, const GemmCase & c,
                              const TilingSpec & ts, const WQuantType & wt);

} // namespace rgd
