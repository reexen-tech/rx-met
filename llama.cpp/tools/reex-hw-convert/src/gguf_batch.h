// Whole-GGUF weight conversion: read a quantized GGUF and write a *new* GGUF
// (same container, same KV, same tensor types/shapes) where the matched Legacy
// block-64 GEMM weight tensors have their bytes replaced by the hardware
// weight_blocks (tiled) layout. CPU-only, streaming (no full-model load).
//
// The output GGUF is marked with `reex.hw_layout=true` (+ the list of converted
// tensors) so consumers know its weight data is HW-tiled and must NOT be fed to
// the stock llama.cpp inference path.
//
// Phase-2 scope: Legacy block-64 only. Unsupported tensor types (F16, K-quant)
// are copied through verbatim.
//
// Selection (which weights become HW-tiled):
//   * Auto (default, no --pattern / --tensor): pick EVERY tensor whose GGUF type
//     is a Legacy block-64 quant (== llama.cpp already decided it is a quantizable
//     GEMM weight), minus the token-embedding (get_rows, not a matmul). A candidate
//     whose shape is not tiling-divisible (e.g. ssm_alpha/beta with N=32) is
//     *skipped*, not fatal. This is architecture-agnostic and reuses llama.cpp's
//     own quant deny-list implicitly (no duplicated name table here).
//   * Explicit (--pattern RE... or --tensor NAME): the user asserts these must
//     convert, so a non-tiling-divisible shape is a hard error.
#pragma once

#include <string>
#include <vector>

namespace rgd {

struct GgufConvItem {
    std::string name;                 // tensor name
    int         expert   = -1;        // -1 for 2D, else expert index
    std::string wtype;                // registry name ("" if unsupported)
    std::string ggml_type;            // ggml type_name as stored in the GGUF
    long long   N        = 0;
    long long   K        = 0;
    std::string status;               // "ok" | "skip:{unsupported-type,embedding,shape-not-divisible}" | "fail:..."
    unsigned long long bytes = 0;     // converted bytes (when ok)
};

struct GgufConvSummary {
    int n_tensors = 0;   // total tensors in the GGUF
    int n_ok      = 0;   // converted (per-expert counted)
    int n_skip    = 0;   // matched but unsupported type
    int n_fail    = 0;
    std::vector<GgufConvItem> items;
    std::string out_gguf;             // path of the written HW GGUF ("" if none)
};

// Read `in_path`, write a HW-tiled GGUF to `out_gguf`.
//   patterns    : regex list; empty -> AUTO type-based selection (see above).
//   only_tensor : if non-empty, convert exactly this tensor (patterns ignored).
//   dump_dir    : if non-empty, ALSO dump per-tensor weight_blocks.bin + meta.json there.
//   dry_run     : classify + report only; write the sidecar index but NO GGUF.
// A sidecar `<out_gguf>.hw_index.json` is always written with the summary.
// Returns 0 on success; 1 if nothing convertible was found (no GGUF written);
// 2 on a hard error (bad shape / read / parse / write).
int wconvert_gguf(const std::string & in_path, const std::string & out_gguf,
                  const std::vector<std::string> & patterns,
                  const std::string & only_tensor,
                  const std::string & dump_dir,
                  GgufConvSummary & summary,
                  bool dry_run = false);

} // namespace rgd
