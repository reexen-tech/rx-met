#pragma once
#include "common.h"
#include <string>
#include <vector>
#include <regex>

// common debug functions and structs

// Print a tensor's detailed data
// data - the tensor's data in byte format
// type - the tensor's quantization type
// ne   - the tensor dimensions array
// nb   - the tensor strides array
// n    - the number of rows/columns to fully print
template <bool abort_on_nan> void common_debug_print_tensor(uint8_t * data, ggml_type type, const int64_t * ne, const size_t * nb, int64_t n);

// Intended to use as callback for ggml_backend_sched_eval_callback
// prints tensors that are processed in the computation graph
// by default prints all tensors, but can be configured by creating a `base_callback_data` instance with
// non-empty filter_patterns. See examples/debug.ccp for possible usage patterns
// The template parameter determines whether an error should be thrown whenever a NaN is encountered
// in a tensor (useful for stopping debug sessions on first erroneous tensor)
// The callback data will be passed as the third parameter (user_data)
template <bool abort_on_nan> bool common_debug_cb_eval(struct ggml_tensor * t, bool ask, void * user_data);

// ---------------------------------------------------------------------------
// MoE golden-reference dump (for RTL "专家重排" verification).
//
// This dump path is compiled UNCONDITIONALLY (it is NOT gated by GGML_USE_REEX,
// because the Q4_0_64 build only defines GGML_USE_REEX_Q64). It is gated at
// RUNTIME, either through this configuration API (preferred, used by
// dump_moe_prefill so the target directory can change per case) or through the
// environment variables REEX_DUMP_MOE_ONLY=1 + REEX_DUMP_DIR (+ optional
// REEX_DUMP_LAYER as a single-layer filter, reusing the existing semantics).
//
// When active, only MoE-related tensors emitted by build_moe_ffn() are dumped:
//   ffn_moe_topk-<L>            top-k selected expert ids (i32)
//   ffn_moe_weights-<L>         router weights (pre-norm)
//   ffn_moe_weights_norm-<L>    router weights (normalized; used for weighting)
//   ffn_moe_down-<L>            per-token per-expert output (unweighted)
//   ffn_moe_weighted-<L>        per-token per-expert output (weighted)
//   ffn_moe_out-<L>             final summed MoE output (sanity check)
//
// Each tensor is written as a raw `.bin` (contiguous in ggml logical order,
// i0 fastest, runtime dtype preserved) plus a `.json` sidecar capturing the
// tensor name, ggml dtype, type size, ne[4], nb[4] and layer index. The
// Python post-processor turns these into normalized `.npy` artifacts.
//
// Set dir to a valid directory and (optionally) a list of layer indices to
// restrict the dump. Pass layers=nullptr or n_layers<=0 to dump all layers.
void common_debug_moe_dump_set(const char * dir, const int * layers, int n_layers);

// Disable the configuration-API MoE dump (environment-variable mode, if set,
// still applies afterwards).
void common_debug_moe_dump_clear();
struct base_callback_data {
    std::vector<uint8_t>    data;
    std::vector<std::regex> tensor_filters;

    base_callback_data() = default;

    base_callback_data(common_params & params, const std::vector<std::string> & filter_patterns) {
        for (const auto & pattern : filter_patterns) {
            try {
                std::string anchored_pattern = "^" + pattern;
                tensor_filters.emplace_back(anchored_pattern, std::regex::optimize);
            } catch (const std::regex_error & e) {
                throw std::runtime_error("Invalid regex pattern '" + pattern + "': " + e.what());
            }
        }
        params.cb_eval           = common_debug_cb_eval<false>;
        params.cb_eval_user_data = this;
    }
};
