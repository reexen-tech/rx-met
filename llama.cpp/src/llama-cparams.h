#pragma once

#include "llama.h"

#include <cstdint>

#define LLAMA_MAX_SEQ 256

struct llama_cparams {
    uint32_t n_ctx;           // context size used during inference
    uint32_t n_ctx_seq;       // context for a single sequence
    uint32_t n_batch;
    uint32_t n_ubatch;
    uint32_t n_seq_max;
    int32_t  n_threads;       // number of threads to use for generation
    int32_t  n_threads_batch; // number of threads to use for batch processing

    float rope_freq_base;
    float rope_freq_scale;

    uint32_t n_ctx_orig_yarn;
    // These hyperparameters are not exposed in GGUF, because all
    // existing YaRN models use the same values for them.
    float yarn_ext_factor;
    float yarn_attn_factor;
    float yarn_beta_fast;
    float yarn_beta_slow;

    bool embeddings;
    bool causal_attn;
    bool offload_kqv;
    bool flash_attn;
    bool auto_fa;
    bool fused_gdn_ar;       // use fused gated delta net (autoregressive)
    bool fused_gdn_ch;       // use fused gated delta net (chunked)
    bool auto_fgdn;
    bool no_perf;
    bool warmup;
    bool op_offload;
    bool kv_unified;
    bool pipeline_parallel;

    enum llama_pooling_type pooling_type;

    ggml_backend_sched_eval_callback cb_eval;
    void * cb_eval_user_data;

#ifdef REEX_TURBOQUANT
    // === REEX_TURBOQUANT BEGIN ===
    // TurboQuant KV-cache compression knobs.  See docs/turboquant/01_设计与实现计划.md §3.1.
    // Whether TQ actually applies to a layer is decided by the user-supplied
    // global cache type (`-ctk tq_k3` / `-ctv tq_v2|tq_v4`); see
    // reex_turboquant::resolve_layer_spec.  The remaining knobs below are kept
    // as orthogonal user-facing semantic dimensions (K bits / V bits / InnerQ
    // toggle / PRNG seed placeholder).
    int32_t tq_key_bits;        // bits per K element after rotation (currently only 3 -> TQ_K3 is implemented)
    int32_t tq_value_bits;      // bits per V element (2 -> TQ_V2 or 4 -> TQ_V4)
    int32_t tq_seed;            // PRNG seed; currently runtime no-op (WHT signs / Lloyd-Max codebook are build-time)
    bool    tq_innerq_enable;   // InnerQ EMA per-channel scale (P4 A2.4 plumbing; currently no-op)
    // === REEX_TURBOQUANT END ===
#endif
};
