# aimet_llama: JSON-driven LLM quantization pipeline

> **Goal of this module (P0)**: make llama.cpp quantization experiments
> **reproducible, auditable, version-controllable**. One JSON file replaces
> ad-hoc shell scripts that string together `llama-imatrix` → `llama-quantize`
> → `llama-perplexity`. The pipeline produces a fingerprinted manifest and
> a Markdown report so any prior run can be exactly re-executed and compared.

This is the LLM-side counterpart of the AIMET v2 `quick_start.py` flow used
for KWS / CV models. The configuration language deliberately mirrors AIMET v2
(`defaults / tensor_name_config / report`) so the same team can author
quantization recipes for both classical NN and LLM workloads.

---

## 1. What's in the box

Following the AIMET layout convention (``aimet_torch/`` / ``aimet_onnx/``
are pure source packages, runnable demos live under the sibling
``examples/`` directory):

```
aimet_rx/
├── aimet_llama/                         # <-- source package
│   ├── __init__.py
│   ├── schema.py                        # JSON schema, defaults, validators
│   ├── cli.py                           # JSON → llama.cpp CLI argv translators
│   ├── pipeline.py                      # End-to-end orchestrator + report + manifest
│   ├── tests.py                         # self-checks
│   └── README.md                        # (this file)
├── llama.cpp/                           # <-- in-tree REEX-enabled llama.cpp
│   └── REEX_Q64_USAGE.md                # block-64 quant + Psum truncation guide
└── examples/                            # <-- runnable demos
    ├── llm_quick_start.py
    └── config/
        ├── qwen3_30b_q4_0_minimal.json
        ├── qwen3_30b_mixed_precision.json
        ├── qwen3_reex_q4_k_64.json      # REEX block-64 + Psum truncation demo
        └── qwen3_30b_legacy_hw_export.json  # Legacy block-64 + hardware weight_blocks export
```

We wrap the upstream binaries plus the in-tree REEX-enabled `llama.cpp`
(`llama.cpp/build_cuda/bin/*`), which adds the block-64 quant family
(`Q4_0_64`, `Q4_K_64`, …) and runtime Psum bit-width truncation. See
`llama.cpp/REEX_Q64_USAGE.md` for the low-level details.

---

## 2. Two-line usage

All commands below assume you ``cd`` into ``aimet_rx/`` first.

```bash
# preview the exact CLI invocations this JSON will run (no side effects)
python -m aimet_llama.pipeline examples/config/qwen3_30b_q4_0_minimal.json --dry-run

# run end-to-end (imatrix → quantize → perplexity)
python -m aimet_llama.pipeline examples/config/qwen3_30b_q4_0_minimal.json
```

Or use the AIMET-style entry point that mirrors ``examples/quick_start.py``:

```bash
python examples/llm_quick_start.py examples/config/qwen3_30b_mixed_precision.json
```

---

## 3. What gets produced

Every run materialises a self-contained, **byte-fingerprinted** directory:

```
runs/<experiment_name>/
├── resolved_config.json    # the JSON after defaults + validation (re-runnable)
├── manifest.json           # SHA-256 of every input/binary/output, timings
├── imatrix.log             # stdout/stderr of the imatrix stage
├── quantize.log            # stdout/stderr of the quantize stage
├── perplexity.log          # stdout/stderr of the perplexity stage
├── imatrix.gguf            # (if calibration enabled)
├── <model>.quantized.gguf  # the quantized output
├── hw_export.log           # stdout/stderr of the hw_export stage (if enabled)
├── <model>.quantized-hw.gguf          # (if hw_export enabled) hardware-tiled GGUF
├── <model>.quantized-hw.gguf.hw_index.json  #   per-tensor summary (type/shape/status)
└── experiment_report.md    # human-readable report with all of the above
```

The manifest records:

- SHA-256 + size of input GGUF and calibration / PPL datasets
- SHA-256 of `llama-quantize`, `llama-imatrix`, `llama-perplexity` binaries
- SHA-256 + size of every produced artifact (imatrix, quantized GGUF)
- Per-stage `exit_code` and `elapsed_s`
- Final parsed perplexity (`PPL ± stderr`) parsed out of the log
- Host, Python version, wall clock start / finish

Any two runs of the same `resolved_config.json` on the same inputs must
produce identical fingerprints, otherwise something silently changed.

---

## 4. JSON schema (one-page tour)

```json
{
  "schema_version": "1.0",

  "experiment": {
    "name": "qwen3_30b_q4_0_minimal",
    "description": "...",
    "output_dir": "./runs/qwen3_30b_q4_0_minimal"
  },

  "binaries": {
    "llama_quantize":   "/path/to/llama-quantize",
    "llama_imatrix":    "/path/to/llama-imatrix",
    "llama_perplexity": "/path/to/llama-perplexity",
    "ld_library_path":  "/path/to/build/bin"
  },

  "model": {
    "gguf_fp16_path": "/path/to/model-f16.gguf"
  },

  "calibration": {
    "enabled":        true,
    "dataset_file":   "/path/to/wiki.train.raw",
    "imatrix_output": "imatrix.gguf",
    "n_chunks":       100,
    "context_length": 512,
    "n_gpu_layers":   99,
    "reuse_imatrix":  null
  },

  "quantization": {
    "default_type":         "Q4_K_M",
    "output_quantized":     "out.gguf",
    "output_tensor_type":   "Q6_K",
    "token_embedding_type": "Q4_K",
    "use_imatrix":          true,
    "n_threads":            16,
    "tensor_type_overrides": [
      { "pattern": ".*\\.attn_v\\.weight$",  "type": "Q5_K" },
      { "pattern": ".*\\.ffn_down\\..*",     "type": "Q5_K" },
      { "pattern": "^blk\\.[0-3]\\..*",      "type": "Q6_K" }
    ]
  },

  "evaluation": {
    "enabled": true,
    "perplexity": {
      "dataset_file":   "/path/to/wiki.test.raw",
      "context_length": 2048,
      "n_gpu_layers":   99,
      "parallel":       16,
      "flash_attn":     true
    }
  }
}
```

See `examples/*.json` for runnable copies. Every field has a default in
`schema.DEFAULT_CONFIG` so most configs are short.

### Field → CLI flag mapping (cheat sheet)

| JSON field | llama.cpp CLI |
|---|---|
| `quantization.default_type` | positional `type` arg of `llama-quantize` |
| `quantization.output_tensor_type` | `--output-tensor-type` |
| `quantization.token_embedding_type` | `--token-embedding-type` |
| `quantization.tensor_type_overrides[i]` | `--tensor-type "pattern=type"` (repeatable) |
| `quantization.use_imatrix` + `calibration.*` | `--imatrix imatrix.gguf` |
| `quantization.imatrix_include_weights[]` | `--include-weights` (repeatable) |
| `quantization.imatrix_exclude_weights[]` | `--exclude-weights` (repeatable) |
| `quantization.allow_requantize` | `--allow-requantize` |
| `quantization.leave_output_tensor` | `--leave-output-tensor` |
| `quantization.pure` | `--pure` |
| `quantization.prune_layers[]` | `--prune-layers L0,L1,L2` |
| `quantization.override_kv[]` | `--override-kv` (repeatable) |
| `calibration.dataset_file` | `llama-imatrix -f` |
| `calibration.n_chunks` | `--chunks` |
| `calibration.context_length` | `-c` |
| `calibration.n_gpu_layers` | `-ngl` |
| `calibration.process_output` | `--process-output` |
| `evaluation.perplexity.*` | `llama-perplexity` flags |
| `hw_export.input_gguf` / (default) | `reex-hw-convert --in-gguf` |
| `hw_export.output_gguf` / (default) | `--out` (default `<quantized>-hw.gguf`) |
| `hw_export.patterns[]` | `--pattern` (repeatable) |
| `hw_export.only_tensor` | `--tensor` |
| `hw_export.dump_dir` | `--dump-dir` |

---

## 4b. REEX block-64 quantization (in-tree llama.cpp)

The bundled `llama.cpp` build exposes a fully block=64-aligned quant family
on top of the standard ggml types. Select any of them as `default_type`,
`output_tensor_type`, `token_embedding_type` or in a `tensor_type_overrides`
entry exactly like a normal type:

```
Q4_0_64  Q5_0_64  Q8_0_64  Q8_1_64
Q4_1_64  Q5_1_64
Q2_K_64  Q3_K_64  Q4_K_64  Q5_K_64  Q6_K_64
Q2_K_64S Q4_K_64S Q5_K_64S
```

Example (`examples/config/qwen3_reex_q4_k_64.json`):

```json
"quantization": { "default_type": "Q4_K_64" }
```

### Psum bit-width truncation at evaluation time

The REEX runtime can truncate the fixed-point integer Psum to `B` bits
(64-element granularity, CPU and GPU stay numerically aligned). Drive it from
the config — the pipeline exports `REEX_Q64_PSUM_BITS=<B>` only for the
perplexity stage:

```json
"evaluation": {
  "perplexity": { "reex_psum_bits": 8 }
}
```

`reex_psum_bits` of `null` or `<= 0` disables truncation (full precision Psum).
Changing `B` needs **no** re-quantization or re-compilation, so a single
quantized GGUF can be swept across bit-widths just by editing the JSON. The
exported env var is recorded in `quantize.log` / `perplexity.log` and the
manifest for auditability.

---

## 4c. Hardware-tiled GGUF export (`hw_export`, Phase 2)

An **optional** stage that runs after `quantize` and rewrites the quantized
GGUF into a **new GGUF** (`<quantized>-hw.gguf`) whose GEMM weights carry the
hardware **tiled** `weight_blocks` layout — the exact byte layout consumed by
the simulator / hardware and produced by `reex-gemm-datagen`. Same container,
same KV metadata, same tensor types/shapes; only the data bytes of the matched
Legacy weights are re-tiled, everything else is copied verbatim. It reuses the
same validated tiling logic (`tools/reex-hw-convert`, a CPU-only tool) and
streams the file (no full-model load).

```json
"hw_export": {
  "enabled":         true,
  "binary":          "./llama.cpp/build_cpu_wconvert/bin/reex-hw-convert",
  "ld_library_path": "./llama.cpp/build_cpu_wconvert/bin",
  "input_gguf":      null,          // null -> the quantized GGUF this run produced
  "output_gguf":     null,          // null -> "<quantized_stem>-hw.gguf"
  "patterns":        [],            // empty -> attn_{q,k,v,output} + ffn_{gate,up,down}(_exps)
  "only_tensor":     null,          // convert exactly one tensor (overrides patterns)
  "dump_dir":        null           // optional: also dump per-tensor weight_blocks.bin
}
```

The output GGUF is tagged with `reex.hw_layout=true`, `reex.hw_tool` and
`reex.hw_converted_tensors` (the list of re-tiled tensors). **It must not be fed
to the stock llama.cpp inference path** — the tiled bytes keep the original
ggml type id, so a normal loader would misread them.

Behaviour (**Legacy block-64 scope**):

- **Converted**: `Q4_0_64 Q4_1_64 Q5_0_64 Q5_1_64 Q8_0_64 Q8_1_64` weight
  tensors whose name matches a pattern. MoE 3D expert tensors are converted
  **per expert**.
- **Copied verbatim**: everything else (F16 / K-quant, norms, `token_embd`,
  non-matching names) — the output is still a complete model container.
- **Hard error**: a matched, convertible tensor whose shape is not divisible by
  the tiling (`N % 64` / `K % 64`) — the stage fails loudly (exit 2).
- **Warning (non-fatal)**: if *nothing* convertible is matched (e.g. the model
  was quantized to a K-quant type), the tool exits 1, **no GGUF is written**,
  and the stage is marked with a `warning` in the report rather than failing.

A sidecar `<output>-hw.gguf.hw_index.json` summarises every matched tensor
(name, ggml type, resolved registry type, `N`, `K`, status, byte count); its
`summary` counts are copied into the manifest / report. Build the CPU tool once
with:

```bash
cmake -S llama.cpp -B llama.cpp/build_cpu_wconvert \
      -DGGML_CUDA=OFF -DREEX_HW_CONVERT=ON -DLLAMA_BUILD_TOOLS=ON
cmake --build llama.cpp/build_cpu_wconvert -j --target reex-hw-convert
```

See `tools/reex-hw-convert/README.md` for the standalone (single-tensor) usage
and the layout / validation details.

---

## 5. What this gives you over a shell script

| Concern | Shell script | aimet_llama |
|---|---|---|
| **Reproducibility** | `$ENV` and `pwd` state leak in | resolved_config.json is a hermetic re-runnable description |
| **Auditability** | "What was the exact CLI?" → re-read history | every stage logs `cmd_str`, `exit_code`, `elapsed_s` |
| **Tamper-evidence** | none | SHA-256 of inputs/binaries/outputs in manifest |
| **PPL extraction** | hand-grep | parsed automatically into `summary.perplexity` |
| **Version control** | tracking many `.sh` files | one `.json` per experiment, diffs are semantic |
| **Sharing a recipe** | "here is my bash, hope your paths are the same" | "here is my JSON, point the binaries at your build" |
| **AIMET parity** | n/a | same `defaults / *_config / report` shape as `quick_start.py` |

---

## 6. Programmatic API

```python
from aimet_llama import LLMQuantPipeline

pipe = LLMQuantPipeline.from_json("examples/qwen3_30b_mixed_precision.json")

# dry inspection (no subprocess calls)
for stage, argv in pipe.plan().items():
    print(stage, argv)

# real run
summary = pipe.run()
print(summary["perplexity"])   # {'ppl': 9.68, 'stderr': 0.077}
```

`LLMQuantPipeline.plan()` is a **pure function of the config**, useful for
unit tests and for showing reviewers the exact CLI that will run before
launching a multi-hour quantization job.

---

## 7. Scope (and what is intentionally out of scope)

Inside scope (this module):

- Reproducible, JSON-driven orchestration of upstream `llama.cpp` binaries.
- AIMET-style configuration syntax for tensor-level mixed precision.
- Manifest + Markdown report for every run.

Out of scope (deferred to later milestones; see
`docs/aimet_llamacpp_integration_plan.md`):

- Custom `block_size` quantization (P1, hooks into `tools/quantize_custom_blocksize.py`).
- Power-of-2 scale alignment / NPU-friendly post-quant transforms (P2).
- AIMET-style encodings sidecar export (P1).
- HF → GGUF conversion (kept as a separate stage, easy to add later).
- QAT / AdaRound / GPTQ (must remain in PyTorch; this module only consumes their output).
