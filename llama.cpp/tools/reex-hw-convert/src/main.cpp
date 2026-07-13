// reex-hw-convert — offline single-tensor weight converter (CPU-only).
//
// Converts a Legacy block-64 weight tensor W[N,K] from native (GGUF / reex
// row-major struct) layout into the hardware weight_blocks layout (§4.1 tile
// order), plus a self-describing meta.json. Input comes from a native quantized
// .bin, an fp32 .bin (quantized on the fly), or deterministic random data.
//
//   reex-hw-convert --wtype q8_0_64 --shape N,K
//                   ( --in native.bin | --in-fp32 fp32.bin | --random [--seed S] )
//                   --out-dir DIR
#include "wconvert.h"
#include "wquant.h"
#include "reex_layout.h"
#include "gguf_batch.h"

#include "ggml.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace rgd;

enum class InMode { None, Native, Fp32, Random, Gguf };

static bool read_file(const std::string & path, std::vector<uint8_t> & buf) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "cannot open '%s'\n", path.c_str()); return false; }
    fseek(f, 0, SEEK_END);
    const long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    buf.resize((size_t) (sz < 0 ? 0 : sz));
    const size_t got = buf.empty() ? 0 : fread(buf.data(), 1, buf.size(), f);
    fclose(f);
    if (got != buf.size()) { fprintf(stderr, "short read on '%s'\n", path.c_str()); return false; }
    return true;
}

static bool write_file(const std::string & path, const void * data, size_t bytes) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) { fprintf(stderr, "cannot write '%s'\n", path.c_str()); return false; }
    const size_t put = bytes ? fwrite(data, 1, bytes, f) : 0;
    fclose(f);
    if (put != bytes) { fprintf(stderr, "short write on '%s'\n", path.c_str()); return false; }
    return true;
}

static bool parse_shape(const char * s, int64_t & N, int64_t & K) {
    // "N,K"
    const char * comma = strchr(s, ',');
    if (!comma) return false;
    N = atoll(s);
    K = atoll(comma + 1);
    return N > 0 && K > 0;
}

static void usage(const char * prog) {
    printf("Usage:\n"
           "  Single tensor:\n"
           "    %s --wtype NAME --shape N,K\n"
           "       ( --in FILE | --in-fp32 FILE | --random ) [--seed S] [--out-dir DIR]\n"
           "  Whole GGUF -> HW GGUF (Legacy only):\n"
           "    %s --in-gguf MODEL.gguf [--out MODEL-hw.gguf] [--pattern RE]... [--tensor NAME] [--dump-dir DIR] [--dry-run]\n\n"
           "  --wtype     Legacy block-64: q8_0_64 q8_1_64 q4_0_64 q4_1_64 q5_0_64 q5_1_64\n"
           "  --shape     N,K  (N=output rows=ne1, K=input dim=ne0; K contiguous)\n"
           "  --in        native quantized bytes (reex struct, row-major) -> reorder only\n"
           "  --in-fp32   fp32 weights (N*K floats, row-major) -> encode then reorder\n"
           "  --random    generate deterministic fp32 weights -> encode then reorder\n"
           "  --seed      RNG seed for --random (default 1234)\n"
           "  --out-dir   single-tensor output directory (default output/wconvert)\n"
           "  --in-gguf   read a GGUF and write a HW-tiled GGUF (matched Legacy weights only)\n"
           "  --out       output GGUF path (default: <input>-hw.gguf)\n"
           "  --pattern   name regex to match (repeatable; overrides auto selection)\n"
           "  --tensor    convert exactly this tensor name (overrides --pattern)\n"
           "  --dump-dir  also dump per-tensor weight_blocks.bin + meta.json here (debug/validate)\n"
           "  --dry-run   classify + report the plan (writes .hw_index.json, no GGUF)\n"
           "  (default selection: every Legacy block-64 quant weight except token_embd;\n"
           "   non-tiling-divisible candidates are skipped)\n",
           prog, prog);
}

// "model.gguf" -> "model-hw.gguf"; "model" -> "model-hw.gguf"
static std::string default_hw_out(const std::string & in) {
    const std::string suf = ".gguf";
    if (in.size() >= suf.size() && in.compare(in.size() - suf.size(), suf.size(), suf) == 0)
        return in.substr(0, in.size() - suf.size()) + "-hw.gguf";
    return in + "-hw.gguf";
}

int main(int argc, char ** argv) {
    std::string wtype;
    std::string in_path;
    std::string gguf_path;
    std::string out_gguf;
    std::string dump_dir;
    std::string only_tensor;
    std::vector<std::string> patterns;
    bool        dry_run = false;
    std::string out_dir = "output/wconvert";
    int64_t     N = 0, K = 0;
    uint64_t    seed = 1234;
    InMode      mode = InMode::None;

    for (int i = 1; i < argc; ++i) {
        if      (!strcmp(argv[i], "--wtype")   && i + 1 < argc) wtype = argv[++i];
        else if (!strcmp(argv[i], "--shape")   && i + 1 < argc) {
            if (!parse_shape(argv[++i], N, K)) { fprintf(stderr, "bad --shape (want N,K)\n"); return 1; }
        }
        else if (!strcmp(argv[i], "--in")      && i + 1 < argc) { in_path = argv[++i]; mode = InMode::Native; }
        else if (!strcmp(argv[i], "--in-fp32") && i + 1 < argc) { in_path = argv[++i]; mode = InMode::Fp32; }
        else if (!strcmp(argv[i], "--random"))                  { mode = InMode::Random; }
        else if (!strcmp(argv[i], "--in-gguf") && i + 1 < argc) { gguf_path = argv[++i]; mode = InMode::Gguf; }
        else if (!strcmp(argv[i], "--out")     && i + 1 < argc) out_gguf = argv[++i];
        else if (!strcmp(argv[i], "--dump-dir")&& i + 1 < argc) dump_dir = argv[++i];
        else if (!strcmp(argv[i], "--pattern") && i + 1 < argc) patterns.push_back(argv[++i]);
        else if (!strcmp(argv[i], "--tensor")  && i + 1 < argc) only_tensor = argv[++i];
        else if (!strcmp(argv[i], "--dry-run"))                 dry_run = true;
        else if (!strcmp(argv[i], "--seed")    && i + 1 < argc) seed = strtoull(argv[++i], nullptr, 10);
        else if (!strcmp(argv[i], "--out-dir") && i + 1 < argc) out_dir = argv[++i];
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) { usage(argv[0]); return 0; }
        else { fprintf(stderr, "Unknown arg: %s\n", argv[i]); usage(argv[0]); return 1; }
    }

    // ---- whole-GGUF -> HW GGUF mode ----
    if (mode == InMode::Gguf) {
        if (out_gguf.empty()) out_gguf = default_hw_out(gguf_path);
        GgufConvSummary s;
        const int rc = wconvert_gguf(gguf_path, out_gguf, patterns, only_tensor, dump_dir, s, dry_run);
        fprintf(stderr,
                "[reex-hw-convert] gguf=%s  tensors=%d  ok=%d skip=%d fail=%d%s\n"
                "  -> %s\n"
                "  -> %s.hw_index.json\n",
                gguf_path.c_str(), s.n_tensors, s.n_ok, s.n_skip, s.n_fail,
                dry_run ? "  (dry-run)" : "",
                s.out_gguf.empty() ? "(no GGUF written)" : s.out_gguf.c_str(),
                out_gguf.c_str());
        return rc;
    }

    if (wtype.empty())        { fprintf(stderr, "missing --wtype\n"); return 1; }
    if (mode == InMode::None) { fprintf(stderr, "missing input (--in / --in-fp32 / --random)\n"); return 1; }
    if (N <= 0 || K <= 0)     { fprintf(stderr, "missing/invalid --shape N,K\n"); return 1; }

    const int wid = wquant_find(wtype.c_str());
    if (wid < 0) { fprintf(stderr, "wtype '%s' not in registry\n", wtype.c_str()); return 1; }
    const WQuantType & t = wquant_get(wid);

    if (t.family != Family::Legacy && t.family != Family::Kquant) {
        fprintf(stderr, "wtype '%s' (family %s) not supported — Legacy block-64 "
                        "(q8_0_64/q8_1_64/q4_0_64/q4_1_64/q5_0_64/q5_1_64) or "
                        "K-quant block-64 (Q6_K_64/Q5_K_64S/Q4_K_64S/Q3_K_64/Q2_K_64S)\n",
                t.name, family_name(t.family));
        return 1;
    }

    const TilingSpec ts = tiling_for(t.family);
    if (K % ts.Kt || N % ts.Nt) {
        fprintf(stderr, "shape N=%lld K=%lld not divisible by tiling (Nt=%d Kt=%d)\n",
                (long long) N, (long long) K, ts.Nt, ts.Kt);
        return 1;
    }

    // ---- produce native quantized bytes ----
    std::vector<uint8_t> native;
    if (mode == InMode::Native) {
        if (!read_file(in_path, native)) return 1;
    } else {
        std::vector<float> W;
        if (mode == InMode::Fp32) {
            std::vector<uint8_t> raw;
            if (!read_file(in_path, raw)) return 1;
            if (raw.size() != (size_t) (N * K) * sizeof(float)) {
                fprintf(stderr, "fp32 input size %zu != N*K*4 = %zu\n",
                        raw.size(), (size_t) (N * K) * sizeof(float));
                return 1;
            }
            W.resize((size_t) (N * K));
            std::memcpy(W.data(), raw.data(), raw.size());
        } else { // Random
            wconvert_random_weights(W, N, K, seed);
        }
        wconvert_encode_native(wid, W.data(), N, K, native);
    }

    // ---- convert native -> HW weight_blocks ----
    const WConvResult r = wconvert_weight(wid, native.data(), native.size(), N, K);
    if (r.bytes.empty()) { fprintf(stderr, "conversion failed\n"); return 1; }

    // ---- write outputs ----
    // best-effort mkdir -p via std filesystem-free approach
    {
        std::string cmd = "mkdir -p '" + out_dir + "'";
        if (system(cmd.c_str()) != 0) { fprintf(stderr, "mkdir failed: %s\n", out_dir.c_str()); return 1; }
    }
    const std::string p_blocks = out_dir + "/weight_blocks.bin";
    const std::string p_native = out_dir + "/weight_native.bin";
    const std::string p_meta   = out_dir + "/meta.json";
    if (!write_file(p_blocks, r.bytes.data(), r.bytes.size())) return 1;
    if (!write_file(p_native, native.data(), native.size()))   return 1;
    const std::string meta = wconvert_meta_json(wid, N, K, r);
    if (!write_file(p_meta, meta.data(), meta.size()))         return 1;

    fprintf(stderr,
            "[reex-hw-convert] %s  N=%lld K=%lld  blocks=%lld  block=%zuB (native %zuB)\n"
            "  -> %s (%zu bytes)\n  -> %s (%zu bytes)\n  -> %s\n",
            t.name, (long long) N, (long long) K, (long long) r.n_blocks,
            r.block_bytes, t.block_bytes,
            p_blocks.c_str(), r.bytes.size(),
            p_native.c_str(), native.size(),
            p_meta.c_str());
    return 0;
}
