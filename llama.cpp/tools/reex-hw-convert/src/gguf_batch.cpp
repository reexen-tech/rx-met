#include "gguf_batch.h"

#include "wconvert.h"
#include "wquant.h"
#include "reex_layout.h"

#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

namespace rgd {

// ---- ggml GGUF type -> reex registry name (Legacy + symmetric K-quant b64) --
static const char * ggml_type_to_registry(enum ggml_type t) {
    switch (t) {
        case GGML_TYPE_Q4_0_64:  return "q4_0_64";
        case GGML_TYPE_Q8_0_64:  return "q8_0_64";
        case GGML_TYPE_Q4_1_64:  return "q4_1_64";
        case GGML_TYPE_Q5_0_64:  return "q5_0_64";
        case GGML_TYPE_Q5_1_64:  return "q5_1_64";
        case GGML_TYPE_Q8_1_64:  return "q8_1_64";
        // K-quant block-64: the reex struct already IS the HW bit-stream, so
        // wconvert_weight is a whole-block reorder (repack is the identity).
        case GGML_TYPE_Q6_K_64:  return "Q6_K_64";
        case GGML_TYPE_Q5_K_64S: return "Q5_K_64S";
        case GGML_TYPE_Q4_K_64S: return "Q4_K_64S";
        case GGML_TYPE_Q3_K_64:  return "Q3_K_64";
        case GGML_TYPE_Q2_K_64S: return "Q2_K_64S";
        default:                 return "";         // unsupported for HW conversion
    }
}

// Tensors that llama.cpp quantizes to a block type but that are NOT matmul
// weights (consumed via ggml_get_rows). These must never be HW-tiled even though
// they carry a convertible quant type. Mirrors llama-quant.cpp's token-embd set.
static bool is_token_embd(const std::string & name) {
    return name == "token_embd.weight" || name == "per_layer_token_embd.weight";
}

static bool name_matches(const std::string & name, const std::vector<std::regex> & res) {
    for (const auto & re : res)
        if (std::regex_search(name, re)) return true;
    return false;
}

static bool mkdir_p(const std::string & path) {
    std::string cur;
    for (size_t i = 0; i < path.size(); ++i) {
        cur += path[i];
        if (path[i] == '/' || i + 1 == path.size()) {
            std::string d = (path[i] == '/') ? cur.substr(0, cur.size() - 1) : cur;
            if (d.empty()) continue;
            if (mkdir(d.c_str(), 0755) != 0 && errno != EEXIST) return false;
        }
    }
    return true;
}

static std::string json_escape(const std::string & s) {
    std::string o;
    for (char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += c; }
        else o += c;
    }
    return o;
}

static void write_index(const std::string & path, const GgufConvSummary & s,
                        const std::string & in_gguf) {
    std::ostringstream o;
    o << "{\n";
    o << "  \"tool\": \"reex-hw-convert\",\n";
    o << "  \"source_gguf\": \"" << json_escape(in_gguf) << "\",\n";
    o << "  \"output_gguf\": \"" << json_escape(s.out_gguf) << "\",\n";
    o << "  \"summary\": { \"n_tensors\": " << s.n_tensors << ", \"n_ok\": " << s.n_ok
      << ", \"n_skip\": " << s.n_skip << ", \"n_fail\": " << s.n_fail << " },\n";
    o << "  \"tensors\": [\n";
    for (size_t i = 0; i < s.items.size(); ++i) {
        const GgufConvItem & it = s.items[i];
        o << "    { \"name\": \"" << json_escape(it.name) << "\""
          << ", \"expert\": " << it.expert
          << ", \"ggml_type\": \"" << json_escape(it.ggml_type) << "\""
          << ", \"wtype\": \"" << json_escape(it.wtype) << "\""
          << ", \"N\": " << it.N << ", \"K\": " << it.K
          << ", \"status\": \"" << json_escape(it.status) << "\""
          << ", \"bytes\": " << it.bytes << " }";
        o << (i + 1 == s.items.size() ? "\n" : ",\n");
    }
    o << "  ]\n}\n";
    std::ofstream f(path, std::ios::binary);
    f << o.str();
}

// Copy `n` raw bytes from the input stream (at absolute `off`) to output stream.
static bool copy_bytes(std::ifstream & in, uint64_t off, std::ofstream & out, size_t n) {
    in.clear();
    in.seekg((std::streamoff) off, std::ios::beg);
    std::vector<char> buf(1 << 20);
    size_t left = n;
    while (left) {
        const size_t chunk = left < buf.size() ? left : buf.size();
        in.read(buf.data(), (std::streamsize) chunk);
        if ((size_t) in.gcount() != chunk) return false;
        out.write(buf.data(), (std::streamsize) chunk);
        left -= chunk;
    }
    return true;
}

static bool read_at(std::ifstream & f, uint64_t off, void * dst, size_t n) {
    f.clear();
    f.seekg((std::streamoff) off, std::ios::beg);
    f.read((char *) dst, (std::streamsize) n);
    return (size_t) f.gcount() == n;
}

int wconvert_gguf(const std::string & in_path, const std::string & out_gguf,
                  const std::vector<std::string> & patterns_in,
                  const std::string & only_tensor,
                  const std::string & dump_dir,
                  GgufConvSummary & summary,
                  bool dry_run) {
    const std::string index_path = out_gguf + ".hw_index.json";

    // Selection mode:
    //   auto   : no --tensor and no --pattern -> pick by GGUF quant type
    //   explicit: --tensor NAME, or one/more --pattern RE (user asserts intent)
    const bool has_only  = !only_tensor.empty();
    const bool has_pat   = !patterns_in.empty();
    const bool auto_mode = !has_only && !has_pat;

    std::vector<std::regex> res;
    if (has_pat) {
        try {
            for (const auto & p : patterns_in) res.emplace_back(p);
        } catch (const std::regex_error & e) {
            fprintf(stderr, "[wconvert-gguf] bad pattern regex: %s\n", e.what());
            return 2;
        }
    }

    // open input GGUF (metadata only, no data alloc)
    struct ggml_context * meta = nullptr;
    struct gguf_init_params gp;
    gp.no_alloc = true;
    gp.ctx      = &meta;
    struct gguf_context * gg = gguf_init_from_file(in_path.c_str(), gp);
    if (!gg) {
        fprintf(stderr, "[wconvert-gguf] failed to open GGUF: %s\n", in_path.c_str());
        return 2;
    }
    std::ifstream fin(in_path, std::ios::binary);
    if (!fin) {
        fprintf(stderr, "[wconvert-gguf] cannot open for read: %s\n", in_path.c_str());
        gguf_free(gg); if (meta) ggml_free(meta);
        return 2;
    }
    const uint64_t data_off_in = (uint64_t) gguf_get_data_offset(gg);
    const int64_t  nt          = gguf_get_n_tensors(gg);
    summary.n_tensors = (int) nt;

    const TilingSpec tsL = tiling_for(Family::Legacy);

    // ---- pass 1: classify + validate (hard fail on bad shape) ----
    struct Sel { std::string wtype; int64_t N, K, nexp; };
    std::vector<Sel> selN(nt);         // per-tensor selection (nexp==0 => not converted)
    std::vector<std::string> converted_names;

    for (int64_t i = 0; i < nt; ++i) {
        selN[i].nexp = 0;
        const std::string name = gguf_get_tensor_name(gg, i);
        const enum ggml_type gt = gguf_get_tensor_type(gg, i);
        const char * reg = ggml_type_to_registry(gt);

        // ---- is this tensor a candidate for conversion? ----
        bool candidate;
        if (has_only)      candidate = (name == only_tensor);
        else if (has_pat)  candidate = name_matches(name, res);
        else               candidate = (reg[0] != 0);   // auto: any Legacy block-64 weight
        if (!candidate) continue;

        // Explicit mode may match a non-convertible type; auto mode never does.
        if (!reg[0]) {
            GgufConvItem it; it.name = name; it.ggml_type = ggml_type_name(gt);
            it.status = "skip:unsupported-type";
            summary.items.push_back(it); summary.n_skip++;
            continue;
        }

        // Auto mode: exclude the token embedding (quantized but used via get_rows).
        if (auto_mode && is_token_embd(name)) {
            GgufConvItem it; it.name = name; it.ggml_type = ggml_type_name(gt);
            it.wtype = reg; it.status = "skip:embedding";
            summary.items.push_back(it); summary.n_skip++;
            continue;
        }

        const struct ggml_tensor * t = ggml_get_tensor(meta, name.c_str());
        if (!t) {
            fprintf(stderr, "[wconvert-gguf] tensor meta missing: %s\n", name.c_str());
            gguf_free(gg); if (meta) ggml_free(meta);
            write_index(index_path, summary, in_path);
            return 2;
        }
        const int64_t K    = t->ne[0];
        const int64_t N    = t->ne[1];
        const int64_t nexp = t->ne[2] > 0 ? t->ne[2] : 1;
        // Divisibility depends on the target family's tile geometry (K-quant
        // uses Kt=256 vs Legacy Kt=64), so resolve the tiling per selected type.
        const int        wid_chk = wquant_find(reg);
        const TilingSpec ts      = wid_chk >= 0 ? tiling_for(wquant_get(wid_chk).family) : tsL;
        if (t->ne[3] > 1 || K % ts.Kt || N % ts.Nt) {
            // Auto mode: not tileable -> skip (e.g. ssm_alpha/beta with N=32).
            // Explicit mode: the user named it, so a bad shape is a hard error.
            if (auto_mode) {
                GgufConvItem it; it.name = name; it.ggml_type = ggml_type_name(gt);
                it.wtype = reg; it.N = N; it.K = K; it.status = "skip:shape-not-divisible";
                summary.items.push_back(it); summary.n_skip++;
                continue;
            }
            fprintf(stderr,
                    "[wconvert-gguf] %s: shape N=%lld K=%lld ne3=%lld not tiling-divisible "
                    "(Nt=%d Kt=%d) — aborting\n",
                    name.c_str(), (long long) N, (long long) K, (long long) t->ne[3],
                    ts.Nt, ts.Kt);
            GgufConvItem it; it.name = name; it.ggml_type = ggml_type_name(gt);
            it.wtype = reg; it.N = N; it.K = K; it.status = "fail:shape-not-divisible";
            summary.items.push_back(it); summary.n_fail++;
            gguf_free(gg); if (meta) ggml_free(meta);
            write_index(index_path, summary, in_path);
            return 2;
        }
        selN[i] = Sel{ reg, N, K, nexp };
        converted_names.push_back(name);
    }

    // ---- dry-run: report the plan (per-expert ok counts) and stop ----
    if (dry_run) {
        for (int64_t i = 0; i < nt; ++i) {
            if (selN[i].nexp == 0) continue;
            const std::string name = gguf_get_tensor_name(gg, i);
            const enum ggml_type gt = gguf_get_tensor_type(gg, i);
            const Sel & sel = selN[i];
            const int wid = wquant_find(sel.wtype.c_str());
            const size_t per_exp = wid >= 0 ? wquant_blocks_bytes(wid, sel.N, sel.K) : 0;
            for (int64_t e = 0; e < sel.nexp; ++e) {
                GgufConvItem it;
                it.name = name; it.expert = (sel.nexp > 1 ? (int) e : -1);
                it.wtype = sel.wtype; it.ggml_type = ggml_type_name(gt);
                it.N = sel.N; it.K = sel.K; it.status = "ok"; it.bytes = per_exp;
                summary.items.push_back(it); summary.n_ok++;
            }
        }
        {
            const size_t slash = index_path.find_last_of('/');
            if (slash != std::string::npos) mkdir_p(index_path.substr(0, slash));
        }
        write_index(index_path, summary, in_path);
        printf("[dry-run] would convert %d tensor-slices, skip %d, fail %d "
               "(of %d tensors)\n", summary.n_ok, summary.n_skip, summary.n_fail,
               summary.n_tensors);
        for (const auto & it : summary.items) {
            if (it.status == "ok") continue;   // list only the non-obvious decisions
            printf("  %-26s  %-22s  N=%-7lld K=%-7lld  %s\n",
                   it.status.c_str(), it.name.c_str(), it.N, it.K,
                   it.ggml_type.c_str());
        }
        gguf_free(gg); if (meta) ggml_free(meta);
        return converted_names.empty() ? 1 : 0;
    }

    if (converted_names.empty()) {
        fprintf(stderr, "[wconvert-gguf] no convertible (Legacy / K-quant block-64) weight "
                        "tensors matched; no GGUF written\n");
        write_index(index_path, summary, in_path);
        gguf_free(gg); if (meta) ggml_free(meta);
        return 1;
    }

    // ---- build the output GGUF context (KV + tensor infos + HW markers) ----
    struct gguf_context * out = gguf_init_empty();
    gguf_set_kv(out, gg);                         // copy every KV pair
    for (int64_t i = 0; i < nt; ++i) {
        const struct ggml_tensor * t = ggml_get_tensor(meta, gguf_get_tensor_name(gg, i));
        gguf_add_tensor(out, t);                  // same name/type/shape; data written manually
    }
    gguf_set_val_bool(out, "reex.hw_layout", true);
    gguf_set_val_str (out, "reex.hw_tool", "reex-hw-convert");
    {
        std::vector<const char *> cptr;
        cptr.reserve(converted_names.size());
        for (const auto & s : converted_names) cptr.push_back(s.c_str());
        gguf_set_arr_str(out, "reex.hw_converted_tensors", cptr.data(), cptr.size());
    }
    const size_t alignment = gguf_get_alignment(out);
    gguf_set_val_u32(out, "general.alignment", (uint32_t) alignment);

    // ---- stream-write: meta, then each tensor's data at meta+offset (padded) ----
    if (!dump_dir.empty() && !mkdir_p(dump_dir)) {
        fprintf(stderr, "[wconvert-gguf] mkdir failed: %s\n", dump_dir.c_str());
        gguf_free(out); gguf_free(gg); if (meta) ggml_free(meta);
        return 2;
    }
    // ensure parent dir of out_gguf exists
    {
        const size_t slash = out_gguf.find_last_of('/');
        if (slash != std::string::npos) mkdir_p(out_gguf.substr(0, slash));
    }

    std::ofstream fout(out_gguf, std::ios::binary);
    if (!fout) {
        fprintf(stderr, "[wconvert-gguf] cannot open for write: %s\n", out_gguf.c_str());
        gguf_free(out); gguf_free(gg); if (meta) ggml_free(meta);
        return 2;
    }

    const size_t size_meta = gguf_get_meta_size(out);
    {
        std::vector<uint8_t> metabuf(size_meta);
        gguf_get_meta_data(out, metabuf.data());
        fout.write((const char *) metabuf.data(), (std::streamsize) size_meta);
    }

    int rc = 0;
    uint64_t written = size_meta;  // running file position (must equal size_meta+off_i)
    auto pad_to_align = [&]() {
        while (written % alignment != 0) { fout.put('\0'); written++; }
    };

    for (int64_t i = 0; i < nt && rc == 0; ++i) {
        const std::string name = gguf_get_tensor_name(gg, i);
        const enum ggml_type gt = gguf_get_tensor_type(gg, i);
        const uint64_t in_abs = data_off_in + (uint64_t) gguf_get_tensor_offset(gg, i);
        const size_t tsize    = gguf_get_tensor_size(gg, i);

        if (selN[i].nexp == 0) {
            // copy through verbatim
            if (!copy_bytes(fin, in_abs, fout, tsize)) {
                fprintf(stderr, "[wconvert-gguf] copy failed: %s\n", name.c_str());
                rc = 2; break;
            }
            written += tsize;
            pad_to_align();
            continue;
        }

        // convert (per expert for MoE 3D)
        const Sel & sel = selN[i];
        const int wid = wquant_find(sel.wtype.c_str());
        const size_t per_exp = wquant_blocks_bytes(wid, sel.N, sel.K);
        if (per_exp == 0 || per_exp * (size_t) sel.nexp != tsize) {
            fprintf(stderr, "[wconvert-gguf] %s: size mismatch (tensor=%zu, %lld*%zu)\n",
                    name.c_str(), tsize, (long long) sel.nexp, per_exp);
            GgufConvItem it; it.name = name; it.wtype = sel.wtype;
            it.N = sel.N; it.K = sel.K; it.status = "fail:size-mismatch";
            summary.items.push_back(it); summary.n_fail++;
            rc = 2; break;
        }

        std::vector<uint8_t> native(per_exp);
        for (int64_t e = 0; e < sel.nexp && rc == 0; ++e) {
            if (!read_at(fin, in_abs + (uint64_t) e * per_exp, native.data(), per_exp)) {
                fprintf(stderr, "[wconvert-gguf] read failed: %s expert %lld\n",
                        name.c_str(), (long long) e);
                GgufConvItem it; it.name = name; it.expert = (sel.nexp > 1 ? (int) e : -1);
                it.wtype = sel.wtype; it.N = sel.N; it.K = sel.K; it.status = "fail:read";
                summary.items.push_back(it); summary.n_fail++;
                rc = 2; break;
            }
            const WConvResult r = wconvert_weight(wid, native.data(), native.size(),
                                                  sel.N, sel.K);
            if (r.bytes.size() != per_exp) {
                GgufConvItem it; it.name = name; it.expert = (sel.nexp > 1 ? (int) e : -1);
                it.wtype = sel.wtype; it.N = sel.N; it.K = sel.K; it.status = "fail:convert";
                summary.items.push_back(it); summary.n_fail++;
                rc = 2; break;
            }
            fout.write((const char *) r.bytes.data(), (std::streamsize) r.bytes.size());
            written += r.bytes.size();

            if (!dump_dir.empty()) {
                std::string sub = name;
                if (sel.nexp > 1) {
                    char eb[32]; snprintf(eb, sizeof(eb), "/expert_%03d", (int) e);
                    sub += eb;
                }
                const std::string dir = dump_dir + "/" + sub;
                if (mkdir_p(dir)) {
                    std::ofstream wf(dir + "/weight_blocks.bin", std::ios::binary);
                    wf.write((const char *) r.bytes.data(), (std::streamsize) r.bytes.size());
                    std::ofstream mf(dir + "/meta.json", std::ios::binary);
                    mf << wconvert_meta_json(wid, sel.N, sel.K, r);
                }
            }

            GgufConvItem it;
            it.name = name; it.expert = (sel.nexp > 1 ? (int) e : -1);
            it.wtype = sel.wtype; it.ggml_type = ggml_type_name(gt);
            it.N = sel.N; it.K = sel.K; it.status = "ok"; it.bytes = r.bytes.size();
            summary.items.push_back(it); summary.n_ok++;
        }
        if (rc == 0) pad_to_align();
    }

    fout.flush();
    const bool write_ok = (bool) fout;
    fout.close();

    if (rc == 0 && write_ok) {
        summary.out_gguf = out_gguf;
    } else if (rc == 0) {
        rc = 2; // stream error
    }
    write_index(index_path, summary, in_path);

    gguf_free(out);
    gguf_free(gg);
    if (meta) ggml_free(meta);
    return rc;
}

} // namespace rgd
