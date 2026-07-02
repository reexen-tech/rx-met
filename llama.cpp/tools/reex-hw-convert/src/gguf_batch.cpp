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

// ---- ggml GGUF type -> reex registry name (Legacy block-64 only) -----------
static const char * ggml_type_to_registry(enum ggml_type t) {
    switch (t) {
        case GGML_TYPE_Q4_0_64: return "q4_0_64";
        case GGML_TYPE_Q8_0_64: return "q8_0_64";
        case GGML_TYPE_Q4_1_64: return "q4_1_64";
        case GGML_TYPE_Q5_0_64: return "q5_0_64";
        case GGML_TYPE_Q5_1_64: return "q5_1_64";
        case GGML_TYPE_Q8_1_64: return "q8_1_64";
        default:                return "";         // unsupported for HW conversion
    }
}

static std::vector<std::string> default_patterns() {
    return {
        "(^|\\.)attn_(q|k|v|output)\\.weight$",
        "(^|\\.)ffn_(gate|up|down)(_exps)?\\.weight$",
    };
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
                  GgufConvSummary & summary) {
    const std::string index_path = out_gguf + ".hw_index.json";

    // compile patterns
    std::vector<std::string> pats = patterns_in.empty() ? default_patterns() : patterns_in;
    std::vector<std::regex> res;
    try {
        for (const auto & p : pats) res.emplace_back(p);
    } catch (const std::regex_error & e) {
        fprintf(stderr, "[wconvert-gguf] bad pattern regex: %s\n", e.what());
        return 2;
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
        const bool matched = only_tensor.empty() ? name_matches(name, res)
                                                 : (name == only_tensor);
        if (!matched) continue;

        const char * reg = ggml_type_to_registry(gt);
        if (!reg[0]) {
            GgufConvItem it; it.name = name; it.ggml_type = ggml_type_name(gt);
            it.status = "skip:unsupported-type";
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
        if (t->ne[3] > 1 || K % tsL.Kt || N % tsL.Nt) {
            fprintf(stderr,
                    "[wconvert-gguf] %s: shape N=%lld K=%lld ne3=%lld not tiling-divisible "
                    "(Nt=%d Kt=%d) — aborting\n",
                    name.c_str(), (long long) N, (long long) K, (long long) t->ne[3],
                    tsL.Nt, tsL.Kt);
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

    if (converted_names.empty()) {
        fprintf(stderr, "[wconvert-gguf] no convertible (Legacy block-64) weight tensors "
                        "matched; no GGUF written\n");
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
