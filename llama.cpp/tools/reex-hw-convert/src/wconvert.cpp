#include "wconvert.h"

#include "ggml.h"

#include <cstdio>
#include <cstring>
#include <random>
#include <sstream>

namespace rgd {

void wconvert_encode_native(int id, const float * W, int64_t N, int64_t K,
                            std::vector<uint8_t> & native) {
    const WQuantType & t = wquant_get(id);
    native.assign(wquant_blocks_bytes(id, N, K), 0);
    t.encode(W, native.data(), N, K);
}

void wconvert_random_weights(std::vector<float> & W, int64_t N, int64_t K,
                             uint64_t seed) {
    W.resize((size_t) (N * K));
    // Mirror datagen_fill's weight recipe: N(0, 0.5) rounded through fp16.
    std::mt19937_64 rng(seed ^ 0x9E3779B97F4A7C15ULL);
    std::normal_distribution<float> dist(0.0f, 0.5f);
    for (auto & x : W) x = ggml_fp16_to_fp32(ggml_fp32_to_fp16(dist(rng)));
}

WConvResult wconvert_weight(int id, const void * native, size_t native_bytes,
                            int64_t N, int64_t K) {
    const WQuantType & t  = wquant_get(id);
    const TilingSpec   ts = tiling_for(t.family);

    WConvResult r;

    const size_t nbytes = wquant_blocks_bytes(id, N, K);
    if (native_bytes != nbytes) {
        fprintf(stderr,
                "[wconvert] native size mismatch: got %zu, expected %zu "
                "(N=%lld K=%lld block=%zuB)\n",
                native_bytes, nbytes, (long long) N, (long long) K, t.block_bytes);
        return r; // empty -> caller treats as error
    }

    // Whole-block reorder into §4.1 tile order (block bytes opaque). Both Legacy
    // and K-quant reex structs already store the final HW byte layout — K-quant
    // is the LSB-first bit-stream itself — so no per-block repack is needed and
    // this reorder is the entire conversion.
    r.bytes.assign(nbytes, 0);
    wquant_reorder_to_tiled(id, native, r.bytes.data(), N, K, ts);
    r.block_bytes = t.block_bytes;
    r.n_blocks    = N * K / t.elems_per_block;
    r.repacked    = false;
    r.layout_desc = (t.family == Family::Kquant)
        ? "LSB-first bitstream: glb_scale(fp16,16b) + 4*[sub_scale(scale_bits, "
          "signed) + 64 codes @W bits] (whole-block reorder only)"
        : "reex native struct {d(fp16); ...; qs} (scale-first), block bytes "
          "opaque (whole-block reorder only)";
    return r;
}

std::string wconvert_meta_json(int id, int64_t N, int64_t K, const WConvResult & r) {
    const WQuantType & t  = wquant_get(id);
    const TilingSpec   ts = tiling_for(t.family);
    const int64_t Ntiles  = (ts.Nt > 0) ? N / ts.Nt : 0;

    std::ostringstream o;
    o << "{\n";
    o << "  \"tool\": \"reex-hw-convert\",\n";
    o << "  \"gemm\": { \"N\": " << N << ", \"K\": " << K << " },\n";
    o << "  \"weight\": {\n";
    o << "    \"wtype\": \"" << t.name << "\",\n";
    o << "    \"family\": \"" << family_name(t.family) << "\",\n";
    o << "    \"W_bits\": " << t.W_bits << ",\n";
    o << "    \"has_min\": " << (t.has_min ? "true" : "false") << ",\n";
    o << "    \"elems_per_block\": " << t.elems_per_block << ",\n";
    o << "    \"native_block_bytes\": " << t.block_bytes << ",\n";
    o << "    \"block_bytes\": " << r.block_bytes << ",\n";
    o << "    \"n_blocks\": " << r.n_blocks << ",\n";
    o << "    \"repacked\": " << (r.repacked ? "true" : "false") << ",\n";
    o << "    \"scale_bits\": " << t.scale_bits << ",\n";
    o << "    \"byte_layout\": \"" << r.layout_desc << "\"\n";
    o << "  },\n";
    o << "  \"tiling\": { \"Mt\": " << ts.Mt << ", \"Nt\": " << ts.Nt
      << ", \"Kt\": " << ts.Kt << ", \"wgroup\": " << ts.wgroup
      << ", \"Ntiles\": " << Ntiles << " },\n";
    o << "  \"index\": {\n";
    o << "    \"_doc\": \"weight_blocks stored in §4.1 block-tile order; each block "
         "spans Kt K-elements of ONE output row n.\",\n";
    o << "    \"sb\": \"k / Kt  (0.." << (K / ts.Kt - 1) << ")\",\n";
    o << "    \"weight_block_slot(n,sb)\": \"(sb*Ntiles + n/Nt)*Nt + n%Nt\",\n";
    o << "    \"note\": \"inter-tile (kt,nt) row-major; intra-tile column(N)-major; "
         "block bytes unchanged for Legacy.\"\n";
    o << "  },\n";
    o << "  \"files\": {\n";
    o << "    \"weight_blocks.bin\": { \"total_bytes\": " << r.bytes.size()
      << ", \"block_bytes\": " << r.block_bytes << ", \"n_blocks\": " << r.n_blocks
      << ", \"order\": \"weight_block_slot\" },\n";
    o << "    \"weight_native.bin\": { \"total_bytes\": "
      << ((size_t) r.n_blocks * t.block_bytes) << ", \"block_bytes\": "
      << t.block_bytes << ", \"n_blocks\": " << r.n_blocks
      << ", \"order\": \"row-major native[n*(K/Kt)+sb]\" }\n";
    o << "  }\n";
    o << "}\n";
    return o.str();
}

} // namespace rgd
