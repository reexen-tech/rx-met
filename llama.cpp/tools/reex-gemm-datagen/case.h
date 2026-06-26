// REEX GEMM datagen — core types, tiling spec and §4.1 tile-address helpers.
#pragma once

#include "ggml.h"

#include <cstdint>
#include <string>

namespace rgd {

#if defined(__CUDACC__)
    #define RGD_HD __host__ __device__
#else
    #define RGD_HD
#endif


// -------------------------------------------------------------------------
// Orthogonal config axes.
// -------------------------------------------------------------------------
enum class Family  { Kquant, Legacy, IntBlock };
enum class ActDType { F32, F16, BF16, E5M2, E4M3 };
enum class OutDType { F16, BF16, I16, I8, I6, I4 };

inline const char * family_name(Family f) {
    switch (f) {
        case Family::Kquant:   return "Kquant";
        case Family::Legacy:   return "Legacy";
        case Family::IntBlock: return "IntBlock";
    }
    return "?";
}
inline const char * actdtype_name(ActDType d) {
    switch (d) {
        case ActDType::F32:  return "F32";
        case ActDType::F16:  return "F16";
        case ActDType::BF16: return "BF16";
        case ActDType::E5M2: return "E5M2";
        case ActDType::E4M3: return "E4M3";
    }
    return "?";
}
// Native byte size of one element stored in the act-source dtype.
inline int act_src_elem_bytes(ActDType d) {
    switch (d) {
        case ActDType::F32:  return 4;
        case ActDType::F16:  return 2;
        case ActDType::BF16: return 2;
        case ActDType::E5M2: return 1;
        case ActDType::E4M3: return 1;
    }
    return 4;
}
inline const char * outdtype_name(OutDType d) {
    switch (d) {
        case OutDType::F16:  return "F16";
        case OutDType::BF16: return "BF16";
        case OutDType::I16:  return "I16";
        case OutDType::I8:   return "I8";
        case OutDType::I6:   return "I6";
        case OutDType::I4:   return "I4";
    }
    return "?";
}

// -------------------------------------------------------------------------
// Output dtype table: container bytes + (for int outputs) symmetric qmax.
// I6/I4 are carried in a signed int8 container (sign-extended). F16/BF16 are
// is_float. The full set is emitted per case (compute once, convert to all).
// -------------------------------------------------------------------------
struct OutSpec { OutDType dt; const char * name; bool is_float; int bytes; int qmax; };

inline const OutSpec * out_specs(int & n) {
    static const OutSpec s[] = {
        { OutDType::F16,  "F16",  true,  2, 0 },
        { OutDType::BF16, "BF16", true,  2, 0 },
        { OutDType::I16,  "I16",  false, 2, 32767 },
        { OutDType::I8,   "I8",   false, 1, 127 },
        { OutDType::I6,   "I6",   false, 1, 31 },
        { OutDType::I4,   "I4",   false, 1, 7 },
    };
    n = (int) (sizeof(s) / sizeof(s[0]));
    return s;
}

// -------------------------------------------------------------------------
// Activation block = ONE contiguous quant group (agroup K-elems), CPU-style:
//   { ggml_fp16_t d; int8_t qs[agroup]; }   (1 scale + agroup int8, no 32-split)
// agroup == Kt (Legacy 64, Kquant 256). Stored as a raw byte buffer because the
// element count varies by family; use act_block_bytes() / act_group_slot().
// -------------------------------------------------------------------------

// -------------------------------------------------------------------------
// §4.1 tiling: tile = [Mt rows x Nt cols(N) x Kt cols(K)]. weight/act tiled at
// quant-block granularity (Kt == quant block K-elems), result at element.
//   wgroup : weight quant-block K-elems   agroup : activation scale group K-elems
// -------------------------------------------------------------------------
struct TilingSpec {
    int Mt, Nt, Kt;
    int wgroup, agroup;
};

inline TilingSpec tiling_for(Family f) {
    switch (f) {
        case Family::Kquant:   return TilingSpec{ 1, 64, 256, 64, 256 }; // weight[256,64], act[1,256]
        case Family::Legacy:   return TilingSpec{ 4, 64,  64, 64,  64 }; // weight[64,64],  act[4,64]
        case Family::IntBlock: return TilingSpec{ 16,16,  16, 16,  16 }; // weight[16,16],  act[16,16]
    }
    return TilingSpec{ 1, 64, 256, 64, 256 };
}

// -------------------------------------------------------------------------
// One experiment case (orthogonal config). Shapes fixed for the milestone.
// -------------------------------------------------------------------------
struct GemmCase {
    int64_t  M = 8192, N = 2048, K = 2048;
    int      wtype_id = 0;
    ActDType act_in   = ActDType::F16;
    int      A_bits   = 8;
    OutDType out      = OutDType::F16;
    int      psum_bits = 0;         // <=0 disabled
    uint64_t seed     = 1234;
    bool     dump_intermediate = false;
    int64_t  dump_m = 64, dump_n = 64;
    std::string name;               // auto-generated

    // IntBlock pure-integer path only (ignored by Kquant/Legacy):
    int      w_bits     = 8;        // weight bit-width (8/6/5/4/3/2)
    bool     a_unsigned = false;    // activations are UINT (else signed two's-comp)
    bool     w_unsigned = false;    // weights are UINT
};

// -------------------------------------------------------------------------
// §4.1 tile address formulas.
//   - weight / activation : QUANT-BLOCK granularity (structs are indivisible)
//   - result              : ELEMENT granularity
// All return a flat slot index into the corresponding tiled buffer.
// -------------------------------------------------------------------------

// Weight blocks: matrix W[N,K] -> blocks[(n, sb)] where sb = K-superblock index
// (each Kt elements). Tile [Kt,Nt] = ONE K-block × Nt N-blocks.
//   inter-tile (kt,nt) row-major; intra-tile column(=N)-major.
RGD_HD inline int64_t weight_block_slot(int64_t n, int64_t sb, int64_t N,
                                        const TilingSpec & ts) {
    const int64_t Ntiles = N / ts.Nt;
    const int64_t nt = n / ts.Nt, c = n % ts.Nt;   // c = column within tile (N)
    const int64_t kt = sb;                          // one K-block per tile
    const int64_t t  = kt * Ntiles + nt;            // inter-tile row-major
    return t * ts.Nt + c;                           // intra-tile column-major
}

// Activation quant container element bytes: int16 when A_bits>8 (A16), else int8.
RGD_HD inline int act_elem_bytes(int A_bits) { return A_bits > 8 ? 2 : 1; }

// Bytes of one activation block = { fp16 scale ; intX qs[agroup] }, X per A_bits.
RGD_HD inline int act_block_bytes(const TilingSpec & ts, int A_bits) {
    return 2 + ts.agroup * act_elem_bytes(A_bits);
}

// Activation group blocks: matrix A[M,K] -> blocks[(m, kg)] where kg = quant
// group index along K (each agroup elems; agroup == Kt). Tile [Mt,Kt] holds Mt
// rows × (Kt/agroup) groups. inter-tile (mt,kt) row-major; intra-tile row-major.
RGD_HD inline int64_t act_group_slot(int64_t m, int64_t kg, int64_t M, int64_t K,
                                     const TilingSpec & ts) {
    (void) M;
    const int     gpt    = ts.Kt / ts.agroup;       // groups along K per tile (=1)
    const int64_t Ktiles = K / ts.Kt;
    const int64_t mt = m / ts.Mt, r = m % ts.Mt;
    const int64_t kt = kg / gpt, kc = kg % gpt;
    const int64_t t  = mt * Ktiles + kt;            // inter-tile row-major
    const int64_t o  = r * gpt + kc;                // intra-tile row-major
    return t * ((int64_t) ts.Mt * gpt) + o;
}

// IntBlock element-granularity tile slots (raw INT values, no scale). Both store
// a 2D [16x16] block: act = [Mt M-rows x Kt K-cols], weight = [Nt N-rows x Kt
// K-cols], block row-major; inter-block (act) (mt,kt) row-major / (weight)
// (kt,nt) row-major. (Identical layout to the FP16-source de-tile formulas.)
RGD_HD inline int64_t act_elem_slot(int64_t m, int64_t k, int64_t K,
                                    const TilingSpec & ts) {
    const int64_t Ktiles = K / ts.Kt;
    const int64_t mt = m / ts.Mt, r = m % ts.Mt;
    const int64_t kt = k / ts.Kt, c = k % ts.Kt;
    return ((mt * Ktiles + kt) * ts.Mt + r) * ts.Kt + c;
}
// Weight intra-block ordering is COLUMN-major (contiguous along N): block viewed
// as [16 N-rows x 16 K-cols], element (n_local,k_local) stored at k_local*Nt +
// n_local. inter-block (kt,nt) row-major. (Activation stays row-major; this
// mirrors the FP groups, whose weight blocks are also N-contiguous within a tile.)
RGD_HD inline int64_t weight_elem_slot(int64_t n, int64_t k, int64_t N,
                                       const TilingSpec & ts) {
    const int64_t Ntiles = N / ts.Nt;
    const int64_t nt = n / ts.Nt, r = n % ts.Nt;   // r = N-row within block
    const int64_t kt = k / ts.Kt, c = k % ts.Kt;   // c = K-col within block
    return ((kt * Ntiles + nt) * ts.Kt + c) * ts.Nt + r;  // intra-block column(=N)-major
}

// Result C[M,N], tile [Mt,Nt] (element granularity):
//   inter-block (mt,nt) column-major, intra-block row-major.
RGD_HD inline int64_t result_tiled_index(int64_t m, int64_t n, int64_t M,
                                  const TilingSpec & ts) {
    const int64_t Mtiles = M / ts.Mt;
    const int64_t mt = m / ts.Mt, r = m % ts.Mt;
    const int64_t nt = n / ts.Nt, c = n % ts.Nt;
    const int64_t t  = nt * Mtiles + mt;           // column-major over tiles
    const int64_t o  = r * ts.Nt + c;
    return t * ((int64_t) ts.Mt * ts.Nt) + o;
}

} // namespace rgd
