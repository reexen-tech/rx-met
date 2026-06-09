// SPDX-License-Identifier: MIT
//
// TurboQuant algorithm reference implementation (oracle) — see header.
//
// Lifted from `src/reex/reex_turboquant_core.cpp` of the pre-v2 prototype
// branch, with all `CompressedKVStore`, `KVCaptureEngine`, ggml integration
// glue and the soft-max-bearing `reex_turboquant_attention_compute` removed.
// Test-only target; never linked into libllama / libggml.

#include "reex_turboquant_oracle_core.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <mutex>
#include <random>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace reex_turboquant_oracle {

namespace {

constexpr float   kEps              = 1e-10f;
constexpr int32_t kCodebookGridSize = 10000;
constexpr int32_t kCodebookMaxIter  = 200;
constexpr double  kCodebookTol      = 1e-12;

static int32_t vals_per_byte_for_bits(int32_t bits, int32_t & packed_bits) {
    if (bits == 1) { packed_bits = 1; return 8; }
    if (bits == 2) { packed_bits = 2; return 4; }
    if (bits <= 4) { packed_bits = 4; return 2; }
    packed_bits = 8;
    return 1;
}

static std::vector<uint8_t> pack_indices(const std::vector<uint8_t> & indices, int32_t bits, int32_t d, int32_t rows) {
    int32_t packed_bits = bits;
    const int32_t vals_per_byte = vals_per_byte_for_bits(bits, packed_bits);
    if (vals_per_byte == 1) {
        return indices;
    }
    const int32_t padded_d = ((d + vals_per_byte - 1) / vals_per_byte) * vals_per_byte;
    const int32_t packed_d = padded_d / vals_per_byte;
    std::vector<uint8_t> out(static_cast<size_t>(rows) * packed_d, 0);
    for (int32_t r = 0; r < rows; ++r) {
        for (int32_t j = 0; j < padded_d; ++j) {
            const uint8_t v = (j < d) ? indices[static_cast<size_t>(r) * d + j] : 0;
            const int32_t byte_idx = j / vals_per_byte;
            const int32_t lane     = j % vals_per_byte;
            out[static_cast<size_t>(r) * packed_d + byte_idx] |= static_cast<uint8_t>(v << (lane * packed_bits));
        }
    }
    return out;
}

static std::vector<uint8_t> unpack_indices(const std::vector<uint8_t> & packed, int32_t bits, int32_t d, int32_t rows) {
    int32_t packed_bits = bits;
    const int32_t vals_per_byte = vals_per_byte_for_bits(bits, packed_bits);
    if (vals_per_byte == 1) {
        return packed;
    }
    const int32_t padded_d = ((d + vals_per_byte - 1) / vals_per_byte) * vals_per_byte;
    const int32_t packed_d = padded_d / vals_per_byte;
    const uint8_t mask = static_cast<uint8_t>((1 << packed_bits) - 1);
    std::vector<uint8_t> out(static_cast<size_t>(rows) * d, 0);
    for (int32_t r = 0; r < rows; ++r) {
        for (int32_t j = 0; j < d; ++j) {
            const int32_t byte_idx = j / vals_per_byte;
            const int32_t lane     = j % vals_per_byte;
            const uint8_t pv       = packed[static_cast<size_t>(r) * packed_d + byte_idx];
            out[static_cast<size_t>(r) * d + j] = static_cast<uint8_t>((pv >> (lane * packed_bits)) & mask);
        }
    }
    return out;
}

static std::vector<float> make_gaussian_matrix(int32_t d, int32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> m(static_cast<size_t>(d) * d);
    for (float & v : m) {
        v = dist(rng);
    }
    return m;
}

static inline void matmul_row_vec(const float * row, const std::vector<float> & mat, int32_t d, float * out, bool transpose_mat) {
    // out = row @ mat^T when transpose_mat = true, else out = row @ mat.
    for (int32_t j = 0; j < d; ++j) {
        float acc = 0.0f;
        for (int32_t k = 0; k < d; ++k) {
            if (transpose_mat) {
                acc += row[k] * mat[static_cast<size_t>(j) * d + k];
            } else {
                acc += row[k] * mat[static_cast<size_t>(k) * d + j];
            }
        }
        out[j] = acc;
    }
}

static double beta_pdf(double x, int32_t d) {
    if (d <= 2) {
        throw std::runtime_error("TurboQuant codebook requires dim >= 3");
    }
    x = std::max(-1.0 + 1e-15, std::min(1.0 - 1e-15, x));
    const double log_const =
        std::lgamma(d / 2.0) -
        0.5 * std::log(M_PI) -
        std::lgamma((d - 1) / 2.0);
    const double exponent = (d - 3) / 2.0;
    return std::exp(log_const + exponent * std::log(1.0 - x * x));
}

static Codebook build_lloyd_max_codebook(int32_t dim, int32_t bits) {
    const int32_t n_clusters = 1 << bits;
    const double  lo = -1.0 + 1e-10;
    const double  hi =  1.0 - 1e-10;
    const double  dx = (hi - lo) / static_cast<double>(kCodebookGridSize - 1);

    std::vector<double> x_grid(kCodebookGridSize);
    std::vector<double> pdf_vals(kCodebookGridSize);
    std::vector<double> w_prefix  (kCodebookGridSize + 1, 0.0);
    std::vector<double> xw_prefix (kCodebookGridSize + 1, 0.0);
    std::vector<double> x2w_prefix(kCodebookGridSize + 1, 0.0);
    for (int32_t i = 0; i < kCodebookGridSize; ++i) {
        const double x = lo + dx * static_cast<double>(i);
        const double w = beta_pdf(x, dim) * dx;
        x_grid[i]          = x;
        pdf_vals[i]        = w;
        w_prefix  [i + 1]  = w_prefix  [i] + w;
        xw_prefix [i + 1]  = xw_prefix [i] + x * w;
        x2w_prefix[i + 1]  = x2w_prefix[i] + x * x * w;
    }
    const double total_mass = std::max(w_prefix.back(), static_cast<double>(kEps));
    for (double & v : w_prefix)   v /= total_mass;
    for (double & v : xw_prefix)  v /= total_mass;
    for (double & v : x2w_prefix) v /= total_mass;

    std::vector<double> centroids(n_clusters, 0.0);
    for (int32_t i = 0; i < n_clusters; ++i) {
        const double q_mid = (static_cast<double>(i) + 0.5) / static_cast<double>(n_clusters);
        auto it = std::lower_bound(w_prefix.begin(), w_prefix.end(), q_mid);
        int32_t idx = static_cast<int32_t>(std::distance(w_prefix.begin(), it));
        idx = std::max(1, std::min(idx, kCodebookGridSize)) - 1;
        centroids[i] = x_grid[idx];
    }

    auto prefix_at = [&](const std::vector<double> & pref, double boundary) -> double {
        if (boundary <= lo) return 0.0;
        if (boundary >= hi) return pref.back();
        const double pos = (boundary - lo) / dx;
        int32_t idx = static_cast<int32_t>(std::floor(pos));
        idx = std::max(0, std::min(idx, kCodebookGridSize - 2));
        const double frac = pos - idx;
        return pref[idx + 1] + frac * (pref[idx + 2] - pref[idx + 1]);
    };

    auto interval_stats = [&](double b0, double b1, double & mass, double & x_sum, double & x2_sum) {
        const double p0  = prefix_at(w_prefix,   b0);
        const double p1  = prefix_at(w_prefix,   b1);
        const double px0 = prefix_at(xw_prefix,  b0);
        const double px1 = prefix_at(xw_prefix,  b1);
        const double px20 = prefix_at(x2w_prefix, b0);
        const double px21 = prefix_at(x2w_prefix, b1);
        mass   = std::max(0.0, p1 - p0);
        x_sum  = px1  - px0;
        x2_sum = px21 - px20;
    };

    double prev_cost = std::numeric_limits<double>::infinity();
    std::vector<double> boundaries(n_clusters + 1, 0.0);
    for (int32_t iter = 0; iter < kCodebookMaxIter; ++iter) {
        boundaries.front() = -1.0;
        boundaries.back()  =  1.0;
        for (int32_t i = 0; i < n_clusters - 1; ++i) {
            boundaries[i + 1] = 0.5 * (centroids[i] + centroids[i + 1]);
        }

        std::vector<double> new_centroids(n_clusters, 0.0);
        double cost = 0.0;
        for (int32_t i = 0; i < n_clusters; ++i) {
            double mass = 0.0, x_sum = 0.0, x2_sum = 0.0;
            interval_stats(boundaries[i], boundaries[i + 1], mass, x_sum, x2_sum);
            if (mass < 1e-30) {
                new_centroids[i] = 0.5 * (boundaries[i] + boundaries[i + 1]);
            } else {
                new_centroids[i] = x_sum / mass;
            }
            const double c = new_centroids[i];
            cost += std::max(0.0, x2_sum - 2.0 * c * x_sum + c * c * mass);
        }

        centroids.swap(new_centroids);
        if (std::abs(prev_cost - cost) < kCodebookTol) {
            break;
        }
        prev_cost = cost;
    }

    boundaries.front() = -1.0;
    boundaries.back()  =  1.0;
    for (int32_t i = 0; i < n_clusters - 1; ++i) {
        boundaries[i + 1] = 0.5 * (centroids[i] + centroids[i + 1]);
    }

    Codebook cb;
    cb.dim  = dim;
    cb.bits = bits;
    cb.centroids.resize(n_clusters);
    cb.boundaries.resize(n_clusters + 1);
    for (int32_t i = 0; i < n_clusters; ++i) {
        cb.centroids[i] = static_cast<float>(centroids[i]);
    }
    for (int32_t i = 0; i <= n_clusters; ++i) {
        cb.boundaries[i] = static_cast<float>(boundaries[i]);
    }
    cb.decision_boundaries.assign(cb.boundaries.begin() + 1, cb.boundaries.end() - 1);
    return cb;
}

static std::vector<float> make_orthogonal_matrix(int32_t d, int32_t seed) {
    // Householder QR decomposition of a Gaussian matrix → uniform random
    // orthogonal Q (as in `numpy.linalg.qr`).  The seed/layer offsetting
    // policy mirrors the Python reference at /home/zcx/CLionProjects/turboquant.
    std::vector<float> r = make_gaussian_matrix(d, seed);
    std::vector<float> q(static_cast<size_t>(d) * d, 0.0f);
    for (int32_t i = 0; i < d; ++i) {
        q[static_cast<size_t>(i) * d + i] = 1.0f;
    }

    std::vector<float> v(d, 0.0f);
    for (int32_t k = 0; k < d; ++k) {
        float norm_x = 0.0f;
        for (int32_t i = k; i < d; ++i) {
            const float val = r[static_cast<size_t>(i) * d + k];
            norm_x += val * val;
        }
        norm_x = std::sqrt(std::max(norm_x, kEps));
        if (norm_x <= kEps) continue;

        std::fill(v.begin(), v.end(), 0.0f);
        for (int32_t i = k; i < d; ++i) {
            v[i] = r[static_cast<size_t>(i) * d + k];
        }
        v[k] += (v[k] >= 0.0f ? norm_x : -norm_x);

        float v_norm2 = 0.0f;
        for (int32_t i = k; i < d; ++i) v_norm2 += v[i] * v[i];
        if (v_norm2 <= kEps) continue;
        const float beta = 2.0f / v_norm2;

        for (int32_t j = k; j < d; ++j) {
            float vr = 0.0f;
            for (int32_t i = k; i < d; ++i) vr += v[i] * r[static_cast<size_t>(i) * d + j];
            vr *= beta;
            for (int32_t i = k; i < d; ++i) r[static_cast<size_t>(i) * d + j] -= vr * v[i];
        }

        for (int32_t i = 0; i < d; ++i) {
            float qv = 0.0f;
            for (int32_t j = k; j < d; ++j) qv += q[static_cast<size_t>(i) * d + j] * v[j];
            qv *= beta;
            for (int32_t j = k; j < d; ++j) q[static_cast<size_t>(i) * d + j] -= qv * v[j];
        }
    }

    for (int32_t j = 0; j < d; ++j) {
        const float diag = r[static_cast<size_t>(j) * d + j];
        const float sign = diag >= 0.0f ? 1.0f : -1.0f;
        for (int32_t i = 0; i < d; ++i) q[static_cast<size_t>(i) * d + j] *= sign;
    }
    return q;
}

static inline int32_t lower_bound_idx(const std::vector<float> & v, float x) {
    return static_cast<int32_t>(std::lower_bound(v.begin(), v.end(), x) - v.begin());
}

static inline int32_t packed_d_for(int32_t d, int32_t bits) {
    int32_t packed_bits = bits;
    const int32_t vals_per_byte = vals_per_byte_for_bits(bits, packed_bits);
    if (vals_per_byte == 1) return d;
    return ((d + vals_per_byte - 1) / vals_per_byte);
}

}  // namespace

Codebook reex_turboquant_oracle_codebook_get(int32_t dim, int32_t bits) {
    if (dim <= 0 || bits <= 0 || bits > 8) {
        throw std::runtime_error("invalid codebook parameters");
    }
    static std::mutex mu;
    static std::map<std::pair<int32_t, int32_t>, Codebook> cache;
    const auto key = std::make_pair(dim, bits);
    std::lock_guard<std::mutex> lock(mu);
    auto it = cache.find(key);
    if (it == cache.end()) {
        it = cache.emplace(key, build_lloyd_max_codebook(dim, bits)).first;
    }
    return it->second;
}

std::vector<float> reex_turboquant_oracle_rotation_get(int32_t dim, int32_t seed, int32_t layer_idx) {
    static std::mutex mu;
    static std::map<std::tuple<int32_t, int32_t, int32_t>, std::vector<float>> cache;
    const auto key = std::make_tuple(dim, seed, layer_idx);
    std::lock_guard<std::mutex> lock(mu);
    auto it = cache.find(key);
    if (it == cache.end()) {
        it = cache.emplace(key, make_orthogonal_matrix(dim, seed + layer_idx * 7)).first;
    }
    return it->second;
}

std::vector<float> reex_turboquant_oracle_qjl_get(int32_t dim, int32_t seed, int32_t layer_idx) {
    static std::mutex mu;
    static std::map<std::tuple<int32_t, int32_t, int32_t>, std::vector<float>> cache;
    const auto key = std::make_tuple(dim, seed, layer_idx);
    std::lock_guard<std::mutex> lock(mu);
    auto it = cache.find(key);
    if (it == cache.end()) {
        it = cache.emplace(key, make_gaussian_matrix(dim, seed + layer_idx * 7 + 1000)).first;
    }
    return it->second;
}

TurboQuantMSE::TurboQuantMSE(int32_t dim, int32_t bits, int32_t seed)
    : dim_(dim),
      bits_(bits),
      codebook_(reex_turboquant_oracle_codebook_get(dim, bits)),
      pi_(reex_turboquant_oracle_rotation_get(dim, seed, 0)) {
    if (dim_ <= 0 || bits_ <= 0) {
        throw std::runtime_error("invalid TurboQuantMSE config");
    }
}

TurboQuantMSE::TurboQuantMSE(int32_t dim, int32_t bits, const Codebook & codebook, const std::vector<float> & rotation)
    : dim_(dim), bits_(bits), codebook_(codebook), pi_(rotation) {
    if (dim_ <= 0 || bits_ <= 0) {
        throw std::runtime_error("invalid TurboQuantMSE config");
    }
    if (static_cast<int32_t>(codebook_.centroids.size()) != (1 << bits_)) {
        throw std::runtime_error("invalid codebook centroid count");
    }
    if (static_cast<int32_t>(pi_.size()) != dim_ * dim_) {
        throw std::runtime_error("invalid rotation size");
    }
    if (codebook_.decision_boundaries.empty() && codebook_.boundaries.size() >= 2) {
        codebook_.decision_boundaries.assign(codebook_.boundaries.begin() + 1, codebook_.boundaries.end() - 1);
    }
}

MSEQuantized TurboQuantMSE::quantize(const float * x, int32_t batch, int32_t heads, int32_t tokens) const {
    const int32_t rows = batch * heads * tokens;
    std::vector<uint8_t> raw_indices(static_cast<size_t>(rows) * dim_, 0);
    std::vector<float>   norms      (rows, 0.0f);
    std::vector<float>   x_unit     (dim_, 0.0f);
    std::vector<float>   y          (dim_, 0.0f);

    for (int32_t r = 0; r < rows; ++r) {
        const float * xr = x + static_cast<size_t>(r) * dim_;
        float n = 0.0f;
        for (int32_t j = 0; j < dim_; ++j) n += xr[j] * xr[j];
        n = std::sqrt(std::max(n, kEps));
        norms[r] = n;
        for (int32_t j = 0; j < dim_; ++j) x_unit[j] = xr[j] / n;
        matmul_row_vec(x_unit.data(), pi_, dim_, y.data(), true);
        for (int32_t j = 0; j < dim_; ++j) {
            int32_t idx = lower_bound_idx(codebook_.decision_boundaries, y[j]);
            idx = std::max(0, std::min(idx, static_cast<int32_t>(codebook_.centroids.size()) - 1));
            raw_indices[static_cast<size_t>(r) * dim_ + j] = static_cast<uint8_t>(idx);
        }
    }

    MSEQuantized out;
    out.indices = pack_indices(raw_indices, bits_, dim_, rows);
    out.norms   = std::move(norms);
    out.bits    = bits_;
    out.dim     = dim_;
    out.batch   = batch;
    out.heads   = heads;
    out.tokens  = tokens;
    return out;
}

std::vector<float> TurboQuantMSE::dequantize(const MSEQuantized & q) const {
    const int32_t rows = q.batch * q.heads * q.tokens;
    std::vector<uint8_t> indices = unpack_indices(q.indices, q.bits, dim_, rows);
    std::vector<float> y    (dim_, 0.0f);
    std::vector<float> x_hat(dim_, 0.0f);
    std::vector<float> out  (static_cast<size_t>(rows) * dim_, 0.0f);

    for (int32_t r = 0; r < rows; ++r) {
        for (int32_t j = 0; j < dim_; ++j) {
            y[j] = codebook_.centroids[indices[static_cast<size_t>(r) * dim_ + j]];
        }
        matmul_row_vec(y.data(), pi_, dim_, x_hat.data(), false);
        const float n = q.norms[r];
        for (int32_t j = 0; j < dim_; ++j) {
            out[static_cast<size_t>(r) * dim_ + j] = x_hat[j] * n;
        }
    }
    return out;
}

TurboQuantPolar::TurboQuantPolar(int32_t dim, int32_t bits, int32_t seed)
    : dim_(dim),
      bits_(bits),
      codebook_(reex_turboquant_oracle_codebook_get(dim, bits)),
      pi_(reex_turboquant_oracle_rotation_get(dim, seed, 0)) {
    if (dim_ <= 0 || bits_ <= 0) {
        throw std::runtime_error("invalid TurboQuantPolar config");
    }
}

TurboQuantPolar::TurboQuantPolar(int32_t dim, int32_t bits, const Codebook & codebook, const std::vector<float> & rotation)
    : dim_(dim), bits_(bits), codebook_(codebook), pi_(rotation) {
    if (dim_ <= 0 || bits_ <= 0) {
        throw std::runtime_error("invalid TurboQuantPolar config");
    }
    if (static_cast<int32_t>(codebook_.centroids.size()) != (1 << bits_)) {
        throw std::runtime_error("invalid codebook centroid count");
    }
    if (static_cast<int32_t>(pi_.size()) != dim_ * dim_) {
        throw std::runtime_error("invalid rotation size");
    }
    if (codebook_.decision_boundaries.empty() && codebook_.boundaries.size() >= 2) {
        codebook_.decision_boundaries.assign(codebook_.boundaries.begin() + 1, codebook_.boundaries.end() - 1);
    }
}

MSEQuantized TurboQuantPolar::quantize(const float * x, int32_t batch, int32_t heads, int32_t tokens) const {
    const int32_t rows = batch * heads * tokens;
    std::vector<uint8_t> raw_indices(static_cast<size_t>(rows) * dim_, 0);
    std::vector<float>   norms      (rows, 0.0f);
    std::vector<float>   x_unit     (dim_, 0.0f);
    std::vector<float>   y          (dim_, 0.0f);

    for (int32_t r = 0; r < rows; ++r) {
        const float * xr = x + static_cast<size_t>(r) * dim_;
        float n = 0.0f;
        for (int32_t j = 0; j < dim_; ++j) n += xr[j] * xr[j];
        n = std::sqrt(std::max(n, kEps));
        const float inv_n = 1.0f / n;
        for (int32_t j = 0; j < dim_; ++j) x_unit[j] = xr[j] * inv_n;
        matmul_row_vec(x_unit.data(), pi_, dim_, y.data(), true);
        // Quantize each rotated component to nearest centroid and accumulate
        // Σ centroid² for the norm-correction trick.  Same convention as
        // TurboQuantMSE::quantize but we also need recon_sq.
        float recon_sq = 0.0f;
        for (int32_t j = 0; j < dim_; ++j) {
            int32_t idx = lower_bound_idx(codebook_.decision_boundaries, y[j]);
            idx = std::max(0, std::min(idx, static_cast<int32_t>(codebook_.centroids.size()) - 1));
            raw_indices[static_cast<size_t>(r) * dim_ + j] = static_cast<uint8_t>(idx);
            const float c = codebook_.centroids[idx];
            recon_sq += c * c;
        }
        const float recon_norm = std::sqrt(std::max(recon_sq, 1e-20f));
        const float corrected  = (recon_norm > 1e-10f) ? (n / recon_norm) : n;
        norms[r] = corrected;
    }

    MSEQuantized out;
    out.indices = pack_indices(raw_indices, bits_, dim_, rows);
    out.norms   = std::move(norms);
    out.bits    = bits_;
    out.dim     = dim_;
    out.batch   = batch;
    out.heads   = heads;
    out.tokens  = tokens;
    return out;
}

std::vector<float> TurboQuantPolar::dequantize(const MSEQuantized & q) const {
    const int32_t rows = q.batch * q.heads * q.tokens;
    std::vector<uint8_t> indices = unpack_indices(q.indices, q.bits, dim_, rows);
    std::vector<float> y    (dim_, 0.0f);
    std::vector<float> x_hat(dim_, 0.0f);
    std::vector<float> out  (static_cast<size_t>(rows) * dim_, 0.0f);

    for (int32_t r = 0; r < rows; ++r) {
        for (int32_t j = 0; j < dim_; ++j) {
            y[j] = codebook_.centroids[indices[static_cast<size_t>(r) * dim_ + j]];
        }
        matmul_row_vec(y.data(), pi_, dim_, x_hat.data(), false);
        const float corr = q.norms[r];  // already includes ‖x‖/‖recon‖
        for (int32_t j = 0; j < dim_; ++j) {
            out[static_cast<size_t>(r) * dim_ + j] = x_hat[j] * corr;
        }
    }
    return out;
}

TurboQuantProd::TurboQuantProd(int32_t dim, int32_t bits, int32_t seed)
    : dim_ (dim),
      bits_(bits),
      mse_ (dim, std::max(bits - 1, 1), seed),
      s_   (reex_turboquant_oracle_qjl_get(dim, seed, 0)),
      qjl_scale_(std::sqrt(static_cast<float>(M_PI) / 2.0f) / static_cast<float>(dim)) {
    if (bits_ < 2) {
        throw std::runtime_error("TurboQuantProd requires bits >= 2");
    }
}

TurboQuantProd::TurboQuantProd(int32_t dim, int32_t bits, const QuantizerResources & resources)
    : dim_ (dim),
      bits_(bits),
      mse_ (dim, std::max(bits - 1, 1), resources.mse_codebook, resources.rotation),
      s_   (resources.qjl),
      qjl_scale_(std::sqrt(static_cast<float>(M_PI) / 2.0f) / static_cast<float>(dim)) {
    if (bits_ < 2) {
        throw std::runtime_error("TurboQuantProd requires bits >= 2");
    }
    if (static_cast<int32_t>(s_.size()) != dim_ * dim_) {
        throw std::runtime_error("invalid qjl matrix size");
    }
}

ProdQuantized TurboQuantProd::quantize(const float * x, int32_t batch, int32_t heads, int32_t tokens) const {
    MSEQuantized       mse_q = mse_.quantize(x, batch, heads, tokens);
    std::vector<float> x_hat = mse_.dequantize(mse_q);
    const int32_t rows = batch * heads * tokens;

    std::vector<float>   residual      (static_cast<size_t>(rows) * dim_, 0.0f);
    std::vector<float>   residual_norms(rows, 0.0f);
    std::vector<uint8_t> qjl_sign_bits (static_cast<size_t>(rows) * ((dim_ + 7) / 8), 0);
    std::vector<float>   projected     (dim_, 0.0f);

    for (int32_t r = 0; r < rows; ++r) {
        const float * xr = x + static_cast<size_t>(r) * dim_;
        float * rr = residual.data() + static_cast<size_t>(r) * dim_;
        float rn = 0.0f;
        for (int32_t j = 0; j < dim_; ++j) {
            rr[j] = xr[j] - x_hat[static_cast<size_t>(r) * dim_ + j];
            rn += rr[j] * rr[j];
        }
        residual_norms[r] = std::sqrt(std::max(rn, kEps));
        matmul_row_vec(rr, s_, dim_, projected.data(), true);
        for (int32_t j = 0; j < dim_; ++j) {
            const int32_t b    = j / 8;
            const int32_t lane = j % 8;
            if (projected[j] > 0.0f) {
                qjl_sign_bits[static_cast<size_t>(r) * ((dim_ + 7) / 8) + b] |= static_cast<uint8_t>(1u << lane);
            }
        }
    }

    ProdQuantized out;
    out.mse_indices    = std::move(mse_q.indices);
    out.qjl_signs      = std::move(qjl_sign_bits);
    out.residual_norms = std::move(residual_norms);
    out.norms          = std::move(mse_q.norms);
    out.mse_bits       = mse_q.bits;
    out.dim            = dim_;
    out.batch          = batch;
    out.heads          = heads;
    out.tokens         = tokens;
    return out;
}

std::vector<float> TurboQuantProd::dequantize(const ProdQuantizedView & v) const {
    const int32_t rows         = v.batch * v.heads * v.tokens;
    const int32_t mse_packed_d = packed_d_for(v.dim, std::max(v.mse_bits, 1));
    const int32_t qjl_packed_d = (v.dim + 7) / 8;

    MSEQuantized mse_q;
    mse_q.bits   = v.mse_bits;
    mse_q.dim    = v.dim;
    mse_q.batch  = v.batch;
    mse_q.heads  = v.heads;
    mse_q.tokens = v.tokens;
    mse_q.indices.resize(static_cast<size_t>(rows) * mse_packed_d);
    mse_q.norms  .resize(rows);
    if (v.row_stride == v.tokens) {
        std::copy_n(v.mse_indices, mse_q.indices.size(), mse_q.indices.begin());
        std::copy_n(v.norms,       mse_q.norms.size(),   mse_q.norms.begin());
    } else {
        for (int32_t b = 0; b < v.batch; ++b) {
            for (int32_t h = 0; h < v.heads; ++h) {
                const size_t src_row = static_cast<size_t>(h) * v.row_stride;
                const size_t dst_row = (static_cast<size_t>(b) * v.heads + h) * v.tokens;
                std::copy_n(
                    v.mse_indices + src_row * mse_packed_d,
                    static_cast<size_t>(v.tokens) * mse_packed_d,
                    mse_q.indices.begin() + dst_row * mse_packed_d);
                std::copy_n(
                    v.norms + src_row,
                    static_cast<size_t>(v.tokens),
                    mse_q.norms.begin() + dst_row);
            }
        }
    }
    std::vector<float> out = mse_.dequantize(mse_q);

    std::vector<float> signs   (dim_, -1.0f);
    std::vector<float> qjl_part(dim_,  0.0f);
    for (int32_t b = 0; b < v.batch; ++b) {
        for (int32_t h = 0; h < v.heads; ++h) {
            for (int32_t t = 0; t < v.tokens; ++t) {
                const size_t src_row = static_cast<size_t>(h) * v.row_stride + t;
                const size_t dst_row = (static_cast<size_t>(b) * v.heads + h) * v.tokens + t;
                for (int32_t j = 0; j < dim_; ++j) {
                    const int32_t bb   = j / 8;
                    const int32_t lane = j % 8;
                    const uint8_t pck  = v.qjl_signs[src_row * qjl_packed_d + bb];
                    signs[j] = ((pck >> lane) & 0x1) ? 1.0f : -1.0f;
                }
                matmul_row_vec(signs.data(), s_, dim_, qjl_part.data(), false);
                const float scale = qjl_scale_ * v.residual_norms[src_row];
                for (int32_t j = 0; j < dim_; ++j) {
                    out[dst_row * dim_ + j] += qjl_part[j] * scale;
                }
            }
        }
    }

    return out;
}

std::vector<float> TurboQuantProd::attention_score(
    const float * query,
    int32_t batch,
    int32_t q_heads,
    int32_t q_tokens,
    const ProdQuantizedView & key_v,
    int32_t kv_heads
) const {
    if (kv_heads <= 0 || q_heads % kv_heads != 0) {
        throw std::runtime_error("invalid GQA ratio");
    }
    const int32_t gqa       = q_heads / kv_heads;
    const int32_t kv_tokens = key_v.tokens;
    std::vector<float> keys = dequantize(key_v); // (batch, kv_heads, kv_tokens, dim)
    std::vector<float> out(static_cast<size_t>(batch) * q_heads * q_tokens * kv_tokens, 0.0f);
    for (int32_t b = 0; b < batch; ++b) {
        for (int32_t qh = 0; qh < q_heads; ++qh) {
            const int32_t kh = qh / gqa;
            for (int32_t qt = 0; qt < q_tokens; ++qt) {
                const float * qv = query + (((static_cast<size_t>(b) * q_heads + qh) * q_tokens + qt) * dim_);
                for (int32_t kt = 0; kt < kv_tokens; ++kt) {
                    const float * kv = keys.data() + (((static_cast<size_t>(b) * kv_heads + kh) * kv_tokens + kt) * dim_);
                    float dot = 0.0f;
                    for (int32_t j = 0; j < dim_; ++j) dot += qv[j] * kv[j];
                    out[(((static_cast<size_t>(b) * q_heads + qh) * q_tokens + qt) * kv_tokens + kt)] = dot;
                }
            }
        }
    }
    return out;
}

ValueQuantized reex_turboquant_oracle_value_quantize(
    const float * v,
    int32_t batch,
    int32_t heads,
    int32_t tokens,
    int32_t dim,
    int32_t bits,
    int32_t group_size
) {
    if (group_size <= 0 || dim % group_size != 0) {
        throw std::runtime_error("invalid group_size");
    }
    const int32_t rows     = batch * heads * tokens;
    const int32_t n_groups = dim / group_size;
    const int32_t levels   = (1 << bits) - 1;
    std::vector<uint8_t> q_raw (static_cast<size_t>(rows) * dim, 0);
    std::vector<float>   scales(static_cast<size_t>(rows) * n_groups, 0.0f);
    std::vector<float>   zeros (static_cast<size_t>(rows) * n_groups, 0.0f);

    for (int32_t r = 0; r < rows; ++r) {
        const float * vr = v + static_cast<size_t>(r) * dim;
        for (int32_t g = 0; g < n_groups; ++g) {
            const int32_t base = g * group_size;
            float mn = vr[base], mx = vr[base];
            for (int32_t j = 1; j < group_size; ++j) {
                mn = std::min(mn, vr[base + j]);
                mx = std::max(mx, vr[base + j]);
            }
            float scale = (mx - mn) / static_cast<float>(levels);
            if (scale < kEps) scale = kEps;
            scales[static_cast<size_t>(r) * n_groups + g] = scale;
            zeros [static_cast<size_t>(r) * n_groups + g] = mn;
            for (int32_t j = 0; j < group_size; ++j) {
                const float qf = (vr[base + j] - mn) / scale;
                int32_t qi = static_cast<int32_t>(std::lround(qf));
                qi = std::max(0, std::min(levels, qi));
                q_raw[static_cast<size_t>(r) * dim + base + j] = static_cast<uint8_t>(qi);
            }
        }
    }

    ValueQuantized out;
    out.data       = pack_indices(q_raw, bits, dim, rows);
    out.scales     = std::move(scales);
    out.zeros      = std::move(zeros);
    out.bits       = bits;
    out.group_size = group_size;
    out.dim        = dim;
    out.batch      = batch;
    out.heads      = heads;
    out.tokens     = tokens;
    return out;
}

std::vector<float> reex_turboquant_oracle_value_dequantize(const ValueQuantizedView & vq) {
    const int32_t rows       = vq.batch * vq.heads * vq.tokens;
    const int32_t n_groups   = vq.dim / vq.group_size;
    const int32_t v_packed_d = packed_d_for(vq.dim, vq.bits);

    std::vector<uint8_t> packed(static_cast<size_t>(rows) * v_packed_d);
    if (vq.row_stride == vq.tokens) {
        std::copy_n(vq.data, packed.size(), packed.begin());
    } else {
        for (int32_t bb = 0; bb < vq.batch; ++bb) {
            for (int32_t h = 0; h < vq.heads; ++h) {
                const size_t src_row = static_cast<size_t>(h) * vq.row_stride;
                const size_t dst_row = (static_cast<size_t>(bb) * vq.heads + h) * vq.tokens;
                std::copy_n(
                    vq.data + src_row * v_packed_d,
                    static_cast<size_t>(vq.tokens) * v_packed_d,
                    packed.begin() + dst_row * v_packed_d);
            }
        }
    }
    std::vector<uint8_t> uq = unpack_indices(packed, vq.bits, vq.dim, rows);

    std::vector<float> out(static_cast<size_t>(rows) * vq.dim, 0.0f);
    for (int32_t bb = 0; bb < vq.batch; ++bb) {
        for (int32_t h = 0; h < vq.heads; ++h) {
            for (int32_t t = 0; t < vq.tokens; ++t) {
                const size_t src_row = static_cast<size_t>(h) * vq.row_stride + t;
                const size_t dst_row = (static_cast<size_t>(bb) * vq.heads + h) * vq.tokens + t;
                for (int32_t g = 0; g < n_groups; ++g) {
                    const float scale = vq.scales[src_row * n_groups + g];
                    const float zero  = vq.zeros [src_row * n_groups + g];
                    const int32_t base = g * vq.group_size;
                    for (int32_t j = 0; j < vq.group_size; ++j) {
                        out[dst_row * vq.dim + base + j] =
                            static_cast<float>(uq[dst_row * vq.dim + base + j]) * scale + zero;
                    }
                }
            }
        }
    }
    return out;
}

ProdQuantized reex_turboquant_oracle_compact(const ProdQuantizedView & v) {
    ProdQuantized out;
    out.mse_bits = v.mse_bits;
    out.dim      = v.dim;
    out.batch    = v.batch;
    out.heads    = v.heads;
    out.tokens   = v.tokens;
    if (v.tokens <= 0 || v.heads <= 0 || v.batch <= 0) return out;

    const int32_t rows         = v.batch * v.heads * v.tokens;
    const int32_t mse_packed_d = packed_d_for(v.dim, std::max(v.mse_bits, 1));
    const int32_t qjl_packed_d = (v.dim + 7) / 8;

    out.mse_indices   .resize(static_cast<size_t>(rows) * mse_packed_d);
    out.qjl_signs     .resize(static_cast<size_t>(rows) * qjl_packed_d);
    out.residual_norms.resize(rows);
    out.norms         .resize(rows);

    if (v.row_stride == v.tokens) {
        std::copy_n(v.mse_indices,    out.mse_indices   .size(), out.mse_indices   .begin());
        std::copy_n(v.qjl_signs,      out.qjl_signs     .size(), out.qjl_signs     .begin());
        std::copy_n(v.residual_norms, out.residual_norms.size(), out.residual_norms.begin());
        std::copy_n(v.norms,          out.norms         .size(), out.norms         .begin());
        return out;
    }
    for (int32_t b = 0; b < v.batch; ++b) {
        for (int32_t h = 0; h < v.heads; ++h) {
            const size_t src_row = static_cast<size_t>(h) * v.row_stride;
            const size_t dst_row = (static_cast<size_t>(b) * v.heads + h) * v.tokens;
            std::copy_n(
                v.mse_indices + src_row * mse_packed_d,
                static_cast<size_t>(v.tokens) * mse_packed_d,
                out.mse_indices.begin() + dst_row * mse_packed_d);
            std::copy_n(
                v.qjl_signs + src_row * qjl_packed_d,
                static_cast<size_t>(v.tokens) * qjl_packed_d,
                out.qjl_signs.begin() + dst_row * qjl_packed_d);
            std::copy_n(
                v.residual_norms + src_row,
                static_cast<size_t>(v.tokens),
                out.residual_norms.begin() + dst_row);
            std::copy_n(
                v.norms + src_row,
                static_cast<size_t>(v.tokens),
                out.norms.begin() + dst_row);
        }
    }
    return out;
}

ValueQuantized reex_turboquant_oracle_compact(const ValueQuantizedView & v) {
    ValueQuantized out;
    out.bits       = v.bits;
    out.group_size = v.group_size;
    out.dim        = v.dim;
    out.batch      = v.batch;
    out.heads      = v.heads;
    out.tokens     = v.tokens;
    if (v.tokens <= 0 || v.heads <= 0 || v.batch <= 0) return out;

    const int32_t rows       = v.batch * v.heads * v.tokens;
    const int32_t v_packed_d = packed_d_for(v.dim, v.bits);
    const int32_t groups     = v.dim / v.group_size;

    out.data  .resize(static_cast<size_t>(rows) * v_packed_d);
    out.scales.resize(static_cast<size_t>(rows) * groups);
    out.zeros .resize(static_cast<size_t>(rows) * groups);

    if (v.row_stride == v.tokens) {
        std::copy_n(v.data,   out.data  .size(), out.data  .begin());
        std::copy_n(v.scales, out.scales.size(), out.scales.begin());
        std::copy_n(v.zeros,  out.zeros .size(), out.zeros .begin());
        return out;
    }
    for (int32_t b = 0; b < v.batch; ++b) {
        for (int32_t h = 0; h < v.heads; ++h) {
            const size_t src_row = static_cast<size_t>(h) * v.row_stride;
            const size_t dst_row = (static_cast<size_t>(b) * v.heads + h) * v.tokens;
            std::copy_n(
                v.data + src_row * v_packed_d,
                static_cast<size_t>(v.tokens) * v_packed_d,
                out.data.begin() + dst_row * v_packed_d);
            std::copy_n(
                v.scales + src_row * groups,
                static_cast<size_t>(v.tokens) * groups,
                out.scales.begin() + dst_row * groups);
            std::copy_n(
                v.zeros + src_row * groups,
                static_cast<size_t>(v.tokens) * groups,
                out.zeros.begin() + dst_row * groups);
        }
    }
    return out;
}

}  // namespace reex_turboquant_oracle
