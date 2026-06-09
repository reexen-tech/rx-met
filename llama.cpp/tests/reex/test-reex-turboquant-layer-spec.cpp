// Tests for `reex_turboquant::resolve_layer_spec`.
//
// We construct minimal `llama_hparams` instances directly (no model load, no
// context init) and check that the per-layer override is driven entirely by
// the user-supplied global `type_k` / `type_v` (i.e. `-ctk` / `-ctv`), guarded
// by the WHT block-divisibility rule (head_dim_{k,v} multiple of 128).
//
// Pure-CPU, fast, no I/O.

#include "reex_turboquant_layer_spec.h"

#include "llama-hparams.h"

#include <cassert>
#include <cstdio>
#include <cstring>

namespace {

namespace tq = reex_turboquant;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

llama_hparams make_hparams(uint32_t n_layer,
                           int32_t  n_layer_kv_from_start = -1,
                           uint32_t n_embd_head_k = 128,
                           uint32_t n_embd_head_v = 128) {
    llama_hparams h{};
    // The struct contains std::arrays that we only need partially populated;
    // value-initializing the whole thing is enough because resolve_layer_spec
    // only reads has_kv() (which uses n_layer / n_layer_kv_from_start) and
    // n_embd_head_{k,v} (the WHT block-divisibility gate).
    h.n_layer = n_layer;
    h.n_layer_kv_from_start = n_layer_kv_from_start;
    h.n_embd_head_k = n_embd_head_k;
    h.n_embd_head_v = n_embd_head_v;
    return h;
}

void check_no_override(const tq::LayerSpec & s, const char * tag) {
    if (s.use_tq_k || s.use_tq_v ||
        s.type_k_override != GGML_TYPE_COUNT ||
        s.type_v_override != GGML_TYPE_COUNT ||
        s.boundary_v_active ||
        s.boundary_v_type != GGML_TYPE_COUNT) {
        fprintf(stderr, "[FAIL] %s: expected no-override, got "
                "use_tq_k=%d use_tq_v=%d k=%d v=%d boundary_v=%d btype=%d\n",
                tag,
                (int) s.use_tq_k, (int) s.use_tq_v,
                (int) s.type_k_override, (int) s.type_v_override,
                (int) s.boundary_v_active, (int) s.boundary_v_type);
        assert(false);
    }
}

void check_full_tq(const tq::LayerSpec & s, ggml_type expect_v_type, int expect_v_bits, const char * tag) {
    if (!s.use_tq_k || s.type_k_override != GGML_TYPE_TQ_K3) {
        fprintf(stderr, "[FAIL] %s: K override should be TQ_K3 (got use=%d type=%d)\n",
                tag, (int) s.use_tq_k, (int) s.type_k_override);
        assert(false);
    }
    if (!s.use_tq_v || s.type_v_override != expect_v_type || s.v_bits != expect_v_bits) {
        fprintf(stderr, "[FAIL] %s: V override mismatch (got type=%d v_bits=%d)\n",
                tag, (int) s.type_v_override, (int) s.v_bits);
        assert(false);
    }
}

// ---------------------------------------------------------------------------
// Off — neither -ctk nor -ctv is a TQ_* type
// ---------------------------------------------------------------------------

void test_off_no_tq_types() {
    auto h = make_hparams(/*n_layer=*/4);
    for (uint32_t il = 0; il < h.n_layer; ++il) {
        const auto s = tq::resolve_layer_spec(h, GGML_TYPE_F16, GGML_TYPE_F16, il);
        check_no_override(s, "off_f16");
    }
}

// ---------------------------------------------------------------------------
// Full TQ K3 + V2 — every KV-bearing layer overrides
// ---------------------------------------------------------------------------

void test_full_k3_v2() {
    auto h = make_hparams(/*n_layer=*/8);
    for (uint32_t il = 0; il < h.n_layer; ++il) {
        const auto s = tq::resolve_layer_spec(h, GGML_TYPE_TQ_K3, GGML_TYPE_TQ_V2, il);
        check_full_tq(s, GGML_TYPE_TQ_V2, /*expect_v_bits=*/2, "k3_v2");
    }
}

// ---------------------------------------------------------------------------
// Full TQ K3 + V4 (高质量档)
// ---------------------------------------------------------------------------

void test_full_k3_v4() {
    auto h = make_hparams(/*n_layer=*/4);
    for (uint32_t il = 0; il < h.n_layer; ++il) {
        const auto s = tq::resolve_layer_spec(h, GGML_TYPE_TQ_K3, GGML_TYPE_TQ_V4, il);
        check_full_tq(s, GGML_TYPE_TQ_V4, /*expect_v_bits=*/4, "k3_v4");
    }
}

// ---------------------------------------------------------------------------
// K-only TQ — V stays on user's non-TQ type
// ---------------------------------------------------------------------------

void test_k3_only() {
    auto h = make_hparams(/*n_layer=*/4);
    for (uint32_t il = 0; il < h.n_layer; ++il) {
        const auto s = tq::resolve_layer_spec(h, GGML_TYPE_TQ_K3, GGML_TYPE_F16, il);
        if (!s.use_tq_k || s.type_k_override != GGML_TYPE_TQ_K3) {
            fprintf(stderr, "[FAIL] k3_only il=%u: K should be TQ_K3\n", il);
            assert(false);
        }
        if (s.use_tq_v || s.type_v_override != GGML_TYPE_COUNT) {
            fprintf(stderr, "[FAIL] k3_only il=%u: V should NOT be overridden (was %d)\n",
                    il, (int) s.type_v_override);
            assert(false);
        }
    }
}

// ---------------------------------------------------------------------------
// V-only TQ — K stays on user's non-TQ type
// ---------------------------------------------------------------------------

void test_v2_only() {
    auto h = make_hparams(/*n_layer=*/4);
    for (uint32_t il = 0; il < h.n_layer; ++il) {
        const auto s = tq::resolve_layer_spec(h, GGML_TYPE_F16, GGML_TYPE_TQ_V2, il);
        if (s.use_tq_k || s.type_k_override != GGML_TYPE_COUNT) {
            fprintf(stderr, "[FAIL] v2_only il=%u: K should NOT be overridden (was %d)\n",
                    il, (int) s.type_k_override);
            assert(false);
        }
        if (!s.use_tq_v || s.type_v_override != GGML_TYPE_TQ_V2 || s.v_bits != 2) {
            fprintf(stderr, "[FAIL] v2_only il=%u: V should be TQ_V2\n", il);
            assert(false);
        }
    }
}

// ---------------------------------------------------------------------------
// Non-KV layer skip — even with TQ types selected, layers without KV stay clean
// ---------------------------------------------------------------------------

void test_skip_non_kv() {
    auto h = make_hparams(/*n_layer=*/8, /*n_layer_kv_from_start=*/4);
    for (uint32_t il = 0; il < h.n_layer; ++il) {
        const auto s = tq::resolve_layer_spec(h, GGML_TYPE_TQ_K3, GGML_TYPE_TQ_V2, il);
        if (il < 4) {
            check_full_tq(s, GGML_TYPE_TQ_V2, 2, "k3v2_kv");
        } else {
            check_no_override(s, "k3v2_no_kv");
        }
    }
}

// ---------------------------------------------------------------------------
// WHT block divisibility — head_dim must be a multiple of 128 to enable TQ
// ---------------------------------------------------------------------------

void test_head_dim_guard_skips_non_multiple_of_128() {
    // head_dim=64 should fall back to upstream cache types regardless of
    // whether the user asked for TQ via -ctk/-ctv.
    auto h_64 = make_hparams(/*n_layer=*/4, /*n_layer_kv_from_start=*/-1,
                             /*n_embd_head_k=*/64, /*n_embd_head_v=*/64);
    for (uint32_t il = 0; il < h_64.n_layer; ++il) {
        const auto s = tq::resolve_layer_spec(h_64, GGML_TYPE_TQ_K3, GGML_TYPE_TQ_V2, il);
        check_no_override(s, "head64");
    }

    // Also reject head_dim=192 (multiple of 64 but not 128).
    auto h_192 = make_hparams(/*n_layer=*/4, -1, 192, 192);
    for (uint32_t il = 0; il < h_192.n_layer; ++il) {
        const auto s = tq::resolve_layer_spec(h_192, GGML_TYPE_TQ_K3, GGML_TYPE_TQ_V2, il);
        check_no_override(s, "head192");
    }

    // head_dim=256 (Qwen3.5 9B) should pass the guard.
    auto h_256 = make_hparams(/*n_layer=*/4, -1, 256, 256);
    for (uint32_t il = 0; il < h_256.n_layer; ++il) {
        const auto s = tq::resolve_layer_spec(h_256, GGML_TYPE_TQ_K3, GGML_TYPE_TQ_V2, il);
        check_full_tq(s, GGML_TYPE_TQ_V2, /*expect_v_bits=*/2, "head256");
    }
}

// ---------------------------------------------------------------------------
// D13 boundary-V (pure function — exercises compute_boundary_v_override
// without env mocking; env-reading wrapper helpers are tested implicitly via
// env-off bitwise-identity preserved by the test_off_* cases above).
// ---------------------------------------------------------------------------

void test_boundary_v_off_path() {
    // Boundary off -> always no override regardless of il / n_layer / type_v.
    for (uint32_t il = 0; il < 64; ++il) {
        const auto t = tq::compute_boundary_v_override(
                il, /*n_layer=*/48, GGML_TYPE_TQ_V_POLAR2,
                /*boundary_enabled=*/false, /*n_boundary=*/2);
        if (t != GGML_TYPE_COUNT) {
            fprintf(stderr, "[FAIL] boundary_off il=%u: got type=%d, expected COUNT\n",
                    il, (int) t);
            assert(false);
        }
    }
}

void test_boundary_v_non_polar_skipped() {
    // boundary on but type_v not in TQ_V_POLAR family -> no override.
    const ggml_type non_polar[] = {
        GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_BF16, GGML_TYPE_Q8_0,
        GGML_TYPE_TQ_V2, GGML_TYPE_TQ_V4,
    };
    for (ggml_type tv : non_polar) {
        for (uint32_t il : {0u, 1u, 23u, 47u}) {
            const auto t = tq::compute_boundary_v_override(
                    il, /*n_layer=*/48, tv, /*boundary_enabled=*/true, /*n_boundary=*/2);
            if (t != GGML_TYPE_COUNT) {
                fprintf(stderr, "[FAIL] non_polar_skipped tv=%d il=%u: got type=%d\n",
                        (int) tv, il, (int) t);
                assert(false);
            }
        }
    }
}

void test_boundary_v_n_layer_too_small() {
    // n_layer < 8 -> always no override (mode-7 guard).
    for (uint32_t n_layer : {0u, 1u, 4u, 7u}) {
        for (uint32_t il = 0; il < n_layer; ++il) {
            const auto t = tq::compute_boundary_v_override(
                    il, n_layer, GGML_TYPE_TQ_V_POLAR2,
                    /*boundary_enabled=*/true, /*n_boundary=*/2);
            if (t != GGML_TYPE_COUNT) {
                fprintf(stderr, "[FAIL] small_n_layer n=%u il=%u: got type=%d\n",
                        n_layer, il, (int) t);
                assert(false);
            }
        }
    }
}

void test_boundary_v_zero_n_boundary() {
    for (uint32_t il = 0; il < 16; ++il) {
        const auto t = tq::compute_boundary_v_override(
                il, /*n_layer=*/16, GGML_TYPE_TQ_V_POLAR2,
                /*boundary_enabled=*/true, /*n_boundary=*/0);
        if (t != GGML_TYPE_COUNT) {
            fprintf(stderr, "[FAIL] zero_boundary il=%u: got type=%d\n", il, (int) t);
            assert(false);
        }
    }
}

void test_boundary_v_too_many_boundary_layers() {
    // 2 * n_boundary >= n_layer -> degenerates to all-boundary; refuse.
    for (uint32_t il = 0; il < 8; ++il) {
        const auto t = tq::compute_boundary_v_override(
                il, /*n_layer=*/8, GGML_TYPE_TQ_V_POLAR2,
                /*boundary_enabled=*/true, /*n_boundary=*/4);  // 2*4 == 8 -> refuse
        if (t != GGML_TYPE_COUNT) {
            fprintf(stderr, "[FAIL] too_many il=%u: got type=%d\n", il, (int) t);
            assert(false);
        }
    }
    // 2 * n_boundary > n_layer -> refuse.
    for (uint32_t il = 0; il < 10; ++il) {
        const auto t = tq::compute_boundary_v_override(
                il, /*n_layer=*/10, GGML_TYPE_TQ_V_POLAR2,
                /*boundary_enabled=*/true, /*n_boundary=*/6);
        if (t != GGML_TYPE_COUNT) {
            fprintf(stderr, "[FAIL] too_many il=%u: got type=%d\n", il, (int) t);
            assert(false);
        }
    }
}

void test_boundary_v_mode7_n_layer_48() {
    // 30B-Inst Q4_0 layout: 48 attention layers, mode-7 boundary=2.
    // Expected: il in {0,1,46,47} -> Q8_0; il in [2,45] -> COUNT.
    for (ggml_type tv : {GGML_TYPE_TQ_V_POLAR2, GGML_TYPE_TQ_V_POLAR4}) {
        for (uint32_t il = 0; il < 48; ++il) {
            const auto t = tq::compute_boundary_v_override(
                    il, /*n_layer=*/48, tv,
                    /*boundary_enabled=*/true, /*n_boundary=*/2);
            const bool expect_boundary = (il < 2 || il >= 48 - 2);
            const ggml_type expect = expect_boundary ? GGML_TYPE_Q8_0 : GGML_TYPE_COUNT;
            if (t != expect) {
                fprintf(stderr, "[FAIL] mode7_48 tv=%d il=%u: got type=%d expected=%d\n",
                        (int) tv, il, (int) t, (int) expect);
                assert(false);
            }
        }
    }
}

void test_boundary_v_n_layer_8_min() {
    // n_layer = 8, n_boundary = 2: il in {0,1,6,7} -> Q8_0.
    // 2*2 == 4 < 8 ✓; n_layer >= 8 ✓.
    for (uint32_t il = 0; il < 8; ++il) {
        const auto t = tq::compute_boundary_v_override(
                il, /*n_layer=*/8, GGML_TYPE_TQ_V_POLAR2,
                /*boundary_enabled=*/true, /*n_boundary=*/2);
        const bool expect_boundary = (il < 2 || il >= 8 - 2);
        const ggml_type expect = expect_boundary ? GGML_TYPE_Q8_0 : GGML_TYPE_COUNT;
        if (t != expect) {
            fprintf(stderr, "[FAIL] n8 il=%u: got=%d expected=%d\n",
                    il, (int) t, (int) expect);
            assert(false);
        }
    }
}

void test_boundary_v_custom_n_boundary() {
    // n_layer = 26 (Gemma-3-1B), n_boundary = 4 (custom env override).
    // 2*4 == 8 < 26 ✓.  Expect {0..3, 22..25} -> Q8_0.
    for (uint32_t il = 0; il < 26; ++il) {
        const auto t = tq::compute_boundary_v_override(
                il, /*n_layer=*/26, GGML_TYPE_TQ_V_POLAR2,
                /*boundary_enabled=*/true, /*n_boundary=*/4);
        const bool expect_boundary = (il < 4 || il >= 26 - 4);
        const ggml_type expect = expect_boundary ? GGML_TYPE_Q8_0 : GGML_TYPE_COUNT;
        if (t != expect) {
            fprintf(stderr, "[FAIL] gemma26_b4 il=%u: got=%d expected=%d\n",
                    il, (int) t, (int) expect);
            assert(false);
        }
    }
}

// ---------------------------------------------------------------------------
// Determinism — identical inputs, identical outputs
// ---------------------------------------------------------------------------

void test_determinism() {
    auto h = make_hparams(/*n_layer=*/16);
    for (uint32_t il = 0; il < h.n_layer; ++il) {
        const auto s1 = tq::resolve_layer_spec(h, GGML_TYPE_TQ_K3, GGML_TYPE_TQ_V4, il);
        const auto s2 = tq::resolve_layer_spec(h, GGML_TYPE_TQ_K3, GGML_TYPE_TQ_V4, il);
        if (memcmp(&s1, &s2, sizeof(s1)) != 0) {
            fprintf(stderr, "[FAIL] determinism il=%u\n", il);
            assert(false);
        }
    }
}

}  // namespace

int main() {
    fprintf(stdout, "[layer-spec] off (no TQ types) ... ");
    test_off_no_tq_types();
    fprintf(stdout, "OK\n");

    fprintf(stdout, "[layer-spec] full K3 + V2 ... ");
    test_full_k3_v2();
    fprintf(stdout, "OK\n");

    fprintf(stdout, "[layer-spec] full K3 + V4 ... ");
    test_full_k3_v4();
    fprintf(stdout, "OK\n");

    fprintf(stdout, "[layer-spec] K3 only (V stays on user type) ... ");
    test_k3_only();
    fprintf(stdout, "OK\n");

    fprintf(stdout, "[layer-spec] V2 only (K stays on user type) ... ");
    test_v2_only();
    fprintf(stdout, "OK\n");

    fprintf(stdout, "[layer-spec] skip non-KV layers ... ");
    test_skip_non_kv();
    fprintf(stdout, "OK\n");

    fprintf(stdout, "[layer-spec] head_dim WHT guard ... ");
    test_head_dim_guard_skips_non_multiple_of_128();
    fprintf(stdout, "OK\n");

    fprintf(stdout, "[layer-spec] D13 boundary-V off path ... ");
    test_boundary_v_off_path();
    fprintf(stdout, "OK\n");

    fprintf(stdout, "[layer-spec] D13 boundary-V non-V_POLAR skipped ... ");
    test_boundary_v_non_polar_skipped();
    fprintf(stdout, "OK\n");

    fprintf(stdout, "[layer-spec] D13 boundary-V n_layer<8 guard ... ");
    test_boundary_v_n_layer_too_small();
    fprintf(stdout, "OK\n");

    fprintf(stdout, "[layer-spec] D13 boundary-V n_boundary=0 ... ");
    test_boundary_v_zero_n_boundary();
    fprintf(stdout, "OK\n");

    fprintf(stdout, "[layer-spec] D13 boundary-V too-many boundary refuse ... ");
    test_boundary_v_too_many_boundary_layers();
    fprintf(stdout, "OK\n");

    fprintf(stdout, "[layer-spec] D13 boundary-V mode-7 n_layer=48 ... ");
    test_boundary_v_mode7_n_layer_48();
    fprintf(stdout, "OK\n");

    fprintf(stdout, "[layer-spec] D13 boundary-V n_layer=8 min ... ");
    test_boundary_v_n_layer_8_min();
    fprintf(stdout, "OK\n");

    fprintf(stdout, "[layer-spec] D13 boundary-V custom n_boundary ... ");
    test_boundary_v_custom_n_boundary();
    fprintf(stdout, "OK\n");

    fprintf(stdout, "[layer-spec] determinism ... ");
    test_determinism();
    fprintf(stdout, "OK\n");

    fprintf(stdout, "[layer-spec] PASS\n");
    return 0;
}
