# tools/reex/ — local TurboQuant offline tools.
#
# Guards:
#   - Only built when REEX_TURBOQUANT=ON.
#   - Each tool that depends on the algorithm oracle additionally requires
#     LLAMA_BUILD_TESTS=ON (which is what brings in tests/oracle/).
#
# See tools/reex/README.md and docs/turboquant/04_naming_and_isolation.md.

if (NOT REEX_TURBOQUANT)
    return()
endif()

# reex_turboquant_gen_constants
#   Dumps codebook + scaling constants computed by the algorithm oracle into a
#   header-only `.inc` file consumed by the production CPU / CUDA kernels.
#   Re-run after any change to the oracle algorithm; the generated `.inc` is
#   committed alongside the source so production builds never re-compute them.
if (TARGET reex_turboquant_oracle)
    add_executable(reex_turboquant_gen_constants
        reex_turboquant_gen_constants.cpp
    )
    target_link_libraries(reex_turboquant_gen_constants PRIVATE
        reex_turboquant_oracle
    )
    target_compile_features(reex_turboquant_gen_constants PRIVATE cxx_std_17)
    if (NOT MSVC)
        target_compile_options(reex_turboquant_gen_constants PRIVATE -Wall -Wextra)
    endif()

    # reex_turboquant_gen_constants_v_polar
    #   V-β (P4 A2.5) Lloyd-Max codebook generator for GGML_TYPE_TQ_V_POLAR2 /
    #   TQ_V_POLAR4. Emits both bits=2 (4 centroids) and bits=4 (16 centroids)
    #   codebooks into a single `reex_turboquant_codebook_tq_v_polar.inc` file.
    add_executable(reex_turboquant_gen_constants_v_polar
        reex_turboquant_gen_constants_v_polar.cpp
    )
    target_link_libraries(reex_turboquant_gen_constants_v_polar PRIVATE
        reex_turboquant_oracle
    )
    target_compile_features(reex_turboquant_gen_constants_v_polar PRIVATE cxx_std_17)
    if (NOT MSVC)
        target_compile_options(reex_turboquant_gen_constants_v_polar PRIVATE -Wall -Wextra)
    endif()
else()
    message(STATUS
        "tools/reex: REEX_TURBOQUANT=ON but `reex_turboquant_oracle` target is "
        "missing (LLAMA_BUILD_TESTS=OFF or LLAMA_BUILD_COMMON=OFF?); skipping "
        "reex_turboquant_gen_constants and reex_turboquant_gen_constants_v_polar."
    )
endif()

# reex_turboquant_gen_wht_signs
#   Emits the deterministic ±1 sign tables used by the WHT rotation kernel.
#   Self-contained (only depends on libstdc++ / mt19937) so it builds
#   independently of the oracle target.
add_executable(reex_turboquant_gen_wht_signs
    reex_turboquant_gen_wht_signs.cpp
)
target_compile_features(reex_turboquant_gen_wht_signs PRIVATE cxx_std_17)
if (NOT MSVC)
    target_compile_options(reex_turboquant_gen_wht_signs PRIVATE -Wall -Wextra)
endif()
