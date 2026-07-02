# REEX HW weight converter — CPU-only offline tool + shared layout library.
#
#   reex-hw-layout  : CPU-only STATIC lib (type registry, §4.1 tiling/slots,
#                     reorder, K-quant HW repack, single-tensor convert). Shared
#                     with reex-gemm-datagen so there is a single implementation.
#   reex-hw-convert : CPU-only CLI (native/fp32/random -> HW weight_blocks).
#
# Needs GGML_USE_REEX_Q64 (for the reex block layouts + reference encoders in the
# ggml lib). Does NOT require CUDA.

# --- shared layout/convert library (CPU-only) ---
add_library(reex-hw-layout STATIC
    src/wquant.cpp
    src/wconvert.cpp
)
target_compile_definitions(reex-hw-layout PUBLIC GGML_USE_REEX_Q64)
target_include_directories(reex-hw-layout PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(reex-hw-layout PUBLIC ggml ggml-base)
# reex q64 reference encoders are resolved at runtime from the ggml shared lib.
target_link_options(reex-hw-layout PRIVATE "LINKER:--allow-shlib-undefined")

# --- converter CLI (CPU-only) ---
add_executable(reex-hw-convert src/main.cpp)
target_link_libraries(reex-hw-convert PRIVATE reex-hw-layout)
target_link_options(reex-hw-convert PRIVATE "LINKER:--allow-shlib-undefined")

install(TARGETS reex-hw-convert RUNTIME)
