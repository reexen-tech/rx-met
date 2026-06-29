# REEX GEMM datagen — standalone GPU quant-sweep data collection tool.
# Built only when REEX_GEMM_DATAGEN=ON (needs GGML_CUDA + GGML_USE_REEX_Q64).

set(REEX_GEMM_DATAGEN_CUDA_ARCH "120" CACHE STRING
    "CUDA architecture(s) for reex-gemm-datagen (e.g. 89 for Ada, 120 for Blackwell/RTX5090)")

# CUDA language is enabled deep inside ggml/src/ggml-cuda/ (sibling scope), so it
# is not visible here — enable it for this tool's scope.
enable_language(CUDA)

set(TARGET reex-gemm-datagen)

# Sources live under src/ (headers stay alongside as an internal tool — no public include/).
add_executable(${TARGET}
    src/main.cpp
    src/datagen.cpp
    src/wquant.cpp
    src/reference.cpp
    src/dumper.cpp
    src/actquant.cu
    src/gemm.cu
    src/intgemm.cu
)

set_target_properties(${TARGET} PROPERTIES
    CUDA_ARCHITECTURES "${REEX_GEMM_DATAGEN_CUDA_ARCH}")

target_compile_definitions(${TARGET} PRIVATE GGML_USE_REEX_Q64)

target_link_libraries(${TARGET} PRIVATE ggml ggml-base)

# The reex q64 hw-dump hooks are resolved at runtime from the ggml shared lib.
target_link_options(${TARGET} PRIVATE "LINKER:--allow-shlib-undefined")

install(TARGETS ${TARGET} RUNTIME)
