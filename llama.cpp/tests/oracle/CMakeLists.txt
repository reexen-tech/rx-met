# tests/oracle/ — TurboQuant algorithm reference (oracle), test-only.
#
# Compiled into a static library `reex_turboquant_oracle` and consumed by
# tests under `tests/reex/`.  NOT linked into libllama / libggml.
#
# Guards:
#   - Only built when REEX_TURBOQUANT=ON.
#   - Only built when this directory is included via the parent
#     `tests/CMakeLists.txt`, i.e. when LLAMA_BUILD_TESTS=ON.
#
# See ../oracle/README.md and docs/turboquant/01_设计与实现计划.md.

if (NOT REEX_TURBOQUANT)
    return()
endif()

add_library(reex_turboquant_oracle STATIC
    reex_turboquant_oracle_core.cpp
)

target_include_directories(reex_turboquant_oracle PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_features(reex_turboquant_oracle PUBLIC cxx_std_17)

set_target_properties(reex_turboquant_oracle PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET     hidden
    VISIBILITY_INLINES_HIDDEN ON
)

if (MSVC)
    target_compile_options(reex_turboquant_oracle PRIVATE /utf-8)
else()
    target_compile_options(reex_turboquant_oracle PRIVATE -Wall -Wextra)
endif()
