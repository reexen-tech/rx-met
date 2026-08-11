llama_add_compile_flags()

function(llama_build source)
    set(TEST_SOURCES ${source} ${ARGN})

    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_TARGET ${LLAMA_TEST_NAME})
    else()
        get_filename_component(TEST_TARGET ${source} NAME_WE)
    endif()

    add_executable(${TEST_TARGET} ${TEST_SOURCES})
    target_link_libraries(${TEST_TARGET} PRIVATE llama llama-common)
    if (LLAMA_TESTS_INSTALL)
        install(TARGETS ${TEST_TARGET} RUNTIME)
    endif()
endfunction()

function(llama_test target)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (NOT DEFINED LLAMA_TEST_LABEL)
        set(LLAMA_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LLAMA_TEST_WORKING_DIRECTORY)
        set(LLAMA_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_NAME ${LLAMA_TEST_NAME})
    else()
        set(TEST_NAME ${target})
    endif()

    set(TEST_TARGET ${target})

    add_test(
        NAME ${TEST_NAME}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND $<TARGET_FILE:${TEST_TARGET}>
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_NAME} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

function(llama_test_cmd target)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (NOT DEFINED LLAMA_TEST_LABEL)
        set(LLAMA_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LLAMA_TEST_WORKING_DIRECTORY)
        set(LLAMA_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_NAME ${LLAMA_TEST_NAME})
    else()
        set(TEST_NAME ${target})
    endif()

    add_test(
        NAME ${TEST_NAME}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND ${target}
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_NAME} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

# Builds and runs a test source file.
# Optional args:
# - NAME: name of the executable & test target (defaults to the source file name without extension)
# - LABEL: label for the test (defaults to main)
# - ARGS: arguments to pass to the test executable
# - WORKING_DIRECTORY
function(llama_build_and_test source)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(TEST_SOURCES ${source} ${LLAMA_TEST_UNPARSED_ARGUMENTS} get-model.cpp)

    if (NOT DEFINED LLAMA_TEST_LABEL)
        set(LLAMA_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LLAMA_TEST_WORKING_DIRECTORY)
        set(LLAMA_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_TARGET ${LLAMA_TEST_NAME})
    else()
        get_filename_component(TEST_TARGET ${source} NAME_WE)
    endif()

    add_executable(${TEST_TARGET} ${TEST_SOURCES})
    if (LLAMA_TESTS_INSTALL)
        install(TARGETS ${TEST_TARGET} RUNTIME)
    endif()
    target_link_libraries(${TEST_TARGET} PRIVATE llama-common)

    add_test(
        NAME ${TEST_TARGET}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND $<TARGET_FILE:${TEST_TARGET}>
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_TARGET} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

# build test-tokenizer-0 target once and add many tests
llama_build(test-tokenizer-0.cpp)

llama_test(test-tokenizer-0 NAME test-tokenizer-0-bert-bge          ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-bert-bge.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-command-r         ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-command-r.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-deepseek-coder    ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-deepseek-coder.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-deepseek-llm      ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-deepseek-llm.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-falcon            ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-falcon.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-gemma-4           ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gemma-4.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-gpt-2             ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gpt-2.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-llama-bpe         ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-bpe.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-llama-spm         ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-spm.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-mpt               ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-mpt.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-phi-3             ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-phi-3.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-qwen2             ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-qwen2.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-refact            ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-refact.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-starcoder         ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-starcoder.gguf)

if (NOT WIN32)
    llama_test_cmd(
        ${CMAKE_CURRENT_SOURCE_DIR}/test-tokenizers-repo.sh
        NAME test-tokenizers-ggml-vocabs
        WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        ARGS https://huggingface.co/ggml-org/vocabs ${PROJECT_SOURCE_DIR}/models/ggml-vocabs
    )
endif()

if (LLAMA_LLGUIDANCE)
    llama_build_and_test(test-grammar-llguidance.cpp ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-bpe.gguf)
endif ()

if (NOT WIN32 OR NOT BUILD_SHARED_LIBS)
    # these tests are disabled on Windows because they use internal functions not exported with LLAMA_API (when building with shared libraries)
    llama_build_and_test(test-sampling.cpp)
    llama_build_and_test(test-reasoning-budget.cpp)
    llama_build_and_test(test-grammar-parser.cpp)
    llama_build_and_test(test-grammar-integration.cpp)
    llama_build_and_test(test-llama-grammar.cpp)
    llama_build_and_test(test-chat.cpp WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
    target_include_directories(test-chat PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
    target_link_libraries(test-chat PRIVATE server-context)
    # TODO: disabled on loongarch64 because the ggml-ci node lacks Python 3.8
    if (NOT ${CMAKE_SYSTEM_PROCESSOR} MATCHES "loongarch64")
        llama_build_and_test(test-json-schema-to-grammar.cpp   WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
        target_include_directories(test-json-schema-to-grammar PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
    endif()

    if (NOT GGML_BACKEND_DL)
        llama_build(test-quantize-stats.cpp)
    endif()

    llama_build(test-gbnf-validator.cpp)

    # build test-tokenizer-1-bpe target once and add many tests
    llama_build(test-tokenizer-1-bpe.cpp)

    # TODO: disabled due to slowness
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-aquila    ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-aquila.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-falcon    ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-falcon.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-gpt-2     ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gpt-2.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-gpt-neox  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gpt-neox.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-llama-bpe ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-bpe.gguf --ignore-merges)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-mpt       ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-mpt.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-refact    ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-refact.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-starcoder ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-starcoder.gguf)

    # build test-tokenizer-1-spm target once and add many tests
    llama_build(test-tokenizer-1-spm.cpp)

    llama_test(test-tokenizer-1-spm  NAME test-tokenizer-1-llama-spm ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-spm.gguf)
    #llama_test(test-tokenizer-1-spm  NAME test-tokenizer-1-baichuan  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-baichuan.gguf)

    # llama_build_and_test(test-double-float.cpp) # SLOW

    llama_build_and_test(test-llama-archs.cpp)
endif()

llama_build_and_test(test-chat-peg-parser.cpp peg-parser/simple-tokenize.cpp)
llama_build_and_test(test-jinja.cpp)
llama_test(test-jinja NAME test-jinja-py ARGS -py LABEL python)
llama_build_and_test(test-chat-auto-parser.cpp WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
llama_build_and_test(test-chat-template.cpp)
llama_build_and_test(test-json-partial.cpp)
llama_build_and_test(test-log.cpp)
llama_build_and_test(
    test-peg-parser.cpp
    peg-parser/simple-tokenize.cpp
    peg-parser/test-basic.cpp
    peg-parser/test-gbnf-generation.cpp
    peg-parser/test-json-parser.cpp
    peg-parser/test-json-serialization.cpp
    peg-parser/test-python-dict-parser.cpp
    peg-parser/test-unicode.cpp
    peg-parser/tests.h
)
llama_build_and_test(test-regex-partial.cpp)

if (NOT ${CMAKE_SYSTEM_PROCESSOR} MATCHES "s390x")
    set(MODEL_NAME "tinyllamas/stories15M-q4_0.gguf")
    set(MODEL_HASH "SHA256=66967fbece6dbe97886593fdbb73589584927e29119ec31f08090732d1861739")
else()
    set(MODEL_NAME "tinyllamas/stories15M-be.Q4_0.gguf")
    set(MODEL_HASH "SHA256=9aec857937849d976f30397e97eb1cabb53eb9dcb1ce4611ba8247fb5f44c65d")
endif()
set(MODEL_DEST "${CMAKE_BINARY_DIR}/${MODEL_NAME}")

add_test(NAME test-download-model COMMAND ${CMAKE_COMMAND}
    -DDEST=${MODEL_DEST}
    -DNAME=${MODEL_NAME}
    -DHASH=${MODEL_HASH}
    -P ${CMAKE_SOURCE_DIR}/cmake/download-models.cmake
)
set_tests_properties(test-download-model PROPERTIES FIXTURES_SETUP test-download-model)

llama_build_and_test(test-thread-safety.cpp ARGS -m "${MODEL_DEST}" -ngl 99 -p "The meaning of life is" -n 128 -c 256 -ub 32 -np 4 -t 2)
set_tests_properties(test-thread-safety PROPERTIES FIXTURES_REQUIRED test-download-model)

llama_build_and_test(test-arg-parser.cpp)

if (NOT LLAMA_SANITIZE_ADDRESS AND NOT GGML_SCHED_NO_REALLOC)
  # TODO: repair known memory leaks
  llama_build_and_test(test-opt.cpp)
endif()
llama_build_and_test(test-gguf.cpp)
llama_build_and_test(test-backend-ops.cpp)

if (GGML_USE_REEX)
  add_executable(test-reex-lut test-reex-lut.cpp)
  target_link_libraries(test-reex-lut PRIVATE ggml)
  target_compile_definitions(test-reex-lut PRIVATE
    GGML_USE_REEX
    GGML_LUT_NUM_SEGMENTS_REEX=${GGML_LUT_NUM_SEGMENTS_REEX}
  )
  target_include_directories(test-reex-lut PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src
    ${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src/ggml-cpu
    ${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src/ggml-cuda/reex
  )
  if (TARGET ggml-cuda)
    target_link_libraries(test-reex-lut PRIVATE ggml-cuda)
    target_compile_definitions(test-reex-lut PRIVATE GGML_CUDA)
  endif()

  if (GGML_REEX_SIN_COS_NO_LUT)
    add_executable(test-reex-sin-cos-no-lut test-reex-sin-cos-no-lut.cpp)
    target_link_libraries(test-reex-sin-cos-no-lut PRIVATE ggml)
    target_compile_definitions(test-reex-sin-cos-no-lut PRIVATE
      GGML_USE_REEX
      GGML_REEX_SIN_COS_NO_LUT
    )
    target_include_directories(test-reex-sin-cos-no-lut PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src/ggml-cpu
    )
    if (TARGET ggml-cuda)
      target_link_libraries(test-reex-sin-cos-no-lut PRIVATE ggml-cuda)
      target_compile_definitions(test-reex-sin-cos-no-lut PRIVATE GGML_CUDA)
    endif()
    add_test(NAME test-reex-sin-cos-no-lut COMMAND test-reex-sin-cos-no-lut)
  endif()

  if (GGML_REEX_EXP_NO_LUT OR GGML_REEX_LOG_NO_LUT OR
      GGML_REEX_RECIPROCAL_NO_LUT OR GGML_REEX_RSQRT_NO_LUT OR
      GGML_REEX_SQRT_NO_LUT)
    add_executable(test-reex-no-lut-switches test-reex-no-lut-switches.cpp)
    target_link_libraries(test-reex-no-lut-switches PRIVATE ggml)
    target_compile_definitions(test-reex-no-lut-switches PRIVATE GGML_USE_REEX)
    foreach(REEX_NO_LUT_OPTION IN ITEMS
            GGML_REEX_EXP_NO_LUT
            GGML_REEX_LOG_NO_LUT
            GGML_REEX_RECIPROCAL_NO_LUT
            GGML_REEX_RSQRT_NO_LUT
            GGML_REEX_SQRT_NO_LUT)
      if (${REEX_NO_LUT_OPTION})
        target_compile_definitions(test-reex-no-lut-switches PRIVATE ${REEX_NO_LUT_OPTION})
      endif()
    endforeach()
    target_include_directories(test-reex-no-lut-switches PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src/ggml-cpu
      ${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src/ggml-cuda/reex
    )
    if (TARGET ggml-cuda)
      target_link_libraries(test-reex-no-lut-switches PRIVATE ggml-cuda)
      target_compile_definitions(test-reex-no-lut-switches PRIVATE GGML_CUDA)
    endif()
    add_test(NAME test-reex-no-lut-switches COMMAND test-reex-no-lut-switches)
  endif()
endif()

llama_build_and_test(test-reex-custom-switch.cpp)
llama_build_and_test(test-reex-fattn-policy.cpp)

llama_build_and_test(test-model-load-cancel.cpp LABEL "model")
llama_build_and_test(test-autorelease.cpp       LABEL "model")
llama_build_and_test(test-backend-sampler.cpp   LABEL "model")

# Test for state restore with fragmented KV cache
# Requires a model, uses same args pa tern as test-thread-safety
llama_build_and_test(test-state-restore-fragmented.cpp LABEL "model" ARGS -m "${MODEL_DEST}")
set_tests_properties(test-state-restore-fragmented PROPERTIES FIXTURES_REQUIRED test-download-model)

if (NOT GGML_BACKEND_DL)
    # these tests use the backends directly and cannot be built with dynamic loading
    llama_build_and_test(test-barrier.cpp)
    llama_build_and_test(test-quantize-fns.cpp)
    llama_build_and_test(test-quantize-perf.cpp)
    llama_build_and_test(test-rope.cpp)

    if (GGML_REEX_GEMM)
        add_executable(test-reex-gemm test-reex/test-reex-gemm.cpp)
        target_compile_definitions(test-reex-gemm PRIVATE GGML_USE_REEX_GEMM)
        target_include_directories(test-reex-gemm PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src/ggml-cpu ${PROJECT_SOURCE_DIR}/ggml/src)
        # 静态链接时用 WHOLE_ARCHIVE 强制拉入 ggml-cpu 全部 .o，否则 ggml_vec_dot_q4_K_q8_K 可能未被拉入导致 reex=0
        if(NOT BUILD_SHARED_LIBS)
            if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
                target_link_libraries(test-reex-gemm PRIVATE llama-common llama ggml "$<LINK_GROUP:WHOLE_ARCHIVE,ggml-cpu>")
            else()
                # CMake < 3.24（如 3.22）：用 -Wl,--whole-archive 实现同样效果
                target_link_libraries(test-reex-gemm PRIVATE llama-common llama ggml "-Wl,--whole-archive" ggml-cpu "-Wl,--no-whole-archive")
            endif()
        else()
            target_link_libraries(test-reex-gemm PRIVATE llama-common llama ggml ggml-cpu)
        endif()
        if(UNIX AND NOT APPLE)
            set_target_properties(test-reex-gemm PROPERTIES BUILD_RPATH "\$ORIGIN")
        elseif(APPLE)
            set_target_properties(test-reex-gemm PROPERTIES BUILD_RPATH "@executable_path")
        endif()
        add_test(NAME test-reex-gemm COMMAND $<TARGET_FILE:test-reex-gemm>)
        set_tests_properties(test-reex-gemm PROPERTIES LABELS main TIMEOUT 60)
        install(TARGETS test-reex-gemm RUNTIME)
        # 带中间结果：cmake --build build --target test-reex-gemm-run 会在终端打印精度摘要
        add_custom_target(test-reex-gemm-run
            COMMAND $<TARGET_FILE:test-reex-gemm>
            WORKING_DIRECTORY $<TARGET_FILE_DIR:test-reex-gemm>
            COMMENT "Run test-reex-gemm (show accuracy summary)"
            USES_TERMINAL
            VERBATIM)
        add_dependencies(test-reex-gemm-run test-reex-gemm)
    endif()

    # REEX CUDA 测试：Q8 与 Q16 必须分开配置/构建，避免同一编译宏下的路径混淆
    if (TARGET ggml-cuda)
        if (NOT GGML_REEX_GEMM OR GGML_REEX_GEMM_ACTIVATION STREQUAL "Q8")
            add_executable(test-reex-cuda-q8 test-reex/test-reex-cuda-verify.cpp)
            target_include_directories(test-reex-cuda-q8 PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src)
            target_link_libraries(test-reex-cuda-q8 PRIVATE llama-common llama ggml ggml-cpu ggml-cuda)
            add_test(NAME test-reex-cuda-q8 COMMAND $<TARGET_FILE:test-reex-cuda-q8>)
            set_tests_properties(test-reex-cuda-q8 PROPERTIES LABELS "main;cuda;reex" TIMEOUT 60)
            install(TARGETS test-reex-cuda-q8 RUNTIME)
        endif()

        if (GGML_REEX_GEMM AND GGML_REEX_GEMM_ACTIVATION STREQUAL "Q16")
            add_executable(test-reex-cuda-q16 test-reex/test-reex-cuda-q16.cpp)
            target_include_directories(test-reex-cuda-q16 PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src)
            target_link_libraries(test-reex-cuda-q16 PRIVATE llama-common llama ggml ggml-cpu ggml-cuda)
            add_test(NAME test-reex-cuda-q16 COMMAND $<TARGET_FILE:test-reex-cuda-q16>)
            set_tests_properties(test-reex-cuda-q16 PROPERTIES LABELS "main;cuda;reex" TIMEOUT 60)
            install(TARGETS test-reex-cuda-q16 RUNTIME)
        endif()
    endif()
endif()

# libmtmd
set(LLAMA_TEST_NAME test-mtmd-c-api)
llama_build_and_test(test-mtmd-c-api.c)
target_link_libraries(${LLAMA_TEST_NAME} PRIVATE mtmd)
unset(LLAMA_TEST_NAME)

# GGUF model data fetcher library for tests that need real model metadata
# Only compile when cpp-httplib has SSL support (CPPHTTPLIB_OPENSSL_SUPPORT)
if (TARGET cpp-httplib)
    get_target_property(_cpp_httplib_defs cpp-httplib INTERFACE_COMPILE_DEFINITIONS)
    if (_cpp_httplib_defs MATCHES "CPPHTTPLIB_OPENSSL_SUPPORT")
        add_library(gguf-model-data STATIC gguf-model-data.cpp)
        target_link_libraries(gguf-model-data PRIVATE llama-common cpp-httplib)
        target_include_directories(gguf-model-data PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

        add_executable(test-gguf-model-data test-gguf-model-data.cpp)
        target_link_libraries(test-gguf-model-data PRIVATE gguf-model-data llama-common)
        llama_test(test-gguf-model-data LABEL "model")

        # test-quant-type-selection requires gguf-model-data for remote model metadata
        llama_build_and_test(test-quant-type-selection.cpp LABEL "model")
        target_link_libraries(test-quant-type-selection PRIVATE gguf-model-data)
        target_compile_definitions(test-quant-type-selection PRIVATE
            SNAPSHOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/snapshots")
    endif()
endif()

# dummy executable - not installed
get_filename_component(TEST_TARGET test-c.c NAME_WE)
add_executable(${TEST_TARGET} test-c.c)
target_link_libraries(${TEST_TARGET} PRIVATE llama)

llama_build_and_test(test-alloc.cpp)
target_include_directories(test-alloc PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src)

# Single-token eval for layer dump comparison (Qwen3Moe single-block, etc.)
unset(LLAMA_TEST_NAME)
llama_build(test-reex/eval_single_token.cpp)

# MoE GPU 采数工具: 8k Prefill, dump router/expert-reorder/expert-result/un-reorder
# tensors as golden reference for RTL 专家重排 verification (see output/README.md).
unset(LLAMA_TEST_NAME)
llama_build(test-reex/dump_moe_prefill.cpp)

# Local TurboQuant integration (REEX) — algorithm reference (oracle).
#   See docs/turboquant/01_设计与实现计划.md and tests/oracle/README.md.
#   Only built when REEX_TURBOQUANT=ON; the subdirectory itself early-returns
#   if the option is OFF so this add_subdirectory() call is always safe.
if (REEX_TURBOQUANT)
    add_subdirectory(oracle)
    add_subdirectory(reex)
endif()

llama_build(export-graph-ops.cpp)
target_include_directories(export-graph-ops PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src)
if (TARGET gguf-model-data)
    target_link_libraries(export-graph-ops PRIVATE gguf-model-data)
    target_compile_definitions(export-graph-ops PRIVATE LLAMA_HF_FETCH)
endif()
