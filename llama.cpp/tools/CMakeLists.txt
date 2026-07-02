# dependencies

find_package(Threads REQUIRED)

# third-party

# ...

# flags

llama_add_compile_flags()

# tools

if (EMSCRIPTEN)
else()
    add_subdirectory(batched-bench)
    add_subdirectory(gguf-split)
    add_subdirectory(imatrix)
    add_subdirectory(llama-bench)
    add_subdirectory(completion)
    add_subdirectory(perplexity)
    add_subdirectory(quantize)
    if (LLAMA_BUILD_SERVER)
        add_subdirectory(cli)
        add_subdirectory(server)
    endif()
    add_subdirectory(tokenize)
    add_subdirectory(parser)
    add_subdirectory(tts)
    add_subdirectory(mtmd)
    if (GGML_RPC)
        add_subdirectory(rpc)
    endif()
    if (NOT GGML_BACKEND_DL)
        # these examples use the backends directly and cannot be built with dynamic loading
        add_subdirectory(cvector-generator)
        add_subdirectory(export-lora)
    endif()
    add_subdirectory(fit-params)
    if (REEX_HW_CONVERT OR REEX_GEMM_DATAGEN)
        # CPU-only HW weight converter + shared reex-hw-layout library.
        # Added first so the reex-hw-layout target exists for reex-gemm-datagen.
        add_subdirectory(reex-hw-convert)
    endif()
    if (REEX_GEMM_DATAGEN)
        # Standalone GPU GEMM quant-sweep datagen tool (reuses reex-hw-layout).
        add_subdirectory(reex-gemm-datagen)
    endif()
    if (REEX_TURBOQUANT)
        # === REEX_TURBOQUANT BEGIN ===
        # Local TurboQuant offline tools (see tools/reex/CMakeLists.txt).
        add_subdirectory(reex)
        # === REEX_TURBOQUANT END ===
    endif()
    add_subdirectory(results)
endif()
