cmake_minimum_required(VERSION 3.18)  # for CMAKE_CUDA_ARCHITECTURES

find_package(CUDAToolkit)

if (CUDAToolkit_FOUND)
    message(STATUS "CUDA Toolkit found")

    if (NOT TARGET CUDA::cudadevrt)
        find_library(GGML_CUDA_DEVICE_RUNTIME_LIBRARY
            NAMES cudadevrt
            HINTS
                "${CUDAToolkit_LIBRARY_ROOT}"
                "${CUDAToolkit_TARGET_DIR}"
                "/usr/local/cuda/lib64"
                "/usr/local/cuda/targets/x86_64-linux/lib"
            PATH_SUFFIXES lib lib64
        )
    endif()

    if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        # native == GPUs available at build time
        # 50     == Maxwell, lowest CUDA 12 standard
        # 60     == P100, FP16 CUDA intrinsics
        # 61     == Pascal, __dp4a instruction (per-byte integer dot product)
        # 70     == V100, FP16 tensor cores
        # 75     == Turing, int8 tensor cores
        # 80     == Ampere, asynchronous data loading, faster tensor core instructions
        # 86     == RTX 3000, needs CUDA v11.1
        # 89     == RTX 4000, needs CUDA v11.8
        # 120    == Blackwell, needs CUDA v12.8, FP4 tensor cores
        #
        # XX-virtual == compile CUDA code as PTX, do JIT compilation to binary code on first run
        # XX-real    == compile CUDA code as device code for this specific architecture
        # no suffix  == compile as both PTX and device code
        #
        # The default behavior for a non-native is to build virtual architectures as needed to cover all features needed
        #     for best performance and to also build real architectures for the most commonly used GPUs.
        if (GGML_NATIVE AND CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.6" AND CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
            set(CMAKE_CUDA_ARCHITECTURES "native")
        else()
            if (CUDAToolkit_VERSION VERSION_LESS "13")
                list(APPEND CMAKE_CUDA_ARCHITECTURES 50-virtual 61-virtual 70-virtual)
            endif ()

            list(APPEND CMAKE_CUDA_ARCHITECTURES 75-virtual 80-virtual 86-real)

            if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.8")
                list(APPEND CMAKE_CUDA_ARCHITECTURES 89-real)
            endif()

            if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.8")
                # The CUDA architecture 120f-virtual would in principle work for Blackwell support
                #     but the newly added "f" suffix conflicted with a preexising regex for validating CUDA architectures in CMake.
                # So either a recent CMake version or one with the backported fix is needed.
                # The following versions should work:
                #   - CMake >= v3.31.8 && CMake < v4.0.0
                #   - CMake >= v4.0.2
                # This is NOT documented in the CMake release notes,
                #     check Modules/Internal/CMakeCUDAArchitecturesValidate.cmake in the CMake git repository instead.
                # However, the architectures 120a-real and 121a-real should work with basically any CMake version and
                #     until the release of e.g. Rubin there is no benefit to shipping virtual architectures for Blackwell.
                list(APPEND CMAKE_CUDA_ARCHITECTURES 120a-real)
            endif()
            if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.9")
                list(APPEND CMAKE_CUDA_ARCHITECTURES 121a-real)
            endif()
        endif()
    endif()

    enable_language(CUDA)

    # TODO: Remove once CCCL 3.2 has been released and bundled with CUDA Toolkit
    if (GGML_CUDA_CUB_3DOT2)
        include(FetchContent)

        FetchContent_Declare(
            CCCL
            GIT_REPOSITORY https://github.com/nvidia/cccl.git
            GIT_TAG        v3.2.0
            GIT_SHALLOW    TRUE
        )

        FetchContent_MakeAvailable(CCCL)
    endif()

    # Replace any plain 12X CUDA architectures with their "architecture-specific" equivalents 12Xa.
    # 12X is forwards-compatible, 12Xa is not.
    # Notably the Blackwell FP4 tensor core instructions are not forwards compatible and therefore need 12Xa.
    # But while 12X vs. 12Xa can be checked in device code there is (to my knowledge) no easy way to do the same check in host code.
    # So for now just replace all instances of 12X with 12Xa, this should be fine until Rubin is released.
    foreach(ARCHS IN ITEMS CMAKE_CUDA_ARCHITECTURES CMAKE_CUDA_ARCHITECTURES_NATIVE)
        set(FIXED_ARCHS "")
        foreach(ARCH IN LISTS ${ARCHS})
            if (ARCH MATCHES "^12[0-9](-real|-virtual)?$")
                string(REGEX REPLACE "^(12[0-9])((-real|-virtual)?)$" "\\1a\\2" FIXED_ARCH ${ARCH})
                message(STATUS "Replacing ${ARCH} in ${ARCHS} with ${FIXED_ARCH}")
                list(APPEND FIXED_ARCHS "${FIXED_ARCH}")
            else()
                list(APPEND FIXED_ARCHS "${ARCH}")
            endif()
        endforeach()
        set(${ARCHS} ${FIXED_ARCHS})
    endforeach()

    # If we try to compile a "native" build it will use the 12X architectures and fail.
    # So we should instead use the native architectures as determined by CMake after replacing 12X with 12Xa.
    # But if at the time of the build no GPUs are connected at all CMAKE_CUDA_ARCHITECTURES will contain garbage that we should not use.
    if (CMAKE_CUDA_ARCHITECTURES STREQUAL "native" AND CMAKE_CUDA_ARCHITECTURES_NATIVE MATCHES "^[0-9]+(a|f)?(-real|-virtual)?(;[0-9]+(a|f)?(-real|-virtual)?|;)*$")
        set(CMAKE_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES_NATIVE})
    endif()
    message(STATUS "Using CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES} CMAKE_CUDA_ARCHITECTURES_NATIVE=${CMAKE_CUDA_ARCHITECTURES_NATIVE}")

    file(GLOB   GGML_HEADERS_CUDA "*.cuh")
    list(APPEND GGML_HEADERS_CUDA "../../include/ggml-cuda.h")

    file(GLOB   GGML_SOURCES_CUDA "*.cu")
    list(APPEND GGML_SOURCES_CUDA reex/reex_cuda_trace.cu)
    list(APPEND GGML_SOURCES_CUDA reex/reex_q16_trace.cu)
    file(GLOB   SRCS "template-instances/fattn-tile*.cu")
    list(APPEND GGML_SOURCES_CUDA ${SRCS})
    file(GLOB   SRCS "template-instances/fattn-mma*.cu")
    list(APPEND GGML_SOURCES_CUDA ${SRCS})
    file(GLOB   SRCS "template-instances/mmq*.cu")
    list(APPEND GGML_SOURCES_CUDA ${SRCS})
    file(GLOB   SRCS "template-instances/mmf*.cu")
    list(APPEND GGML_SOURCES_CUDA ${SRCS})

    if (GGML_CUDA_FA_ALL_QUANTS)
        file(GLOB   SRCS "template-instances/fattn-vec*.cu")
        list(APPEND GGML_SOURCES_CUDA ${SRCS})
        add_compile_definitions(GGML_CUDA_FA_ALL_QUANTS)
    else()
        list(APPEND GGML_SOURCES_CUDA
            template-instances/fattn-vec-instance-f16-f16.cu
            template-instances/fattn-vec-instance-q4_0-q4_0.cu
            template-instances/fattn-vec-instance-q8_0-q8_0.cu
            template-instances/fattn-vec-instance-bf16-bf16.cu)
    endif()

    if (GGML_REEX_GEMM)
        list(APPEND GGML_SOURCES_CUDA reex/reex_gemm_cuda.cu)
    endif()

    if (GGML_USE_REEX)
        file(GLOB SRCS "reex/*.cu")
        list(APPEND GGML_SOURCES_CUDA ${SRCS})
    endif()

    if (REEX_TURBOQUANT)
        # === REEX_TURBOQUANT BEGIN ===
        # CUDA-side dequantize / GET_ROWS / SET_ROWS / WHT kernels for
        # GGML_TYPE_TQ_K3 / TQ_V2 / TQ_V4.  All sources are guarded by
        # `#ifdef REEX_TURBOQUANT` so it is safe (but pointless) to include
        # them when the macro is OFF.  We list them explicitly rather than
        # globbing reex/* so the GGML_USE_REEX path above stays decoupled.
        list(APPEND GGML_SOURCES_CUDA
            reex/reex_turboquant_dequant.cu
            reex/reex_turboquant_wht.cu
            template-instances/fattn-vec-instance-f16-tq_v2.cu
            template-instances/fattn-vec-instance-tq_k3-tq_v2.cu
            template-instances/fattn-vec-instance-tq_k3-tq_v4.cu
            template-instances/fattn-vec-instance-tq_k3-tq_v_polar2.cu
            template-instances/fattn-vec-instance-tq_k3-tq_v_polar4.cu
        )
        list(APPEND GGML_HEADERS_CUDA
            reex/reex_turboquant_dequant.cuh
            reex/reex_turboquant_fattn_helpers.cuh
            reex/reex_turboquant_quantize_kv.cuh
            reex/reex_turboquant_wht.cuh
        )
        # === REEX_TURBOQUANT END ===
    endif()

    ggml_add_backend_library(ggml-cuda
                             ${GGML_HEADERS_CUDA}
                             ${GGML_SOURCES_CUDA}
                            )

    if (GGML_USE_REEX)
        target_compile_definitions(ggml-cuda PRIVATE GGML_USE_REEX)
        # REEX LUT uses __constant__ memory defined in unary.cu but referenced
        # from norm.cu, softmax.cu, rope.cu, fattn-*.cuh etc. via extern __constant__.
        # Without separable compilation each TU has its own __constant__ copy
        # and only unary.cu's gets initialized, leaving others as zeros.
        # [REEX custom] BUILD_SHARED_LIBS=OFF 的 CI 会把 ggml-cuda 作为静态库参与链接；
        # 仅开启 separable compilation 还不够，需同时做 device-link 收口，
        # 否则 test-backend-ops 等目标会在链接阶段报 __cudaRegisterLinkedBinary 未定义。
        set_property(TARGET ggml-cuda PROPERTY CUDA_SEPARABLE_COMPILATION ON)
        set_property(TARGET ggml-cuda PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    endif()

    add_compile_definitions(GGML_CUDA_PEER_MAX_BATCH_SIZE=${GGML_CUDA_PEER_MAX_BATCH_SIZE})

    if (GGML_CUDA_GRAPHS)
        add_compile_definitions(GGML_CUDA_USE_GRAPHS)
    endif()

    if (GGML_CUDA_FORCE_MMQ)
        add_compile_definitions(GGML_CUDA_FORCE_MMQ)
    endif()

    if (GGML_CUDA_FORCE_CUBLAS)
        add_compile_definitions(GGML_CUDA_FORCE_CUBLAS)
    endif()

    if (GGML_CUDA_NO_VMM)
        add_compile_definitions(GGML_CUDA_NO_VMM)
    endif()

    if (NOT GGML_CUDA_FA)
        add_compile_definitions(GGML_CUDA_NO_FA)
    endif()

    if (GGML_REEX_GEMM)
        add_compile_definitions(GGML_USE_REEX_GEMM_CUDA)
        if (GGML_REEX_GEMM_ACTIVATION STREQUAL "Q8")
            add_compile_definitions(GGML_REEX_GEMM_ACTIVATION_Q8)
        elseif (GGML_REEX_GEMM_ACTIVATION STREQUAL "FLOAT")
            add_compile_definitions(GGML_REEX_GEMM_ACTIVATION_FLOAT)
        else()
            add_compile_definitions(GGML_REEX_GEMM_ACTIVATION_Q16)
        endif()
    endif()

    if (GGML_CUDA_NO_PEER_COPY)
        add_compile_definitions(GGML_CUDA_NO_PEER_COPY)
    endif()

    if (GGML_STATIC)
        if (WIN32)
            # As of 12.3.1 CUDA Toolkit for Windows does not offer a static cublas library
            target_link_libraries(ggml-cuda PRIVATE CUDA::cudart_static CUDA::cublas)
        else ()
            if (GGML_CUDA_CUB_3DOT2)
                target_link_libraries(ggml-cuda PRIVATE  CCCL::CCCL)
            endif()
            if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL "10.1")
                target_link_libraries(ggml-cuda PRIVATE  CUDA::cudart_static CUDA::cublas_static CUDA::cublasLt_static)
            else()
                target_link_libraries(ggml-cuda PRIVATE  CUDA::cudart_static CUDA::cublas_static)
            endif()
        endif()
    else()
        if (GGML_CUDA_CUB_3DOT2)
            target_link_libraries(ggml-cuda PRIVATE  CCCL::CCCL)
        endif()
        target_link_libraries(ggml-cuda PRIVATE CUDA::cudart CUDA::cublas)
    endif()

    # [REEX custom] 旧分支 12-job CUDA gate 在静态库链路下还需要 device runtime，
    # 否则 cmake_device_link.o 会在最终链接 test-backend-ops 时留下 __fatbinwrap_* 未定义。
    if (TARGET CUDA::cudadevrt)
        target_link_libraries(ggml-cuda PUBLIC CUDA::cudadevrt)
    elseif (GGML_CUDA_DEVICE_RUNTIME_LIBRARY)
        target_link_libraries(ggml-cuda PUBLIC "${GGML_CUDA_DEVICE_RUNTIME_LIBRARY}")
    endif()

    if (GGML_CUDA_NO_VMM)
        # No VMM requested, no need to link directly with the cuda driver lib (libcuda.so)
    else()
        target_link_libraries(ggml-cuda PRIVATE CUDA::cuda_driver)
    endif()

    if (GGML_CUDA_NCCL)
        find_package(NCCL)
        if (NCCL_FOUND)
            add_compile_definitions(GGML_USE_NCCL)
            target_link_libraries(ggml-cuda PRIVATE NCCL::NCCL)
        else()
            message(STATUS "Warning: NCCL not found, performance for multiple CUDA GPUs will be suboptimal")
        endif()
    endif()

    set(CUDA_CXX_FLAGS "")

    set(CUDA_FLAGS -use_fast_math -extended-lambda)

    # CUDA 12.8+ nvcc: 默认仍编译 sm_50/sm_61 等 virtual arch 时会刷屏弃用警告
    if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.8")
        list(APPEND CUDA_FLAGS -Wno-deprecated-gpu-targets)
    endif()

    if (GGML_CUDA_DEBUG)
        list(APPEND CUDA_FLAGS -lineinfo)
        add_compile_definitions(GGML_CUDA_DEBUG)
    endif()

    if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.8")
        # Options are:
        # - none (not recommended)
        # - speed (nvcc's default)
        # - balance
        # - size
        list(APPEND CUDA_FLAGS -compress-mode=${GGML_CUDA_COMPRESSION_MODE})
    endif()

    if (GGML_FATAL_WARNINGS)
        list(APPEND CUDA_FLAGS -Werror all-warnings)
    endif()

    if (GGML_ALL_WARNINGS AND NOT MSVC)
        set(NVCC_CMD ${CMAKE_CUDA_COMPILER} .c)
        if (NOT CMAKE_CUDA_HOST_COMPILER STREQUAL "")
            list(APPEND NVCC_CMD -ccbin ${CMAKE_CUDA_HOST_COMPILER})
        endif()

        execute_process(
            COMMAND ${NVCC_CMD} -Xcompiler --version
            OUTPUT_VARIABLE CUDA_CCFULLVER
            ERROR_QUIET
        )

        if (NOT CUDA_CCFULLVER MATCHES clang)
            set(CUDA_CCID "GNU")
            execute_process(
                COMMAND ${NVCC_CMD} -Xcompiler "-dumpfullversion -dumpversion"
                OUTPUT_VARIABLE CUDA_CCVER
                ERROR_QUIET
                OUTPUT_STRIP_TRAILING_WHITESPACE
            )
        else()
            if (CUDA_CCFULLVER MATCHES Apple)
                set(CUDA_CCID "AppleClang")
            else()
                set(CUDA_CCID "Clang")
            endif()
            string(REGEX REPLACE "^.* version ([0-9.]*).*$" "\\1" CUDA_CCVER ${CUDA_CCFULLVER})
        endif()

        message(STATUS "CUDA host compiler is ${CUDA_CCID} ${CUDA_CCVER}")

        ggml_get_flags(${CUDA_CCID} ${CUDA_CCVER})
        list(APPEND CUDA_CXX_FLAGS ${CXX_FLAGS} ${GF_CXX_FLAGS})  # This is passed to -Xcompiler later
    endif()

    if (NOT MSVC)
        list(APPEND CUDA_CXX_FLAGS -Wno-pedantic)
    else()
        # CCCL 3.2 onwards will require a cpp-standard-compliant preprocessor for MSVC
        # https://github.com/NVIDIA/cccl/pull/6827
        list(APPEND CUDA_CXX_FLAGS /Zc:preprocessor)
    endif()

    list(JOIN   CUDA_CXX_FLAGS " " CUDA_CXX_FLAGS_JOINED)  # pass host compiler flags as a single argument

    if (NOT CUDA_CXX_FLAGS_JOINED STREQUAL "")
        list(APPEND CUDA_FLAGS -Xcompiler ${CUDA_CXX_FLAGS_JOINED})
    endif()

    target_compile_options(ggml-cuda PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${CUDA_FLAGS}>")
else()
    message(FATAL_ERROR "CUDA Toolkit not found")
endif()
