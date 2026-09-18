set(AIMET_ONNX_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src")

if(NOT RX_MET_ONNXRUNTIME_ROOT)
    set(
        RX_MET_ONNXRUNTIME_ROOT
        "${PROJECT_SOURCE_DIR}/third_party/onnxruntime"
    )
endif()

find_path(
    ONNXRUNTIME_INCLUDE_DIR
    onnxruntime_cxx_api.h
    PATHS
        "${RX_MET_ONNXRUNTIME_ROOT}/include"
        "${RX_MET_ONNXRUNTIME_ROOT}/include/onnxruntime/core/session"
    REQUIRED
    NO_DEFAULT_PATH
)

if(RX_MET_AIMET_ENABLE_CUDA)
    find_path(
        ONNXRUNTIME_INTERNAL_INCLUDE_DIR
        core/providers/cuda/cuda_context.h
        PATHS
            "${RX_MET_ONNXRUNTIME_ROOT}/include"
            "${RX_MET_ONNXRUNTIME_ROOT}/onnxruntime"
        REQUIRED
        NO_DEFAULT_PATH
    )
endif()

pybind11_add_module(libquant_info MODULE ${AIMET_ONNX_SRC}/QcQuantizeInfo.cpp)
target_include_directories(libquant_info PRIVATE ${AIMET_ONNX_SRC})
target_link_libraries(libquant_info PRIVATE aimet_quantization_runtime)
set_target_properties(libquant_info PROPERTIES INSTALL_RPATH "$ORIGIN")

add_library(aimet_onnxrt_ops SHARED
    ${AIMET_ONNX_SRC}/AimetOpUtils.cpp
    ${AIMET_ONNX_SRC}/QcQuantizeInfo.cpp
    ${AIMET_ONNX_SRC}/QcQuantizeOp.cpp
    ${AIMET_ONNX_SRC}/RegisterCustomOps.cpp
)
target_include_directories(aimet_onnxrt_ops PRIVATE
    ${AIMET_ONNX_SRC}
    ${PROJECT_SOURCE_DIR}/common/src
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${ONNXRUNTIME_INTERNAL_INCLUDE_DIR}
)
target_link_libraries(aimet_onnxrt_ops PRIVATE
    aimet_quantization_runtime
    pybind11::headers
    Python3::Module
)
set_target_properties(aimet_onnxrt_ops PROPERTIES
    OUTPUT_NAME aimet_onnxrt_ops
    INSTALL_RPATH "$ORIGIN"
)

if(RX_MET_AIMET_ENABLE_CUDA)
    target_compile_definitions(aimet_onnxrt_ops PRIVATE ONNX_CUDA=1)
endif()

install(TARGETS libquant_info aimet_onnxrt_ops LIBRARY DESTINATION .)
