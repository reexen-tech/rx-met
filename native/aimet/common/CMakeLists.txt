set(AIMET_QUANTIZATION_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(AIMET_PYTHON_BINDINGS "${CMAKE_CURRENT_SOURCE_DIR}/bindings/python")

add_library(aimet_quantization STATIC
    ${AIMET_QUANTIZATION_SRC}/IQuantizationAlgorithm.hpp
    ${AIMET_QUANTIZATION_SRC}/MainQuantizationClass.cpp
    ${AIMET_QUANTIZATION_SRC}/MainQuantizationClass.hpp
    ${AIMET_QUANTIZATION_SRC}/spec_functions.hpp
    ${AIMET_QUANTIZATION_SRC}/spec_functions.cpp
    ${AIMET_QUANTIZATION_SRC}/math_functions.cpp
    ${AIMET_QUANTIZATION_SRC}/math_functions.hpp
    ${AIMET_QUANTIZATION_SRC}/quantization_utils.cpp
    ${AIMET_QUANTIZATION_SRC}/quantization_utils.hpp
    ${AIMET_QUANTIZATION_SRC}/QuantizerFactory.cpp
    ${AIMET_QUANTIZATION_SRC}/TfEnhancedQuantizer.cpp
    ${AIMET_QUANTIZATION_SRC}/TfEnhancedQuantizer.hpp
    ${AIMET_QUANTIZATION_SRC}/TfQuantizer.cpp
    ${AIMET_QUANTIZATION_SRC}/TfQuantizer.hpp
    ${AIMET_QUANTIZATION_SRC}/trim_functions.cpp
    ${AIMET_QUANTIZATION_SRC}/trim_functions.hpp
    ${AIMET_QUANTIZATION_SRC}/tensor_utils.cpp
    ${AIMET_QUANTIZATION_SRC}/tensor_utils.hpp
    ${AIMET_QUANTIZATION_SRC}/ContiguousEncodingAnalyzer.cpp
    ${AIMET_QUANTIZATION_SRC}/ContiguousEncodingAnalyzer.h
    ${AIMET_QUANTIZATION_SRC}/EntropyEncodingAnalyzer.cpp
    ${AIMET_QUANTIZATION_SRC}/EntropyEncodingAnalyzer.h
    ${AIMET_QUANTIZATION_SRC}/MinMaxEncodingAnalyzer.cpp
    ${AIMET_QUANTIZATION_SRC}/MinMaxEncodingAnalyzer.h
    ${AIMET_QUANTIZATION_SRC}/MseEncodingAnalyzer.cpp
    ${AIMET_QUANTIZATION_SRC}/MseEncodingAnalyzer.h
    ${AIMET_QUANTIZATION_SRC}/PercentileEncodingAnalyzer.cpp
    ${AIMET_QUANTIZATION_SRC}/PercentileEncodingAnalyzer.h
    ${AIMET_QUANTIZATION_SRC}/TfEnhancedEncodingAnalyzer.cpp
    ${AIMET_QUANTIZATION_SRC}/TfEnhancedEncodingAnalyzer.h
    ${AIMET_QUANTIZATION_SRC}/TfEncodingAnalyzer.h
    ${AIMET_QUANTIZATION_SRC}/TfEncodingAnalyzer.cpp
    ${AIMET_QUANTIZATION_SRC}/EncodingAnalyzerWrapper.h
    ${AIMET_QUANTIZATION_SRC}/EncodingAnalyzerWrapper.cpp
    ${AIMET_QUANTIZATION_SRC}/TensorQuantizationSim.h
    ${AIMET_QUANTIZATION_SRC}/TensorQuantizationSim.cpp
    ${AIMET_QUANTIZATION_SRC}/EncodingAnalyzerForPython.cpp
    ${AIMET_QUANTIZATION_SRC}/TensorQuantizationSimForPython.cpp
    ${AIMET_QUANTIZATION_SRC}/TensorQuantizer.cpp
    ${AIMET_QUANTIZATION_SRC}/GraphQuantizer.cpp
    ${AIMET_QUANTIZATION_SRC}/Fp16Quantization.cpp
)
target_include_directories(aimet_quantization
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${AIMET_QUANTIZATION_SRC}
)
target_link_libraries(aimet_quantization PUBLIC
    Eigen3::Eigen
    pybind11::headers
    Python3::Module
)

add_library(aimet_quantization_runtime INTERFACE)
target_link_libraries(aimet_quantization_runtime INTERFACE aimet_quantization)

if(RX_MET_AIMET_ENABLE_CUDA)
    target_compile_definitions(aimet_quantization PRIVATE GPU_QUANTIZATION_ENABLED=1)
    target_include_directories(aimet_quantization PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

    add_library(aimet_quantization_cuda STATIC
        ${AIMET_QUANTIZATION_SRC}/spec_functions.cuh
        ${AIMET_QUANTIZATION_SRC}/spec_functions.cu
        ${AIMET_QUANTIZATION_SRC}/cuda_util.cu
        ${AIMET_QUANTIZATION_SRC}/trim_functions.cu
        ${AIMET_QUANTIZATION_SRC}/cuda_util.hpp
        ${AIMET_QUANTIZATION_SRC}/trim_functions.cuh
        ${AIMET_QUANTIZATION_SRC}/math_functions.cu
        ${AIMET_QUANTIZATION_SRC}/tensor_utils.cu
    )
    target_include_directories(aimet_quantization_cuda PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${AIMET_QUANTIZATION_SRC}
    )
    target_compile_definitions(aimet_quantization_cuda PRIVATE GPU_QUANTIZATION_ENABLED=1)
    target_compile_options(aimet_quantization_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info;--use_fast_math>
    )
    target_link_libraries(aimet_quantization_cuda PUBLIC CUDA::cudart)
    target_link_libraries(
        aimet_quantization_runtime INTERFACE aimet_quantization_cuda
    )
endif()

pybind11_add_module(_libpymo MODULE
    ${AIMET_PYTHON_BINDINGS}/PyModelOptimizations.cpp
    ${AIMET_PYTHON_BINDINGS}/PyTensorQuantizer.cpp
)
target_include_directories(_libpymo PRIVATE
    ${AIMET_PYTHON_BINDINGS}
    ${AIMET_QUANTIZATION_SRC}
)
target_link_libraries(_libpymo PRIVATE aimet_quantization_runtime)
set_target_properties(_libpymo PROPERTIES INSTALL_RPATH "$ORIGIN")

install(TARGETS _libpymo LIBRARY DESTINATION .)
