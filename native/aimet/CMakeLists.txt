cmake_minimum_required(VERSION 3.19)

project(rx_met_aimet_native LANGUAGES CXX)

option(RX_MET_AIMET_ENABLE_CUDA "Build AIMET native modules with CUDA" OFF)
set(RX_MET_ONNXRUNTIME_ROOT "" CACHE PATH "ONNX Runtime headers root")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python3 3.10 EXACT COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)

if(RX_MET_AIMET_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

add_subdirectory(common)
add_subdirectory(onnx)
