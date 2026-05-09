cmake_minimum_required(VERSION 3.18.0)
project(ML VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# adjust to fit your gpu
set(CMAKE_CUDA_ARCHITECTURES 75)

set(TENSORRT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/external/trt)
# add cuda tool kit lib
find_package(CUDAToolkit REQUIRED)

# build cuda module as static library
add_library(cuPlugins SHARED 
    plugin/tensorRTPlugin.cpp
    plugin/tensorRTPlugin.h
    plugin/tensorRTKernel.cu
    plugin/tensorRTKernel.h
)

# build with nvcc, export to generate lib file to link
set_target_properties(cuPlugins PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)

target_include_directories(cuPlugins PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/plugin
    ${TENSORRT_ROOT}/include)

# link plugin with cuda API
target_link_libraries(cuPlugins
    CUDA::cudart
    ${TENSORRT_ROOT}/lib/nvinfer.lib
    ${TENSORRT_ROOT}/lib/nvinfer_plugin.lib
    ${TENSORRT_ROOT}/lib/nvonnxparser.lib)

add_executable(ML main.cpp
    tensorRTLogger.h
    tensorRTService.h tensorRTService.cpp)

set_property(TARGET ML PROPERTY CXX_STANDARD 17)

# link to the project
target_link_libraries(ML
    cuPlugins)

