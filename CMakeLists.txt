cmake_minimum_required(VERSION 3.18)
project(CudaConvolution LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set(SOURCES
    src/ImageProcessor.cpp
    src/main.cpp
    kernels/convolution.cu
)

add_executable(image_filter ${SOURCES})