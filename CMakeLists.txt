cmake_minimum_required(VERSION 3.28)
set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
project(MyCUDAProject LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

include_directories(include)

add_executable(MyCUDAExecutable src/main.cpp src/CudaSolver.cpp)

target_link_libraries(MyCUDAExecutable PRIVATE CUDA::cudart CUDA::cublas)
