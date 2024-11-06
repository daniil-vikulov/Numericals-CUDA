cmake_minimum_required(VERSION 3.28)
project(LAB_NOW)

set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CUDA_STANDARD 23)

# CUDA linkage for WSL2 RTX 40-series machine
set(CMAKE_CUDA_ARCHITECTURES 89)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
set(CUDA_INCLUDE_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/include)
set(CUDA_LIBRARIES ${CUDA_TOOLKIT_ROOT_DIR}/lib64)

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)


file(GLOB_RECURSE SRC src/*.cpp)

add_executable(${PROJECT_NAME} ${SRC})

target_include_directories(${PROJECT_NAME} PRIVATE ${CUDA_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME}
        ${CUDA_LIBRARIES}/libcudart.so
        ${CUDA_LIBRARIES}/libcublas.so
        ${CUDA_LIBRARIES}/libcusolver.so
)

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
target_link_libraries(${PROJECT_NAME} LAPACK::LAPACK BLAS::BLAS)