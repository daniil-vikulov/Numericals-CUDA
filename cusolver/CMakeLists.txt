cmake_minimum_required(VERSION 3.28)
project(PLAY_ZONE)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CUDA_STANDARD 23)

set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
set(CUDA_INCLUDE_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/include)
set(CUDA_LIBRARIES ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
set(CMAKE_CUDA_ARCHITECTURES 89)

add_executable(${PROJECT_NAME} main.cpp CUDASolver.cpp)

target_include_directories(${PROJECT_NAME} PRIVATE ${CUDA_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME}
        ${CUDA_LIBRARIES}/libcudart.so
        ${CUDA_LIBRARIES}/libcublas.so
        ${CUDA_LIBRARIES}/libcusolver.so
)

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
target_link_libraries(${PROJECT_NAME} LAPACK::LAPACK BLAS::BLAS)