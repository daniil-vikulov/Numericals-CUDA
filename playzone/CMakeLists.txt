cmake_minimum_required(VERSION 3.28)
project(PLAY_ZONE)

set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CUDA_STANDARD 23)

set(CMAKE_CUDA_ARCHITECTURES 86)

add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME} cuda cudart)
target_include_directories(${PROJECT_NAME} PRIVATE ${CUDA_INCLUDE_DIRS})
