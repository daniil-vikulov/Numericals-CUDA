cmake_minimum_required(VERSION 3.10)
project(IterativeSolverProject)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add include directory
include_directories(include)

# Add source files
set(SOURCES
    main.cpp 
    src/iterative_method.cpp 
    tests/tests.cpp
    src/gauss.cpp
)

# Add executable
add_executable(IterativeSolverProject ${SOURCES})

# Add clang-format target
find_program(CLANG_FORMAT "clang-format")
if (CLANG_FORMAT)
    add_custom_target(
        clang-format
        COMMAND ${CLANG_FORMAT} -i ${SOURCES}
        COMMENT "Running clang-format on source files..."
        VERBATIM
    )
    
    # Run clang-format before building the project
    add_dependencies(IterativeSolverProject clang-format)
else()
    message(WARNING "clang-format not found. Code will not be auto-formatted.")
endif()
