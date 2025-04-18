cmake_minimum_required(VERSION 3.22)
project(LLMENGINE LANGUAGES CXX CUDA)

# Specify standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA toolkit
find_package(CUDAToolkit REQUIRED)

# Output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Common flags
set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-Wall --generate-code=arch=compute_86,code=[compute_86,sm_86]")

# Debug flags
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O0 -G -Xcompiler=-Wall")

# Include and library directories
include_directories(
    ${PROJECT_SOURCE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

link_directories(${CUDAToolkit_LIBRARY_DIR})

# Collect source files
file(GLOB_RECURSE LLM_CXX_SOURCES src/*.cpp src/*.cc)
file(GLOB_RECURSE LLM_CUDA_SOURCES src/*.cu)

# Library definition
add_library(llmengine OBJECT ${LLM_CXX_SOURCES} ${LLM_CUDA_SOURCES})

# Subdirectories
add_subdirectory(src)
add_subdirectory(tests)
