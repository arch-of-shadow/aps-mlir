cmake_minimum_required(VERSION 3.20.0)

include(CheckCXXSourceCompiles)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")
set(CMAKE_CXX_STANDARD_REQUIRED YES)

project(aps_mlir LANGUAGES C CXX)

# MLIR LLVM CIRCT should be found from ./circt/build
set(CIRCT_BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/circt/build")
set(MLIR_DIR "${CIRCT_BUILD_DIR}/lib/cmake/mlir")
set(LLVM_DIR "${CIRCT_BUILD_DIR}/lib/cmake/llvm")
set(CIRCT_DIR "${CIRCT_BUILD_DIR}/lib/cmake/circt")

find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

find_package(CIRCT REQUIRED CONFIG)
if(CIRCT_FOUND)
    message(STATUS "Found CIRCT: ${CIRCT_PACKAGE_DIR}")
else()
    message(STATUS "CIRCT not found")
endif()


list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CIRCT_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)

add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(include)
add_subdirectory(lib)
# add_subdirectory(tools)
