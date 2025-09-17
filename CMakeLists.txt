cmake_minimum_required(VERSION 3.20.0)

# Set CMP0116 to NEW to avoid deprecation warning
# This policy is about generated file dependencies
if(POLICY CMP0116)
  cmake_policy(SET CMP0116 NEW)
endif()

# Suppress warnings about runtime path conflicts
if(POLICY CMP0060)
  cmake_policy(SET CMP0060 NEW)
endif()

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

# add lp_solve
find_library(LP_SOLVE_LIB NAMES liblpsolve55.so)
message(STATUS "Found lp_solve: ${LP_SOLVE_LIB}")



add_compile_options(-Wno-dev)
add_compile_options(-fexceptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CIRCT_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)

add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(include)
add_subdirectory(lib)
# add_subdirectory(tools)
