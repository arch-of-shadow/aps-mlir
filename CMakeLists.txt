cmake_minimum_required(VERSION 3.20.0)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set CMP0116 to NEW to avoid deprecation warning
# This policy is about generated file dependencies
if(POLICY CMP0116)
  cmake_policy(SET CMP0116 NEW)
endif()

# Suppress warnings about runtime path conflicts
if(POLICY CMP0060)
  cmake_policy(SET CMP0060 NEW)
endif()

cmake_policy(SET CMP0174 NEW)
cmake_policy(SET CMP0177 NEW)

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

find_package(ortools CONFIG)
if (ortools_FOUND)
    message(STATUS "Found OR-Tools: ${ortools_LIBDIR}")
else()
    message(STATUS "OR-Tools not found")
endif()

find_package(CIRCT REQUIRED CONFIG)
if(CIRCT_FOUND)
    message(STATUS "Found CIRCT: ${CIRCT_PACKAGE_DIR}")
else()
    message(STATUS "CIRCT not found")
endif()


list(APPEND CMAKE_MODULE_PATH "${CIRCT_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(AddCIRCT)
include(HandleLLVMOptions)

include(MLIRDetectPythonEnv)
# Note that both upstream and downstreams often call this macro. It gates
# internally on the MLIR_CONFIGURE_PYTHON_DEV_PACKAGES option.
mlir_configure_python_dev_packages()

message(STATUS "APS Python bindings are enabled.")
set(MLIR_DISABLE_CONFIGURE_PYTHON_DEV_PACKAGES 0)
mlir_detect_nanobind_install()
# Prime the search like mlir_configure_python_dev_modules
find_package(Python3 3.8 COMPONENTS Interpreter Development)
find_package(Python3 3.8 COMPONENTS Interpreter Development.Module)
find_package(nanobind 2.4.0 CONFIG REQUIRED)

# #-------------------------------------------------------------------------------
# LP_SOLVE configuration
#-------------------------------------------------------------------------------
set(LP_SOLVE_PATH "${CMAKE_SOURCE_DIR}/thirdparty/lp_solve_5.5/lpsolve55")
if (WIN32)
    find_library(LP_SOLVE_LIB NAMES liblpsolve55.lib PATHS "${LP_SOLVE_PATH}/bin/win64")
elseif (UNIX)
    if (NOT EXISTS "${LP_SOLVE_PATH}/bin/ux64/liblpsolve55.a")
        execute_process(
                COMMAND sh "${LP_SOLVE_PATH}/ccc"
                WORKING_DIRECTORY ${LP_SOLVE_PATH}
        )
    endif ()
    find_library(LP_SOLVE_LIB NAMES liblpsolve55.a PATHS "${LP_SOLVE_PATH}/bin/ux64")
endif ()

#-------------------------------------------------------------------------------
# JSON configuration
#-------------------------------------------------------------------------------
message(STATUS "Using nlohmann_json in: thirdparty/json")
set(JSON_BuildTests OFF CACHE INTERNAL "")
add_subdirectory(thirdparty/json)
include_directories(thirdparty/json/include)

IF(CMAKE_BUILD_TYPE MATCHES Debug)
  message(STATUS "Build Debug version")
  add_compile_options(-g)
ENDIF()
IF(CMAKE_BUILD_TYPE MATCHES Release)
  message(STATUS "Build Release version, SAFE COMPILER IS ON")
  add_compile_options(-fstack-protector-all)
  add_compile_options(-s)
ENDIF()


# APS CONFIGURATION
set(APS_MAIN_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR} ) # --src-root
set(APS_MAIN_INCLUDE_DIR ${APS_MAIN_SRC_DIR}/include)

set(APS_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(APS_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(APS_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/include)
set(APS_LIBRARY_DIR ${CMAKE_BINARY_DIR}/lib)
set(APS_TOOLS_DIR ${CMAKE_BINARY_DIR}/bin)
set(APS_UTILS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/utils)
set(APS_PYTHON_PACKAGES_DIR ${APS_BINARY_DIR}/python_packages)

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
add_subdirectory(tools)