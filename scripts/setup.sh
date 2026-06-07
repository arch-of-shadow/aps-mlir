#!/usr/bin/env bash

set -e

# This script sets up the environment for the LLVM/MLIR/CIRCT project and its dependencies.

# check if circt is already installed
# ${which circt-opt} should be ${PWD}/install/bin/circt-opt
# ${circt-opt --version} should contain ${CIRCT_COMMIT}
# if so, return success
if [ -f "${PWD}/install/bin/circt-opt" ] && [[ "$(circt-opt --version)" == *"${CIRCT_COMMIT}"* ]]; then
    echo "circt is already installed"
    exit 0
fi

# - CIRCT_COMMIT: The commit to checkout
CIRCT_COMMIT=$1
if [ -z "$CIRCT_COMMIT" ]; then
    echo "Error: CIRCT_COMMIT is not set"
    exit 1
fi

# Submodule update
# be careful, don't update chipyard's submodule, it's fragile
# and must be updated with it's own script!!!
git submodule update --init --recursive circt/

# cd into the CIRCT repository, pushd is better for this
pushd circt

# Mkdir build 
mkdir -p build

# Cd into build
pushd build

export PKG_CONFIG_PATH="$CONDA_PREFIX/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
OR_TOOLS_PIXI_DEPS="$PIXI_PROJECT_ROOT/cmake/OrToolsPixiDeps.cmake"

# Cmake - use explicit paths for everything we need
cmake -G Ninja ../llvm/llvm \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_C_COMPILER="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc" \
    -DCMAKE_CXX_COMPILER="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++" \
    -DLLVM_ENABLE_PROJECTS="mlir" \
    -DLLVM_TARGETS_TO_BUILD="host;RISCV" \
    -DLLVM_ENABLE_BINDINGS=OFF \
    -DOR_TOOLS_PATH=$CONDA_PREFIX \
    -DCMAKE_PROJECT_TOP_LEVEL_INCLUDES="$OR_TOOLS_PIXI_DEPS" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_EXTERNAL_PROJECTS=circt \
    -DLLVM_EXTERNAL_CIRCT_SOURCE_DIR=.. \
    -DLLVM_USE_SPLIT_DWARF=ON \
    -DLLVM_ENABLE_LLD=ON \
    -DLLVM_ENABLE_OCAMLDOC=OFF \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DCIRCT_BINDINGS_PYTHON_ENABLED=ON \
    -DCMAKE_INSTALL_PREFIX=../../install \
    -DLLVM_ENABLE_ZSTD=FORCE_ON \
    -Dzstd_INCLUDE_DIR="$CONDA_PREFIX/include" \
    -Dzstd_LIBRARY="$CONDA_PREFIX/lib/libzstd.so" \
    -DZLIB_LIBRARY_RELEASE="$CONDA_PREFIX/lib/libz.so" \
    -DLLVM_ENABLE_LIBXML2=OFF \
    -DLLVM_INCLUDE_TESTS=OFF \
    -DMLIR_INCLUDE_TESTS=OFF \
    -DMLIR_INCLUDE_INTEGRATION_TESTS=OFF \
    -DCIRCT_INCLUDE_TESTS=ON \
    -DLLVM_INCLUDE_BENCHMARKS=OFF \
    -DLLVM_BUILD_BENCHMARKS=OFF \
    -DLLVM_INCLUDE_EXAMPLES=OFF \
    -DLLVM_BUILD_EXAMPLES=OFF \
    -DLLVM_INCLUDE_DOCS=OFF \
    -DMLIR_INCLUDE_DOCS=OFF \
    -DCIRCT_INCLUDE_DOCS=OFF \
    -DVERILATOR_DISABLE=ON

# Ninja
ninja

# Echo success
echo "llvm/mlir/circt build/install success with python bindings"

popd

popd

# Echo success
echo "circt $CIRCT_COMMIT setup success" > __setup.success
