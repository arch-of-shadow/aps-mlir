#!/usr/bin/env bash

set -e

# This script sets up the environment for the LLVM/MLIR/CIRCT project and its dependencies.

# - CIRCT_COMMIT: The branch, tag, or commit to checkout (defaults to cmt2).
CIRCT_COMMIT=${1:-${CIRCT_COMMIT:-cmt2}}

# Submodule update
# be careful, don't update chipyard's submodule, it's fragile
# and must be updated with it's own script!!!
git submodule update --init circt/
git -C circt checkout "$CIRCT_COMMIT"
git -C circt submodule update --init --recursive

# Rebuild when the selected CIRCT revision changes, including branch updates.
CIRCT_REVISION=$(git -C circt rev-parse HEAD)
SETUP_SUCCESS="circt $CIRCT_COMMIT $CIRCT_REVISION setup success"
if [ -x "${PWD}/circt/build/bin/circt-opt" ] && \
    [ -f __setup.success ] && \
    [ "$(cat __setup.success)" = "$SETUP_SUCCESS" ]; then
    echo "circt $CIRCT_COMMIT is already built at $CIRCT_REVISION"
    exit 0
fi

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
echo "$SETUP_SUCCESS" > __setup.success
