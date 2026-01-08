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

# Clone the CIRCT repository
if [ ! -d "circt" ]; then
    git clone git@github.com:arch-of-shadow/circt-cmt2.git circt
fi

# cd into the CIRCT repository, pushd is better for this
pushd circt

# Fetch latest changes from remote
git fetch origin

# Checkout the CIRCT commit (use origin/ prefix for remote branch)
git checkout origin/$CIRCT_COMMIT

# Submodule update
git submodule init

# Submodule update
git submodule update

# Mkdir build 
mkdir -p build

# Cd into build
pushd build

# Cmake
cmake -G Ninja ../llvm/llvm \
    -DCMAKE_BUILD_TYPE=Debug \
    -DLLVM_ENABLE_PROJECTS="mlir" \
    -DLLVM_TARGETS_TO_BUILD="host;RISCV" \
    -DLLVM_ENABLE_BINDINGS=OFF \
    -DOR_TOOLS_PATH=${OR_TOOLS_PATH} \
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
    -DZSTD_INCLUDE_DIR="$CONDA_PREFIX/include" \
    -DZSTD_LIBRARY="$CONDA_PREFIX/lib/libzstd.so"

# Ninja
ninja

# Ninja check mlir
# ninja check-mlir

# Ninja check mlir python
# ninja check-mlir-python

# Ninja check circt
# ninja check-circt

# Ninja check circt integration
# ninja check-circt-integration

# Ninja install
# ninja install

# Echo success
echo "llvm/mlir/circt build/install success with python bindings"

# Cd back to the circt repository
popd


# Popd from the circt repository
popd

# Echo success
echo "circt $CIRCT_COMMIT setup success" > __setup.success