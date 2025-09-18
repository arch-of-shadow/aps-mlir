#!/usr/bin/env bash

set -e

# This script sets up the environment for the LLVM/MLIR/CIRCT project and its dependencies.

# ORTOOLS: downloads from github and unzip

UBUNTU_VERSION=$(lsb_release -rs)
echo "Ubuntu version: $UBUNTU_VERSION"
if [ "$UBUNTU_VERSION" == "22.04" ]; then
    DOWNLOAD_URL="https://github.com/google/or-tools/releases/download/v9.10/or-tools_amd64_ubuntu-22.04_cpp_v9.10.4067.tar.gz"
elif [ "$UBUNTU_VERSION" == "20.04" ]; then
    DOWNLOAD_URL="https://github.com/google/or-tools/releases/download/v9.10/or-tools_amd64_ubuntu-20.04_cpp_v9.10.4067.tar.gz"
elif [ "$UBUNTU_VERSION" == "24.04" ]; then
    DOWNLOAD_URL="https://github.com/google/or-tools/releases/download/v9.10/or-tools_amd64_ubuntu-24.04_cpp_v9.10.4067.tar.gz"
else
    echo "Error: Ubuntu version $UBUNTU_VERSION is not supported"
    exit 1
fi

# Create install directory if it doesn't exist
mkdir -p ./install

# Download and extract or-tools, stripping the top-level directory
wget -qO- $DOWNLOAD_URL | tar -xz --strip-components=1 -C ./install


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
    git clone git@github.com:circt/circt.git
fi

# cd into the CIRCT repository, pushd is better for this
pushd circt

# Checkout the CIRCT commit
git checkout $CIRCT_COMMIT

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
    -DLLVM_TARGETS_TO_BUILD="host" \
    -DLLVM_ENABLE_BINDINGS=OFF \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_EXTERNAL_PROJECTS=circt \
    -DLLVM_EXTERNAL_CIRCT_SOURCE_DIR=.. \
    -DLLVM_USE_SPLIT_DWARF=ON \
    -DLLVM_ENABLE_LLD=ON \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DCIRCT_BINDINGS_PYTHON_ENABLED=ON \
    -DCMAKE_INSTALL_PREFIX=../../install

# Ninja
ninja

# Ninja check mlir
ninja check-mlir

# Ninja check mlir python
ninja check-mlir-python

# Ninja check circt
ninja check-circt

# Ninja check circt integration
ninja check-circt-integration

# Ninja install
ninja install

# Echo success
echo "llvm/mlir/circt build/install success with python bindings"

# Cd back to the circt repository
popd


# Popd from the circt repository
popd

# Echo success
echo "circt $CIRCT_COMMIT setup success" > __setup.success