#!/usr/bin/env bash

set -euo pipefail

mkdir -p "$PIXI_PROJECT_ROOT/build"
cd "$PIXI_PROJECT_ROOT/build"

cmake .. \
  -G Ninja \
  -DCMAKE_CXX_FLAGS="-fuse-ld=lld" \
  -DCMAKE_BUILD_TYPE=Debug \
  -DProtobuf_INCLUDE_DIR=$CONDA_PREFIX/include \
  -DProtobuf_LIBRARY=$CONDA_PREFIX/lib/libprotobuf.so \
  -DProtobuf_PROTOC_EXECUTABLE=$CONDA_PREFIX/bin/protoc \
  -Dzstd_INCLUDE_DIR="$CONDA_PREFIX/include" \
  -Dzstd_LIBRARY="$CONDA_PREFIX/lib/libzstd.so" \
  -DZLIB_LIBRARY_RELEASE="$CONDA_PREFIX/lib/libz.so"

ninja "$@"
