#!/bin/bash

if [ -z "$APS" ]; then
  echo 'APS environment variable is not set. Please run `pixi shell` first.'
  exit 1
fi

mkdir -p $APS/thirdparty

pushd $APS/thirdparty

git clone https://github.com/povik/yosys-slang.git --recursive

pushd yosys-slang

mkdir -p build

pushd build

cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release

ninja

mkdir -p $CONDA_PREFIX/share/yosys/plugins

cp slang.so $CONDA_PREFIX/share/yosys/plugins/