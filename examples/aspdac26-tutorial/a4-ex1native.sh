#!/bin/bash
set -e

if [ -z "$APS" ]; then
  echo 'APS environment variable is not set. Please run `pixi shell` first.'
  exit 1
fi

mkdir -p $APS/build/examples/aspdac26-tutorial

pixi run compile-native $APS/examples/aspdac26-tutorial/csrc/test_v3ddist_vv.c $APS/build/examples/aspdac26-tutorial/v3ddist_vv_native.riscv

pixi run riscv32-unknown-elf-objdump -j .text -d --disassemble=v3ddist_vv $APS/build/examples/aspdac26-tutorial/v3ddist_vv_native.riscv | grep -v f1202573