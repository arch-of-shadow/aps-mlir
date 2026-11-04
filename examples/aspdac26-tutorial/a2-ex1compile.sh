#!/bin/bash
set -e

if [ -z "$APS" ]; then
  echo 'APS environment variable is not set. Please run `pixi shell` first.'
  exit 1
fi

mkdir -p $APS/build/examples/aspdac26-tutorial

pixi run compile $APS/examples/aspdac26-tutorial/cadl/v3ddist_vv.cadl $APS/examples/aspdac26-tutorial/csrc/test_v3ddist_vv.c $APS/build/examples/aspdac26-tutorial/v3ddist_vv.riscv
