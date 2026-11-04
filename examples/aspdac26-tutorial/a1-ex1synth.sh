#!/bin/bash
set -e

if [ -z "$APS" ]; then
  echo 'APS environment variable is not set. Please run `pixi shell` first.'
  exit 1
fi

mkdir -p $APS/build/examples/aspdac26-tutorial

# Run CADL-to-MLIR parser
pixi run mlir $APS/examples/aspdac26-tutorial/cadl/v3ddist_vv.cadl $APS/build/examples/aspdac26-tutorial/v3ddist_vv.mlir

# Run the APS end-to-end synthesis flow (APS/TOR -> CMT2 -> FIRRTL -> SystemVerilog)
pixi run syn $APS/build/examples/aspdac26-tutorial/v3ddist_vv.mlir $APS/build/examples/aspdac26-tutorial/v3ddist_vv.sv