#!/bin/bash
set -e

if [ -z "$APS" ]; then
  echo 'APS environment variable is not set. Please run `pixi shell` first.'
  exit 1
fi

mkdir -p $APS/tutorial/outputs

# Run CADL-to-MLIR parser
pixi run mlir $APS/tutorial/cadl/v3ddist_vv.cadl $APS/tutorial/outputs/v3ddist_vv.mlir

# Run the APS end-to-end synthesis flow (APS/TOR -> CMT2 -> FIRRTL -> SystemVerilog)
pixi run syn $APS/tutorial/outputs/v3ddist_vv.mlir $APS/tutorial/outputs/v3ddist_vv.sv