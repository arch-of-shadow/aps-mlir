#!/bin/bash

mkdir -p $APS/tutorial/outputs

# Run CADL-to-MLIR parser
pixi run mlir $APS/tutorial/cadl/inference.cadl $APS/tutorial/outputs/inference.mlir

# Run APS synthesis tool
pixi run opt $APS/tutorial/outputs/inference.mlir $APS/tutorial/outputs/inference_cmt.mlir

# Run Cement's synthesis flow
pixi run sv $APS/tutorial/outputs/inference_cmt.mlir $APS/tutorial/outputs/inference.sv