#!/bin/bash

set -e

cd $APS_CHIPYARD/sims/verilator && make CONFIG=APSRocketConfig run-binary-debug BINARY=$APS/tutorial/outputs/v3ddist_vv_native.riscv LOADMEM=1 -j4

cd -