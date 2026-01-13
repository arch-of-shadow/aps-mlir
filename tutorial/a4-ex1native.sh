#!/bin/bash

pixi run compile-native csrc/test_v3ddist_vv.c $APS/tutorial/outputs/v3ddist_vv_baseline.riscv

pixi run riscv32-unknown-elf-objdump -D $APS/tutorial/outputs/v3ddist_vv_baseline.riscv | grep -v f1202573 | grep insn -C 5