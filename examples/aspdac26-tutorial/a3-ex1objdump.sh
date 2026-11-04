#!/bin/bash
set -e

if [ -z "$APS" ]; then
  echo 'APS environment variable is not set. Please run `pixi shell` first.'
  exit 1
fi

cat $APS/build/examples/aspdac26-tutorial/compile_logs/v3ddist_vv.asm | grep -v f1202573 | grep insn -C 5