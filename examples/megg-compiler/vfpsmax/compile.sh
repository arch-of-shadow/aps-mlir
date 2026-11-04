#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
OUT="$ROOT/build/examples/megg/vfpsmax"
mkdir -p "$OUT"
cd "$ROOT"
pixi run compile "$SCRIPT_DIR/vfpsmax.cadl" "$SCRIPT_DIR/test_vfpsmax.c" "$OUT/vfpsmax.riscv" "$@"
