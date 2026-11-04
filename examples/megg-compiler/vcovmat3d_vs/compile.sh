#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
OUT="$ROOT/build/examples/megg/vcovmat3d_vs"
mkdir -p "$OUT"
cd "$ROOT"
pixi run compile "$SCRIPT_DIR/vcovmat3d_vs.cadl" "$SCRIPT_DIR/test_vcovmat3d_vs.c" "$OUT/vcovmat3d_vs.riscv" "$@"
