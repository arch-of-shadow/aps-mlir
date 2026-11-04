#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
OUT="$ROOT/build/examples/megg/vgemv3d"
mkdir -p "$OUT"
cd "$ROOT"
pixi run compile "$SCRIPT_DIR/vgemv3d.cadl" "$SCRIPT_DIR/test_vgemv3d.c" "$OUT/vgemv3d.riscv" "$@"
