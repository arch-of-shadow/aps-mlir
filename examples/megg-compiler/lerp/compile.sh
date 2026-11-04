#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
OUT="$ROOT/build/examples/megg/lerp"
mkdir -p "$OUT"
cd "$ROOT"
pixi run compile "$SCRIPT_DIR/lerp.cadl" "$SCRIPT_DIR/test_lerp.c" "$OUT/lerp.riscv" "$@"
