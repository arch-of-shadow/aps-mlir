#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
OUT="$ROOT/build/examples/megg/v3ddist_vv"
mkdir -p "$OUT"
cd "$ROOT"
pixi run compile "$SCRIPT_DIR/v3ddist_vv.cadl" "$SCRIPT_DIR/test_v3ddist_vv.c" "$OUT/v3ddist_vv.riscv" "$@"
