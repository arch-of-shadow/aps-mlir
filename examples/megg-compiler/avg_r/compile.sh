#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
OUT="$ROOT/build/examples/megg/avg_r"
mkdir -p "$OUT"
cd "$ROOT"
pixi run compile "$SCRIPT_DIR/avg_r.cadl" "$SCRIPT_DIR/test_avg_r.c" "$OUT/avg_r.riscv" "$@"
