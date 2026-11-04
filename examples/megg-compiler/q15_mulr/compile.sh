#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
OUT="$ROOT/build/examples/megg/q15_mulr"
mkdir -p "$OUT"
cd "$ROOT"
pixi run compile "$SCRIPT_DIR/q15_mulr.cadl" "$SCRIPT_DIR/test_q15_mulr.c" "$OUT/q15_mulr.riscv" "$@"
