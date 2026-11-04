#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
OUT="$ROOT/build/examples/megg/horner3"
mkdir -p "$OUT"
cd "$ROOT"
pixi run compile "$SCRIPT_DIR/horner3.cadl" "$SCRIPT_DIR/test_horner3.c" "$OUT/horner3.riscv" "$@"
