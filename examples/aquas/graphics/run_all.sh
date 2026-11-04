#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
OUT="$ROOT/build/examples/aquas/graphics"
mkdir -p "$OUT"
cd "$ROOT"
for name in vmvar mphong vrgb2yuv; do
  pixi run mlir "$SCRIPT_DIR/$name.cadl" "$OUT/$name.mlir"
  pixi run build/tools/aps-opt/aps-e2e -i "$OUT/$name.mlir" --export sv -o "$OUT/$name.sv"
done
