#!/usr/bin/env bash
# Convert CADL to Standard MLIR via APS dialect
# Usage: ./scripts/mlir-std.sh <cadl-file>

set -e

# Check if aps-opt exists
if [ ! -f "./build/tools/aps-opt/aps-opt" ]; then
    echo "Error: aps-opt not found. Run 'pixi run build' first." >&2
    exit 1
fi

# Generate APS MLIR and extract module, then lower to standard
# Use pixi environment's Python which has all dependencies installed
pixi run python aps-frontend mlir "$@" 2>&1 | \
  sed -n '/^module {/,/^}$/p' | \
  ./build/tools/aps-opt/aps-opt --aps-to-standard
