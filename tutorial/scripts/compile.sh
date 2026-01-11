#!/bin/bash
# compile.sh - Full compilation pipeline for tutorial examples
# Usage: compile.sh <example_name> [--handson]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Parse arguments
EXAMPLE_NAME=""
HANDSON_FLAG=""
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --handson)
            HANDSON_FLAG="--handson"
            shift
            ;;
        --*)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
        *)
            if [ -z "$EXAMPLE_NAME" ]; then
                EXAMPLE_NAME="$1"
            else
                EXTRA_ARGS="$EXTRA_ARGS $1"
            fi
            shift
            ;;
    esac
done

if [ -z "$EXAMPLE_NAME" ]; then
    echo "Usage: $0 <example_name> [--handson]"
    echo ""
    echo "Available examples:"
    ls -1 "$PROJECT_ROOT/tutorial/cadl/" | sed 's/.cadl$//'
    exit 1
fi

# Define paths
CADL_FILE="$PROJECT_ROOT/tutorial/cadl/${EXAMPLE_NAME}.cadl"
TEST_C_FILE="$PROJECT_ROOT/tutorial/csrc/test_${EXAMPLE_NAME}.c"
CSRC_DIR="$PROJECT_ROOT/tutorial/csrc"
MLIR_DIR="$PROJECT_ROOT/tutorial/mlir"
OUTPUT_DIR="$PROJECT_ROOT/output/compile_logs"
LOG_FILE="$OUTPUT_DIR/${EXAMPLE_NAME}.log"

# Check input files
if [ ! -f "$CADL_FILE" ]; then
    echo "Error: CADL file not found: $CADL_FILE"
    exit 1
fi

if [ ! -f "$TEST_C_FILE" ]; then
    echo "Error: Test C file not found: $TEST_C_FILE"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Output paths - pattern files go to tutorial/, final outputs go to output/
PATTERN_C="$CSRC_DIR/${EXAMPLE_NAME}.c"
PATTERN_MLIR="$MLIR_DIR/${EXAMPLE_NAME}.mlir"
ENCODING_JSON="$OUTPUT_DIR/${EXAMPLE_NAME}.json"
OUTPUT_EXE="$OUTPUT_DIR/${EXAMPLE_NAME}.out"

echo "Compiling $EXAMPLE_NAME..."

# Step 1: CADL -> C
echo "[1/4] CADL -> C"
pixi run -q python "$PROJECT_ROOT/aps-frontend" cadl2c "$CADL_FILE" -o "$PATTERN_C"

# Step 2: C -> MLIR
echo "[2/4] C -> MLIR"
pixi run -q cgeist -c -S "$PATTERN_C" --raise-scf-to-affine -O3 --memref-fullrank -o "$PATTERN_MLIR"
pixi run -q python "$PROJECT_ROOT/scripts/strip_mlir_attrs.py" "$PATTERN_MLIR"
pixi run -q mlir-opt --lower-affine "$PATTERN_MLIR" -o "$PATTERN_MLIR"

# Step 3: CADL -> encoding JSON
echo "[3/4] CADL -> encoding"
pixi run -q python "$PROJECT_ROOT/aps-frontend" encoding "$CADL_FILE" -o "$ENCODING_JSON"

# Step 4: Megg E2E
echo "[4/4] Megg E2E"
export PYTHONPATH="$PROJECT_ROOT/python:$PROJECT_ROOT/3rdparty/llvm-project/install/python_packages/mlir_core"

if [ -n "$HANDSON_FLAG" ]; then
    # Handson mode: output to stdout for web demo to capture
    pixi run -q python "$PROJECT_ROOT/megg-opt.py" --mode c-e2e \
        "$TEST_C_FILE" \
        --custom-instructions "$PATTERN_MLIR" \
        --encoding-json "$ENCODING_JSON" \
        -o "$OUTPUT_EXE" \
        --keep-intermediate \
        $HANDSON_FLAG \
        $EXTRA_ARGS 2>&1
else
    # Normal mode: redirect to log file
    pixi run -q python "$PROJECT_ROOT/megg-opt.py" --mode c-e2e \
        "$TEST_C_FILE" \
        --custom-instructions "$PATTERN_MLIR" \
        --encoding-json "$ENCODING_JSON" \
        -o "$OUTPUT_EXE" \
        --keep-intermediate \
        $EXTRA_ARGS > "$LOG_FILE" 2>&1
    echo "  Log: $LOG_FILE"
fi

echo ""
echo "Done. Outputs:"
echo "  $OUTPUT_EXE"
[ -f "$OUTPUT_DIR/${EXAMPLE_NAME}.asm" ] && echo "  $OUTPUT_DIR/${EXAMPLE_NAME}.asm"
