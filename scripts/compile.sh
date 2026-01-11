#!/bin/bash
# compile.sh - Full compilation pipeline from CADL + test.c to executable
# Usage: compile.sh <cadl_file> <test_c_file> <output_executable>
#
# This script performs the complete compilation:
# 1. CADL → C (cadl2c)
# 2. C → MLIR (cgeist)
# 3. CADL → encoding JSON
# 4. Full E2E compilation with Megg

set -e

if [ $# -lt 3 ]; then
    echo "Usage: $0 <cadl_file> <test_c_file> <output_executable>"
    echo ""
    echo "Example:"
    echo "  $0 examples/diff_match/vgemv3d/vgemv3d.cadl examples/diff_match/vgemv3d/test_vgemv3d.c vgemv3d.out"
    exit 1
fi

CADL_FILE="$1"
TEST_C_FILE="$2"
OUTPUT="$3"

# Get absolute paths
CADL_FILE=$(realpath "$CADL_FILE")
TEST_C_FILE=$(realpath "$TEST_C_FILE")

# Get project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Extract base name from CADL file
CADL_BASENAME=$(basename "$CADL_FILE" .cadl)

# Create output directory for intermediate files and output
OUTPUT_DIR="${MEGG_OUTPUT_DIR:-$PROJECT_ROOT/output/compile_logs}"
mkdir -p "$OUTPUT_DIR"

# Update OUTPUT to be inside OUTPUT_DIR if it's just a filename
if [[ "$OUTPUT" != /* ]]; then
    OUTPUT="$OUTPUT_DIR/$OUTPUT"
fi

echo "=== Megg Compilation Pipeline ==="
echo "  CADL:   $CADL_FILE"
echo "  Test C: $TEST_C_FILE"
echo "  Output: $OUTPUT"
echo "  Logs:   $OUTPUT_DIR/"
echo ""

# Step 1: CADL → C
echo "[1/4] Converting CADL to C..."
PATTERN_C="$OUTPUT_DIR/${CADL_BASENAME}.c"
pixi run -q python "$PROJECT_ROOT/aps-frontend" cadl2c "$CADL_FILE" -o "$PATTERN_C"
echo "  ✓ Created: $PATTERN_C"

# Step 2: C → MLIR (using cgeist)
echo "[2/4] Converting C to MLIR..."
PATTERN_MLIR="$OUTPUT_DIR/${CADL_BASENAME}.mlir"
pixi run -q cgeist -c -S "$PATTERN_C" --raise-scf-to-affine -O3 --memref-fullrank -o "$PATTERN_MLIR"

# Strip module-level attributes (dlti, llvm.* etc.) that cause parsing issues
# This matches what e2e_compiler.py does in _strip_module_attributes
pixi run -q python "$PROJECT_ROOT/scripts/strip_mlir_attrs.py" "$PATTERN_MLIR"

# Lower affine to scf
pixi run -q mlir-opt --lower-affine "$PATTERN_MLIR" -o "$PATTERN_MLIR"
echo "  ✓ Created: $PATTERN_MLIR"

# Step 3: CADL → encoding JSON
echo "[3/4] Extracting encoding JSON..."
ENCODING_JSON="$OUTPUT_DIR/${CADL_BASENAME}.json"
pixi run -q python "$PROJECT_ROOT/aps-frontend" encoding "$CADL_FILE" -o "$ENCODING_JSON"
echo "  ✓ Created: $ENCODING_JSON"

# Step 4: Full E2E compilation
echo "[4/4] Running Megg E2E compilation..."
pixi run -q python "$PROJECT_ROOT/megg-opt.py" --mode c-e2e \
    "$TEST_C_FILE" \
    --custom-instructions "$PATTERN_MLIR" \
    --encoding-json "$ENCODING_JSON" \
    -o "$OUTPUT" \
    --keep-intermediate

echo ""
echo "=== Compilation Complete ==="
echo "  Executable:         $OUTPUT"
echo "  Intermediate files: $OUTPUT_DIR/"
echo ""
echo "Generated files:"
ls -la "$OUTPUT_DIR/"
