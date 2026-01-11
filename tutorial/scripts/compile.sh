#!/bin/bash
# compile.sh - Tutorial compilation wrapper
# Usage: compile.sh <example_name> [--handson]
# This script wraps pixi run compile for tutorial examples

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Parse arguments
EXAMPLE_NAME=""
HANDSON_FLAG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --handson)
            HANDSON_FLAG="--handson"
            shift
            ;;
        *)
            if [ -z "$EXAMPLE_NAME" ]; then
                EXAMPLE_NAME="$1"
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
CADL_FILE="tutorial/cadl/${EXAMPLE_NAME}.cadl"
TEST_C_FILE="tutorial/csrc/test_${EXAMPLE_NAME}.c"
OUTPUT_EXE="tutorial/${EXAMPLE_NAME}.riscv"

# Check input files
if [ ! -f "$CADL_FILE" ]; then
    echo "Error: CADL file not found: $CADL_FILE"
    exit 1
fi

if [ ! -f "$TEST_C_FILE" ]; then
    echo "Error: Test C file not found: $TEST_C_FILE"
    exit 1
fi

# Call pixi run compile (without --handson, as scripts/compile.sh doesn't support it)
pixi run compile "$CADL_FILE" "$TEST_C_FILE" "$OUTPUT_EXE"
