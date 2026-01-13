#!/bin/bash

# Build the APS builder image with pre-built .pixi
# Usage: ./build.sh [APS_DIR]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ $# -eq 0 ]; then
    # Default to parent directory of docker/
    APS_DIR="$(realpath "$SCRIPT_DIR/..")"
else
    APS_DIR="$(realpath "$1")"
fi

if [ ! -d "$APS_DIR" ]; then
    echo "Error: APS_DIR '$APS_DIR' does not exist"
    exit 1
fi

BUILDER_IMAGE="aps-mlir-builder:latest"

echo "Building APS builder image..."
echo "  APS_DIR: $APS_DIR"
echo "  Image:   $BUILDER_IMAGE"

docker build \
    -t "$BUILDER_IMAGE" \
    -f "$SCRIPT_DIR/Dockerfile.builder" \
    "$APS_DIR"

echo "Builder image '$BUILDER_IMAGE' created successfully."
echo "This image contains a pre-built .pixi directory with correct container paths."
