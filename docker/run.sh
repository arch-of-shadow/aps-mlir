#!/bin/bash

# Run the APS Docker container with mounted directories
# Usage: ./run.sh <APS_DIR> <TUTORIAL_DIR> <TMP_DIR>

set -e

if [ $# -ne 3 ]; then
    echo "Usage: $0 <APS_DIR> <TUTORIAL_DIR> <TMP_DIR>"
    echo "  APS_DIR:      Path to the APS repository on the host machine"
    echo "  TUTORIAL_DIR: Path to the tutorial directory on the host machine"
    echo "  TMP_DIR:      Path to a temporary directory for read-write mounts"
    exit 1
fi

APS_DIR="$(realpath "$1")"
TUTORIAL_DIR="$(realpath "$2")"
TMP_DIR="$(realpath -m "$3")"  # -m allows non-existent path

# Validate directories exist
if [ ! -d "$APS_DIR" ]; then
    echo "Error: APS_DIR '$APS_DIR' does not exist"
    exit 1
fi

if [ ! -d "$TUTORIAL_DIR" ]; then
    echo "Error: TUTORIAL_DIR '$TUTORIAL_DIR' does not exist"
    exit 1
fi

# Create temporary directory structure
mkdir -p "$TMP_DIR"
echo "Setting up temporary directory: $TMP_DIR"

# Copy tutorial directory to tmp
TMP_TUTORIAL_DIR="$TMP_DIR/tutorial"
echo "Copying $TUTORIAL_DIR to $TMP_TUTORIAL_DIR..."
rm -rf "$TMP_TUTORIAL_DIR"
cp -r "$TUTORIAL_DIR" "$TMP_TUTORIAL_DIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILDER_IMAGE="aps-mlir-builder:latest"
CONTAINER_NAME="aps-mlir"

# Check if builder image exists, if not build it
if ! docker image inspect "$BUILDER_IMAGE" &> /dev/null; then
    echo "Builder image '$BUILDER_IMAGE' not found. Building it now..."
    "$SCRIPT_DIR/build.sh" "$APS_DIR"
fi

# Extract .pixi from builder image
TMP_PIXI_DIR="$TMP_DIR/.pixi"
if [ ! -d "$TMP_PIXI_DIR" ] || [ -z "$(ls -A "$TMP_PIXI_DIR" 2>/dev/null)" ]; then
    echo "Extracting .pixi from builder image..."
    TEMP_CONTAINER="aps-mlir-temp-$$"
    docker create --name "$TEMP_CONTAINER" "$BUILDER_IMAGE" > /dev/null
    rm -rf "$TMP_PIXI_DIR"
    docker cp "$TEMP_CONTAINER:/root/aps/.pixi" "$TMP_PIXI_DIR"
    docker rm "$TEMP_CONTAINER" > /dev/null
    echo "Extracted .pixi to $TMP_PIXI_DIR"
else
    echo "Using existing .pixi from $TMP_PIXI_DIR"
fi

# Check if container with this name already exists
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Error: Container '$CONTAINER_NAME' already exists."
    echo "To remove it, run: docker stop $CONTAINER_NAME && docker rm $CONTAINER_NAME"
    exit 1
fi

MODULE_LIBRARY_DIR="$TMP_DIR/ModuleLibrary"
SRC_MODULE_LIBRARY_DIR="$APS_DIR/circt/lib/Dialect/Cmt2/ModuleLibrary"
if [ -d "$SRC_MODULE_LIBRARY_DIR" ]; then
    echo "Copying $SRC_MODULE_LIBRARY_DIR to $MODULE_LIBRARY_DIR..."
    rm -rf "$MODULE_LIBRARY_DIR"
    cp -r "$SRC_MODULE_LIBRARY_DIR" "$MODULE_LIBRARY_DIR"
else
    echo "Source ModuleLibrary not found, creating empty directory..."
    mkdir -p "$MODULE_LIBRARY_DIR"
fi

echo "Starting APS Docker container..."
echo "  APS_DIR:       $APS_DIR -> /root/aps (read-only)"
echo "  TMP_DIR:       $TMP_DIR"
echo "  ModuleLibrary: $MODULE_LIBRARY_DIR -> /root/aps/circt/lib/Dialect/Cmt2/ModuleLibrary (read-write)"
echo "  Tutorial:      $TMP_TUTORIAL_DIR -> /root/aps/tutorial (read-write)"
echo "  Pixi:          $TMP_PIXI_DIR -> /root/aps/.pixi (read-write)"

docker run -it --rm \
    --name "$CONTAINER_NAME" \
    -v "$APS_DIR:/root/aps:ro" \
    -v "$MODULE_LIBRARY_DIR:/root/aps/circt/lib/Dialect/Cmt2/ModuleLibrary:rw" \
    -v "$TMP_TUTORIAL_DIR:/root/aps/tutorial:rw" \
    -v "$TMP_PIXI_DIR:/root/aps/.pixi:rw" \
    uvxiao/aps-mlir:v0
