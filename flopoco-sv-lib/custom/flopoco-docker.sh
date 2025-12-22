#!/bin/bash
# FloPoCo Docker wrapper script
# Usage: ./flopoco-docker.sh [flopoco args...]

WORK_DIR="${PWD}"

docker run --rm -v "${WORK_DIR}:/work" -w /work gdeest/flopoco "$@"
