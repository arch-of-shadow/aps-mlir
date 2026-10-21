#!/usr/bin/env bash
# Full numerical + mapped-netlist + real Liberty setup/hold acceptance.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
cmake --build build -j "${BUILD_JOBS:-3}"
mkdir -p build/hardfloat
(cd build/hardfloat && bash ../../hardware/hardfloat/build-hardfloat-ip.sh --resource-output=resource.json)
python tests/hardfloat/acceptance/generate_cases.py
status=0
python tests/hardfloat/acceptance/run.py --jobs "${JOBS:-6}" "$@" || status=1
# Do not hide unit failures behind an earlier paired-case failure.
python tests/hardfloat/acceptance/run.py --units --jobs "${UNIT_JOBS:-3}" "$@" || status=1
exit "$status"
