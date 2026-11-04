#!/usr/bin/env bash
set -euo pipefail
PACKAGE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/package"
exec python3 "$PACKAGE/build-hardfloat-ip.py" "$@"
