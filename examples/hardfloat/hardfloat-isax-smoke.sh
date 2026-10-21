#!/usr/bin/env bash
# Run under pixi run (Verilator, Yosys, PyYAML and SoftFloat toolchain required).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
OUT="$ROOT/build/hardfloat"
JOBS="${JOBS:-4}"
mkdir -p "$OUT/logs"
(
  cd "$OUT"
  bash "$ROOT/hardware/hardfloat/build-hardfloat-ip.sh" --resource-output="$OUT/resource.json"
)
python3 tests/hardfloat/generate_vectors.py "$OUT/vectors"
RTL="$OUT/rtl/HardFloatIP.sv"
for spec in '0 add 3 5' '1 sub 3 5' '2 mul 3 5' '0 add 3 7' '0 add 3 1'; do
  read -r op name latency capacity <<< "$spec"
  tag="${name}_l${latency}_q${capacity}"
  verilator --cc --exe --build --assert -Wno-fatal --top-module HF32Binary \
    -GOP="$op" -GLATENCY="$latency" -GCAPACITY="$capacity" \
    --Mdir "$OUT/obj_$tag" -j "$JOBS" \
    -CFLAGS "-O1 -DTEST_LATENCY=$latency -DTEST_CAPACITY=$capacity" \
    "$RTL" "$ROOT/tests/hardfloat/test_transaction.cpp" \
    > "$OUT/logs/verilator_$tag.log" 2>&1
  "$OUT/obj_$tag/VHF32Binary" "$OUT/vectors/$name.txt"
done
# Deliberately wrong 0+0=1 golden result must fail numerically, not crash.
if "$OUT/obj_add_l3_q5/VHF32Binary" "$ROOT/tests/hardfloat/incorrect_add.txt" \
    > "$OUT/logs/negative-ip.log" 2>&1; then
  echo "ERROR: incorrect IP golden result was accepted" >&2; exit 1
else
  status=$?
  [[ "$status" == 1 ]] || exit "$status"
  grep -q 'arithmetic/flags/order mismatch' "$OUT/logs/negative-ip.log"
fi
if [[ "${PROTOCOL_ONLY:-0}" == 1 ]]; then
  exit 0
fi
for example in add add_mul; do
  ./tools/aps-frontend mlir "examples/hardfloat/$example.cadl" > "$OUT/$example.frontend.mlir"
  ./build/tools/aps-opt/aps-e2e -i "$OUT/$example.frontend.mlir" \
    --resource "$OUT/resource.json" --export sv \
    --dump-cmt2 "$OUT/$example.cmt2.mlir" -o "$OUT/$example.sv" \
    > "$OUT/logs/$example.lower.log" 2>&1
  ./circt/build/bin/circt-opt "$OUT/$example.cmt2.mlir" \
    --cmt2-print-call-info --cmt2-print-conflict-matrix --cmt2-print-scheduler \
    > "$OUT/logs/$example.analysis.log" 2>&1
  # SV export includes the implementation imported through Yosys and FIRRTL.
  verilator --lint-only -Wno-fatal --top-module main "$OUT/$example.sv" \
    > "$OUT/logs/$example.lint.log" 2>&1
  yosys -Q -T -m slang -p "read_slang -DSYNTHESIS $OUT/$example.sv; synth -top main; check -assert; stat" \
    > "$OUT/logs/$example.synth.log" 2>&1
  ./circt/build/bin/circt-opt "$OUT/$example.cmt2.mlir" \
    --cmt2-inline-modules --cmt2-inline-private-funcs \
    --cmt2-verify-private-funcs-inlined --cmt2-verify-call-sequence \
    > "$OUT/$example.inlined.mlir" 2> "$OUT/logs/$example.inline.log"
  verilator --cc --exe --build --assert -Wno-fatal --top-module main \
    --Mdir "$OUT/obj_isax_$example" -j "$JOBS" -CFLAGS '-O1' \
    "$OUT/$example.sv" "$ROOT/tests/hardfloat/test_isax.cpp" \
    > "$OUT/logs/$example.sim-build.log" 2>&1
  vectors=add
  if [[ "$example" == add_mul ]]; then vectors=chain; fi
  "$OUT/obj_isax_$example/Vmain" "$OUT/vectors/$vectors.txt"
  if [[ "$example" == add ]]; then
    if "$OUT/obj_isax_$example/Vmain" "$ROOT/tests/hardfloat/incorrect_add.txt" \
        > "$OUT/logs/negative-isax.log" 2>&1; then
      echo "ERROR: incorrect ISAX golden result was accepted" >&2; exit 1
    else
      status=$?
      [[ "$status" == 1 ]] || exit "$status"
      grep -q 'ISAX arithmetic/context/order mismatch' "$OUT/logs/negative-isax.log"
    fi
  fi
  echo "PASS $example: CADL -> APS -> CMT2 -> SV -> RTL simulation + synthesis"
done
