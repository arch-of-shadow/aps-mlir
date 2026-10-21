#!/usr/bin/env bash
# Full Rocket + RoCC RTL simulation of the compiled RV32 C program.
set -euo pipefail
ulimit -c 0
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$ROOT"
CHIPYARD=${APS_CHIPYARD:-$ROOT/thirdparty/chipyard}
EXAMPLE="$ROOT/examples/hardfloat/bf16_gemm"
OUT=${OUT:-$ROOT/build/chipyard-bf16-gemm}
JOBS=${JOBS:-16}
mkdir -p "$OUT"
OUT=$(realpath "$OUT")
: "${RISCV:?Run this script through pixi run}"
export JAVA_TOOL_OPTIONS=${JAVA_TOOL_OPTIONS:-"-Xmx16G -Xss8M -XX:ActiveProcessorCount=$JOBS"}
cmake --build build --target aps-e2e -j "${BUILD_JOBS:-3}" > "$OUT/compiler-build.log" 2>&1
(
    cd "$OUT"
    bash "$ROOT/hardware/hardfloat/build-hardfloat-ip.sh" --resource-output="$OUT/resource.json"
) > "$OUT/package.log" 2>&1
python "$EXAMPLE/generate_vectors.py" "$OUT"
cadl_source="$EXAMPLE/gemm.cadl"
case ${LOOP_PIPELINE:-1} in
1) ;;
0)
    # Keep a serial reference with identical arithmetic, partitioning and DMA.
    python - "$cadl_source" "$OUT/gemm.serial.cadl" <<'CADL_PY'
from pathlib import Path
import sys
source, output = map(Path, sys.argv[1:])
text = source.read_text()
annotations = '    [[pipeline(1)]]\n    [[II(1)]]\n'
assert text.count(annotations) == 1, 'Expected one pipelined output loop'
output.write_text(text.replace(annotations, ''))
CADL_PY
    cadl_source="$OUT/gemm.serial.cadl"
    ;;
*) echo 'LOOP_PIPELINE must be 0 or 1' >&2; exit 2 ;;
esac
./tools/aps-frontend mlir "$cadl_source" -o "$OUT/input.mlir"
./build/tools/aps-opt/aps-e2e -i "$OUT/input.mlir" --resource "$OUT/resource.json" \
    --export sv --dump-cmt2 "$OUT/gemm.cmt2.mlir" -o "$OUT/gemm.next.sv" \
    > "$OUT/codegen.log" 2>&1
if cmp -s "$OUT/gemm.next.sv" "$OUT/gemm.sv"; then
    rm "$OUT/gemm.next.sv"
else
    mv "$OUT/gemm.next.sv" "$OUT/gemm.sv"
fi
if [[ ${VERIFY_CMT2:-1} == 1 ]]; then
    ./circt/build/bin/circt-opt "$OUT/gemm.cmt2.mlir" \
        --cmt2-inline-modules --cmt2-inline-private-funcs \
        --cmt2-verify-private-funcs-inlined --cmt2-verify-call-sequence \
        -o "$OUT/gemm.inlined.mlir" > "$OUT/inline.log" 2>&1
else
    echo 'Skipping the separate CMT2 inline/verification sweep (VERIFY_CMT2=0).'
fi
if [[ ${BENCHMARK:-0} == 1 ]]; then
    python "$EXAMPLE/generate_benchmark.py" "$OUT"
    flags=(-std=c11 -O3 -Wall -Wextra -Werror -fno-common -fno-builtin-printf
           -fno-fast-math -ffp-contract=fast
           -march=rv32imaf_zicsr_zifencei -mabi=ilp32f -mcmodel=medany
           -specs=htif_nano.specs -static -T htif.ld -I "$OUT")
    riscv32-unknown-elf-gcc "${flags[@]}" "$EXAMPLE/benchmark.c" "$EXAMPLE/fp32_software.c" \
        -o "$OUT/gemm_bench.riscv"
    riscv32-unknown-elf-objdump -d "$OUT/gemm_bench.riscv" > "$OUT/gemm_bench.asm"
    # Native FP32 baseline may use Rocket's FMA; never emulate BF16 rounding.
    grep -Fq 'fmadd.s' "$OUT/gemm_bench.asm"
    if grep -Eq '__addsf3|__mulsf3|bf16_round' "$OUT/gemm_bench.asm"; then
        echo 'Unexpected software arithmetic in FP32 benchmark' >&2
        exit 1
    fi
else
    flags=(-std=c11 -O2 -Wall -Wextra -Werror -fno-common -fno-builtin-printf
           -march=rv32ima_zicsr_zifencei -mabi=ilp32 -mcmodel=medany
           -specs=htif_nano.specs -static -T htif.ld -I "$OUT")
    riscv32-unknown-elf-gcc "${flags[@]}" "$EXAMPLE/test_gemm.c" -o "$OUT/gemm.riscv"
    riscv32-unknown-elf-gcc "${flags[@]}" -DGEMM_FORCE_FAIL "$EXAMPLE/test_gemm.c" -o "$OUT/gemm_fail.riscv"
    riscv32-unknown-elf-objdump -d "$OUT/gemm.riscv" > "$OUT/gemm.asm"
fi
# Select our accelerator without modifying Chipyard's default aps_config.yaml.
export APS_CONFIG="$OUT/aps_config.yaml"
profile_req=""
if [[ ${PROFILE:-0} == 1 ]]; then
    python "$EXAMPLE/generate_profile.py" "$OUT/gemm.sv" "$OUT/profile.sv"
    profile_req="$OUT/profile.sv"
fi
python - "$OUT" <<'PY'
import json
import os
from pathlib import Path
import sys
out = Path(sys.argv[1])
# JSON is valid YAML, including paths with spaces.
config = json.dumps(dict(
    backend="rocc", arch="rv32", core_count=1,
    top_module="aps_rocc_wrapper", name="aps_rocc_module",
    vsrc=[str(out / "gemm.sv")] + ([str(out / "profile.sv")] if os.environ.get("PROFILE") == "1" else []),
    maxBurstBytes=128, nXacts=2), indent=2)
path = out / "aps_config.yaml"
if not path.exists() or path.read_text() != config:
    path.write_text(config)
PY
make_args=(-C "$CHIPYARD/sims/verilator" CONFIG=APSRocketConfig "-j$JOBS" "nproc=$JOBS"
           "gen_dir=$OUT/generated-src" "output_dir=$OUT/output"
           "sim=$OUT/simulator" "sim_debug=$OUT/simulator-debug"
           'EXTRA_GENERATOR_REQS=$(BOOTROM_TARGETS) '"$APS_CONFIG $OUT/gemm.sv $profile_req"
           "SIM_OPT_CXXFLAGS=${SIM_CXX_OPT:--O0}"
           CXX=/usr/bin/g++ CC=/usr/bin/gcc AR=/usr/bin/ar LINK=/usr/bin/g++)
# Same Chipyard TestHarness/HTIF/LOADMEM flow as sims/verilator/run.sh.
# DEBUG=1 selects its run-binary-debug target (instruction log + waveform).
run_target=run-binary-fast
build_target=default
if [[ ${DEBUG:-0} == 1 ]]; then run_target=run-binary-debug; build_target=debug; fi
make "${make_args[@]}" "$build_target" 2>&1 | tee "$OUT/build.log"
if [[ ${BENCHMARK:-0} == 1 ]]; then
    make "${make_args[@]}" "$run_target" "BINARY=$OUT/gemm_bench.riscv" LOADMEM=1 \
        TIMEOUT_CYCLES=5000000 2>&1 | tee "$OUT/benchmark.log"
    grep -q 'BF16 BENCH PASS trials=2 checked=2048' "$OUT/benchmark.log"
    python "$EXAMPLE/summarize_benchmark.py" "$OUT/benchmark.log" "$OUT/benchmark.json"
    if [[ ${PROFILE:-0} == 1 ]]; then
        python "$EXAMPLE/summarize_profile.py" "$OUT/benchmark.log" "$OUT/profile.json"
    fi
    exit 0
fi
make "${make_args[@]}" "$run_target" "BINARY=$OUT/gemm.riscv" LOADMEM=1 \
    TIMEOUT_CYCLES=5000000 2>&1 | tee "$OUT/run.log"
grep -q 'BF16 GEMM PASS tiles=32 elements=2048 DMA=8tiles partition=8 unroll=2/8' "$OUT/run.log"
if make "${make_args[@]}" "$run_target" "BINARY=$OUT/gemm_fail.riscv" LOADMEM=1 \
    TIMEOUT_CYCLES=5000000 > "$OUT/negative.log" 2>&1; then
    echo 'FAIL: Rocket accepted the deliberately corrupted golden' >&2
    exit 1
fi
grep -q 'BF16 GEMM FAIL: batch=0 pass=0' "$OUT/negative.log"
# Profile stdout can interleave with the simulator's stderr tohost message.
# The TestHarness assertion reports the same HTIF exit code on a complete line.
grep -Eq '\*\*\* FAILED \*\*\* \((tohost|exit code) = *1\)' "$OUT/negative.log"
echo "BF16 GEMM Rocket/RoCC RTL PASS, including HTIF failure-exit check. Logs: $OUT"
