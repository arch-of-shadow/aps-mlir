# ScaleISA / APS benchmark comparison

The 16-case manifest is
[`tests/aps-scaleisa/benchmark_suite16.json`](../../tests/aps-scaleisa/benchmark_suite16.json).
It contains distinct repository kernels, including the earlier application runs.
Input sizes are recorded in each case's provenance. PCL and GTSAM cases use the
repository's standalone snapshots; they are not timings of the upstream frameworks.
Bitcount and squared distance are extracted kernel microbenchmarks with
deterministic inputs and independent C++ references.
Pose2 and Pose3 time their original solver kernels and bounded input staging;
their graph construction runs in C++ before the timed region. Every solver
invocation starts from the same prepared input buffers.

The completed 16-kernel results and exact measured artifacts are archived in
[`reports/aps-scaleisa-16/report.md`](../../reports/aps-scaleisa-16/report.md).
The bundle's `validation.json` records artifact hash, MLIR parsing and link checks.

The linked review bundle is regenerated with:

```bash
.pixi/envs/default/bin/python tests/aps-scaleisa/collect_benchmark_report.py
# To refresh the archived bundle:
.pixi/envs/default/bin/python tests/aps-scaleisa/collect_benchmark_report.py \
  --output-dir reports/aps-scaleisa-16
```

Open [`build/aps-suite-16/review/report.md`](../../build/aps-suite-16/review/report.md)
for the table, and `summary.json` in that directory for raw model and CPU cycles,
area settings, scheduling periods and file hashes. The collector requires passed
native and Chipyard runs, verifies the measured RTL and binary hashes, and checks
that the measured run's estimate matches the selected implementation report.
`--allow-incomplete` produces an explicitly incomplete progress report.

Measured speedup divides warm software baseline cycles by warm accelerated
program cycles on full Rocket RTL. Chipyard runs use `APSRocketConfig`,
`run-binary-debug`, `LOADMEM=1`, an explicit `BINARY=...riscv`, and the selected
RTL/CSR configuration through `APS_CONFIG`. Validation, reference generation and
printing run outside the timed region. Cold measurements are also retained.
ScaleISA's estimate divides its software model cycles by its selected-program
model cycles. Scheduling period is an HLS input; these cycle ratios do not claim
a post-layout accelerator frequency or frequency-adjusted wall-clock speedup.

Every case directory contains:

- `index.md`: all selected AU symbols, their implementation inputs and their
  Hector evaluated schedules.
- `scaleisa-au.mlir`: the exact collected selected definitions supplied to the
  APS adapter, including proven alias attributes.
- `aps-planned.mlir` and `aps-hardware.mlir`: adapted computation and the final
  hardware functions, including register/CSR bindings and memory transfers.
- `selected-program.mlir`, `cpu-ci.mlir`, and `baseline-program.mlir`: selected
  call sites, actual CPU CI lowering, and the software baseline.
- `manifest.json`, target profile, pipeline report and Chipyard result/logs.

Files are copied without rewriting MLIR. The manifest maps copies to their
original source paths and SHA-256 hashes. AU indices include every selected
implementation rather than one representative CI per application.

To independently audit numerical validation on the measured Rocket hardware:

```bash
.pixi/envs/default/bin/python tests/aps-scaleisa/audit_result_correctness.py
```

This replays the accelerated programs with executed element-comparison counts
printed by the original software-reference checkers. A second binary flips one
bit of an actual output before comparison; the checker must report a mismatch
and the simulated test must fail. Integer values and floating-point bit patterns
must match exactly. The audit writes `result-correctness.json` and `correctness.md`
to the archived bundle, including per-case harnesses and simulator logs.
The evidence applies to the benchmark inputs, rather than all possible inputs.

Additional PCL/GTSAM snapshots can be run with:

```bash
.pixi/envs/default/bin/python tests/aps-scaleisa/run_snapshot_case.py ransac \
  --output-dir build/aps-suite-16/ransac
.pixi/envs/default/bin/python tests/aps-scaleisa/run_chipyard_pcl.py \
  --source-dir build/aps-suite-16/ransac \
  --output-dir build/aps-suite-16/ransac-chipyard
```

The snapshot runner captures every entry array from the original C++ reference
and checks the selected program bit for bit. It changes only explicitly listed
dimensions in matching C++ and MLIR inputs, and adds output observers after
selection. `--discovery-dir` reuses a matching completed selection when rebuilding
downstream stages. `--prepare-only` generates inspectable inputs and provenance.
