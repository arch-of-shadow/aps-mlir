# APS ScaleISA driver

[tools/aps-scaleisa-opt/main.cpp](../../tools/aps-scaleisa-opt/main.cpp) owns
full-pipeline orchestration and compile-only mode. It invokes dedicated adapter
and CI-lowering tools; their passes are not registered in `aps-opt`.

## Run and inspect

After the [compiler build](README.md):

```sh
pixi run build/tools/aps-scaleisa-opt/aps-scaleisa-opt input.mlir \
  --area-budget 500 --output-dir build/selected-run
```

`input.mlir` is an application module suitable for ScaleISA discovery. Defaults
are `linalg,affine,scf` discovery, greedy selection, five extra CSR operands,
RV32 soft-float `ilp32`, and object emission. Supported selection methods are
`greedy`, `cbc`, and `eboost`; solver prerequisites depend on the selected method.

| Stage | Inspect |
|---|---|
| Target/resource preparation | Target profile and scheduling resources in the output directory |
| ScaleISA discovery and selection | Discovery stage report, final selected program, selection report |
| Candidate collection and adapter | `candidates.mlir`, `planned.mlir`, `hardware.mlir` |
| APS synthesis | `design.scheduled-tor.mlir`, `design.cmt2.mlir`, `design.sv` |
| Software generation | `selected.mlir`, `program.o` or `program.elf`, compilation intermediates |
| Optional simulation | Matching simulator, simulation log, and runtime statistics |

`report.json` records stage commands, durations, status, selection metrics,
artifact hashes, and backend pipeline outcomes. Each subprocess has a log;
`--verbose` streams logs and `--timeout` sets the per-stage timeout in seconds
(default 3600). Failed stages stop execution and record diagnostics.

A selection with no CIs takes the software-only branch: an empty hardware module
is emitted, synthesis is skipped, and `hardware_selected` is false. With `--cosim`,
that branch runs the software ELF in Spike and reports RTL co-simulation as
inapplicable. Check the report before assuming an accelerator was generated.

## Target and reuse

`--target-profile` supplies the realized CPU/hardware/interface model. Without
one, the driver prepares a profile using ScaleISA resources; physical estimates
use its ASAP7 table. `--resource` supplies APS execution contracts, and `--clock`
overrides the scheduling period (otherwise read from the hardware profile).
These modeled values do not establish post-layout timing or area.

`--discovery-stages=scf` restricts discovery. `--discovery-dir=PATH` reuses a
completed run only when its input path, target profile hash, area budget,
selection method, and discovery stages match. Reevaluate implementations when
changing the target profile.

Selected loop pragmas are preserved during adaptation. The report's
`backend_pipeline_outcomes` records requested/achieved II and sequential fallback
reasons where emitted by the backend. Inspect those outcomes when comparing
selected estimates with realized hardware.

## Linking and execution

Use `--emit=elf --entry=_start` with repeated `--link-object`, `--link-source`,
and `--link-arg` for an executable. Supply startup/runtime code and a linker
script for the intended platform. Add `--cosim` to build and execute the matching
Spike + Verilator model; object output cannot be simulated.

A complete checked-in scalar campaign supplies independent references and runtime
inputs:

```sh
pixi run python tests/aps-scaleisa/run_scalar_cases.py --cases bitcount
```

For CPU floating-point code, select compatible features and ABI, for example
`--mattr=+m,+a,+f,+zicsr,+zifencei --abi=ilp32f`. This CPU target is separate from
the accelerator's HardFloat implementation.

## Compile-only and validation hooks

```sh
pixi run build/tools/aps-scaleisa-opt/aps-scaleisa-opt \
  --compile-only selected.mlir --hardware hardware.mlir -o build/program.o
```

This consumes a matching materialized hardware contract without repeating
selection or RTL generation. Zero recognized CI sites require `--allow-no-ci`.
See [software compilation](scaleisa-compiler.md).

`--rename-main=NAME` renames the selected entry. `--observe-function=NAME` inserts
a callback before its returns, with buffers selected by repeated `--observe-type`
or repeated `--observe-allocation`. Types must uniquely identify entry-block
`memref.alloca` operations; indices are zero-based in that entry allocation list.
The two selection forms are mutually exclusive. Supply the callback implementation
when linking. Instrumentation follows selection, keeping reference validation
outside discovery.

Tests: [test_driver.py](../../tests/aps-scaleisa/test_driver.py).
Numerical campaigns and full Rocket measurements are indexed in the
[benchmark section](#benchmark-reproduction-and-evidence).

## Benchmark reproduction and evidence

The [archived 16-kernel report](../../reports/aps-scaleisa-16/report.md) contains
measured results and exact artifacts. Use its manifests and validation records
for specific numerical claims. Input sizes, provenance, harnesses, and timed
regions define each measurement's scope.

### Reproduce campaigns

After [building the compiler tools](README.md) and preparing the RISC-V/Spike
runtime, a small source-derived campaign is:

```sh
pixi run python tests/aps-scaleisa/run_scalar_cases.py --cases bitcount sq_dist
```

The runner preserves extracted kernel computation, adds deterministic inputs and
independent C++ references, and checks accelerated and software-only execution.
These are kernel microbenchmarks. They do not time complete upstream applications.
The default output is `build/aps-scaleisa-scalars/`, including `summary.json`.

PCL/GTSAM snapshot preparation and full Rocket measurement are separate commands:

```sh
pixi run python tests/aps-scaleisa/run_snapshot_case.py ransac \
  --output-dir build/aps-suite-16/ransac
pixi run python tests/aps-scaleisa/run_chipyard_pcl.py \
  --source-dir build/aps-suite-16/ransac \
  --output-dir build/aps-suite-16/ransac-chipyard
```

Rocket runs require initialized Chipyard and its simulator toolchain; see
[integration](../ARCHITECTURE/README.md). Review each runner's `--help` for
reuse and preparation options before launching a long campaign.

### Collect and audit

```sh
pixi run python tests/aps-scaleisa/collect_benchmark_report.py
pixi run python tests/aps-scaleisa/audit_result_correctness.py
```

The collector checks completed native/Chipyard runs, artifact hashes, selected
implementation estimates, parsing, and links. Its default review output is under
`build/aps-suite-16/review/`. `--allow-incomplete` explicitly creates an incomplete
progress report. To replace the archived bundle intentionally, pass
`--output-dir reports/aps-scaleisa-16`.

The correctness audit replays accelerated binaries with executed comparison
counts and corrupts an actual output in a second run, which must fail validation.
It writes correctness evidence into the archived bundle. Numerical checks apply
to the supplied benchmark inputs.

### Interpret artifacts

- **Modeled speedup:** software model cycles divided by selected model cycles.
- **Measured processor speedup:** warm baseline Rocket cycles divided by warm
  accelerated-program cycles; cold samples are retained separately.
- **CI latency:** measured accelerator invocation cycles, separate from complete
  program cost and processor cache behavior.
- **Physical timing/area:** requires technology-specific implementation reports;
  an HLS clock setting or selection area estimate is not such a measurement.

Reference generation, validation, and printing run outside the timed region.
Standalone PCL/GTSAM snapshots measure the repository inputs, not the complete
upstream frameworks. Compare runs only with matching input provenance, target
profiles, timing scope, and artifact hashes.

Archived cases include selected AU definitions, adapted APS MLIR, actual CPU CI
lowering, baseline software, target profile, and measured binary/RTL provenance.
Use [collect_benchmark_report.py](../../tests/aps-scaleisa/collect_benchmark_report.py)
and [audit_result_correctness.py](../../tests/aps-scaleisa/audit_result_correctness.py)
for the enforced evidence contracts.
