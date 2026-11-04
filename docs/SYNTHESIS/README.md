# Hardware synthesis

The CADL frontend emits functional MLIR. `aps-e2e` owns normalization, memory
binding, TOR scheduling, CMT2 generation, and FIRRTL/SystemVerilog export.
`aps-opt` exposes individual passes for development.

## Build

Use Linux with Pixi installed. From the repository root:

```sh
pixi run build
```

The task runs [setup](../../scripts/setup.sh), configures and builds APS through
[scripts/build.sh](../../scripts/build.sh), and builds the ScaleISA Rust engine
in release mode. Setup builds the repository's CIRCT fork and Python bindings;
expect an initial toolchain build to take substantially longer than incremental builds.
The [Pixi manifest](../../pixi.toml) defines dependencies and activation paths.
Full Rocket simulation additionally needs `pixi run setup-chipyard`; it is not
required for the small RTL export below.

After setup, rebuild specific C++ tools with:

```sh
pixi run cmake --build build --target aps-opt aps-e2e -j 3
```

## First instruction

```sh
mkdir -p build/docs-smoke
pixi run mlir examples/aspdac26-tutorial/cadl/hello.cadl build/docs-smoke/hello.aps.mlir
pixi run build/tools/aps-opt/aps-e2e \
  -i build/docs-smoke/hello.aps.mlir \
  --export sv -o build/docs-smoke/hello.sv \
  --dump-scheduled-tor build/docs-smoke/hello.tor.mlir \
  --dump-cmt2 build/docs-smoke/hello.cmt2.mlir
```

The outputs are frontend MLIR, scheduled TOR, CMT2, and a SystemVerilog design
with a `main` accelerator module. Hardware generation does not compile a CPU
program; use the [compiler workflows](../COMPILER/README.md) for that step.

`pixi run syn INPUT OUTPUT` is the platform-oriented shortcut: it selects a 4 ns
scheduling period, applies `examples/interface.json`, exports SV, and writes
Chipyard configuration. For isolated experiments, invoke `aps-e2e` directly;
its default period is 6 ns and its default export is CMT2.

## Choose the next guide

- [CADL](cadl-language.md): syntax, control flow, and frontend limits.
- [Pipeline](aps-e2e-flow.md): stages, output selection, and diagnostics.
- [HLS](hls-passes.md): affine conversion, partitioning, scheduling, and storage.
- [CMT2 lowering](aps-to-cmt2-transfer-model.md): tokens, values, and loop execution.
- [Floating point](hardfloat-bf16-fp32-acceptance.md): resource generation and acceptance.
- [Co-simulation](aps-spike-verilator-cosim.md): execute software with generated RTL.

## Development checks

```sh
pixi run python -m pytest tests/frontend/parser tests/frontend/test_mlir_conversion.py -q
```

Tests for specific transformations are linked from the HLS and CMT2 guides.
Frontend entry points are [tools/aps-frontend](../../tools/aps-frontend) and
[cadl_frontend/to_mlir/](../../cadl_frontend/to_mlir). Build targets are defined in
[tools/aps-opt/CMakeLists.txt](../../tools/aps-opt/CMakeLists.txt).
