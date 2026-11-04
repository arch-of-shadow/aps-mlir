# Compiler workflows

APS has two software entry points. Choose one based on where instruction
semantics come from; both must agree with the generated hardware encoding and ABI.

| Input and purpose | Entry point |
|---|---|
| Discover and select instructions from application MLIR, then generate hardware and software | [aps-scaleisa-opt](aps-scaleisa-opt.md) |
| Match a supplied CADL instruction description against C code | `tools/aps-megg compile` |

## ScaleISA workflow

The C++ driver orchestrates discovery/selection, adapts the selected implementation
library, synthesizes RTL, and compiles exposed custom-instruction (CI) sites.
It can link a supplied runtime and execute the matching RTL in co-simulation.

```sh
pixi run build
pixi run build/tools/aps-scaleisa-opt/aps-scaleisa-opt \
  scaleisa/benchmarks/pcl/octree_search/octree_search.mlir \
  --area-budget 500 --output-dir build/scaleisa-octree
```

The driver uses the Rust release engine produced by `pixi run build`.
This example starts a full discovery run and may take time.
A valid zero-CI selection produces software without accelerator RTL.

- [Driver](aps-scaleisa-opt.md): invocation, stages, reuse, and artifacts.
- [Adapter](scaleisa-adapter.md): selected kernels to APS hardware contracts.
- [Software compiler](scaleisa-compiler.md): selected call sites to RV32 binaries.
- [Benchmarks](aps-scaleisa-opt.md#benchmark-reproduction-and-evidence): reproduction and measurement scope.

Algorithm and cost-model documentation belongs to
[ScaleISA](../../scaleisa/docs/README.md).

## CADL/Megg workflow

```sh
mkdir -p build/megg-example
pixi run compile examples/aspdac26-tutorial/cadl/v3ddist_vv.cadl examples/aspdac26-tutorial/csrc/test_v3ddist_vv.c build/megg-example/v3ddist_vv.riscv
```

The example uses the same CADL/C pair as `examples/aspdac26-tutorial/a2-ex1compile.sh`. It requires the
Polygeist `cgeist` frontend, APS Python bindings, Megg's compiled `mlir_utils`
extension, and the RISC-V toolchain supplied/configured during setup.

[tools/aps-megg](../../tools/aps-megg) converts CADL to a C pattern, converts it
through `cgeist`, extracts instruction encodings and CSR inputs, and invokes
[E2ECompiler](../../megg/megg/e2e_compiler.py). Megg matches application computation
using e-graph rewrites and emits custom instructions through its LLVM backend.
`--verbose` exposes diagnostics; `--handson` enables tutorial snapshots.
Artifacts are collected in `compile_logs/` beside the executable, including
pattern C/MLIR, encoding JSON, and available assembly/statistics/snapshots.

CSR argument placement must preserve source operand positions. Use
[CSR transport tests](../../tests/megg/test_csr.py) and
[robustness tests](../../tests/megg/test_csr_robustness.py) when changing matching
or emission. Compile the CADL to hardware separately through the
[synthesis flow](../SYNTHESIS/README.md).

Megg internals are documented in [matching and e-graph compilation](pattern-matching.md),
including rewrites, skeleton matching, extraction, and code generation.

## Checks

```sh
pixi run cmake --build build --target check-aps-scaleisa check-scaleisa-adapter check-scaleisa-compiler
pixi run python -m pytest tests/megg/test_csr.py tests/megg/test_cleanup.py -q
```

The CMake targets run driver/pass regressions. Full application correctness and
Rocket cycle measurement are separate benchmark campaigns.
