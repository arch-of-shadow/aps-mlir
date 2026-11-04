# APS tests

Run commands from the repository root in the Pixi environment.

| Directory | Coverage |
|---|---|
| [frontend/](frontend) | CADL parser, MLIR emission, types, and C transpilation; parser cases are under `parser/` |
| [megg/](megg) | Matching, CSR operands, reconstruction, and compiler failure handling |
| [synthesis/](synthesis) | APS/TOR passes, scheduling, storage, memory, and RTL/co-simulation regressions |
| [hardfloat/](hardfloat) | Floating frontend, numerical oracles, library import, and acceptance workflows |
| [scaleisa_adapter/](scaleisa_adapter) | Candidate conversion, instruction splitting, hardware ABI, and correctness runners |
| [scaleisa_compiler/](scaleisa_compiler) | CI software lowering, RV32 ABI, and 2mm execution runners |
| [scaleisa_driver/](scaleisa_driver) | Integrated driver behavior and failure cases |

## Select checks

Collect all regressions without running them:

```sh
pixi run python -m pytest tests --collect-only -q
```

Run frontend checks:

```sh
pixi run python -m pytest tests/frontend -q
```

The `check-aps-scaleisa`, `check-scaleisa-adapter`, and `check-scaleisa-compiler`
CMake targets build their tools and run the corresponding unittest suites.
Other compiler/RTL tests use existing tools, and some invoke synthesis or simulator
compilation. Select suites according to the behavior being changed.

Files named `test_*.py` are collected regressions. The `run_*.py` scripts are
explicit workflows; inspect their `--help` and component guide for required inputs.
Keep source fixtures beside their suite, generated outputs under `build/`, and
shared frontend IR assertions in `frontend/mlir_test_utils.py`.

See the [synthesis](../docs/SYNTHESIS/README.md),
[compiler](../docs/COMPILER/README.md), and
[co-simulation](../docs/SYNTHESIS/aps-spike-verilator-cosim.md) guides for setup.
