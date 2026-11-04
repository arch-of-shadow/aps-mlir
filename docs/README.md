# APS documentation

APS generates RISC-V instruction extensions and the software that uses them.
The supported hardware path is CADL or adapted ScaleISA MLIR → APS/SCF/Affine →
scheduled TOR → CMT2 → FIRRTL → SystemVerilog.

## Start here

| Task | Guide |
|---|---|
| Build APS and synthesize a first instruction | [Hardware synthesis](SYNTHESIS/README.md) |
| Discover instructions or compile against a CADL description | [Compiler workflows](COMPILER/README.md) |
| Package hardware for Rocket and run full-system simulation | [Processor integration](ARCHITECTURE/README.md) |

Run commands from the repository root unless a command explicitly changes directories.
APS binaries are under `build/tools/`; CIRCT binaries are under `circt/build/bin/`.
The Pixi environment supplies Python bindings, libraries, and tool paths.

## Topic index

| Synthesis | Compiler | Integration |
|---|---|---|
| [CADL language](SYNTHESIS/cadl-language.md) | [APS ScaleISA driver](COMPILER/aps-scaleisa-opt.md) | [Memory interfaces and DMA](ARCHITECTURE/aps-itfc.md) |
| [End-to-end pipeline](SYNTHESIS/aps-e2e-flow.md) | [ScaleISA adapter](COMPILER/scaleisa-adapter.md) | [Ports, CSRs, and completion](ARCHITECTURE/aps-port-convention.md) |
| [HLS and scheduling](SYNTHESIS/hls-passes.md) | [ScaleISA software compiler](COMPILER/scaleisa-compiler.md) | |
| [CMT2 lowering](SYNTHESIS/aps-to-cmt2-transfer-model.md) | [Benchmark reproduction](COMPILER/aps-scaleisa-opt.md#benchmark-reproduction-and-evidence) | |
| [BF16/FP32 hardware](SYNTHESIS/hardfloat-bf16-fp32-acceptance.md) | | |
| [Spike + Verilator](SYNTHESIS/aps-spike-verilator-cosim.md) | [Megg matching and e-graphs](COMPILER/pattern-matching.md) | |

## Implementation references

Each guide links to its implementing code and relevant regression tests. Use those
contracts when changing behavior; CLI defaults belong to the tool that owns them.
The [ScaleISA documentation](../scaleisa/docs/README.md) covers discovery,
anti-unification, extraction, and its cost model. The
[CMT2 concepts guide](../circt/docs/Cmt2/features/Concepts.md) covers dialect semantics.
Benchmark evidence lives in [reports/](../reports/), with reproduction instructions
linked above.

This index covers current usage and implementation. Historical refactor notes,
superseded transfer designs, and the Aquas-IR mapping proposal are available in
Git history. Their useful implementation details have been consolidated into the
pipeline, HLS, CMT2, and integration guides.

## Consolidation map

| Previous material | Current location |
|---|---|
| FIFO flow and scope-FIFO notes | CMT2 lowering |
| SCF/affine debugging notes and standalone scheduling page | HLS transformations and scheduling |
| Frontend refactor and control-flow-if notes | CADL language and frontend |
| Megg e-graph architecture and matching | Megg matching and e-graph compilation |
| CSR integration notes | Generated ports, CSRs, and completion |
| Standalone DMA guide | Memory interfaces and DMA |
| Deployment guide | Processor integration |
| Standalone benchmark report guide | Benchmark section of the ScaleISA driver guide |
| Aquas-IR mapping proposal | Git history; no active implementation guide |
