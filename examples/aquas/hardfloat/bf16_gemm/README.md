# BF16 GEMM on Rocket + RoCC

[gemm.cadl](gemm.cadl) implements tiled 8x8 BF16 GEMM with partitioned
scratchpads, loop unrolling, and RoCC DMA. The runner generates hardware,
compiles the RISC-V validation program, and executes it on Chipyard's
`APSRocketConfig` RTL.

## Run

Build the repository tools and initialize Chipyard, then run:

```sh
pixi run bash examples/aquas/hardfloat/bf16_gemm/run.sh
DEBUG=1 pixi run bash examples/aquas/hardfloat/bf16_gemm/run.sh
BENCHMARK=1 pixi run bash examples/aquas/hardfloat/bf16_gemm/run.sh
PROFILE=1 BENCHMARK=1 pixi run bash examples/aquas/hardfloat/bf16_gemm/run.sh
```

The default output directory is `build/chipyard-bf16-gemm/`. `OUT` overrides
it; generated MLIR, RTL, headers, binaries, logs, and summaries stay there.
`JOBS` controls simulator parallelism and `BUILD_JOBS` controls the compiler
build. `DEBUG=1` selects Chipyard's debug simulator. `LOOP_PIPELINE=0` generates
a serial variant with the same arithmetic, partitioning, and DMA.

The run requires the APS/CIRCT tools, packaged HardFloat IP, Chipyard's RV32
GCC/libgloss-HTIF, Verilator, and Java. Run through Pixi to activate the tools.

## Files

- `gemm.cadl`: accelerator source.
- `run.sh`: hardware generation, RV32 compilation, and Rocket simulation.
- `src/`: C validation programs and software references.
- `tools.py`: vector generation, passive RTL profiling, and result summaries.

## Arithmetic and memory layout

Each of `matA`, `matB`, and `matC` contains 131072 BF16 elements with eight-way
cyclic partitioning. Tiles contain 64 elements. B is transposed within each
tile. The flattened output loop unrolls two columns, and K unrolls eight terms.

BF16 multiplication and addition round separately to nearest-even; the
accelerator does not contract them into FMA. `use_c=0` starts with zero and
`use_c=1` adds the existing C tile. Native FP32 software uses its own rounding
contract and independent golden results.

All commands use `funct3=7` and return zero after completion:

| Opcode | funct7 | Operation | rs1 | rs2 |
|---|---|---|---|---|
| `0x0b` | 0 | GEMM | `(offsetA<<16) \| (use_c<<15) \| offsetB` | `(offsetC<<16) \| offsetD` |
| `0x7b` | 0 | Load A tiles | DRAM address | `(spm_element_offset<<16) \| row_pitch_bytes` |
| `0x7b` | 1 | Load transposed B tiles | DRAM address | same |
| `0x7b` | 2 | Store C/D tiles | DRAM address | same |

GEMM offsets count tiles; DMA offsets count elements. Each DMA command transfers
eight rows of 64 BF16 elements. DRAM addresses and row pitches must align to
128 bytes. All CPU/accelerator data movement uses DMA.

## Validation and measurement

`tools.py vectors` creates exact BF16 goldens using
the independent IEEE oracle under `tests/hardfloat/`. [test_gemm.c](src/test_gemm.c)
checks 32 computed tiles, 2048 output elements, nonzero scratchpad offsets,
padded DRAM pitches, in-place results, and DMA guards. A second executable
corrupts a golden value and must fail numerically.

`BENCHMARK=1` compares BF16 acceleration with native FP32 software on the same
Rocket core. `PROFILE=1` adds passive RTL counters. The shared `tools.py` generates
headers and instrumentation, and summarizes simulator logs into `benchmark.json`
and `profile.json` in the output directory. Measurements cover batches of 8x8
tiles; they do not establish large-matrix GEMM performance.
