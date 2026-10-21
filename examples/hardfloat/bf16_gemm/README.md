# Partitioned BF16 GEMM on Rocket + RoCC RTL

This is the BF16 version of `examples/features/gemm.cadl`. It keeps the original
8×8 tile algorithm, packed offsets, transposed B layout, array partitioning,
loop unrolling, and tiled DMA helpers.

```sh
pixi run bf16-gemm-rocket
# Same debug target as thirdparty/chipyard/sims/verilator/run.sh:
DEBUG=1 pixi run bf16-gemm-rocket
```

The script compiles CADL through APS/TOR → CMT2 → FIRRTL → SystemVerilog,
cross-compiles the C test into `gemm.riscv` (RV32 ELF), and runs the whole
program in Chipyard's `APSRocketConfig` Verilator TestHarness. Rocket fills DRAM
buffers, invokes RoCC DMA and GEMM instructions, reads DMA results back from
DRAM, checks exact BF16 encodings, and exits through HTIF.

## Preserved structure and BF16 arithmetic

- `matA`, `matB`, `matC`: **131072 BF16 elements each**, 2048 tiles, 256 KiB per
  array. All three retain 8-way cyclic partitioning on dimension 0.
- GEMM: one flattened loop over 64 outputs, with unroll factor 2 and pipeline
  enabled: **32 continuous output-pair iterations per tile**. Row and column
  are `output / 8` and `output % 8`; K retains unroll factor 8. The requested
  II=1 is scheduled at II=9. There is no per-row pipeline exit/re-entry.
- DMA: all three helpers retain their eight-way unrolled row loop and
  `stride_y(8)`. Each row transfers 64 **elements**, now 128 bytes. A/B's
  `stride_x` changes from 8 to 16 bytes for BF16; C's remains 16 bytes.
- B is stored transposed inside each tile: `matB[offsetB*64 + j*8 + k]`.
- Each `zacc + a_val * b_val` performs a BF16 multiply followed by a BF16 add,
  rounding each operation to nearest-even. The accumulator starts at BF16 +0.
  The final `zacc + c_val` also rounds to BF16. There is no FMA contraction or
  reassociation. `use_c=0` supplies +0; `use_c=1` adds the existing C tile.

The generated CMT2 currently contains 24 memory banks (16384 × 16 bits each),
16 BF16 multiply instances and 18 BF16 add instances, confirming that the
partitioning and two-column/eight-term unrolling survive synthesis.

Besides changing element types/literals and the required DMA byte stride,
this version clears `use_c` (bit 15) from `offsetB`. The original example's
`0xffff` mask includes that flag in the B tile address; `0x7fff` lets the
optional-C path use the intended B tile.

## RoCC ABI

All commands use `funct3=7` and return zero after completion.

| Opcode | funct7 | Operation | rs1 | rs2 |
|---|---|---|---|---|
| `0x0b` | 0 | `D = A*B + optional C` | `(offsetA<<16) \| (use_c<<15) \| offsetB` | `(offsetC<<16) \| offsetD` |
| `0x7b` | 0 | DMA load eight A tiles | DRAM address | `(spm_element_offset<<16) \| row_pitch_bytes` |
| `0x7b` | 1 | DMA load eight transposed B tiles | DRAM address | same |
| `0x7b` | 2 | DMA store eight C/D tiles | DRAM address | same |

GEMM offsets are tile indices; DMA offsets are element indices. A tile contains
64 elements. The packed DMA starting offset has 16 bits. DRAM addresses and
row pitches must be aligned to 128 bytes for these bursts.

DRAM DMA layout consists of eight rows. Each row contains eight consecutive
8-element tile rows (64 BF16 elements); a configurable row pitch permits
padding. DMA scatters/gathers those rows into/from eight contiguous tiles in
the partitioned scratchpads. All input and output movement uses DMA; no scalar
scratchpad load/read commands exist in this example.

## C test and reference results

`generate_vectors.py` embeds 16 input pairs and exact goldens in `vectors.h`.
It uses the independent integer/Fraction IEEE oracle, with separately rounded
BF16 multiply/add operations matching the CADL source. Cases cover zero,
identity, cancellation, rounding, subnormals, overflow, Inf/NaN, and random
finite/full-bit-pattern inputs.

`test_gemm.c` processes two batches of eight tiles, with two GEMM passes each:

1. DMA load A and transposed B, compute with `use_c=0`, DMA store, check results.
2. Recompute with `use_c=1`, adding the first result; DMA store and check again.

The test checks **32 computed tiles / 2048 output elements**, nonzero/distinct
SPM offsets, both 128-byte and padded 256-byte DRAM row pitches, separate C/D
and in-place C/D, and untouched DMA guards/padding. CPU memory fences order
buffer writes, DMA commands and result reads. A second ELF compiled with
`GEMM_FORCE_FAIL` corrupts one expected output; the script requires a numerical
mismatch and `*** FAILED *** (tohost = 1)` from the full-system simulator.

The flattened-loop full Rocket/RoCC RTL positive run in
`build/chipyard-bf16-flat/run.log` produced:

```text
BF16 GEMM batch=0 pass=0: 8 tiles + DMA guards PASS
BF16 GEMM batch=0 pass=1: 8 tiles + DMA guards PASS
BF16 GEMM batch=1 pass=0: 8 tiles + DMA guards PASS
BF16 GEMM batch=1 pass=1: 8 tiles + DMA guards PASS
BF16 GEMM PASS tiles=32 elements=2048 DMA=8tiles partition=8 unroll=2/8
```

The negative run also passed its expected-failure check:

```text
BF16 GEMM FAIL: batch=0 pass=0 row=0 col=0 expected=0001 got=0000
*** FAILED *** (tohost = 1)
```

## Build and logs

Use the repository's existing Chipyard toolchain setup:

```sh
pixi run setup-chipyard
# If Chipyard is already initialized, install/rebuild only the extra tools:
# pixi run setup-chipyard-riscv-extra
```

That flow builds upstream libgloss with the existing Chipyard RV32 patch.
GEMM links the installed `htif_nano.specs` and `htif.ld`, just like
`compile-native` and Chipyard's C tests; no separate libgloss fork or
GEMM-specific toolchain setup is required.

Requires the configured APS/CIRCT build plus Chipyard's RV32 GCC/libgloss-HTIF,
Spike/libfesvr, Verilator and Java. The default is `run-binary-fast` with
`CONFIG=APSRocketConfig` and `LOADMEM=1`; `DEBUG=1` selects `run-binary-debug`.
An isolated `APS_CONFIG` YAML points to the generated GEMM RTL.

Artifacts are under `build/chipyard-bf16-gemm/`: `gemm.riscv`, `gemm_fail.riscv`,
`gemm.asm`, `vectors.h`, `input.mlir`, `gemm.cmt2.mlir`, `gemm.sv`,
`generated-src/`, `simulator`, `build.log`, `run.log`, and `negative.log`.
`OUT` overrides the output directory; `JOBS` defaults to 16 and `BUILD_JOBS` to 3.

## Native FP32 software versus BF16 ISAX cycle benchmark

```sh
BENCHMARK=1 pixi run bf16-gemm-rocket
# Also bind passive counters into the accelerator inside the full Rocket SoC:
PROFILE=1 BENCHMARK=1 OUT=build/chipyard-bf16-profile pixi run bf16-gemm-rocket
python examples/hardfloat/bf16_gemm/summarize_profile.py \
    build/chipyard-bf16-profile/benchmark.log build/chipyard-bf16-profile/profile.json
```

`benchmark.c` runs both implementations in one RV32F ELF on the same
`APSRocketConfig` RTL using Chipyard's TestHarness, HTIF and `run-binary-fast`.
`fp32_software.c` is a pure C implementation with FP32 inputs, accumulation and
outputs. Rocket has a native FP32 FPU, not a BF16 arithmetic instruction set.
The baseline uses two column accumulators and an ordinary eight-iteration K
loop, with no unroll pragma. Loop unrolling is left to the compiler. It is
compiled separately with `-O3 -fno-fast-math -ffp-contract=fast`, without LTO.
Disassembly must contain native `fmadd.s`; no BF16 rounding is performed in the
software kernel. Input widening happens before its timed interval.

The earlier baseline explicitly requested `#pragma GCC unroll 8`; this has
been removed. With the current toolchain and flags, the software function's
disassembly is identical because GCC still unrolls the fixed-size loop.
A fresh full Rocket/RoCC RTL run in `build/chipyard-bf16-flat-no-pragma/`
passed all 2048 benchmark comparisons and reproduced 13202/12741 software
cycles and the **2.174×** aggregate end-to-end speedup. Removing the pragma
does not disable automatic compiler unrolling. The two column accumulators
remain explicit in the C source.

Both engines compute `D=A*B` for the same **eight independent dense finite 8×8
tiles**, with 512 outputs and 4096 MACs per engine per trial. This is not a large
matrix GEMM. Both use the same element layout and transposed B; FP32 buffers
have twice the byte size of BF16 buffers. Software results are checked bitwise
against independent FP32 FMA goldens, and ISAX results against independent BF16
multiply/add goldens. The precisions and rounding semantics intentionally differ;
this comparison measures native FP32 software against BF16 hardware, not numerical
equivalence between them.

Two trials reverse engine order (software, ISAX, ISAX, software). Input setup,
conversion, transposition, checks and printing are outside timing. No cache flush
is performed: this is a prepared-input workload on the ordinary Rocket cache
hierarchy, not a forced cold-DRAM benchmark. Timing uses `mcycle`, fences and an
explicit dependency on each RoCC response. Software includes its normal loads
and stores. ISAX separately measures A/B DMA, eight SPM-resident compute commands,
and output DMA. End-to-end speedup divides software cycles by the sum of all
three ISAX regions. Values include command, function-call and counter overhead;
clock frequency is assumed equal. Host simulation wall time is not a performance
metric.

`benchmark.log` records 2048 correctness comparisons over two engines and two
trials; `benchmark.json` records ratios of summed cycles. `gemm_bench.riscv` and
`gemm_bench.asm` preserve the executable and its disassembly. With `PROFILE=1`,
`generate_profile.py` creates simulation-only SV bind counters observing actual
RoCC command/response handshakes, iteration admission/completion, and all 34 FP
units' issue/collect handshakes. It does not drive any accelerator input.
`summarize_profile.py` validates 16 commands, 512 output-pair iterations and
512 transactions per FP unit, then saves `profile.json`. Issue-slot utilization
counts accepted issues divided by unit count times kernel cycles; it is not
power/activity utilization. The normal correctness/negative-test task is unchanged.

### Continuous output-pair pipeline

The output traversal is flattened into a single loop, preserving `unroll(2)`,
the eight-way K unroll, array partitioning, ordered BF16 operations and DMA.
All 32 output pairs belong to the same pipeline invocation; it drains only at
the end of the tile. This replaces the previous nested row/column traversal,
which drained after every four output pairs.

Use the default full-system validation, or select the serial reference without
changing the example source:

```sh
OUT="$PWD/build/chipyard-bf16-flat" PROFILE=1 pixi run bf16-gemm-rocket
OUT="$PWD/build/chipyard-bf16-flat" PROFILE=1 BENCHMARK=1 pixi run bf16-gemm-rocket
LOOP_PIPELINE=0 OUT="$PWD/build/chipyard-bf16-serial-reference" pixi run bf16-gemm-rocket
```

`LOOP_PIPELINE=0` writes `gemm.serial.cadl` in the output directory. The normal
flow performs CMT2 inline/call-sequence verification before building Chipyard.
The recorded pipelined run performed that same verification separately in
parallel (`gemm.inlined.mlir`, `inline.log`) and passed it. `VERIFY_CMT2=0`
only skips that separate sweep and should not be treated as a verification pass.

The scheduler may increase the requested II to satisfy resource/dependency
constraints; the generated controller enforces that achieved II at actual
body admission, including after stalls.

To retain bank partitioning after flattening, affine index recognition handles
constant unsigned division/remainder of a loop variable only when its bounds
prove a nonnegative, signed-representable range. Bank selection also drops terms
whose entire coefficient is divisible by the bank factor, including terms such
as `(output floordiv 8) * 8`. These divisions by eight synthesize as bit selection.

The compiler gives these overlapping iterations explicit memory semantics:

- Dynamic bank stores use `aps.write_smem_if`. Only the selected bank writes;
  there are no preservation reads or stale-value writes to unselected banks.
  Bank readiness remains conservative, even when a predicate is false, because
  CMT2 call arguments are driven under the caller's fire signal.
- Each static read site has its own two-entry request/response FIFOs and
  outstanding credit limit. Request admission is independent of bank
  arbitration, avoiding conflicting caller/callee priorities. A tagged bank
  request captures the synchronous BRAM response before subsequent requests
  can overwrite it. Collection returns the credit.
- RAW/WAR/WAW dependencies retain their iteration distance and wait on actual
  producer completion, rather than assuming scheduled cycles survive stalls.
  Read completion means response collection. Conditional stores advance their
  completion counters even when their predicate is false.
- Completion counters use modular differences and balance at loop drain, so
  zero-trip loops and re-entry do not inherit unmatched dependencies. Reset
  clears tokens, credits, responses and admission state.
- Stage tokens use two-entry FIFOs with registered readiness, breaking
  combinational cycles between bank arbitration and downstream backpressure.
  They still accept and retire an element together when singly occupied.
- The shared `FIFO2_I` implementation preserves oldest-first order when a
  second element arrives without a dequeue.

Currently supported elastic bodies are single basic blocks without scalar
loop-carried results. Nested control flow and CPU/DMA side effects inside an
elastic body are rejected explicitly; the GEMM's DMA helpers remain outside
that body. BF16 multiply/add operations still use the native three-cycle
HardFloat units. This change does not reassociate the eight-add recurrence.

Compiler and generated-RTL protocol regressions can be run with:

```sh
pixi run python -m pytest tests/test_pipeline_memory.py \
    tests/test_mlir/test_aps_raise_scf_to_affine_pass.py \
    tests/test_mlir/test_tor_schedule_pass.py -q
```

The flattened-loop change passed **36 tests**, with one unavailable real-case
fixture skipped (`build/chipyard-bf16-flat/regressions.log`). These cover
flattened affine addresses, retained bank partitioning, independently stalled
read sites, shared-bank arbitration,
response ordering, reset with
pending responses, dynamic and zero trip counts, loop re-entry, scheduled
II=1/2/10, distance-one and distance-two dependencies, same-address read/modify/
write, and partitioned u32/BF16/FP32 writes without preservation reads,
including one-element banks. Unsupported scalar recurrences are diagnosed.
GEMM acceptance uses the compiled RISC-V program on the full Rocket/RoCC RTL,
including the deliberately corrupted golden and native FP32 software baseline.

### Continuous pipeline measurements

`build/chipyard-bf16-flat/benchmark.log`, `benchmark.json` and `profile.json`
record the flattened loop on the full Rocket/RoCC RTL. Both engines passed all
2048 benchmark comparisons against their respective numerical goldens.

| Trial | Native FP32 C | BF16 DMA load | BF16 compute | BF16 DMA store | BF16 total |
|---|---:|---:|---:|---:|---:|
| 0 | 13202 | 623 | 5108 | 300 | 6031 |
| 1 | 12741 | 640 | 4962 | 300 | 5902 |

Ratios of summed cycles are **2.174× end-to-end** and **2.576× compute-only**.
DMA accounts for **15.61%** of the hardware interval. Compared with the previous
per-row pipeline below, the measured compute interval falls by **40.87%**.

The passive RTL profile confirms:

- All 32 output-pair iterations enter continuously, at offsets
  **0, 9, 18, ..., 279** in the first tile. Across all 16 benchmark tiles,
  all 496 within-tile admission gaps are **exactly 9 cycles**.
- Up to **20 output-pair iterations** overlap. All 512 iterations retire and
  all 34 FP units issue and collect 512 transactions.
- Each accelerator command takes **609 cycles**, down from 1044 (**41.67%**).
- Iteration lifetimes are **75–323 cycles**. The longer tail shows that queued
  work still waits inside the pipeline; II=9 at admission is not proof of a
  sustained result throughput of one pair every nine cycles.
- Aggregate FP issue-slot use is **5.255%**, with up to six outstanding
  transactions per unit. The summed issue-credit-unavailable count is 4512
  unit-cycles; it measures unavailable capacity, not necessarily blocked
  requests or 4512 additional elapsed cycles.

The remaining compute cost includes internal stage waiting and the final
pipeline drain. Flattening removes the seven intermediate row drains while
retaining the ordered BF16 accumulation. The three-cycle HardFloat IP latency
is unchanged. These measurements still cover eight independent 8×8 tiles per
trial, not a large matrix GEMM.

The compiler reuses one `FIFO2_I` module definition per width to keep this
larger pipeline practical to compile. Every FIFO instance retains independent
state; this only eliminates duplicate definitions and repeated analyses.

### Comparison with the previous per-row pipeline

The previous implementation drained after each row (four output pairs).
Its measured admission gaps averaged 30.226 cycles, each tile took 1044 cycles,
and at most four iterations overlapped. Aggregate speedup over native FP32 C
was 1.373× including DMA and 1.523× for compute. The flattened implementation
above removes the intermediate row drains, retaining the same arithmetic,
partitioning and DMA. Historical logs remain in
`build/chipyard-bf16-pipeline-fixed/`; build artifacts are not committed.
