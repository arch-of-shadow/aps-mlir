# ScaleISA adapter

The adapter converts selected kernel computations into APS instruction hardware
contracts. Public passes are declared in
[Passes.td](../../include/scaleisa-adapter/Passes.td); implementation lives in
[lib/scaleisa-adapter/](../../lib/scaleisa-adapter/).

## Planning and hardware export

```sh
pixi run build/tools/aps-scaleisa-opt/scaleisa-adapter-opt candidate.mlir \
  --scaleisa-adapter='max-extra-csrs=5 pragma=none' -o build/planned.mlir
pixi run build/tools/aps-scaleisa-opt/scaleisa-adapter-opt build/planned.mlir \
  --scaleisa-materialize-aps='csr-base=2049 opcode=43 funct7-base=0' \
  -o build/hardware.mlir
```

Create `build/` first and supply candidate functions marked `au.generated = true`
or `scaleisa.kernel`. The registered pipeline in
[Pipelines.cpp](../../lib/scaleisa-adapter/Pipelines.cpp) runs:

1. `scaleisa-convert`: lower buffer-based linalg/affine accesses to APS-compatible
   computation and clear stale scheduling/cost metadata.
2. `scaleisa-split`: fit each scalar helper into two register inputs plus the
   configured CSR budget. The pipeline's default CSR budget is zero.
3. `scaleisa-insert-pragmas`: apply `none`, `pipeline`, or `unroll` mode after
   splitting. The default is `none`.

The production driver invokes conversion with
`preserve-implementation-pragmas=true`, then splitting, to preserve the selected
implementation's pipeline/II/unroll choices. Use that explicit conversion option
when adapting an already selected implementation outside the driver.

Oversized pure single-block scalar graphs can be partitioned into connected,
single-result instructions. The deterministic greedy planner preserves shared
computation and passes intermediates through CPU values. Oversized effects or
region-bearing graphs are rejected. Splitting is an operand-feasibility step;
it does not promise an optimal latency partition.

Planning IR contains helpers and `scaleisa.sequences` metadata describing their
composition. It is not yet an input for APS synthesis. Materialization creates
hardware functions with five-bit register selectors, `aps.read_irf`, CSR reads,
result writes, encoding attributes, and operand bindings. The software compiler
uses that same metadata to emit instruction sequences.

## Operand and memory ABI

Narrow integers travel as raw low bits; f32 travels through bitcasts; indices use
RV32 unsigned values. Extra scalar/pointer operands use dedicated CSRs.

External memrefs represent CPU memory and use aligned base pointers. Supported
buffered operands have positive static shapes, contiguous row-major layouts,
default memory space, and i8/i16/i32/f32 elements. Local scratchpads and copy-in/out
operations implement access effects. Static source offsets are incorporated by
the hardware export. Dynamic descriptor fields, noncontiguous views, unsupported
widths, and unbound external symbols need legalization first.

Read-only operands may alias. With multiple accessed external operands and
buffering, writable memrefs require the unit argument attribute `scaleisa.noalias`,
promising disjointness from other arguments:

```mlir
func.func @kernel(%src: memref<5xi32>,
                  %dst: memref<5xi32> {scaleisa.noalias})
```

The driver adds such annotations only for proven distinct allocation origins,
including distinct defined globals; repeated use of one buffer is not proof.
Keep this contract consistent when changing call-site analysis.

## Verification

```sh
pixi run cmake --build build --target check-scaleisa-adapter
```

The numerical runner `tests/scaleisa-adapter/run_correctness.py` additionally
requires a completed compatible 2mm discovery run; select it with
`--discovery-dir PATH`. Its default cache path is `scaleisa/tmp/scaleisa-test`.

Review [test_adapter.py](../../tests/scaleisa-adapter/test_adapter.py) for positive
and rejected forms, [MaterializeAPS.cpp](../../lib/scaleisa-adapter/MaterializeAPS.cpp)
for hardware ABI construction, and [MemoryABI.cpp](../../lib/scaleisa-adapter/MemoryABI.cpp)
for layout/alias contracts. Numerical model checks and actual RTL/CPU execution
validate different parts of the flow; successful MLIR parsing alone is insufficient.
