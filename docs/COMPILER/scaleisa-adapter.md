# ScaleISA adapter

`scaleisa-adapter-opt` prepares ScaleISA candidate functions for APS. Public
interfaces live in `include/scaleisa-adapter`; implementations live in
`lib/scaleisa-adapter`. It does not register passes in `aps-opt`.

Build with `cmake --build build --target scaleisa-adapter-opt`. The pipeline is:

```bash
build/tools/aps-scaleisa-opt/scaleisa-adapter-opt candidate.mlir \
  --scaleisa-adapter='max-extra-csrs=5 pragma=none' -o planned.mlir
```

Only functions marked `au.generated = true` or `scaleisa.kernel` are candidates.
The three stages run in this order:

Pragma insertion defaults to `none`. Pipeline and unroll hints require an
explicit request; correctness runs use `max-extra-csrs=5 pragma=none`.

1. `--scaleisa-convert`: lower buffer-based linalg to affine loops and convert
   affine/memref loads and stores to APS scratchpad operations. Preserve arithmetic
   and loop semantics; remove previous scheduling attributes and loop pragmas.
   `preserve-implementation-pragmas=true` retains the selected `pipeline`, `II`,
   and `unroll` choices while still clearing stale schedules and costs. The
   [APS ScaleISA driver](aps-scaleisa-opt.md) uses this mode for selected kernels.
2. `--scaleisa-split='max-extra-csrs=1'`: limit each instruction helper to two
   register inputs plus the configured number of CSR inputs. Keep a candidate
   intact when it fits. Otherwise partition a pure, single-block scalar SSA graph
   into connected, single-output helpers. Joins of computed branches remain
   separate instructions, with intermediate results passed through CPU registers.
   Constants are internal, repeated operands count once, and shared computations
   are preserved. The deterministic greedy planner fuses legal chains; it is
   not a latency-optimal ILP. Oversized candidates must be extracted before
   splitting if other functions already reference them.
3. `--scaleisa-insert-pragmas='mode=pipeline'`: annotate innermost loops after
   splitting. `mode=unroll` requests full unrolling of static affine or constant-bound SCF loops within
   an expanded-operation limit; `mode=none` retains sequential loops. Ineligible
   loops remain sequential and carry an explicit rejection reason.

Each instruction helper has `scaleisa.instruction`, `scaleisa.operand_count`,
`scaleisa.csr_count`, and `scaleisa.max_extra_csrs` attributes. These counts
describe the helper signature, before narrow-operand packing. Oversized memory or
region-bearing graphs are rejected rather than split across effects or loop
dependencies. Tensor inputs require bufferization before this adapter.

The split output is **planning IR**, not an APS hardware input. It contains
scalar helper signatures and non-executable `scaleisa.sequences` module metadata;
there is no `func.call` wrapper. A sequence records the original name and type,
ordered steps with a helper `callee` and `inputs`, and a final `result`.
Input/result bindings refer to an original `argument`, an earlier `step`, or
a typed `constant`. This preserves the composition for later CPU code generation.
Pass it through `--scaleisa-materialize-aps` before APS scheduling or synthesis.
That export contains only hardware helpers with `i5` register selectors,
register/CSR reads, and `aps.write_irf`. Emitting
the corresponding CPU instruction sequence is handled by the
[ScaleISA software compiler](scaleisa-compiler.md).

Materialize the RV32 scalar and CPU-memory operand ABI:

```bash
build/tools/aps-scaleisa-opt/scaleisa-adapter-opt planned.mlir \
  --scaleisa-materialize-aps='csr-base=2049 opcode=43 funct7-base=0' \
  -o hardware.mlir
```

This exports only the hardware instruction functions and retains sequencing
metadata for subsequent software lowering. Each exported
function accepts `rs1`, `rs2`, and `rd` register selectors, reads extra inputs from
dedicated CSR globals, and writes its result with `aps.write_irf`. Register reads
are emitted at function entry before unpacking or memory copies. Encoding and
operand/CSR bindings are recorded on the function for software generation. Narrow
integers are transported as raw low bits, f32 through bitcasts, and index values
as RV32 unsigned indices.

External memref operands are CPU memory, each represented by one RV32 aligned
base pointer in a register or CSR. Array operands receive flat local scratchpads.
Scratchpads follow the CADL frontend convention: module-level `memref.global`
declarations with a `var_name` attribute and `memref.get_global` references in
the instruction body. The adapter emits no `memref.alloca` or `memref.alloc`;
static local candidate buffers are hoisted to globals as well. The existing APS
memory-map and partition passes can therefore discover them.

```text
aps.copy CPU -> scratchpad, length = logical element count
compute using aps.read_smem / aps.write_smem
aps.copy scratchpad -> CPU for writable operands
```

Input copies preserve untouched elements for partial writes. A read-free buffer
with one unconditional store covering every element of a static rectangular
loop nest needs only its output copy. The proof requires distinct induction
variables, zero-based unit-step bounds matching each extent, and no enclosing
conditionals or extra loops. Other buffers retain their input copy.
Output copies precede `aps.write_irf` and function completion.
rank-zero scalar references use one-element scratchpads with the same
import/writeback protocol and alias contract as arrays. Local scalar slots
remain local hardware state. No physical interface declarations, interface-bound operations, or
transfer-length canonicalization are emitted. Copy lengths remain the original
logical counts, including non-power-of-two lengths. For 32-bit elements, copies
use a whole-beat DMA prefix (8-byte reads, 16-byte writes) and exact scalar CPU
accesses for the remaining elements. This preserves adjacent CPU storage for
12-byte and other short buffers. Interface binding and decomposition of the
DMA prefix belong to downstream APS lowering.
An explicit cyclic partition hint is refined when needed to provide one physical
bank per 64-bit DMA lane. For example, factor 1 on an f32 buffer becomes two
physical banks. Larger compatible partition factors are preserved. The scheduled
TOR dump records the realized memory map. CMT2 lowering rejects a DMA copy whose
banks cannot carry a complete beat.
Software must provide the memref's aligned base pointer, not an MLIR descriptor
or an offset-adjusted pointer; the operand bindings record the memref type and
`value_kind = "cpu_aligned_base"` for that purpose. Static source offsets are
added to the CPU address; scratchpad indices begin at logical element zero.

CPU arrays support positive static shapes and contiguous row-major layouts with
i8/i16/i32/f32 elements in the default memory space. Statically known
`memref.dim` is supported; noncontiguous layouts, views, and dynamic descriptor fields
require explicit legalization and are rejected. This restriction concerns
descriptor aliases inside the function. Read-only arguments may alias. When
there are multiple accessed external operands and arrays are buffered, each
writable memref must carry the unit argument attribute `scaleisa.noalias`,
promising disjointness from the other arguments. For example:

```mlir
func.func @kernel(%src: memref<5xi32>,
                  %dst: memref<5xi32> {scaleisa.noalias})
```

The contract is preserved in the exported operand bindings. Overlapping writable
buffers and mixtures with unmodeled CPU-memory effects require explicit alias
legalization. Wide scalars, multiple results, and unbound external symbols are
also rejected explicitly. Buffering optimizations, packing, and target
latency/area feedback are subsequent work. The
output is functional APS input for synthesis, not a claim of completed processor
integration.

Run adapter regression tests with:

```bash
.pixi/envs/default/bin/python -m unittest discover \
  -s tests/scaleisa-adapter -p 'test_*.py'
```

The build target `check-scaleisa-adapter` builds the dedicated tool and runs the
same suite. Integer scalar export has also been smoke-tested through `aps-e2e`
to CMT2 and SystemVerilog. f32 transport preserves the operand bits; downstream
floating-point hardware support requires separate target validation.
The suite now runs the split scalar example through `aps-e2e --export=sv`,
requiring all three helpers to become `tor.func` operations and produce SystemVerilog.
Checking MLIR parsing or scalar arithmetic alone does not establish APS legality.

## Real-candidate numerical correctness

After a completed ScaleISA multi-level discovery run:

```bash
.pixi/envs/default/bin/python tests/scaleisa-adapter/run_correctness.py \
  --discovery-dir scaleisa/tmp/scaleisa-test \
  --output-dir build/scaleisa-adapter-correctness
```

The driver reuses discovery and sequential Hector evaluations, filters out all
pipeline/unroll implementations, and reruns selection at budgets 800 and 1000.
It exports the selected helpers with `max-extra-csrs=5 pragma=none`, canonicalizes
the IR, and checks that no implementation pragma remains.

The test-only `functional_model.py` converts APS register/CSR accesses,
scratchpad operations, and functional copies to executable standard MLIR. It
runs in a separate process with the APS/CIRCT bindings, serializing standard
MLIR for the upstream MLIR execution engine. CPU memory uses aligned virtual
RV32 byte addresses backed by a host word array. The model explicitly supports
i32/f32 transfers; it emits no physical interface or timing model.

Both the original candidate and the APS host model are compiled with the MLIR
execution engine. Outputs are checked bitwise against each other and an
independent float32 oracle, preserving reduction order. The run also checks that
read-only CPU memory, CSR state, and non-result registers remain unchanged.

For the existing `2mm_new` discovery artifacts, budget 800 selects scalar f32
add/multiply (606.43 square microns, 144 numerical cases); budget 1000 selects
the sequential 64-element dot product (881.29 square microns, 24 cases, two extra
CSR operands). All 168 cases pass. These are functional candidate checks, not
RTL or whole-application validation. The area values are existing Hector
estimates and exclude added adapter scratchpad and transfer/interface overhead.

For full-computation functional validation using the budget-1000 selection:

```bash
.pixi/envs/default/bin/python tests/scaleisa-adapter/run_2mm_correctness.py \
  --candidate-run build/scaleisa-adapter-correctness \
  --output-dir build/scaleisa-adapter-2mm-correctness
```

This test explicitly substitutes calls at both sequential dot-product sites in
the original `two_stage_mm`, checking the bounds and reduction dataflow first.
A test-only host bridge stages array data in CPU memory, supplies the register
and two CSR operands, and invokes the generated APS functional model. The
remainder of the original computation is preserved. The harness does not test
ScaleISA e-graph extraction or emit real CPU custom instructions.

Seven full 64-by-64 cases cover zero matrices, identity, boundary basis entries,
signed constants, and three seeded random inputs. All 28,672 output elements
match both the original MLIR JIT and a separate sequential float32 reference
bitwise. The 57,344 APS calls reuse state across cases to expose stale-buffer
bugs; checks cover input preservation, unrelated registers/CSRs, CPU-memory
guards, and the final staged matrices. The report and executable standard MLIR
are saved in the output directory. This remains functional validation, without
physical interfaces, instruction encoding execution, RTL, timing, or new area
measurements. No pragmas are inserted.

When preserving ScaleISA implementation pragmas, the adapter translates
`pipeline = 0` (automatic II in ScaleISA) to APS's `pipeline = 1` enable flag.
A positive ScaleISA pipeline value supplies `II` separately. This preserves the
implementation's scheduling intent across the two pragma conventions.
