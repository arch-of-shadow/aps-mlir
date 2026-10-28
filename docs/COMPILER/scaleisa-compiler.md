# ScaleISA software compiler

`lib/scaleisa-compiler` lowers the CI sites already exposed by ScaleISA into
RV32 custom instructions. It uses the matching APS hardware export as the
encoding and operand contract. It does not repeat MEGG's pattern matching.
The dedicated tool is `scaleisa-compiler-opt`; nothing is registered in `aps-opt`.
The [C++ APS ScaleISA driver](aps-scaleisa-opt.md) invokes CI lowering, LLVM
code generation, and linking. It also drives discovery, hardware generation,
and cosimulation for full runs. The compiler driver and both pass executables
are built under `build/tools/aps-scaleisa-opt/`.

```bash
cmake --build build --target scaleisa-compiler-opt

build/tools/aps-scaleisa-opt/aps-scaleisa-opt --compile-only selected.mlir --hardware hardware.mlir -o program.o

build/tools/aps-scaleisa-opt/aps-scaleisa-opt --compile-only selected.mlir --hardware hardware.mlir \
  --emit elf --entry _start --link-object startup.o --link-object main.o \
  -o program.elf
```

`selected.mlir` is ScaleISA's final selected program, including the exposed
helper definitions or declarations. `hardware.mlir` is produced by
`--scaleisa-materialize-aps`, with the same selected implementations and operand
contract. The compiler uses the existing `circt/build` LLVM tools;
it does not build another LLVM. Linking uses `ld.lld`, found in that build,
the Pixi environment, or PATH. Additional runtime objects and linker options
can be supplied with repeated `--link-object` and `--link-arg=-Tplatform.ld`.
An ELF entry must exist; the compiler does not invent platform startup code.

The default target is RV32IMA with Zicsr/Zifencei and the soft-float `ilp32`
ABI, matching MEGG's convention. CPU floating-point code needs the corresponding
runtime at link time, or an explicitly selected floating-point target, for
example `--mattr=+m,+a,+f,+zicsr,+zifencei --abi ilp32f`. Linked objects must
use the same ABI. The default bare-pointer memref function ABI lets a C caller
pass pointers for statically shaped memrefs. `--memref-abi=descriptor` selects
MLIR's expanded descriptor ABI instead. Indices lower to 32 bits.

Each output has a sibling `<output>.scaleisa/` directory containing the CI
MLIR, LLVM-dialect MLIR, LLVM IR, assembly, object, disassembly, stage logs, and
`compile-report.json`. The report records CI site/instruction counts, commands,
target configuration, and hashes of the program, hardware contract, and binary.
Failed compilation records a failure and does not publish a partial binary.
Zero matching CI sites fail unless `--allow-no-ci` is explicitly requested.

## CI binding and instruction emission

Hardware functions provide `scaleisa.source_type`,
`scaleisa.operand_bindings`, `opcode`, `funct3`, and `funct7`. Source argument
and result types must match exactly. The compiler recognizes the hardware
implementation name and ScaleISA's native `<implementation>_site_<number>`
helper names. An explicit `scaleisa.ci = @implementation` attribute on a helper
or call supports other names. Unknown marked CI helpers, incompatible signatures,
invalid encodings, invalid CSR addresses, and indirect helper uses fail explicitly.

Each instruction emits a single side-effecting LLVM inline-assembly block:

```text
csrrw x0, <CSR for argument 2>, <operand 2>
...remaining CSR operands...
.insn r <opcode>, <funct3>, <funct7>, <result>, <operand 0>, <operand 1>
```

CSR writes and the instruction stay together. All operands have register
constraints, so the RISC-V backend allocates their actual CPU registers. The
block has a memory clobber. Memory-taking CIs also have `fence rw, rw` before
and after the block. These order accesses; they do not implement cache flushing
or a new coherence protocol. Memory visibility remains the APS platform's
responsibility.

f32 arguments/results travel as raw i32 bit patterns. Narrow integers preserve
their low bits. CPU memory operands use the memref's aligned base pointer,
not its descriptor or an offset-adjusted logical pointer: the hardware export
already incorporates static source offsets. No hardware interfaces, scratchpad
layouts, copy lengths, or pragmas are changed by the software compiler.

The default APS DMA interface uses naturally aligned bursts of up to 64 bytes.
CI lowering raises compiler-owned buffered allocations and globals, including
rank-zero scalar references, to at least
64-byte alignment, tracing buffers through casts and ordinary helper calls;
stronger existing alignment is retained. External callers must supply buffers
with the alignment required by their hardware interface. The compiler pass's
`dma-alignment` option controls this requirement for other interface sizes.
An element-aligned memref pointer alone does not establish DMA burst alignment.

Split candidates use the hardware module's `scaleisa.sequences` metadata.
For the one-CSR expression example, a software site becomes three CI assembly
blocks:

```text
t0 = CI0(a, x; CSR w)
t1 = CI1(y, z)
r  = CI2(t0, t1)
```

The intermediate results are ordinary CPU SSA values, allocated to CPU registers
or spilled by LLVM. There are no calls between hardware instructions and no
instruction-internal persistent registers. Under the five-CSR cap, that whole
five-input expression fits one CI. Known helper definitions and their exposed
calls are removed from the software module after lowering; normal CPU functions
remain software.

The lowering can also be used separately:

```bash
build/tools/aps-scaleisa-opt/scaleisa-compiler-opt selected.mlir \
  --scaleisa-lower-ci='hardware-file=hardware.mlir max-extra-csrs=5' \
  -o ci.mlir
```

## Validation

```bash
cmake --build build --target check-scaleisa-compiler

.pixi/envs/default/bin/python tests/scaleisa-compiler/run_2mm_binary.py \
  --discovery-dir scaleisa/tmp/scaleisa-test \
  --output-dir build/scaleisa-compiler-2mm
```

The regression suite compiles and links actual RV32 ELF files. A small test-only
instruction interpreter decodes the scalar ELF machine words, supplies the
expected CI semantics, and checks that both the one-CI and three-CI expression
programs return 54. This validates register allocation, encodings, CSR setup,
and intermediate-result flow in the emitted binary, independently of assembly
spelling. Other tests cover native site names, explicit bindings, memory bases,
f32/narrow transport, zero-input/void instructions, preserved side effects,
and rejection paths. This is not an RTL simulator.

The 2mm test reuses real discovery artifacts, selects sequential implementations
at area budget 1000 with a five-CSR cap, and uses ScaleISA's selected-egraph
materialization to obtain both actual CI call sites. It exports the matching
pragma-free APS hardware, compiles the software, and links a C caller plus
freestanding startup into `2mm.elf`. It verifies both custom instruction words
in the linked ELF. The C fixture initializes four 64-by-64 matrices to one and
checks all outputs against 4097 if run on the matching target. The test compiles
and inspects that binary; it does not execute this floating-point/memory workload
against APS hardware. Area remains the existing Hector estimate and excludes
adapter overhead.

## APS co-simulation

```bash
.pixi/envs/default/bin/python tests/scaleisa-compiler/run_2mm_cosim.py --timeout 7200

# Shorter complete two-stage run, using the same 64-term CI hardware:
.pixi/envs/default/bin/python tests/scaleisa-compiler/run_2mm_cosim.py \
  --binary-dir build/scaleisa-cosim-small --output-rows 2 --output-cols 3
```

The runner prepares the selected binary, checks a software-only 2mm baseline on
Spike, generates matching RTL, builds `tools/aps-cosim`, and runs the CI binary
with both CSR bindings. Each stage has a log and the result is recorded in
`build/scaleisa-compiler-2mm/cosim/cosim-report.json`. The startup enables
floating-point state and reports success/failure through Spike's `tohost`.

The runner generates a HardFloat resource configuration from `ip_catalog.yaml`
and passes it to APS scheduling so the scheduled float latencies match the
imported hardware. Unsigned index casts are legalized to the hardware's 32-bit
index representation before CMT2 lowering.

APS first raises the adapter's scratchpad accesses back to memref/affine form
for burst partitioning. FP32 scratchpads need at least two banks for the 64-bit
burst interface; runtime indices select the bank through a mux. Generated bank
names are uniqued against existing globals and carried through to the CMT2
read/write channels using the memory map's bank symbols.

Both the software baseline and the CI binary initialize four 64-by-64 matrices
to one and check all 4096 output elements against 4097. A successful report
requires the CI binary to execute against the generated RTL and exit through
`tohost` with success; compiling the ELF or building the simulator alone does
not count as a co-simulation pass. Use `--binary-dir` to keep separate runs and
`--skip-prepare` to reuse an already prepared selected program and hardware.
`--output-rows` and `--output-cols` default to 64. Smaller output domains keep
the discovered 64-term dot-product kernels and the full matrix layouts: the
first stage computes every intermediate column needed by the second stage.
The 2-by-3 case checks six outputs and executes 134 custom instructions across
both selected call sites. Full 64-by-64 co-simulation executes 8192 instructions
and can take roughly an hour with this unoptimized simulator build.

Generated RTL exposes `instruction_completed_<id>_`, where
`id = (opcode << 8) | funct7`. This bit toggles when the instruction's final
block finishes, after any explicit memory-copy waits. The co-simulator waits
for this transition and an idle DMA engine before returning to Spike; an
instruction with `xd=1` also requires its register response. Void instructions
retain `xd=0` and do not manufacture a register result. Previously generated
RTL without the completion port supports response-producing instructions;
regenerate it before simulating void instructions.

The full `scaleisa/benchmarks/pcl/iss_keypoint` case has also passed Spike +
Verilator validation at its original 40-by-40 size. Greedy selection retained
17 kernels; the compiled program matched all 1,600 FP32 response values bit
for bit, all 1,600 keypoint flags, and the C++ reference signature 16415.
The complete cosimulation took about 20 minutes. Its stage commands and
validation evidence are recorded in `build/scaleisa-cosim-pcl-iss/report.json`.
The default eboost optimality proof was not completed for this experiment.

Non-affine scratchpad indices, such as clamped neighbor coordinates, use
runtime cyclic bank selection with guarded writes. The banked buffers retain
64-bit burst DMA support. Set `APS_COSIM_PROGRESS=1` when starting the
simulator to report progress every 10,000 completed custom instructions;
`APS_COSIM_TRACE=1` enables detailed command and cycle traces.
