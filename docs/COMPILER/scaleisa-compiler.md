# ScaleISA software compiler

`lib/scaleisa-compiler` lowers the CI sites already exposed by ScaleISA into
RV32 custom instructions. It uses the matching APS hardware export as the
encoding and operand contract. It does not repeat MEGG's pattern matching.
The dedicated tool is `scaleisa-compiler-opt`; nothing is registered in `aps-opt`.

```bash
cmake --build build --target scaleisa-compiler-opt

tools/scaleisa-compile selected.mlir --hardware hardware.mlir -o program.o

tools/scaleisa-compile selected.mlir --hardware hardware.mlir \
  --emit elf --entry _start --link-object startup.o --link-object main.o \
  -o program.elf
```

`selected.mlir` is ScaleISA's final selected program, including the exposed
helper definitions or declarations. `hardware.mlir` is produced by
`--scaleisa-materialize-aps`, with the same selected implementations, five-CSR
cap, and no pragmas. The compiler uses the existing `circt/build` LLVM tools;
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
build/tools/scaleisa-compiler-opt/scaleisa-compiler-opt selected.mlir \
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

## APS co-simulation attempt

```bash
.pixi/envs/default/bin/python tests/scaleisa-compiler/run_2mm_cosim.py
```

The runner prepares the selected binary, checks a software-only 2mm baseline on
Spike, generates matching RTL, builds `tools/aps-cosim`, and runs the CI binary
with both CSR bindings. Each stage has a log and the result is recorded in
`build/scaleisa-compiler-2mm/cosim/cosim-report.json`. The startup enables
floating-point state and reports success/failure through Spike's `tohost`.

The software-only baseline executes successfully on Spike and checks all 4096
outputs against 4097. The CI run currently stops before simulator construction:
APS-to-CMT2 rejects `arith.index_castui` in the hardware input. The scheduled TOR
also retains `arith.bitcast`, `tor.mulf`, and `tor.addf`, for which the current
CMT2 generators have no handlers. Cast legalization and floating-point hardware
lowering are required before this f32 example can execute in APS co-simulation.
The compiler/functional-model successes above are not an RTL execution result.
