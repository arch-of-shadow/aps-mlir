# ScaleISA software compiler

The compiler lowers CI sites exposed by ScaleISA using the matching materialized
APS hardware module. It preserves the selected instruction/operand contract;
it does not perform Megg pattern matching.

## Compile and link

```sh
pixi run build/tools/aps-scaleisa-opt/aps-scaleisa-opt \
  --compile-only selected.mlir --hardware hardware.mlir -o build/program.o
```

`selected.mlir` contains the selected software and exposed helper sites.
`hardware.mlir` is the adapter's matching export. For an ELF, add `--emit=elf`,
`--entry=_start`, runtime objects/sources, and linker arguments:

```sh
pixi run build/tools/aps-scaleisa-opt/aps-scaleisa-opt \
  --compile-only selected.mlir --hardware hardware.mlir \
  --emit=elf --entry=_start --link-object=startup.o --link-object=main.o \
  --link-arg=-Tplatform.ld -o build/program.elf
```

The default target uses RV32, `+m,+a,+zicsr,+zifencei`, soft-float `ilp32`,
32-bit indices, and the bare-pointer memref ABI. `--memref-abi=descriptor`
selects expanded MLIR descriptors. For native CPU floating point use compatible
features, ABI, and runtime objects. Startup and platform linkage are explicit inputs.

The [driver](../../tools/aps-scaleisa-opt/main.cpp) invokes the existing LLVM
utilities under `circt/build`, then Pixi/PATH fallbacks. Compilation artifacts
live beside the output in `<output>.scaleisa/`, including CI/LLVM MLIR, LLVM IR,
assembly, object/disassembly, logs, and `compile-report.json`. Reports record
counts, target settings, commands, and hashes. A failed compile does not publish
a partial binary. Use `--allow-no-ci` only for an intended software baseline.

## Binding and emission

[LowerCI.cpp](../../lib/scaleisa-compiler/LowerCI.cpp) matches hardware source
types and `scaleisa.operand_bindings`. Sites may use implementation names,
`<implementation>_site_<number>`, or explicit `scaleisa.ci = @implementation`.
Unknown marked sites, signature mismatches, illegal encodings, and invalid CSR
bindings fail explicitly.

Each instruction emits one side-effecting LLVM inline-assembly block containing
CSR operand writes followed by `.insn r`. Register constraints leave register
allocation to LLVM. A memory clobber keeps relevant memory effects ordered;
memory-taking instructions additionally emit `fence rw, rw` before and after.
Platform memory visibility must satisfy the hardware's interface contract.

f32 operands/results are raw i32 bits. Narrow values preserve low bits. Memref
operands use aligned base pointers, with static source offsets already accounted
for by the hardware export. Split instructions follow `scaleisa.sequences`;
intermediates remain CPU SSA values, allocated or spilled by LLVM.

Compiler-owned buffered allocations/globals are raised to the CI lowering pass's
DMA alignment requirement (default 64 bytes), retaining stronger alignment.
External buffers must meet the actual chosen interface's burst alignment.
The pass's `dma-alignment` option configures other requirements; check alignment
again when changing burst sizes.

## Pass-level use and tests

```sh
pixi run build/tools/aps-scaleisa-opt/scaleisa-compiler-opt selected.mlir \
  --scaleisa-lower-ci='hardware-file=hardware.mlir max-extra-csrs=5' \
  -o build/ci.mlir
pixi run cmake --build build --target check-scaleisa-compiler
```

The dedicated pass tool is separate from `aps-opt`. See
[test_compiler.py](../../tests/scaleisa-compiler/test_compiler.py) for ABI,
encoding, helper, and failure coverage. The
[2mm runner](../../tests/scaleisa-compiler/run_2mm_cosim.py) builds a selected
program and matching RTL for execution. Full-pipeline runs and numerical/performance
campaigns are described in the [driver](aps-scaleisa-opt.md) and
[benchmark](aps-scaleisa-opt.md#benchmark-reproduction-and-evidence) guides.
