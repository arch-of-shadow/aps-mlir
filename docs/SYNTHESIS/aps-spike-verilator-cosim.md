# Spike + Verilator co-simulation

[tools/aps-cosim/](../../tools/aps-cosim/) runs generated accelerator RTL with a
Spike CPU or a direct command harness. It supports functional checks of instruction
results, CPU-memory effects, CSR transport, and completion. Full Rocket cycle
measurement uses [Chipyard](../ARCHITECTURE/README.md).

## Prerequisites and execution

Build APS/CIRCT, initialize the Chipyard RISC-V tools, and provide Verilator plus
the Spike static libraries under the configured `SPIKE_BUILD`. The
[Makefile](../../tools/aps-cosim/Makefile) links `riscv`, `disasm`, `softfloat`,
`fesvr`, and `fdt` libraries. The Pixi `setup-verilator` task prepares its build
configuration.

After generating matching RTL and an RV32 ELF:

```sh
pixi run cosim-run build/program.elf build/design.sv
```

The task uses top module `main`, builds the model, and runs `run-spike`.
For isolated models or custom build locations:

```sh
pixi run make -C tools/aps-cosim spike RTL=build/design.sv TOP=main \
  SPIKE_OBJ_DIR=../../build/docs-cosim VERILATE_JOBS=2 BUILD_JOBS=2
build/docs-cosim/Vmain build/program.elf
```

Supply the appropriate `--isa` when needed. Add each CSR used by the generated
hardware with `--add-csr name=addr[,mask=...,init=...]`. Names must match generated
CSR suffixes. The ScaleISA driver's `--cosim` mode derives these bindings from
its hardware export automatically.

## Memory and completion

[spike_memory_bridge.cc](../../tools/aps-cosim/spike_memory_bridge.cc) connects
accelerator accesses to Spike memory, including virtual-address translation.
[fake_dma.cc](../../tools/aps-cosim/fake_dma.cc) services burst transfers against
that memory. Scalar requests and DMA therefore observe the program's buffers.
This bridge does not reproduce Rocket caches or TileLink performance.

[aps_rocc_extension.cc](../../tools/aps-cosim/aps_rocc_extension.cc) dispatches
custom instructions. [aps_rtl_model.cc](../../tools/aps-cosim/aps_rtl_model.cc)
drives the RTL protocol and waits for results or the generated completion toggle
for instructions with `xd=0`. Scalar stores and copy-out must complete before the
instruction is considered done. CSR value/set ports bind by name through
[csr_runtime_bridge.cc](../../tools/aps-cosim/csr_runtime_bridge.cc).

`APS_COSIM_PROGRESS=1` enables progress reporting. `APS_COSIM_STATS=PATH` records
runtime statistics. CI cycles describe accelerator invocations; simulator wall
time and Spike instruction counts are separate quantities.

## Direct RTL checks

```sh
pixi run make -C tools/aps-cosim direct RTL=build/design.sv \
  OBJ_DIR=../../build/docs-direct VERILATE_JOBS=2 BUILD_JOBS=2
build/docs-direct/Vmain --help
```

The direct harness accepts command and memory load/dump options. Use
[test_storage_cosim.py](../../tests/test_mlir/test_storage_cosim.py) for worked
examples covering scalar/DMA outputs with and without register results.
[run_direct.cc](../../tools/aps-cosim/run_direct.cc) defines its current options.

Other regressions cover [Spike memory translation](../../tests/test_mlir/test_spike_memory_bridge.py)
and [pipeline memory](../../tests/test_pipeline_memory.py). Independent numerical
checks and failing corrupted-output checks are required for a correctness campaign;
a successful simulator build alone validates only model compilation.
