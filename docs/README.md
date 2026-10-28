# APS Documentation

This documentation covers the APS (Agile Processor Specialization) framework, organized into three major components.

## Documentation Structure

- **[SYNTHESIS/](SYNTHESIS/)** - Hardware synthesis from CADL to Verilog
- **[COMPILER/](COMPILER/)** - E-graph based pattern matching compiler
- **[ARCHITECTURE/](ARCHITECTURE/)** - Processor integration and deployment

## Quick Start

### Hardware Synthesis

```bash
pixi run mlir tutorial/cadl/hello.cadl outputs/hello.mlir
pixi run opt outputs/hello.mlir outputs/hello_cmt.mlir
pixi run sv outputs/hello_cmt.mlir outputs/hello.sv
```

### Compiler

```bash
pixi run compile tutorial/cadl/hello.cadl csrc/test.c outputs/test.riscv
```

### Deployment

```bash
cd $APS_CHIPYARD/sims/verilator
make CONFIG=APSRocketConfig run-binary-debug BINARY=test.riscv LOADMEM=1
```

## Key Concepts

### CADL (Computer Architecture Description Language)

A high-level DSL for describing ISAXs:

```cadl
#[opcode(7'b0101011)]
#[funct7(7'b0000000)]
rtype hello(rs1: u5, rs2: u5, rd: u5) {
  let a: u32 = _irf[rs1];
  let b: u32 = _irf[rs2];
  _irf[rd] = a + b;
}
```

### E-graph Based Compilation

Uses equality saturation to find optimal instruction mappings through algebraic and control flow transformations.

### APS-Itfc (Interface Abstraction)

A unified interface for ISAX integration across RISC-V platforms (Rocket/RoCC, CV-X-IF, etc.).

## Resources

- [APS ScaleISA driver](COMPILER/aps-scaleisa-opt.md) — discovery, selected AU
  software, matching RTL, and cosimulation through one C++ tool.
- [ScaleISA benchmark comparison](COMPILER/scaleisa-benchmark-report.md) — full
  Rocket cycle measurements, model estimates, and exact AU/APS MLIR artifacts.
- [ScaleISA adapter](COMPILER/scaleisa-adapter.md) — candidate conversion,
  instruction splitting, loop pragmas, and scalar APS interfaces.
- [ScaleISA software compiler](COMPILER/scaleisa-compiler.md) — exposed CI sites
  to RISC-V instructions, objects, and linked program binaries.

- **Project Home**: [http://aps.ericlyun.me/](http://aps.ericlyun.me/)
- **Tutorials**: [http://aps.ericlyun.me/tutorials/](http://aps.ericlyun.me/tutorials/)
