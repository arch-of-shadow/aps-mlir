# APS MLIR Dialect

This document describes the APS (Accelerated Processing System) MLIR dialect, which provides the target representation for CADL code after lowering. The APS dialect models hardware accelerators with custom operations for register files, memory access, and burst transfers.

## Overview

The APS dialect is part of the APS-MLIR compiler infrastructure and provides:

- **Register File Operations**: Access to CPU register files
- **Memory Operations**: Scratchpad memory management
- **Burst Transfers**: Efficient bulk data movement between CPU and accelerator memory
- **Hardware Semantics**: Operations that map directly to hardware structures

## Dialect Namespace

The APS dialect uses the namespace `aps` in MLIR IR:

```mlir
%result = aps.readrf %rs : i5 -> i32
```

## Operations

### Register File Operations

#### `aps.readrf` - Read CPU Register File

Read a value from the CPU register file.

**Syntax:**
```mlir
%result = aps.readrf %rs : i5 -> i32
```

**Semantics:**
- Reads from the CPU register file at index `%rs`
- Returns the register value as `%result`
- Side effect: Memory read

**Example:**
```mlir
%rs1_idx = arith.constant 1 : i5
%rs1_val = aps.readrf %rs1_idx : i5 -> i32
```

**CADL Equivalent:**
```cadl
let r1: u32 = _irf[rs1];
```

#### `aps.writerf` - Write CPU Register File

Write a value to the CPU register file.

**Syntax:**
```mlir
aps.writerf %rd, %value : i5, i32
```

**Semantics:**
- Writes `%value` to the CPU register file at index `%rd`
- Side effect: Memory write

**Example:**
```mlir
%rd_idx = arith.constant 3 : i5
%result = arith.addi %a, %b : i32
aps.writerf %rd_idx, %result : i5, i32
```

**CADL Equivalent:**
```cadl
_irf[rd] = result;
```

### Memory Operations

#### `aps.memdeclare` - Declare Memory Instance

Declare a scratchpad memory instance for the accelerator.

**Syntax:**
```mlir
%mem = aps.memdeclare : memref<16xi32>
```

**Semantics:**
- Allocates a memory resource of the specified type
- Similar to `memref.alloc` but with APS-specific semantics
- Returns a memref handle for the memory instance

**Example:**
```mlir
%mem_a = aps.memdeclare : memref<16xi32>
%mem_b = aps.memdeclare : memref<1024xf32>
```

**CADL Equivalent:**
```cadl
static mem_a: [u32; 16];
```

#### `aps.memload` - Load from Memory

Load a value from scratchpad memory.

**Syntax:**
```mlir
%value = aps.memload %memref[%index] : memref<16xi32>, i32 -> i32
```

**Semantics:**
- Loads element at `%index` from `%memref`
- Returns the loaded value
- Side effect: Memory read

**Example:**
```mlir
%i = arith.constant 5 : i32
%data = aps.memload %mem_a[%i] : memref<16xi32>, i32 -> i32
```

**Multi-dimensional access:**
```mlir
%data = aps.memload %mem[%i, %j] : memref<8x8xi32>, i32, i32 -> i32
```

**CADL Equivalent:**
```cadl
let data: u32 = mem_a[i];
```

#### `aps.memstore` - Store to Memory

Store a value to scratchpad memory.

**Syntax:**
```mlir
aps.memstore %value, %memref[%index] : i32, memref<16xi32>, i32
```

**Semantics:**
- Stores `%value` to `%memref` at `%index`
- Side effect: Memory write

**Example:**
```mlir
%i = arith.constant 5 : i32
%value = arith.constant 42 : i32
aps.memstore %value, %mem_a[%i] : i32, memref<16xi32>, i32
```

**CADL Equivalent:**
```cadl
mem_a[i] = value;
```

### Burst Transfer Operations

Burst operations enable efficient bulk data movement between CPU memory and accelerator scratchpad memory.

#### `aps.memburstload` - Burst Load from CPU Memory

Transfer data from CPU memory to accelerator scratchpad in a burst.

**Syntax:**
```mlir
aps.memburstload %cpu_addr, %memref[%start], %length
  : i64, memref<1024xi32>, i32, i32
```

**Arguments:**
- `%cpu_addr` - Starting address in CPU memory
- `%memref` - Target scratchpad memory
- `%start` - Starting index in scratchpad
- `%length` - Number of elements to transfer

**Semantics:**
- Reads `%length` elements from CPU memory starting at `%cpu_addr`
- Writes them to scratchpad `%memref` starting at index `%start`
- Efficient bulk transfer operation
- Side effects: Memory read and write

**Example:**
```mlir
// Load 16 words from CPU memory to scratchpad
%cpu_addr = aps.readrf %rs1 : i5 -> i64
%start = arith.constant 0 : i32
%length = arith.constant 16 : i32
%mem_a = aps.memdeclare : memref<16xi32>
aps.memburstload %cpu_addr, %mem_a[%start], %length
  : i64, memref<16xi32>, i32, i32
```

**CADL Equivalent:**
```cadl
mem_a[0 +: ] = _burst_read[r1 +: 16];
```

#### `aps.memburststore` - Burst Store to CPU Memory

Transfer data from accelerator scratchpad to CPU memory in a burst.

**Syntax:**
```mlir
aps.memburststore %memref[%start], %cpu_addr, %length
  : memref<1024xi32>, i32, i64, i32
```

**Arguments:**
- `%memref` - Source scratchpad memory
- `%start` - Starting index in scratchpad
- `%cpu_addr` - Target address in CPU memory
- `%length` - Number of elements to transfer

**Semantics:**
- Reads `%length` elements from scratchpad `%memref` starting at `%start`
- Writes them to CPU memory starting at `%cpu_addr`
- Efficient bulk transfer operation
- Side effects: Memory read and write

**Example:**
```mlir
// Store 16 words from scratchpad to CPU memory
%cpu_addr = aps.readrf %rs1 : i5 -> i64
%start = arith.constant 0 : i32
%length = arith.constant 16 : i32
aps.memburststore %mem_a[%start], %cpu_addr, %length
  : memref<16xi32>, i32, i64, i32
```

**CADL Equivalent:**
```cadl
_burst_write[r1 +: 16] = mem_a[0 +: ];
```

## Memory Model

### Memory Hierarchy

The APS dialect models a two-level memory hierarchy:

1. **CPU Memory** - Large, slower, shared system memory
2. **Scratchpad Memory** - Small, fast, accelerator-local memory

### Access Patterns

**Direct Access:**
```mlir
// Load/store to scratchpad
%val = aps.memload %mem[%i] : memref<16xi32>, i32 -> i32
aps.memstore %val, %mem[%i] : i32, memref<16xi32>, i32
```

**Burst Transfer:**
```mlir
// Bulk transfer between CPU and scratchpad
aps.memburstload %cpu_addr, %mem[%start], %len : ...
aps.memburststore %mem[%start], %cpu_addr, %len : ...
```

### Memory Attributes

Memory instances can have implementation attributes:

- **Port Configuration**: `#[impl("1rw")]` - Single read-write port
- **Storage Type**: `#[impl("regs")]` - Register implementation

These attributes guide hardware synthesis.

## Type System

### Integer Types

Register indices and data values:
- `i5` - 5-bit register index (0-31)
- `i32` - 32-bit data
- `i64` - 64-bit addresses

### Memory Types

Scratchpad memory as memref:
- `memref<16xi32>` - 16-element array of 32-bit integers
- `memref<1024xf32>` - 1024-element array of floats
- `memref<8x8xi32>` - 8x8 2D array

## Complete Examples

### Simple Register Addition

**CADL:**
```cadl
rtype add(rs1: u5, rs2: u5, rd: u5) {
  let r1: u32 = _irf[rs1];
  let r2: u32 = _irf[rs2];
  _irf[rd] = r1 + r2;
}
```

**APS MLIR:**
```mlir
func.func @add(%rs1: i5, %rs2: i5, %rd: i5) {
  %r1 = aps.readrf %rs1 : i5 -> i32
  %r2 = aps.readrf %rs2 : i5 -> i32
  %result = arith.addi %r1, %r2 : i32
  aps.writerf %rd, %result : i5, i32
  return
}
```

### Memory Access

**CADL:**
```cadl
rtype mem_read(rs1: u5, rs2: u5, rd: u5) {
  let addr: u32 = _irf[rs1];
  let data: u32 = _mem[addr];
  _irf[rd] = data;
}
```

**APS MLIR:**
```mlir
func.func @mem_read(%rs1: i5, %rs2: i5, %rd: i5) {
  %addr = aps.readrf %rs1 : i5 -> i32
  %data = aps.memload %cpu_mem[%addr] : memref<?xi32>, i32 -> i32
  aps.writerf %rd, %data : i5, i32
  return
}
```

### Burst Vector Addition

**CADL:**
```cadl
#[impl("1rw")]
static mem_a: [u32; 16];
static mem_b: [u32; 16];

rtype burst_add(rs1: u5, rs2: u5, rd: u5) {
    let r1: u32 = _irf[rs1];
    let r2: u32 = _irf[rs2];

    mem_a[0 +: ] = _burst_read[r1 +: 16];
    mem_b[0 +: ] = _burst_read[r2 +: 16];

    with i: u32 = (0, i_) do {
        let a: u32 = mem_a[i];
        let b: u32 = mem_b[i];
        mem_a[i] = a + b;
        let i_: u32 = i + 1;
    } while (i_ < 16);

    _burst_write[r1 +: 16] = mem_a[0 +: ];
    _irf[rd] = 42;
}
```

**APS MLIR (simplified):**
```mlir
func.func @burst_add(%rs1: i5, %rs2: i5, %rd: i5) {
  // Declare scratchpad memories
  %mem_a = aps.memdeclare : memref<16xi32>
  %mem_b = aps.memdeclare : memref<16xi32>

  // Read CPU addresses from registers
  %r1 = aps.readrf %rs1 : i5 -> i64
  %r2 = aps.readrf %rs2 : i5 -> i64

  // Burst load from CPU memory
  %c0 = arith.constant 0 : i32
  %c16 = arith.constant 16 : i32
  aps.memburstload %r1, %mem_a[%c0], %c16 : i64, memref<16xi32>, i32, i32
  aps.memburstload %r2, %mem_b[%c0], %c16 : i64, memref<16xi32>, i32, i32

  // Loop: element-wise addition
  %c1 = arith.constant 1 : i32
  scf.for %i = %c0 to %c16 step %c1 {
    %a = aps.memload %mem_a[%i] : memref<16xi32>, i32 -> i32
    %b = aps.memload %mem_b[%i] : memref<16xi32>, i32 -> i32
    %sum = arith.addi %a, %b : i32
    aps.memstore %sum, %mem_a[%i] : i32, memref<16xi32>, i32
  }

  // Burst store back to CPU memory
  aps.memburststore %mem_a[%c0], %r1, %c16 : memref<16xi32>, i32, i64, i32

  // Write result to register
  %result = arith.constant 42 : i32
  aps.writerf %rd, %result : i5, i32
  return
}
```

## Lowering from CADL

The CADL frontend lowers to APS dialect as follows:

| CADL Construct | APS Operation |
|----------------|---------------|
| `_irf[rs1]` (read) | `aps.readrf %rs1` |
| `_irf[rd] = val` (write) | `aps.writerf %rd, %val` |
| `static mem: [u32; 16]` | `aps.memdeclare : memref<16xi32>` |
| `mem[i]` (read) | `aps.memload %mem[%i]` |
| `mem[i] = val` (write) | `aps.memstore %val, %mem[%i]` |
| `_burst_read[addr +: len]` | `aps.memburstload %addr, %mem[...], %len` |
| `_burst_write[addr +: len] = mem` | `aps.memburststore %mem[...], %addr, %len` |

## Integration with Standard Dialects

The APS dialect works alongside standard MLIR dialects:

- **`arith`** - Arithmetic operations (`addi`, `muli`, etc.)
- **`scf`** - Control flow (loops, conditionals)
- **`func`** - Functions and calls
- **`memref`** - Additional memory operations

**Example with multiple dialects:**
```mlir
func.func @example(%rs1: i5) {
  %val = aps.readrf %rs1 : i5 -> i32
  %c10 = arith.constant 10 : i32
  %cond = arith.cmpi sgt, %val, %c10 : i32
  scf.if %cond {
    %doubled = arith.muli %val, %c2 : i32
    aps.writerf %rd, %doubled : i5, i32
  }
  return
}
```

## Hardware Synthesis

APS operations map to hardware constructs:

- **`aps.readrf`/`aps.writerf`** → Register file ports
- **`aps.memdeclare`** → SRAM/BRAM allocation
- **`aps.memload`/`aps.memstore`** → Memory access circuits
- **`aps.memburstload`/`aps.memburststore`** → DMA controllers

The APS dialect enables hardware-aware optimizations:
- Memory port allocation
- Access pattern analysis
- Burst transfer optimization
- Register file scheduling

## Future Directions

Planned extensions to the APS dialect:

- Streaming interfaces
- Multi-port memory operations
- Hardware synchronization primitives
- Pipeline and dataflow annotations

## See Also

- [CADL Flows](flows.md) - Source language constructs
- [MLIR Documentation](https://mlir.llvm.org/) - General MLIR concepts
- **TOR Dialect** - Transformation and optimization passes
- **Schedule Dialect** - Scheduling and timing analysis
