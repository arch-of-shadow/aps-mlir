# CADL Flows

This document describes flow definitions in CADL, the primary construct for defining custom RISC-V instructions and hardware accelerator behavior.

## Flow Types

CADL provides two types of flows:

### 1. Flow (Default, DO NOT USE)

Generic hardware flow for accelerator behavior:

```cadl
flow name(arg1: type1, arg2: type2, ...) {
  // Flow body
}
```

**Use for:**
- Hardware accelerator modules
- Dataflow operations
- Reusable hardware components

### 2. RType Flow

RISC-V R-type instruction definition:

```cadl
rtype name(rs1: u5, rs2: u5, rd: u5) {
  // Instruction body
}
```

**Use for:**
- Custom RISC-V instructions
- ISA extensions
- Instructions that interact with the register file

**Standard R-type signature:**
- `rs1: u5` - Source register 1 (5-bit register index)
- `rs2: u5` - Source register 2 (5-bit register index)
- `rd: u5` - Destination register (5-bit register index)

## Flow Syntax

### Basic Structure

```cadl
[attributes]
flow_type flow_name(parameters) {
  statements
}
```

**Components:**
1. **Attributes** (optional) - Metadata and configuration
2. **Flow type** - `flow` or `rtype`
3. **Name** - Identifier for the flow
4. **Parameters** - Input arguments with types
5. **Body** - Statements defining behavior

### Parameters

Flow parameters specify inputs:

```cadl
flow_name(param1: type1, param2: type2, ...)
```

**Examples:**
```cadl
flow add_vectors(a: u32, b: u32)
flow process_data(addr: u32, count: u32, enable: u1)
rtype custom_mult(rs1: u5, rs2: u5, rd: u5)
```

**Parameter Types:**
- Basic types: `u32`, `i16`, `f32`, etc.
- Arrays: `[u32; 16]`
- Any valid CADL type

### Empty Body

A flow can have an empty body (forward declaration):

```cadl
flow placeholder(x: u32);
```

## Attributes

Flows can be annotated with attributes that provide metadata for hardware generation and RISC-V integration.

### Attribute Syntax

```cadl
#[attribute_name]
#[attribute_name(value)]
```

### RISC-V Attributes

#### Opcode

Specify the RISC-V opcode (7 bits):

```cadl
#[opcode(7'b0101011)]
rtype my_instruction(...) { ... }
```

**Common opcodes:**
- `7'b0001011` - custom0
- `7'b0101011` - custom1
- `7'b1011011` - custom2
- `7'b1111011` - custom3

#### Funct7

Specify the funct7 field (7 bits):

```cadl
#[funct7(7'b0000000)]
rtype my_instruction(...) { ... }
```

**Example with both:**
```cadl
#[opcode(7'b0101011)]
#[funct7(7'b0000000)]
rtype cordic(rs1: u5, rs2: u5, rd: u5) {
  // Implementation
}
```

### Implementation Hints

#### Register Implementation

Indicate variables should be implemented as registers:

```cadl
#[impl("regs")]
static thetas: [u32; 8] = {...};
```

#### Memory Port Configuration

Specify memory port configuration:

```cadl
#[impl("1rw")]
static mem_a: [u32; 16];  // Single read-write port
```

**Port configurations:**
- `"1rw"` - One read-write port
- `"1r1w"` - One read port, one write port
- `"2rw"` - Two read-write ports

### Multiple Attributes

Chain multiple attributes:

```cadl
#[opcode(7'b0101011)]
#[funct7(7'b0000000)]
#[impl("pipeline")]
rtype my_custom_op(rs1: u5, rs2: u5, rd: u5) {
  // Implementation
}
```

## Flow Body

The flow body contains statements that define the flow's behavior.

### Typical Structure

```cadl
rtype example(rs1: u5, rs2: u5, rd: u5) {
  // 1. Read inputs
  let r1: u32 = _irf[rs1];
  let r2: u32 = _irf[rs2];

  // 2. Compute result
  let result: u32 = r1 + r2;

  // 3. Write output
  _irf[rd] = result;
}
```

### Register File Access

**Reading from registers:**
```cadl
let value: u32 = _irf[register_index];
```

**Writing to registers:**
```cadl
_irf[register_index] = value;
```

**Example:**
```cadl
rtype add(rs1: u5, rs2: u5, rd: u5) {
  let r1: u32 = _irf[rs1];
  let r2: u32 = _irf[rs2];
  _irf[rd] = r1 + r2;
}
```

### Memory Access

**Reading from memory:**
```cadl
let data: u32 = _mem[address];
```

**Writing to memory:**
```cadl
_mem[address] = data;
```

**Example:**
```cadl
rtype mem_write(rs1: u5, rs2: u5, rd: u5) {
  let r1: u32 = _irf[rs1];  // Address
  let r2: u32 = _irf[rs2];  // Data
  _mem[r1] = r2;
  _irf[rd] = 1;  // Success indicator
}
```

## Complete Examples

### Simple Arithmetic

```cadl
#[opcode(7'b0101011)]
#[funct7(7'b0000000)]
rtype custom_add(rs1: u5, rs2: u5, rd: u5) {
  let r1: u32 = _irf[rs1];
  let r2: u32 = _irf[rs2];
  _irf[rd] = r1 + r2;
}
```

### SIMD Addition

```cadl
#[opcode(7'b0101011)]
#[funct7(7'b1111111)]
rtype simd_add(rs1: u5, rs2: u5, rd: u5) {
  let r1: u32 = _irf[rs1];
  let r2: u32 = _irf[rs2];
  _irf[rd] = {
      (r1[31:24] + r2[31:24])[7:0],
      (r1[23:16] + r2[23:16])[7:0],
      (r1[15: 8] + r2[15: 8])[7:0],
      (r1[ 7: 0] + r2[ 7: 0])[7:0]
  };
}
```

### Conditional Operation

```cadl
#[opcode(7'b1011011)]
#[funct7(7'b1111111)]
rtype if_test(rs1: u5, rs2: u5, rd: u5) {
  let r1: u32 = _irf[rs1];
  let r2: u32 = _irf[rs2];
  _irf[rd] = if r1 > 32'd6 {r1} else {r2};
}
```

### Memory Accumulation

```cadl
#[opcode(7'b1011011)]
#[funct7(7'b0000000)]
rtype accum(rs1: u5, rs2: u5, rd: u5) {
  let r1: u32 = _irf[rs1];
  let a : u32 = _mem[r1];
  let b : u32 = _mem[r1 + 4];
  let c : u32 = _mem[r1 + 8];
  let d : u32 = _mem[r1 + 12];
  let rst: u32 = a + b + c + d;
  _mem[r1 + 16] = rst;
  _irf[rd] = rst;
}
```

### Stateful Flow

```cadl
static counter: u32 = 0;

#[opcode(7'b0101011)]
#[funct7(7'b0000000)]
rtype increment_counter(rs1: u5, rs2: u5, rd: u5) {
  let r1: u32 = _irf[rs1];
  counter = counter + r1;
  _irf[rd] = counter;
}
```

### Loop-based Processing

```cadl
#[opcode(7'b1011011)]
#[funct7(7'b1111100)]
rtype loop_test(rs1: u5, rs2: u5, rd: u5) {
  let sum0: u32 = _irf[rs1];
  let i0: u32 = 0;
  let n0: u32 = _irf[rs2];

  with
    i: u32 = (i0, i_)
    sum: u32 = (sum0, sum_)
    n: u32 = (n0, n_)
  do {
    let n_: u32 = n;
    let sum_: u32 = sum + 4;
    let i_: u32 = i + 1;
  } while (i_ < n);

  _irf[rd] = sum;
}
```

### Complex Multiplication

```cadl
#[opcode(7'b0101011)]
#[funct7(7'b0000000)]
rtype cplx_mult(rs1: u5, rs2: u5, rd: u5) {
  let r1: i32 = _irf[rs1];
  let r2: i32 = _irf[rs2];

  // Extract real and imaginary parts
  let ar: i16 = r1[31:16];
  let ai: i16 = r1[15:0];
  let br: i16 = r2[31:16];
  let bi: i16 = r2[15:0];

  // Complex multiplication: (ar + ai*i) * (br + bi*i)
  let zr: i32 = ar * br - ai * bi;
  let zi: i32 = ai * br + ar * bi;

  // Pack result
  _irf[rd] = {zr[15:0], zi[15:0]};
}
```

### CORDIC Algorithm

```cadl
#[impl("regs")]
static thetas: [u32; 8] = {1474560, 870484, 459940, 233473, 117189, 58652, 29333, 14667};

#[opcode(7'b0101011)]
#[funct7(7'b0000000)]
rtype cordic(rs1: u5, rs2: u5, rd: u5) {
    let x0 : u32 = 19898;
    let y0 : u32 = 0;
    let z0 : u32 = _irf[rs1];
    let n0 : u32 = 8;
    let it0: u32 = 0;

    with
      it: u32 = (it0, it_)
      x: u32 = (x0, x_)
      y: u32 = (y0, y_)
      z: u32 = (z0, z_)
      n: u32 = (n0, n_)
    do {
      let z_neg: u1  = z[31:31];
      let theta: u32 = thetas[it];
      let x_shift: u32 = x >> it;
      let y_shift: u32 = y >> it;
      let x_ : u32 = if z_neg {x + y_shift} else {x - y_shift};
      let y_ : u32 = if z_neg {y - x_shift} else {y + x_shift};
      let z_ : u32 = if z_neg {z + theta} else {z - theta};
      let it_: u32 = it + 1;
      let n_ : u32 = n;
    } while (it < n);

    _irf[rd] = y;
}
```

### Burst Memory Operations

```cadl
#[impl("1rw")]
static mem_a: [u32; 16];
static mem_b: [u32; 16];

#[opcode(7'b0101011)]
#[funct7(7'b0000000)]
rtype burst_add(rs1: u5, rs2: u5, rd: u5) {
    let r1: u32 = _irf[rs1];
    let r2: u32 = _irf[rs2];

    // Burst read arrays from memory
    mem_a[0 +: ] = _burst_read[r1 +: 16];
    mem_b[0 +: ] = _burst_read[r2 +: 16];

    // Element-wise addition
    with i: u32 = (0, i_) do {
        let a: u32 = mem_a[i];
        let b: u32 = mem_b[i];
        let c: u32 = a + b;
        mem_a[i] = c;
        let i_: u32 = i + 1;
    } while (i_ < 16);

    // Burst write result back
    _burst_write[r1 +: 16] = mem_a[0 +: ];
    _irf[rd] = 42;
}
```

## Design Patterns

### Input-Process-Output

Standard pattern for most instructions:

```cadl
rtype operation(rs1: u5, rs2: u5, rd: u5) {
  // 1. Input: Read from register file
  let input1: u32 = _irf[rs1];
  let input2: u32 = _irf[rs2];

  // 2. Process: Compute
  let result: u32 = process(input1, input2);

  // 3. Output: Write result
  _irf[rd] = result;
}
```

### Memory Access Pattern

```cadl
rtype mem_operation(rs1: u5, rs2: u5, rd: u5) {
  let addr: u32 = _irf[rs1];
  let data: u32 = _mem[addr];
  let result: u32 = process(data);
  _mem[addr + 4] = result;
  _irf[rd] = result;
}
```

### Stateful Accumulator

```cadl
static state: u32 = 0;

rtype accumulate(rs1: u5, rs2: u5, rd: u5) {
  let input: u32 = _irf[rs1];
  state = state + input;
  _irf[rd] = state;
}
```

### Pipeline with Local Storage

```cadl
#[impl("1rw")]
static buffer: [u32; 8];

rtype pipeline_stage(rs1: u5, rs2: u5, rd: u5) {
  let idx: u32 = _irf[rs1];
  let data: u32 = _irf[rs2];
  buffer[idx] = process(data);
  _irf[rd] = buffer[idx];
}
```

## Best Practices

1. **Use Descriptive Names**: Flow names should describe their function
   ```cadl
   rtype vector_dot_product(...)  // Good
   rtype vdp(...)                  // Less clear
   ```

2. **Add RISC-V Attributes**: Always specify opcode and funct7 for rtype flows
   ```cadl
   #[opcode(7'b0101011)]
   #[funct7(7'b0000000)]
   rtype my_instruction(...) { ... }
   ```

3. **Explicit Types**: Always specify types for variables
   ```cadl
   let result: u32 = a + b;  // Good
   let result = a + b;        // Type inferred, but less clear
   ```

4. **Consistent Register Access**: Read inputs first, write outputs last
   ```cadl
   // Good pattern:
   let r1: u32 = _irf[rs1];
   let r2: u32 = _irf[rs2];
   let result: u32 = r1 + r2;
   _irf[rd] = result;
   ```

5. **Document Complex Logic**: Use comments for non-obvious operations
   ```cadl
   // Extract sign bit
   let sign: u1 = value[31:31];
   ```

6. **Use Static for State**: Declare persistent state as static
   ```cadl
   static counter: u32 = 0;
   static buffer: [u32; 16];
   ```

## Register File Definition

Before using `_irf`, you may need to define the register file:

```cadl
regfile _irf(32, 32);  // 32 registers, 32 bits wide
```

**Syntax:**
```cadl
regfile name(depth, width);
```

## See Also

- [Statements](statements.md) - Statement syntax
- [Expressions](expressions.md) - Expression syntax
- [Type System](type-system.md) - Data types
- [Examples](examples.md) - Complete working examples
- [APS Dialect](aps-dialect.md) - MLIR lowering
