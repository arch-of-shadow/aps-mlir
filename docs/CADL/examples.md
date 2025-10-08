# CADL Examples

This document provides complete, working examples of CADL code demonstrating common patterns and use cases.

## Basic Arithmetic Operations

### Simple Addition

```cadl
#[opcode(7'b0001011)]
#[funct7(7'b0000000)]
rtype add(rs1: u5, rs2: u5, rd: u5) {
  let r1: u32 = _irf[rs1];
  let r2: u32 = _irf[rs2];
  _irf[rd] = r1 + r2;
}
```

**Purpose**: Custom RISC-V instruction for addition
**Opcode**: custom0 (7'b0001011)
**Behavior**: Adds two registers and stores result in rd

### Many Multiplications

```cadl
#[opcode(7'b0001011)]
#[funct7(7'b0000001)]
rtype many_mult(rs1: u5, rs2: u5, rd: u5) {
  let r1: u32 = _irf[rs1];
  let r2: u32 = _irf[rs2];
  _irf[rd] = r1 * r2 * r2 * r2 * r2 * r2 * r2 * r2 * r2;
}
```

**Purpose**: Multiple multiplication operations
**Use**: Testing compiler optimization and hardware resource sharing

### Chain of Additions

```cadl
#[opcode(7'b1011011)]
#[funct7(7'b1111100)]
rtype many_add_test(rs1: u5, rs2: u5, rd: u5) {
  let r1: u32 = _irf[rs1];
  let r2: u32 = _irf[rs2];
  let d1: u32 = r1 + r2;
  let d2: u32 = d1 + r1;
  let d3: u32 = d2 + r1;
  let d4: u32 = d3 + r1;
  let d5: u32 = d4 + r1;
  let d6: u32 = d5 + r1;
  let d7: u32 = d6 + r1;
  let d8: u32 = d7 + r1;
  _irf[rd] = d8;
}
```

**Purpose**: Chain of dependent operations
**Result**: `rd = rs2 + 8 * rs1`
**Use**: Testing pipeline depth and latency

## SIMD and Bit Manipulation

### SIMD Byte Addition

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

**Purpose**: SIMD-style byte-wise addition
**Behavior**: Treats 32-bit values as 4 bytes, adds corresponding bytes
**Pattern**: Bit slicing, aggregate construction

### Shift Operation

```cadl
#[opcode(7'b0101011)]
#[funct7(7'b0000001)]
rtype shift_test(rs1: u5, rs2: u5, rd: u5) {
  let r1: u32 = _irf[rs1];
  _irf[rd] = r1 << 1;
}
```

**Purpose**: Simple left shift by 1
**Use**: Basic bit manipulation

## Conditional Operations

### If Expression

```cadl
#[opcode(7'b1011011)]
#[funct7(7'b1111111)]
rtype if_test(rs1: u5, rs2: u5, rd: u5) {
  let r1: u32 = _irf[rs1];
  let r2: u32 = _irf[rs2];
  _irf[rd] = if r1 > 32'd6 {r1} else {r2};
}
```

**Purpose**: Conditional selection based on comparison
**Behavior**: Returns rs1 if > 6, otherwise rs2

### Select Expression

```cadl
#[opcode(7'b0101011)]
#[funct7(7'b0010000)]
rtype categorize(rs1: u5, rs2: u5, rd: u5) {
  let value: u32 = _irf[rs1];

  // Categorize value into ranges
  let category: u32 = sel {
    value == 0: 0,
    value < 10: 1,
    value < 100: 2,
    value < 1000: 3,
    value >= 1000: 4,
  };

  _irf[rd] = category;
}
```

**Purpose**: Multi-way conditional selection
**Behavior**: Categorizes value into 5 ranges (0, 1-9, 10-99, 100-999, 1000+)
**Pattern**: Last arm serves as default

```cadl
#[opcode(7'b0101011)]
#[funct7(7'b0010001)]
rtype compute_function(rs1: u5, rs2: u5, rd: u5) {
  let x: u32 = _irf[rs1];
  let mode: u32 = _irf[rs2];

  // Different computations based on mode
  let result: u32 = sel {
    mode == 0: x + 10,
    mode == 1: x * 2,
    mode == 2: x << 1,
    mode == 3: x >> 1,
    mode >= 4: x,
  };

  _irf[rd] = result;
}
```

**Purpose**: Mode-based operation selection
**Behavior**: Applies different operations based on mode parameter
**Use case**: Configurable accelerators, polymorphic operations

## Memory Operations

### Simple Memory Write

```cadl
#[opcode(7'b1011011)]
#[funct7(7'b0000000)]
rtype mem_simplewrite(rs1: u5, rs2: u5, rd: u5) {
  let r1: u32 = _irf[rs1];
  _mem[r1] = _irf[rs2];
  _irf[rd] = 1437;
}
```

**Purpose**: Write register value to memory
**Inputs**: rs1 = address, rs2 = data
**Output**: rd = status code (1437)

### Memory Read

```cadl
#[opcode(7'b1011011)]
#[funct7(7'b0000001)]
rtype mem_read(rs1: u5, rs2: u5, rd: u5) {
  let r1: u32 = _irf[rs1];
  let r2: u32 = _irf[rs2];
  let rst: u32 = _mem[r1 + r2];
  _irf[rd] = rst;
}
```

**Purpose**: Read from computed memory address
**Address**: rs1 + rs2

### Memory Write with Computation

```cadl
#[opcode(7'b0000000)]
#[funct7(7'b0000000)]
rtype mem_write(rs1: u5, rs2: u5, rd: u5) {
  let r1: u32 = _irf[rs1];
  let r2: u32 = _irf[rs2];
  let a: u32 = r1 + r2;
  _mem[r1] = a;
  _irf[rd] = a + r2;
}
```

**Purpose**: Compute value, write to memory, return modified result

### Memory Accumulation

```cadl
#[opcode(7'b1011011)]
#[funct7(7'b0000010)]
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

**Purpose**: Sum four consecutive memory words
**Pattern**: Multiple memory reads, computation, write result

## Loop-based Algorithms

### Simple Loop Counter

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

**Purpose**: Loop-based accumulation
**Result**: `rd = rs1 + 4 * rs2`
**Pattern**: State threading in do-while loop

### Streaming with State

```cadl
#[opcode(7'b0101011)]
#[funct7(7'b0000010)]
rtype stream_stateful(rs1: u5, rs2: u5, rd: u5) {
  let s0: u32 = _irf[rs1];  // Source address
  let d0: u32 = _irf[rs2];  // Destination address
  let it0: u32 = 0;
  let n0: u32 = 8;

  with
    s: u32 = (s0, s_)
    d: u32 = (d0, d_)
    it: u32 = (it0, it_)
    n: u32 = (n0, n_)
  do {
    let mem1: u32 = _mem[s];
    let mem2: u32 = _mem[s + 4];
    _mem[d] = mem1 + mem2;
    let s_: u32 = s + 8;
    let d_: u32 = d + 4;
    let it_: u32 = it + 1;
    let n_: u32 = n;
  } while (it < n);

  _irf[rd] = 1;
}
```

**Purpose**: Stream processing with loop
**Behavior**: Read pairs, add, write result, update pointers
**Iterations**: 8 loop iterations

## Complex Algorithms

### CRC-8 Calculation

```cadl
#[opcode(7'b0101011)]
#[funct7(7'b0000011)]
rtype crc8(rs1: u5, rs2: u5, rd: u5) {
  let x0: u32 = _irf[rs1];
  let i0: u32 = 0;
  let n0: u32 = 8;

  with
    i: u32 = (i0, i_)
    x: u32 = (x0, x_)
    n: u32 = (n0, n_)
  do {
    let a: u32 = x >> 1;
    let x_: u32 = a ^ (0xEDB88320 & ~((x & 1) - 1));
    let i_: u32 = i + 1;
    let n_: u32 = n;
  } while (i < n);

  _irf[rd] = x;
}
```

**Purpose**: CRC-8 hash computation
**Algorithm**: 8 iterations of shift and conditional XOR
**Pattern**: Bit manipulation, masking, loop

### Complex Number Multiplication

```cadl
#[opcode(7'b0101011)]
#[funct7(7'b0000100)]
rtype cplx_mult(rs1: u5, rs2: u5, rd: u5) {
  let r1: i32 = _irf[rs1];
  let r2: i32 = _irf[rs2];

  // Extract real and imaginary parts (16-bit each)
  let ar: i16 = r1[31:16];
  let ai: i16 = r1[15:0];
  let br: i16 = r2[31:16];
  let bi: i16 = r2[15:0];

  // Complex multiplication: (ar + ai*i) * (br + bi*i)
  let zr: i32 = ar * br - ai * bi;  // Real part
  let zi: i32 = ai * br + ar * bi;  // Imaginary part

  // Pack result back into 32 bits
  _irf[rd] = {zr[15:0], zi[15:0]};
}
```

**Purpose**: 16-bit complex number multiplication
**Input format**: Each 32-bit register holds real (high 16) + imag (low 16)
**Pattern**: Bit slicing, signed arithmetic, bit packing

### CORDIC Algorithm

```cadl
#[impl("regs")]
static thetas: [u32; 8] = {
  1474560, 870484, 459940, 233473,
  117189, 58652, 29333, 14667
};

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
      let z_neg: u1  = z[31:31];        // Extract sign bit
      let theta: u32 = thetas[it];      // Lookup table
      let x_shift: u32 = x >> it;       // Variable shift
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

**Purpose**: CORDIC rotation algorithm for trigonometric functions
**Features**:
- Static lookup table
- Variable-amount shifts
- Conditional operations
- Multiple state variables

## Stateful Operations

### Auto-Incrementing Address

```cadl
static addr: u32 = 0;

#[opcode(7'b0101011)]
#[funct7(7'b0000101)]
rtype autoinc(rs1: u5, rs2: u5, rd: u5) {
  let r1: u32 = _irf[rs1];
  let r2: u32 = _irf[rs2];
  addr = if r2 == 0 {addr + 4} else {r1};
  _irf[rd] = _mem[addr];
}
```

**Purpose**: Maintain persistent address pointer
**Behavior**:
- If rs2 == 0: auto-increment by 4
- Else: reset to rs1
**Use**: Efficient sequential memory access

### State Priority Counter

```cadl
static st: u32 = 0;

#[opcode(7'b0101011)]
#[funct7(7'b0000110)]
rtype state_priority(rs1: u5, rs2: u5, rd: u5) {
  let r1: u32 = _irf[rs1];
  let mr1: u32 = _mem[r1];
  _irf[rd] = mr1 + st;
  st = st + 1;
}
```

**Purpose**: Stateful counter across invocations
**Pattern**: Static variable modification

## Burst Memory Operations

### Burst Vector Addition

```cadl
#[impl("1rw")]
static mem_a: [u32; 16];
static mem_b: [u32; 16];

#[opcode(7'b0101011)]
#[funct7(7'b0000111)]
rtype burst_add(rs1: u5, rs2: u5, rd: u5) {
    let r1: u32 = _irf[rs1];
    let r2: u32 = _irf[rs2];

    // Burst read: copy from memory to local arrays
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

    // Burst write: copy result back to memory
    _burst_write[r1 +: 16] = mem_a[0 +: ];
    _irf[rd] = 42;
}
```

**Purpose**: Efficient vector addition using burst transfers
**Features**:
- Burst memory reads
- Local array storage
- Loop processing
- Burst memory write
**Pattern**: Load-compute-store paradigm

## Multiple Register Reads

### Read Same Register Multiple Times

```cadl
#[opcode(7'b0101011)]
#[funct7(7'b0001000)]
rtype multiple_rs_read(rs1: u5, rs2: u5, rd: u5) {
  let r1: u32 = _irf[rs1];
  let r2: u32 = _irf[rs1];  // Read rs1 again
  let r3: u32 = _irf[rs2];
  let r4: u32 = _irf[rs2];  // Read rs2 again
  _irf[rd] = r1 + r2 + r3 + r4;
}
```

**Purpose**: Test register file multi-read capability
**Pattern**: Multiple reads from same register

## Common Patterns Summary

### Pattern: Input-Process-Output
```cadl
let inputs = read_inputs();
let result = process(inputs);
write_outputs(result);
```

### Pattern: Memory Pipeline
```cadl
let data = _mem[addr];
let processed = transform(data);
_mem[addr + offset] = processed;
```

### Pattern: Stateful Accumulator
```cadl
static state = init;
state = update(state, input);
return state;
```

### Pattern: Loop with Multiple State
```cadl
with
  var1 = (init1, next1)
  var2 = (init2, next2)
do {
  compute next1, next2
} while condition;
```

### Pattern: Conditional Bit Manipulation
```cadl
let flag = extract_bit(value);
let result = if flag {option_a} else {option_b};
```

## Testing Patterns

### Latency Test
```cadl
// Chain operations to test pipeline depth
let d1 = a * b;
let d2 = d1 * c;
let d3 = d2 * d;
```

### Throughput Test
```cadl
spawn {
  op1(); op2(); op3(); op4();
}
```

### Memory Bandwidth Test
```cadl
mem_a[0 +: ] = _burst_read[addr +: 1024];
```

## See Also

- [Flows](flows.md) - Flow definition syntax
- [Statements](statements.md) - Statement types
- [Expressions](expressions.md) - Expression operators
- [Type System](type-system.md) - Data types and literals
