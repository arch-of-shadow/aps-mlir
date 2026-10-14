# CADL Language Reference

CADL (Computer Architecture Description Language) is a high-level DSL for describing custom RISC-V instruction extensions (ISAXs).

## Basic Structure

```cadl
// Global scratchpad memory
static array: [u32; 16];

// Instruction definition
#[opcode(7'b0101011)]
#[funct7(7'b0000000)]
rtype instruction_name(rs1: u5, rs2: u5, rd: u5) {
  // Instruction body
}
```

## Type System

| Type | Description |
|------|-------------|
| `u8`, `u16`, `u32`, `u64` | Unsigned integers |
| `i8`, `i16`, `i32`, `i64` | Signed integers |
| `u1` - `u256` | Arbitrary width unsigned |

### Width-Aware Literals

```cadl
5'b10101    // 5-bit binary
8'hFF       // 8-bit hexadecimal
32'd1000    // 32-bit decimal
```

## Variables

```cadl
let x: u32 = 42;                  // Local variable
static buffer: [u32; 64];         // Scratchpad memory
static acc: i32;                  // Scalar static
```

## Processor Interface (APS-Itfc)

```cadl
// Register file access
let a: u32 = _irf[rs1];           // Read GPR
_irf[rd] = result;                // Write GPR

// Single memory access
let data: u32 = _mem[addr];       // Read CPU memory
_mem[addr] = data;                // Write CPU memory

// Burst DMA access
matrix[0 +: ] = _burst_read[addr +: 16];      // Burst read
_burst_write[addr +: 16] = matrix[0 +: ];     // Burst write
```

## Control Flow

### Do-While Loops

```cadl
with i: u32 = (init, next) do {
  // Loop body
  let next: u32 = i + 1;
} while (condition);
```

### Multiple Loop-Carried Variables

```cadl
with i: u32 = (0, i_), acc: i32 = (0, acc_) do {
  acc_ = acc + array[i];
  let i_: u32 = i + 1;
} while (i_ < 16);
```

## Instruction Definitions

```cadl
#[opcode(7'b0101011)]     // Custom-1 opcode
#[funct7(7'b0000000)]
rtype add_custom(rs1: u5, rs2: u5, rd: u5) {
  let a: u32 = _irf[rs1];
  let b: u32 = _irf[rs2];
  _irf[rd] = a + b;
}
```

### RISC-V Custom Opcodes

| Opcode | Binary | Description |
|--------|--------|-------------|
| custom-0 | `0001011` | User extension 0 |
| custom-1 | `0101011` | User extension 1 |
| custom-2 | `1011011` | User extension 2 |
| custom-3 | `1111011` | User extension 3 |

## Optimization Directives

### Loop Unrolling

```cadl
[[unroll(4)]]
with i: u32 = (0, i + 1) do { ... } while (i + 1 < 16);
```

### Array Partitioning

```cadl
#[partition_dim_array([0])]
#[partition_factor_array([4])]
#[partition_cyclic_array([1])]     // 1=cyclic, 0=block
static array: [i32; 16];
```

### DMA Stride

```cadl
[[stride_x(2)]]
[[stride_y(2)]]
mat[sa +: ] = _burst_read[da +: 8];
```

## Example: Vector Distance

```cadl
#[partition_dim_array([0])]
#[partition_factor_array([4])]
#[partition_cyclic_array([1])]
static points1_x: [u32; 16];
static points2_x: [u32; 16];
static dist_out: [u32; 16];

#[opcode(7'b0001011)]
#[funct7(7'b0101000)]
rtype v3ddist_vv(rs1: u5, rs2: u5, rd: u5) {
  let addr1: u32 = _irf[rs1];
  let out_addr: u32 = _irf[rs2];

  points1_x[0 +: ] = _burst_read[addr1 +: 16];
  points2_x[0 +: ] = _burst_read[addr1 + 64 +: 16];

  [[unroll(4)]]
  with i: u32 = (0, i + 1) do {
    let dx: u32 = points1_x[i] - points2_x[i];
    dist_out[i] = dx * dx;
  } while (i + 1 < 16);

  _burst_write[out_addr +: 16] = dist_out[0 +: ];
  _irf[rd] = 0;
}
```

## Frontend Usage

```bash
pixi run mlir examples/features/csr_partition_loop.cadl out.mlir  # Convert to MLIR
```

## CSR Parameters in MEGG

Declare an instruction's configuration input as a CSR and read it with
`_csr[name]`:

```rust
#[csr_address(0x801)]
register gain_cfg: u32;
```

During semantic alignment, MEGG converts each CSR read into a value parameter
after the two `rs1`/`rs2` parameters in the generated C pattern. Repeated reads
of the same CSR share one parameter. Application C uses ordinary values and
computations; it does not need to write the CSR explicitly.

The instruction encoding JSON carries `csr_inputs` entries with the CSR name,
address, and original pattern argument index. Matching records the values bound
to those parameters, including constants and computed expressions. When a
selected custom instruction is lowered, MEGG emits `csrrw x0, address, value`
for each bound CSR, followed by the R-type instruction. These operations share
one side-effecting inline assembly block with a memory clobber. Only the first
two pattern parameters supply `rs1` and `rs2`; CSR parameters supply the setup
writes. Unused source registers and void destinations use `x0`.
The RoCC `xd` flag (bit 2 of `funct3`) is derived from the emitted result:
value-producing calls set it, while void calls clear it even if encoding
metadata specifies `funct3 = 0x7`. The `xs1`/`xs2` bits are preserved.

An instruction's input expressions must be available before the region it
replaces. A CSR input such as `gain + i`, where `i` belongs to the replaced
loop, prevents that offload and keeps the software loop. An outer-loop value
can configure an instruction replacing only an inner loop. The matcher checks
the expressions selected by the extraction cost model; memory reads and other
effects are conservatively rejected as inputs that would need hoisting.

Integer matching also recovers subtraction patterns after constant folding,
using negation modulo the operand width. Reconstruction exceptions fail the
compilation with a function-specific diagnostic instead of silently omitting
the function from the output module.

The current target is RV32: CSR parameters must have scalar integer types of
at most 32 bits and literal writable 12-bit addresses. CSR writes inside CADL
matching patterns are unsupported. Each selected instruction invocation sets
its CSR inputs; writes are not cached or hoisted across invocations.

For a memory loop with two CSR parameters:

```bash
pixi run compile examples/megg_example/csr_scale/csr_scale.cadl \
  examples/megg_example/csr_scale/test_csr_scale.c /tmp/csr_scale.elf
```

Executing that binary requires hardware implementing the declared CSRs and
custom instruction.

## Best Practices

1. Use explicit bit widths for all operations
2. Match partition factor with unroll factor
3. Use burst transfers for large data movements
4. Minimize loop-carried dependencies
