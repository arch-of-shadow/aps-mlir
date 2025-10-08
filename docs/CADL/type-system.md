# CADL Type System

Complete reference for the CADL type system, which provides precise hardware modeling with explicit bit widths.

## Table of Contents

- [Overview](#overview)
- [Type Hierarchy](#type-hierarchy)
- [BasicType - Scalar Types](#basictype---scalar-types)
- [DataType - Composite Types](#datatype---composite-types)
- [CompoundType - Function Types](#compoundtype---function-types)
- [Literal System](#literal-system)
- [Type Inference](#type-inference)
- [Type Conversions](#type-conversions)

---

## Overview

CADL's type system is designed for hardware synthesis with these key principles:

1. **Explicit Bit Widths**: Every type has a known, fixed bit width
2. **No Implicit Conversions**: All type changes must be explicit
3. **Hardware Semantics**: Types map directly to hardware structures
4. **Static Typing**: All types are known at compile time

The type system consists of three layers:

```
CompoundType (highest level - function signatures)
    ↓
DataType (arrays and composite structures)
    ↓
BasicType (scalar types - fundamental building blocks)
```

---

## Type Hierarchy

### Python AST Class Hierarchy

The Python frontend implements types matching the Rust `type_sys_ir.rs`:

```python
# BasicType variants
BasicType_ApFixed(width: int)        # Signed fixed-point
BasicType_ApUFixed(width: int)       # Unsigned fixed-point
BasicType_Float32()                  # 32-bit float
BasicType_Float64()                  # 64-bit float
BasicType_String()                   # String type
BasicType_USize()                    # Architecture-dependent size

# DataType variants
DataType_Single(BasicType)           # Single value
DataType_Array(BasicType, List[int]) # Array with dimensions
DataType_Instance()                  # Instance type

# CompoundType variants
CompoundType_Basic(DataType)         # Basic compound type
CompoundType_FnTy(...)               # Function type (future)
```

---

## BasicType - Scalar Types

BasicType represents the fundamental scalar types in CADL.

### Unsigned Fixed-Point: `BasicType_ApUFixed`

**Syntax:** `u{N}` where N is the bit width (1 to any positive integer)

**AST Class:** `BasicType_ApUFixed(width: int)`

**Examples:**
```cadl
let x: u1 = 1;           // 1-bit unsigned (boolean-like)
let addr: u5 = 15;       // 5-bit unsigned (0-31)
let data: u32 = 1000;    // 32-bit unsigned
let wide: u64 = 0;       // 64-bit unsigned
```

**Common Widths:**
- `u1` - Single bit (often used for flags)
- `u5` - RISC-V register addresses (0-31)
- `u8`, `u16`, `u32`, `u64` - Standard byte-aligned widths
- `u7` - Custom width for opcodes/funct7

**Hardware Mapping:** Directly maps to unsigned hardware wires/registers of specified width.

### Signed Fixed-Point: `BasicType_ApFixed`

**Syntax:** `i{N}` where N is the bit width (1 to any positive integer)

**AST Class:** `BasicType_ApFixed(width: int)`

**Examples:**
```cadl
let temp: i8 = -42;      // 8-bit signed (-128 to 127)
let coord: i16 = -1000;  // 16-bit signed
let value: i32 = 0;      // 32-bit signed
```

**Representation:** Two's complement representation (standard for hardware).

**Hardware Mapping:** Signed hardware wires/registers with two's complement arithmetic.

### Float Types

#### 32-bit Float: `BasicType_Float32`

**Syntax:** `f32`

**AST Class:** `BasicType_Float32()`

**Format:** IEEE 754 single-precision (32 bits)
- 1 sign bit
- 8 exponent bits
- 23 mantissa bits

**Examples:**
```cadl
let pi: f32 = 3.14159;
let epsilon: f32 = 1.0e-6;
```

#### 64-bit Float: `BasicType_Float64`

**Syntax:** `f64`

**AST Class:** `BasicType_Float64()`

**Format:** IEEE 754 double-precision (64 bits)
- 1 sign bit
- 11 exponent bits
- 52 mantissa bits

**Examples:**
```cadl
let precise: f64 = 3.141592653589793;
let large: f64 = 1.0e100;
```

### Architecture Size: `BasicType_USize`

**Syntax:** `usize`

**AST Class:** `BasicType_USize()`

**Description:** Architecture-dependent unsigned integer size (typically 32 or 64 bits).

**Use Cases:**
- Array indices
- Size calculations
- Pointer-like values

**Example:**
```cadl
let index: usize = 0;
let count: usize = 100;
```

### String Type: `BasicType_String`

**Syntax:** `string`

**AST Class:** `BasicType_String()`

**Description:** String type for metadata and debugging (limited hardware synthesis support).

**Example:**
```cadl
let message: string = "hello";
```

---

## DataType - Composite Types

DataType builds on BasicType to create arrays and composite structures.

### Single Type: `DataType_Single`

**AST Class:** `DataType_Single(basic_type: BasicType)`

**Description:** Wraps a single BasicType value.

**Syntax:**
```cadl
u32              // DataType_Single(BasicType_ApUFixed(32))
i16              // DataType_Single(BasicType_ApFixed(16))
f32              // DataType_Single(BasicType_Float32())
```

**Examples:**
```cadl
let x: u32 = 100;        // Single u32 value
let y: i16 = -50;        // Single i16 value
let z: f32 = 3.14;       // Single f32 value
```

### Array Type: `DataType_Array`

**AST Class:** `DataType_Array(element_type: BasicType, dimensions: List[int])`

**Syntax:** `[{element_type}; {dim1}; {dim2}; ...]`

**Description:** Multi-dimensional arrays with specified element type and dimensions.

#### One-Dimensional Arrays

```cadl
let buffer: [u32; 8];           // 8 elements of u32
let coeffs: [i16; 16];          // 16 elements of i16
let flags: [u1; 32];            // 32 single-bit flags
```

**Memory Layout:** Contiguous array of `dim1` elements.

#### Multi-Dimensional Arrays

```cadl
let matrix: [u32; 4; 4];        // 4x4 matrix of u32
let image: [u8; 256; 256; 3];   // 256x256x3 RGB image
```

**Memory Layout:** Row-major order (rightmost dimension varies fastest).

#### Array Access

```cadl
// 1D indexing
let elem: u32 = buffer[3];

// 2D indexing
let cell: u32 = matrix[i, j];

// 3D indexing
let pixel: u8 = image[x, y, channel];
```

#### Static Array Initialization

```cadl
#[impl("regs")]
static lookup_table: [u32; 4] = {10, 20, 30, 40};

#[impl("regs")]
static thetas: [u32; 8] = {1474560, 870484, 459940, 233473, 117189, 58652, 29333, 14667};
```

### Instance Type: `DataType_Instance`

**AST Class:** `DataType_Instance()`

**Syntax:** `Instance`

**Description:** Special type for flow instantiation (future feature).

**Example:**
```cadl
let flow_inst: Instance;
```

---

## CompoundType - Function Types

CompoundType is used in function signatures and high-level type abstractions.

### Basic Compound Type: `CompoundType_Basic`

**AST Class:** `CompoundType_Basic(data_type: DataType)`

**Description:** Wraps a DataType for use in function arguments and compound contexts.

**Examples:**
```cadl
// Function arguments use CompoundType
rtype my_instr(rs1: u5, rs2: u5, rd: u5) {
    // rs1, rs2, rd are CompoundType_Basic(DataType_Single(BasicType_ApUFixed(5)))
}

flow process(input: [u32; 8], count: u32) {
    // input is CompoundType_Basic(DataType_Array(...))
    // count is CompoundType_Basic(DataType_Single(...))
}
```

### Function Type: `CompoundType_FnTy` (Future)

**AST Class:** `CompoundType_FnTy(args: List[CompoundType], ret: List[CompoundType])`

**Description:** Function type for higher-order functions (not yet fully implemented).

---

## Literal System

CADL literals have both a value and a type, matching the Rust `literal.rs` implementation.

### Literal Structure

```python
@dataclass
class Literal:
    lit: LiteralInner      # The value
    ty: BasicType          # The type
```

### LiteralInner Variants

#### Fixed-Point Literal: `LiteralInner_Fixed`

```python
@dataclass
class LiteralInner_Fixed(LiteralInner):
    value: int
```

**Used for:** All integer literals (signed and unsigned).

#### Float Literal: `LiteralInner_Float`

```python
@dataclass
class LiteralInner_Float(LiteralInner):
    value: float
```

**Used for:** All floating-point literals.

### Width-Specified Literals

CADL supports explicit bit-width specification in literals:

```cadl
// Binary
5'b101010         // Literal(LiteralInner_Fixed(42), BasicType_ApUFixed(5))

// Hexadecimal
8'hFF             // Literal(LiteralInner_Fixed(255), BasicType_ApUFixed(8))

// Decimal
15'd123           // Literal(LiteralInner_Fixed(123), BasicType_ApUFixed(15))

// Octal
3'o7              // Literal(LiteralInner_Fixed(7), BasicType_ApUFixed(3))
```

### Default Width Literals

Literals without width specification default to `u32`:

```cadl
42                // Literal(LiteralInner_Fixed(42), BasicType_ApUFixed(32))
0x1234            // Literal(LiteralInner_Fixed(4660), BasicType_ApUFixed(32))
0b1010            // Literal(LiteralInner_Fixed(10), BasicType_ApUFixed(32))
```

### Float Literals

```cadl
3.14159           // Literal(LiteralInner_Float(3.14159), BasicType_Float64())
1.0e-6            // Literal(LiteralInner_Float(0.000001), BasicType_Float64())
```

---

## Type Inference

CADL uses limited type inference in specific contexts:

### Let Bindings with Type Annotations

```cadl
let x: u32 = 42;         // Explicit type
let y = 5'b10101;        // Type inferred from literal (u5)
```

### Expression Type Propagation

```cadl
let a: u32 = 10;
let b: u32 = 20;
let sum = a + b;         // Type inferred as u32
```

### Function Arguments

```cadl
rtype add(rs1: u5, rs2: u5, rd: u5) {
    let r1 = _irf[rs1];  // Type inferred from _irf read (u32)
}
```

---

## Type Conversions

CADL requires explicit type conversions using cast operators:

### Cast Operators

```cadl
$signed(expr)       // Convert to signed
$unsigned(expr)     // Convert to unsigned
$f32(expr)          // Convert to f32
$f64(expr)          // Convert to f64
$int(expr)          // Convert to integer
$uint(expr)         // Convert to unsigned integer
```

### Examples

```cadl
let unsigned_val: u32 = 0xFFFFFFFF;
let signed_val: i32 = $signed(unsigned_val);  // -1

let x: u16 = 100;
let y: u32 = $uint(x);                        // Extend to u32

let float_val: f32 = 3.14;
let int_val: u32 = $uint(float_val);          // Convert to integer
```

### Bit Slicing (Implicit Conversion)

Bit slicing extracts portions of values:

```cadl
let value: u32 = 0x12345678;
let byte0: u8 = value[7:0];    // Extract low byte: 0x78
let byte3: u8 = value[31:24];  // Extract high byte: 0x12
let nibble: u4 = value[3:0];   // Extract low nibble (u4)
```

Bit slicing automatically determines the result type from the range width.

---

## Type Compatibility Rules

### Assignment Compatibility

```cadl
let x: u32 = 100;
let y: u32 = x;         // ✓ Same type

let z: u16 = x;         // ✗ Different widths - requires cast
let z: u16 = x[15:0];   // ✓ Explicit slice
```

### Arithmetic Type Rules

```cadl
let a: u32 = 10;
let b: u32 = 20;
let sum: u32 = a + b;   // ✓ Same types

let c: u16 = 5;
let d: u32 = a + c;     // ✗ Mixed widths - requires cast
let d: u32 = a + $uint(c);  // ✓ Explicit cast
```

### Array Element Type

```cadl
let arr: [u32; 8];
arr[0] = 100;           // ✓ u32 value to u32 array

let val: u16 = 50;
arr[1] = val;           // ✗ Type mismatch
arr[1] = $uint(val);    // ✓ With cast
```

---

## Hardware Implications

### Bit Width and Area

- Wider types consume more hardware resources
- Use minimum required bit width for efficiency

```cadl
// Good - uses only 5 bits
let reg_addr: u5 = 15;

// Wasteful - uses 32 bits for 0-31 range
let reg_addr: u32 = 15;
```

### Signed vs. Unsigned

- Signed arithmetic requires additional logic
- Use unsigned when sign is not needed

```cadl
let counter: u32 = 0;       // Good for counters (always ≥ 0)
let temperature: i16 = -5;  // Good when negative values needed
```

### Array Storage

- Arrays map to memory structures
- Multi-dimensional arrays use row-major layout
- Large arrays may require external memory

---

## Type System Implementation

### Python Classes Location

All type classes are defined in:
```
cadl_frontend/ast.py
```

### Helper Functions

```python
# Parse type from string
basic_type = parse_basic_type_from_string("u32")
# Returns: BasicType_ApUFixed(32)

# Parse literal with width
literal = parse_literal_from_string("5'b101010")
# Returns: Literal(LiteralInner_Fixed(42), BasicType_ApUFixed(5))
```

### Type Matching with Rust IR

The Python type system exactly matches:
```
cadl_rust/type_sys_ir.rs
cadl_rust/literal.rs
```

This ensures compatibility between the Python frontend and Rust backend.

---

## See Also

- [Grammar Reference](grammar.md) - Type syntax
- [Expressions](expressions.md) - Type conversions in expressions
- [Flows](flows.md) - Function argument types
- [Examples](examples.md) - Type usage examples

---

**Note**: The type system is designed for hardware synthesis and matches the MLIR type system for seamless lowering.
