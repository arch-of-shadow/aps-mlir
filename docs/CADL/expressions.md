# CADL Expressions

This document describes expressions in CADL, including operators, function calls, indexing, slicing, and special hardware-specific constructs.

## Expression Categories

### Primary Expressions

#### Literals

**Number Literals**
```cadl
42              // Decimal, type u32
0xFF            // Hexadecimal, type u32
0b1010          // Binary, type u32
0o755           // Octal, type u32
```

**Width-Specified Literals**
```cadl
5'b10101        // 5-bit binary (21), type u5
8'hFF           // 8-bit hex (255), type u8
16'd1000        // 16-bit decimal (1000), type u16
3'o7            // 3-bit octal (7), type u3
```

**Float Literals**
```cadl
3.14            // Float literal
1.5e-3          // Scientific notation
```

**Boolean Literals**
```cadl
true
false
```

**String Literals**
```cadl
"hello world"
```

#### Identifiers

Variable and parameter names:
```cadl
x               // Identifier
result
temp_val
```

#### Parenthesized Expressions

Group expressions with parentheses:
```cadl
(a + b)
(x * y, z)      // Tuple expression
```

### Binary Operators

CADL supports a comprehensive set of binary operators with C-like precedence.

#### Arithmetic Operators

```cadl
a + b           // Addition
a - b           // Subtraction
a * b           // Multiplication
a / b           // Division
a % b           // Remainder/modulo
```

#### Comparison Operators

```cadl
a == b          // Equality
a != b          // Inequality
a < b           // Less than
a <= b          // Less than or equal
a > b           // Greater than
a >= b          // Greater than or equal
```

#### Logical Operators

```cadl
a && b          // Logical AND
a || b          // Logical OR
```

#### Bitwise Operators

```cadl
a & b           // Bitwise AND
a | b           // Bitwise OR
a ^ b           // Bitwise XOR
a << n          // Left shift
a >> n          // Right shift (arithmetic for signed, logical for unsigned)
```

### Unary Operators

#### Arithmetic Unary

```cadl
-x              // Negation
```

#### Logical Unary

```cadl
!x              // Logical NOT
```

#### Bitwise Unary

```cadl
~x              // Bitwise NOT/complement
```

### Type Cast Operators

CADL provides explicit type casting operators:

```cadl
$signed(x)      // Cast to signed
$unsigned(x)    // Cast to unsigned
$f32(x)         // Cast to 32-bit float
$f64(x)         // Cast to 64-bit float
$int(x)         // Cast to integer
$uint(x)        // Cast to unsigned integer
```

**Example:**
```cadl
let a: i32 = $signed(my_unsigned_value);
let b: u32 = $unsigned(my_signed_value);
```

### Operator Precedence

From highest to lowest precedence:

1. **Primary** - Literals, identifiers, parentheses, function calls, indexing, slicing
2. **Unary** - `-`, `!`, `~`, type casts
3. **Multiplicative** - `*`, `/`, `%`
4. **Additive** - `+`, `-`
5. **Shift** - `<<`, `>>`
6. **Relational** - `<`, `<=`, `>`, `>=`
7. **Equality** - `==`, `!=`
8. **Bitwise XOR** - `^`
9. **Bitwise OR** - `|`
10. **Bitwise AND** - `&`
11. **Logical AND** - `&&`
12. **Logical OR** - `||`

**Examples:**
```cadl
a + b * c       // Equivalent to: a + (b * c)
a << 2 + 1      // Equivalent to: a << (2 + 1)
a && b || c     // Equivalent to: (a && b) || c
a & b == c      // Equivalent to: a & (b == c)
```

### Postfix Expressions

#### Function Calls

Call a function or flow:
```cadl
foo(x, y)       // Call function 'foo' with arguments x, y
bar()           // Call with no arguments
```

#### Array/Vector Indexing

Access elements of arrays or bit ranges:

**Single Index:**
```cadl
arr[0]          // First element
bits[5]         // Bit 5
mem_a[i]        // Dynamic index
```

**Multiple Indices (Multi-dimensional):**
```cadl
matrix[i, j]    // 2D array access
```

**Register File Access:**
```cadl
_irf[rs1]       // Read from CPU register file
_irf[rd] = x    // Write to CPU register file
```

**Memory Access:**
```cadl
_mem[addr]      // Read from memory
_mem[r1] = val  // Write to memory
```

#### Bit Slicing

Extract a range of bits:

**Standard Slice (high:low):**
```cadl
x[31:24]        // Bits 31 down to 24 (8 bits)
r1[15:0]        // Lower 16 bits
val[7:0]        // Lowest byte
```

**Range Slice (start +: length):**
```cadl
arr[0 +: ]      // All elements starting from 0
arr[start +: len]  // 'len' elements starting at 'start'
mem_a[i +: 4]   // 4 elements starting at index i
```

**Example combining slicing:**
```cadl
// Extract and combine byte fields
{r1[31:24], r1[23:16], r1[15:8], r1[7:0]}
```

### Conditional Expressions

#### If Expression

Ternary-like conditional:
```cadl
if condition {true_value} else {false_value}
```

**Examples:**
```cadl
let x: u32 = if a > b {a} else {b};  // Maximum
let sign: u1 = if val[31:31] {1} else {0};  // Sign bit
```

**Nested:**
```cadl
if x > 0 {
    1
} else {
    if x < 0 {-1} else {0}
}
```

#### Select Expression

Multi-way conditional selection (like switch/case with ordered evaluation):

```cadl
sel {
    condition1: value1,
    condition2: value2,
    condition3: value3,
    default_condition: default_value,
}
```

**Semantics:**
- Conditions are evaluated **in order** from first to last
- The first condition that evaluates to true determines the result
- The **last arm** serves as the default (its condition is ignored, only its value is used)
- At least one arm is required

**Examples:**

Basic categorization:
```cadl
let category: u32 = sel {
    value == 0: 0,
    value < 10: 1,
    value < 100: 2,
    value >= 100: 3,  // Last arm becomes default
};
```

With computed values:
```cadl
let result: u32 = sel {
    mode == 0: x + 10,
    mode == 1: x * 2,
    mode == 2: x << 1,
    mode >= 3: x,  // Default case
};
```

With complex conditions (use parentheses):
```cadl
let quadrant: u32 = sel {
    (x == 0) && (y == 0): 0,
    (x > 0) && (y > 0): 1,
    (x < 0) && (y > 0): 2,
    1: 3,  // Always true - acts as default
};
```

**Note:** Due to operator precedence in the current grammar, use parentheses around comparison operators in logical expressions:
- ✅ `(x == 0) && (y == 0)`
- ❌ `x == 0 && y == 0` (parsed incorrectly)

### Aggregate Expressions

Create composite values:

**Vector/Array Aggregate:**
```cadl
{a, b, c, d}        // 4-element aggregate
{1, 2, 3, 4}        // Literal aggregate
```

**Bit Concatenation:**
```cadl
{r1[31:24], r2[23:16], r1[15:8], r2[7:0]}  // Combine bit fields
{zr[15:0], zi[15:0]}  // Pack two 16-bit values into 32-bit
```

**Static Array Initialization:**
```cadl
static thetas: [u32; 8] = {1474560, 870484, 459940, 233473, 117189, 58652, 29333, 14667};
```

## Hardware-Specific Expressions

### Register File Access

Access the CPU register file:
```cadl
let r1: u32 = _irf[rs1];    // Read from register
_irf[rd] = result;           // Write to register
```

### Memory Access

Access system memory:
```cadl
let data: u32 = _mem[addr];      // Memory read
_mem[addr] = value;              // Memory write
let combined: u32 = _mem[r1 + r2];  // Computed address
```

### Burst Memory Access

Specialized burst read/write operations for efficient memory transfers:
```cadl
// Burst read: copy 16 words from memory address r1 to local array
mem_a[0 +: ] = _burst_read[r1 +: 16];

// Burst write: copy local array to memory address r1
_burst_write[r1 +: 16] = mem_a[0 +: ];
```

**Syntax:**
- `_burst_read[address +: count]` - Read 'count' elements from memory
- `_burst_write[address +: count] = source` - Write to memory

## Expression Examples

### Complex Arithmetic

```cadl
// Complex number multiplication: (ar + ai*i) * (br + bi*i)
let ar: i16 = r1[31:16];
let ai: i16 = r1[15:0];
let br: i16 = r2[31:16];
let bi: i16 = r2[15:0];
let zr: i32 = ar * br - ai * bi;  // Real part
let zi: i32 = ai * br + ar * bi;  // Imaginary part
_irf[rd] = {zr[15:0], zi[15:0]};
```

### SIMD-like Byte Addition

```cadl
// Add corresponding bytes in two 32-bit values
_irf[rd] = {
    (r1[31:24] + r2[31:24])[7:0],
    (r1[23:16] + r2[23:16])[7:0],
    (r1[15: 8] + r2[15: 8])[7:0],
    (r1[ 7: 0] + r2[ 7: 0])[7:0]
};
```

### Conditional Bit Manipulation

```cadl
// Conditional shift based on sign bit
let z_neg: u1 = z[31:31];
let x_: u32 = if z_neg {x + y_shift} else {x - y_shift};
let y_: u32 = if z_neg {y - x_shift} else {y + x_shift};
```

### Masking Operations

```cadl
// XOR with conditional mask
let a: u32 = x >> 1;
let x_: u32 = a ^ (0xEDB88320 & ~((x & 1) - 1));
```

## Type Inference

CADL performs type inference for expressions:

- Literals get types based on their suffix or default width
- Binary operations promote operands to a common type
- Slicing operations infer result width from bit range
- Aggregate expressions infer element types

**Examples:**
```cadl
5'b101         // Type: u5
32'd100        // Type: u32
x[7:0]         // Type: u8 (8 bits)
{a, b}         // Type inferred from a and b
```

## Best Practices

1. **Use Explicit Widths**: Specify bit widths for literals when precision matters
   ```cadl
   let mask: u8 = 8'hFF;  // Better than: 0xFF
   ```

2. **Parenthesize Complex Expressions**: Make precedence explicit
   ```cadl
   (a + b) * (c + d)  // Clear intent
   ```

3. **Name Intermediate Results**: Improve readability
   ```cadl
   let sum: u32 = a + b;
   let product: u32 = sum * c;
   ```

4. **Use Type Casts Explicitly**: Don't rely on implicit conversions
   ```cadl
   let signed_val: i32 = $signed(unsigned_input);
   ```

5. **Align Slice Widths**: Ensure slices match target types
   ```cadl
   let byte: u8 = val[7:0];  // 8-bit slice for u8
   ```

## See Also

- [Type System](type-system.md) - Data types and literals
- [Statements](statements.md) - Using expressions in statements
- [Grammar Reference](grammar.md) - Complete syntax
