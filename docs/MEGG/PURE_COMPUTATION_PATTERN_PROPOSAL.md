# Pure Computation Pattern Proposals for tests/diff_match

## Executive Summary

This document proposes a set of **pure computation patterns** (without control flow) to complement the existing control-flow-heavy patterns in `tests/diff_match`. These patterns are designed to:

1. **Leverage e-graph capabilities**: Algebraic simplification, CSE, strength reduction
2. **Represent classic ASIP applications**: Common in DSP, graphics, ML accelerators
3. **Be moderately simple**: Easy to understand but non-trivial enough to demonstrate value
4. **Fill gaps in current test coverage**: Focus on computation-only patterns

---

## Current State Analysis

### Existing tests/diff_match Patterns
All current patterns are **control-flow heavy**:
- `v3ddist_vs`: Loop with 3D distance calculation (squared differences + sum)
- `v3ddist_vv`: Vector-vector variant
- `vcovmat3d_vs`: Covariance matrix computation with loops
- `vgemv3d`: Matrix-vector multiplication with nested loops

### Existing tests/match/match_cases Patterns
Simple patterns exist but are basic:
- `01-arithmetic`: Simple add/sub/mul chain
- `02-bitwise`: and/or/xor operations
- `03-mux`: Compare + select

### Gap Identified
**Missing**: Classic computation patterns that showcase e-graph's strength in:
- Recognizing algebraically equivalent forms
- Common subexpression elimination across complex expressions
- Strength reduction (e.g., `x*2` → `x<<1`)
- Pattern matching across different implementations of the same algorithm

---

## Proposed Pattern Categories

### Category 1: Fused Multiply-Add Variants ⭐ **HIGH PRIORITY**

**Rationale**: FMA is the most common custom instruction in ASIPs for DSP/ML/Graphics

#### Pattern 1.1: Simple FMA (Fused Multiply-Add)
```mlir
func.func @fma(%a: i32, %b: i32, %c: i32) -> i32 {
  %mul = arith.muli %a, %b : i32
  %add = arith.addi %mul, %c : i32
  return %add : i32
}
```
**Application**: `result = a * b + c` (ubiquitous in matrix operations, convolutions)

**E-graph leverage**: Can match even if code computes `c + a*b` (commutative rewrite)

---

#### Pattern 1.2: MAC (Multiply-Accumulate)
```mlir
func.func @mac(%acc: i32, %a: i32, %b: i32) -> i32 {
  %mul = arith.muli %a, %b : i32
  %new_acc = arith.addi %acc, %mul : i32
  return %new_acc : i32
}
```
**Application**: Accumulator-based operations (dot product, convolution)

**Difference from FMA**: Semantically emphasizes accumulation pattern

---

#### Pattern 1.3: MSUB (Multiply-Subtract)
```mlir
func.func @msub(%a: i32, %b: i32, %c: i32) -> i32 {
  %mul = arith.muli %a, %b : i32
  %sub = arith.subi %mul, %c : i32
  return %sub : i32
}
```
**Application**: `result = a * b - c` (common in numerical methods)

**E-graph leverage**: Can rewrite from `c - a*b` → `-(a*b - c)`

---

#### Pattern 1.4: NMSUB (Negated Multiply-Subtract)
```mlir
func.func @nmsub(%a: i32, %b: i32, %c: i32) -> i32 {
  %mul = arith.muli %a, %b : i32
  %sub = arith.subi %c, %mul : i32  // c - a*b
  return %sub : i32
}
```
**Application**: `result = c - a * b` (FIR filters, error calculations)

---

### Category 2: Arithmetic Optimization Patterns ⭐ **HIGH PRIORITY**

**Rationale**: Demonstrate strength reduction and algebraic identities

#### Pattern 2.1: Average (Arithmetic Mean)
```mlir
func.func @avg_shift(%a: i32, %b: i32) -> i32 {
  %sum = arith.addi %a, %b : i32
  %c1 = arith.constant 1 : i32
  %avg = arith.shrui %sum, %c1 : i32  // (a+b) >> 1
  return %avg : i32
}
```
**Application**: Fast average without division (image processing, interpolation)

**E-graph leverage**: Can match equivalent `(a+b)/2` via strength reduction rewrite

**Alternative form** (tests algebraic equivalence):
```mlir
func.func @avg_div(%a: i32, %b: i32) -> i32 {
  %sum = arith.addi %a, %b : i32
  %c2 = arith.constant 2 : i32
  %avg = arith.divui %sum, %c2 : i32
  return %avg : i32
}
```

---

#### Pattern 2.2: Squared Difference
```mlir
func.func @sqdiff(%a: i32, %b: i32) -> i32 {
  %diff = arith.subi %a, %b : i32
  %sq = arith.muli %diff, %diff : i32
  return %sq : i32
}
```
**Application**: MSE calculation, distance metrics (used in v3ddist but in loop)

**E-graph leverage**:
- CSE if `%diff` used elsewhere
- Can expand to `a*a + b*b - 2*a*b` and match that form too

**Expanded algebraic form** (tests equivalence detection):
```mlir
func.func @sqdiff_expanded(%a: i32, %b: i32) -> i32 {
  %a2 = arith.muli %a, %a : i32
  %b2 = arith.muli %b, %b : i32
  %ab = arith.muli %a, %b : i32
  %c2 = arith.constant 2 : i32
  %ab2 = arith.muli %ab, %c2 : i32
  %sum = arith.addi %a2, %b2 : i32
  %result = arith.subi %sum, %ab2 : i32
  return %result : i32
}
```

---

#### Pattern 2.3: 3-Term Dot Product (Small Vector Dot)
```mlir
func.func @dot3(%a0: i32, %a1: i32, %a2: i32,
                %b0: i32, %b1: i32, %b2: i32) -> i32 {
  %p0 = arith.muli %a0, %b0 : i32
  %p1 = arith.muli %a1, %b1 : i32
  %p2 = arith.muli %a2, %b2 : i32
  %sum01 = arith.addi %p0, %p1 : i32
  %sum012 = arith.addi %sum01, %p2 : i32
  return %sum012 : i32
}
```
**Application**: 3D vector dot product (graphics, physics)

**E-graph leverage**:
- CSE of multiplications
- Can match different association orders: `(p0+p1)+p2` vs `p0+(p1+p2)`

---

#### Pattern 2.4: Linear Interpolation (LERP)
```mlir
func.func @lerp(%a: i32, %b: i32, %t: i32) -> i32 {
  %diff = arith.subi %b, %a : i32
  %scaled = arith.muli %diff, %t : i32
  %result = arith.addi %a, %scaled : i32
  return %result : i32  // a + t*(b-a)
}
```
**Application**: Animation, color interpolation, numerical methods

**E-graph leverage**: Can match equivalent forms:
- `a + t*b - t*a` (distributive)
- `(1-t)*a + t*b` (after rewrite)

---

### Category 3: Bitwise Computation Patterns

**Rationale**: Common in embedded systems, crypto, data processing

#### Pattern 3.1: Clear Lowest Set Bit
```mlir
func.func @clear_lowest_bit(%x: i32) -> i32 {
  %c1 = arith.constant 1 : i32
  %x_minus_1 = arith.subi %x, %c1 : i32
  %result = arith.andi %x, %x_minus_1 : i32
  return %result : i32  // x & (x-1)
}
```
**Application**: Bit manipulation, population count algorithms

**E-graph leverage**: Classic bit-twiddling idiom recognition

---

#### Pattern 3.2: Isolate Lowest Set Bit
```mlir
func.func @isolate_lowest_bit(%x: i32) -> i32 {
  %neg_x = arith.subi %c0, %x : i32  // -x (two's complement)
  %result = arith.andi %x, %neg_x : i32
  return %result : i32  // x & (-x)
}
```
**Application**: Fast bit operations, binary search on bits

---

#### Pattern 3.3: Byte Swap (16-bit)
```mlir
func.func @bswap16(%x: i32) -> i32 {
  %c8 = arith.constant 8 : i32
  %c0xff = arith.constant 255 : i32
  %c0xff00 = arith.constant 65280 : i32

  %low_byte = arith.andi %x, %c0xff : i32
  %high_byte = arith.andi %x, %c0xff00 : i32

  %low_shifted = arith.shli %low_byte, %c8 : i32
  %high_shifted = arith.shrui %high_byte, %c8 : i32

  %result = arith.ori %low_shifted, %high_shifted : i32
  return %result : i32
}
```
**Application**: Endianness conversion, network protocols

**E-graph leverage**: Complex pattern with multiple subexpressions

---

### Category 4: Saturation & Clamping Patterns

**Rationale**: Common in signal processing and image/video codecs

#### Pattern 4.1: Saturating Add (with min/max)
```mlir
func.func @sat_add(%a: i32, %b: i32, %max_val: i32) -> i32 {
  %sum = arith.addi %a, %b : i32
  %cmp = arith.cmpi ult, %sum, %max_val : i32
  %result = arith.select %cmp, %sum, %max_val : i32
  return %result : i32  // min(a+b, max_val)
}
```
**Application**: Fixed-point arithmetic, audio processing

---

#### Pattern 4.2: Clamp (to range)
```mlir
func.func @clamp(%x: i32, %min_val: i32, %max_val: i32) -> i32 {
  %cmp_min = arith.cmpi slt, %x, %min_val : i32
  %clamped_low = arith.select %cmp_min, %min_val, %x : i32

  %cmp_max = arith.cmpi sgt, %clamped_low, %max_val : i32
  %result = arith.select %cmp_max, %max_val, %clamped_low : i32
  return %result : i32  // clamp(x, min, max)
}
```
**Application**: Color quantization, sensor data normalization

**E-graph leverage**: Two-stage select pattern (nested conditionals without scf.if)

---

### Category 5: Polynomial Evaluation Patterns

**Rationale**: Approximations of math functions, checksums

#### Pattern 5.1: Horner's Method (degree 2)
```mlir
func.func @horner_deg2(%x: i32, %c0: i32, %c1: i32, %c2: i32) -> i32 {
  // Evaluates: c2*x^2 + c1*x + c0
  // Horner form: c0 + x*(c1 + x*c2)
  %inner = arith.muli %x, %c2 : i32
  %inner_plus_c1 = arith.addi %inner, %c1 : i32
  %outer = arith.muli %x, %inner_plus_c1 : i32
  %result = arith.addi %outer, %c0 : i32
  return %result : i32
}
```
**Application**: Fast polynomial evaluation (Taylor series, checksums)

**E-graph leverage**: Can match both Horner and naive `c0 + c1*x + c2*x*x` forms

---

#### Pattern 5.2: CRC-style XOR Chain
```mlir
func.func @crc_step(%crc: i32, %data: i32, %poly: i32) -> i32 {
  %xor1 = arith.xori %crc, %data : i32
  %c1 = arith.constant 1 : i32
  %shifted = arith.shrui %xor1, %c1 : i32
  %xor2 = arith.xori %shifted, %poly : i32
  return %xor2 : i32
}
```
**Application**: Checksums, error detection codes

---

### Category 6: Specialized DSP Patterns

**Rationale**: Common in signal processing ASIPs

#### Pattern 6.1: Absolute Difference
```mlir
func.func @abs_diff(%a: i32, %b: i32) -> i32 {
  %diff = arith.subi %a, %b : i32
  %c0 = arith.constant 0 : i32
  %is_neg = arith.cmpi slt, %diff, %c0 : i32
  %neg_diff = arith.subi %c0, %diff : i32
  %result = arith.select %is_neg, %neg_diff, %diff : i32
  return %result : i32  // |a - b|
}
```
**Application**: SAD (Sum of Absolute Differences) in video encoding

---

#### Pattern 6.2: Sign Extension (8-bit to 32-bit)
```mlir
func.func @sign_extend_i8(%x: i32) -> i32 {
  %c0xff = arith.constant 255 : i32
  %c0x80 = arith.constant 128 : i32
  %c0xffffff00 = arith.constant -256 : i32

  %low_byte = arith.andi %x, %c0xff : i32
  %is_neg = arith.cmpi uge, %low_byte, %c0x80 : i32
  %extended = arith.ori %low_byte, %c0xffffff00 : i32
  %result = arith.select %is_neg, %extended, %low_byte : i32
  return %result : i32
}
```
**Application**: Data type conversions, packed arithmetic

---

## Recommended Implementation Priority

### Tier 1: Must-Have (Immediate Value) ⭐⭐⭐
1. **Pattern 1.1**: FMA (most common ASIP instruction)
2. **Pattern 1.2**: MAC (accumulator variant)
3. **Pattern 2.2**: Squared Difference (complements existing v3ddist)
4. **Pattern 2.3**: 3-Term Dot Product (common vector operation)

**Rationale**: These are ubiquitous in real ASIP designs and demonstrate clear value

---

### Tier 2: High Value (Good Demonstrations) ⭐⭐
5. **Pattern 2.1**: Average (strength reduction showcase)
6. **Pattern 2.4**: LERP (algebraic equivalence)
7. **Pattern 4.2**: Clamp (nested select pattern)
8. **Pattern 6.1**: Absolute Difference (DSP common)

**Rationale**: Show different e-graph capabilities (strength reduction, equivalence, nested patterns)

---

### Tier 3: Nice-to-Have (Completeness) ⭐
9. **Pattern 3.1**: Clear Lowest Bit (classic bit hack)
10. **Pattern 5.1**: Horner's Method (polynomial eval)
11. **Pattern 1.3**: MSUB (FMA variant)

**Rationale**: Educational value, demonstrate pattern diversity

---

## Implementation Guidelines

### Directory Structure
```
tests/diff_match/
├── fma/                    # Pattern 1.1
│   ├── fma.c
│   ├── fma.mlir           # Pattern definition
│   ├── fma.cadl
│   ├── fma.json
│   └── compile.sh
├── mac/                    # Pattern 1.2
├── sqdiff/                 # Pattern 2.2
├── dot3/                   # Pattern 2.3
└── ...
```

### File Templates

**Example: `fma.c`**
```c
#include <stdint.h>

uint8_t fma(uint32_t *rs1, uint32_t *rs2) {
    uint32_t a = rs1[0];
    uint32_t b = rs1[1];
    uint32_t c = rs1[2];

    // Pattern: a * b + c
    uint32_t result = a * b + c;

    rs2[0] = result;
    return 0;
}
```

**Example: `fma.cadl` (minimal)**
```cadl
instruction fma (rs1: uint32[3], rs2: uint32[1]) -> uint8 {
    let a = rs1[0];
    let b = rs1[1];
    let c = rs1[2];
    rs2[0] = a * b + c;
    return 0u8;
}
```

**Example: `fma.json`** (encoding)
```json
{
  "fma": {
    "opcode": "0001011",
    "funct3": "000",
    "funct7": "0000001"
  }
}
```

---

## E-graph Leverage Analysis

### How Each Pattern Demonstrates E-graph Value

| Pattern | E-graph Capability | Demonstration |
|---------|-------------------|---------------|
| FMA | Commutativity | Matches `a*b+c` and `c+a*b` |
| MAC | CSE | Reuses intermediate `%mul` |
| Squared Diff | Algebraic expansion | `(a-b)^2` ↔ `a²+b²-2ab` |
| Dot3 | Associativity | Different sum orders |
| Average | Strength reduction | `(a+b)/2` ↔ `(a+b)>>1` |
| LERP | Distributive law | `a+t*(b-a)` ↔ `a+t*b-t*a` |
| Clamp | Nested pattern | Composite select chains |
| Abs Diff | CSE + select | Common subexpression in branches |

---

## Testing Strategy

### Test Case Structure
For each pattern, create:
1. **Exact match**: Code identical to pattern
2. **Algebraic variant**: Equivalent but different form (tests e-graph rewrite)
3. **CSE variant**: Pattern embedded with shared subexpressions

**Example for FMA**:

**Test 1**: `fma_exact.mlir` (identical to pattern)
```mlir
%mul = arith.muli %a, %b : i32
%add = arith.addi %mul, %c : i32
```

**Test 2**: `fma_commuted.mlir` (tests commutativity)
```mlir
%mul = arith.muli %a, %b : i32
%add = arith.addi %c, %mul : i32  // c + a*b instead of a*b + c
```

**Test 3**: `fma_with_cse.mlir` (tests CSE)
```mlir
%mul = arith.muli %a, %b : i32
%add1 = arith.addi %mul, %c : i32   // FMA
%add2 = arith.addi %mul, %d : i32   // Reuses %mul
```

---

## Success Metrics

A pattern is successfully implemented when:

1. ✅ **Pattern extraction**: Megg correctly builds skeleton/pattern
2. ✅ **Exact match**: Identical code matches 100%
3. ✅ **Algebraic match**: Equivalent forms match (after rewrites)
4. ✅ **CSE benefit**: Shared subexpressions handled correctly
5. ✅ **RISC-V generation**: Generates correct `.insn` encoding
6. ✅ **Disassembly verification**: Custom instruction visible in `.asm`

---

## Integration with Existing Tests

### Relationship to tests/match/match_cases
- **match_cases**: Test *matching engine* correctness (control flow, edge cases)
- **diff_match**: Test *realistic ASIP patterns* for end-to-end compilation

### Relationship to existing diff_match patterns
- **Current**: Control-flow heavy (loops + computation)
- **Proposed**: Pure computation (no loops, leverage algebraic rewrites)
- **Complementary**: Together provide full coverage of ASIP pattern types

---

## Future Extensions

Once pure computation patterns are established:

1. **Hybrid patterns**: Combine computation patterns with minimal control flow
   - Example: FMA inside a simple `scf.if` (predicated execution)

2. **Chained patterns**: Multiple computation patterns in sequence
   - Example: `FMA → LERP → CLAMP` pipeline

3. **SIMD variants**: Vector versions of scalar patterns
   - Example: `vec_fma` operating on vectors

4. **Floating-point variants**: FP32/FP64 versions
   - Example: `fmaf` for `f32` FMA

---

## Conclusion

These pure computation patterns will:

✅ **Fill a gap** in current test coverage (computation-only patterns)
✅ **Demonstrate e-graph strength** (algebraic rewrites, CSE, strength reduction)
✅ **Represent real ASIP needs** (FMA, MAC, dot product are ubiquitous)
✅ **Be easy to implement** (no control flow complexity)
✅ **Be educational** (classic algorithms like Horner, bit hacks)

**Recommended starting point**: Implement **Tier 1** patterns (FMA, MAC, Squared Diff, Dot3) first to establish value, then expand to Tier 2/3 based on needs.

---

**Document Version**: 1.0
**Date**: 2025-01-09
**Status**: Proposal (awaiting review)
