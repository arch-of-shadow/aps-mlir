# CADL Affine Compatibility Guide

## Overview

This document explains how to write CADL code that is compatible with MLIR's affine dialect and array partitioning optimizations. When running the affine transformation pipeline, non-affine array access patterns will generate warnings like:

```
warning: global variable "array_name" with applying array_partition pragma is failed, because non standard affine access.
```

The affine pipeline being tested:
```bash
./build/tools/aps-opt/aps-opt \
  --memory-map \
  --scf-for-index-cast \
  --aps-mem-to-memref \
  --canonicalize \
  --raise-scf-to-affine \
  --canonicalize \
  --affine-raise-from-memref \
  --canonicalize \
  --hls-unroll \
  --canonicalize \
  --affine-loop-normalize \
  --canonicalize \
  --new-array-partition \
  --canonicalize
```

## What is Affine Access?

**Affine access** means array indices can be expressed as linear combinations of loop variables plus constants:
- `array[i]` ✅ Direct loop variable
- `array[2*i + 3]` ✅ Linear: `a*i + b` where a,b are constants
- `array[i + j]` ✅ Multiple loop variables (linear)
- `array[CONST]` ✅ Constant index

**Non-affine access** includes:
- `array[i / 8]` ❌ Division by non-constant
- `array[i * j]` ❌ Multiplication of two variables
- `array[f(i)]` ❌ Where f is non-linear or data-dependent
- `array[vidx]` ❌ Where vidx is computed from array data

## Analysis of Example Files

### ✅ SUCCESS: v3ddist_vs.cadl (0 warnings)

**Pattern**: Simple linear indexing only

**Key characteristics**:
```cadl
[[unroll(4)]]
with i: u32 = (0, i_) do {
    let x: u32 = points_x[i];      // ✅ array[i]
    let y: u32 = points_y[i];      // ✅ array[i]
    let z: u32 = points_z[i];      // ✅ array[i]

    // ... computation ...

    dist_out[i] = dist_sq;         // ✅ array[i]
    let i_: u32 = i + 1;
} while (i_ < vl);
```

**Why it succeeds**:
- All array accesses use the loop variable `i` directly
- No arithmetic operations on indices
- Simple affine map: `array[i]` → `affine_map<(d0) -> (d0)>`

**Location**: [examples/pcl/v3ddist_vs.cadl:47-64](../examples/pcl/v3ddist_vs.cadl#L47-L64)

---

### ✅ SUCCESS: vcovmat3d.cadl (0 warnings)

**Pattern**: Constant indexing (no loops on array accesses)

**Key characteristics**:
```cadl
// No loops - all accesses are constant indices
let x: i32 = points[0];    // ✅ Constant
let y: i32 = points[1];    // ✅ Constant
let z: i32 = points[2];    // ✅ Constant
// ...
cov_out[0] = dx * dx;      // ✅ Constant
cov_out[1] = dx * dy;      // ✅ Constant
// ...
```

**Why it succeeds**:
- All array indices are compile-time constants
- No loop-dependent indexing
- Trivially affine

**Location**: [examples/pcl/vcovmat3d.cadl:30-61](../examples/pcl/vcovmat3d.cadl#L30-L61)

---

### ⚠️ PARTIAL: vgemv3d.cadl (0 warnings after fixes)

**Pattern**: Multi-dimensional array flattening with `i*CONST+j`

**Previous problematic pattern** (would cause warnings):
```cadl
// This would be non-affine if matrix is 2D:
matrix[i][j]  // If lowered incorrectly

// But flattened 1D with linear expression is affine:
matrix[i * 4 + j]  // ✅ Affine: (d0, d1) -> (d0 * 4 + d1)
```

**Current implementation**:
```cadl
with i: u32 = (0, i_) do {
    acc = 0;
    [[unroll(4)]]
    with j: u32 = (0, j_) do {
        acc = acc + matrix[i * 4 + j] * vec[j];  // ✅ Linear in i,j
        let j_: u32 = j + 1;
    } while (j_ < 4);
    result[i] = acc;
    let i_: u32 = i + 1;
} while (i_ < 4);
```

**Why it now succeeds**:
- Expression `i * 4 + j` is affine: multiplication by constant is allowed
- Unrolling the inner loop may help eliminate the index computation
- Complete partitioning allows parallel access

**Note**: If there were warnings previously, they were likely fixed by adjusting partition attributes or loop unrolling.

**Location**: [examples/pcl/vgemv3d.cadl:36-47](../examples/pcl/vgemv3d.cadl#L36-L47)

---

### ⚠️ PARTIAL: vfpsmax.cadl (0 warnings after fixes)

**Pattern**: Tree reduction with double-buffering

**Challenge**: Data-dependent access in reduction tree
```cadl
// Stage 1: Read from _in buffers
let idx1: u32 = i * 2;
let idx2: u32 = i * 2 + 1;
let val1: u32 = reduction_vals_in[idx1];   // ✅ i*2 is affine
let val2: u32 = reduction_vals_in[idx2];   // ✅ i*2+1 is affine

// Stage 2: Alternate buffers (_out to _in)
let val1: u32 = reduction_vals_out[idx1];  // ✅ Linear
```

**Why it now succeeds**:
- Index expressions like `i*2` and `i*2+1` are affine (multiplication by constant)
- Double-buffering pattern (_in↔_out) avoids read-after-write hazards
- Complete unrolling (`[[unroll(4)]]`, `[[unroll(2)]]`) may eliminate dynamic indexing
- All accesses are ultimately affine maps

**Previous issues likely fixed by**:
- Proper unroll factors matching access patterns
- Complete array partitioning for small reduction buffers
- Simplified addressing (no data-dependent vidx in reduction stages)

**Location**: [examples/pcl/vfpsmax.cadl:54-135](../examples/pcl/vfpsmax.cadl#L54-L135)

---

### ❌ FAILURE: deca_decompress.cadl (3 warnings)

**Warnings generated**:
1. `warning: global variable "bitmask" with applying array_partition pragma is failed`
2. `warning: global variable "values" with applying array_partition pragma is failed`
3. `warning: global variable "scales" with applying array_partition pragma is failed`

**Problematic patterns**:

#### 1. Division-based indexing (bitmask)
```cadl
// Line 61-63
let byte_idx: u32 = idx / 8;           // ❌ Division creates non-affine
let mask_byte: u8 = bitmask[byte_idx]; // ❌ array[idx/8]
```

**Why it fails**:
- `idx / 8` is NOT affine (division by variable)
- MLIR affine dialect requires indices be polynomial of degree 1
- Division would need to be by a power-of-2 constant AND recognized as a shift

**Lowered MLIR**:
```mlir
%c8_i32 = arith.constant 8 : i32
%9 = arith.divui %arg3, %c8_i32 : i32    // Non-affine operation
%12 = aps.memload %2[%9] : memref<64xi8> // Uses non-affine index
```

#### 2. Division with bit-slice (scales)
```cadl
// Line 68-69
let group_idx: u8 = (idx / 16)[4:0];   // ❌ Division + bit extraction
let scale: i16 = scales[group_idx];    // ❌ array[(idx/16)[4:0]]
```

**Why it fails**:
- `idx / 16` is non-affine (division)
- Bit slice `[4:0]` adds another transformation
- Double non-affine transformation

**Lowered MLIR**:
```mlir
%c16_i32 = arith.constant 16 : i32
%14 = arith.divui %arg3, %c16_i32 : i32  // Non-affine
%15 = comb.extract %14 from 0 : (i32) -> i5
%16 = arith.extui %15 : i5 to i8
%17 = aps.memload %6[%16] : memref<32xi16> // Non-affine index
```

#### 3. Data-dependent indexing (values)
```cadl
// Line 56-58, 72, 84-85
with idx: u32 = (0, idx + 1)
     vidx: u32 = (0, vidx_next)          // ❌ vidx is loop-carried
do {
    let sparse_val: i8 = values[vidx];   // ❌ array[data_dependent_var]
    // ...
    let inc_val: u32 = if is_nonzero {1} else {0};
    let vidx_next: u32 = vidx + inc_val; // ❌ Updates based on data
}
```

**Why it fails**:
- `vidx` increments conditionally based on bitmask data
- Not expressible as `f(idx)` where f is affine
- Data-dependent control flow breaks affine analysis

**Lowered MLIR**:
```mlir
%7 = scf.for %arg3 = %c0_i32 to %c512_i32 step %c1_i32
      iter_args(%arg4 = %c0_i32) -> (i32) {  // %arg4 is vidx
    // ...
    %18 = aps.memload %4[%arg4] : memref<256xi8> // Uses data-dependent vidx
    // ...
    %27 = arith.addi %arg4, %26 : i32  // Conditional increment
    scf.yield %27 : i32
}
```

**Location**: [examples/deca/deca_decompress.cadl:54-87](../examples/deca/deca_decompress.cadl#L54-L87)

---

## Patterns to AVOID (Non-Affine)

### 1. Division/Modulo by Loop Variables
```cadl
❌ let idx: u32 = i / 8;
❌ let idx: u32 = i % 16;
❌ array[i / CONST]  // Even division by constant is problematic
❌ array[i % CONST]
```

### 2. Data-Dependent Indices
```cadl
❌ with vidx: u32 = (0, vidx_next) do {
    let val: i8 = array[vidx];  // vidx updated based on data
    vidx_next = vidx + (condition ? 1 : 0);
}
```

### 3. Bit Extraction in Indices
```cadl
❌ let idx: u8 = (i / 16)[4:0];
❌ array[value[7:4]]
```

### 4. Multiplication of Two Loop Variables
```cadl
❌ array[i * j]  // Both i and j are loop variables
```

### 5. Non-Linear Functions
```cadl
❌ array[i * i]     // Quadratic
❌ array[2 ** i]    // Exponential
❌ array[log2(i)]   // Logarithmic
```

---

## Patterns to USE (Affine)

### 1. Direct Loop Variable Indexing
```cadl
✅ array[i]
✅ array[j]
```

### 2. Linear Combinations with Constants
```cadl
✅ array[2*i + 3]
✅ array[i*4 + j]      // i*CONST + j
✅ array[3*i - 2*j + 5]
```

### 3. Constant Indexing
```cadl
✅ array[0]
✅ array[42]
```

### 4. Multiple Induction Variables (all linear)
```cadl
✅ with i: u32 = (0, i+1) do {
    with j: u32 = (0, j+1) do {
        array[i*COL + j]  // Flattened 2D access
    }
}
```

---

## Recommended Transformations for deca_decompress.cadl

### Option 1: Complete Loop Unrolling

Since the loop bound is constant (512), consider full unrolling:

```cadl
// Instead of:
with idx: u32 = (0, idx + 1) do { ... } while idx + 1 < 512;

// Could unroll to 512 explicit statements (impractical but affine)
// Or use aggressive unroll pragma to let HLS tool handle it
[[unroll(512)]]
```

**Pros**: Eliminates all loop variables, making all accesses constant
**Cons**: Huge code size, may exceed HLS tool limits

---

### Option 2: Lookup Table for Group Indices

Pre-compute group_idx mapping:

```cadl
// Add static lookup table
static group_lut: [u8; 512] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // idx 0-15 → group 0
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  // idx 16-31 → group 1
    // ... (generate full table)
];

// Then in loop:
let group_idx: u8 = group_lut[idx];  // ✅ Affine: array[i]
let scale: i16 = scales[group_idx];  // Still non-affine (data-dependent)
```

**Limitation**: `scales[group_idx]` is still data-dependent

---

### Option 3: Restructure to Separate Affine and Non-Affine Parts

Split the algorithm into two passes:

**Pass 1: Affine bitmask expansion** (generate dense mask)
```cadl
static dense_mask: [u1; 512];

[[unroll(64)]]  // 512 / 8 = 64 bytes
with byte_i: u32 = (0, byte_i + 1) do {
    let mask_byte: u8 = bitmask[byte_i];  // ✅ Affine
    // Unroll bit extraction
    dense_mask[byte_i * 8 + 0] = mask_byte[0:0];
    dense_mask[byte_i * 8 + 1] = mask_byte[1:1];
    // ... expand all 8 bits
} while byte_i + 1 < 64;
```

**Pass 2: Use dense mask** (still data-dependent on vidx)
```cadl
with idx: u32 = (0, idx + 1), vidx: u32 = (0, vidx_next) do {
    let is_nonzero: u1 = dense_mask[idx];  // ✅ Affine
    let group_idx: u8 = idx / 16;  // Still problematic
    // ...
}
```

**Limitation**: Still has vidx problem and division

---

### Option 4: Tile-Based Processing with Fixed Groups

Process groups sequentially instead of all elements:

```cadl
// Process 32 groups of 16 elements each
with group: u32 = (0, group + 1) do {
    let scale: i16 = scales[group];  // ✅ Affine: array[group]

    [[unroll(16)]]
    with elem: u32 = (0, elem + 1) do {
        let idx: u32 = group * 16 + elem;  // ✅ Affine
        let byte_idx: u32 = idx / 8;       // Still division issue
        // ...
    } while elem + 1 < 16;
} while group + 1 < 32;
```

**Pros**: Simplifies scale indexing
**Cons**: Still has bitmask division problem, nested loops

---

### Option 5: Bit-Level Unrolling (Recommended)

Unroll the byte-level loop and explicitly handle bits:

```cadl
[[unroll(64)]]  // 512 elements / 8 bits per byte
with byte_i: u32 = (0, byte_i + 1), vidx: u32 = (0, vidx_) do {
    let mask_byte: u8 = bitmask[byte_i];  // ✅ Affine

    // Explicitly unroll 8 bits
    let base_idx: u32 = byte_i * 8;

    // Bit 0
    let is_nz_0: u1 = mask_byte[0:0];
    let val_0: i8 = values[vidx];  // Still data-dependent
    let group_0: u8 = (base_idx / 16)[4:0];  // Still division
    // ... process element 0

    // Bit 1
    let is_nz_1: u1 = mask_byte[1:1];
    let val_1: i8 = values[vidx + is_nz_0];
    // ... process element 1

    // ... repeat for bits 2-7
}
```

**Pros**: Eliminates inner loop, makes bit extraction explicit
**Cons**:
- Still has data-dependent `values[vidx]` access
- Still has division for group_idx
- Very verbose

---

### Option 6: Accept Non-Affine Warnings (Pragmatic)

**Reality check**: DECA decompression is inherently non-affine due to:
1. **Sparse format**: Compressed index tracking is fundamentally data-dependent
2. **Variable-length codes**: Number of values read depends on bitmask
3. **Group-wise scaling**: Requires division to determine group

**Recommendation**:
- Keep partition pragmas as hints to HLS tools
- Warnings indicate array partitioning may not be optimal, but doesn't prevent synthesis
- Focus optimization on:
  - Burst memory access (already done)
  - Unroll factors for parallelism
  - Pipelining the main loop

**Partition strategy**:
```cadl
// bitmask: Complete partition (64 elements is small)
#[partition_factor_array([64])]  // Full partition

// values: Cyclic for streaming access
#[partition_factor_array([4])]   // Partial partition

// scales: Complete partition (32 elements)
#[partition_factor_array([32])]  // Full partition
```

**Why this is acceptable**:
- Affine warnings don't prevent hardware generation
- Non-affine access patterns still synthesize (just may not get optimal banking)
- The algorithm's inherent structure (sparse decompression) is naturally irregular

---

## Summary Table

| Example | Warnings | Key Pattern | Affine? | Notes |
|---------|----------|-------------|---------|-------|
| v3ddist_vs | 0 | `array[i]` | ✅ | Simple direct indexing |
| vcovmat3d | 0 | `array[CONST]` | ✅ | No loops on accesses |
| vgemv3d | 0 | `array[i*4+j]` | ✅ | Linear multi-dim (after fixes) |
| vfpsmax | 0 | `array[i*2]`, `array[i*2+1]` | ✅ | Tree reduction with constant strides (after fixes) |
| deca_decompress | 3 | `array[i/8]`, `array[vidx]` | ❌ | Division + data-dependent |

---

## Quick Checklist for Affine-Compatible CADL

- [ ] Array indices only use loop variables with addition/subtraction
- [ ] Multiplication only with constants (e.g., `i*4`, not `i*j`)
- [ ] No division or modulo with loop variables
- [ ] No bit extraction in index calculations
- [ ] No data-dependent indexing (where index depends on array values)
- [ ] Loop bounds are constants or known at compile time
- [ ] Consider unrolling small loops to eliminate variables

---

## References

- **Affine Dialect Documentation**: [MLIR Affine Dialect](https://mlir.llvm.org/docs/Dialects/Affine/)
- **Polyhedral Compilation**: Affine analysis enables loop transformations like tiling, fusion, and vectorization
- **Hardware Synthesis**: Array partitioning requires predictable access patterns for banking

---

## Conclusion

The PCL examples (v3ddist_vs, vcovmat3d, vgemv3d, vfpsmax) demonstrate that **regular, affine-compatible computation patterns** can be achieved by:

1. Using direct loop variable indexing
2. Applying constant stride patterns
3. Unrolling when necessary
4. Avoiding division and data-dependent access

The DECA decompression example shows that **some algorithms are inherently non-affine** due to their sparse/irregular nature. In such cases:

- Accept the warnings as informational
- Focus on other optimizations (burst access, pipelining)
- Ensure partition pragmas are still provided as hints
- Consider algorithmic restructuring only if critical for performance

**When in doubt**: Test with the affine pipeline and analyze warnings to guide optimization.
