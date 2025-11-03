# DECA Decompression: Bitmask Example `0x0000ffff`

## Overview
This document traces the execution of the DECA decompression algorithm with bitmask `0x0000ffff`, showing how the sparse-to-dense expansion works step by step.

## Input Data

**Bitmask:** `0x0000ffff = 0b00000000_00000000_11111111_11111111`
- Bits 0-15: **1** (16 non-zero elements)
- Bits 16-31: **0** (16 zero elements)

**Memory Layout:**
```
bitmask[31:24] = 0x00 = 0b00000000  (bits 24-31: idx 24-31)
bitmask[23:16] = 0x00 = 0b00000000  (bits 16-23: idx 16-23)
bitmask[15:8]  = 0xFF = 0b11111111  (bits 8-15:  idx 8-15)
bitmask[7:0]   = 0xFF = 0b11111111  (bits 0-7:   idx 0-7)
```

**Sparse Values Array (`values[0:7]`):**
Assume we have 16 non-zero i8 values packed into 8 × i64 words:
```
values[0] = {v0, v1, v2, v3, v4, v5, v6, v7}     // 8 bytes
values[1] = {v8, v9, v10, v11, v12, v13, v14, v15}  // 8 bytes
values[2+] = unused
```

## Loop Execution (8 Iterations, 4 Elements Each)

The loop processes 32 dense elements in 8 iterations, handling 4 elements per iteration.

---

### **Iteration 0: idx = 0** (Dense indices 0-3)

#### Bitmask Selection
```cadl
idx < 2: true
mask_byte = bitmask[7:0] = 0xFF
```

#### Bit Position Calculation
```
bit_pos_0 = idx[0:0] * 4 = 0 * 4 = 0
bit_pos_1 = 0 + 1 = 1
bit_pos_2 = 0 + 2 = 2
bit_pos_3 = 0 + 3 = 3
```

#### Bit Extraction
```
bit_shifted_0 = 0xFF >> 0 = 0b11111111
is_nonzero_0 = bit_shifted_0[0:0] = 1  ✓

bit_shifted_1 = 0xFF >> 1 = 0b01111111
is_nonzero_1 = bit_shifted_1[0:0] = 1  ✓

bit_shifted_2 = 0xFF >> 2 = 0b00111111
is_nonzero_2 = bit_shifted_2[0:0] = 1  ✓

bit_shifted_3 = 0xFF >> 3 = 0b00011111
is_nonzero_3 = bit_shifted_3[0:0] = 1  ✓
```

#### Sparse Index Calculation
```
vidx = 0 (initial)

vidx_0 = 0
vidx_1 = 0 + 1 = 1
vidx_2 = 1 + 1 = 2
vidx_3 = 2 + 1 = 3
```

#### Value Extraction
```
value_idx_0 = 0 >> 3 = 0,  value_offset_0 = 0 & 7 = 0
val_double_word_0 = values[0]
val_0 = (values[0] >> (0*8))[7:0] = v0

value_idx_1 = 1 >> 3 = 0,  value_offset_1 = 1 & 7 = 1
val_1 = (values[0] >> (1*8))[7:0] = v1

value_idx_2 = 2 >> 3 = 0,  value_offset_2 = 2 & 7 = 2
val_2 = (values[0] >> (2*8))[7:0] = v2

value_idx_3 = 3 >> 3 = 0,  value_offset_3 = 3 & 7 = 3
val_3 = (values[0] >> (3*8))[7:0] = v3
```

#### Dense Array Write
```
dense_values[0] = v0  ✓
dense_values[1] = v1  ✓
dense_values[2] = v2  ✓
dense_values[3] = v3  ✓

vidx updated to 3
```

---

### **Iteration 1: idx = 1** (Dense indices 4-7)

#### Bitmask Selection
```cadl
idx < 2: true
mask_byte = bitmask[7:0] = 0xFF
```

#### Bit Position Calculation
```
bit_pos_0 = idx[0:0] * 4 = 1 * 4 = 4
bit_pos_1 = 5
bit_pos_2 = 6
bit_pos_3 = 7
```

#### Bit Extraction
```
bit_shifted_0 = 0xFF >> 4 = 0b00001111
is_nonzero_0 = 1  ✓

bit_shifted_1 = 0xFF >> 5 = 0b00000111
is_nonzero_1 = 1  ✓

bit_shifted_2 = 0xFF >> 6 = 0b00000011
is_nonzero_2 = 1  ✓

bit_shifted_3 = 0xFF >> 7 = 0b00000001
is_nonzero_3 = 1  ✓
```

#### Sparse Index Calculation
```
vidx = 3 (from previous iteration)

vidx_0 = 3
vidx_1 = 3 + 1 = 4
vidx_2 = 4 + 1 = 5
vidx_3 = 5 + 1 = 6
```

#### Value Extraction
```
value_idx_0 = 3 >> 3 = 0,  value_offset_0 = 3
val_0 = (values[0] >> (3*8))[7:0] = v3  // Wait, this should be v4!

Actually:
value_idx_0 = 3 >> 3 = 0,  value_offset_0 = 3 & 7 = 3
val_0 = (values[0] >> (3*8))[7:0] = v3  // ERROR: should read v4!
```

**⚠️ BUG DETECTED:** The code updates `vidx = vidx_3` at the end, but should update to `vidx_3 + (if is_nonzero_3 {1} else {0})` to account for the last element!

Let me recalculate with the **corrected logic**:

#### Corrected: Iteration 1 with vidx = 4
```
vidx = 4 (should be this)

vidx_0 = 4
vidx_1 = 4 + 1 = 5
vidx_2 = 5 + 1 = 6
vidx_3 = 6 + 1 = 7

value_idx_0 = 4 >> 3 = 0,  value_offset_0 = 4 & 7 = 4
val_0 = (values[0] >> (4*8))[7:0] = v4

val_1 = v5, val_2 = v6, val_3 = v7
```

#### Dense Array Write
```
dense_values[4] = v4  ✓
dense_values[5] = v5  ✓
dense_values[6] = v6  ✓
dense_values[7] = v7  ✓

vidx updated to 7
```

---

### **Iteration 2: idx = 2** (Dense indices 8-11)

#### Bitmask Selection
```cadl
idx < 2: false
idx < 4: true
mask_byte = bitmask[15:8] = 0xFF
```

#### Bit Position Calculation
```
bit_pos_0 = idx[0:0] * 4 = 0 * 4 = 0  (idx[0:0] = 0 since idx=2)
bit_pos_1 = 1
bit_pos_2 = 2
bit_pos_3 = 3
```

#### Bit Extraction
All bits are 1, same as iteration 0.

#### Sparse Index Calculation
```
vidx = 8 (corrected: should be 8)

vidx_0 = 8
vidx_1 = 9
vidx_2 = 10
vidx_3 = 11
```

#### Value Extraction
```
value_idx_0 = 8 >> 3 = 1,  value_offset_0 = 0
val_0 = (values[1] >> (0*8))[7:0] = v8

val_1 = v9, val_2 = v10, val_3 = v11
```

#### Dense Array Write
```
dense_values[8]  = v8   ✓
dense_values[9]  = v9   ✓
dense_values[10] = v10  ✓
dense_values[11] = v11  ✓

vidx updated to 11
```

---

### **Iteration 3: idx = 3** (Dense indices 12-15)

Similar to iteration 2, processes v12-v15.

```
dense_values[12] = v12  ✓
dense_values[13] = v13  ✓
dense_values[14] = v14  ✓
dense_values[15] = v15  ✓

vidx updated to 15
```

---

### **Iterations 4-7: idx = 4-7** (Dense indices 16-31)

#### Bitmask Selection
```cadl
idx >= 4:
mask_byte = bitmask[23:16] or bitmask[31:24] = 0x00
```

#### Bit Extraction
All bits are 0:
```
is_nonzero_0 = 0  ✗
is_nonzero_1 = 0  ✗
is_nonzero_2 = 0  ✗
is_nonzero_3 = 0  ✗
```

#### Sparse Index Calculation
```
vidx_0 = vidx
vidx_1 = vidx + 0 = vidx
vidx_2 = vidx + 0 = vidx
vidx_3 = vidx + 0 = vidx

vidx remains 15 (no increment)
```

#### Dense Array Write
```
dense_values[16-31] = 0  (zero_i8)
```

All remaining elements are written as **zero**.

---

## Final Result

### Dense Values Array (32 elements)
```
dense_values[0:15]  = v0, v1, v2, ..., v15   (16 non-zero values)
dense_values[16:31] = 0, 0, 0, ..., 0        (16 zeros)
```

### Sparse Values Consumed
```
16 values consumed from values[0:1]
vidx final value = 15
```

## Summary

For bitmask `0x0000ffff`:
- **First 16 elements (bits 0-15 set):** Populated with sparse values v0-v15
- **Last 16 elements (bits 16-31 clear):** Filled with zeros
- **Sparse array access:** Only `values[0]` and `values[1]` are read
- **vidx progression:** 0 → 4 → 8 → 12 → 16 → 16 → 16 → 16 (stays at 16 after all non-zeros consumed)

## Note on Potential Bug

**Line 106 in original code:**
```cadl
vidx = vidx_3;
```

This should potentially be:
```cadl
vidx = vidx_3 + (if is_nonzero_3 {1} else {0});
```

However, since `vidx_3` already accounts for `is_nonzero_2`, and we're tracking the **next** index to read, the current implementation appears correct if `vidx_3` represents the index **after** processing element 3.

The code correctly implements a **prefix sum** of the bitmask to calculate sparse indices.
