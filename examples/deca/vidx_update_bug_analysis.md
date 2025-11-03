# DECA Decompression: vidx Update Bug Analysis

## The Question

In line 106 of `deca_decompress.cadl`:
```cadl
vidx = vidx_3;
```

Should this be:
```cadl
vidx = vidx_3 + (if is_nonzero_3 {1} else {0});
```

Let's trace through what happens with bitmask `0x0000ffff` under both scenarios.

---

## Scenario A: Current Code (vidx = vidx_3)

### Understanding vidx_3 Calculation

```cadl
let vidx_0: u32 = vidx;
let vidx_1: u32 = vidx + (if is_nonzero_0 {1} else {0});
let vidx_2: u32 = vidx_1 + (if is_nonzero_1 {1} else {0});
let vidx_3: u32 = vidx_2 + (if is_nonzero_2 {1} else {0});
```

**Key insight:** `vidx_3` represents the index **AFTER** consuming element 2, but **BEFORE** consuming element 3!

### Iteration 0: idx = 0, all bits set

```
Initial: vidx = 0

vidx_0 = 0     → reads values[0] byte 0 → val_0 = v0
vidx_1 = 0 + 1 = 1 → reads values[0] byte 1 → val_1 = v1
vidx_2 = 1 + 1 = 2 → reads values[0] byte 2 → val_2 = v2
vidx_3 = 2 + 1 = 3 → NOT USED for reading in this iteration!

Writes:
dense_values[0] = v0  ✓
dense_values[1] = v1  ✓
dense_values[2] = v2  ✓
dense_values[3] = if is_nonzero_3 { val_3 } else { 0 }
```

**Wait!** Let's check what `val_3` actually reads:

```cadl
let value_idx_3: u32 = vidx_3 >> 3;  // = 3 >> 3 = 0
let value_offset_3: u32 = vidx_3 & 7; // = 3 & 7 = 3
let val_double_word_3: i64 = values[value_idx_3];  // = values[0]
let val_3: i8 = val_double_word_3 >> (value_offset_3 * 8) [7:0];  // = values[0] byte 3
```

So `val_3` uses `vidx_3` to compute its index! This means:
```
dense_values[3] = v3  ✓
```

**Update vidx:**
```
vidx = vidx_3 = 3
```

### Iteration 1: idx = 1, all bits set

```
Initial: vidx = 3

vidx_0 = 3     → reads values[0] byte 3 → val_0 = v3  ❌ WRONG!
vidx_1 = 3 + 1 = 4 → reads values[0] byte 4 → val_1 = v4  ✓
vidx_2 = 4 + 1 = 5 → reads values[0] byte 5 → val_2 = v5  ✓
vidx_3 = 5 + 1 = 6 → reads values[0] byte 6 → val_3 = v6  ✓

Writes:
dense_values[4] = v3  ❌ DUPLICATE!
dense_values[5] = v4  ✓
dense_values[6] = v5  ✓
dense_values[7] = v6  ✓

vidx = 6
```

**BUG DETECTED!** Element `v3` is written twice:
- `dense_values[3] = v3` (iteration 0)
- `dense_values[4] = v3` (iteration 1)

Element `v7` is never read!

### Full Result with Current Code

Continuing this pattern:

```
Iteration 0: vidx 0→3,  reads v0,v1,v2,v3,  writes to dense[0:3]
Iteration 1: vidx 3→6,  reads v3,v4,v5,v6,  writes to dense[4:7]   ❌ v3 duplicate
Iteration 2: vidx 6→9,  reads v6,v7,v8,v9,  writes to dense[8:11]  ❌ v6 duplicate
Iteration 3: vidx 9→12, reads v9,v10,v11,v12, writes to dense[12:15] ❌ v9 duplicate
```

**Final dense array (WRONG):**
```
dense_values[0]  = v0
dense_values[1]  = v1
dense_values[2]  = v2
dense_values[3]  = v3
dense_values[4]  = v3  ❌ should be v4
dense_values[5]  = v4  ❌ should be v5
dense_values[6]  = v5  ❌ should be v6
dense_values[7]  = v6  ❌ should be v7
dense_values[8]  = v6  ❌ should be v8
dense_values[9]  = v7  ❌ should be v9
dense_values[10] = v8  ❌ should be v10
dense_values[11] = v9  ❌ should be v11
dense_values[12] = v9  ❌ should be v12
dense_values[13] = v10 ❌ should be v13
dense_values[14] = v11 ❌ should be v14
dense_values[15] = v12 ❌ should be v15
dense_values[16:31] = 0
```

**Every value starting from position 4 is OFF BY ONE!**

---

## Scenario B: Fixed Code (vidx = vidx_3 + increment_3)

```cadl
vidx = vidx_3 + (if is_nonzero_3 {1} else {0});
```

### Iteration 0: idx = 0, all bits set

```
Initial: vidx = 0

vidx_0 = 0     → reads v0
vidx_1 = 0 + 1 = 1 → reads v1
vidx_2 = 1 + 1 = 2 → reads v2
vidx_3 = 2 + 1 = 3 → reads v3

Writes:
dense_values[0] = v0  ✓
dense_values[1] = v1  ✓
dense_values[2] = v2  ✓
dense_values[3] = v3  ✓

Update vidx:
vidx = vidx_3 + (if is_nonzero_3 {1} else {0})
     = 3 + 1 = 4
```

### Iteration 1: idx = 1, all bits set

```
Initial: vidx = 4

vidx_0 = 4     → reads v4  ✓
vidx_1 = 4 + 1 = 5 → reads v5  ✓
vidx_2 = 5 + 1 = 6 → reads v6  ✓
vidx_3 = 6 + 1 = 7 → reads v7  ✓

Writes:
dense_values[4] = v4  ✓
dense_values[5] = v5  ✓
dense_values[6] = v6  ✓
dense_values[7] = v7  ✓

vidx = 7 + 1 = 8
```

### Full Result with Fixed Code

```
Iteration 0: vidx 0→4,  reads v0,v1,v2,v3,   writes to dense[0:3]    ✓
Iteration 1: vidx 4→8,  reads v4,v5,v6,v7,   writes to dense[4:7]    ✓
Iteration 2: vidx 8→12, reads v8,v9,v10,v11, writes to dense[8:11]   ✓
Iteration 3: vidx 12→16, reads v12,v13,v14,v15, writes to dense[12:15] ✓
```

**Final dense array (CORRECT):**
```
dense_values[0:15]  = v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15  ✓
dense_values[16:31] = 0, 0, 0, ... (all zeros)  ✓
```

---

## Visual Comparison

### Bitmask: `0x0000ffff` (16 ones, 16 zeros)

| Dense Index | Expected | Current Code (BUGGY) | Fixed Code |
|-------------|----------|----------------------|------------|
| 0-3         | v0-v3    | v0-v3 ✓              | v0-v3 ✓    |
| 4           | v4       | **v3** ❌            | v4 ✓       |
| 5           | v5       | **v4** ❌            | v5 ✓       |
| 6           | v6       | **v5** ❌            | v6 ✓       |
| 7           | v7       | **v6** ❌            | v7 ✓       |
| 8           | v8       | **v6** ❌            | v8 ✓       |
| 9           | v9       | **v7** ❌            | v9 ✓       |
| 10          | v10      | **v8** ❌            | v10 ✓      |
| 11          | v11      | **v9** ❌            | v11 ✓      |
| 12          | v12      | **v9** ❌            | v12 ✓      |
| 13          | v13      | **v10** ❌           | v13 ✓      |
| 14          | v14      | **v11** ❌           | v14 ✓      |
| 15          | v15      | **v12** ❌           | v15 ✓      |
| 16-31       | 0        | 0 ✓                  | 0 ✓        |

---

## Root Cause Analysis

The bug occurs because:

1. **`vidx_3` is calculated but not used** for the actual read of element 3 in the **current** iteration
2. Element 3 reads using `vidx_3`, which **includes** the increment for element 2
3. But `vidx = vidx_3` **does not include** the increment for element 3
4. This causes a **one-element lag** starting from the second iteration

### The Pattern

Each iteration processes 4 elements indexed 0-3:
- Element 0 uses `vidx_0` (current vidx)
- Element 1 uses `vidx_1` (vidx_0 + increment_0)
- Element 2 uses `vidx_2` (vidx_1 + increment_1)
- Element 3 uses `vidx_3` (vidx_2 + increment_2)

At the end, we need `vidx_next` for the next iteration:
- `vidx_next = vidx_3 + increment_3`

**Current code misses `increment_3`!**

---

## Conclusion

### Current Code Behavior (Line 106: `vidx = vidx_3`)
- **INCORRECT** for consecutive non-zero elements
- Causes **off-by-one duplication** starting from the 4th element
- Last sparse value in each 4-element group is duplicated in the next group
- Values v13, v14, v15 are **never read**

### Fixed Code Behavior (Line 106: `vidx = vidx_3 + (if is_nonzero_3 {1} else {0})`)
- **CORRECT** for all patterns
- Each sparse value maps to exactly one dense position
- All 16 sparse values are consumed exactly once

### Impact on Decompression Quality

For bitmask `0x0000ffff` with 16 distinct values:
- **Current code:** Produces corrupted output with repeated values
- **Fixed code:** Produces correct decompressed data

**Recommendation:** Fix line 106 to include the final increment.

---

## Exception Case: What if the Last Element is Zero?

### Example: Iteration with pattern `0b1110` (3 ones, 1 zero)

**Current code:**
```
vidx_3 = vidx_2 + 0 (no increment)
vidx = vidx_3  (correct, since element 3 contributes nothing)
```

**Fixed code:**
```
vidx = vidx_3 + (if is_nonzero_3 {1} else {0})
     = vidx_3 + 0 = vidx_3  (same result)
```

**Both codes produce the same result when element 3 is zero!**

This explains why the bug might not have been caught in testing if test cases had:
- Sparse patterns with zeros in positions 3, 7, 11, 15, etc.
- Or tests that don't verify the exact values, only counts

---

## Test Case for Verification

```cadl
// Input:
bitmask = 0x0000ffff
values = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
          0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10}

// Expected output (correct):
dense_values[0:15] = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
                      0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10}

// Actual output (with current buggy code):
dense_values[0:15] = {0x01, 0x02, 0x03, 0x04, 0x04, 0x05, 0x06, 0x07,
                      0x07, 0x08, 0x09, 0x0a, 0x0a, 0x0b, 0x0c, 0x0d}
//                                          ^^                ^^      duplicates!
```

Notice the pattern: Every 4th element gets duplicated in the next group.
