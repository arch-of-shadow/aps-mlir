# Bank Wrapper Address Calculation

## Overview

This document explains how each bank wrapper calculates whether it participates in a 64-bit burst access and where its data should be positioned in the 64-bit word.

## Key Concepts

- **Cyclic Partitioning**: Elements are distributed round-robin across banks
- **Burst Access**: Each 64-bit memory transaction accesses multiple consecutive elements
- **Element Index**: `element_idx = byte_address / element_size`
- **Bank Selection**: `bank_idx = element_idx % num_banks`

## Algorithm

For each bank wrapper receiving a burst address:

1. **Calculate starting element and bank**:
   ```
   element_idx = addr / element_size
   start_bank_idx = element_idx % num_banks
   ```

2. **Calculate this bank's position in the cyclic sequence**:
   ```
   position = (my_bank_idx - start_bank_idx + num_banks) % num_banks
   ```

3. **Check if this bank participates**:
   ```
   elements_per_burst = 64 / data_width
   isMine = (position < elements_per_burst)
   ```

4. **Calculate bit position in 64-bit word**:
   ```
   element_offset_in_burst = position % elements_per_burst
   bit_start = element_offset_in_burst * data_width
   ```

## Examples

### u32 (32-bit, 4 bytes per element)

**Configuration:**
- Element size: 4 bytes
- Elements per burst: 64 / 32 = 2
- Num banks: 4

**mem_a_0:** (1) 0 <= addr < 256 (2) (addr / 8) % 2 == 0
    => Offset = ((addr - base) / 8) / 2
    => Place it in lower 32bit

**mem_a_1:** (1) 0 <= addr < 256 (2) (addr / 8) % 2 == 0
    => Offset = ((addr - base) / 8) / 2
    => Place it in higher 32bit

**mem_a_2:** (1) 0 <= addr < 256 (2) (addr / 8) % 2 == 1
    => Offset = ((addr - base) / 8 - 1) / 2
    => Place it in lower 32bit

**mem_a_3:** (1) 0 <= addr < 256 (2) (addr / 8) % 2 == 1
    => Offset = ((addr - base) / 8 - 1) / 2
    => Place it in higher 32bit

**Verification:**
- Address 0: burst_idx=0, burst%2=0 → banks 0,1 ✓
- Address 8: burst_idx=1, burst%2=1 → banks 2,3 ✓
- Address 12: burst_idx=1, burst%2=1 → banks 2,3... Wait, addr 12 / 8 = 1, so burst 1.
  But element 3 is at addr 12, which is bank 3. And element 4 is at addr 16 (burst 2).

Actually for addr 12:
- element_idx = 12/4 = 3, bank = 3%4 = 3
- But we're reading a burst starting at 12, which spans elements 3,4
- Element 3 (addr 12): bank 3
- Element 4 (addr 16): bank 0, but this is in the NEXT burst (burst 2)!

So burst starting at addr 12 actually only contains element 3 (from bank 3).
Let me recalculate:

Burst alignment: 64-bit = 8 bytes, so bursts start at addresses 0, 8, 16, 24...
- Address 12 is NOT burst-aligned!
- If we're reading at addr 12, we're reading within burst 1 (addresses 8-15)

For u32 with 2 elements per burst:
- Burst at addr 0: elements 0,1 (banks 0,1)
- Burst at addr 8: elements 2,3 (banks 2,3)
- Burst at addr 16: elements 4,5 (banks 0,1)

So the specification assumes burst-aligned accesses!

### u16 (16-bit, 2 bytes per element)

**Configuration:**
- Element size: 2 bytes
- Elements per burst: 64 / 16 = 4
- Num banks: 4
- Burst alignment: 8 bytes (addresses 0, 8, 16, 24...)

**mem_a_0:** (1) 0 <= addr < size (2) (addr / 8) % 4 == 0
    => Offset = ((addr - base) / 8) / 4
    => Place it in bits [15:0]

**mem_a_1:** (1) 0 <= addr < size (2) (addr / 8) % 4 == 0
    => Offset = ((addr - base) / 8) / 4
    => Place it in bits [31:16]

**mem_a_2:** (1) 0 <= addr < size (2) (addr / 8) % 4 == 0
    => Offset = ((addr - base) / 8) / 4
    => Place it in bits [47:32]

**mem_a_3:** (1) 0 <= addr < size (2) (addr / 8) % 4 == 0
    => Offset = ((addr - base) / 8) / 4
    => Place it in bits [63:48]

**Explanation:**
For u16, all 4 banks participate in every burst because elements_per_burst (4) == num_banks (4).
The condition `(addr / 8) % 4 == 0` means all banks are used for every burst that is 4-burst aligned.
However, for intermediate bursts:

- Burst 0 (addr 0): elements 0,1,2,3 → banks 0,1,2,3
- Burst 1 (addr 8): elements 4,5,6,7 → banks 0,1,2,3
- Burst 2 (addr 16): elements 8,9,10,11 → banks 0,1,2,3

### u8 (8-bit, 1 byte per element)

**Configuration:**
- Element size: 1 byte
- Elements per burst: 64 / 8 = 8
- Num banks: 4
- **INVALID**: This violates the constraint `num_banks × data_width ≥ 64`
  - 4 banks × 8 bits = 32 < 64 ✗

**Why this doesn't work:**
With 8 elements per burst but only 4 banks, each bank would need to contribute TWO elements per burst. The current bank wrapper design assumes each bank contributes at most once per burst.

**To support u8, you need:**
- **Option 1**: Use 8 banks (8 × 8 = 64 ≥ 64) ✓
- **Option 2**: Redesign bank wrappers to handle multiple contributions

**If using 8 banks:**

**mem_a_0:** (1) 0 <= addr < size (2) (addr / 8) % 8 == 0
    => Offset = ((addr - base) / 8) / 8
    => Place it in bits [7:0]

**mem_a_1:** (1) 0 <= addr < size (2) (addr / 8) % 8 == 0
    => Offset = ((addr - base) / 8) / 8
    => Place it in bits [15:8]

**mem_a_2:** (1) 0 <= addr < size (2) (addr / 8) % 8 == 0
    => Offset = ((addr - base) / 8) / 8
    => Place it in bits [23:16]

**mem_a_3:** (1) 0 <= addr < size (2) (addr / 8) % 8 == 0
    => Offset = ((addr - base) / 8) / 8
    => Place it in bits [31:24]

**mem_a_4:** (1) 0 <= addr < size (2) (addr / 8) % 8 == 0
    => Offset = ((addr - base) / 8) / 8
    => Place it in bits [39:32]

**mem_a_5:** (1) 0 <= addr < size (2) (addr / 8) % 8 == 0
    => Offset = ((addr - base) / 8) / 8
    => Place it in bits [47:40]

**mem_a_6:** (1) 0 <= addr < size (2) (addr / 8) % 8 == 0
    => Offset = ((addr - base) / 8) / 8
    => Place it in bits [55:48]

**mem_a_7:** (1) 0 <= addr < size (2) (addr / 8) % 8 == 0
    => Offset = ((addr - base) / 8) / 8
    => Place it in bits [63:56]

### u64 (64-bit, 8 bytes per element)

**Configuration:**
- Element size: 8 bytes
- Elements per burst: 64 / 64 = 1
- Num banks: 4
- Burst alignment: 8 bytes (addresses 0, 8, 16, 24...)

**mem_b_0:** (1) 0 <= addr < size (2) (addr / 8) % 4 == 0
    => Offset = ((addr - base) / 8) / 4
    => Place it in bits [63:0]

**mem_b_1:** (1) 0 <= addr < size (2) (addr / 8) % 4 == 1
    => Offset = ((addr - base) / 8 - 1) / 4
    => Place it in bits [63:0]

**mem_b_2:** (1) 0 <= addr < size (2) (addr / 8) % 4 == 2
    => Offset = ((addr - base) / 8 - 2) / 4
    => Place it in bits [63:0]

**mem_b_3:** (1) 0 <= addr < size (2) (addr / 8) % 4 == 3
    => Offset = ((addr - base) / 8 - 3) / 4
    => Place it in bits [63:0]

**Explanation:**
For u64, only ONE bank participates per burst because elements_per_burst (1) < num_banks (4).
Each burst accesses exactly one element, which belongs to one bank based on cyclic distribution:

- Burst 0 (addr 0): element 0 → bank 0
- Burst 1 (addr 8): element 1 → bank 1
- Burst 2 (addr 16): element 2 → bank 2
- Burst 3 (addr 24): element 3 → bank 3
- Burst 4 (addr 32): element 4 → bank 0

## Constraint

The design assumes: **`num_banks × data_width ≥ 64`**

This ensures that `elements_per_burst ≤ num_banks`, meaning each bank contributes at most once per burst.

**Valid configurations:**
- 4 banks × 16 bits = 64 ✓
- 4 banks × 32 bits = 128 ✓
- 4 banks × 64 bits = 256 ✓
- 8 banks × 8 bits = 64 ✓

**Invalid configuration (NOT SUPPORTED):**
- 4 banks × 8 bits = 32 < 64 ✗ (each bank would need to contribute twice per burst)

## Local Address Calculation

Once a bank determines it owns an element:
```
my_element_idx = start_element_idx + position
local_addr = my_element_idx / num_banks
```

For bank 0 at address 12 (u32 example):
- `start_element_idx = 12 / 4 = 3`
- `position = 1`
- `my_element_idx = 3 + 1 = 4`
- `local_addr = 4 / 4 = 1` ✓
