# Burst Bus Interface Specification

## Overview

The scratchpad memory pool uses a 64-bit burst bus for high-bandwidth memory access. This document specifies the burst bus interface and data layout constraints.

## Burst Bus Characteristics

- **Width**: 64 bits (8 bytes)
- **Address**: Byte address (globally addressed across all memory entries)
- **Alignment**: Naturally aligned to 8-byte boundaries
- **Operations**: `burst_read(addr: u64) -> u64`, `burst_write(addr: u64, data: u64)`

## Memory Entry Structure

Each `aps.mem_entry` defines a region in the global address space:

```mlir
aps.mem_entry "mem_a" : banks([@mem_a_0, @mem_a_1, @mem_a_2, @mem_a_3]),
                       base(0), size(256), count(4), cyclic(1)
```

**Attributes:**
- **base**: Starting byte address in global address space
- **size**: Total size of this entry in bytes (NOT per-bank)
- **count**: Number of banks
- **cyclic**: Partition mode (1=cyclic, 0=block)
- **banks**: Array of memref.global symbols for each bank

## Data Width Handling

Memory banks support data widths that are **multiples of 8 bits**: u8, u16, u24, u32, u40, u48, u56, u64.

### Constraint: Bank Conflict Prevention

**Required**: `num_banks × data_width >= 64`

This ensures that each 64-bit burst accesses at most **one element per bank**, preventing write conflicts.

### Valid Configurations

| Banks | Data Width | Product | Elements/Burst | Status |
|-------|------------|---------|----------------|--------|
| 8     | 8 bits     | 64      | 8              | ✅ All banks used |
| 4     | 16 bits    | 64      | 4              | ✅ All banks used |
| 2     | 32 bits    | 64      | 2              | ✅ All banks used |
| 1     | 64 bits    | 64      | 1              | ✅ One bank used |
| 4     | 32 bits    | 128     | 2              | ✅ 2 of 4 banks per burst |
| 8     | 16 bits    | 128     | 4              | ✅ 4 of 8 banks per burst |

### Invalid Configurations

| Banks | Data Width | Product | Elements/Burst | Issue |
|-------|------------|---------|----------------|-------|
| 2     | 16 bits    | 32      | 4              | ❌ 2 elements/bank = conflict |
| 4     | 8 bits     | 32      | 8              | ❌ 2 elements/bank = conflict |

**Why invalid**: With `elements_per_burst > num_banks`, multiple elements in a single burst map to the same bank, causing write conflicts.

## Cyclic Partitioning Data Layout

### Bit Position Calculation

For cyclic partitioning, each bank occupies fixed bit positions based on its index:

```
elements_per_burst = 64 / data_width
position = bank_idx % elements_per_burst
bit_start = position * data_width
bit_end = bit_start + data_width - 1
```

### Examples

#### 4 Banks × u32 (32-bit elements)

**Configuration:**
- Elements per burst: 64 / 32 = 2
- Product: 4 × 32 = 128 bits (> 64, so 2 of 4 banks per burst)

**Burst 0 (addr 0):**
- Element 0 (bank 0) → bits [31:0]
- Element 1 (bank 1) → bits [63:32]

**Burst 1 (addr 8):**
- Element 2 (bank 2) → bits [31:0]
- Element 3 (bank 3) → bits [63:32]

#### 4 Banks × u16 (16-bit elements)

**Configuration:**
- Elements per burst: 64 / 16 = 4
- Product: 4 × 16 = 64 bits (all banks used per burst)

**Every burst:**
- Bank 0 → bits [15:0]
- Bank 1 → bits [31:16]
- Bank 2 → bits [47:32]
- Bank 3 → bits [63:48]

#### 4 Banks × u64 (64-bit elements)

**Configuration:**
- Elements per burst: 64 / 64 = 1
- Product: 4 × 64 = 256 bits (only 1 bank per burst)

**Burst 0 (addr 0):** Bank 0 → bits [63:0]
**Burst 1 (addr 8):** Bank 1 → bits [63:0]
**Burst 2 (addr 16):** Bank 2 → bits [63:0]
**Burst 3 (addr 24):** Bank 3 → bits [63:0]

## Block Partitioning Data Layout

For block partitioning, data position depends on byte offset within the burst:

```
byte_offset_in_burst = addr % 8
bit_start = byte_offset_in_burst * 8
bit_end = bit_start + (data_width - 1)
```

**Example: u32 with block partitioning:**
- Offset 0 → bits [31:0]
- Offset 4 → bits [63:32]

## Burst Read Operation

1. Calculate which banks participate in this burst
2. Each bank reads its element and positions data at correct bits
3. OR all bank outputs together
4. Non-participating banks return 0

**Pseudo-code:**
```
result = 0
for each bank:
  if bank_participates(addr, bank_idx):
    data = bank.read(local_addr)
    positioned = shift_to_position(data, bank_idx, data_width)
    result |= positioned
return result
```

## Burst Write Operation

1. Calculate which banks participate in this burst
2. Each bank extracts its data slice from 64-bit word
3. Banks write conditionally based on participation

**Pseudo-code:**
```
for each bank:
  if bank_participates(addr, bank_idx):
    bit_slice = extract_bits(data, bank_idx, data_width)
    bank.write(local_addr, bit_slice)
```

## Address Translation

### Global to Bank-Local Address

**For cyclic partitioning:**
```
element_idx = addr / element_size_bytes
bank_idx = element_idx % num_banks
local_addr = element_idx / num_banks
```

**For block partitioning:**
```
offset = addr - base
size_per_bank = total_size / num_banks
bank_idx = offset / size_per_bank
local_addr = (offset % size_per_bank) / element_size_bytes
```

## Implementation Notes

- **OR aggregation**: Non-participating banks return 0, so simple OR combines results
- **Zero padding**: Banks add zero padding to position data correctly
- **Natural alignment**: Burst accesses should be 8-byte aligned for efficiency
- **Concurrent access**: Multiple banks can be read/written in parallel

## See Also

- `BANK_WRAPPER_ADDRESS_CALCULATION.md` - Detailed address calculation algorithms
- `IMPLEMENTATION_STATUS.md` - Current implementation status and known issues
