# Scratchpad Memory Pool Implementation Status

## Overview

The APSMemoryPoolGenPass generates a hierarchical CMT2-based scratchpad memory pool with burst access support for cyclic-partitioned memory banks.

## Current Architecture

### Module Hierarchy

```
ScratchpadMemoryPool (top-level)
├─ burst_read(addr: u64) -> u64
├─ burst_write(addr: u64, data: u64)
└─ For each memory entry:
   └─ BankWrapper_<entry>_<idx> (one per bank)
      ├─ mem_bank (Mem1r1w instance)
      ├─ write_enable_wire (WireDefault instance)
      ├─ write_data_wire (WireDefault instance)
      ├─ write_addr_wire (WireDefault instance)
      ├─ burst_read(addr: u64) -> u64
      ├─ burst_write(addr: u64, data: u64)
      └─ do_bank_write (rule with wire guards)
```

### Key Design Decisions

1. **Bank Wrapper Modules**: Each physical memory bank is wrapped in a module that:
   - Determines if it participates in a burst access
   - Calculates its local address
   - Aligns data to correct bit position in 64-bit burst word
   - Uses WireDefault pattern for conditional writes

2. **WireDefault Pattern**:
   - Wraps FIRRTL Wire modules with default value (0)
   - Write method overrides default
   - Read value returns current wire value
   - Used for enable/data/addr signals in conditional bank writes

3. **Conditional Write via Wires**:
   - `burst_write` method writes enable/data/addr to wires
   - Separate `do_bank_write` rule with guard checking wire enable
   - Rule body reads from wires and calls bank write

## Implemented Features

### ✅ Cyclic Partitioning Address Calculation

**Algorithm** (see `BANK_WRAPPER_ADDRESS_CALCULATION.md` for details):
```
burst_idx = addr / 8
burst_pattern = burst_idx % num_banks
position = (my_bank - burst_pattern + num_banks) % num_banks
isMine = position < elements_per_burst
offset = (burst_idx - position) / num_banks
```

### ✅ Data Width Support

- Supports u8, u16, u32, u64 with constraint: `num_banks × data_width >= 64`
- Automatic bit positioning based on bank index and data width
- Zero padding for alignment

### ✅ Burst Read

- Each bank wrapper returns aligned data or 0
- Top-level ORs all wrapper outputs
- Handles multi-element bursts (e.g., 2 × u32 per 64-bit word)

### ✅ Burst Write

- Broadcasts addr/data to all bank wrappers
- Each wrapper:
  - Calculates if it participates
  - Extracts its data slice
  - Writes enable/data/addr to internal wires
- Conditional write rule fires when enable wire is high

## Known Issues

### 🐛 Critical Bug: CMT2 Instance.cpp

**Location**: `circt/lib/Dialect/Cmt2/ECMT2/Instance.cpp:69-167`

**Problem**: `CallBuilder::buildCall()` only handles result type inference for `ExtModuleFirrtlOp`. For regular CMT2 `ModuleOp` instances (like WireDefault), it leaves `resultTypes` empty, causing `CallOp` to be created with zero results.

**Impact**: Calling `callValue()` on WireDefault instances returns empty vector, causing assertion failure when accessing `[0]`.

**Workaround**: None currently - requires fix in ECMT2 core.

**Code Location**: `lib/APS/APSMemoryPoolGenPass.cpp:509-510`
```cpp
// BUG: writeEnableWire->callValue returns empty vector for regular Module instances
auto enableValues = writeEnableWire->callValue("read", guardBuilder);
auto isEnabled = guardBuilder.create<EQPrimOp>(loc, enableValues[0], oneConst.getResult());
```

**Required Fix**: Add logic in `Instance.cpp` to look up Method/Value definitions in regular CMT2 Modules and extract their return types, similar to how it handles ExtModuleFirrtlOp.

## File Locations

- **Implementation**: `lib/APS/APSMemoryPoolGenPass.cpp`
- **Core Algorithm**: `docs/SCRATCHPAD_MEM/BANK_WRAPPER_ADDRESS_CALCULATION.md`
- **This Document**: `docs/SCRATCHPAD_MEM/IMPLEMENTATION_STATUS.md`

## Testing

```bash
# Build
pixi run build

# Run pass (currently crashes due to CMT2 bug)
./build/bin/aps-opt --aps-memory-pool-gen test.mlir
```

## Next Steps

1. **Fix CMT2 bug** - Requires ECMT2 maintainer to fix `Instance.cpp`
2. **Test with fixed CMT2** - Verify wire-based conditional writes work correctly
3. **Add block partitioning** - Currently only cyclic is implemented
4. **Optimize generated code** - Consider inlining or flattening

## Statistics

- **Lines of implementation**: ~550 lines in APSMemoryPoolGenPass.cpp
- **Modules generated**: 1 top-level + N bank wrappers (N = total banks across all entries)
- **Dependencies**: ModuleLibrary (for Wire and Mem1r1w modules)
- **Constraint validation**: Checks `num_banks × data_width >= 64` at generation time
