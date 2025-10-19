# Scratchpad Memory Pool Documentation

This directory contains documentation for the APS scratchpad memory pool implementation using CMT2 (Circuit Module Type 2) dialect.

## Documents

### 📘 [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)
**Current implementation status and known issues**

- Module hierarchy and architecture
- Implemented features (cyclic partitioning, data widths, burst access)
- Known bugs (CMT2 Instance.cpp issue)
- File locations and testing instructions

**Read this first** to understand the current state of the implementation.

### 📐 [BANK_WRAPPER_ADDRESS_CALCULATION.md](BANK_WRAPPER_ADDRESS_CALCULATION.md)
**Core address calculation algorithms**

- Cyclic partitioning formulas
- Bank selection logic
- Bit position calculation
- Examples for u8, u16, u32, u64
- Constraint validation

**Read this** when implementing or debugging address decoding logic.

### 🔌 [BURST_BUS_INTERFACE.md](BURST_BUS_INTERFACE.md)
**Burst bus interface specification**

- 64-bit burst bus characteristics
- Data width constraints (`num_banks × data_width >= 64`)
- Data layout for different configurations
- Cyclic vs block partitioning
- Address translation formulas

**Read this** to understand the bus interface and data layout requirements.

## Quick Reference

### Core Formula (Cyclic Partitioning)

```
burst_idx = addr / 8
burst_pattern = burst_idx % num_banks
position = (my_bank - burst_pattern + num_banks) % num_banks
isMine = position < elements_per_burst
offset = (burst_idx - position) / num_banks
```

### Constraint

```
num_banks × data_width >= 64
```

This prevents bank write conflicts by ensuring at most one element per bank per burst.

### Valid Configurations

| Banks | Data Width | Status |
|-------|------------|--------|
| 8     | 8 bits     | ✅     |
| 4     | 16 bits    | ✅     |
| 4     | 32 bits    | ✅     |
| 2     | 32 bits    | ✅     |
| 1     | 64 bits    | ✅     |

### Invalid Configurations

| Banks | Data Width | Issue |
|-------|------------|-------|
| 4     | 8 bits     | ❌ Multiple elements per bank |
| 2     | 16 bits    | ❌ Multiple elements per bank |

## Implementation Files

- **Pass**: `lib/APS/APSMemoryPoolGenPass.cpp`
- **Tests**: `test/APS/mempool.mlir`
- **Dependencies**: ModuleLibrary (Wire, Mem1r1w modules)

## Known Issues

### 🐛 CMT2 Bug: callValue on Regular Modules

**Status**: Blocking - requires ECMT2 maintainer fix

**Location**: `circt/lib/Dialect/Cmt2/ECMT2/Instance.cpp:69-167`

**Problem**: `CallBuilder::buildCall()` doesn't infer result types for regular CMT2 Module instances, only for ExtModuleFirrtlOp.

**Impact**: Calling `callValue()` on WireDefault instances returns empty vector, causing assertion crash.

**Workaround**: None - requires core ECMT2 fix.

## See Also

- [CMT2 Dialect Documentation](../../circt/docs/Dialects/Cmt2/)
- [ECMT2 Class API](../../circt/docs/Dialects/Cmt2/ecmt2-Class-API.md)
- [Module Library](../../circt/docs/Dialects/Cmt2/ModuleLibrary.md)
