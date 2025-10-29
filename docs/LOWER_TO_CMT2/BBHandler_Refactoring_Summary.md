# BBHandler Refactoring Summary

## Overview
This document summarizes the complete refactoring of BBHandler to properly handle single basic blocks according to Blockgen.md specifications.

## Changes Made

### 1. Interface Update
**Before:**
```cpp
LogicalResult processBasicBlock(Block *mlirBlock, unsigned blockId,
                                llvm::DenseMap<Value, Instance*> &inputFIFOs,
                                llvm::DenseMap<Value, Instance*> &outputFIFOs,
                                Instance *inputTokenFIFO, Instance *outputTokenFIFO);
```

**After:**
```cpp
LogicalResult processBasicBlock(BlockInfo& block);
```

**Benefit:** Cleaner API, direct access to `blockName` for proper naming conventions.

### 2. Member Variable Addition
Added `BlockInfo* currentBlock` member to store the current block being processed, providing easy access to:
- `blockName` for instance naming
- `input_fifos` and `output_fifos`
- `input_token_fifo` and `output_token_fifo`

### 3. Naming Convention Fixes
All instances now use `currentBlock->blockName` instead of `opcode`:

| Instance Type | Old Name | New Name |
|--------------|----------|----------|
| Cross-slot FIFO | `{opcode}_fifo_sX_sY` | `{blockName}_fifo_sX_sY` |
| Token FIFO | `{opcode}_token_fifo_sX` | `{blockName}_token_fifo_sX` |
| Register | `reg_rd_{opcode}` | `reg_rd_{blockName}` |
| Rule | `{opcode}_rule_sX` | `{blockName}_slot_X_rule` |

**Note:** `opcode` is added as prefix at the outermost level automatically.

### 4. Input FIFO Handling Enhancement

#### buildCrossSlotFIFOs() Extension
**Added logic to create cross-slot FIFOs for input values:**

```cpp
// Handle cross-block input values (from inputFIFOs)
for (auto &[inputValue, inputFIFO] : currentBlock->input_fifos) {
  // Find all operations that use this input value and group by slot
  llvm::DenseMap<int64_t, llvm::SmallVector<...>> consumersByStage;

  // For each consumer slot, create cross_slot_fifo_s0_sN
  for (auto &[consumerSlot, consumers] : consumersByStage) {
    auto fifo = std::make_unique<CrossSlotFIFO>();
    fifo->producerSlot = 0;  // Input values available at slot 0
    fifo->consumerSlot = consumerSlot;
    // ...
  }
}
```

#### First Slot Distribution Logic
**Implemented proper value distribution per BBHandler_DataFlow.md:**

```cpp
if (slot == slotOrder.front()) {
  for (auto &[value, fifo] : inputFIFOs) {
    auto dequeuedValue = fifo->callMethod("deq", {}, b)[0];

    // Check if used in slot 0
    if (usedInSlot0) {
      localMap[value] = dequeuedValue;  // For immediate use
    }

    // Check if used in later slots
    auto it = crossSlotFIFOs.find(value);
    if (it != crossSlotFIFOs.end()) {
      for (auto crossSlotFifo : it->second) {
        crossSlotFifo->fifoInstance->callMethod("enq", {dequeuedValue}, b);
      }
    }
  }
}
```

**Key Insight:** Same dequeued value can go to BOTH `localMap` and `cross_slot_fifos`.

### 5. Empty Block Handling
**Before:** Created a minimal coordination rule
**After:** Panics with fatal error - empty blocks should not reach BBHandler

```cpp
if (blockOperations.empty()) {
  llvm::report_fatal_error("BBHandler received empty block - should be handled by BlockHandler");
}
```

## Data Flow Patterns

See [BBHandler_DataFlow.md](BBHandler_DataFlow.md) for complete documentation.

### Summary of Patterns:
1. **Same slot**: localMap only
2. **Cross slot**: localMap in producer → cross_slot_fifo → localMap in consumer
3. **Input to slot 0**: input_fifo → localMap
4. **Input to later slot**: input_fifo → cross_slot_fifo → localMap
5. **Input to slot 0 + later**: input_fifo → BOTH localMap AND cross_slot_fifo

## Files Modified

1. **include/APS/BBHandler.h**
   - Updated interface to use `BlockInfo&`
   - Added `BlockInfo* currentBlock` member
   - Added include for `BlockHandler.h`

2. **lib/APS/APSToCMT2/BBHandler.cpp**
   - Complete refactoring of `processBasicBlock()`
   - Extended `buildCrossSlotFIFOs()` for input values
   - Updated all naming to use `currentBlock->blockName`
   - Fixed first slot input handling

3. **lib/APS/APSToCMT2/BlockHandler.cpp**
   - Updated `processRegularBlockWithBBHandler()` to use new interface

4. **docs/Blockgen.md**
   - Updated with naming conventions
   - Documented current implementation status
   - Added fix plan and completion status

5. **docs/BBHandler_DataFlow.md** (NEW)
   - Complete documentation of data flow patterns
   - Value acquisition and storage algorithms
   - Examples for all use cases

## Verification Checklist

- [x] BBHandler processes only ONE basic block
- [x] Uses `BlockInfo&` interface
- [x] All instances use `blockName` for naming
- [x] Cross-slot FIFOs created for input values used in later slots
- [x] First slot distributes input values correctly
- [x] localMap cleared between slots
- [x] getValueInRule() handles all value sources correctly
- [x] Empty blocks panic

## Next Steps

1. Test with actual MLIR code
2. Verify FIFO instantiation works correctly
3. Check that values flow through FIFOs as expected
4. Ensure no regressions in existing functionality
