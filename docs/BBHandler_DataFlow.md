# BBHandler Data Flow and FIFO Usage

This document describes how BBHandler acquires and stores values through FIFOs and localMap across time slots.

## Overview

BBHandler processes a basic block by dividing operations into time slots based on their `starttime` attribute. Values flow between slots using FIFOs, and within a slot using `localMap`.

## Value Sources

A value used by an operation can come from three sources:

1. **Same Slot** - Produced by another operation in the current slot
2. **Earlier Slot** - Produced by an operation in a previous slot
3. **Cross-Block Input** - From `input_fifos` (previous block's output)

## Data Structures

### localMap
- **Type**: `llvm::DenseMap<mlir::Value, mlir::Value>`
- **Scope**: Per-slot (cleared at the beginning of each slot rule)
- **Purpose**: Stores values available in the current slot
- **Contents**:
  - Values produced by operations in this slot
  - Values dequeued from FIFOs for use in this slot
  - Values from input_fifos if used in slot 0

### cross_slot_fifos
- **Type**: `llvm::DenseMap<mlir::Value, llvm::SmallVector<CrossSlotFIFO*>>`
- **Scope**: Entire block
- **Purpose**: Transfer values from producer slot to consumer slot(s)
- **Naming**: `{blockName}_fifo_s{producer}_s{consumer}[_{count}]`
- **Created for**:
  - Values produced in slot X and used in slot Y (where Y > X)
  - Input values used in later slots (slot 0 → slot Y)

## Value Flow Patterns

### Pattern 1: Value Produced and Used in Same Slot

```
Slot 3:
  op1: %result = some_op(...)
  op2: another_op(%result)  // Uses %result
```

**Flow:**
1. `op1` executes, produces `%result`
2. Store in localMap: `localMap[%result] = result_value`
3. `op2` executes, reads from localMap: `getValueInRule(%result)` → finds in localMap
4. No FIFO needed

**Rule:**
```cpp
// In slot 3 rule body:
auto result_value = /* generate op1 */;
localMap[%result] = result_value;

// Later in same rule:
auto operand = localMap[%result];  // Direct lookup
```

### Pattern 2: Value Produced in Slot X, Used in Slot Y (X < Y)

```
Slot 2:
  op1: %result = some_op(...)

Slot 5:
  op2: another_op(%result)  // Uses %result from slot 2
```

**Flow:**
1. **Slot 2 (Producer)**:
   - `op1` executes, produces `%result`
   - Store in localMap: `localMap[%result] = result_value`
   - Enqueue to cross_slot_fifo_s2_s5: `fifo->enq(result_value)`

2. **Slot 5 (Consumer)**:
   - `op2` needs `%result`
   - Check localMap: not found
   - Check cross_slot_fifos: found fifo_s2_s5
   - Dequeue: `result_value = fifo->deq()`
   - Store in localMap: `localMap[%result] = result_value`
   - Use `result_value`

**FIFO Created By:** `buildCrossSlotFIFOs()`

**Rule Generation:**
```cpp
// Slot 2 rule:
auto result_value = /* generate op1 */;
localMap[%result] = result_value;
// After all ops in slot:
for (auto result : op1->getResults()) {
  auto fifoIt = crossSlotFIFOs.find(result);
  if (fifoIt != crossSlotFIFOs.end()) {
    for (auto fifo : fifoIt->second) {
      fifo->fifoInstance->callMethod("enq", {localMap[result]}, b);
    }
  }
}

// Slot 5 rule:
// getValueInRule(%result) automatically:
//   1. Checks localMap (not found)
//   2. Finds cross_slot_fifo_s2_s5
//   3. Dequeues from FIFO
//   4. Stores in localMap for subsequent uses
```

### Pattern 3: Input Value Used Only in Slot 0

```
Input: %input_val (from input_fifos)

Slot 0:
  op1: some_op(%input_val)
  op2: another_op(%input_val)  // Uses same input twice
```

**Flow:**
1. **Slot 0 (First operation using input)**:
   - Dequeue from input_fifo: `val = input_fifo->deq()`
   - Store in localMap: `localMap[%input_val] = val`
   - `op1` uses value from localMap

2. **Slot 0 (Second operation using input)**:
   - Check localMap: found!
   - Use value from localMap (no dequeue)

**Key Point**: Input FIFO is dequeued **exactly once** at the beginning of slot 0, then stored in localMap for all uses in that slot.

**Rule Generation:**
```cpp
// Slot 0 rule:
if (slot == slotOrder.front()) {
  // Dequeue from input_fifos
  for (auto &[value, fifo] : inputFIFOs) {
    auto dequeuedValue = fifo->callMethod("deq", {}, b)[0];

    // Check if used in slot 0
    bool usedInSlot0 = /* check if value used by ops in slot 0 */;
    if (usedInSlot0) {
      localMap[value] = dequeuedValue;  // Available for all slot 0 ops
    }
  }
}
```

### Pattern 4: Input Value Used in Slot 0 AND Later Slots

```
Input: %input_val (from input_fifos)

Slot 0:
  op1: some_op(%input_val)

Slot 3:
  op2: another_op(%input_val)
```

**Flow:**
1. **Slot 0**:
   - Dequeue from input_fifo: `val = input_fifo->deq()`
   - Used in slot 0? → Store in localMap: `localMap[%input_val] = val`
   - Used in later slots? → Enqueue to cross_slot_fifo_s0_s3: `fifo->enq(val)`
   - **Same value goes to BOTH destinations**

2. **Slot 3**:
   - `op2` needs `%input_val`
   - Check localMap: not found (localMap cleared between slots)
   - Check cross_slot_fifos: found fifo_s0_s3
   - Dequeue: `val = fifo->deq()`
   - Store in localMap: `localMap[%input_val] = val`
   - Use value

**FIFO Created By:** `buildCrossSlotFIFOs()` - handles input values

**Rule Generation:**
```cpp
// Slot 0 rule:
if (slot == slotOrder.front()) {
  for (auto &[value, fifo] : inputFIFOs) {
    auto dequeuedValue = fifo->callMethod("deq", {}, b)[0];

    // Check if used in slot 0
    bool usedInSlot0 = /* check ops in slot 0 */;
    if (usedInSlot0) {
      localMap[value] = dequeuedValue;
    }

    // Check if used in later slots (has cross_slot_fifos)
    auto it = crossSlotFIFOs.find(value);
    if (it != crossSlotFIFOs.end()) {
      for (auto crossSlotFifo : it->second) {
        crossSlotFifo->fifoInstance->callMethod("enq", {dequeuedValue}, b);
      }
    }
  }
}
```

### Pattern 5: Input Value Used Only in Later Slots (Not Slot 0)

```
Input: %input_val (from input_fifos)

Slot 3:
  op1: some_op(%input_val)
```

**Flow:**
1. **Slot 0**:
   - Dequeue from input_fifo: `val = input_fifo->deq()`
   - NOT used in slot 0 → Don't store in localMap
   - Used in slot 3 → Enqueue to cross_slot_fifo_s0_s3: `fifo->enq(val)`

2. **Slot 3**:
   - Same as Pattern 4, dequeue from cross_slot_fifo

**Rule Generation:**
```cpp
// Slot 0 rule:
if (slot == slotOrder.front()) {
  for (auto &[value, fifo] : inputFIFOs) {
    auto dequeuedValue = fifo->callMethod("deq", {}, b)[0];

    // NOT used in slot 0, skip localMap

    // Enqueue to cross_slot_fifos for later use
    auto it = crossSlotFIFOs.find(value);
    if (it != crossSlotFIFOs.end()) {
      for (auto crossSlotFifo : it->second) {
        crossSlotFifo->fifoInstance->callMethod("enq", {dequeuedValue}, b);
      }
    }
  }
}
```

## Value Acquisition Algorithm (getValueInRule)

When an operation needs a value, the priority is:

1. **Check localMap first**
   - If found: return immediately
   - Most common case: value produced earlier in same slot

2. **Check if it's a constant**
   - If `arith::ConstantOp`: generate inline, store in localMap

3. **Check cross_slot_fifos**
   - Find FIFO for this value with matching consumer (current op)
   - Dequeue from FIFO
   - Store in localMap (for subsequent uses in same slot)
   - Return value

4. **Error**
   - Value not available

```cpp
FailureOr<mlir::Value> getValueInRule(mlir::Value v, Operation *currentOp,
                                      unsigned operandIndex, ...) {
  // 1. Check localMap
  if (auto it = localMap.find(v); it != localMap.end())
    return it->second;

  // 2. Handle constants
  if (auto constOp = v.getDefiningOp<arith::ConstantOp>()) {
    auto constant = /* generate constant */;
    localMap[v] = constant;
    return constant;
  }

  // 3. Check cross-slot FIFOs
  auto it = crossSlotFIFOs.find(v);
  if (it != crossSlotFIFOs.end()) {
    for (CrossSlotFIFO *fifo : it->second) {
      for (auto [consumerOp, opIndex] : fifo->consumers) {
        if (consumerOp == currentOp && opIndex == operandIndex) {
          auto result = fifo->fifoInstance->callValue("deq", b)[0];
          localMap[v] = result;  // Cache for later uses
          return result;
        }
      }
    }
  }

  // 4. Error
  return failure();
}
```

## Value Storage Algorithm

### Within Slot (Immediate Storage)

When an operation produces a value:
```cpp
auto result_value = /* generate operation */;
localMap[result] = result_value;  // Immediate storage for same-slot uses
```

### Cross-Slot (Deferred Storage)

After all operations in a slot are processed:
```cpp
// At end of slot rule
for (Operation *op : slotMap[slot].ops) {
  for (mlir::Value result : op->getResults()) {
    auto fifoIt = crossSlotFIFOs.find(result);
    if (fifoIt != crossSlotFIFOs.end()) {
      auto valueIt = localMap.find(result);
      if (valueIt != localMap.end()) {
        for (CrossSlotFIFO *fifo : fifoIt->second) {
          fifo->fifoInstance->callMethod("enq", {valueIt->second}, b);
        }
      }
    }
  }
}
```

## Output Values

At the end of the last slot, values that are in `output_fifos` are enqueued:

```cpp
if (slot == slotOrder.back()) {
  for (auto &[value, fifo] : outputFIFOs) {
    auto valueIt = localMap.find(value);
    if (valueIt != localMap.end()) {
      fifo->callMethod("enq", {valueIt->second}, b);
    }
  }
}
```

## Summary

| Value Source | Storage Location | When Stored | When Retrieved |
|-------------|------------------|-------------|----------------|
| Same slot operation | localMap | Immediately after op executes | By subsequent ops in same slot |
| Earlier slot operation | cross_slot_fifo | End of producer slot | Beginning of consumer slot |
| Input FIFO (used in slot 0) | localMap | Beginning of slot 0 | By ops in slot 0 |
| Input FIFO (used later) | cross_slot_fifo | Beginning of slot 0 | Beginning of consumer slot |
| Input FIFO (used in slot 0 + later) | Both localMap + cross_slot_fifo | Beginning of slot 0 | localMap in slot 0, FIFO in later slots |

## Key Principles

1. **localMap is slot-scoped** - Cleared at the beginning of each slot rule
2. **FIFOs are persistent** - Values enqueued in one slot, dequeued in another
3. **Single dequeue** - Each FIFO is dequeued exactly once per consumer
4. **Caching in localMap** - Once a value is obtained (from any source), it's cached in localMap for subsequent uses in the same slot
5. **Input FIFOs dequeued once** - In slot 0, input values are dequeued once and distributed to both localMap and cross_slot_fifos as needed
