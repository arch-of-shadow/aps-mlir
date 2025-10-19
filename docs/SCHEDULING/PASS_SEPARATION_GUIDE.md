# TOR Scheduling Pass Separation Guide

## Overview

The TOR scheduling system has been split into **two separate passes** to allow transformations between scheduling and time graph generation:

1. **TORSchedulePass** (`schedule-tor`) - Performs scheduling and sets `ref_*` attributes
2. **TORTimeGraphPass** (`tor-time-graph`) - Generates time graph from `ref_*` attributes

## What Changed

### Before (Single Pass)
```
TORSchedulePass:
  1. Run scheduling algorithm
  2. Set ref_* attributes
  3. Build time graph
  4. Set starttime/endtime
  5. Emit tor.time_graph op
```

### After (Two Passes)
```
TORSchedulePass:
  1. Run scheduling algorithm
  2. Set ref_* attributes
  3. Set "scheduled" attribute
  4. STOP ✋

[YOUR TRANSFORMATION PASS GOES HERE]

TORTimeGraphPass:
  1. Read ref_* attributes
  2. Build time graph
  3. Set starttime/endtime
  4. Emit tor.time_graph op
```

## Attribute Types

### `ref_starttime` / `ref_endtime` (Cycle Numbers)
- Set by: **TORSchedulePass**
- Type: `IntegerAttr` (i32)
- Meaning: **Absolute cycle times** from scheduling
- Example: `ref_starttime = 3`, `ref_endtime = 4`

### `starttime` / `endtime` (Node IDs)
- Set by: **TORTimeGraphPass**
- Type: `IntegerAttr` (i32)
- Meaning: **Time graph node IDs**
- Example: `starttime = 5`, `endtime = 7`

### `scheduled` (Marker)
- Set by: **TORSchedulePass**
- Type: `BoolAttr`
- Meaning: Function has been scheduled and is ready for time graph generation

## Usage

### Pass Pipeline (Old)
```bash
# Old way (single pass)
aps-schd --schedule-tor input.mlir -o output.mlir
```

### Pass Pipeline (New)
```bash
# Step 1: Schedule (sets ref_* attributes)
aps-schd --schedule-tor input.mlir -o scheduled.mlir

# Step 2: Your transformation (e.g., split aps.load)
your-transform scheduled.mlir -o transformed.mlir

# Step 3: Generate time graph
aps-schd --tor-time-graph transformed.mlir -o final.mlir
```

### Programmatic Usage
```cpp
// In your pass pipeline
pm.addPass(mlir::createTORSchedulePass());        // Sets ref_*

// Insert your transformation here
pm.addPass(createYourTransformationPass());

pm.addPass(mlir::createTORTimeGraphPass());       // Builds time graph
```

## Example Transformation: Split aps.load

### Input (after TORSchedulePass)
```mlir
func.func @example() {
  %val = aps.load %mem[%i] {
    ref_starttime = 3 : i32,
    ref_endtime = 4 : i32
  }
  %result = arith.addi %val, %c1 {
    ref_starttime = 4 : i32,
    ref_endtime = 5 : i32
  }
}
```

### Your Transformation Pass
```cpp
// Split aps.load into aps.load_req + aps.load_collect
void splitApsLoad(aps::Load loadOp) {
    auto req_time = loadOp->getAttr("ref_starttime");
    auto collect_time = loadOp->getAttr("ref_endtime");

    // Create req at same time as load
    auto req = builder.create<aps::LoadReqOp>(
        loadOp.getLoc(), loadOp.getMemref(), loadOp.getIndices());
    req->setAttr("ref_starttime", req_time);
    req->setAttr("ref_endtime", req_time);

    // Create collect at next time
    auto collect = builder.create<aps::LoadCollectOp>(
        loadOp.getLoc(), req.getResult());
    collect->setAttr("ref_starttime", collect_time);
    collect->setAttr("ref_endtime", collect_time);

    loadOp.replaceAllUsesWith(collect.getResult());
    loadOp.erase();
}
```

### Output (ready for TORTimeGraphPass)
```mlir
func.func @example() {
  %req = aps.load_req %mem[%i] {
    ref_starttime = 3 : i32,
    ref_endtime = 3 : i32
  }
  %val = aps.load_collect %req {
    ref_starttime = 4 : i32,
    ref_endtime = 4 : i32
  }
  %result = arith.addi %val, %c1 {
    ref_starttime = 4 : i32,
    ref_endtime = 5 : i32
  }
}
```

### Final (after TORTimeGraphPass)
```mlir
tor.time_graph {
  tor.succ 1 : [0 : i32] [{type = "static"}]
  tor.succ 2 : [1 : i32] [{type = "static:1"}]
  tor.finish
}

func.func @example() {
  %req = aps.load_req %mem[%i] {
    starttime = 0 : i32,
    endtime = 1 : i32
  }
  %val = aps.load_collect %req {
    starttime = 1 : i32,
    endtime = 1 : i32
  }
  %result = arith.addi %val, %c1 {
    starttime = 1 : i32,
    endtime = 2 : i32
  }
}
```

## Key Validation Rules

When inserting transformations between the passes:

### ✅ Safe Operations
- Read `ref_*` attributes
- Create new operations with `ref_*` attributes
- Modify operation order (but keep `ref_*` times valid)
- Replace operations with equivalent ones

### ❌ Unsafe Operations
- Change `ref_*` values (breaks scheduling contract)
- Remove `ref_*` attributes (time graph pass needs them)
- Create operations without `ref_*` (will be ignored by time graph)
- Violate data dependencies (user expects value at earlier time)

### Validation Checklist
```python
def validate_transformation(op):
    # 1. Check ref_* attributes exist
    assert op.hasAttr("ref_starttime") and op.hasAttr("ref_endtime")

    # 2. Check users see value at correct time
    for user in op.getUsers():
        user_time = user.getAttr("ref_starttime").getInt()
        op_end_time = op.getAttr("ref_endtime").getInt()
        assert user_time >= op_end_time, "Data hazard!"

    # 3. Check producer completes before consumer
    for operand in op.getOperands():
        if producer := operand.getDefiningOp():
            producer_time = producer.getAttr("ref_endtime").getInt()
            consumer_time = op.getAttr("ref_starttime").getInt()
            assert consumer_time >= producer_time, "Dependency violation!"
```

## File Changes Summary

### Modified Files
1. **`lib/TOR/TORSchedulePass.cpp`**
   - Removed time graph building logic
   - Added `"scheduled"` attribute marker
   - Stops after setting `ref_*` attributes

### New Files
2. **`lib/TOR/TORTimeGraphPass.cpp`** (NEW)
   - Reads `ref_*` attributes
   - Builds time graph from scratch
   - Sets `starttime`/`endtime` attributes
   - Emits `tor.time_graph` operation

### Registration Files
3. **`include/TOR/Passes.td`**
   - Updated `TORSchedule` summary
   - Added `TORTimeGraph` pass definition

4. **`include/TOR/Passes.h`**
   - Added `createTORTimeGraphPass()` declaration

5. **`lib/TOR/CMakeLists.txt`**
   - Added `TORTimeGraphPass.cpp` to build

## Pass Characteristics

### TORSchedulePass
- **Input**: Unscheduled TOR IR
- **Output**: TOR IR with `ref_*` attributes
- **Side Effects**: Resource usage calculation
- **Failure Conditions**:
  - Schedule infeasible (II too small)
  - Resource constraints violated
  - Dataflow region resource exceeded

### TORTimeGraphPass
- **Input**: Scheduled TOR IR (with `ref_*` attributes)
- **Output**: TOR IR with time graph
- **Side Effects**: None
- **Failure Conditions**:
  - Missing `ref_*` attributes
  - Missing `"scheduled"` marker

## Debugging

### Check if function is scheduled
```bash
# Look for ref_* attributes and "scheduled" marker
mlir-opt --mlir-print-debuginfo your.mlir | grep -E "(ref_starttime|ref_endtime|scheduled)"
```

### Check if time graph was generated
```bash
# Look for tor.time_graph operation
mlir-opt your.mlir | grep -A 5 "tor.time_graph"
```

### Common Issues

**Issue**: Time graph pass fails silently
```
Cause: Function missing "scheduled" attribute
Fix: Ensure TORSchedulePass ran successfully
```

**Issue**: Operations missing in time graph
```
Cause: Operations don't have ref_* attributes
Fix: Add ref_* when creating new operations
```

**Issue**: Data hazard at runtime
```
Cause: User expects value before it's produced
Fix: Validate ref_endtime <= user's ref_starttime
```

## Benefits of Separation

1. ✅ **Flexibility**: Insert arbitrary transformations between scheduling and time graph
2. ✅ **Modularity**: Each pass has single responsibility
3. ✅ **Testability**: Can test scheduling without time graph generation
4. ✅ **Debuggability**: Intermediate IR with `ref_*` is human-readable
5. ✅ **Composability**: Mix and match transformations in pass pipeline

## Migration Guide

### For Pass Users
```diff
# Old pipeline
- pm.addPass(createTORSchedulePass());
+ pm.addPass(createTORSchedulePass());
+ // Your transformations here
+ pm.addPass(createTORTimeGraphPass());
```

### For Transformation Writers
```cpp
// Read scheduling results
auto ref_start = op->getAttr("ref_starttime");
auto ref_end = op->getAttr("ref_endtime");

// Create new operations with same timing
newOp->setAttr("ref_starttime", ref_start);
newOp->setAttr("ref_endtime", ref_end);

// Time graph pass will handle the rest!
```

## Advanced Usage

### Conditional Time Graph Generation
```cpp
// Only generate time graph for certain functions
if (funcOp->hasAttr("scheduled") &&
    !funcOp->hasAttr("skip_timegraph")) {
    pm.addPass(createTORTimeGraphPass());
}
```

### Multiple Transformation Stages
```cpp
pm.addPass(createTORSchedulePass());

// Stage 1: Split memory operations
pm.addPass(createSplitMemoryOpsPass());

// Stage 2: Optimize pipeline
pm.addPass(createPipelineOptimizationPass());

// Stage 3: Insert control logic
pm.addPass(createControlLogicPass());

// Finally: Generate time graph
pm.addPass(createTORTimeGraphPass());
```

## References

- **Scheduling Algorithm Report**: `SCHEDULING_ALGORITHM_REPORT.md`
- **TOR Dialect Documentation**: `include/TOR/TORStmt.td`
- **Pass Infrastructure**: `include/TOR/PassDetail.h`
