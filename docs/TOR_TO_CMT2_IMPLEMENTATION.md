# TOR-to-CMT2 Verilog Generation Implementation Guide

**Document Version:** 1.0
**Last Updated:** 2025-10-19
**Purpose:** Step-by-step guide for implementing Verilog generation from scheduled TOR MLIR using the CMT2 dialect

---

## Table of Contents

1. [Overview](#overview)
2. [Background and Prerequisites](#background-and-prerequisites)
3. [Architecture Overview](#architecture-overview)
4. [Implementation Phases](#implementation-phases)
5. [Reference Information](#reference-information)
6. [Troubleshooting](#troubleshooting)

---

## Overview

### Goal

Convert scheduled TOR (Temporal Ordering) MLIR to synthesizable SystemVerilog by:
1. Grouping operations by scheduling cycles into **CMT2 Rules**
2. Connecting rules via **FIFOs** for data flow
3. Using CMT2's scheduler and conflict analysis
4. Lowering to FIRRTL and generating Verilog

### Key Concept

```
TOR MLIR (scheduled) → CMT2 MLIR (rules + FIFOs) → FIRRTL → SystemVerilog
```

Each basic block at the same start cycle becomes a **CMT2 Rule**. Rules communicate via **FIFOs** to enforce sequential execution and data transfer.

### Input Example

**File:** `cadl-frontend/2.mlir`

Scheduled TOR operations with attributes:
- `ref_starttime`: When operation begins
- `ref_endtime`: When operation completes
- `dump`: Operation identifier

### Expected Final Output

**File:** `output.sv` (SystemVerilog)

A synthesizable hardware module with:
- Pipelined execution following original schedule
- FIFO-based communication
- Ready/enable handshaking
- Clock and reset signals

---

## Background and Prerequisites

### Required Reading

#### 1. CMT2 Dialect Documentation

**Location:** `cadl-frontend/circt/CMT2.md`

**Key Sections to Read:**
- **Status**: Understanding CMT2 dialect capabilities
- **Tasks** (especially completed ones):
  - InstanceGraph Analysis
  - CallInfo Analysis
  - ConflictMatrix Analysis
  - Scheduler Analysis
  - Cmt2ToFIRRTL Conversion
  - Embedded DSL

**What to Learn:**
- CMT2 operations: `cmt2.module`, `cmt2.rule`, `cmt2.method`, `cmt2.value`
- How rules work: guard region (when to fire) + body region (what to do)
- Conflict matrices: `<>` (conflict), `/` (conflict-free), `<` (sequential before)
- External modules: `cmt2.module.extern.firrtl` with `cmt2.bind.method` and `cmt2.bind.value`

#### 2. CMT2 Example (GCD)

**Location:** `cadl-frontend/circt/test/Dialect/Cmt2/gcd.mlir`

**Study These Patterns:**
```mlir
// External module definition with conflict matrix
cmt2.module.extern.firrtl @reg : @Reg32(%clk, %rst) {
  cmt2.bind.value @read : (!firrtl.uint<1>) -> (!firrtl.uint<32>) [...]
  cmt2.bind.method @write : (!firrtl.uint<1>, !firrtl.uint<32>) -> (...) [...]
} {
  conflict = [[@write, @write]],
  sequenceBefore = [[@read, @write]]
}

// Module with rules
cmt2.module @gcd(%clk, %rst) {
  cmt2.instance @x = @reg(%clk, %rst)

  // Rule: guard region checks condition, body region executes
  cmt2.rule @swap() {
    // Guard region
    %condition = ... compute when to fire ...
    cmt2.return %condition : !firrtl.uint<1>
  } {
    // Body region
    ... execute operations ...
  }
}
```

**Key Observations:**
- Rules have two regions: guard (when) and body (what)
- External modules define interfaces with conflict relationships
- Instances connect to external modules
- Calls use `cmt2.call @instance @method (...)`

#### 3. TOR Scheduled MLIR

**Location:** `cadl-frontend/2.mlir`

**Key Attributes:**
- `ref_starttime`: Cycle when operation starts
- `ref_endtime`: Cycle when operation ends
- `dump`: Unique identifier for debugging
- `slot`: Memory port slot
- `unroll`: Loop unrolling factor

**Operations to Understand:**
- `aps.readrf`: Read from register file (input)
- `aps.writerf`: Write to register file (output)
- `aps.memload`: Load from memory
- `aps.memstore`: Store to memory
- `aps.itfc.burst_load_req`: Request burst memory load
- `aps.itfc.burst_load_collect`: Wait for burst load completion
- `aps.itfc.burst_store_req`: Request burst memory store
- `aps.itfc.burst_store_collect`: Wait for burst store completion
- `tor.addi`: Addition operation with timing
- `tor.for`: Loop with scheduling

#### 4. CIRCT Build System

**Location:** `cadl-frontend/circt/CLAUDE.md`

**Build Commands:**
```bash
# Build CIRCT
ninja -C build

# Run tests
ninja -C build check-circt

# Test specific file
build/bin/llvm-lit -v test/Dialect/Cmt2/gcd.mlir

# Run circt-opt
build/bin/circt-opt <input.mlir> [passes]

# Run firtool (FIRRTL to Verilog)
build/bin/firtool <input.mlir> --format=mlir -o output.sv
```

---

## Architecture Overview

### Conceptual Model

```
┌─────────────────────────────────────────────────────────┐
│ TOR Function (Scheduled)                                │
├─────────────────────────────────────────────────────────┤
│ Cycle 0-1:   Read inputs                                │
│ Cycle 1-17:  Burst load A                               │
│ Cycle 17-33: Burst load B (overlaps with A!)            │
│ Cycle 35-37: Compute (4-way unrolled loop)              │
│ Cycle 37-53: Burst store C                              │
│ Cycle 53-54: Write outputs                              │
└─────────────────────────────────────────────────────────┘
                          ↓
                    [Analysis]
                          ↓
┌─────────────────────────────────────────────────────────┐
│ CMT2 Module (Rules + FIFOs)                             │
├─────────────────────────────────────────────────────────┤
│ FIFO Instances:                                         │
│   @fifo_init_to_load                                    │
│   @fifo_load_to_compute                                 │
│   @fifo_compute_to_store                                │
│                                                          │
│ Rules (Sequential Execution):                           │
│   @rule_init       → enqueue to fifo_init_to_load       │
│   @rule_load_a     → dequeue, load, enqueue             │
│   @rule_load_b     → dequeue, load, enqueue             │
│   @rule_compute_0  → dequeue, compute, enqueue          │
│   @rule_compute_1  → (parallel with compute_0)          │
│   @rule_compute_2  → (parallel with compute_0)          │
│   @rule_compute_3  → (parallel with compute_0)          │
│   @rule_store      → dequeue, store                     │
│   @rule_finish     → write result                       │
└─────────────────────────────────────────────────────────┘
                          ↓
                  [CMT2 Scheduler]
                          ↓
┌─────────────────────────────────────────────────────────┐
│ FIRRTL Module (Scheduled Hardware)                      │
├─────────────────────────────────────────────────────────┤
│ - Ready/enable signals for each rule                    │
│ - Fire signals based on guards + conflicts              │
│ - Sequential logic with proper timing                   │
└─────────────────────────────────────────────────────────┘
                          ↓
                     [firtool]
                          ↓
                  SystemVerilog (.sv)
```

### FIFO-Based Communication

**Why FIFOs?**
1. **Enforce ordering:** Producer rule must complete before consumer
2. **Data transfer:** Values produced in cycle N used in cycle N+M
3. **Handshaking:** `notEmpty()` and `notFull()` provide natural guards
4. **Buffering:** Handle timing variations and pipeline stalls

**FIFO Protocol:**
```mlir
// Producer rule
cmt2.rule @producer() {
  %can_produce = ... // Check input conditions
  %fifo_ready = cmt2.call @fifo @notFull()
  %guard = firrtl.and %can_produce, %fifo_ready
  cmt2.return %guard
} {
  %data = ... // Compute data
  cmt2.call @fifo @enqueue(%data)
}

// Consumer rule
cmt2.rule @consumer() {
  %data_available = cmt2.call @fifo @notEmpty()
  cmt2.return %data_available
} {
  %data = cmt2.call @fifo @dequeue()
  ... // Use data
}
```

**Conflict Matrix for FIFO:**
```
enqueue <> enqueue  (conflict: only one enqueue per cycle)
dequeue <  enqueue  (sequential before: read before write)
dequeue /  dequeue  (conflict-free: multiple reads OK if non-destructive)
```

---

## Implementation Phases

### Phase 1: Scheduling Analysis

**Goal:** Extract scheduling information from TOR MLIR and group operations by cycle.

#### Step 1.1: Parse TOR Function

**Input File:** `cadl-frontend/2.mlir`

**Code Location to Create:** `lib/Conversion/TORToCmt2/ScheduleAnalysis.h`

**Algorithm:**
```cpp
struct ScheduledBlock {
  int startCycle;
  int endCycle;
  SmallVector<Operation*> operations;
};

class ScheduleAnalyzer {
  SmallVector<ScheduledBlock> analyzeFunction(tor::FuncOp funcOp) {
    // 1. Iterate all operations in function
    // 2. Extract ref_starttime and ref_endtime attributes
    // 3. Group operations with same starttime
    // 4. Sort by starttime
    // 5. Return ordered blocks
  }
};
```

**Expected Output (Intermediate):**

```
Schedule Analysis Results for @main:

Block 0: Cycles 0-1
  - op_23: aps.readrf %arg0
  - op_24: aps.readrf %arg1
  - op_25-28: memref.get_global @mem_a_*
  - op_30-33: memref.get_global @mem_b_*

Block 1: Cycles 1-17
  - op_29: aps.itfc.burst_load_req (mem_a)

Block 2: Cycles 17-18
  - op_34: aps.itfc.burst_load_collect (mem_a)

Block 3: Cycles 17-33
  - op_35: aps.itfc.burst_load_req (mem_b)

Block 4: Cycles 33-34
  - op_39: aps.itfc.burst_load_collect (mem_b)

Block 5: Cycles 35-37
  - Loop body (4-way unrolled):
    - op_36, 38, 39: Load a[0], b[0], add
    - op_45: Store c[0]
    - ... (repeat for 1, 2, 3)

Block 6: Cycles 37-53
  - op_73: aps.itfc.burst_store_req (mem_c)

Block 7: Cycles 53-54
  - op_74: aps.itfc.burst_store_collect
  - op_75: aps.writerf (result)
```

**How to Test:**
```bash
# Create test pass to print schedule analysis
build/bin/circt-opt 2.mlir -print-tor-schedule-analysis
```

#### Step 1.2: Analyze Data Dependencies

**Code Location:** `lib/Conversion/TORToCmt2/DataFlowAnalysis.h`

**Algorithm:**
```cpp
struct DataDependency {
  ScheduledBlock* producer;
  ScheduledBlock* consumer;
  SmallVector<Value> values;  // Values flowing between blocks
  int latency;  // Cycles between producer.endCycle and consumer.startCycle
};

class DataFlowAnalyzer {
  SmallVector<DataDependency> analyzeDependencies(
      ArrayRef<ScheduledBlock> blocks) {
    // For each pair of blocks (earlier, later):
    //   1. Find values defined in earlier block
    //   2. Find values used in later block
    //   3. Compute intersection
    //   4. If non-empty, create dependency
  }
};
```

**Expected Output:**
```
Data Flow Dependencies:

Block 0 → Block 1: [%0 (addr), %2-%5 (mem_refs)]
  Latency: 0 cycles
  FIFO needed: @fifo_init_to_load_a (depth=2)

Block 0 → Block 3: [%1 (addr), %7-%10 (mem_refs)]
  Latency: 16 cycles
  FIFO needed: @fifo_init_to_load_b (depth=2)

Block 2 → Block 5: [data from mem_a]
  Latency: 17 cycles
  FIFO needed: @fifo_load_a_to_compute (depth=16)

Block 4 → Block 5: [data from mem_b]
  Latency: 1 cycle
  FIFO needed: @fifo_load_b_to_compute (depth=16)

Block 5 → Block 6: [compute results]
  Latency: 0 cycles
  FIFO needed: @fifo_compute_to_store (depth=16)
```

**Reference:** See `lib/Dialect/Cmt2/Cmt2CallInfo.cpp` for similar dependency analysis in CMT2.

---

### Phase 2: External Module Library

**Goal:** Create reusable FIRRTL modules for primitives (FIFOs, memories, burst controllers).

#### Step 2.1: Design FIFO Module

**File to Create:** `lib/Dialect/Cmt2/ModuleLibrary/chisel/fifo/FIFO.scala`

**FIFO Interface:**
```scala
// Parameterized FIFO
class FIFO(width: Int, depth: Int) extends Module {
  val io = IO(new Bundle {
    // Data interface
    val enqData = Input(UInt(width.W))
    val enqEnable = Input(Bool())
    val enqReady = Output(Bool())

    val deqData = Output(UInt(width.W))
    val deqEnable = Input(Bool())
    val deqReady = Output(Bool())

    // Status
    val notEmpty = Output(Bool())
    val notFull = Output(Bool())
    val count = Output(UInt(log2Ceil(depth+1).W))
  })

  // Implementation: circular buffer with read/write pointers
  // ...
}
```

**CMT2 Wrapper (to create):** `lib/Dialect/Cmt2/ModuleLibrary/static/FIFO.mlir`

```mlir
firrtl.circuit "FIFO" {
  firrtl.module @FIFO(
    in %clock: !firrtl.clock,
    in %reset: !firrtl.uint<1>,
    in %enqData: !firrtl.uint<32>,
    in %enqEnable: !firrtl.uint<1>,
    out %enqReady: !firrtl.uint<1>,
    out %deqData: !firrtl.uint<32>,
    in %deqEnable: !firrtl.uint<1>,
    out %deqReady: !firrtl.uint<1>,
    out %notEmpty: !firrtl.uint<1>,
    out %notFull: !firrtl.uint<1>
  ) {
    // FIRRTL implementation or instantiate Chisel-generated module
  }
}
```

**CMT2 External Module Declaration:**

**File to Create:** `lib/Conversion/TORToCmt2/ExternalModules.cpp`

```cpp
void createFIFOExternalModule(OpBuilder &builder, int width, int depth) {
  auto extMod = builder.create<cmt2::ExtModuleFirrtlOp>(
    loc,
    builder.getStringAttr("FIFO_" + std::to_string(width) + "_" + std::to_string(depth)),
    builder.getStringAttr("FIFO")
  );

  Block *body = extMod.getBodyBlock();
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(body);

  // Add bind.method for enqueue
  builder.create<cmt2::BindMethodOp>(
    loc,
    builder.getStringAttr("enqueue"),
    TypeRange{FIRRTLBaseType::get(...)},  // data: uintN
    TypeRange{}  // no outputs
  );

  // Add bind.value for dequeue
  builder.create<cmt2::BindValueOp>(
    loc,
    builder.getStringAttr("dequeue"),
    TypeRange{},  // no inputs
    TypeRange{FIRRTLBaseType::get(...)}  // data: uintN
  );

  // Add bind.value for notEmpty
  builder.create<cmt2::BindValueOp>(
    loc,
    builder.getStringAttr("notEmpty"),
    TypeRange{},
    TypeRange{builder.getI1Type()}
  );

  // Add bind.value for notFull
  builder.create<cmt2::BindValueOp>(
    loc,
    builder.getStringAttr("notFull"),
    TypeRange{},
    TypeRange{builder.getI1Type()}
  );

  // Set conflict matrix
  extMod->setAttr("conflict", builder.getArrayAttr({
    builder.getArrayAttr({
      builder.getStringAttr("enqueue"),
      builder.getStringAttr("enqueue")
    })
  }));

  extMod->setAttr("sequenceBefore", builder.getArrayAttr({
    builder.getArrayAttr({
      builder.getStringAttr("dequeue"),
      builder.getStringAttr("enqueue")
    })
  }));
}
```

**Expected Output:** When instantiated in CMT2:

```mlir
cmt2.module.extern.firrtl @FIFO_32_16 : @FIFO(%clk: !firrtl.clock, %rst: !firrtl.uint<1>) {
  cmt2.bind.bare %clk, @clock : !firrtl.clock
  cmt2.bind.bare %rst, @reset : !firrtl.uint<1>

  cmt2.bind.method @enqueue : (!firrtl.uint<1>, !firrtl.uint<32>) -> (!firrtl.uint<1>) [
    enable = @enqEnable,
    ready = @enqReady,
    inputs = [@enqData],
    outputs = []
  ]

  cmt2.bind.value @dequeue : (!firrtl.uint<1>) -> (!firrtl.uint<32>) [
    ready = @deqReady,
    data = [@deqData]
  ]

  cmt2.bind.value @notEmpty : (!firrtl.uint<1>) -> (!firrtl.uint<1>) [
    ready = @c1,  // Always ready
    data = [@notEmpty]
  ]

  cmt2.bind.value @notFull : (!firrtl.uint<1>) -> (!firrtl.uint<1>) [
    ready = @c1,
    data = [@notFull]
  ]
} {
  conflict = [[@enqueue, @enqueue]],
  sequenceBefore = [[@dequeue, @enqueue]]
}
```

**How to Test:**
```bash
# Build Chisel FIFO
cd lib/Dialect/Cmt2/ModuleLibrary/chisel/fifo
sbt "runMain FIFO --target-dir ../../generated"

# Verify FIRRTL output
cat ../../generated/FIFO.fir

# Test in CMT2
build/bin/circt-opt test_fifo.mlir -cmt2-print-conflict-matrix
```

#### Step 2.2: Create Memory Bank Module

**File to Create:** `lib/Dialect/Cmt2/ModuleLibrary/chisel/memory/MemoryBank.scala`

**Interface:**
```scala
class MemoryBank(width: Int, depth: Int) extends Module {
  val io = IO(new Bundle {
    val readAddr = Input(UInt(log2Ceil(depth).W))
    val readEnable = Input(Bool())
    val readData = Output(UInt(width.W))
    val readReady = Output(Bool())

    val writeAddr = Input(UInt(log2Ceil(depth).W))
    val writeData = Input(UInt(width.W))
    val writeEnable = Input(Bool())
    val writeReady = Output(Bool())
  })

  val mem = SyncReadMem(depth, UInt(width.W))
  // Read-first semantics: read returns old value if simultaneous write
}
```

**CMT2 Declaration:**
```mlir
cmt2.module.extern.firrtl @MemoryBank_32_4 : @MemoryBank {
  cmt2.bind.value @read : (!firrtl.uint<1>, !firrtl.uint<2>) -> (!firrtl.uint<32>) [
    ready = @readReady,
    data = [@readData]
  ]

  cmt2.bind.method @write : (!firrtl.uint<1>, !firrtl.uint<2>, !firrtl.uint<32>) -> (...) [
    enable = @writeEnable,
    ready = @writeReady,
    inputs = [@writeAddr, @writeData],
    outputs = []
  ]
} {
  conflict = [[@write, @write], [@read, @write]],
  conflictFree = [[@read, @read]]
}
```

**Reference:** See `cadl-frontend/circt/test/Dialect/Cmt2/gcd.mlir` lines 23-40 for register example.

#### Step 2.3: Create Burst Controller Module

**File to Create:** `lib/Dialect/Cmt2/ModuleLibrary/chisel/burst/BurstController.scala`

**Purpose:** Interface with AXI-like burst memory interface.

**Interface:**
```scala
class BurstController extends Module {
  val io = IO(new Bundle {
    // Load interface
    val loadReqAddr = Input(UInt(32.W))
    val loadReqLen = Input(UInt(32.W))
    val loadReqEnable = Input(Bool())
    val loadReqReady = Output(Bool())

    val loadCollectEnable = Input(Bool())
    val loadCollectReady = Output(Bool())

    // Store interface
    val storeReqAddr = Input(UInt(32.W))
    val storeReqLen = Input(UInt(32.W))
    val storeReqEnable = Input(Bool())
    val storeReqReady = Output(Bool())

    val storeCollectEnable = Input(Bool())
    val storeCollectReady = Output(Bool())

    // AXI master port
    // ... AXI signals ...
  })
}
```

**CMT2 Declaration:**
```mlir
cmt2.module.extern.firrtl @BurstController : @BurstController {
  cmt2.bind.method @loadReq : (..., !firrtl.uint<32>, !firrtl.uint<32>) -> (...) [...]
  cmt2.bind.method @loadCollect : (...) -> (...) [...]
  cmt2.bind.method @storeReq : (..., !firrtl.uint<32>, !firrtl.uint<32>) -> (...) [...]
  cmt2.bind.method @storeCollect : (...) -> (...) [...]
} {
  conflict = [
    [@loadReq, @loadReq],
    [@storeReq, @storeReq],
    [@loadReq, @storeReq]
  ],
  sequenceBefore = [
    [@loadReq, @loadCollect],
    [@storeReq, @storeCollect]
  ]
}
```

#### Step 2.4: Update Module Library Manifest

**File to Edit:** `lib/Dialect/Cmt2/ModuleLibrary/manifest.yaml`

```yaml
modules:
  - name: FIFO
    type: chisel
    path: chisel/fifo/
    parameters:
      - width: {type: integer, default: 32}
      - depth: {type: integer, default: 16}
    conflict_matrix:
      conflict: [["enqueue", "enqueue"]]
      sequenceBefore: [["dequeue", "enqueue"]]

  - name: MemoryBank
    type: chisel
    path: chisel/memory/
    parameters:
      - width: {type: integer, default: 32}
      - depth: {type: integer, default: 4}
    conflict_matrix:
      conflict: [["write", "write"], ["read", "write"]]
      conflictFree: [["read", "read"]]

  - name: BurstController
    type: chisel
    path: chisel/burst/
    conflict_matrix:
      conflict: [["loadReq", "loadReq"], ["storeReq", "storeReq"], ["loadReq", "storeReq"]]
      sequenceBefore: [["loadReq", "loadCollect"], ["storeReq", "storeCollect"]]
```

**Expected Output:** Modules available for instantiation in CMT2 conversion pass.

---

### Phase 3: TOR-to-CMT2 Conversion Pass

**Goal:** Implement the main conversion pass that transforms TOR MLIR to CMT2 MLIR.

#### Step 3.1: Create Pass Infrastructure

**File to Create:** `lib/Conversion/TORToCmt2/TORToCmt2.cpp`

**File to Create:** `include/circt/Conversion/TORToCmt2.h`

**Basic Structure:**
```cpp
#include "circt/Dialect/Cmt2/Cmt2Ops.h"
#include "circt/Dialect/TOR/TORDialect.h"
#include "mlir/Pass/Pass.h"

namespace circt {

struct ConvertTORToCmt2Pass : public PassWrapper<ConvertTORToCmt2Pass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder builder(module.getContext());

    // 1. Find tor.design operation
    auto designOp = findDesignOp(module);
    if (!designOp) return signalPassFailure();

    // 2. Analyze scheduling
    ScheduleAnalyzer schedAnalyzer;
    auto blocks = schedAnalyzer.analyzeFunction(designOp);

    // 3. Analyze data flow
    DataFlowAnalyzer dfAnalyzer;
    auto dependencies = dfAnalyzer.analyzeDependencies(blocks);

    // 4. Create CMT2 circuit
    auto cmt2Circuit = createCmt2Circuit(builder, module);

    // 5. Create external modules
    createExternalModules(builder, cmt2Circuit, dependencies);

    // 6. Create main module
    auto cmt2Module = createMainModule(builder, cmt2Circuit, designOp);

    // 7. Create FIFO instances
    createFIFOInstances(builder, cmt2Module, dependencies);

    // 8. Convert blocks to rules
    for (auto &block : blocks) {
      createRuleForBlock(builder, cmt2Module, block, dependencies);
    }

    // 9. Create entry/exit methods
    createStartMethod(builder, cmt2Module);
    createResultValue(builder, cmt2Module);

    // 10. Erase original TOR operations
    designOp.erase();
  }
};

} // namespace circt
```

**Reference:** See similar conversion in `lib/Conversion/CalyxToHW/CalyxToHW.cpp`

#### Step 3.2: Implement Block-to-Rule Conversion

**Algorithm:**

```cpp
void createRuleForBlock(OpBuilder &builder,
                        cmt2::ModuleOp module,
                        ScheduledBlock &block,
                        ArrayRef<DataDependency> deps) {

  // Determine rule name
  std::string ruleName = "rule_block_" + std::to_string(block.startCycle);

  // Find input FIFOs (dependencies where this block is consumer)
  SmallVector<cmt2::InstanceOp> inputFIFOs;
  for (auto &dep : deps) {
    if (dep.consumer == &block) {
      inputFIFOs.push_back(findFIFOInstance(dep));
    }
  }

  // Find output FIFOs (dependencies where this block is producer)
  SmallVector<cmt2::InstanceOp> outputFIFOs;
  for (auto &dep : deps) {
    if (dep.producer == &block) {
      outputFIFOs.push_back(findFIFOInstance(dep));
    }
  }

  // Create rule operation
  auto ruleOp = builder.create<cmt2::RuleOp>(loc, ruleName);

  // Build guard region
  Block *guardBlock = new Block();
  ruleOp.getGuardRegion().push_back(guardBlock);
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(guardBlock);

    // Guard: all input FIFOs must have data
    SmallVector<Value> guardConditions;
    for (auto fifo : inputFIFOs) {
      auto notEmpty = builder.create<cmt2::CallOp>(
        loc,
        builder.getI1Type(),
        fifo.getSymName(),
        builder.getStringAttr("notEmpty"),
        ValueRange{}
      );
      guardConditions.push_back(notEmpty.getResult(0));
    }

    // Guard: all output FIFOs must have space
    for (auto fifo : outputFIFOs) {
      auto notFull = builder.create<cmt2::CallOp>(
        loc,
        builder.getI1Type(),
        fifo.getSymName(),
        builder.getStringAttr("notFull"),
        ValueRange{}
      );
      guardConditions.push_back(notFull.getResult(0));
    }

    // AND all conditions
    Value finalGuard = guardConditions[0];
    for (size_t i = 1; i < guardConditions.size(); ++i) {
      finalGuard = builder.create<firrtl::AndPrimOp>(
        loc, finalGuard, guardConditions[i]
      );
    }

    builder.create<cmt2::ReturnOp>(loc, finalGuard);
  }

  // Build body region
  Block *bodyBlock = new Block();
  ruleOp.getBodyRegion().push_back(bodyBlock);
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(bodyBlock);

    // Dequeue from input FIFOs
    SmallVector<Value> inputValues;
    for (auto fifo : inputFIFOs) {
      auto deqResult = builder.create<cmt2::CallOp>(
        loc,
        getDataType(fifo),  // FIFO data type
        fifo.getSymName(),
        builder.getStringAttr("dequeue"),
        ValueRange{}
      );
      inputValues.push_back(deqResult.getResult(0));
    }

    // Convert TOR operations to CMT2/FIRRTL operations
    IRMapping valueMap;
    for (Operation *op : block.operations) {
      convertTOROperation(builder, op, valueMap, inputValues);
    }

    // Enqueue to output FIFOs
    for (size_t i = 0; i < outputFIFOs.size(); ++i) {
      auto outputValue = getOutputValue(block, i, valueMap);
      builder.create<cmt2::CallOp>(
        loc,
        TypeRange{},  // No return type
        outputFIFOs[i].getSymName(),
        builder.getStringAttr("enqueue"),
        ValueRange{outputValue}
      );
    }

    builder.create<cmt2::ReturnOp>(loc);
  }
}
```

**Expected Output (for one block):**

```mlir
cmt2.rule @rule_block_35() {
  // Guard: check input data available and output space available
  %a_valid = cmt2.call @fifo_load_a @notEmpty() : () -> (!firrtl.uint<1>)
  %b_valid = cmt2.call @fifo_load_b @notEmpty() : () -> (!firrtl.uint<1>)
  %c_space = cmt2.call @fifo_compute_to_store @notFull() : () -> (!firrtl.uint<1>)
  %cond1 = firrtl.and %a_valid, %b_valid : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  %guard = firrtl.and %cond1, %c_space : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  cmt2.return %guard : !firrtl.uint<1>
} {
  // Body: dequeue, compute, enqueue
  %a = cmt2.call @fifo_load_a @dequeue() : () -> (!firrtl.uint<32>)
  %b = cmt2.call @fifo_load_b @dequeue() : () -> (!firrtl.uint<32>)
  %sum = firrtl.add %a, %b : (!firrtl.uint<32>, !firrtl.uint<32>) -> !firrtl.uint<33>
  %truncated = firrtl.bits %sum 31 to 0 : (!firrtl.uint<33>) -> !firrtl.uint<32>
  cmt2.call @fifo_compute_to_store @enqueue(%truncated) : (!firrtl.uint<32>) -> ()
  cmt2.return
}
```

#### Step 3.3: Implement Operation Conversion

**File to Create:** `lib/Conversion/TORToCmt2/TOROpConversion.cpp`

**Conversion Table:**

```cpp
void convertTOROperation(OpBuilder &builder,
                         Operation *torOp,
                         IRMapping &valueMap,
                         ArrayRef<Value> blockInputs) {

  // Map TOR operations to CMT2/FIRRTL operations

  if (auto addOp = dyn_cast<tor::AddIOp>(torOp)) {
    // tor.addi %a, %b -> firrtl.add %a, %b
    auto lhs = valueMap.lookup(addOp.getLhs());
    auto rhs = valueMap.lookup(addOp.getRhs());
    auto result = builder.create<firrtl::AddPrimOp>(
      addOp.getLoc(), lhs, rhs
    );
    valueMap.map(addOp.getResult(), result);
  }

  else if (auto loadOp = dyn_cast<aps::MemLoadOp>(torOp)) {
    // aps.memload %mem[%idx] -> cmt2.call @mem @read(%idx)
    auto mem = valueMap.lookup(loadOp.getMemRef());
    auto idx = valueMap.lookup(loadOp.getIndex());
    auto result = builder.create<cmt2::CallOp>(
      loadOp.getLoc(),
      loadOp.getType(),
      mem,  // Instance name
      builder.getStringAttr("read"),
      ValueRange{idx}
    );
    valueMap.map(loadOp.getResult(), result.getResult(0));
  }

  else if (auto storeOp = dyn_cast<aps::MemStoreOp>(torOp)) {
    // aps.memstore %val, %mem[%idx] -> cmt2.call @mem @write(%idx, %val)
    auto mem = valueMap.lookup(storeOp.getMemRef());
    auto idx = valueMap.lookup(storeOp.getIndex());
    auto val = valueMap.lookup(storeOp.getValue());
    builder.create<cmt2::CallOp>(
      storeOp.getLoc(),
      TypeRange{},
      mem,
      builder.getStringAttr("write"),
      ValueRange{idx, val}
    );
  }

  else if (auto burstLoadReq = dyn_cast<aps::BurstLoadReqOp>(torOp)) {
    // aps.itfc.burst_load_req %addr, %len -> cmt2.call @burst_ctrl @loadReq
    auto addr = valueMap.lookup(burstLoadReq.getAddr());
    auto len = valueMap.lookup(burstLoadReq.getLen());
    auto result = builder.create<cmt2::CallOp>(
      burstLoadReq.getLoc(),
      builder.getNoneType(),  // Token type
      builder.getStringAttr("burst_ctrl"),
      builder.getStringAttr("loadReq"),
      ValueRange{addr, len}
    );
    valueMap.map(burstLoadReq.getResult(), result.getResult(0));
  }

  else if (auto burstLoadCollect = dyn_cast<aps::BurstLoadCollectOp>(torOp)) {
    // aps.itfc.burst_load_collect %token -> cmt2.call @burst_ctrl @loadCollect
    auto token = valueMap.lookup(burstLoadCollect.getToken());
    builder.create<cmt2::CallOp>(
      burstLoadCollect.getLoc(),
      TypeRange{},
      builder.getStringAttr("burst_ctrl"),
      builder.getStringAttr("loadCollect"),
      ValueRange{token}
    );
  }

  else if (auto constOp = dyn_cast<arith::ConstantOp>(torOp)) {
    // arith.constant -> firrtl.constant
    auto value = constOp.getValue().cast<IntegerAttr>();
    auto result = builder.create<firrtl::ConstantOp>(
      constOp.getLoc(),
      convertToFIRRTLType(constOp.getType()),
      value
    );
    valueMap.map(constOp.getResult(), result);
  }

  // ... Add more conversions as needed
}
```

**Reference:**
- CMT2 operations: `include/circt/Dialect/Cmt2/Cmt2Ops.td`
- FIRRTL operations: `include/circt/Dialect/FIRRTL/FIRRTLOps.td`
- Example conversion: `lib/Conversion/CalyxToHW/CalyxToHW.cpp`

#### Step 3.4: Handle Loop Unrolling

**Special Case:** `tor.for` with `unroll` attribute

**Input (from 2.mlir lines 41-63):**
```mlir
tor.for %arg3 = (%c0_i32 : i32) to (%c3_i32 : i32) step (%c1_i32 : i32) {
  // Loop body with 4 parallel operations
} {unroll = 4 : i32}
```

**Conversion Strategy:**

**Option 1: Create Parallel Rules (Recommended)**
```cpp
void convertUnrolledLoop(OpBuilder &builder,
                         tor::ForOp forOp,
                         int unrollFactor) {

  // Create separate rule for each unrolled iteration
  for (int i = 0; i < unrollFactor; ++i) {
    std::string ruleName = "rule_compute_" + std::to_string(i);
    auto ruleOp = builder.create<cmt2::RuleOp>(loc, ruleName);

    // Clone loop body for this iteration
    // Replace loop induction variable with constant i
    IRMapping valueMap;
    valueMap.map(forOp.getInductionVar(),
                 builder.create<firrtl::ConstantOp>(loc, i));

    // ... build guard and body regions
  }
}
```

**Expected Output:**
```mlir
cmt2.rule @rule_compute_0() {
  %guard = ... // Check FIFOs
  cmt2.return %guard
} {
  %c0 = firrtl.constant 0 : !firrtl.uint<32>
  %a0 = cmt2.call @mem_a_0 @read(%c0) : (!firrtl.uint<32>) -> (!firrtl.uint<32>)
  %b0 = cmt2.call @mem_b_0 @read(%c0) : (!firrtl.uint<32>) -> (!firrtl.uint<32>)
  %sum0 = firrtl.add %a0, %b0 : ...
  cmt2.call @mem_c_0 @write(%c0, %sum0) : ...
}

cmt2.rule @rule_compute_1() {
  %guard = ...
  cmt2.return %guard
} {
  %c0 = firrtl.constant 0 : !firrtl.uint<32>
  %a1 = cmt2.call @mem_a_1 @read(%c0) : (!firrtl.uint<32>) -> (!firrtl.uint<32>)
  %b1 = cmt2.call @mem_b_1 @read(%c0) : (!firrtl.uint<32>) -> (!firrtl.uint<32>)
  %sum1 = firrtl.add %a1, %b1 : ...
  cmt2.call @mem_c_1 @write(%c0, %sum1) : ...
}

// rule_compute_2 and rule_compute_3 similarly
```

**Conflict Matrix:**
Since all 4 compute rules access different memory banks, they are **conflict-free** and can execute in parallel:
```cpp
module->setAttr("conflictFree", builder.getArrayAttr({
  builder.getArrayAttr({
    builder.getStringAttr("rule_compute_0"),
    builder.getStringAttr("rule_compute_1")
  }),
  builder.getArrayAttr({
    builder.getStringAttr("rule_compute_0"),
    builder.getStringAttr("rule_compute_2")
  }),
  // ... all pairs
}));
```

#### Step 3.5: Create Entry and Exit Points

**Entry Method (@start):**
```cpp
void createStartMethod(OpBuilder &builder, cmt2::ModuleOp module) {
  // Extract function arguments from original TOR function
  // For 2.mlir: %arg0, %arg1, %arg2 (addresses)

  auto methodOp = builder.create<cmt2::MethodOp>(
    loc,
    builder.getStringAttr("start"),
    FunctionType::get(
      builder.getContext(),
      {FIRRTLType::UInt(5), FIRRTLType::UInt(5), FIRRTLType::UInt(5)},  // args
      {}  // no returns
    )
  );

  // Guard: check not already running
  Block *guardBlock = new Block();
  methodOp.getGuardRegion().push_back(guardBlock);
  builder.setInsertionPointToStart(guardBlock);
  auto notRunning = builder.create<cmt2::CallOp>(
    loc, builder.getI1Type(),
    builder.getStringAttr("this"),
    builder.getStringAttr("isIdle"),
    ValueRange{}
  );
  builder.create<cmt2::ReturnOp>(loc, notRunning.getResult(0));

  // Body: store arguments to internal state, trigger first rule
  Block *bodyBlock = methodOp.addEntryBlock();
  builder.setInsertionPointToStart(bodyBlock);
  // Store arguments for use by rules
  // Set state to RUNNING
  builder.create<cmt2::ReturnOp>(loc);
}
```

**Exit Value (@result):**
```cpp
void createResultValue(OpBuilder &builder, cmt2::ModuleOp module) {
  auto valueOp = builder.create<cmt2::ValueOp>(
    loc,
    builder.getStringAttr("result"),
    FunctionType::get(
      builder.getContext(),
      {},  // no args
      {FIRRTLType::UInt(32)}  // return result
    )
  );

  // Guard: check computation finished
  Block *guardBlock = new Block();
  valueOp.getGuardRegion().push_back(guardBlock);
  builder.setInsertionPointToStart(guardBlock);
  auto isDone = builder.create<cmt2::CallOp>(
    loc, builder.getI1Type(),
    builder.getStringAttr("this"),
    builder.getStringAttr("isDone"),
    ValueRange{}
  );
  builder.create<cmt2::ReturnOp>(loc, isDone.getResult(0));

  // Body: return stored result
  Block *bodyBlock = valueOp.addEntryBlock();
  builder.setInsertionPointToStart(bodyBlock);
  // Read result from internal storage
  auto result = ...;  // Get result value
  builder.create<cmt2::ReturnOp>(loc, result);
}
```

---

### Phase 4: Integration and Testing

**Goal:** Wire everything together and validate the conversion.

#### Step 4.1: Register the Pass

**File to Edit:** `include/circt/Conversion/Passes.h`

```cpp
namespace circt {

std::unique_ptr<mlir::Pass> createConvertTORToCmt2Pass();

#define GEN_PASS_REGISTRATION
#include "circt/Conversion/Passes.h.inc"

} // namespace circt
```

**File to Edit:** `include/circt/Conversion/Passes.td`

```tablegen
def ConvertTORToCmt2 : Pass<"convert-tor-to-cmt2", "mlir::ModuleOp"> {
  let summary = "Convert scheduled TOR dialect to CMT2 dialect";
  let description = [{
    Converts scheduled TOR operations to CMT2 rules connected by FIFOs.
    Each basic block at the same scheduling cycle becomes a CMT2 rule.

    Prerequisites:
    - Input must be scheduled (have ref_starttime/ref_endtime attributes)
    - Loop unrolling should be applied before this pass

    Output:
    - CMT2 circuit with external modules (FIFOs, memories)
    - CMT2 module with rules implementing scheduled operations
    - FIFO instances for inter-rule communication
  }];

  let constructor = "circt::createConvertTORToCmt2Pass()";
  let dependentDialects = ["circt::cmt2::Cmt2Dialect",
                           "circt::firrtl::FIRRTLDialect"];
}
```

**File to Edit:** `lib/Conversion/CMakeLists.txt`

```cmake
add_subdirectory(TORToCmt2)
```

**File to Create:** `lib/Conversion/TORToCmt2/CMakeLists.txt`

```cmake
add_circt_conversion_library(CIRCTTORToCmt2
  TORToCmt2.cpp
  TOROpConversion.cpp
  ScheduleAnalysis.cpp
  DataFlowAnalysis.cpp
  ExternalModules.cpp

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  CIRCTCmt2
  CIRCTFIRRTL
  CIRCTTOR
  MLIRIR
  MLIRPass
  MLIRTransforms
)
```

#### Step 4.2: Create Test Cases

**File to Create:** `test/Conversion/TORToCmt2/simple_add.mlir`

**Purpose:** Minimal test with 2 blocks and 1 FIFO

```mlir
// RUN: circt-opt %s -convert-tor-to-cmt2 | FileCheck %s

module {
  tor.design @simple_add {
    tor.func @main(%arg0: i32, %arg1: i32) {
      // Block 0: Cycles 0-1
      %sum = tor.addi %arg0, %arg1 on (0 to 1) {ref_starttime = 0, ref_endtime = 1} : (i32, i32) -> i32

      // Block 1: Cycles 2-3
      %result = tor.addi %sum, %sum on (0 to 1) {ref_starttime = 2, ref_endtime = 3} : (i32, i32) -> i32

      tor.return {ref_starttime = 4, ref_endtime = 4}
    }
  }
}

// CHECK: cmt2.circuit {
// CHECK:   cmt2.module.extern.firrtl @FIFO
// CHECK:   cmt2.module @simple_add
// CHECK:     cmt2.instance @fifo_0_to_1 = @FIFO
// CHECK:     cmt2.rule @rule_block_0
// CHECK:       cmt2.call @fifo_0_to_1 @notFull
// CHECK:       cmt2.call @fifo_0_to_1 @enqueue
// CHECK:     cmt2.rule @rule_block_2
// CHECK:       cmt2.call @fifo_0_to_1 @notEmpty
// CHECK:       cmt2.call @fifo_0_to_1 @dequeue
```

**File to Create:** `test/Conversion/TORToCmt2/burst_add.mlir`

**Purpose:** Test with burst operations (simplified version of 2.mlir)

```mlir
// RUN: circt-opt %s -convert-tor-to-cmt2 | FileCheck %s

module {
  tor.design @burst_add {
    memref.global @mem_a : memref<4xi32> = dense<[1, 2, 3, 4]>
    memref.global @mem_b : memref<4xi32> = dense<[5, 6, 7, 8]>

    tor.func @main(%addr: i32) {
      %mem_a = memref.get_global @mem_a {ref_starttime = 0, ref_endtime = 1}
      %mem_b = memref.get_global @mem_b {ref_starttime = 0, ref_endtime = 1}
      %len = arith.constant 4 {ref_starttime = 0, ref_endtime = 1}

      // Block 1: Burst load
      %token = aps.itfc.burst_load_req %addr, %mem_a, %len {ref_starttime = 1, ref_endtime = 10}
      aps.itfc.burst_load_collect %token {ref_starttime = 10, ref_endtime = 11}

      // Block 2: Compute
      %c0 = arith.constant 0 {ref_starttime = 11, ref_endtime = 12}
      %a = aps.memload %mem_a[%c0] {ref_starttime = 11, ref_endtime = 12}
      %b = aps.memload %mem_b[%c0] {ref_starttime = 11, ref_endtime = 12}
      %sum = tor.addi %a, %b on (0 to 1) {ref_starttime = 12, ref_endtime = 13}

      tor.return {ref_starttime = 13, ref_endtime = 13}
    }
  }
}

// CHECK: cmt2.circuit {
// CHECK:   cmt2.module.extern.firrtl @BurstController
// CHECK:   cmt2.module.extern.firrtl @MemoryBank
// CHECK:   cmt2.module.extern.firrtl @FIFO
// CHECK:   cmt2.module @burst_add
// CHECK:     cmt2.instance @burst_ctrl = @BurstController
// CHECK:     cmt2.instance @mem_a = @MemoryBank
// CHECK:     cmt2.instance @mem_b = @MemoryBank
// CHECK:     cmt2.instance @fifo_{{.*}} = @FIFO
// CHECK:     cmt2.rule @rule_block_1
// CHECK:       cmt2.call @burst_ctrl @loadReq
// CHECK:     cmt2.rule @rule_block_10
// CHECK:       cmt2.call @burst_ctrl @loadCollect
// CHECK:     cmt2.rule @rule_block_11
// CHECK:       cmt2.call @mem_a @read
// CHECK:       firrtl.add
```

**How to Run:**
```bash
# Build with new pass
ninja -C build

# Run simple test
build/bin/circt-opt test/Conversion/TORToCmt2/simple_add.mlir -convert-tor-to-cmt2

# Run with FileCheck
build/bin/circt-opt test/Conversion/TORToCmt2/simple_add.mlir -convert-tor-to-cmt2 | \
  build/bin/FileCheck test/Conversion/TORToCmt2/simple_add.mlir

# Full test suite
ninja -C build check-circt-conversion-tortocmt2
```

#### Step 4.3: Test Full Pipeline (2.mlir → Verilog)

**Test Script:** `test_tor_to_verilog.sh`

```bash
#!/bin/bash
set -e

INPUT="cadl-frontend/2.mlir"
OUTPUT_DIR="build/test_output"
mkdir -p $OUTPUT_DIR

echo "Step 1: Convert TOR to CMT2"
build/bin/circt-opt $INPUT \
  -convert-tor-to-cmt2 \
  > $OUTPUT_DIR/step1_cmt2.mlir

echo "Step 2: Verify CMT2 (parse check)"
build/bin/circt-opt $OUTPUT_DIR/step1_cmt2.mlir

echo "Step 3: Print schedule analysis"
build/bin/circt-opt $OUTPUT_DIR/step1_cmt2.mlir \
  -cmt2-print-scheduler

echo "Step 4: Print conflict matrix"
build/bin/circt-opt $OUTPUT_DIR/step1_cmt2.mlir \
  -cmt2-print-conflict-matrix

echo "Step 5: Inline private functions"
build/bin/circt-opt $OUTPUT_DIR/step1_cmt2.mlir \
  -cmt2-inline-private-funcs \
  > $OUTPUT_DIR/step2_inlined.mlir

echo "Step 6: Verify and convert to FIRRTL"
build/bin/circt-opt $OUTPUT_DIR/step2_inlined.mlir \
  -cmt2-verify-private-funcs-inlined \
  -cmt2-verify-call-sequence \
  --lower-cmt2-to-firrtl \
  > $OUTPUT_DIR/step3_firrtl.mlir

echo "Step 7: Generate SystemVerilog"
build/bin/firtool $OUTPUT_DIR/step3_firrtl.mlir \
  --format=mlir \
  --disable-reg-randomization \
  -o $OUTPUT_DIR/output.sv

echo "Success! Generated Verilog at: $OUTPUT_DIR/output.sv"
cat $OUTPUT_DIR/output.sv
```

**Expected Output Files:**

1. **step1_cmt2.mlir** - CMT2 representation with rules and FIFOs
2. **step2_inlined.mlir** - CMT2 with private functions inlined
3. **step3_firrtl.mlir** - FIRRTL with ready/enable signals
4. **output.sv** - Synthesizable SystemVerilog

**Validation Checks:**
```bash
# Check CMT2 structure
grep -q "cmt2.circuit" $OUTPUT_DIR/step1_cmt2.mlir
grep -q "cmt2.rule @rule_block_0" $OUTPUT_DIR/step1_cmt2.mlir
grep -q "cmt2.instance @fifo" $OUTPUT_DIR/step1_cmt2.mlir

# Check FIRRTL conversion
grep -q "firrtl.circuit" $OUTPUT_DIR/step3_firrtl.mlir
grep -q "firrtl.module @flow_burst_add" $OUTPUT_DIR/step3_firrtl.mlir

# Check Verilog output
grep -q "module flow_burst_add" $OUTPUT_DIR/output.sv
grep -q "always @(posedge clock)" $OUTPUT_DIR/output.sv
```

---

### Phase 5: Optimization and Refinement

**Goal:** Improve generated code quality and performance.

#### Step 5.1: FIFO Depth Optimization

**Problem:** Default FIFO depth may be too large (waste resources) or too small (cause stalls).

**Analysis:**
```cpp
int computeOptimalFIFODepth(DataDependency &dep) {
  // Calculate based on producer/consumer cycle difference
  int latency = dep.consumer->startCycle - dep.producer->endCycle;

  // Estimate data volume
  int dataVolume = estimateDataVolume(dep.values);

  // Heuristic: depth = max(2, min(latency, dataVolume))
  return std::max(2, std::min(latency, dataVolume));
}
```

**For 2.mlir:**
- `fifo_load_a_to_compute`: Producer ends at cycle 18, consumer starts at 35
  - Latency = 17 cycles
  - Data volume = 16 elements (burst length)
  - Optimal depth = 16

- `fifo_compute_to_store`: Producer ends at cycle 37, consumer starts at 37
  - Latency = 0 cycles
  - Data volume = 16 elements
  - Optimal depth = 16 (buffer entire output)

**Implementation:**
```cpp
auto fifo = builder.create<cmt2::InstanceOp>(
  loc,
  builder.getStringAttr(fifoName),
  builder.getStringAttr("FIFO"),
  /*args=*/ValueRange{clk, rst},
  /*parameters=*/DictionaryAttr::get(builder.getContext(), {
    builder.getNamedAttr("width", builder.getI32IntegerAttr(dataWidth)),
    builder.getNamedAttr("depth", builder.getI32IntegerAttr(optimalDepth))
  })
);
```

#### Step 5.2: Rule Merging

**Opportunity:** Merge rules that always fire together to reduce overhead.

**Example:** `rule_block_17` (burst_load_collect) always fires immediately after `rule_block_1` (burst_load_req) completes.

**Criteria for Merging:**
1. No intermediate FIFO between rules
2. Consumer guard only depends on producer completion
3. No other rules can fire between them

**Implementation:**
```cpp
bool canMergeRules(RuleOp rule1, RuleOp rule2) {
  // Check if rule2's guard only checks rule1's completion
  // Check no other rules scheduled between them
  // Check no FIFO communication between them
  return ...;
}

void mergeRules(RuleOp rule1, RuleOp rule2) {
  // Combine guard regions (AND conditions)
  // Concatenate body regions
  // Update dependencies
}
```

**Benefit:** Reduces number of rules, simplifies scheduling, potentially improves timing.

#### Step 5.3: Memory Banking Optimization

**Analysis:** 2.mlir uses 4 memory banks for parallel access.

**Verify in CMT2:**
```mlir
cmt2.instance @mem_a_0 = @MemoryBank_32_4(%clk, %rst)
cmt2.instance @mem_a_1 = @MemoryBank_32_4(%clk, %rst)
cmt2.instance @mem_a_2 = @MemoryBank_32_4(%clk, %rst)
cmt2.instance @mem_a_3 = @MemoryBank_32_4(%clk, %rst)
```

**Conflict Matrix:**
- Accesses to different banks are conflict-free
- Accesses to same bank have read/write conflicts

**Check Generated Conflict Matrix:**
```bash
build/bin/circt-opt output.mlir -cmt2-print-conflict-matrix | grep mem_a
```

**Expected:**
```
rule_compute_0 / rule_compute_1  (conflict-free: different banks)
rule_compute_0 / rule_compute_2  (conflict-free: different banks)
rule_compute_0 / rule_compute_3  (conflict-free: different banks)
```

---

## Reference Information

### File Locations Summary

#### Input Files
- **TOR Scheduled MLIR:** `cadl-frontend/2.mlir`
- **CMT2 Documentation:** `cadl-frontend/circt/CMT2.md`
- **CMT2 Example:** `cadl-frontend/circt/test/Dialect/Cmt2/gcd.mlir`
- **CIRCT Build Docs:** `cadl-frontend/circt/CLAUDE.md`

#### Files to Create

**Pass Implementation:**
- `include/circt/Conversion/TORToCmt2.h` - Pass interface
- `lib/Conversion/TORToCmt2/TORToCmt2.cpp` - Main pass
- `lib/Conversion/TORToCmt2/ScheduleAnalysis.cpp` - Cycle grouping
- `lib/Conversion/TORToCmt2/DataFlowAnalysis.cpp` - Dependency analysis
- `lib/Conversion/TORToCmt2/ExternalModules.cpp` - FIFO/Memory module creation
- `lib/Conversion/TORToCmt2/TOROpConversion.cpp` - Operation conversion
- `lib/Conversion/TORToCmt2/CMakeLists.txt` - Build configuration

**External Modules:**
- `lib/Dialect/Cmt2/ModuleLibrary/chisel/fifo/FIFO.scala` - FIFO implementation
- `lib/Dialect/Cmt2/ModuleLibrary/chisel/memory/MemoryBank.scala` - Memory implementation
- `lib/Dialect/Cmt2/ModuleLibrary/chisel/burst/BurstController.scala` - Burst controller
- `lib/Dialect/Cmt2/ModuleLibrary/manifest.yaml` - Module catalog

**Tests:**
- `test/Conversion/TORToCmt2/simple_add.mlir` - Basic test
- `test/Conversion/TORToCmt2/burst_add.mlir` - Burst operations test
- `test/Conversion/TORToCmt2/loop_unroll.mlir` - Loop unrolling test
- `test/Conversion/TORToCmt2/lit.local.cfg` - Test configuration

**Build System:**
- Edit `include/circt/Conversion/Passes.h` - Declare pass
- Edit `include/circt/Conversion/Passes.td` - Define pass
- Edit `lib/Conversion/CMakeLists.txt` - Add subdirectory

#### Expected Outputs

**Intermediate (CMT2):**
- `build/test_output/step1_cmt2.mlir` - Generated CMT2 from 2.mlir

**Final (SystemVerilog):**
- `build/test_output/output.sv` - Synthesizable Verilog

### Key MLIR Operations Reference

#### CMT2 Operations
**Location:** `include/circt/Dialect/Cmt2/Cmt2Ops.td`

```mlir
cmt2.circuit { ... }                           // Top-level container
cmt2.module.extern.firrtl @name : @type { ... }  // External module
cmt2.module @name(%args...) { ... }            // CMT2 module
cmt2.instance @name = @type(%args...)          // Instantiate module
cmt2.rule @name() { guard } { body }           // Rule with guard and body
cmt2.method @name(%args...) -> (...) { guard } { body }  // Method
cmt2.value @name() -> (...) { guard } { body }  // Value
cmt2.call @instance @method(%args...)          // Call method/value
cmt2.bind.bare %value, @port                   // Bind clock/reset
cmt2.bind.method @name : (types) -> (types) [mapping]  // Bind method
cmt2.bind.value @name : (types) -> (types) [mapping]   // Bind value
cmt2.return %values...                         // Return from region
```

#### FIRRTL Operations
**Location:** `include/circt/Dialect/FIRRTL/FIRRTLOps.td`

```mlir
firrtl.circuit "name" { ... }                  // Top-level circuit
firrtl.module @name(...) { ... }               // FIRRTL module
firrtl.instance name @target                   // Instance
firrtl.add %a, %b                              // Addition
firrtl.sub %a, %b                              // Subtraction
firrtl.and %a, %b                              // Bitwise AND
firrtl.or %a, %b                               // Bitwise OR
firrtl.bits %val N to M                        // Bit extraction
firrtl.mux(%cond, %true, %false)               // Multiplexer
firrtl.when %cond { ... }                      // Conditional
firrtl.connect %dest, %src                     // Wire connection
firrtl.regreset %clk, %reset, %init            // Register with reset
firrtl.constant N                              // Constant value
```

### Common Patterns and Idioms

#### Pattern 1: FIFO Communication

```mlir
// Producer rule
cmt2.rule @producer() {
  %space_available = cmt2.call @output_fifo @notFull() : () -> (!firrtl.uint<1>)
  cmt2.return %space_available
} {
  %data = ... // Compute data
  cmt2.call @output_fifo @enqueue(%data) : (!firrtl.uint<32>) -> ()
}

// Consumer rule
cmt2.rule @consumer() {
  %data_available = cmt2.call @input_fifo @notEmpty() : () -> (!firrtl.uint<1>)
  cmt2.return %data_available
} {
  %data = cmt2.call @input_fifo @dequeue() : () -> (!firrtl.uint<32>)
  ... // Use data
}
```

#### Pattern 2: Memory Access

```mlir
// Read from memory
%data = cmt2.call @memory @read(%addr) : (!firrtl.uint<32>) -> (!firrtl.uint<32>)

// Write to memory
cmt2.call @memory @write(%addr, %data) : (!firrtl.uint<32>, !firrtl.uint<32>) -> ()
```

#### Pattern 3: Conditional Guard

```mlir
cmt2.rule @conditional() {
  %cond1 = ... // Some condition
  %cond2 = ... // Another condition
  %guard = firrtl.and %cond1, %cond2 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  cmt2.return %guard
} {
  // Only executes when both conditions true
}
```

---

## Troubleshooting

### Issue 1: Scheduling Information Missing

**Symptom:** Pass fails with "Operation missing ref_starttime attribute"

**Cause:** Input TOR MLIR not scheduled

**Solution:**
1. Ensure scheduling pass runs before conversion:
   ```bash
   circt-opt input.mlir -tor-schedule -convert-tor-to-cmt2
   ```
2. Check that all operations have `ref_starttime` and `ref_endtime` attributes

### Issue 2: FIFO Type Mismatches

**Symptom:** Type errors in generated CMT2 MLIR

**Cause:** FIFO data type doesn't match producer/consumer types

**Solution:**
1. Check data flow analysis correctly infers types
2. Ensure FIFO instantiation uses correct width parameter
3. Verify type conversion from TOR types to FIRRTL types

**Debug:**
```cpp
// Add debug output in DataFlowAnalyzer
llvm::errs() << "Producer type: " << producerValue.getType() << "\n";
llvm::errs() << "Consumer type: " << consumerValue.getType() << "\n";
llvm::errs() << "FIFO width: " << fifoWidth << "\n";
```

### Issue 3: CMT2 Scheduler Fails

**Symptom:** `-cmt2-print-scheduler` reports conflicts or precedence violations

**Cause:** Generated conflict matrix or precedence constraints incorrect

**Solution:**
1. Check external module conflict matrices match hardware semantics
2. Verify precedence attribute correctly encodes block ordering:
   ```mlir
   module @main {
     ...
   } {
     precedence = [[@rule_0, @rule_1], [@rule_1, @rule_2]]
   }
   ```
3. Run conflict matrix analysis to debug:
   ```bash
   circt-opt output.mlir -cmt2-print-conflict-matrix
   ```

### Issue 4: FIRRTL Conversion Fails

**Symptom:** `--lower-cmt2-to-firrtl` fails with errors

**Cause:**
- Private functions not inlined
- Call sequence violations

**Solution:**
1. Always run required passes before conversion:
   ```bash
   circt-opt input.mlir \
     -cmt2-inline-private-funcs \
     -cmt2-verify-private-funcs-inlined \
     -cmt2-verify-call-sequence \
     --lower-cmt2-to-firrtl
   ```
2. Check for `@this` calls in output (should be none after inlining)

### Issue 5: Generated Verilog Doesn't Synthesize

**Symptom:** Synthesis tool reports timing violations or combinational loops

**Cause:**
- Incorrect conflict matrices allowing simultaneous conflicting operations
- Missing pipeline registers

**Solution:**
1. Review conflict matrices for all external modules
2. Ensure rules with dependencies have proper sequencing
3. Check FIFO depths are sufficient to avoid deadlock
4. Verify scheduling in FIRRTL:
   ```bash
   circt-opt output.mlir --lower-cmt2-to-firrtl | grep "firrtl.when"
   ```

### Issue 6: Build Errors

**Symptom:** CMake or Ninja build fails

**Common Causes:**
1. Missing dependencies in CMakeLists.txt
2. Incorrect include paths
3. Pass not registered

**Solution:**
1. Ensure all dialect dependencies listed:
   ```cmake
   LINK_LIBS PUBLIC
     CIRCTCmt2
     CIRCTFIRRTL
     CIRCTTOR
   ```
2. Check header includes:
   ```cpp
   #include "circt/Dialect/Cmt2/Cmt2Ops.h"
   #include "circt/Dialect/FIRRTL/FIRRTLOps.h"
   #include "circt/Dialect/TOR/TORDialect.h"
   ```
3. Verify pass registration in Passes.td and Passes.h

### Debug Tools

**1. Print Pass IR:**
```bash
circt-opt input.mlir -convert-tor-to-cmt2 -mlir-print-ir-after-all
```

**2. Verify Operation:**
```bash
circt-opt output.mlir -verify-each
```

**3. Print Specific Analysis:**
```bash
# Print call info
circt-opt output.mlir -cmt2-print-call-info

# Print conflict matrix
circt-opt output.mlir -cmt2-print-conflict-matrix

# Print scheduler result
circt-opt output.mlir -cmt2-print-scheduler
```

**4. FileCheck Debugging:**
```bash
# Show what FileCheck sees
circt-opt test.mlir -convert-tor-to-cmt2 > /tmp/output.mlir
cat /tmp/output.mlir

# Run FileCheck manually
build/bin/FileCheck test.mlir < /tmp/output.mlir
```

---

## Success Criteria

### Phase 1: Analysis
- [ ] Can parse 2.mlir and extract scheduling information
- [ ] Can group operations by cycle correctly
- [ ] Can identify data dependencies between blocks

### Phase 2: External Modules
- [ ] FIFO module builds and generates correct FIRRTL
- [ ] MemoryBank module builds and has correct conflict matrix
- [ ] BurstController module matches AXI semantics
- [ ] Modules registered in ModuleLibrary manifest

### Phase 3: Conversion
- [ ] Pass parses TOR MLIR without errors
- [ ] Generates valid CMT2 MLIR
- [ ] Creates correct number of rules (one per block)
- [ ] Creates FIFOs for data dependencies
- [ ] Guards check FIFO status correctly
- [ ] Bodies perform correct operations
- [ ] Precedence constraints encode original schedule

### Phase 4: Validation
- [ ] Generated CMT2 MLIR parses successfully
- [ ] CMT2 scheduler completes without conflicts
- [ ] Private function inlining succeeds
- [ ] CMT2-to-FIRRTL conversion succeeds
- [ ] Generated FIRRTL is well-formed
- [ ] firtool generates SystemVerilog
- [ ] Verilog passes synthesis

### Phase 5: Quality
- [ ] FIFO depths optimized (no wasted resources)
- [ ] Rules merged where beneficial
- [ ] Conflict matrices accurate
- [ ] Generated code readable and debuggable
- [ ] Performance meets timing requirements

---

## Next Steps After Implementation

1. **Validate Correctness:**
   - Run simulation on generated Verilog
   - Compare outputs with TOR simulation
   - Verify cycle-accurate timing

2. **Performance Analysis:**
   - Measure resource usage (LUTs, FFs, BRAM)
   - Check timing closure at target frequency
   - Profile FIFO utilization

3. **Extend to More Examples:**
   - Test on different CADL programs
   - Support more TOR operations
   - Handle edge cases (nested loops, conditionals)

4. **Integration with CADL Frontend:**
   - Connect CADL parser → TOR → CMT2 → Verilog pipeline
   - Add end-to-end tests
   - Document full toolchain

5. **Optimize Generated Code:**
   - Implement retiming passes
   - Add pipelining for critical paths
   - Explore DSE (Design Space Exploration)

---

## Appendix: Full Example Walkthrough

### Input: 2.mlir (Simplified)

```mlir
tor.func @main(%addr_a: i32, %addr_b: i32) {
  // Cycle 0: Load memories
  %mem_a = memref.get_global @mem_a {ref_starttime = 0, ref_endtime = 1}

  // Cycle 1-17: Burst load A
  %c16 = arith.constant 16 {ref_starttime = 1, ref_endtime = 2}
  %token_a = aps.itfc.burst_load_req %addr_a, %mem_a, %c16 {ref_starttime = 1, ref_endtime = 17}
  aps.itfc.burst_load_collect %token_a {ref_starttime = 17, ref_endtime = 18}

  // Cycle 18-20: Compute
  %c0 = arith.constant 0 {ref_starttime = 18, ref_endtime = 19}
  %a = aps.memload %mem_a[%c0] {ref_starttime = 18, ref_endtime = 19}
  %b = aps.memload %mem_a[%c0] {ref_starttime = 18, ref_endtime = 19}
  %sum = tor.addi %a, %b on (0 to 1) {ref_starttime = 19, ref_endtime = 20}

  tor.return {ref_starttime = 20, ref_endtime = 20}
}
```

### Step 1: Schedule Analysis Output

```
Block 0 (cycles 0-1): get_global
Block 1 (cycles 1-17): burst_load_req
Block 2 (cycles 17-18): burst_load_collect
Block 3 (cycles 18-20): compute (load + add)
```

### Step 2: Data Flow Analysis Output

```
Dependency: Block 0 → Block 1
  Values: [%mem_a]
  FIFO: @fifo_init_to_load (depth=2, width=64)

Dependency: Block 2 → Block 3
  Values: [loaded data]
  FIFO: @fifo_load_to_compute (depth=16, width=32)
```

### Step 3: Generated CMT2 MLIR

```mlir
cmt2.circuit {
  // External modules
  cmt2.module.extern.firrtl @FIFO_64_2 : @FIFO(%clk, %rst) { ... }
  cmt2.module.extern.firrtl @FIFO_32_16 : @FIFO(%clk, %rst) { ... }
  cmt2.module.extern.firrtl @BurstController : @BurstController(%clk, %rst) { ... }
  cmt2.module.extern.firrtl @MemoryBank : @MemoryBank(%clk, %rst) { ... }

  cmt2.module @main(%clk: !firrtl.clock, %rst: !firrtl.uint<1>) {
    // Instances
    cmt2.instance @fifo_init = @FIFO_64_2(%clk, %rst)
    cmt2.instance @fifo_data = @FIFO_32_16(%clk, %rst)
    cmt2.instance @burst_ctrl = @BurstController(%clk, %rst)
    cmt2.instance @mem_a = @MemoryBank(%clk, %rst)

    // Rules
    cmt2.rule @rule_block_0() {
      %ready = cmt2.call @fifo_init @notFull() : () -> (!firrtl.uint<1>)
      cmt2.return %ready
    } {
      %mem_ref = ... // Get memory reference
      cmt2.call @fifo_init @enqueue(%mem_ref) : (!firrtl.uint<64>) -> ()
    }

    cmt2.rule @rule_block_1() {
      %has_data = cmt2.call @fifo_init @notEmpty() : () -> (!firrtl.uint<1>)
      cmt2.return %has_data
    } {
      %mem_ref = cmt2.call @fifo_init @dequeue() : () -> (!firrtl.uint<64>)
      %addr = ... // Extract address
      %len = firrtl.constant 16
      cmt2.call @burst_ctrl @loadReq(%addr, %len) : ...
    }

    cmt2.rule @rule_block_17() {
      %c1 = firrtl.constant 1
      cmt2.return %c1  // Always ready after previous block
    } {
      cmt2.call @burst_ctrl @loadCollect() : () -> ()
    }

    cmt2.rule @rule_block_18() {
      %has_data = cmt2.call @fifo_data @notEmpty() : () -> (!firrtl.uint<1>)
      cmt2.return %has_data
    } {
      %a = cmt2.call @mem_a @read(%c0) : (!firrtl.uint<32>) -> (!firrtl.uint<32>)
      %b = cmt2.call @mem_a @read(%c0) : (!firrtl.uint<32>) -> (!firrtl.uint<32>)
      %sum = firrtl.add %a, %b : (!firrtl.uint<32>, !firrtl.uint<32>) -> !firrtl.uint<33>
      %trunc = firrtl.bits %sum 31 to 0
      // Store result
    }

    // Entry method
    cmt2.method @start(%addr_a: !firrtl.uint<32>, %addr_b: !firrtl.uint<32>) -> () {
      ... // Guard and body
    }
  } {
    precedence = [
      [@rule_block_0, @rule_block_1],
      [@rule_block_1, @rule_block_17],
      [@rule_block_17, @rule_block_18]
    ]
  }
}
```

### Step 4: FIRRTL Output (after conversion)

```mlir
firrtl.circuit "main" {
  firrtl.module @main(in %clk: !firrtl.clock, in %rst: !firrtl.uint<1>, ...) {
    // Instances
    firrtl.instance fifo_init @FIFO_64_2
    firrtl.instance fifo_data @FIFO_32_16
    ...

    // Ready signals
    %rule_block_0_ready = ...
    %rule_block_1_ready = ...
    ...

    // Fire signals
    %rule_block_0_fire = firrtl.and %rule_block_0_ready, %can_fire_0
    %rule_block_1_fire = firrtl.and %rule_block_1_ready, %can_fire_1
    ...

    // Conditional execution
    firrtl.when %rule_block_0_fire {
      // Body of rule_block_0
    }

    firrtl.when %rule_block_1_fire {
      // Body of rule_block_1
    }
    ...
  }
}
```

### Step 5: SystemVerilog Output

```verilog
module main(
  input clock,
  input reset,
  input [31:0] addr_a,
  input [31:0] addr_b,
  input start_enable,
  output start_ready,
  output done
);

  // FIFO instances
  FIFO_64_2 fifo_init (
    .clock(clock),
    .reset(reset),
    ...
  );

  // Rule ready/fire signals
  wire rule_block_0_ready = fifo_init_notFull;
  wire rule_block_0_fire = rule_block_0_ready & can_fire_0;

  // Conditional execution
  always @(posedge clock) begin
    if (rule_block_0_fire) begin
      // Rule block 0 body
    end

    if (rule_block_1_fire) begin
      // Rule block 1 body
    end

    ...
  end

endmodule
```

---

**End of Document**
