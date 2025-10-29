# BlockHandler Input Distribution Fix

## Problem Statement

When BlockHandler processes a block with multiple sub-blocks, and an input value from `input_fifos` is needed by multiple sub-blocks, we encounter a **FIFO underflow problem**:

- **1 enqueue**: Parent block (or loop) enqueues value to `input_fifo` once
- **N dequeues**: Each of the N sub-blocks tries to dequeue from the same `input_fifo`
- **Result**: FIFO underflow after first dequeue

### Example Scenario

```
Loop (has input value X)
  └─> Loop Body Block (BlockHandler)
       ├─> Sub-block 1 (needs X)
       └─> Sub-block 2 (needs X)
```

Currently:
1. Loop enqueues X to `body_input_fifo` once
2. Sub-block 1 dequeues from `body_input_fifo` ✓
3. Sub-block 2 tries to dequeue from `body_input_fifo` ✗ (empty!)

## Solution Overview

Implement the same pattern as LoopHandler's `loop_to_body_fifos`:

1. **Analyze** which input values are used by which sub-blocks
2. **If no input values needed by sub-blocks**: Skip distribution entirely (0 extra cycles)
3. **If input values needed**: Create distribution infrastructure:
   - Create separate FIFOs for each sub-block that needs each input value
   - Add a **distribution rule** that dequeues from `input_fifos` and enqueues to sub-block-specific FIFOs
   - Pass sub-block-specific FIFOs to sub-blocks instead of original `input_fifos`

## Detailed Design

### 1. New Data Structures in BlockHandler

Add to `BlockHandler` class (or `BlockInfo` struct):

```cpp
// Maps: input_value -> sub_block_index -> dedicated FIFO for that sub-block
llvm::DenseMap<Value, llvm::DenseMap<unsigned, Instance*>> input_distribution_fifos;

// Flag: whether this block needs input distribution rule
bool needsInputDistribution = false;

// Distribution rule instance (created only if needsInputDistribution == true)
Rule* inputDistributionRule = nullptr;

// Token FIFO: parent -> distribution rule (only if distribution needed)
Instance* input_distribution_token_fifo = nullptr;
```

### 2. Analysis Phase

In `processFunctionAsBlocks` or `processLoopBodyAsBlocks`, after creating sub-blocks:

```cpp
LogicalResult analyzeInputDistributionNeeds() {
  // For each value in input_fifos
  for (auto &[inputValue, inputFIFO] : input_fifos) {
    if (!inputFIFO) continue;

    // Find which sub-blocks use this input value
    llvm::SmallVector<unsigned> subBlocksUsingValue;

    for (unsigned i = 0; i < blocks.size(); i++) {
      BlockInfo &subBlock = blocks[i];
      if (isValueUsedInBlock(inputValue, subBlock)) {
        subBlocksUsingValue.push_back(i);
      }
    }

    // If multiple sub-blocks use it, OR even if one sub-block uses it
    // (because parent already enqueued to input_fifo, we must dequeue)
    if (!subBlocksUsingValue.empty()) {
      needsInputDistribution = true;

      // Record which sub-blocks need this value
      for (unsigned subBlockIdx : subBlocksUsingValue) {
        // Will create FIFO for this (value, sub-block) pair
        // Actual FIFO creation happens in createInputDistributionInfrastructure
      }
    }
  }

  return success();
}
```

### 3. Infrastructure Creation

Create FIFOs and token coordination:

```cpp
LogicalResult createInputDistributionInfrastructure() {
  if (!needsInputDistribution) {
    llvm::outs() << "[BlockHandler] No input distribution needed, skipping\n";
    return success();
  }

  llvm::outs() << "[BlockHandler] Creating input distribution infrastructure\n";

  // 1. Create token FIFO: parent -> distribution rule
  auto *tokenFifoMod = STLLibrary::createFIFO1PushModule(1, circuit);
  std::string tokenFifoName = currentBlock->blockName + "_dist_token_fifo";
  input_distribution_token_fifo = mainModule->addInstance(
      tokenFifoName, tokenFifoMod, {mainClk.getValue(), mainRst.getValue()});

  // 2. For each input value used by sub-blocks, create distribution FIFOs
  for (auto &[inputValue, inputFIFO] : input_fifos) {
    if (!inputFIFO) continue;

    // Find which sub-blocks use this value
    for (unsigned i = 0; i < blocks.size(); i++) {
      BlockInfo &subBlock = blocks[i];
      if (!isValueUsedInBlock(inputValue, subBlock)) continue;

      // Create dedicated FIFO for this sub-block
      unsigned bitWidth = getBitWidth(inputValue.getType());
      auto *fifoMod = STLLibrary::createFIFO1PushModule(bitWidth, circuit);

      std::string fifoName = currentBlock->blockName + "_dist_to_block" +
                             std::to_string(i) + "_fifo_" +
                             std::to_string(input_distribution_fifos[inputValue].size());

      Instance *distFifo = mainModule->addInstance(
          fifoName, fifoMod, {mainClk.getValue(), mainRst.getValue()});

      input_distribution_fifos[inputValue][i] = distFifo;

      llvm::outs() << "[BlockHandler] Created distribution FIFO: " << fifoName
                   << " for sub-block " << i << "\n";
    }
  }

  return success();
}
```

### 4. Distribution Rule Generation

Create the rule that performs 1-to-N distribution:

```cpp
LogicalResult generateInputDistributionRule() {
  if (!needsInputDistribution) {
    return success();  // No rule needed
  }

  std::string ruleName = currentBlock->blockName + "_input_distribution";
  inputDistributionRule = mainModule->addRule(ruleName);

  // === GUARD ===
  // Per CMT2 pattern: always return 1'b1
  // Coordination is handled automatically by FIFO availability (deq/enq operations)
  inputDistributionRule->guard([](mlir::OpBuilder &b) {
    auto loc = b.getUnknownLoc();
    auto alwaysTrue = UInt::constant(1, 1, b, loc);
    b.create<circt::cmt2::ReturnOp>(loc, mlir::ValueRange{alwaysTrue.getValue()});
  });

  // === BODY ===
  inputDistributionRule->body([&](mlir::OpBuilder &b) {
    // 1. Dequeue input token
    input_token_fifo->callMethod("deq", {}, b);
    llvm::outs() << "[BlockHandler] Distribution rule: dequeued input token\n";

    // 2. For each input value, dequeue once and distribute to all sub-blocks
    for (auto &[inputValue, inputFIFO] : input_fifos) {
      if (!inputFIFO) continue;

      // Skip if no distribution targets for this value
      if (input_distribution_fifos.count(inputValue) == 0) continue;

      // Dequeue ONCE from input FIFO
      auto dequeuedValue = inputFIFO->callMethod("deq", {}, b)[0];
      llvm::outs() << "[BlockHandler] Distribution rule: dequeued input value\n";

      // Enqueue to ALL sub-block distribution FIFOs that need it
      for (auto &[subBlockIdx, distFifo] : input_distribution_fifos[inputValue]) {
        distFifo->callMethod("enq", {dequeuedValue}, b);
        llvm::outs() << "[BlockHandler] Distribution rule: enqueued to sub-block "
                     << subBlockIdx << " FIFO\n";
      }
    }

    // 3. Enqueue token to distribution token FIFO (signals distribution complete)
    auto tokenVal = UInt::constant(1, 1, b, mainModule->getLoc());
    input_distribution_token_fifo->callMethod("enq", {tokenVal.getValue()}, b);
    llvm::outs() << "[BlockHandler] Distribution rule: enqueued distribution token\n";

    b.create<circt::cmt2::ReturnOp>(mainModule->getLoc(), mlir::ValueRange{});
  });

  inputDistributionRule->finalize();

  return success();
}
```

### 5. Token Flow Modification

Update token coordination:

#### Case 1: No input distribution needed

```
Parent Block/Loop
  |
  | (enqueue to input_token_fifo)
  |
  v
First Sub-Block (blocks[0])
  |
  v
Second Sub-Block (blocks[1])
  |
  v
...
  |
  | (enqueue to output_token_fifo)
  |
  v
Next Block/Loop
```

#### Case 2: Input distribution needed

```
Parent Block/Loop
  |
  | (enqueue to input_token_fifo)
  |
  v
Input Distribution Rule
  | - dequeue from input_fifos
  | - enqueue to input_distribution_fifos
  | - enqueue to input_distribution_token_fifo
  |
  | (input_distribution_token_fifo)
  |
  v
First Sub-Block (blocks[0])
  | - uses input_distribution_fifos instead of input_fifos
  |
  v
Second Sub-Block (blocks[1])
  | - uses input_distribution_fifos instead of input_fifos
  |
  v
...
  |
  | (enqueue to output_token_fifo)
  |
  v
Next Block/Loop
```

### 6. Sub-Block Creation and FIFO Mapping

When creating sub-blocks, we need to properly handle token FIFOs and input FIFOs based on whether distribution is needed.

#### Key principle:
- **First sub-block**: Receives token from distribution rule (if exists) or parent, receives values from distribution FIFOs (if exist) or parent input_fifos
- **Subsequent sub-blocks**: Receive token from previous sub-block, receive values from their own output_fifos

```cpp
LogicalResult createAndProcessSubBlocks() {
  // After analyzing and creating infrastructure, process each sub-block
  for (unsigned i = 0; i < blocks.size(); i++) {
    BlockInfo &subBlock = blocks[i];

    // === 1. Determine input token FIFO for this sub-block ===
    Instance *subBlockInputTokenFifo;
    if (i == 0) {
      // First sub-block receives token from distribution rule (if exists) or parent
      if (needsInputDistribution) {
        // Distribution rule consumes parent's input_token_fifo
        // Distribution rule enqueues to input_distribution_token_fifo
        // First sub-block dequeues from input_distribution_token_fifo
        subBlockInputTokenFifo = input_distribution_token_fifo;
        llvm::outs() << "[BlockHandler] First sub-block uses distribution token FIFO\n";
      } else {
        // No distribution needed - first sub-block directly uses parent's input_token_fifo
        subBlockInputTokenFifo = input_token_fifo;
        llvm::outs() << "[BlockHandler] First sub-block uses parent input token FIFO\n";
      }
    } else {
      // Subsequent sub-blocks receive token from previous sub-block
      subBlockInputTokenFifo = blocks[i-1].output_token_fifo;
      llvm::outs() << "[BlockHandler] Sub-block " << i
                   << " uses previous sub-block's output token FIFO\n";
    }

    // === 2. Determine output token FIFO for this sub-block ===
    Instance *subBlockOutputTokenFifo;
    if (i == blocks.size() - 1) {
      // Last sub-block enqueues to parent's output_token_fifo
      subBlockOutputTokenFifo = output_token_fifo;
    } else {
      // Not last sub-block - will create its own output token FIFO
      // (handled by sub-block handler itself)
      subBlockOutputTokenFifo = nullptr;  // Will be created by sub-block
    }

    // === 3. Determine input value FIFOs for this sub-block ===
    llvm::DenseMap<Value, Instance*> subBlockInputFifos;

    if (i == 0 && needsInputDistribution) {
      // First sub-block with distribution: use distribution FIFOs
      // Only include values that THIS sub-block actually uses
      for (auto &[inputValue, subBlockMap] : input_distribution_fifos) {
        if (subBlockMap.count(i)) {
          subBlockInputFifos[inputValue] = subBlockMap[i];
          llvm::outs() << "[BlockHandler] First sub-block uses distribution FIFO for value\n";
        }
      }
    } else if (i == 0 && !needsInputDistribution) {
      // First sub-block without distribution: use parent's input_fifos directly
      subBlockInputFifos = input_fifos;
      llvm::outs() << "[BlockHandler] First sub-block uses parent input FIFOs directly\n";
    } else {
      // Subsequent sub-blocks (i > 0): use empty input_fifos
      // They get their inputs from cross-block FIFOs created by BlockHandler analysis
      // (cross-block dataflow between blocks[0] -> blocks[1], blocks[1] -> blocks[2], etc.)
      subBlockInputFifos.clear();
      llvm::outs() << "[BlockHandler] Sub-block " << i
                   << " uses empty input FIFOs (cross-block coordination)\n";
    }

    // === 4. Determine output value FIFOs for this sub-block ===
    llvm::DenseMap<Value, Instance*> subBlockOutputFifos;

    if (i == blocks.size() - 1) {
      // Last sub-block: use parent's output_fifos for values needed by parent's consumers
      subBlockOutputFifos = output_fifos;
    } else {
      // Not last sub-block: will create output FIFOs for cross-block dataflow
      // (handled by BlockHandler's cross-block analysis)
      subBlockOutputFifos.clear();
    }

    // === 5. Create and invoke sub-block handler ===
    std::string subBlockName = currentBlock->blockName + "_sub" + std::to_string(i);

    if (isLoopBlock(subBlock)) {
      // Create LoopHandler for loop sub-block
      LoopHandler loopHandler(
          pass, mainModule, funcOp,
          poolInstance, roccInstance, hellaMemInstance, dmaItfc,
          circuit, mainClk, mainRst, opcode, regRdInstance,
          subBlockInputTokenFifo,
          subBlockOutputTokenFifo,
          subBlockInputFifos,
          subBlockOutputFifos,
          subBlockName);

      if (failed(loopHandler.processLoopBlock(subBlock))) {
        return failure();
      }
    } else {
      // Create BBHandler for basic block sub-block
      BBHandler bbHandler(
          pass, mainModule, funcOp,
          poolInstance, roccInstance, hellaMemInstance, dmaItfc,
          circuit, mainClk, mainRst, opcode, regRdInstance,
          subBlockInputTokenFifo,
          subBlockOutputTokenFifo,
          subBlockInputFifos,
          subBlockOutputFifos,
          subBlockName);

      if (failed(bbHandler.processBasicBlock(subBlock))) {
        return failure();
      }
    }
  }

  return success();
}
```

#### Summary of FIFO Routing:

**Without Distribution (needsInputDistribution = false):**
```
Parent
  └─> input_token_fifo
       └─> First Sub-Block (directly uses parent's input_token_fifo and input_fifos)
            └─> output_token_fifo
                 └─> Second Sub-Block
                      └─> ...
                           └─> output_token_fifo
                                └─> Next Block
```

**With Distribution (needsInputDistribution = true):**
```
Parent
  └─> input_token_fifo
       └─> Distribution Rule
            ├─> deq from input_fifos (once per value)
            ├─> enq to input_distribution_fifos (once per sub-block that needs it)
            └─> enq to input_distribution_token_fifo
                 └─> First Sub-Block (uses input_distribution_token_fifo and input_distribution_fifos)
                      └─> output_token_fifo
                           └─> Second Sub-Block (also uses input_distribution_fifos)
                                └─> ...
                                     └─> output_token_fifo
                                          └─> Next Block
```

**Important Notes:**

1. **Distribution FIFOs are shared**: When distribution is needed, ALL sub-blocks that need a particular input value will dequeue from their respective distribution FIFOs (not just the first sub-block).

2. **Token flow is sequential**: Only the token flows sequentially through sub-blocks. Input values can be consumed by any sub-block at any time (as long as they have the token).

3. **Cross-block FIFOs still exist**: Sub-blocks still create cross-block FIFOs between each other for values produced within the body (not from parent inputs).

### 7. Implementation Order

1. **Add data structures** to BlockHandler.h
2. **Implement analysis** (`analyzeInputDistributionNeeds`)
3. **Implement infrastructure creation** (`createInputDistributionInfrastructure`)
4. **Implement distribution rule** (`generateInputDistributionRule`)
5. **Update sub-block processing** to use distribution FIFOs
6. **Test** with loop body containing multiple sub-blocks that share input values

## Edge Cases

### Edge Case 1: No sub-blocks
- `needsInputDistribution = false`
- No extra rule created
- 0 cycle overhead

### Edge Case 2: Input value used by only one sub-block
- Still create distribution FIFO (because parent already enqueued to input_fifo)
- Distribution rule dequeues once, enqueues once
- 1 cycle overhead (necessary to avoid FIFO protocol mismatch)

### Edge Case 3: Input value not used by any sub-block
- Don't create distribution FIFO for this value
- Don't dequeue from this input_fifo in distribution rule
- Parent's enqueue will remain in FIFO (potential issue - may need to dequeue and discard?)

**Resolution for Edge Case 3**: If an input value is not used by any sub-block, we should still dequeue it in the distribution rule (but not enqueue anywhere). This keeps FIFO protocol consistent.

### Edge Case 4: Nested blocks with distribution
- Outer block creates distribution for its sub-blocks
- Each sub-block (if it has further sub-blocks) creates its own distribution
- Each level handles its own distribution independently

## Performance Impact

- **No input distribution needed**: 0 extra cycles
- **Input distribution needed**: 1 extra cycle per block invocation
- Trade-off: 1 cycle cost vs. correct FIFO protocol

## Comparison with LoopHandler

| Aspect | LoopHandler | BlockHandler (this fix) |
|--------|-------------|-------------------------|
| Purpose | Distribute loop inputs to loop body iterations | Distribute block inputs to sub-blocks |
| Distribution target | Loop body blocks (via `loop_to_body_fifos`) | Sub-blocks (via `input_distribution_fifos`) |
| Token flow | Entry rule → body → next rule | Parent → distribution rule → sub-blocks |
| Conditional creation | Based on `isValueUsedInLoopBody` | Based on `isValueUsedInBlock` for each sub-block |
| Rule name | Loop entry rule (multi-purpose) | Dedicated `input_distribution` rule |
| Cycle cost | Included in loop entry overhead | 0 if not needed, 1 if needed |

## Testing Strategy

1. **Test 1**: Loop with body containing 2 blocks, both using same input value
   - Verify distribution rule created
   - Verify 1 dequeue, 2 enqueues
   - Verify both blocks receive correct values

2. **Test 2**: Loop with body containing 2 blocks, only 1 using input value
   - Verify distribution rule created
   - Verify 1 dequeue, 1 enqueue
   - Verify correct block receives value

3. **Test 3**: Loop with body containing 2 blocks, neither using input value
   - Verify distribution rule NOT created (or dequeues and discards)
   - Verify 0 cycle overhead

4. **Test 4**: Nested blocks with distribution
   - Verify each level creates its own distribution infrastructure
   - Verify correct token flow through all levels

## Open Questions

1. **Should we dequeue unused input values?**
   - Yes: Keeps FIFO protocol consistent, but wastes cycles
   - No: Leaves values in FIFO, may cause issues with subsequent invocations

   **Recommendation**: Yes, dequeue but don't enqueue anywhere (discard)

2. **Should distribution rule be part of BlockHandler or separate?**
   - Current design: Part of BlockHandler, conditionally created
   - Alternative: Separate DistributionHandler class

   **Recommendation**: Keep in BlockHandler for simplicity

3. **How to handle cross-block output values from sub-blocks?**
   - Current design: Sub-blocks enqueue to their output_fifos as usual
   - Parent block aggregates outputs (already implemented)

   **Recommendation**: No change needed
