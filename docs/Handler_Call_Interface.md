# Handler Call Interface Documentation

This document describes how BlockHandler, BBHandler, and LoopHandler call each other and what data they pass through their interfaces.

## Handler Architecture Overview

```
APSToCMT2GenPass::generateRulesForFunction()
└── BlockHandler (top-level coordinator - function level)
    ├── BBHandler (for regular basic blocks)
    └── LoopHandler (for loop blocks)
        └── BlockHandler (for loop body - recursive)
            └── BBHandler (for basic blocks within loop body)
```

## 1. BlockHandler → BBHandler

**Location**: `lib/APS/APSToCMT2/BlockHandler.cpp:581-591`

### Call Site: `processRegularBlockWithBBHandler()`

```cpp
LogicalResult BlockHandler::processRegularBlockWithBBHandler(BlockInfo& block) {
  // Create BBHandler with all required infrastructure
  BBHandler bbHandler(pass, mainModule, funcOp, poolInstance, roccInstance,
                     hellaMemInstance, dmaItfc, circuit, mainClk, mainRst,
                     opcode);

  // Use the new BlockInfo interface for cleaner API and proper blockName access
  return bbHandler.processBasicBlock(block);
}
```

### Interface: `BBHandler::processBasicBlock(BlockInfo& block)`

**Location**: `lib/APS/APSToCMT2/BBHandler.cpp:550`

**Signature**:
```cpp
LogicalResult BBHandler::processBasicBlock(BlockInfo& block);
```

**Data Passed via BlockInfo**:
- `block.mlirBlock` - MLIR basic block to process
- `block.blockId` - Unique block identifier
- `block.blockName` - Block name for instance naming (e.g., "b0", "b1")
- `block.input_fifos` - Map of input values to their producer FIFOs
- `block.output_fifos` - Map of output values to their consumer FIFOs
- `block.input_token_fifo` - Token FIFO from previous block in sequence
- `block.output_token_fifo` - Token FIFO to next block in sequence
- `block.type` - Block type (REGULAR, LOOP_HEADER, etc.)
- `block.is_loop_block` - Whether this is a loop block
- `block.is_conditional_block` - Whether this is a conditional block

**BBHandler Responsibilities**:
1. Process only ONE basic block (not whole function)
2. Organize operations by time slots (starttime attribute)
3. Create cross-slot FIFOs for values flowing between slots
4. Generate slot rules with proper coordination
5. Handle input FIFO dequeuing in first slot
6. Handle output FIFO enqueuing in last slot
7. Use `blockName` for all instance naming

## 2. BlockHandler → LoopHandler

**Location**: `lib/APS/APSToCMT2/BlockHandler.cpp:405-424`

### Call Site: `processBlock()` for loop blocks

```cpp
LogicalResult BlockHandler::processBlock(BlockInfo &block) {
  // Check if this is a loop block and handle it with LoopHandler
  if (block.is_loop_block || block.type == BlockType::LOOP_HEADER) {
    // PANIC: Ensure loop blocks have proper infrastructure
    if (!block.input_token_fifo || !block.output_token_fifo) {
      llvm::report_fatal_error("Loop block missing required token FIFOs");
    }

    // Create a LoopHandler to process this loop block with proper FIFO arguments
    LoopHandler loopHandler(pass, mainModule, funcOp, poolInstance,
                           roccInstance, hellaMemInstance, dmaItfc, circuit,
                           mainClk, mainRst, opcode,
                           block.input_token_fifo, block.output_token_fifo,
                           block.input_fifos, block.output_fifos);

    // Process the loop block with the LoopHandler
    if (failed(loopHandler.processLoopBlock(block))) {
      return failure();
    }

    return success();
  }
  // ... handle other block types ...
}
```

### Interface: `LoopHandler` Constructor + `processLoopBlock()`

**Constructor Location**: `include/APS/LoopHandler.h`

**Signature**:
```cpp
LoopHandler(APSToCMT2GenPass *pass, Module *mainModule, tor::FuncOp funcOp,
            Instance *poolInstance, Instance *roccInstance,
            Instance *hellaMemInstance, InterfaceDecl *dmaItfc,
            Circuit &circuit, Clock mainClk, Reset mainRst,
            unsigned long opcode,
            Instance *input_token_fifo,
            Instance *output_token_fifo,
            llvm::DenseMap<Value, Instance*> &input_fifos,
            llvm::DenseMap<Value, Instance*> &output_fifos);
```

**Data Passed**:
- `pass` - Compiler pass context
- `mainModule` - CMT2 main module
- `funcOp` - MLIR function operation
- `poolInstance` - Memory pool instance
- `roccInstance` - RoCC interface instance
- `hellaMemInstance` - Hella memory interface instance
- `dmaItfc` - DMA interface declaration
- `circuit` - FIRRTL circuit
- `mainClk`, `mainRst` - Clock and reset signals
- `opcode` - Instruction opcode
- `input_token_fifo` - Token from previous block (BlockInfo.input_token_fifo)
- `output_token_fifo` - Token to next block (BlockInfo.output_token_fifo)
- `input_fifos` - Input value FIFOs (BlockInfo.input_fifos)
- `output_fifos` - Output value FIFOs (BlockInfo.output_fifos)

**LoopHandler Responsibilities**:
1. Create loop infrastructure (entry, next, body token FIFOs)
2. Create state registers for input values (size 1 input FIFOs)
3. Generate entry rule (dequeue from input_fifos → write to state registers)
4. Generate next rule (read from state registers → enqueue to body FIFOs)
5. Create induction variable FIFO
6. Delegate loop body processing to BlockHandler

## 3. LoopHandler → BlockHandler (Recursive)

**Location**: `lib/APS/APSToCMT2/LoopHandler.cpp:500-527`

### Call Site: `processLoopBody()`

```cpp
LogicalResult LoopHandler::processLoopBody(tor::ForOp forOp) {
  // ... create loop infrastructure ...

  // Prepare input FIFOs for loop body (original inputs + induction variable)
  llvm::DenseMap<Value, Instance*> loopBodyInputFIFOs = input_fifos;

  // Add the induction variable FIFO to the input map
  if (loop.inductionVar && inductionVarFIFO) {
    loopBodyInputFIFOs[loop.inductionVar] = inductionVarFIFO;
  }

  // Use BlockHandler's processLoopBodyAsBlocks for proper loop body processing
  BlockHandler loopBodyHandler(
      pass, mainModule, funcOp, poolInstance, roccInstance,
      hellaMemInstance, dmaItfc, circuit, mainClk, mainRst, opcode,
      loop.token_fifos.to_body,      // Input token: signals body can start
      loop.token_fifos.body_to_next, // Output token: signals body completion
      loopBodyInputFIFOs,            // Input data FIFOs (including loop variables)
      output_fifos                   // Output data FIFOs (to loop handler)
  );

  if (failed(loopBodyHandler.processLoopBodyAsBlocks(forOp))) {
    return failure();
  }

  return success();
}
```

### Interface: `BlockHandler::processLoopBodyAsBlocks()`

**Location**: `lib/APS/APSToCMT2/BlockHandler.cpp:109-139`

**Signature**:
```cpp
LogicalResult BlockHandler::processLoopBodyAsBlocks(tor::ForOp loopOp);
```

**Data Passed via Constructor** (before calling processLoopBodyAsBlocks):
- All standard BlockHandler constructor arguments
- `loop.token_fifos.to_body` as `input_token_fifo`
- `loop.token_fifos.body_to_next` as `output_token_fifo`
- `loopBodyInputFIFOs` (includes induction variable FIFO) as `input_fifos`
- `output_fifos` from original LoopHandler

**BlockHandler (loop body) Responsibilities**:
1. Segment loop body into blocks
2. Analyze cross-block dataflow within loop body
3. Create FIFOs for values flowing between blocks in loop body
4. Process each block (may call BBHandler for basic blocks, recursively call LoopHandler for nested loops)

## 4. BlockHandler → BlockHandler (Segmentation)

**Location**: `lib/APS/APSToCMT2/BlockHandler.cpp:344-367`

### Context: `segmentBlockIntoBlocks()`

When BlockHandler segments a large block into smaller blocks, it filters input FIFOs based on usage:

```cpp
// Propagate input FIFOs only if the value is actually used in this segment
for (const auto &pair : input_fifos) {
    Value value = pair.first;
    Instance *fifo = pair.second;

    // Check if this value is actually used in this segment
    if (isValueUsedInBlock(value, block)) {
        block.input_fifos[value] = fifo;
    }
}
```

This ensures subblocks only receive FIFOs for values they actually use.

## Key Data Structures

### BlockInfo (defined in BlockHandler.h)

```cpp
struct BlockInfo {
  unsigned blockId;
  std::string blockName;              // e.g., "b0", "b1" - used for naming
  Block *mlirBlock;
  BlockType type;
  bool is_loop_block;
  bool is_conditional_block;

  // Coordination FIFOs
  Instance *input_token_fifo;         // Token from previous block
  Instance *output_token_fifo;        // Token to next block

  // Data FIFOs
  llvm::DenseMap<Value, Instance*> input_fifos;   // Value → producer FIFO
  llvm::DenseMap<Value, Instance*> output_fifos;  // Value → consumer FIFO

  // Analysis data
  int64_t startTime;
  int64_t endTime;
  llvm::SmallVector<Value> produced_values;
  llvm::SmallVector<Value> consumed_values;
};
```

### LoopInfo (defined in LoopHandler.h)

```cpp
struct LoopInfo {
  tor::ForOp forOp;
  Value inductionVar;

  // Token FIFOs for loop coordination
  struct {
    Instance *to_entry;      // Start → entry rule
    Instance *entry_to_body; // Entry → body
    Instance *to_body;       // For body execution
    Instance *body_to_next;  // Body → next rule
    Instance *next_to_entry; // Next → entry (loop back)
    Instance *exit;          // Exit loop
  } token_fifos;

  // State registers: persistent storage for input values
  // (because input_fifos are size 1, both entry and next need access)
  llvm::DenseMap<Value, Instance*> input_state_registers;
};
```

## Call Flow Diagrams

### Top-Level Function Processing

```
BlockHandler::processAllBlocks()
  └─> for each block:
      └─> processBlock(BlockInfo& block)
          ├─> if loop block:
          │   └─> LoopHandler::processLoopBlock()
          │       └─> LoopHandler::processLoopBody()
          │           └─> BlockHandler::processLoopBodyAsBlocks() [RECURSIVE]
          │               └─> (back to processAllBlocks for loop body blocks)
          │
          └─> if regular block:
              └─> processRegularBlockWithBBHandler()
                  └─> BBHandler::processBasicBlock()
```

### Data FIFO Flow

```
BlockHandler (outer)
├─> Creates cross-block FIFOs for values
├─> Passes BlockInfo with input_fifos/output_fifos to handlers
│
├─> To BBHandler (regular block):
│   └─> BBHandler receives:
│       ├─> input_fifos (from previous blocks)
│       ├─> output_fifos (to next blocks)
│       ├─> input_token_fifo (coordination)
│       └─> output_token_fifo (coordination)
│
└─> To LoopHandler (loop block):
    └─> LoopHandler receives:
        ├─> input_fifos (from previous blocks)
        ├─> output_fifos (to next blocks)
        ├─> input_token_fifo (coordination)
        └─> output_token_fifo (coordination)
        │
        └─> LoopHandler creates state registers for input values
        │
        └─> To BlockHandler (loop body):
            └─> BlockHandler receives:
                ├─> loopBodyInputFIFOs (input_fifos + induction var FIFO)
                ├─> output_fifos (passed through from outer)
                ├─> to_body token FIFO (loop coordination)
                └─> body_to_next token FIFO (loop coordination)
```

## Interface Verification Checklist

- ✅ **BlockHandler → BBHandler**: Passes `BlockInfo&` with all FIFOs
- ✅ **BlockHandler → LoopHandler**: Passes token FIFOs and data FIFOs via constructor
- ✅ **LoopHandler → BlockHandler**: Passes loop token FIFOs + input FIFOs (with induction var)
- ✅ **BBHandler naming**: Uses `currentBlock->blockName` for all instances
- ✅ **LoopHandler state registers**: Created for input values (input_fifos are size 1)
- ✅ **Input FIFO filtering**: BlockHandler filters by usage before passing to subblocks
- ✅ **Token FIFO coordination**: All handlers receive and use token FIFOs correctly
- ✅ **Cross-slot FIFOs**: BBHandler creates them internally, including for input values

## Summary

All handler interfaces are properly connected with the correct data structures:

1. **BlockHandler** is the top-level coordinator that segments code into blocks and delegates to specialized handlers
2. **BBHandler** processes single basic blocks with slot-based execution
3. **LoopHandler** processes loops with entry/next rules and delegates body to BlockHandler (recursive)
4. All handlers use `BlockInfo` or equivalent structured data for clean interfaces
5. FIFOs (both token and data) are properly threaded through all call chains
6. Input FIFOs are filtered by usage to avoid unnecessary connections
