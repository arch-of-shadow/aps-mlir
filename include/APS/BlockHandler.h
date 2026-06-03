//===- BlockHandler.h - Unified Block Handler -*- C++ -*-===//
//
// This file declares the unified block handler that treats all control flow
// as blocks with explicit token coordination and non-pipeline value registers
//
//===----------------------------------------------------------------------===//

#ifndef APS_BLOCKHANDLER_H
#define APS_BLOCKHANDLER_H

#include "APS/APSOps.h"
#include "APS/APSToCMT2.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {

using namespace mlir;
using namespace mlir::tor;
using namespace circt::cmt2::ecmt2;
using namespace circt::cmt2::ecmt2::stl;
using namespace circt::firrtl;

//===----------------------------------------------------------------------===//
// Block Types
//===----------------------------------------------------------------------===//

enum class BlockType {
  REGULAR,           // Sequential operations
  LOOP_HEADER,       // Loop initialization (tor.for)
  LOOP_BODY,         // Loop iteration body
  LOOP_EXIT,         // Loop termination
  CONDITIONAL_THEN,  // If-then branch
  CONDITIONAL_ELSE,  // If-else branch
  CONDITIONAL_EXIT   // After conditional
};

//===----------------------------------------------------------------------===//
// Block Information
//===----------------------------------------------------------------------===//

/// Represents a unified block with token-based communication.
struct BlockInfo {
  unsigned blockId;
  std::string blockName;  // Hierarchical name for nested blocks
  Block* mlirBlock;       // Parent MLIR block (for context)
  BlockType type;
  int64_t startTime, endTime;

  // Operations belonging to this specific block segment
  llvm::SmallVector<Operation*> operations;

  // Values produced/consumed by this block
  llvm::SmallVector<Value> producedValues;
  llvm::SmallVector<Value> consumedValues;

  // Scope-boundary FIFOs inherited from the parent. Internal block-to-block
  // values are carried by scopeResources registers in non-pipeline mode.
  llvm::DenseMap<Value, llvm::SmallVector<std::pair<BlockInfo*, Instance*>, 4>> output_fifos;
  llvm::DenseMap<Value, Instance*> input_fifos;   // Values this block consumes

  // Block execution coordination - unified token system
  Instance* input_token_fifo;     // Token coordination (prev block complete -> this block ready)
  Instance* output_token_fifo;    // Token to next block (this block complete -> next block ready)

  // Block-specific data (union-like pattern)
  bool is_loop_block;
  bool is_conditional_block;
  BlockScopeResources scopeResources;

  BlockInfo(unsigned blockId, const std::string& blockName, Block* block, BlockType type)
    : blockId(blockId), blockName(blockName), mlirBlock(block), type(type),
      startTime(-1), endTime(-1), input_token_fifo(nullptr), output_token_fifo(nullptr),
      is_loop_block(false), is_conditional_block(false) {}
};

/// Cross-block value flow information
struct CrossBlockValueFlow {
  Value value;
  BlockInfo* producer_block;
  BlockInfo* consumer_block;
  Instance* storage;
};

//===----------------------------------------------------------------------===//
// Block Handler Base Class
//===----------------------------------------------------------------------===//

/// Unified block handler with explicit token coordination.
class BlockHandler {
public:
  BlockHandler(APSToCMT2Pass *pass, Module *mainModule, tor::FuncOp funcOp,
               Instance *poolInstance, Instance *roccInstance,
               Instance *hellaMemInstance, InterfaceDecl *dmaItfc,
               InterfaceDecl *csrItfc, Circuit &circuit, Clock mainClk,
               Reset mainRst,
               unsigned long opcode, Instance *regRdInstance,
               Instance *inputTokenFIFO, Instance *outputTokenFIFO,
              llvm::DenseMap<Value, Instance*> &input_fifos,
              llvm::DenseMap<Value, llvm::SmallVector<std::pair<BlockInfo*, Instance*>, 4>> &output_fifos,
              const std::string &namePrefix = "");

  /// Process all blocks in the function
  LogicalResult processFunctionAsBlocks();

  LogicalResult processLoopBodyAsBlocks(tor::ForOp loopOp);

  /// Process a specific block (virtual for specialization)
  virtual LogicalResult processBlock(BlockInfo& block);

  /// Create producer-owned register for one produced value.
  Instance* createCrossBlockValueReg(Value value, unsigned producerBlockId,
                                     unsigned counter);

  /// Find all consumers of a value
  llvm::SmallVector<BlockInfo*> findValueConsumers(Value value);

  /// Register-backed parent live-in available to generated sub-blocks.
  void addInputRegister(Value value, Instance *reg);

  /// Enable per-iteration FIFO dataflow inside this block hierarchy.
  void setPipelineMode(bool enabled) { pipelineMode = enabled; }

protected:
  // Core components
  APSToCMT2Pass *pass;
  Module *mainModule;
  tor::FuncOp funcOp;
  Instance *poolInstance;
  Instance *roccInstance;
  Instance *hellaMemInstance;
  Instance *regRdInstance;  // Shared reg_rd register for all blocks
  InterfaceDecl *dmaItfc;
  InterfaceDecl *csrItfc;
  Circuit &circuit;
  Clock mainClk;
  Reset mainRst;
  unsigned long opcode;

  // Name prefix for hierarchical naming (e.g., "43_" for opcode, "43_loop_1_" for nested)
  std::string namePrefix;

  bool pipelineMode = false;

  // External token FIFOs for top-level block coordination
  Instance *inputTokenFIFO;
  Instance *outputTokenFIFO;

  // Block information
  llvm::SmallVector<BlockInfo, 4> blocks;
  llvm::DenseMap<unsigned, BlockInfo*> blockMap;
  llvm::DenseMap<Block*, BlockInfo*> mlirBlockMap;

  // Cross-block value flows backed by producer-owned registers.
  llvm::SmallVector<CrossBlockValueFlow> crossBlockFlows;

  llvm::SmallVector<CrossBlockValueFlow> pipelineInputFanoutFlows;
  
  // Unified token FIFOs for cross-block coordination (block i -> block i+1)
  llvm::DenseMap<std::pair<unsigned, unsigned>, Instance*> unifiedTokenFIFOs;

  // Scope-boundary input and output FIFOs inherited from the parent.
  llvm::DenseMap<Value, Instance*> input_fifos;
  llvm::DenseMap<Value, Instance*> input_regs;
  // Output FIFOs: for each value, list of (consumer_block, FIFO) pairs
  llvm::DenseMap<Value, llvm::SmallVector<std::pair<BlockInfo*, Instance*>, 4>> output_fifos;

  //===--------------------------------------------------------------------===//
  // Input Capture Infrastructure (for multi-consumer parent live-ins)
  //===--------------------------------------------------------------------===//

  // Maps: input_value -> sub_block_index -> shared capture register.
  // All sub-blocks using the same input value read the same register.
  llvm::DenseMap<Value, llvm::DenseMap<unsigned, Instance*>> input_distribution_regs;

  // Flag: whether this block needs an input capture rule
  bool needsInputDistribution = false;

  // Token FIFO: capture rule -> first sub-block (only if capture is needed)
  Instance* input_distribution_token_fifo = nullptr;

  //===--------------------------------------------------------------------===//
  // Block Analysis
  //===--------------------------------------------------------------------===//

  /// Identify all blocks in the function
  LogicalResult identifyBlocksByFuncOp();

  LogicalResult identifyBlocksByLoop(tor::ForOp loopOp);

  /// Segment a block into blocks based on control flow with FIFO propagation
  LogicalResult segmentBlockIntoBlocks(Block *mlirBlock, unsigned &blockId);

  /// Analyze a single operation within a block
  void analyzeOperationInBlock(Operation *op, BlockInfo &block);

  /// Analyze data flow between blocks
  LogicalResult analyzeCrossBlockDataflow();

  /// Determine block types (regular, loop, conditional, etc.)
  BlockType determineBlockType(Block* block);

  /// Check if block contains loop operations
  bool containsLoop(Block* block);

  /// Check if block contains conditional operations
  bool containsConditional(Block* block);

  //===--------------------------------------------------------------------===//
  // Register Infrastructure
  //===--------------------------------------------------------------------===//

  /// Create producer-owned registers for cross-block communication
  LogicalResult createCrossBlockValueRegs();

  /// In pipeline mode, create FIFO fanout for parent FIFO inputs used by
  /// multiple sub-blocks.
  LogicalResult createPipelineInputFanoutFIFOs();

  /// Get unique FIFO name
  std::string getFIFOName(StringRef prefix, unsigned blockId, StringRef suffix = "");

  //===--------------------------------------------------------------------===//
  // Input Capture (for sub-blocks with shared inputs)
  //===--------------------------------------------------------------------===//

  /// Analyze if input distribution is needed for sub-blocks
  LogicalResult analyzeInputDistributionNeeds();

  /// Create input capture registers and token coordination.
  LogicalResult createInputDistributionRegs();

  /// Generate input capture rule (dequeue once, write shared registers)
  LogicalResult generateInputCaptureRule();

  //===--------------------------------------------------------------------===//
  // Rule Generation
  //===--------------------------------------------------------------------===//

  /// Process all blocks through specialized handlers (BBHandler/LoopHandler)
  LogicalResult processAllBlocks();

  /// Process a regular block using internal BB logic
  LogicalResult processRegularBlockWithBBHandler(BlockInfo& block);

  /// Create intra-block coordination FIFOs (ready/complete for slot-to-slot)
  LogicalResult createBlockTokenFIFOs();

  //===--------------------------------------------------------------------===//
  // Utility Methods
  //===--------------------------------------------------------------------===//

  /// Get bit width for FIFO sizing
  unsigned getBitWidth(mlir::Type type);

  /// Check if value is used in target block
  bool isValueUsedInBlock(Value value, BlockInfo& targetBlock);

  /// Check if a value comes from a virtual operation (doesn't need FIFO)
  bool isVirtualValue(Value value);

  /// Generate hierarchical block name for nested blocks
  std::string generateBlockName(unsigned blockId, BlockType type, const std::string& parentName = "");
};

} // namespace mlir

#endif // APS_BLOCKHANDLER_H
