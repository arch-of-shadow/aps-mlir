//===- LoopHandler.h - Loop Handler for FIFO-based Block Coordination -*- C++ -*-===//
//
// This file declares the loop handler that processes loops first, then
// delegates blocks to BBHandler using FIFO-based coordination
//
//===----------------------------------------------------------------------===//

#ifndef APS_LOOPHANDLER_H
#define APS_LOOPHANDLER_H

#include "APS/APSOps.h"
#include "APS/BlockHandler.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {

using namespace mlir;
using namespace mlir::tor;
using namespace circt::cmt2::ecmt2;
using namespace circt::cmt2::ecmt2::stl;
using namespace circt::firrtl;

//===----------------------------------------------------------------------===//
// Forward Declarations
//===----------------------------------------------------------------------===//

struct APSToCMT2GenPass;
class BBHandler;

//===----------------------------------------------------------------------===//
// Loop Data Structures
//===----------------------------------------------------------------------===//

/// Represents a loop with unified block-based coordination
struct LoopInfo {
  tor::ForOp forOp;
  std::string loopName;  // Hierarchical name for nested loops (e.g., "loop_0", "loop_0_body_1")

  // Loop control values
  Value inductionVar;
  Value lowerBound, upperBound, step;
  llvm::SmallVector<Value> iterArgs;
  llvm::SmallVector<Type> iterArgTypes;

  // Loop body blocks (using standard BlockInfo from BlockHandler)
  llvm::SmallVector<BlockInfo, 4> blocks;
  llvm::DenseMap<unsigned, BlockInfo *> blockMap;
  llvm::DenseMap<Block *, BlockInfo *> mlirBlockMap;

  // Token FIFOs for canonical loop coordination (entry → body → next)
  struct TokenFIFOs {
    Instance *to_body; // Entry rule signals body can start
    Instance *body_to_next;  // Body signals completion to next rule
    Instance *next_to_exit;  // Next rule signals loop completion
  } token_fifos;

  // State management FIFOs
  Instance *loop_state_fifo;  // Carries (counter, bound, step, iter_args)
  Instance *loop_result_fifo; // Final iter_args results
  llvm::SmallVector<Instance *, 4> iter_arg_fifos; // Individual iter_arg FIFOs

  LoopInfo() : loopName("uninitialized"),
        token_fifos{nullptr, nullptr, nullptr},
        loop_state_fifo(nullptr), loop_result_fifo(nullptr) {}

  // Initialize with actual loop information
  void initialize(tor::ForOp forOp, const std::string &name) {
    this->forOp = forOp;
    this->loopName = name;
  }
};

//===----------------------------------------------------------------------===//
// Loop Handler
//===----------------------------------------------------------------------===//

/// Specialized loop handler that derives from BlockHandler
/// Handles loop control structure (entry → body → next) while integrating
/// with the unified block system and producer-responsible FIFO coordination
class LoopHandler : public BlockHandler {
public:
  LoopHandler(APSToCMT2GenPass *pass, Module *mainModule, tor::FuncOp funcOp,
              Instance *poolInstance, Instance *roccInstance,
              Instance *hellaMemInstance, InterfaceDecl *dmaItfc,
              Circuit &circuit, Clock mainClk, Reset mainRst,
              unsigned long opcode, Instance *input_token_fifo,
              Instance *output_token_fifo,
              llvm::DenseMap<Value, Instance *> &input_fifos,
              llvm::DenseMap<Value, Instance *> &output_fifos);

  /// Process a loop block within the unified block system
  LogicalResult processLoopBlock(BlockInfo &loopBlock);

  /// Get the loop that this handler processes
  const LoopInfo &getLoop() const { return loop; }

protected:
  // Override processBlock to handle loop specialization
  LogicalResult processBlock(BlockInfo &block) override;

private:
  // Single loop that this handler processes
  LoopInfo loop;

  //===--------------------------------------------------------------------===//
  // Loop Analysis
  //===--------------------------------------------------------------------===//

  /// Extract blocks from loop body using BlockHandler segmentation
  LogicalResult extractLoopBlocks();

  /// Analyze data flow between loop blocks
  LogicalResult analyzeCrossBlockDataflow();

  //===--------------------------------------------------------------------===//
  // FIFO Infrastructure
  //===--------------------------------------------------------------------===//

  /// Simplified canonical loop infrastructure - no complex FIFOs needed
  /// The canonical entry/next rule pattern handles loop coordination directly

  //===--------------------------------------------------------------------===//
  // Rule Generation
  //===--------------------------------------------------------------------===//

  /// Generate canonical loop rules following Blockgen.md (entry → body → next)
  LogicalResult generateCanonicalLoopRules(BlockInfo &loopBlock);

  /// Helper: Generate loop entry rule (token coordination and state init)
  LogicalResult generateLoopEntryRule(BlockInfo &loopBlock);

  /// Helper: Generate loop next rule (termination check and state update)
  LogicalResult generateLoopNextRule(BlockInfo &loopBlock);

  /// Create loop infrastructure (FIFOs for coordination)
  LogicalResult createLoopInfrastructure();

  //===--------------------------------------------------------------------===//
  // Utility Methods
  //===--------------------------------------------------------------------===//

  /// Get unique FIFO name
  std::string getFIFOName(StringRef prefix, unsigned loopId,
                          unsigned blockId = 0, StringRef suffix = "");

  /// Generate hierarchical block name for nested blocks
  std::string generateBlockName(unsigned loopId, unsigned blockId,
                                const std::string &parentName = "");

  /// Find all values flowing between blocks
  llvm::SmallVector<CrossBlockValueFlow>
  findCrossBlockValues(BlockInfo &producerBlock, BlockInfo &consumerBlock);

  /// Check if value is used in target block
  bool isValueUsedInBlock(Value value, BlockInfo &targetBlock);

  /// Check if value is used in target loop
  bool isValueUsedInLoop(Value value, LoopInfo &targetLoop);

  /// Get loop for a given block
  LoopInfo *getLoopForBlock(Block *block);

  /// Analyze a single operation within a loop block
  void analyzeOperationInLoopBlock(Operation *op, BlockInfo &block);

  /// Get bit width for a type
  unsigned getBitWidth(mlir::Type type);

  /// Process loop body operations using BlockHandler with token coordination
  LogicalResult processLoopBodyOperations(tor::ForOp forOp, BlockInfo &loopBlock);

  /// Process loop body as blocks using proper segmentation (similar to processFunctionAsBlocks)
  LogicalResult processLoopBodyAsBlocks(tor::ForOp forOp, Block *loopBody);

  /// Check if a value is used in the loop body
  bool isValueUsedInLoopBody(Value value, Block *loopBody);

private:
  // Induction variable FIFO - created in infrastructure, used by entry rule
  Instance *inductionVarFIFO = nullptr;
};

} // namespace mlir

#endif // APS_LOOPHANDLER_H