//===- RuleGeneration.cpp - Rule Generation for TOR Functions -------------===//
//
// This file implements the rule generation functionality for TOR functions
// that was previously in APSToCMT2GenPass.cpp
//
// REFACTORED: Now uses BBHandler for object-oriented basic block management
// and LoopHandler for FIFO-based loop coordination
//
//===----------------------------------------------------------------------===//

#include "APS/APSToCMT2.h"
#include "APS/BlockHandler.h"
#include "APS/BBHandler.h"
#include "APS/LoopHandler.h"

namespace mlir {

using namespace mlir;
using namespace mlir::tor;
using namespace circt::cmt2::ecmt2;
using namespace circt::cmt2::ecmt2::stl;
using namespace circt::firrtl;

//===----------------------------------------------------------------------===//
// Main Rule Generation Function - Refactored to use BBHandler and LoopHandler
//===----------------------------------------------------------------------===//

/// Generate rules for a specific TOR function - proper implementation from
/// rulegenpass.cpp
void APSToCMT2GenPass::generateRulesForFunction(
    Module *mainModule, tor::FuncOp funcOp, Instance *poolInstance,
    Instance *roccInstance, Instance *hellaMemInstance, InterfaceDecl *dmaItfc,
    Circuit &circuit, Clock mainClk, Reset mainRst, unsigned long opcode) {

  // First, check if function contains loops - if so, use LoopHandler
  bool hasLoops = false;
  funcOp.walk([&](tor::ForOp forOp) {
    hasLoops = true;
    return WalkResult::interrupt();
  });

  // Unified entry point: Always use BlockHandler, regardless of whether there are loops
  llvm::outs() << "[RuleGen] Function " << funcOp.getName() 
               << " - using unified BlockHandler for all processing\n";

  // Create BlockHandler to manage all blocks with FIFO coordination
  // BlockHandler will internally delegate to specialized handlers (LoopHandler, BBHandler) as needed
  BlockHandler blockHandler(this, mainModule, funcOp, poolInstance,
                           roccInstance, hellaMemInstance, dmaItfc, circuit,
                           mainClk, mainRst, opcode);

  // Process all blocks - BlockHandler will handle loops, basic blocks, conditionals, etc.
  if (failed(blockHandler.processFunctionAsBlocks())) {
    funcOp.emitError("failed to process blocks for rule generation");
    return;
  }

  llvm::outs() << "[BlockHandler] Successfully processed all blocks for function "
               << funcOp.getName() << "\n";
}

} // namespace mlir