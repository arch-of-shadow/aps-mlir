//===- RuleGeneration.cpp - Rule Generation for TOR Functions -------------===//
//
// This file implements the rule generation functionality for TOR functions
// that was previously in APSToCMT2.cpp
//
// REFACTORED: Now uses BBHandler for object-oriented basic block management
// and LoopHandler for token-based loop coordination
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
void APSToCMT2Pass::generateRulesForFunction(
    Module *mainModule, tor::FuncOp funcOp, Instance *poolInstance,
    Instance *roccInstance, Instance *hellaMemInstance, InterfaceDecl *dmaItfc,
    InterfaceDecl *csrItfc, Circuit &circuit, Clock mainClk, Reset mainRst,
    unsigned long instructionId) {

  // First, check if function contains loops - if so, use LoopHandler
  bool hasLoops = false;
  funcOp.walk([&](tor::ForOp forOp) {
    hasLoops = true;
    return WalkResult::interrupt();
  });

  // Unified entry point: Always use BlockHandler, regardless of whether there are loops
  llvm::dbgs() << "[RuleGen] Function " << funcOp.getName()
               << " - using unified BlockHandler for all processing\n";

  // Create reg_rd register instance (one per instruction id, shared across all blocks)
  auto &builder = mainModule->getBuilder();
  auto savedIP = builder.saveInsertionPoint();
  auto *regRdMod = STLLibrary::createRegModule(5, 0, circuit);
  Instance *regRdInstance = mainModule->addInstance("reg_rd_" + (std::ostringstream() << std::hex << std::setw(4) << std::setfill('0') << instructionId).str(), regRdMod,
                                                    {mainClk.getValue(), mainRst.getValue()});
  builder.restoreInsertionPoint(savedIP);
  llvm::dbgs() << "[RuleGen] Created shared reg_rd instance: reg_rd_" << instructionId << "\n";

  // For top-level function processing, we don't have external FIFOs
  // Token FIFOs and value FIFOs will be created internally by BlockHandler
  Instance *topLevelInputTokenFIFO = nullptr;
  Instance *topLevelOutputTokenFIFO = nullptr;
  // Floating instructions use shared cross-block registers and reg_rd. A
  // block-local busy bit cannot protect those across a loop or conditional.
  // Admit one execution context for the whole function, while the RoCC
  // adapter may still queue commands/responses. Loop iterations can pipeline
  // internally; only independent instruction invocations are serialized.
  bool hasFloat = false;
  funcOp.walk([&](Operation *op) {
    for (Type type : op->getOperandTypes())
      hasFloat |= isa<FloatType>(type);
    for (Type type : op->getResultTypes())
      hasFloat |= isa<FloatType>(type);
  });
  if (hasFloat) {
    std::string prefix = "fp_context_" + std::to_string(instructionId);
    auto *tokenMod = STLLibrary::createFIFO2IModule(1, circuit);
    auto *busyMod = STLLibrary::createRegModule(1, 0, circuit);
    builder.restoreInsertionPoint(savedIP);
    auto *busy = mainModule->addInstance(prefix + "_busy", busyMod,
                                        {mainClk.getValue(), mainRst.getValue()});
    topLevelInputTokenFIFO = mainModule->addInstance(
        prefix + "_entry", tokenMod, {mainClk.getValue(), mainRst.getValue()});
    topLevelOutputTokenFIFO = mainModule->addInstance(
        prefix + "_exit", tokenMod, {mainClk.getValue(), mainRst.getValue()});
    builder.restoreInsertionPoint(savedIP);
    auto *admit = mainModule->addRule(prefix + "_admit");
    admit->guard([&](OpBuilder &b) {
      auto loc = b.getUnknownLoc();
      auto idle = ~Signal(busy->callValue("read", b)[0], &b, loc);
      b.create<circt::cmt2::ReturnOp>(loc, idle.getValue());
    });
    admit->body([&](OpBuilder &b) {
      auto loc = b.getUnknownLoc();
      auto one = UInt::constant(1, 1, b, loc).getValue();
      busy->callMethod("write", {one}, b);
      topLevelInputTokenFIFO->callMethod("enq", {one}, b);
      b.create<circt::cmt2::ReturnOp>(loc);
    });
    auto *release = mainModule->addRule(prefix + "_release");
    release->guard([&](OpBuilder &b) {
      auto loc = b.getUnknownLoc();
      b.create<circt::cmt2::ReturnOp>(loc,
          UInt::constant(1, 1, b, loc).getValue());
    });
    release->body([&](OpBuilder &b) {
      auto loc = b.getUnknownLoc();
      topLevelOutputTokenFIFO->callMethod("deq", {}, b);
      busy->callMethod("write", {UInt::constant(0, 1, b, loc).getValue()}, b);
      b.create<circt::cmt2::ReturnOp>(loc);
    });
    builder.restoreInsertionPoint(savedIP);
  }
  llvm::DenseMap<Value, Instance*> topLevelInputFIFOs;  // Empty for top level
  llvm::DenseMap<Value, llvm::SmallVector<std::pair<BlockInfo*, Instance*>, 4>> topLevelOutputFIFOs; // Empty for top level

  // Create BlockHandler to manage all blocks with FIFO coordination
  // BlockHandler will internally delegate to specialized handlers (LoopHandler, BBHandler) as needed
  BlockHandler blockHandler(this, mainModule, funcOp, poolInstance,
                           roccInstance, hellaMemInstance, dmaItfc, csrItfc,
                           circuit, mainClk, mainRst, instructionId, regRdInstance,
                           topLevelInputTokenFIFO, topLevelOutputTokenFIFO,
                           topLevelInputFIFOs, topLevelOutputFIFOs);

  // Process all blocks - BlockHandler will handle loops, basic blocks, conditionals, etc.
  if (failed(blockHandler.processFunctionAsBlocks())) {
    funcOp.emitError("failed to process blocks for rule generation");
    signalPassFailure();
    return;
  }

  llvm::dbgs() << "[BlockHandler] Successfully processed all blocks for function "
               << funcOp.getName() << "\n";
}

} // namespace mlir
