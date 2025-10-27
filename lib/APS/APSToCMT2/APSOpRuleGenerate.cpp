//===- RuleGeneration.cpp - Rule Generation for TOR Functions -------------===//
//
// This file implements the rule generation functionality for TOR functions
// that was previously in APSToCMT2GenPass.cpp
//
// REFACTORED: Now uses BBHandler for object-oriented basic block management
//
//===----------------------------------------------------------------------===//

#include "APS/APSToCMT2.h"
#include "APS/BBHandler.h"

namespace mlir {

using namespace mlir;
using namespace mlir::tor;
using namespace circt::cmt2::ecmt2;
using namespace circt::cmt2::ecmt2::stl;
using namespace circt::firrtl;

//===----------------------------------------------------------------------===//
// Main Rule Generation Function - Refactored to use BBHandler
//===----------------------------------------------------------------------===//

/// Generate rules for a specific TOR function - proper implementation from
/// rulegenpass.cpp
void APSToCMT2GenPass::generateRulesForFunction(
    Module *mainModule, tor::FuncOp funcOp, Instance *poolInstance,
    Instance *roccInstance, Instance *hellaMemInstance, InterfaceDecl *dmaItfc,
    Circuit &circuit, Clock mainClk, Reset mainRst, unsigned long opcode) {

  // Create BBHandler to manage basic block operations
  BBHandler bbHandler(this, mainModule, funcOp, poolInstance, roccInstance,
                      hellaMemInstance, dmaItfc, circuit, mainClk, mainRst,
                      opcode);

  // Process all basic blocks using the new OO approach
  if (failed(bbHandler.processBasicBlocks())) {
    funcOp.emitError("failed to process basic blocks for rule generation");
    return;
  }

  llvm::outs()
      << "[BBHandler] Successfully processed basic blocks for function "
      << funcOp.getName() << "\n";
}

} // namespace mlir