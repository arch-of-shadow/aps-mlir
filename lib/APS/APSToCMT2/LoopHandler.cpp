//===- LoopHandler.cpp - Canonical Loop Handler with Block Coordination ---===//
//
// This file implements the canonical loop handler using Signal EDSL for
// clean hardware generation with proper token flow coordination
//
//===----------------------------------------------------------------------===//

#include "APS/LoopHandler.h"
#include "APS/APSOps.h"
#include "APS/BBHandler.h"
#include "APS/BlockHandler.h"
#include "circt/Dialect/Cmt2/ECMT2/SignalHelpers.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {

using namespace mlir;
using namespace mlir::tor;
using namespace circt::cmt2::ecmt2;
using namespace circt::cmt2::ecmt2::stl;
using namespace circt::firrtl;

//===----------------------------------------------------------------------===//
// Canonical LoopHandler Implementation with Signal EDSL
//===----------------------------------------------------------------------===//

LoopHandler::LoopHandler(APSToCMT2Pass *pass, Module *mainModule,
                         tor::FuncOp funcOp, Instance *poolInstance,
                         Instance *roccInstance, Instance *hellaMemInstance,
                         InterfaceDecl *dmaItfc, InterfaceDecl *csrItfc,
                         Circuit &circuit,
                         Clock mainClk, Reset mainRst, unsigned long instructionId, Instance *regRdInstance,
                         Instance *input_token_fifo,
                         Instance *output_token_fifo,
                         llvm::DenseMap<Value, Instance *> &input_fifos,
                         llvm::DenseMap<Value, llvm::SmallVector<std::pair<BlockInfo*, Instance*>, 4>> &output_fifos,
                         const std::string &namePrefix)
    : BlockHandler(pass, mainModule, funcOp, poolInstance, roccInstance,
                   hellaMemInstance, dmaItfc, csrItfc, circuit, mainClk,
                   mainRst, instructionId, regRdInstance,
                   input_token_fifo, output_token_fifo, input_fifos,
                   output_fifos, namePrefix) {}

LogicalResult LoopHandler::processLoopBlock(BlockInfo &loopBlock) {
  // Process a single loop block following Blockgen.md canonical pattern
  // entry → body → next with proper token coordination

  loop.scopeResources.clear();
  inductionVarReg = nullptr;
  inductionVarFIFO = nullptr;

  // 1. Extract the tor.for operation from this block segment
  tor::ForOp forOp = nullptr;
  for (Operation *op : loopBlock.operations) {
    if (auto candidate = dyn_cast<tor::ForOp>(op)) {
      forOp = candidate;
      break;
    }
  }

  if (!forOp) {
    llvm::dbgs() << "[LoopHandler] ERROR: Loop block segment contains " << loopBlock.operations.size() << " operations:\n";
    for (Operation *op : loopBlock.operations) {
      llvm::dbgs() << "  - " << op->getName() << "\n";
    }
    llvm::report_fatal_error("Loop block segment does not contain tor.for operation");
  }

  // 2. Initialize the single loop with the enclosing compact block name.
  std::string loop_name = namePrefix;
  if (!loop_name.empty() && loop_name.back() == '_')
    loop_name.pop_back();
  loop.initialize(forOp, loop_name);
  loop.isPipeline = hasPipelineAttr(forOp);
  loop.context_token_reg = nullptr;

  // 3. Extract loop control information
  loop.inductionVar = forOp.getInductionVar();
  loop.lowerBound = forOp.getLowerBound();
  loop.upperBound = forOp.getUpperBound();
  loop.step = forOp.getStep();

  // Extract iter_args if present
  for (Value iterArg : forOp.getRegionIterArgs()) {
    loop.iterArgs.push_back(iterArg);
    loop.iterArgTypes.push_back(iterArg.getType());
  }

  if (!loop.iterArgs.empty() || forOp->getNumResults() != 0) {
    return forOp.emitError(
        "loop with iter_args/results is not supported by APSToCMT2 loop "
        "lowering yet");
  }

  // 4. Create simplified loop infrastructure.
  if (failed(createLoopInfrastructure(loopBlock)))
    return failure();

  if (outputTokenFIFO)
    loop.scopeResources.boundary.exitTokenFIFO = outputTokenFIFO;

  if (loop.isPipeline) {
    if (!loop.scopeResources.isValidForPipeline())
      llvm::report_fatal_error("pipeline loop scope resources are incomplete");
  } else if (!loop.scopeResources.isValidForNonPipeline()) {
    llvm::report_fatal_error("non-pipeline loop scope resources are incomplete");
  }

  // 5. Process loop body operations using BBHandler with token coordination
  if (failed(processLoopBodyOperations(forOp, loopBlock)))
    return failure();

  // 6. Generate loop control rules.
  if (loop.isPipeline) {
    if (failed(generatePipelineLoopRules(loopBlock)))
      return failure();
  } else {
    if (failed(generateCanonicalLoopRules(loopBlock)))
      return failure();
  }

  return success();
}

LogicalResult LoopHandler::processBlock(BlockInfo &block) {
  llvm::dbgs() << "[LoopHandler] Processing block " << block.blockId
               << " as loop block\n";

  // Delegate to processLoopBlock for loop-specific processing
  return processLoopBlock(block);
}

LogicalResult LoopHandler::generateCanonicalLoopRules(BlockInfo &loopBlock) {
  // Generate exactly 2 rules per Blockgen.md: entry rule and next rule
  llvm::dbgs() << "[LoopHandler] Generating canonical loop rules (entry + "
                  "next) for loop "
               << loop.loopName << "\n";

  // Create entry rule - handles loop initialization and first iteration
  if (failed(generateLoopEntryRule(loopBlock)))
    return failure();

  // Create next rule - handles loop iteration and termination
  if (failed(generateLoopNextRule(loopBlock)))
    return failure();

  return success();
}

LogicalResult LoopHandler::generatePipelineLoopRules(BlockInfo &loopBlock) {
  llvm::dbgs() << "[LoopHandler] Generating pipeline loop rules (entry + "
                  "issue + retire) for loop "
               << loop.loopName << "\n";

  if (failed(generatePipelineLoopEntryRule(loopBlock)))
    return failure();
  if (failed(generatePipelineLoopIssueRule(loopBlock)))
    return failure();
  if (failed(generatePipelineLoopRetireRule(loopBlock)))
    return failure();

  return success();
}

LogicalResult LoopHandler::generateLoopEntryRule(BlockInfo &loopBlock) {
  // Per Blockgen.md: Create entry rule that handles loop initialization
  // Use loop name as distinguisher, not loop ID
  auto *rule = mainModule->addRule(loop.loopName + "_entry_rule");

  rule->guard([&](mlir::OpBuilder &b) {
    auto loc = b.getUnknownLoc();
    if (loop.scopeResources.contextTokenReg) {
      auto tokenValues = loop.scopeResources.contextTokenReg->callValue("read", b);
      if (tokenValues.empty())
        llvm::report_fatal_error("LoopHandler: context token read returned no value");
      Signal token(tokenValues[0], &b, loc);
      auto one = UInt::constant(1, 1, b, loc);
      b.create<circt::cmt2::ReturnOp>(loc, (token == one).getValue());
      return;
    }
    auto alwaysTrue = UInt::constant(1, 1, b, loc);
    b.create<circt::cmt2::ReturnOp>(loc, alwaysTrue.getValue());
  });

  rule->body([&](mlir::OpBuilder &b) {
    auto loc = b.getUnknownLoc();

    llvm::dbgs() << "[LoopHandler] Generating entry rule for loop "
                 << loop.loopName << "\n";

    if (loop.scopeResources.contextTokenReg) {
      auto unavailable = UInt::constant(0, 1, b, loc);
      loop.scopeResources.contextTokenReg->callMethod(
          "write", {unavailable.getValue()}, b);
      llvm::dbgs() << "[LoopHandler] Acquired loop context token\n";
    }

    // 1. Dequeue token from previous block (token input fifo)
    if (loop.scopeResources.boundary.entryTokenFIFO) {
      auto prevToken = loop.scopeResources.boundary.entryTokenFIFO->callMethod("deq", {}, b);
      llvm::dbgs() << "[LoopHandler] Dequeued token from previous block\n";
    } else {
      llvm::dbgs() << "[LoopHandler] No input token FIFO (top-level loop)\n";
    }

    llvm::SmallVector<std::pair<Instance *, mlir::Value>, 4>
        pendingLoopToBodyValues;

    // 2. Handle parent live-ins from direct input FIFOs.
    // IMPORTANT: Only dequeue values that are actually used by this loop.
    for (auto &[value, fifo] : input_fifos) {
      if (!fifo)
        continue;

      // Skip constants - they don't need to be dequeued
      if (value.getDefiningOp<arith::ConstantOp>()) {
        llvm::dbgs() << "[LoopHandler] Skipping constant value in entry rule\n";
        continue;
      }

      // Check if this value has a corresponding loop-to-body FIFO or state register
      // If not, it means this value is NOT used by the loop - skip dequeuing
      bool hasLoopToBodyFifo = loop.loop_to_body_fifos.count(value) > 0;
      bool hasStateRegister = loop.input_state_registers.count(value) > 0;

      if (!hasLoopToBodyFifo && !hasStateRegister) {
        llvm::dbgs() << "[LoopHandler] Skipping input FIFO - value not used by loop\n";
        continue;
      }

      // Dequeue only if value is actually used by the loop
      auto dequeuedValue = fifo->callMethod("deq", {}, b)[0];
      llvm::dbgs() << "[LoopHandler] Dequeued cross-block value from input FIFO\n";

      // Write to state register for persistent storage (used by next rule)
      if (hasStateRegister) {
        Instance *stateReg = loop.input_state_registers[value];
        stateReg->callMethod("write", {dequeuedValue}, b);
        llvm::dbgs() << "[LoopHandler] Wrote dequeued value to state register\n";
      }

      // Enqueue to loop-to-body FIFO for first iteration
      if (hasLoopToBodyFifo) {
        Instance *loopToBodyFifo = loop.loop_to_body_fifos[value];
        pendingLoopToBodyValues.push_back({loopToBodyFifo, dequeuedValue});
      }
    }

    // 3. Handle parent live-ins captured by an enclosing block register.
    for (auto &[value, reg] : loopBlock.scopeResources.inputValueRegs) {
      if (!reg)
        continue;

      if (value.getDefiningOp<arith::ConstantOp>()) {
        llvm::dbgs() << "[LoopHandler] Skipping constant value in captured input regs\n";
        continue;
      }

      bool hasLoopToBodyFifo = loop.loop_to_body_fifos.count(value) > 0;
      bool hasStateRegister = loop.input_state_registers.count(value) > 0;

      if (!hasLoopToBodyFifo && !hasStateRegister) {
        llvm::dbgs() << "[LoopHandler] Skipping captured input reg - value not used by loop\n";
        continue;
      }

      auto capturedValue = reg->callValue("read", b);
      if (capturedValue.empty())
        continue;

      if (hasStateRegister) {
        Instance *stateReg = loop.input_state_registers[value];
        stateReg->callMethod("write", {capturedValue[0]}, b);
        llvm::dbgs() << "[LoopHandler] Wrote captured input value to state register\n";
      }

      if (hasLoopToBodyFifo) {
        Instance *loopToBodyFifo = loop.loop_to_body_fifos[value];
        pendingLoopToBodyValues.push_back({loopToBodyFifo, capturedValue[0]});
      }
    }

    // 4. Initialize loop state in loop carry fifo
    // Pack state: [counter][bound][step][iter_args...]
    // Convert all values to FIRRTL types before creating Signals
    auto convertToFIRRTL = [&](mlir::Value val) -> mlir::Value {
      if (isa<circt::firrtl::FIRRTLBaseType>(val.getType())) {
        llvm::report_fatal_error("LoopHandler: loop boundary is not a constant!");
      }
      if (auto constOp = val.getDefiningOp<arith::ConstantOp>()) {
        auto intAttr = mlir::cast<IntegerAttr>(constOp.getValueAttr());
        unsigned width = mlir::cast<IntegerType>(intAttr.getType()).getWidth();
        return UInt::constant(intAttr.getValue().getZExtValue(), width, b, loc).getValue();
      }
      llvm::report_fatal_error("LoopHandler: Cannot convert non-constant MLIR type to FIRRTL");
    };

    // high: lowerBound (start), medium: upperBound (stop), low: step
    // TODO: we only handle increment bound here...
    Signal loopState(convertToFIRRTL(loop.lowerBound), &b, loc);
    loopState = loopState.cat(Signal(convertToFIRRTL(loop.upperBound), &b, loc));
    loopState = loopState.cat(Signal(convertToFIRRTL(loop.step), &b, loc));

    // TODO: ignore for now..
    // for (unsigned i = 0; i < loop.iterArgs.size(); i++) {
    //   loopState = loopState.cat(Signal(convertToFIRRTL(loop.iterArgs[i]), &b, loc));
    // }

    auto *loopStateReg = loop.scopeResources.loopStateReg ? loop.scopeResources.loopStateReg : loop.loop_state_reg;
    loopStateReg->callMethod("write", {loopState.getValue()}, b);
    llvm::dbgs() << "[LoopHandler] Initialized loop state register\n";

    Signal lowerSig(convertToFIRRTL(loop.lowerBound), &b, loc);
    Signal upperSig(convertToFIRRTL(loop.upperBound), &b, loc);
    auto shouldEnter = lowerSig <= upperSig;

    If(
        shouldEnter,
        [&](mlir::OpBuilder &b) {
          for (auto &[loopToBodyFifo, liveInValue] : pendingLoopToBodyValues) {
            loopToBodyFifo->callMethod("enq", {liveInValue}, b);
            llvm::dbgs() << "[LoopHandler] Enqueued live-in value to "
                            "loop-to-body FIFO for first iteration\n";
          }

          // Extract and publish loop variables for the loop body.
          if (loop.inductionVar && inductionVarReg) {
            Signal stateSig(loopState.getValue(), &b, loc);
            auto inductionVar = stateSig.bits(95, 64);
            inductionVarReg->callMethod("write", {inductionVar.getValue()}, b);
            llvm::dbgs() << "[LoopHandler] Wrote induction variable register\n";
          }

          auto startToken = UInt::constant(1, 1, b, loc);
          if (loop.scopeResources.bodyAdmitFIFO) {
            loop.scopeResources.bodyAdmitFIFO->callMethod(
                "enq", {startToken.getValue()}, b);
            llvm::dbgs()
                << "[LoopHandler] Signaled loop body to start via token FIFO\n";
          }
        },
        [&](mlir::OpBuilder &b) {
          auto *exitTokenFIFO = loop.scopeResources.boundary.exitTokenFIFO
                                    ? loop.scopeResources.boundary.exitTokenFIFO
                                    : outputTokenFIFO;
          if (exitTokenFIFO) {
            auto exitToken = UInt::constant(1, 1, b, loc);
            exitTokenFIFO->callMethod("enq", {exitToken.getValue()}, b);
          }
          if (loop.scopeResources.contextTokenReg) {
            auto available = UInt::constant(1, 1, b, loc);
            loop.scopeResources.contextTokenReg->callMethod(
                "write", {available.getValue()}, b);
          }
          emitLoopExitValues(b, loc);
        },
        b, loc);

    b.create<circt::cmt2::ReturnOp>(loc);
  });

  rule->finalize();
  llvm::dbgs() << "[LoopHandler] Generated entry rule for loop "
               << loop.loopName << "\n";
  return success();
}

LogicalResult LoopHandler::generateLoopNextRule(BlockInfo &loopBlock) {
  auto *rule = mainModule->addRule(loop.loopName + "_next_rule");

  rule->guard([&](mlir::OpBuilder &b) {
    auto loc = b.getUnknownLoc();
    // Always return 1'b1 - coordination handled by FIFO availability
    auto alwaysReady = UInt::constant(1, 1, b, loc);
    b.create<circt::cmt2::ReturnOp>(loc, alwaysReady.getValue());
  });

  rule->body([&](mlir::OpBuilder &b) {
    auto loc = b.getUnknownLoc();

    // Canonical next rule: handle iteration logic and loop termination
    llvm::dbgs()
        << "[LoopHandler] Generating canonical next rule body for loop "
        << loop.loopName << "\n";

    // 1. Dequeue body completion token from body_to_next fifo
    if (loop.scopeResources.bodyDoneTokenFIFO) {
      auto bodyCompleteToken =
          loop.scopeResources.bodyDoneTokenFIFO->callMethod("deq", {}, b);
      llvm::dbgs() << "[LoopHandler] Dequeued body completion token\n";
    }

    // 2. Read loop state from the single non-pipeline loop context.
    auto *loopStateReg = loop.scopeResources.loopStateReg ? loop.scopeResources.loopStateReg : loop.loop_state_reg;
    if (loopStateReg) {
      auto loopStateValues = loopStateReg->callValue("read", b);
      if (loopStateValues.empty()) {
        llvm::report_fatal_error("LoopHandler: loop state register read returned no value");
      }
      auto loopState = loopStateValues[0];

      // Extract state components using Signal operations
      // [counter:32][bound:32][step:32][iter_arg0...]
      Signal stateSig(loopState, &b, loc);

      auto currentCounter = stateSig.bits(95, 64); // Extract counter (bits 31:0)
      auto upperBound = stateSig.bits(63, 32);    // Extract bound (bits 63:32)
      auto step = stateSig.bits(31, 0);          // Extract step (bits 95:64)

      // Extract iter_args
      llvm::SmallVector<mlir::Value> iterArgs;
      unsigned stateOffset = 96;
      for (unsigned i = 0; i < loop.iterArgs.size(); i++) {
        unsigned width = getBitWidth(loop.iterArgTypes[i]);
        unsigned highBit = stateOffset + width - 1;
        auto iterArg = stateSig.bits(highBit, stateOffset);
        iterArgs.push_back(iterArg.getValue());
        stateOffset += width;
      }

      // 4. Canonical loop decision:
      // If shouldContinue: increment counter and continue loop
      // If not shouldContinue: exit loop and pass control to next block
      auto nextCounter = currentCounter + step;

      // 3. Check if loop should continue: counter <= upper_bound
      auto shouldContinue = nextCounter <= upperBound;
      llvm::dbgs()
          << "[LoopHandler] Next rule: checking if counter < upper_bound\n";

      // 5. Update loop state and either continue or exit using ECMT2 If
      // construct for proper signal-based conditional execution
      If(
          shouldContinue,
          // Then branch: Continue looping
          [&](mlir::OpBuilder &b) {
            llvm::dbgs() << "[LoopHandler] Next rule: continuing loop, counter "
                            "updated\n";

            // Pack updated state: [nextCounter][upperBound][step][iterArgs...]
            Signal updatedState(nextCounter.getValue(), &b, loc);
            updatedState = updatedState.cat(upperBound);
            updatedState = updatedState.cat(step);

            // Add updated iter_args (for now, just pass through)
            for (auto iterArg : iterArgs) {
              updatedState = updatedState.cat(Signal(iterArg, &b, loc));
            }

            // Write updated state for next iteration.
            loopStateReg->callMethod("write", {updatedState.bits(stateOffset - 1, 0).getValue()}, b);

            // Re-issue every loop live-in payload for the next iteration.
            for (auto &[value, loopToBodyFifo] : loop.loop_to_body_fifos) {
              Instance *stateReg = loop.input_state_registers.lookup(value);
              if (!stateReg || !loopToBodyFifo)
                continue;
              auto storedValue = stateReg->callValue("read", b);
              if (storedValue.empty())
                continue;
              loopToBodyFifo->callMethod("enq", {storedValue[0]}, b);
              llvm::dbgs() << "[LoopHandler] Next rule: re-issued live-in "
                              "value to loop-to-body FIFO\n";
            }

            if (loop.inductionVar && inductionVarReg) {
              auto inductionVar = nextCounter.bits(31, 0);
              inductionVarReg->callMethod("write", {inductionVar.getValue()}, b);
              llvm::dbgs() << "[LoopHandler] Wrote induction variable register\n";
            }

            // Signal next iteration via token FIFO coordination
            if (loop.scopeResources.bodyAdmitFIFO) {
              auto continueToken = UInt::constant(1, 1, b, loc);
              loop.scopeResources.bodyAdmitFIFO->callMethod("enq", {continueToken.getValue()}, b);
              llvm::dbgs() << "[LoopHandler] Next rule: signaling next "
                              "iteration via token FIFO\n";
            }
          },
          // Else branch: Loop complete
          [&](mlir::OpBuilder &b) {
            llvm::dbgs()
                << "[LoopHandler] Next rule: loop complete, signaling exit\n";

            // Signal loop completion directly to next block via output token FIFO
            // (no intermediate next_to_exit FIFO needed)
            auto *exitTokenFIFO = loop.scopeResources.boundary.exitTokenFIFO ? loop.scopeResources.boundary.exitTokenFIFO
                                                                             : outputTokenFIFO;
            if (exitTokenFIFO) {
              auto outputExitToken = UInt::constant(1, 1, b, loc);
              exitTokenFIFO->callMethod("enq", {outputExitToken.getValue()}, b);
              llvm::dbgs() << "[LoopHandler] Next rule: enqueued output token to next block\n";
            } else {
              llvm::dbgs() << "[LoopHandler] No output token FIFO (top-level loop exit)\n";
            }

            if (loop.scopeResources.contextTokenReg) {
              auto available = UInt::constant(1, 1, b, loc);
              loop.scopeResources.contextTokenReg->callMethod(
                  "write", {available.getValue()}, b);
              llvm::dbgs() << "[LoopHandler] Released loop context token\n";
            }

            emitLoopExitValues(b, loc);
          },
          b, loc);
    } 
    
    b.create<circt::cmt2::ReturnOp>(loc);

  });

  rule->finalize();

  llvm::dbgs() << "[LoopHandler] Generated canonical loop next rule for loop "
              << loop.loopName << "\n";
  return success();
}

LogicalResult LoopHandler::generatePipelineLoopEntryRule(BlockInfo &loopBlock) {
  auto *rule = mainModule->addRule(loop.loopName + "_entry_rule");

  rule->guard([&](mlir::OpBuilder &b) {
    auto loc = b.getUnknownLoc();
    if (loop.scopeResources.contextTokenReg) {
      auto tokenValues = loop.scopeResources.contextTokenReg->callValue("read", b);
      if (tokenValues.empty())
        llvm::report_fatal_error("LoopHandler: context token read returned no value");
      Signal token(tokenValues[0], &b, loc);
      auto one = UInt::constant(1, 1, b, loc);
      b.create<circt::cmt2::ReturnOp>(loc, (token == one).getValue());
      return;
    }
    auto alwaysTrue = UInt::constant(1, 1, b, loc);
    b.create<circt::cmt2::ReturnOp>(loc, alwaysTrue.getValue());
  });

  rule->body([&](mlir::OpBuilder &b) {
    auto loc = b.getUnknownLoc();

    if (loop.scopeResources.contextTokenReg) {
      auto unavailable = UInt::constant(0, 1, b, loc);
      loop.scopeResources.contextTokenReg->callMethod(
          "write", {unavailable.getValue()}, b);
    }

    if (loop.scopeResources.boundary.entryTokenFIFO)
      loop.scopeResources.boundary.entryTokenFIFO->callMethod("deq", {}, b);

    for (auto &[value, fifo] : input_fifos) {
      if (!fifo || value.getDefiningOp<arith::ConstantOp>())
        continue;
      if (!loop.input_state_registers.count(value))
        continue;
      auto dequeuedValue = fifo->callMethod("deq", {}, b);
      if (!dequeuedValue.empty())
        loop.input_state_registers[value]->callMethod("write",
                                                      {dequeuedValue[0]}, b);
    }

    for (auto &[value, reg] : loopBlock.scopeResources.inputValueRegs) {
      if (!reg || value.getDefiningOp<arith::ConstantOp>())
        continue;
      if (!loop.input_state_registers.count(value))
        continue;
      auto capturedValue = reg->callValue("read", b);
      if (!capturedValue.empty())
        loop.input_state_registers[value]->callMethod("write",
                                                      {capturedValue[0]}, b);
    }

    auto convertToFIRRTL = [&](mlir::Value val) -> mlir::Value {
      if (isa<circt::firrtl::FIRRTLBaseType>(val.getType()))
        llvm::report_fatal_error(
            "LoopHandler: loop boundary is not a constant!");
      if (auto constOp = val.getDefiningOp<arith::ConstantOp>()) {
        auto intAttr = mlir::cast<IntegerAttr>(constOp.getValueAttr());
        unsigned width = mlir::cast<IntegerType>(intAttr.getType()).getWidth();
        return UInt::constant(intAttr.getValue().getZExtValue(), width, b, loc)
            .getValue();
      }
      llvm::report_fatal_error(
          "LoopHandler: Cannot convert non-constant MLIR type to FIRRTL");
    };

    Signal loopState(convertToFIRRTL(loop.lowerBound), &b, loc);
    loopState = loopState.cat(Signal(convertToFIRRTL(loop.upperBound), &b, loc));
    loopState = loopState.cat(Signal(convertToFIRRTL(loop.step), &b, loc));
    loop.scopeResources.loopStateReg->callMethod("write",
                                                 {loopState.getValue()}, b);

    loop.scopeResources.doneCounterReg->callMethod(
        "write", {convertToFIRRTL(loop.lowerBound)}, b);

    auto token = UInt::constant(1, 1, b, loc);
    loop.scopeResources.issueTokenFIFO->callMethod("enq", {token.getValue()},
                                                   b);

    b.create<circt::cmt2::ReturnOp>(loc);
  });

  rule->finalize();
  return success();
}

LogicalResult LoopHandler::generatePipelineLoopIssueRule(BlockInfo &loopBlock) {
  auto *rule = mainModule->addRule(loop.loopName + "_issue_rule");

  rule->guard([](mlir::OpBuilder &b) {
    auto loc = b.getUnknownLoc();
    auto alwaysTrue = UInt::constant(1, 1, b, loc);
    b.create<circt::cmt2::ReturnOp>(loc, alwaysTrue.getValue());
  });

  rule->body([&](mlir::OpBuilder &b) {
    auto loc = b.getUnknownLoc();

    loop.scopeResources.issueTokenFIFO->callMethod("deq", {}, b);

    auto stateValues = loop.scopeResources.loopStateReg->callValue("read", b);
    if (stateValues.empty())
      llvm::report_fatal_error(
          "LoopHandler: pipeline issue state register read returned no value");

    Signal stateSig(stateValues[0], &b, loc);
    auto currentCounter = stateSig.bits(95, 64);
    auto upperBound = stateSig.bits(63, 32);
    auto step = stateSig.bits(31, 0);
    auto shouldIssue = currentCounter <= upperBound;

    If(
        shouldIssue,
        [&](mlir::OpBuilder &b) {
          if (inductionVarFIFO)
            inductionVarFIFO->callMethod("enq",
                                         {currentCounter.bits(31, 0).getValue()},
                                         b);

          auto bodyToken = UInt::constant(1, 1, b, loc);
          loop.scopeResources.bodyAdmitFIFO->callMethod(
              "enq", {bodyToken.getValue()}, b);

          auto nextCounter = currentCounter + step;
          Signal updatedState(nextCounter.bits(31, 0).getValue(), &b, loc);
          updatedState = updatedState.cat(upperBound);
          updatedState = updatedState.cat(step);
          loop.scopeResources.loopStateReg->callMethod(
              "write", {updatedState.getValue()}, b);

          auto shouldReissue = nextCounter <= upperBound;
          If(
              shouldReissue,
              [&](mlir::OpBuilder &b) {
                auto issueToken = UInt::constant(1, 1, b, loc);
                loop.scopeResources.issueTokenFIFO->callMethod(
                    "enq", {issueToken.getValue()}, b);
              },
              [&](mlir::OpBuilder &b) {}, b, loc);
        },
        [&](mlir::OpBuilder &b) {
          auto *exitTokenFIFO = loop.scopeResources.boundary.exitTokenFIFO
                                    ? loop.scopeResources.boundary.exitTokenFIFO
                                    : outputTokenFIFO;
          if (exitTokenFIFO) {
            auto exitToken = UInt::constant(1, 1, b, loc);
            exitTokenFIFO->callMethod("enq", {exitToken.getValue()}, b);
          }
          if (loop.scopeResources.contextTokenReg) {
            auto available = UInt::constant(1, 1, b, loc);
            loop.scopeResources.contextTokenReg->callMethod(
                "write", {available.getValue()}, b);
          }
          emitLoopExitValues(b, loc);
        },
        b, loc);

    b.create<circt::cmt2::ReturnOp>(loc);
  });

  rule->finalize();
  return success();
}

LogicalResult LoopHandler::generatePipelineLoopRetireRule(BlockInfo &loopBlock) {
  auto *rule = mainModule->addRule(loop.loopName + "_retire_rule");

  rule->guard([](mlir::OpBuilder &b) {
    auto loc = b.getUnknownLoc();
    auto alwaysTrue = UInt::constant(1, 1, b, loc);
    b.create<circt::cmt2::ReturnOp>(loc, alwaysTrue.getValue());
  });

  rule->body([&](mlir::OpBuilder &b) {
    auto loc = b.getUnknownLoc();

    loop.scopeResources.bodyDoneTokenFIFO->callMethod("deq", {}, b);

    auto stateValues = loop.scopeResources.loopStateReg->callValue("read", b);
    auto doneValues = loop.scopeResources.doneCounterReg->callValue("read", b);
    if (stateValues.empty() || doneValues.empty())
      llvm::report_fatal_error(
          "LoopHandler: pipeline retire register read returned no value");

    Signal stateSig(stateValues[0], &b, loc);
    auto upperBound = stateSig.bits(63, 32);
    auto step = stateSig.bits(31, 0);

    Signal doneSig(doneValues[0], &b, loc);
    auto nextDone = doneSig + step;
    loop.scopeResources.doneCounterReg->callMethod(
        "write", {nextDone.bits(31, 0).getValue()}, b);

    auto shouldHaveMoreCompletions = nextDone <= upperBound;
    If(
        shouldHaveMoreCompletions,
        [&](mlir::OpBuilder &b) {},
        [&](mlir::OpBuilder &b) {
          auto *exitTokenFIFO = loop.scopeResources.boundary.exitTokenFIFO
                                    ? loop.scopeResources.boundary.exitTokenFIFO
                                    : outputTokenFIFO;
          if (exitTokenFIFO) {
            auto exitToken = UInt::constant(1, 1, b, loc);
            exitTokenFIFO->callMethod("enq", {exitToken.getValue()}, b);
          }
          if (loop.scopeResources.contextTokenReg) {
            auto available = UInt::constant(1, 1, b, loc);
            loop.scopeResources.contextTokenReg->callMethod(
                "write", {available.getValue()}, b);
          }
          emitLoopExitValues(b, loc);
        },
        b, loc);

    b.create<circt::cmt2::ReturnOp>(loc);
  });

  rule->finalize();
  return success();
}

void LoopHandler::emitLoopExitValues(mlir::OpBuilder &b, mlir::Location loc) {
  (void)loc;
  for (auto &[value, consumers] : output_fifos) {
    Instance *stateReg = loop.input_state_registers.lookup(value);
    if (!stateReg)
      continue;

    auto storedValue = stateReg->callValue("read", b);
    if (storedValue.empty())
      continue;

    for (const auto &[consumerBlock, outFIFO] : consumers) {
      (void)consumerBlock;
      if (outFIFO)
        outFIFO->callMethod("enq", {storedValue[0]}, b);
    }
  }
}

LogicalResult LoopHandler::createLoopInfrastructure(BlockInfo &loopBlock) {
  llvm::dbgs() << "[LoopHandler] Creating loop infrastructure for loop "
               << loop.loopName << "\n";

  auto &builder = mainModule->getBuilder();
  auto savedIP = builder.saveInsertionPoint();

  // Create token FIFOs for canonical loop coordination (entry → body → next)
  // Entry -> Body: signals that loop body can start
  auto *entryToBodyMod = STLLibrary::createFIFO2IModule(1, circuit);
  builder.restoreInsertionPoint(savedIP);
  std::string entryToBodyName = loop.loopName + "_entok";
  loop.token_fifos.to_body =
      mainModule->addInstance(entryToBodyName, entryToBodyMod,
                              {mainClk.getValue(), mainRst.getValue()});
  loop.scopeResources.boundary.entryTokenFIFO = inputTokenFIFO;
  loop.scopeResources.bodyAdmitFIFO = loop.token_fifos.to_body;
  llvm::dbgs() << "[LoopHandler] Created entry-to-body token FIFO: "
               << entryToBodyName << "\n";

  // Body -> Next: signals that body execution is complete
  auto *bodyToNextMod = STLLibrary::createFIFO2IModule(1, circuit);
  builder.restoreInsertionPoint(savedIP);
  std::string bodyToNextName = loop.loopName + "_dntok";
  loop.token_fifos.body_to_next = mainModule->addInstance(
      bodyToNextName, bodyToNextMod, {mainClk.getValue(), mainRst.getValue()});
  loop.scopeResources.bodyDoneTokenFIFO = loop.token_fifos.body_to_next;
  loop.scopeResources.loopFrameToNextFIFO = loop.token_fifos.body_to_next;
  llvm::dbgs() << "[LoopHandler] Created body-to-next token FIFO: "
               << bodyToNextName << "\n";

  // Note: next_to_exit is NOT created - we use the loop block's outputTokenFIFO instead
  // This connects the loop's exit directly to the next block
  loop.token_fifos.next_to_exit = nullptr;

  // Create single loop state register for the one non-pipeline loop context.
  // Calculate total bit width: counter(32) + bound(32) + step(32) + iter_args
  unsigned stateWidth = 32 + 32 + 32; // counter + bound + step
  for (unsigned i = 0; i < loop.iterArgs.size(); i++) {
    stateWidth += getBitWidth(loop.iterArgTypes[i]);
  }

  auto *stateMod = STLLibrary::createRegModule(stateWidth, 0, circuit);
  builder.restoreInsertionPoint(savedIP);
  std::string stateName = loop.loopName + "_st";
  loop.loop_state_reg = mainModule->addInstance(
      stateName, stateMod, {mainClk.getValue(), mainRst.getValue()});
  loop.scopeResources.loopStateReg = loop.loop_state_reg;
  llvm::dbgs() << "[LoopHandler] Created loop state register: " << stateName
               << " (width=" << stateWidth << ")\n";

  if (loop.isPipeline) {
    auto *issueTokMod = STLLibrary::createFIFO2IModule(1, circuit);
    builder.restoreInsertionPoint(savedIP);
    std::string issueTokName = loop.loopName + "_istok";
    loop.pipeline_issue_token_fifo = mainModule->addInstance(
        issueTokName, issueTokMod, {mainClk.getValue(), mainRst.getValue()});
    loop.scopeResources.issueTokenFIFO = loop.pipeline_issue_token_fifo;

    auto *doneRegMod = STLLibrary::createRegModule(32, 0, circuit);
    builder.restoreInsertionPoint(savedIP);
    std::string doneRegName = loop.loopName + "_dn";
    loop.pipeline_done_reg = mainModule->addInstance(
        doneRegName, doneRegMod, {mainClk.getValue(), mainRst.getValue()});
    loop.scopeResources.doneCounterReg = loop.pipeline_done_reg;
  }

  if (requireContextToken) {
    auto *contextTokenMod = STLLibrary::createRegModule(1, 1, circuit);
    builder.restoreInsertionPoint(savedIP);
    std::string contextTokenName = loop.loopName + "_ctok";
    loop.context_token_reg = mainModule->addInstance(
        contextTokenName, contextTokenMod,
        {mainClk.getValue(), mainRst.getValue()});
    loop.scopeResources.contextTokenReg = loop.context_token_reg;
    llvm::dbgs() << "[LoopHandler] Created loop context token register: "
                 << contextTokenName << "\n";
  }

  // Create register for induction variable if the body reads it.
  // Only create it if the induction variable is actually used in the loop body
  if (loop.inductionVar) {
    // Get the loop body to check if induction variable is used
    Block *loopBody = loop.forOp ? loop.forOp.getBody() : nullptr;
    if (loopBody && isValueUsedInLoopBody(loop.inductionVar, loopBody)) {
      unsigned ivWidth = getBitWidth(loop.inductionVar.getType());
      std::string indVarName = loop.loopName + "_iv";
      if (loop.isPipeline) {
        auto *indVarMod = STLLibrary::createFIFO2IModule(ivWidth, circuit);
        builder.restoreInsertionPoint(savedIP);
        Instance *indVarInstance = mainModule->addInstance(
            indVarName, indVarMod, {mainClk.getValue(), mainRst.getValue()});
        inductionVarFIFO = indVarInstance;
      } else {
        auto *indVarMod = STLLibrary::createRegModule(ivWidth, 0, circuit);
        builder.restoreInsertionPoint(savedIP);
        Instance *indVarInstance = mainModule->addInstance(
            indVarName, indVarMod, {mainClk.getValue(), mainRst.getValue()});
        inductionVarReg = indVarInstance;
      }
      llvm::dbgs() << "[LoopHandler] Created induction variable "
                   << (loop.isPipeline ? "FIFO: " : "register: ")
                   << indVarName << "\n";
    } else {
      llvm::dbgs() << "[LoopHandler] Skipping induction variable register creation (not used in loop body)\n";
    }
  }

  // Create state registers and loop-to-body FIFOs for parent live-ins that
  // reach the loop through either a direct FIFO or a captured parent register.
  // State registers store values for reuse across iterations. Loop-to-body
  // FIFOs are fed by entry/next and consumed by the body.
  Block *loopBody = loop.forOp.getBody();
  llvm::dbgs() << "[LoopHandler] Creating state registers and loop-to-body FIFOs for input values used in loop body\n";
  auto ensureLoopLiveInStorage = [&](Value value) {
    if (!value)
      return;
    if (loop.input_state_registers.count(value))
      return;
    if (value.getDefiningOp<arith::ConstantOp>()) {
      llvm::dbgs() << "[LoopHandler] Skipping constant value (constants don't need loop live-in storage)\n";
      return;
    }
    if (!isValueUsedInLoopBody(value, loopBody)) {
      llvm::dbgs() << "[LoopHandler] Skipping value (not used in loop body)\n";
      return;
    }

    unsigned bitWidth = getBitWidth(value.getType());

    auto *regMod = STLLibrary::createRegModule(bitWidth, 0, circuit);
    builder.restoreInsertionPoint(savedIP);
    std::string regName = loop.loopName + "_isr" +
                          std::to_string(loop.input_state_registers.size());
    Instance *regInstance = mainModule->addInstance(
        regName, regMod, {mainClk.getValue(), mainRst.getValue()});
    loop.input_state_registers[value] = regInstance;
    loop.scopeResources.inputStateRegs[value] = regInstance;
    llvm::dbgs() << "[LoopHandler] Created state register: " << regName
                 << " (width=" << bitWidth << ")\n";

    if (!loop.isPipeline) {
      auto *loopToBodyFifoMod = STLLibrary::createFIFO2IModule(bitWidth, circuit);
      builder.restoreInsertionPoint(savedIP);
      std::string loopToBodyFifoName = loop.loopName + "_in" +
                                       std::to_string(loop.loop_to_body_fifos.size());
      Instance *loopToBodyFifo = mainModule->addInstance(
          loopToBodyFifoName, loopToBodyFifoMod,
          {mainClk.getValue(), mainRst.getValue()});
      loop.loop_to_body_fifos[value] = loopToBodyFifo;
      loop.scopeResources.loopToBodyFIFOs[value] = loopToBodyFifo;
      llvm::dbgs() << "[LoopHandler] Created loop-to-body FIFO: "
                   << loopToBodyFifoName << " (width=" << bitWidth << ")\n";
    }
  };

  for (auto &[value, fifo] : input_fifos) {
    if (fifo)
      ensureLoopLiveInStorage(value);
  }
  for (auto &[value, reg] : loopBlock.scopeResources.inputValueRegs) {
    if (reg)
      ensureLoopLiveInStorage(value);
  }

  loop.scopeResources.boundary.exitTokenFIFO = outputTokenFIFO;
  loop.scopeResources.loopCarriedRegs = loop.input_state_registers;
  loop.scopeResources.frameLocalRegs = loop.loop_to_body_fifos;

  return success();
}

unsigned LoopHandler::getBitWidth(mlir::Type type) {
  if (auto intType = dyn_cast<mlir::IntegerType>(type)) {
    return intType.getWidth();
  }
  return 32; // Default width
}

bool LoopHandler::isValueUsedInLoopBody(Value value, Block *loopBody) {
  if (!value || !loopBody) return false;

  // Walk through all operations in the loop body
  for (Operation &op : loopBody->getOperations()) {
    // Check if this operation uses the value
    for (Value operand : op.getOperands()) {
      if (operand == value) {
        return true;
      }
    }

    // Recursively check nested operations
    if (op.getNumRegions() > 0) {
      for (Region &region : op.getRegions()) {
        for (Block &block : region.getBlocks()) {
          if (isValueUsedInLoopBody(value, &block)) {
            return true;
          }
        }
      }
    }
  }

  return false;
}

bool LoopHandler::hasPipelineAttr(tor::ForOp forOp) const {
  if (!forOp)
    return false;
  Attribute attr = forOp->getAttr("pipeline");
  if (!attr)
    return false;
  if (auto boolAttr = dyn_cast<BoolAttr>(attr))
    return boolAttr.getValue();
  if (auto intAttr = dyn_cast<IntegerAttr>(attr))
    return intAttr.getValue().getBoolValue();
  if (isa<UnitAttr>(attr))
    return true;
  return false;
}

LogicalResult LoopHandler::processLoopBodyOperations(tor::ForOp forOp, BlockInfo &loopBlock) {
  llvm::dbgs() << "[LoopHandler] Processing loop body operations for loop " << loop.loopName << "\n";

  // Get the loop body block
  Block *loopBody = forOp.getBody();
  if (!loopBody) {
    return forOp.emitError("loop has no body block");
  }

  // Create input fifos that include loop variables for the loop body
  // Use loop-to-body FIFOs instead of external input_fifos to hide cross-block FIFOs from subblocks
  llvm::DenseMap<Value, Instance*> loopBodyInputFIFOs = loop.loop_to_body_fifos;
  if (loop.isPipeline && loop.inductionVar && inductionVarFIFO)
    loopBodyInputFIFOs[loop.inductionVar] = inductionVarFIFO;
  llvm::DenseMap<Value, llvm::SmallVector<std::pair<BlockInfo *, Instance *>, 4>>
      loopBodyOutputFIFOs;

  // Use BlockHandler's processLoopBodyAsBlocks for proper loop body processing
  // This will handle block segmentation, dataflow analysis, and rule generation
  BlockHandler loopBodyHandler(
      pass, mainModule, funcOp, poolInstance, roccInstance,
      hellaMemInstance, dmaItfc, csrItfc, circuit, mainClk, mainRst, instructionId,
      regRdInstance,
      loop.scopeResources.bodyAdmitFIFO ? loop.scopeResources.bodyAdmitFIFO : loop.token_fifos.to_body,
      loop.scopeResources.bodyDoneTokenFIFO ? loop.scopeResources.bodyDoneTokenFIFO : loop.token_fifos.body_to_next,
      loopBodyInputFIFOs,            // Input data FIFOs (including loop variables)
      loopBodyOutputFIFOs,           // Loop-level outputs are emitted once on exit.
      loop.loopName + "_"            // Name prefix for nested blocks
  );
  loopBodyHandler.setPipelineMode(loop.isPipeline);
  if (loop.inductionVar && inductionVarReg) {
    llvm::dbgs() << "[LoopHandler] Adding induction variable register to loop body inputs\n";
    loopBodyHandler.addInputRegister(loop.inductionVar, inductionVarReg);
  }
  if (loop.isPipeline) {
    for (auto &[value, reg] : loop.input_state_registers)
      loopBodyHandler.addInputRegister(value, reg);
  }

  llvm::dbgs() << "[LoopHandler] Processing loop body using BlockHandler::processLoopBodyAsBlocks\n";

  if (failed(loopBodyHandler.processLoopBodyAsBlocks(forOp))) {
    llvm::dbgs() << "[LoopHandler] Failed to process loop body operations\n";
    return failure();
  }

  llvm::dbgs() << "[LoopHandler] Successfully processed loop body operations\n";
  return success();
}

} // namespace mlir
