//===- IfHandler.cpp - tor.if Handler with Scope Tokens -------------------===//
//
// This file implements non-pipeline tor.if lowering using explicit dispatch,
// branch-entry, and join token FIFOs.
//
//===----------------------------------------------------------------------===//

#include "APS/IfHandler.h"
#include "APS/APSOps.h"
#include "circt/Dialect/Cmt2/ECMT2/SignalHelpers.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {

using namespace mlir::tor;
using namespace circt::cmt2::ecmt2;
using namespace circt::cmt2::ecmt2::stl;
using namespace circt::firrtl;

IfHandler::IfHandler(
    APSToCMT2Pass *pass, Module *mainModule, tor::FuncOp funcOp,
    Instance *poolInstance, Instance *roccInstance, Instance *hellaMemInstance,
    InterfaceDecl *dmaItfc, InterfaceDecl *csrItfc, Circuit &circuit,
    Clock mainClk, Reset mainRst, unsigned long instructionId,
    Instance *regRdInstance, Instance *inputTokenFIFO,
    Instance *outputTokenFIFO, llvm::DenseMap<Value, Instance *> &input_fifos,
    llvm::DenseMap<Value, llvm::SmallVector<std::pair<BlockInfo *, Instance *>, 4>>
        &output_fifos,
    const std::string &namePrefix)
    : BlockHandler(pass, mainModule, funcOp, poolInstance, roccInstance,
                   hellaMemInstance, dmaItfc, csrItfc, circuit, mainClk,
                   mainRst, instructionId, regRdInstance, inputTokenFIFO,
                   outputTokenFIFO, input_fifos, output_fifos, namePrefix) {}

LogicalResult IfHandler::processBlock(BlockInfo &block) {
  return processIfBlock(block);
}

LogicalResult IfHandler::processIfBlock(BlockInfo &ifBlock) {
  ifOp = nullptr;
  thenEntryTokenFIFO = nullptr;
  elseEntryTokenFIFO = nullptr;
  thenDoneTokenFIFO = nullptr;
  elseDoneTokenFIFO = nullptr;
  joinTokenFIFO = nullptr;
  contextTokenReg = nullptr;
  inputStateRegs.clear();
  thenInputFIFOs.clear();
  elseInputFIFOs.clear();
  thenOutputFIFOs.clear();
  elseOutputFIFOs.clear();
  thenResultFIFOs.clear();
  elseResultFIFOs.clear();
  currentIfBlock = &ifBlock;

  for (Operation *op : ifBlock.operations) {
    if (auto candidate = dyn_cast<tor::IfOp>(op)) {
      ifOp = candidate;
      break;
    }
  }

  if (!ifOp)
    return ifBlock.mlirBlock->getParentOp()->emitError()
           << "APSToCMT2 conditional block does not contain tor.if";

  if (ifOp->getNumResults() != 0 && !hasElseRegion())
    return ifOp.emitError()
           << "tor.if with results requires a non-empty else region";

  ifName = namePrefix;
  if (!ifName.empty() && ifName.back() == '_')
    ifName.pop_back();

  if (failed(createIfInfrastructure(ifBlock)))
    return ifOp.emitError()
           << "APSToCMT2 if lowering failed while creating token "
              "infrastructure for "
           << ifName;

  if (failed(generateDispatchRule(ifBlock)))
    return ifOp.emitError()
           << "APSToCMT2 if lowering failed while generating dispatch rule "
              "for "
           << ifName;

  Block &thenBlock = ifOp.getThenRegion().front();
  if (failed(processBranchRegion(thenBlock, thenEntryTokenFIFO,
                                 thenDoneTokenFIFO, "then")))
    return ifOp.emitError()
           << "APSToCMT2 if lowering failed while processing then region for "
           << ifName;
  if (failed(generateBranchTagRule("then", thenDoneTokenFIFO, true)))
    return ifOp.emitError()
           << "APSToCMT2 if lowering failed while generating then completion "
              "tag rule for "
           << ifName;

  if (hasElseRegion()) {
    Block &elseBlock = ifOp.getElseRegion().front();
    if (failed(processBranchRegion(elseBlock, elseEntryTokenFIFO,
                                   elseDoneTokenFIFO, "else")))
      return ifOp.emitError()
             << "APSToCMT2 if lowering failed while processing else region "
                "for "
             << ifName;
    if (failed(generateBranchTagRule("else", elseDoneTokenFIFO, false)))
      return ifOp.emitError()
             << "APSToCMT2 if lowering failed while generating else "
                "completion tag rule for "
             << ifName;
  }

  if (ifOp->getNumResults() == 0) {
    if (failed(generateJoinRule()))
      return ifOp.emitError()
             << "APSToCMT2 if lowering failed while generating join rule for "
             << ifName;
  }

  return success();
}

Instance *IfHandler::createTokenFIFO(StringRef suffix) {
  auto &builder = mainModule->getBuilder();
  auto savedIP = builder.saveInsertionPoint();
  auto *tokenMod = STLLibrary::createFIFO2IModule(1, circuit);
  builder.restoreInsertionPoint(savedIP);
  return mainModule->addInstance(ifName + suffix.str(), tokenMod,
                                 {mainClk.getValue(), mainRst.getValue()});
}

Instance *IfHandler::createDataFIFO(StringRef suffix, Value value) {
  auto &builder = mainModule->getBuilder();
  auto savedIP = builder.saveInsertionPoint();
  auto *fifoMod = STLLibrary::createFIFO2IModule(getBitWidth(value.getType()),
                                                 circuit);
  builder.restoreInsertionPoint(savedIP);
  return mainModule->addInstance(ifName + suffix.str(), fifoMod,
                                 {mainClk.getValue(), mainRst.getValue()});
}

Instance *IfHandler::createStateReg(StringRef suffix, Value value,
                                    unsigned index) {
  auto &builder = mainModule->getBuilder();
  auto savedIP = builder.saveInsertionPoint();
  auto *regMod = STLLibrary::createRegModule(getBitWidth(value.getType()), 0,
                                             circuit);
  builder.restoreInsertionPoint(savedIP);
  return mainModule->addInstance(
      ifName + suffix.str() + std::to_string(index), regMod,
      {mainClk.getValue(), mainRst.getValue()});
}

LogicalResult IfHandler::createIfInfrastructure(BlockInfo &ifBlock) {
  thenEntryTokenFIFO = createTokenFIFO("_thentok");
  thenDoneTokenFIFO = createTokenFIFO("_thendone");
  joinTokenFIFO = createTokenFIFO("_jointok");
  if (hasElseRegion()) {
    elseEntryTokenFIFO = createTokenFIFO("_elsetok");
    elseDoneTokenFIFO = createTokenFIFO("_elsedone");
  }

  if (requireContextToken) {
    auto &builder = mainModule->getBuilder();
    auto savedIP = builder.saveInsertionPoint();
    auto *contextMod = STLLibrary::createRegModule(2, 1, circuit);
    builder.restoreInsertionPoint(savedIP);
    contextTokenReg = mainModule->addInstance(
        ifName + "_ctok", contextMod, {mainClk.getValue(), mainRst.getValue()});
  }

  if (failed(createResultMergeFIFOs()))
    return failure();

  unsigned stateIdx = 0;
  unsigned thenInputIdx = 0;
  unsigned elseInputIdx = 0;
  auto ensureState = [&](Value value) -> Instance * {
    if (!value || value.getDefiningOp<arith::ConstantOp>())
      return nullptr;
    auto it = inputStateRegs.find(value);
    if (it != inputStateRegs.end())
      return it->second;
    Instance *reg = createStateReg("_isr", value, stateIdx++);
    inputStateRegs[value] = reg;
    return reg;
  };

  auto considerValue = [&](Value value) {
    if (!value || value.getDefiningOp<arith::ConstantOp>())
      return;
    bool usedInThen = isValueUsedInRegion(value, ifOp.getThenRegion());
    bool usedInElse =
        hasElseRegion() && isValueUsedInRegion(value, ifOp.getElseRegion());
    bool liveThrough = output_fifos.count(value) > 0;
    bool isCondition = value == ifOp.getCondition();
    if (!usedInThen && !usedInElse && !liveThrough && !isCondition)
      return;

    ensureState(value);
    if (usedInThen)
      thenInputFIFOs[value] =
          createDataFIFO("_thenin" + std::to_string(thenInputIdx++), value);
    if (usedInElse)
      elseInputFIFOs[value] =
          createDataFIFO("_elsein" + std::to_string(elseInputIdx++), value);
  };

  for (auto &[value, fifo] : input_fifos)
    if (fifo)
      considerValue(value);
  for (auto &[value, reg] : ifBlock.scopeResources.inputValueRegs)
    if (reg)
      considerValue(value);

  return success();
}

LogicalResult IfHandler::createResultMergeFIFOs() {
  if (ifOp->getNumResults() == 0)
    return success();

  auto &builder = mainModule->getBuilder();
  auto savedIP = builder.saveInsertionPoint();

  for (auto result : llvm::enumerate(ifOp.getResults())) {
    unsigned bitWidth = getBitWidth(result.value().getType());
    auto *thenMod = STLLibrary::createFIFO2IModule(bitWidth, circuit);
    builder.restoreInsertionPoint(savedIP);
    thenResultFIFOs.push_back(mainModule->addInstance(
        ifName + "_thenres" + std::to_string(result.index()), thenMod,
        {mainClk.getValue(), mainRst.getValue()}));

    auto *elseMod = STLLibrary::createFIFO2IModule(bitWidth, circuit);
    builder.restoreInsertionPoint(savedIP);
    elseResultFIFOs.push_back(mainModule->addInstance(
        ifName + "_elseres" + std::to_string(result.index()), elseMod,
        {mainClk.getValue(), mainRst.getValue()}));
  }

  if (failed(populateBranchResultOutputs(ifOp.getThenRegion(), thenResultFIFOs,
                                         thenOutputFIFOs)))
    return failure();
  if (failed(populateBranchResultOutputs(ifOp.getElseRegion(), elseResultFIFOs,
                                         elseOutputFIFOs)))
    return failure();

  return success();
}

LogicalResult IfHandler::populateBranchResultOutputs(
    Region &region, ArrayRef<Instance *> resultFIFOs,
    llvm::DenseMap<Value, llvm::SmallVector<std::pair<BlockInfo *, Instance *>, 4>>
        &branchOutputFIFOs) {
  if (ifOp->getNumResults() == 0)
    return success();
  if (region.empty())
    return ifOp.emitError("tor.if result branch has no region block");

  auto yieldOp = dyn_cast<tor::YieldOp>(region.front().getTerminator());
  if (!yieldOp)
    return ifOp.emitError("tor.if result branch must terminate with tor.yield");
  if (yieldOp->getNumOperands() != ifOp->getNumResults())
    return ifOp.emitError("tor.if yield operand count does not match results");

  for (auto indexedOperand : llvm::enumerate(yieldOp->getOperands())) {
    Instance *fifo = resultFIFOs[indexedOperand.index()];
    if (!fifo)
      continue;
    branchOutputFIFOs[indexedOperand.value()].push_back({nullptr, fifo});
  }
  return success();
}

bool IfHandler::hasNonEmptyRegion(Region &region) const {
  if (region.empty())
    return false;
  Block &block = region.front();
  for (Operation &op : block.getOperations()) {
    if (!isa<tor::YieldOp>(&op))
      return true;
  }
  return false;
}

bool IfHandler::hasElseRegion() {
  return ifOp && !ifOp.getElseRegion().empty();
}

bool IfHandler::isValueUsedInRegion(Value value, Region &region) const {
  if (!value || region.empty())
    return false;
  for (Block &block : region.getBlocks()) {
    for (Operation &op : block.getOperations()) {
      for (Value operand : op.getOperands()) {
        if (operand == value)
          return true;
      }
      bool nestedUse = false;
      op.walk([&](Operation *nestedOp) {
        for (Value operand : nestedOp->getOperands()) {
          if (operand == value) {
            nestedUse = true;
            return WalkResult::interrupt();
          }
        }
        return WalkResult::advance();
      });
      if (nestedUse)
        return true;
    }
  }
  return false;
}

Value IfHandler::materializeCondition(OpBuilder &builder, Location loc,
                                      BlockInfo &ifBlock) {
  Value condition = ifOp.getCondition();

  auto regIt = ifBlock.scopeResources.inputValueRegs.find(condition);
  if (regIt != ifBlock.scopeResources.inputValueRegs.end() && regIt->second) {
    auto values = regIt->second->callValue("read", builder);
    if (!values.empty())
      return values[0];
  }

  auto inputRegIt = input_regs.find(condition);
  if (inputRegIt != input_regs.end() && inputRegIt->second) {
    auto values = inputRegIt->second->callValue("read", builder);
    if (!values.empty())
      return values[0];
  }

  auto fifoIt = ifBlock.input_fifos.find(condition);
  if (fifoIt != ifBlock.input_fifos.end() && fifoIt->second) {
    auto values = fifoIt->second->callMethod("deq", {}, builder);
    if (!values.empty())
      return values[0];
  }

  if (auto constOp = condition.getDefiningOp<arith::ConstantOp>())
  {
    auto intAttr = cast<IntegerAttr>(constOp.getValueAttr());
    unsigned width = cast<IntegerType>(intAttr.getType()).getWidth();
    return UInt::constant(intAttr.getValue().getZExtValue(), width, builder,
                          loc)
        .getValue();
  }

  if (isa<circt::firrtl::FIRRTLBaseType>(condition.getType()))
    return condition;

  llvm::report_fatal_error(
      "IfHandler: condition is not available as a FIRRTL value");
}

LogicalResult IfHandler::generateDispatchRule(BlockInfo &ifBlock) {
  auto *rule = mainModule->addRule(ifName + "_dispatch_rule");

  rule->guard([&](OpBuilder &builder) {
    auto loc = builder.getUnknownLoc();
    if (contextTokenReg) {
      auto tokenValues = contextTokenReg->callValue("read", builder);
      if (tokenValues.empty())
        llvm::report_fatal_error(
            "IfHandler: context token read returned no value");
      Signal token(tokenValues[0], &builder, loc);
      auto one = UInt::constant(1, 2, builder, loc);
      builder.create<circt::cmt2::ReturnOp>(loc, (token == one).getValue());
      return;
    }
    auto alwaysTrue = UInt::constant(1, 1, builder, loc);
    builder.create<circt::cmt2::ReturnOp>(loc, alwaysTrue.getValue());
  });

  rule->body([&](OpBuilder &builder) {
    auto loc = builder.getUnknownLoc();

    if (contextTokenReg) {
      auto unavailable = UInt::constant(0, 2, builder, loc);
      contextTokenReg->callMethod("write", {unavailable.getValue()}, builder);
    }

    if (inputTokenFIFO)
      inputTokenFIFO->callMethod("deq", {}, builder);

    llvm::DenseMap<Value, Value> capturedValues;
    for (auto &[value, stateReg] : inputStateRegs) {
      if (!stateReg)
        continue;

      Value captured;
      auto fifoIt = ifBlock.input_fifos.find(value);
      if (fifoIt != ifBlock.input_fifos.end() && fifoIt->second) {
        auto values = fifoIt->second->callMethod("deq", {}, builder);
        if (!values.empty())
          captured = values[0];
      }
      if (!captured) {
        auto regIt = ifBlock.scopeResources.inputValueRegs.find(value);
        if (regIt != ifBlock.scopeResources.inputValueRegs.end() &&
            regIt->second) {
          auto values = regIt->second->callValue("read", builder);
          if (!values.empty())
            captured = values[0];
        }
      }
      if (!captured) {
        auto regIt = input_regs.find(value);
        if (regIt != input_regs.end() && regIt->second) {
          auto values = regIt->second->callValue("read", builder);
          if (!values.empty())
            captured = values[0];
        }
      }
      if (!captured)
        continue;
      stateReg->callMethod("write", {captured}, builder);
      capturedValues[value] = captured;
    }

    Value condValue = capturedValues.lookup(ifOp.getCondition());
    if (!condValue)
      condValue = materializeCondition(builder, loc, ifBlock);
    Signal cond(condValue, &builder, loc);

    If(cond,
       [&](OpBuilder &thenBuilder) {
         emitBranchInputs(thenBuilder, loc, capturedValues, thenInputFIFOs);
         auto token = UInt::constant(1, 1, thenBuilder, loc);
         thenEntryTokenFIFO->callMethod("enq", {token.getValue()},
                                        thenBuilder);
       },
       [&](OpBuilder &elseBuilder) {
         auto tag = UInt::constant(0, 1, elseBuilder, loc);
         if (elseEntryTokenFIFO) {
           emitBranchInputs(elseBuilder, loc, capturedValues, elseInputFIFOs);
           elseEntryTokenFIFO->callMethod("enq", {tag.getValue()},
                                          elseBuilder);
         } else {
           joinTokenFIFO->callMethod("enq", {tag.getValue()}, elseBuilder);
         }
       },
       builder, loc);

    builder.create<circt::cmt2::ReturnOp>(loc);
  });

  rule->finalize();
  return success();
}

LogicalResult IfHandler::processBranchRegion(Block &region,
                                             Instance *entryTokenFIFO,
                                             Instance *doneTokenFIFO,
                                             StringRef branchName) {
  if (!hasNonEmptyRegion(*region.getParent()))
    return generateYieldOnlyBranchRule(branchName, entryTokenFIFO,
                                       doneTokenFIFO);

  BlockHandler branchHandler(pass, mainModule, funcOp, poolInstance,
                             roccInstance, hellaMemInstance, dmaItfc, csrItfc,
                             circuit, mainClk, mainRst, instructionId,
                             regRdInstance, entryTokenFIFO, doneTokenFIFO,
                             branchName == "then" ? thenInputFIFOs
                                                  : elseInputFIFOs,
                             branchName == "then" ? thenOutputFIFOs
                                                  : elseOutputFIFOs,
                             ifName + "_" + branchName.str() + "_");

  for (auto &[value, reg] : input_regs)
    branchHandler.addInputRegister(value, reg);

  return branchHandler.processRegionAsBlocks(&region, ifOp.getOperation());
}

LogicalResult IfHandler::generateYieldOnlyBranchRule(StringRef branchName,
                                                     Instance *entryTokenFIFO,
                                                     Instance *doneTokenFIFO) {
  auto *rule = mainModule->addRule(ifName + "_" + branchName.str() +
                                   "_yield_rule");
  auto &branchInputFIFOs =
      branchName == "then" ? thenInputFIFOs : elseInputFIFOs;
  auto &branchOutputFIFOs =
      branchName == "then" ? thenOutputFIFOs : elseOutputFIFOs;

  rule->guard([](OpBuilder &builder) {
    auto loc = builder.getUnknownLoc();
    auto alwaysTrue = UInt::constant(1, 1, builder, loc);
    builder.create<circt::cmt2::ReturnOp>(loc, alwaysTrue.getValue());
  });

  rule->body([&](OpBuilder &builder) {
    auto loc = builder.getUnknownLoc();
    llvm::DenseMap<Value, Value> localMap;

    entryTokenFIFO->callMethod("deq", {}, builder);

    for (auto &[value, fifo] : branchInputFIFOs) {
      if (!fifo)
        continue;
      auto values = fifo->callMethod("deq", {}, builder);
      if (!values.empty())
        localMap[value] = values[0];
    }

    for (auto &[value, consumers] : branchOutputFIFOs) {
      Value payload = localMap.lookup(value);
      if (!payload) {
        Instance *stateReg = inputStateRegs.lookup(value);
        if (stateReg) {
          auto values = stateReg->callValue("read", builder);
          if (!values.empty())
            payload = values[0];
        }
      }
      if (!payload) {
        if (auto constOp = value.getDefiningOp<arith::ConstantOp>()) {
          auto intAttr = cast<IntegerAttr>(constOp.getValueAttr());
          unsigned width = cast<IntegerType>(intAttr.getType()).getWidth();
          payload = UInt::constant(intAttr.getValue().getZExtValue(), width,
                                   builder, loc)
                        .getValue();
        }
      }
      if (!payload)
        return;
      for (auto &[consumerBlock, fifo] : consumers) {
        (void)consumerBlock;
        if (fifo)
          fifo->callMethod("enq", {payload}, builder);
      }
    }

    auto token = UInt::constant(1, 1, builder, loc);
    doneTokenFIFO->callMethod("enq", {token.getValue()}, builder);
    builder.create<circt::cmt2::ReturnOp>(loc);
  });

  rule->finalize();
  return success();
}

void IfHandler::emitBranchInputs(
    OpBuilder &builder, Location loc, llvm::DenseMap<Value, Value> &capturedValues,
    llvm::DenseMap<Value, Instance *> &branchFIFOs) {
  (void)loc;
  for (auto &[value, fifo] : branchFIFOs) {
    if (!fifo)
      continue;
    Value payload = capturedValues.lookup(value);
    if (!payload) {
      Instance *stateReg = inputStateRegs.lookup(value);
      if (!stateReg)
        continue;
      auto values = stateReg->callValue("read", builder);
      if (values.empty())
        continue;
      payload = values[0];
    }
    fifo->callMethod("enq", {payload}, builder);
  }
}

void IfHandler::emitLiveThroughOutputs(OpBuilder &builder, Location loc) {
  (void)loc;
  for (auto &[value, consumers] : output_fifos) {
    Instance *stateReg = inputStateRegs.lookup(value);
    if (!stateReg)
      continue;
    auto values = stateReg->callValue("read", builder);
    if (values.empty())
      continue;
    for (auto &[consumerBlock, fifo] : consumers) {
      (void)consumerBlock;
      if (fifo)
        fifo->callMethod("enq", {values[0]}, builder);
    }
  }
}

void IfHandler::emitResultOutputs(OpBuilder &builder, Location loc,
                                  ArrayRef<Instance *> resultFIFOs) {
  (void)loc;
  for (auto result : llvm::enumerate(ifOp.getResults())) {
    if (result.index() >= resultFIFOs.size())
      continue;
    Instance *resultFIFO = resultFIFOs[result.index()];
    if (!resultFIFO)
      continue;
    auto payloadValues = resultFIFO->callMethod("deq", {}, builder);
    if (payloadValues.empty())
      continue;

    auto consumersIt = output_fifos.find(result.value());
    if (consumersIt == output_fifos.end())
      consumersIt = output_fifos.end();

    if (currentIfBlock) {
      auto regIt =
          currentIfBlock->scopeResources.outputValueRegs.find(result.value());
      if (regIt != currentIfBlock->scopeResources.outputValueRegs.end()) {
        for (Instance *reg : regIt->second) {
          if (reg)
            reg->callMethod("write", {payloadValues[0]}, builder);
        }
      }
    }

    if (consumersIt != output_fifos.end()) {
      for (auto &[consumerBlock, fifo] : consumersIt->second) {
        (void)consumerBlock;
        if (fifo)
          fifo->callMethod("enq", {payloadValues[0]}, builder);
      }
    }
  }
}

LogicalResult IfHandler::generateBranchTagRule(StringRef branchName,
                                               Instance *branchDoneFIFO,
                                               bool isThen) {
  auto *rule =
      mainModule->addRule(ifName + "_" + branchName.str() + "_tag_rule");

  rule->guard([](OpBuilder &builder) {
    auto loc = builder.getUnknownLoc();
    auto alwaysTrue = UInt::constant(1, 1, builder, loc);
    builder.create<circt::cmt2::ReturnOp>(loc, alwaysTrue.getValue());
  });

  rule->body([&](OpBuilder &builder) {
    auto loc = builder.getUnknownLoc();

    branchDoneFIFO->callMethod("deq", {}, builder);

    if (ifOp->getNumResults() != 0) {
      emitResultOutputs(builder, loc,
                        isThen ? thenResultFIFOs : elseResultFIFOs);
      emitLiveThroughOutputs(builder, loc);

      if (outputTokenFIFO) {
        auto token = UInt::constant(1, 1, builder, loc);
        outputTokenFIFO->callMethod("enq", {token.getValue()}, builder);
      }

      if (contextTokenReg) {
        auto available = UInt::constant(1, 2, builder, loc);
        contextTokenReg->callMethod("write", {available.getValue()}, builder);
      }
    } else {
      auto tag = UInt::constant(isThen ? 1 : 0, 1, builder, loc);
      joinTokenFIFO->callMethod("enq", {tag.getValue()}, builder);
    }

    builder.create<circt::cmt2::ReturnOp>(loc);
  });

  rule->finalize();
  return success();
}

LogicalResult IfHandler::generateJoinRule() {
  auto *rule = mainModule->addRule(ifName + "_join_rule");

  rule->guard([](OpBuilder &builder) {
    auto loc = builder.getUnknownLoc();
    auto alwaysTrue = UInt::constant(1, 1, builder, loc);
    builder.create<circt::cmt2::ReturnOp>(loc, alwaysTrue.getValue());
  });

  rule->body([&](OpBuilder &builder) {
    auto loc = builder.getUnknownLoc();

    auto tagValues = joinTokenFIFO->callMethod("deq", {}, builder);

    if (ifOp->getNumResults() != 0) {
      if (tagValues.empty())
        llvm::report_fatal_error("IfHandler: join token returned no tag");
      Signal tag(tagValues[0], &builder, loc);
      If(tag,
         [&](OpBuilder &thenBuilder) {
           emitResultOutputs(thenBuilder, loc, thenResultFIFOs);
         },
         [&](OpBuilder &elseBuilder) {
           emitResultOutputs(elseBuilder, loc, elseResultFIFOs);
         },
         builder, loc);
    }

    emitLiveThroughOutputs(builder, loc);

    if (outputTokenFIFO) {
      auto token = UInt::constant(1, 1, builder, loc);
      outputTokenFIFO->callMethod("enq", {token.getValue()}, builder);
    }

    if (contextTokenReg) {
      auto available = UInt::constant(1, 2, builder, loc);
      contextTokenReg->callMethod("write", {available.getValue()}, builder);
    }

    builder.create<circt::cmt2::ReturnOp>(loc);
  });

  rule->finalize();
  return success();
}

} // namespace mlir
