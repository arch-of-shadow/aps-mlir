//===- BBHandler.cpp - Basic Block Handler Implementation
//------------------===//
//
// This file implements the object-oriented basic block handling for TOR
// function rule generation
//
//===----------------------------------------------------------------------===//

#include "APS/BBHandler.h"
#include "APS/APSOps.h"
#include "circt/Dialect/Cmt2/ECMT2/Signal.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Operation.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {

using namespace mlir;
using namespace mlir::tor;
using namespace circt::cmt2::ecmt2;
using namespace circt::cmt2::ecmt2::stl;
using namespace circt::firrtl;

//===----------------------------------------------------------------------===//
// BBHandler Implementation
//===----------------------------------------------------------------------===//

BBHandler::BBHandler(APSToCMT2Pass *pass, Module *mainModule,
                     tor::FuncOp funcOp, Instance *poolInstance,
                     Instance *roccInstance, Instance *hellaMemInstance,
                     Instance *regRdInstance, InterfaceDecl *dmaItfc,
                     InterfaceDecl *csrItfc, Circuit &circuit, Clock mainClk,
                     Reset mainRst, unsigned long instructionId)
    : pass(pass), mainModule(mainModule), funcOp(funcOp),
      poolInstance(poolInstance), roccInstance(roccInstance),
      hellaMemInstance(hellaMemInstance), dmaItfc(dmaItfc), csrItfc(csrItfc),
      circuit(circuit), mainClk(mainClk), mainRst(mainRst), instructionId(instructionId),
      regRdInstance(regRdInstance) {

  // Initialize operation generators
  arithmeticGen = std::make_unique<ArithmeticOpGenerator>(this);
  memoryGen = std::make_unique<MemoryOpGenerator>(this);
  interfaceGen = std::make_unique<InterfaceOpGenerator>(this);
  registerGen = std::make_unique<RegisterOpGenerator>(this);

  // Set up register generator with required instances (shared across all
  // blocks)
  registerGen->setRegRdInstance(regRdInstance);
}

void BBHandler::addReverseSlotRulePrecedence() {
  if (!currentBlock || slotOrder.size() < 2)
    return;

  auto makeSlotRuleName = [&](int64_t slot) {
    return llvm::formatv("{0}_slot_{1}_rule", currentBlock->blockName, slot)
        .str();
  };

  llvm::SmallVector<std::pair<std::string, std::string>, 16> pairs;
  for (size_t laterIdx = slotOrder.size(); laterIdx > 1; --laterIdx) {
    int64_t laterSlot = slotOrder[laterIdx - 1];
    std::string laterRuleName = makeSlotRuleName(laterSlot);
    for (size_t earlierIdx = 0; earlierIdx < laterIdx - 1; ++earlierIdx) {
      int64_t earlierSlot = slotOrder[earlierIdx];
      std::string earlierRuleName = makeSlotRuleName(earlierSlot);
      pairs.push_back({laterRuleName, earlierRuleName});
    }
  }

  if (!pairs.empty()) {
    mainModule->setPrecedence(pairs);
    llvm::dbgs() << "[BBHandler] Set reverse slot precedence for "
                 << pairs.size() << " rule pairs in block "
                 << currentBlock->blockName << "\n";
  }
}

void BBHandler::collectOperationsFromList(
    llvm::SmallVector<Operation *> &operations) {
  // Organize the provided operations by their time slots
  // Clear existing slot map and order
  slotMap.clear();
  slotOrder.clear();

  // Process each operation and assign to appropriate slot
  for (Operation *op : operations) {
    if (auto startAttr = op->getAttrOfType<IntegerAttr>("starttime")) {
      int64_t slot = startAttr.getInt();
      slotMap[slot].ops.push_back(op);
    } else {
      llvm::report_fatal_error(
          llvm::Twine("Operation missing required starttime attribute: ") +
          op->getName().getStringRef());
    }
  }

  // Populate sorted slot order
  for (auto &kv : slotMap)
    slotOrder.push_back(kv.first);
  llvm::sort(slotOrder);
}

LogicalResult BBHandler::validateOperations() {
  for (int64_t slot : slotOrder) {
    for (Operation *op : slotMap[slot].ops) {
      if (isa<arith::ConstantOp>(op) || isa<memref::GetGlobalOp>(op) ||
          arithmeticGen->canHandle(op) || memoryGen->canHandle(op) ||
          interfaceGen->canHandle(op) || registerGen->canHandle(op))
        continue;

      op->emitError("unsupported operation for APSToCMT2 rule generation");
      llvm::report_fatal_error("unsupported operation reached BBHandler");
    }
  }
  return success();
}

void BBHandler::handleRoCCCommandBundle(mlir::OpBuilder &b, Location loc) {
  if (!funcOp) {
    llvm::report_fatal_error(
        "handleRoCCCommandBundle requires a valid function context");
  }

  // Call cmd_to_user once to get the RoCC command bundle
  std::string cmdMethod =
      llvm::formatv("cmd_to_user_{0}", llvm::format_hex_no_prefix(instructionId, 4))
          .str();
  auto cmdResult = roccInstance->callMethod(cmdMethod, {}, b)[0];
  auto instruction = Bundle(cmdResult, &b, loc);
  regRdInstance->callMethod("write", {instruction["rd"].getValue()}, b);

  // Set the cached bundle in register generator
  registerGen->setCachedRoCCCmdBundle(cmdResult);
}

LogicalResult BBHandler::generateRuleForOperation(
    Operation *op, mlir::OpBuilder &b, Location loc, int64_t slot,
    llvm::DenseMap<mlir::Value, mlir::Value> &localMap) {
  // Try each operation generator in order
  if (arithmeticGen->canHandle(op)) {
    return arithmeticGen->generateRule(op, b, loc, slot, localMap);
  } else if (memoryGen->canHandle(op)) {
    return memoryGen->generateRule(op, b, loc, slot, localMap);
  } else if (interfaceGen->canHandle(op)) {
    return interfaceGen->generateRule(op, b, loc, slot, localMap);
  } else if (registerGen->canHandle(op)) {
    return registerGen->generateRule(op, b, loc, slot, localMap);
  }

  op->emitError("no operation generator can handle this operation");
  return failure();
}

std::optional<int64_t> BBHandler::getSlotForOp(Operation *op) {
  if (auto attr = op->getAttrOfType<IntegerAttr>("starttime"))
    return attr.getInt();
  return {};
}

LogicalResult BBHandler::processBasicBlock(BlockInfo &block) {
  llvm::dbgs() << "[BBHandler] Processing basic block " << block.blockId << " ("
               << block.blockName << ") with "
               << block.mlirBlock->getOperations().size() << " operations\n";

  // Store block reference for use throughout the handler
  currentBlock = &block;

  unsigned blockId = block.blockId;
  llvm::DenseMap<Value, Instance *> &inputFIFOs = block.input_fifos;
  llvm::DenseMap<Value,
                 llvm::SmallVector<std::pair<BlockInfo *, Instance *>, 4>>
      &outputFIFOs = block.output_fifos;
  Instance *block_input_token_fifo = block.input_token_fifo;
  Instance *block_output_token_fifo = block.output_token_fifo;

  // Use the operations specifically assigned to this block segment
  // BlockHandler has already filtered out control flow operations
  llvm::SmallVector<Operation *> blockOperations;

  for (Operation *op : block.operations) {
    // Skip terminators and special operations
    if (op->hasTrait<mlir::OpTrait::IsTerminator>() ||
        isa<tor::TimeGraphOp>(op) || isa<tor::ReturnOp>(op)) {
      continue;
    }
    blockOperations.push_back(op);
  }

  // PANIC: Empty blocks should not reach BBHandler
  if (blockOperations.empty()) {
    llvm::report_fatal_error("BBHandler received empty block");
  }

  // Phase 2: Organize operations by time slots
  collectOperationsFromList(blockOperations);

  if (failed(validateOperations())) {
    llvm::dbgs() << "[BBHandler] Unsupported operation found in basic block "
                 << block.blockId << "\n";
    return failure();
  }

  if (pipelineMode) {
    processPipelineBasicBlock(block);
    return success();
  }

  auto &builder = mainModule->getBuilder();
  auto savedIP = builder.saveInsertionPoint();

  llvm::DenseMap<int64_t, Instance *> slotTokenFIFOs;
  auto makeTokenName = [&](int64_t slot) {
    return llvm::formatv("{0}_s{1}tok", currentBlock->blockName, slot).str();
  };
  auto makeSlotRuleName = [&](int64_t slot) {
    return llvm::formatv("{0}_slot_{1}_rule", currentBlock->blockName, slot)
        .str();
  };
  auto makeRegName = [&](unsigned index) {
    return llvm::formatv("{0}_reg{1}", currentBlock->blockName, index).str();
  };
  for (size_t i = 0; i + 1 < slotOrder.size(); ++i) {
    int64_t slot = slotOrder[i];
    auto *tokenMod = STLLibrary::createFIFO1PushModule(1, circuit);
    builder.restoreInsertionPoint(savedIP);
    std::string tokenName = makeTokenName(slot);
    slotTokenFIFOs[slot] = mainModule->addInstance(
        tokenName, tokenMod, {mainClk.getValue(), mainRst.getValue()});
  }

  auto isRequestTokenProducer = [](Operation *op) {
    return isa<aps::ItfcBurstLoadReq, aps::ItfcBurstStoreReq, aps::ItfcLoadReq,
               aps::ItfcStoreReq, aps::SpmLoadReq>(op);
  };

  auto isOpInSlot = [&](Operation *candidate, int64_t slot) {
    auto slotIt = slotMap.find(slot);
    if (slotIt == slotMap.end())
      return false;
    return llvm::is_contained(slotIt->second.ops, candidate);
  };

  auto isValueUsedInSlot = [&](Value value, int64_t slot) {
    for (OpOperand &use : value.getUses()) {
      if (isOpInSlot(use.getOwner(), slot))
        return true;
    }
    return false;
  };

  auto isValueUsedAfterSlot = [&](Value value, size_t producerIdx) {
    if (producerIdx >= slotOrder.size())
      return false;
    for (size_t i = producerIdx + 1; i < slotOrder.size(); ++i) {
      if (isValueUsedInSlot(value, slotOrder[i]))
        return true;
    }
    return false;
  };

  auto needsBlockOutput = [&](Value value) {
    auto outputIt = outputFIFOs.find(value);
    return outputIt != outputFIFOs.end() && !outputIt->second.empty();
  };

  llvm::DenseMap<Value, Instance *> localValueRegs;
  unsigned localRegCounter = 0;
  auto ensureLocalValueReg = [&](Value value) -> Instance * {
    if (!value || !isa<mlir::IntegerType>(value.getType()))
      return nullptr;
    if (auto *defOp = value.getDefiningOp()) {
      if (isa<arith::ConstantOp, memref::GetGlobalOp>(defOp) ||
          isRequestTokenProducer(defOp))
        return nullptr;
    }
    auto existing = localValueRegs.find(value);
    if (existing != localValueRegs.end())
      return existing->second;

    unsigned bitWidth = 32;
    if (auto intType = dyn_cast<mlir::IntegerType>(value.getType()))
      bitWidth = intType.getWidth();
    auto *regMod = STLLibrary::createRegModule(bitWidth, 0, circuit);
    builder.restoreInsertionPoint(savedIP);
    std::string regName = makeRegName(localRegCounter++);
    Instance *reg = mainModule->addInstance(
        regName, regMod, {mainClk.getValue(), mainRst.getValue()});
    localValueRegs[value] = reg;
    block.scopeResources.stageLocalRegs[value] = reg;
    return reg;
  };

  for (size_t slotIdx = 0; slotIdx < slotOrder.size(); ++slotIdx) {
    int64_t slot = slotOrder[slotIdx];
    for (Operation *op : slotMap[slot].ops) {
      if (isRequestTokenProducer(op))
        continue;
      for (Value result : op->getResults()) {
        if (isValueUsedAfterSlot(result, slotIdx) ||
            (needsBlockOutput(result) && slotIdx + 1 < slotOrder.size()))
          ensureLocalValueReg(result);
      }
    }
  }

  for (auto &[value, fifo] : inputFIFOs) {
    if (isValueUsedAfterSlot(value, 0) || needsBlockOutput(value)) {
      if (!fifo) {
        llvm::report_fatal_error(
            "BBHandler: expected live input FIFO for value used after slot 0");
      }
      ensureLocalValueReg(value);
    }
  }
  for (auto &[value, reg] : block.scopeResources.inputValueRegs) {
    if (isValueUsedAfterSlot(value, 0) || needsBlockOutput(value)) {
      if (!reg) {
        llvm::report_fatal_error("BBHandler: expected input value register for "
                                 "value used after slot 0");
      }
      ensureLocalValueReg(value);
    }
  }

  for (size_t slotIdx = 0; slotIdx < slotOrder.size(); ++slotIdx) {
    int64_t slot = slotOrder[slotIdx];
    auto *rule = mainModule->addRule(makeSlotRuleName(slot));
    rule->guard([&, slotIdx](mlir::OpBuilder &b) {
      auto loc = b.getUnknownLoc();
      if (slotIdx != 0) {
        int64_t prevSlot = slotOrder[slotIdx - 1];
        if (Instance *tokenFIFO = slotTokenFIFOs.lookup(prevSlot)) {
          auto full = tokenFIFO->callValue("full", b);
          if (!full.empty()) {
            b.create<circt::cmt2::ReturnOp>(loc, full[0]);
            return;
          }
        }
      }
      auto one = UInt::constant(1, 1, b, loc);
      b.create<circt::cmt2::ReturnOp>(loc, one.getValue());
    });

    rule->body([&, slot, slotIdx](mlir::OpBuilder &b) {
      auto loc = b.getUnknownLoc();
      llvm::DenseMap<mlir::Value, mlir::Value> localMap;

      if (slotIdx == 0) {
        if (block_input_token_fifo)
          block_input_token_fifo->callMethod("deq", {}, b);

        for (auto &[value, fifo] : inputFIFOs) {
          if (!fifo)
            continue;
          auto dequeuedValue = fifo->callMethod("deq", {}, b);
          if (dequeuedValue.empty())
            continue;
          if (isValueUsedInSlot(value, slot))
            localMap[value] = dequeuedValue[0];
          if (Instance *reg = localValueRegs.lookup(value))
            reg->callMethod("write", {dequeuedValue[0]}, b);
        }

        for (auto &[value, reg] : block.scopeResources.inputValueRegs) {
          if (!reg)
            continue;
          auto storedValue = reg->callValue("read", b);
          if (storedValue.empty())
            continue;
          if (isValueUsedInSlot(value, slot))
            localMap[value] = storedValue[0];
          if (Instance *localReg = localValueRegs.lookup(value))
            localReg->callMethod("write", {storedValue[0]}, b);
        }
      } else {
        int64_t prevSlot = slotOrder[slotIdx - 1];
        if (Instance *tokenFIFO = slotTokenFIFOs.lookup(prevSlot))
          tokenFIFO->callMethod("deq", {}, b);

        for (Operation *op : slotMap[slot].ops) {
          for (Value operand : op->getOperands()) {
            if (localMap.count(operand))
              continue;
            if (Instance *reg = localValueRegs.lookup(operand)) {
              auto storedValue = reg->callValue("read", b);
              if (!storedValue.empty())
                localMap[operand] = storedValue[0];
            }
          }
        }
      }

      if (slotIdx == 0 && block.captures_rocc_command) {
        handleRoCCCommandBundle(b, loc);
      }

      for (Operation *op : slotMap[slot].ops) {
        if (failed(generateRuleForOperation(op, b, loc, slot, localMap))) {
          llvm::report_fatal_error(
              llvm::Twine("BBHandler: failed to process operation in "
                          "non-pipeline rule: ") +
              op->getName().getStringRef());
        }
      }

      for (Operation *op : slotMap[slot].ops) {
        if (isRequestTokenProducer(op))
          continue;
        for (mlir::Value result : op->getResults()) {
          if (!isa<mlir::IntegerType>(result.getType()))
            continue;
          auto valueIt = localMap.find(result);
          if (valueIt == localMap.end())
            continue;

          if (Instance *reg = localValueRegs.lookup(result))
            reg->callMethod("write", {valueIt->second}, b);

          auto regIt = block.scopeResources.outputValueRegs.find(result);
          if (regIt != block.scopeResources.outputValueRegs.end()) {
            for (Instance *reg : regIt->second) {
              if (reg)
                reg->callMethod("write", {valueIt->second}, b);
            }
          }
        }
      }

      if (slotIdx == slotOrder.size() - 1) {
        for (auto &[value, consumers] : outputFIFOs) {
          Value payload;
          auto valueIt = localMap.find(value);
          if (valueIt != localMap.end())
            payload = valueIt->second;
          if (!payload) {
            if (Instance *reg = localValueRegs.lookup(value)) {
              auto storedValue = reg->callValue("read", b);
              if (!storedValue.empty())
                payload = storedValue[0];
            }
          }
          if (!payload) {
            auto regIt = block.scopeResources.inputValueRegs.find(value);
            if (regIt != block.scopeResources.inputValueRegs.end() &&
                regIt->second) {
              auto storedValue = regIt->second->callValue("read", b);
              if (!storedValue.empty())
                payload = storedValue[0];
            }
          }
          if (!payload) {
            if (auto constOp = value.getDefiningOp<arith::ConstantOp>()) {
              auto intAttr = cast<IntegerAttr>(constOp.getValueAttr());
              unsigned width = cast<IntegerType>(intAttr.getType()).getWidth();
              payload = UInt::constant(intAttr.getValue().getZExtValue(),
                                       width, b, loc)
                            .getValue();
            }
          }
          if (!payload)
            continue;
          for (const auto &[_, fifo] : consumers) {
            if (fifo)
              fifo->callMethod("enq", {payload}, b);
          }
        }
      }

      if (slot != slotOrder.back()) {
        if (Instance *tokenFIFO = slotTokenFIFOs.lookup(slot)) {
          auto outputToken = UInt::constant(1, 1, b, loc);
          tokenFIFO->callMethod("enq", {outputToken.getValue()}, b);
        }
      } else if (block_output_token_fifo) {
        auto outputToken = UInt::constant(1, 1, b, loc);
        block_output_token_fifo->callMethod("enq", {outputToken.getValue()}, b);
      }

      b.create<circt::cmt2::ReturnOp>(loc);
    });

    rule->finalize();
  }

  llvm::dbgs() << "[BBHandler] Successfully generated " << slotOrder.size()
               << " non-pipeline slot rules for basic block " << blockId
               << "\n";
  addReverseSlotRulePrecedence();
  return success();
}

void BBHandler::processPipelineBasicBlock(BlockInfo &block) {
  unsigned blockId = block.blockId;
  llvm::DenseMap<Value, Instance *> &inputFIFOs = block.input_fifos;
  llvm::DenseMap<Value,
                 llvm::SmallVector<std::pair<BlockInfo *, Instance *>, 4>>
      &outputFIFOs = block.output_fifos;
  Instance *blockInputTokenFIFO = block.input_token_fifo;
  Instance *blockOutputTokenFIFO = block.output_token_fifo;

  auto &builder = mainModule->getBuilder();
  auto savedIP = builder.saveInsertionPoint();

  llvm::DenseMap<int64_t, Instance *> slotTokenFIFOs;
  auto makeTokenName = [&](int64_t slot) {
    return llvm::formatv("{0}_s{1}tok", currentBlock->blockName, slot).str();
  };
  auto makeSlotRuleName = [&](int64_t slot) {
    return llvm::formatv("{0}_slot_{1}_rule", currentBlock->blockName, slot)
        .str();
  };
  auto makeLiveEdgeName = [&](size_t edgeIdx, unsigned fifoCounter) {
    return llvm::formatv("{0}_s{1}v{2}s{3}", currentBlock->blockName,
                         slotOrder[edgeIdx], fifoCounter,
                         slotOrder[edgeIdx + 1])
        .str();
  };
  for (size_t i = 0; i + 1 < slotOrder.size(); ++i) {
    int64_t slot = slotOrder[i];
    auto *tokenMod = STLLibrary::createFIFO1PushModule(1, circuit);
    builder.restoreInsertionPoint(savedIP);
    std::string tokenName = makeTokenName(slot);
    slotTokenFIFOs[slot] = mainModule->addInstance(
        tokenName, tokenMod, {mainClk.getValue(), mainRst.getValue()});
  }

  auto isRequestTokenProducer = [](Operation *op) {
    return isa<aps::ItfcBurstLoadReq, aps::ItfcBurstStoreReq, aps::ItfcLoadReq,
               aps::ItfcStoreReq, aps::SpmLoadReq>(op);
  };

  auto isOpInSlot = [&](Operation *candidate, int64_t slot) {
    auto slotIt = slotMap.find(slot);
    if (slotIt == slotMap.end())
      return false;
    return llvm::is_contained(slotIt->second.ops, candidate);
  };

  auto isValueUsedInSlot = [&](Value value, int64_t slot) {
    for (OpOperand &use : value.getUses()) {
      if (isOpInSlot(use.getOwner(), slot))
        return true;
    }
    return false;
  };

  auto lastUseIndex = [&](Value value) -> int64_t {
    int64_t last = -1;
    for (size_t i = 0; i < slotOrder.size(); ++i)
      if (isValueUsedInSlot(value, slotOrder[i]))
        last = static_cast<int64_t>(i);
    return last;
  };

  auto needsBlockOutput = [&](Value value) {
    auto outputIt = outputFIFOs.find(value);
    return outputIt != outputFIFOs.end() && !outputIt->second.empty();
  };

  llvm::SmallVector<llvm::DenseMap<Value, Instance *>, 4> liveEdgeFIFOs;
  if (slotOrder.size() > 1)
    liveEdgeFIFOs.resize(slotOrder.size() - 1);
  unsigned dataFifoCounter = 0;

  auto ensureLiveEdgeFIFO = [&](Value value, size_t edgeIdx) -> Instance * {
    if (!value || !isa<mlir::IntegerType>(value.getType()))
      return nullptr;
    if (edgeIdx >= liveEdgeFIFOs.size())
      return nullptr;
    auto existing = liveEdgeFIFOs[edgeIdx].find(value);
    if (existing != liveEdgeFIFOs[edgeIdx].end())
      return existing->second;

    unsigned bitWidth = dyn_cast<mlir::IntegerType>(value.getType()).getWidth();
    auto *fifoMod = STLLibrary::createFIFO2IModule(bitWidth, circuit);
    builder.restoreInsertionPoint(savedIP);
    std::string fifoName = makeLiveEdgeName(edgeIdx, dataFifoCounter++);
    Instance *fifo = mainModule->addInstance(
        fifoName, fifoMod, {mainClk.getValue(), mainRst.getValue()});
    liveEdgeFIFOs[edgeIdx][value] = fifo;
    return fifo;
  };

  auto createLivePath = [&](Value value, size_t producerIdx) {
    if (!value || !isa<mlir::IntegerType>(value.getType()))
      return;

    int64_t lastRequiredIdx = lastUseIndex(value);
    if (needsBlockOutput(value))
      lastRequiredIdx = std::max<int64_t>(
          lastRequiredIdx, static_cast<int64_t>(slotOrder.size() - 1));
    if (lastRequiredIdx <= static_cast<int64_t>(producerIdx))
      return;

    for (size_t edgeIdx = producerIdx;
         edgeIdx < static_cast<size_t>(lastRequiredIdx); ++edgeIdx)
      ensureLiveEdgeFIFO(value, edgeIdx);
  };

  for (auto &[value, fifo] : inputFIFOs) {
    if (!fifo)
      continue;
    createLivePath(value, 0);
  }
  for (auto &[value, reg] : block.scopeResources.inputValueRegs) {
    if (!reg)
      continue;
    createLivePath(value, 0);
  }

  for (size_t producerIdx = 0; producerIdx < slotOrder.size(); ++producerIdx) {
    int64_t slot = slotOrder[producerIdx];
    for (Operation *op : slotMap[slot].ops) {
      if (isRequestTokenProducer(op))
        continue;
      for (Value result : op->getResults()) {
        if (!isa<mlir::IntegerType>(result.getType()))
          continue;
        createLivePath(result, producerIdx);
      }
    }
  }

  for (size_t slotIdx = 0; slotIdx < slotOrder.size(); ++slotIdx) {
    int64_t slot = slotOrder[slotIdx];
    auto *rule = mainModule->addRule(makeSlotRuleName(slot));
    rule->guard([&, slotIdx](mlir::OpBuilder &b) {
      auto loc = b.getUnknownLoc();
      if (slotIdx != 0) {
        int64_t prevSlot = slotOrder[slotIdx - 1];
        if (Instance *tokenFIFO = slotTokenFIFOs.lookup(prevSlot)) {
          auto full = tokenFIFO->callValue("full", b);
          if (!full.empty()) {
            b.create<circt::cmt2::ReturnOp>(loc, full[0]);
            return;
          }
        }
      }
      auto one = UInt::constant(1, 1, b, loc);
      b.create<circt::cmt2::ReturnOp>(loc, one.getValue());
    });

    rule->body([&, slot, slotIdx](mlir::OpBuilder &b) {
      auto loc = b.getUnknownLoc();
      llvm::DenseMap<mlir::Value, mlir::Value> localMap;

      if (slotIdx == 0) {
        if (blockInputTokenFIFO)
          blockInputTokenFIFO->callMethod("deq", {}, b);

        for (auto &[value, fifo] : inputFIFOs) {
          if (!fifo)
            continue;
          auto dequeuedValue = fifo->callMethod("deq", {}, b);
          if (dequeuedValue.empty())
            continue;
          localMap[value] = dequeuedValue[0];
        }

        for (auto &[value, reg] : block.scopeResources.inputValueRegs) {
          if (!reg)
            continue;
          auto storedValue = reg->callValue("read", b);
          if (!storedValue.empty())
            localMap[value] = storedValue[0];
        }
      } else {
        int64_t prevSlot = slotOrder[slotIdx - 1];
        if (Instance *tokenFIFO = slotTokenFIFOs.lookup(prevSlot))
          tokenFIFO->callMethod("deq", {}, b);

        for (auto &[value, fifo] : liveEdgeFIFOs[slotIdx - 1]) {
          if (!fifo)
            continue;
          auto dequeuedValue = fifo->callMethod("deq", {}, b);
          if (!dequeuedValue.empty())
            localMap[value] = dequeuedValue[0];
        }
      }

      if (slotIdx == 0 && block.captures_rocc_command) {
        handleRoCCCommandBundle(b, loc);
      }

      for (Operation *op : slotMap[slot].ops) {
        if (failed(generateRuleForOperation(op, b, loc, slot, localMap))) {
          llvm::report_fatal_error(
              llvm::Twine(
                  "BBHandler: failed to process operation in pipeline rule: ") +
              op->getName().getStringRef());
        }
      }

      if (slotIdx + 1 < slotOrder.size()) {
        for (auto &[value, fifo] : liveEdgeFIFOs[slotIdx]) {
          auto valueIt = localMap.find(value);
          if (valueIt == localMap.end()) {
            llvm::errs() << "[BBHandler] Missing live-through value at slot "
                         << slot << "\n";
            llvm::report_fatal_error(
                "pipeline live-through value is not available");
          }
          fifo->callMethod("enq", {valueIt->second}, b);
        }
      }

      if (slotIdx == slotOrder.size() - 1) {
        for (auto &[value, consumers] : outputFIFOs) {
          auto valueIt = localMap.find(value);
          if (valueIt == localMap.end())
            continue;
          for (const auto &[_, outFIFO] : consumers) {
            if (outFIFO)
              outFIFO->callMethod("enq", {valueIt->second}, b);
          }
        }
      }

      if (slotIdx + 1 < slotOrder.size()) {
        if (Instance *tokenFIFO = slotTokenFIFOs.lookup(slot)) {
          auto outputToken = UInt::constant(1, 1, b, loc);
          tokenFIFO->callMethod("enq", {outputToken.getValue()}, b);
        }
      } else if (blockOutputTokenFIFO) {
        auto outputToken = UInt::constant(1, 1, b, loc);
        blockOutputTokenFIFO->callMethod("enq", {outputToken.getValue()}, b);
      }

      b.create<circt::cmt2::ReturnOp>(loc);
    });

    rule->finalize();
  }

  llvm::dbgs() << "[BBHandler] Successfully generated " << slotOrder.size()
               << " pipeline slot rules for basic block " << blockId << "\n";
  addReverseSlotRulePrecedence();
}

// Implementation of missing BBHandler methods
bool BBHandler::isControlFlowBoundary(Operation *op) {
  return isa<tor::ForOp, tor::IfOp, tor::WhileOp>(op);
}

mlir::Type BBHandler::toFirrtlType(mlir::Type type, mlir::MLIRContext *ctx) {
  if (auto intType = dyn_cast<mlir::IntegerType>(type)) {
    return circt::firrtl::UIntType::get(ctx, intType.getWidth());
  }
  return nullptr;
}

unsigned int BBHandler::roundUpToPowerOf2(unsigned int n) {
  if (n == 0)
    return 1;
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  n++;
  return n;
}

unsigned int BBHandler::log2Floor(unsigned int n) {
  if (n == 0)
    return 0;
  unsigned int log = 0;
  while (n > 1) {
    n >>= 1;
    log++;
  }
  return log;
}

FailureOr<mlir::Value> OperationGenerator::getValueInRule(
    mlir::Value v, Operation *currentOp, mlir::OpBuilder &b,
    llvm::DenseMap<mlir::Value, mlir::Value> &localMap, Location loc) {
  if (auto it = localMap.find(v); it != localMap.end())
    return it->second;

  if (auto constOp = v.getDefiningOp<arith::ConstantOp>()) {
    auto intAttr = mlir::cast<IntegerAttr>(constOp.getValueAttr());
    unsigned width = mlir::cast<IntegerType>(intAttr.getType()).getWidth();
    auto constant =
        UInt::constant(intAttr.getValue().getZExtValue(), width, b, loc)
            .getValue();
    localMap[v] = constant;
    return constant;
  }

  if (auto globalOp = v.getDefiningOp<memref::GetGlobalOp>()) {
    return currentOp->emitError()
           << "memref.get_global value @" << globalOp.getName()
           << " is not available as a scalar rule value; this operand should "
              "be handled through memory symbol resolution";
  }

  currentOp->emitError("value is not available in this rule");
  return failure();
}

} // namespace mlir
