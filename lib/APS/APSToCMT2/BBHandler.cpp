//===- BBHandler.cpp - Basic Block Handler Implementation
//------------------===//
//
// This file implements the object-oriented basic block handling for TOR
// function rule generation
//
//===----------------------------------------------------------------------===//

#include "APS/BBHandler.h"
#include "APS/FloatOpGenerator.h"
#include "APS/HardwareValueUtils.h"
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
  floatGen = std::make_unique<FloatOpGenerator>(this);
  memoryGen = std::make_unique<MemoryOpGenerator>(this);
  interfaceGen = std::make_unique<InterfaceOpGenerator>(this);
  registerGen = std::make_unique<RegisterOpGenerator>(this);

  // Set up register generator with required instances (shared across all
  // blocks)
  registerGen->setRegRdInstance(regRdInstance);
}

BBHandler::~BBHandler() = default;

llvm::SmallVector<Value> BBHandler::getProducedValuesForSlot(int64_t slot) {
  auto values = floatGen->getResults(slot);
  for (Operation *op : slotMap[slot].ops) {
    if (floatGen->canHandle(op) ||
        isa<aps::CopyIssue, aps::LoadIssue, aps::StoreIssue, aps::ReadSmemIssue>(op))
      continue;
    llvm::append_range(values, op->getResults());
  }
  return values;
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
      if (floatGen->canHandle(op))
        if (auto end = op->getAttrOfType<IntegerAttr>("endtime"))
          slotMap.try_emplace(end.getInt());
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
      for (Type type : op->getOperandTypes())
        if (isa<FloatType>(type) && !type.isF32() && !type.isBF16())
          return op->emitError("HardFloat backend supports only scalar f32/bf16");
      for (Type type : op->getResultTypes())
        if (isa<FloatType>(type) && !type.isF32() && !type.isBF16())
          return op->emitError("HardFloat backend supports only scalar f32/bf16");
      if (isa<arith::ConstantOp>(op) || isa<memref::GetGlobalOp>(op) ||
          arithmeticGen->canHandle(op) || floatGen->canHandle(op) ||
          memoryGen->canHandle(op) ||
          interfaceGen->canHandle(op) || registerGen->canHandle(op))
        continue;

      return op->emitError("unsupported operation for APSToCMT2 rule generation");
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
  } else if (floatGen->canHandle(op)) {
    return floatGen->generateRule(op, b, loc, slot, localMap);
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

  // A constant-bound loop can leave an empty entry segment. It still owns
  // command capture and boundary token forwarding, so emit one empty stage.
  collectOperationsFromList(blockOperations);
  if (slotOrder.empty()) {
    slotMap.try_emplace(0);
    slotOrder.push_back(0);
  }

  if (failed(validateOperations())) {
    llvm::dbgs() << "[BBHandler] Unsupported operation found in basic block "
                 << block.blockId << "\n";
    return failure();
  }

  floatGen->clear();
  for (Operation *op : blockOperations)
    if (floatGen->canHandle(op) && failed(floatGen->prepare(op)))
      return failure();

  if (pipelineMode) {
    return processPipelineBasicBlock(block);
  }

  auto &builder = mainModule->getBuilder();
  auto savedIP = builder.saveInsertionPoint();

  // Non-pipeline stage-local registers belong to one context. A multicycle FP
  // operation must not admit the next context until the final slot completes.
  Instance *floatContextBusy = nullptr;
  if (llvm::any_of(blockOperations, [&](Operation *op) { return floatGen->canHandle(op); })) {
    auto *busyModule = STLLibrary::createRegModule(1, 0, circuit);
    floatContextBusy = mainModule->addInstance(
        currentBlock->blockName + "_fp_busy", busyModule,
        {mainClk.getValue(), mainRst.getValue()});
    builder.restoreInsertionPoint(savedIP);
  }

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
    auto *tokenMod = pass->createTokenFIFO(circuit, 1);
    builder.restoreInsertionPoint(savedIP);
    std::string tokenName = makeTokenName(slot);
    slotTokenFIFOs[slot] = mainModule->addInstance(
        tokenName, tokenMod, {mainClk.getValue(), mainRst.getValue()});
  }

  auto isRequestTokenProducer = [](Operation *op) {
    return isa<aps::CopyIssue, aps::LoadIssue,
               aps::StoreIssue, aps::ReadSmemIssue>(op);
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
    if (!value || !getHardwareBitWidth(value.getType()))
      return nullptr;
    if (auto *defOp = value.getDefiningOp()) {
      if (isa<arith::ConstantOp, memref::GetGlobalOp>(defOp) ||
          isRequestTokenProducer(defOp))
        return nullptr;
    }
    auto existing = localValueRegs.find(value);
    if (existing != localValueRegs.end())
      return existing->second;

    unsigned bitWidth = *getHardwareBitWidth(value.getType());
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
    for (Value result : getProducedValuesForSlot(slot)) {
      if (isValueUsedAfterSlot(result, slotIdx) ||
          (needsBlockOutput(result) && slotIdx + 1 < slotOrder.size()))
        ensureLocalValueReg(result);
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
      if (slotIdx == 0 && floatContextBusy) {
        auto busy = floatContextBusy->callValue("read", b)[0];
        auto idle = ~Signal(busy, &b, loc);
        b.create<circt::cmt2::ReturnOp>(loc, idle.getValue());
        return;
      }
      auto one = UInt::constant(1, 1, b, loc);
      b.create<circt::cmt2::ReturnOp>(loc, one.getValue());
    });

    rule->body([&, slot, slotIdx](mlir::OpBuilder &b) {
      auto loc = b.getUnknownLoc();
      llvm::DenseMap<mlir::Value, mlir::Value> localMap;

      if (slotIdx == 0) {
        if (floatContextBusy)
          floatContextBusy->callMethod("write", {UInt::constant(1, 1, b, loc).getValue()}, b);
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

      if (failed(floatGen->collectResults(slot, b, localMap)))
        llvm::report_fatal_error("failed to collect floating-point results");

      for (Operation *op : slotMap[slot].ops) {
        if (failed(generateRuleForOperation(op, b, loc, slot, localMap))) {
          llvm::report_fatal_error(
              llvm::Twine("BBHandler: failed to process operation in "
                          "non-pipeline rule: ") +
              op->getName().getStringRef());
        }
      }

      for (Value result : getProducedValuesForSlot(slot)) {
        if (!getHardwareBitWidth(result.getType()))
          continue;
        auto valueIt = localMap.find(result);
        if (valueIt == localMap.end())
          continue;

        if (Instance *reg = localValueRegs.lookup(result))
          reg->callMethod("write", {valueIt->second}, b);

        auto regIt = block.scopeResources.outputValueRegs.find(result);
        if (regIt != block.scopeResources.outputValueRegs.end()) {
          for (Instance *reg : regIt->second)
            if (reg)
              reg->callMethod("write", {valueIt->second}, b);
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
              auto constant = materializeHardwareConstant(constOp.getValueAttr(), b, loc);
              if (succeeded(constant))
                payload = *constant;
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

      if (slotIdx + 1 == slotOrder.size() && floatContextBusy)
        floatContextBusy->callMethod("write", {UInt::constant(0, 1, b, loc).getValue()}, b);
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

LogicalResult BBHandler::processPipelineBasicBlock(BlockInfo &block) {
  unsigned blockId = block.blockId;
  llvm::DenseMap<Value, Instance *> &inputFIFOs = block.input_fifos;
  llvm::DenseMap<Value,
                 llvm::SmallVector<std::pair<BlockInfo *, Instance *>, 4>>
      &outputFIFOs = block.output_fifos;
  Instance *blockInputTokenFIFO = block.input_token_fifo;
  Instance *blockOutputTokenFIFO = block.output_token_fifo;

  auto &builder = mainModule->getBuilder();
  auto savedIP = builder.saveInsertionPoint();

  // Completion counts implement dependency distance in iteration space,
  // independently of stalls and of the numeric initiation interval.
  llvm::DenseMap<int64_t, size_t> completionSlots;
  for (size_t i = 0; i < slotOrder.size(); ++i)
    for (Operation *op : slotMap[slotOrder[i]].ops)
      if (isa<aps::ReadSmemWait, aps::WriteSmem, aps::WriteSmemIf>(op))
        if (auto id = op->getAttrOfType<IntegerAttr>("aps.mem_id"))
          completionSlots[id.getInt()] = i;
  SmallVector<SmallVector<std::pair<size_t, int64_t>>> dependencies(slotOrder.size());
  llvm::DenseSet<size_t> countedSlots;
  for (size_t i = 0; i < slotOrder.size(); ++i) {
    for (Operation *op : slotMap[slotOrder[i]].ops) {
      if (!isa<aps::ReadSmemIssue, aps::WriteSmem, aps::WriteSmemIf>(op)) continue;
      if (auto deps = op->getAttrOfType<ArrayAttr>("aps.mem_deps")) {
        for (Attribute attr : deps) {
          auto dep = cast<DictionaryAttr>(attr);
          int64_t id = cast<IntegerAttr>(dep.get("source")).getInt();
          int64_t distance = cast<IntegerAttr>(dep.get("distance")).getInt();
          auto source = completionSlots.find(id);
          if (source == completionSlots.end() || distance < 0 || distance >= (1LL << 30))
            return op->emitError("unsupported elastic memory dependency source/distance");
          // Same-stage read collection and store are one atomic firing.
          if (source->second == i && distance == 0) continue;
          dependencies[i].push_back({source->second, distance});
          countedSlots.insert(i);
          countedSlots.insert(source->second);
        }
      }
    }
  }
  auto makeReg = [&](StringRef suffix, unsigned width) {
    auto *mod = STLLibrary::createRegModule(width, 0, circuit);
    builder.restoreInsertionPoint(savedIP);
    return mainModule->addInstance(currentBlock->blockName + suffix.str(), mod,
                                   {mainClk.getValue(), mainRst.getValue()});
  };
  llvm::DenseMap<size_t, Instance *> progress;
  for (size_t i = 0; i < slotOrder.size(); ++i)
    if (countedSlots.contains(i))
      progress[i] = makeReg("_completed_s" + std::to_string(slotOrder[i]), 32);

  auto parentLoop = dyn_cast<tor::ForOp>(block.mlirBlock->getParentOp());
  auto iiAttr = parentLoop ? parentLoop->getAttrOfType<IntegerAttr>("II") : IntegerAttr();
  if (!iiAttr || iiAttr.getInt() < 1 || iiAttr.getInt() > INT32_MAX)
    return block.mlirBlock->getParentOp()->emitError("missing/invalid scheduled pipeline II");
  unsigned ii = iiAttr.getInt();
  llvm::dbgs() << "[BBHandler] Pipeline scheduled II=" << ii << "\n";
  unsigned cooldownWidth = std::max(1u, llvm::Log2_32_Ceil(ii));
  Instance *cooldown = ii == 1 ? nullptr : makeReg("_admit_cooldown", cooldownWidth);
  if (cooldown) {
    auto *tick = mainModule->addRule(currentBlock->blockName + "_cooldown_tick");
    tick->guard([&](mlir::OpBuilder &b) {
      auto loc = b.getUnknownLoc();
      auto count = Signal(cooldown->callValue("read", b)[0], &b, loc);
      b.create<circt::cmt2::ReturnOp>(loc,
          (count != UInt::constant(0, cooldownWidth, b, loc)).getValue());
    });
    tick->body([&](mlir::OpBuilder &b) {
      auto loc = b.getUnknownLoc();
      auto count = Signal(cooldown->callValue("read", b)[0], &b, loc);
      auto next = (count - UInt::constant(1, cooldownWidth, b, loc)).bits(cooldownWidth-1, 0);
      cooldown->callMethod("write", {next.getValue()}, b);
      b.create<circt::cmt2::ReturnOp>(loc);
    });
    tick->finalize();
  }

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
    // Registered readiness breaks combinational cycles between downstream
    // stage backpressure and arbitration among read sites sharing a bank.
    auto *tokenMod = pass->createTokenFIFO(circuit, 2);
    builder.restoreInsertionPoint(savedIP);
    std::string tokenName = makeTokenName(slot);
    slotTokenFIFOs[slot] = mainModule->addInstance(
        tokenName, tokenMod, {mainClk.getValue(), mainRst.getValue()});
  }

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
    if (!value || !getHardwareBitWidth(value.getType()))
      return nullptr;
    if (edgeIdx >= liveEdgeFIFOs.size())
      return nullptr;
    auto existing = liveEdgeFIFOs[edgeIdx].find(value);
    if (existing != liveEdgeFIFOs[edgeIdx].end())
      return existing->second;

    unsigned bitWidth = *getHardwareBitWidth(value.getType());
    auto *fifoMod = STLLibrary::createFIFO2IModule(bitWidth, circuit);
    builder.restoreInsertionPoint(savedIP);
    std::string fifoName = makeLiveEdgeName(edgeIdx, dataFifoCounter++);
    Instance *fifo = mainModule->addInstance(
        fifoName, fifoMod, {mainClk.getValue(), mainRst.getValue()});
    liveEdgeFIFOs[edgeIdx][value] = fifo;
    return fifo;
  };

  auto createLivePath = [&](Value value, size_t producerIdx) {
    if (!value || !getHardwareBitWidth(value.getType()))
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
    for (Value result : getProducedValuesForSlot(slot))
      createLivePath(result, producerIdx);
  }

  for (size_t slotIdx = 0; slotIdx < slotOrder.size(); ++slotIdx) {
    int64_t slot = slotOrder[slotIdx];
    auto *rule = mainModule->addRule(makeSlotRuleName(slot));
    rule->guard([&, slotIdx](mlir::OpBuilder &b) {
      auto loc = b.getUnknownLoc();
      Signal ready = UInt::constant(1, 1, b, loc);
      // Token availability is enforced by the body deq method's readiness.
      if (slotIdx == 0 && cooldown) {
        auto count = Signal(cooldown->callValue("read", b)[0], &b, loc);
        ready = ready & (count == UInt::constant(0, cooldownWidth, b, loc));
      }
      for (auto [source, distance] : dependencies[slotIdx]) {
        auto produced = Signal(progress[source]->callValue("read", b)[0], &b, loc);
        auto consumed = Signal(progress[slotIdx]->callValue("read", b)[0], &b, loc);
        // Modular comparison: live iteration differences are bounded by the
        // pipeline's finite FIFOs, far below half of the 32-bit sequence space.
        auto delta = (produced - consumed + UInt::constant(distance, 32, b, loc)).bits(31, 0);
        ready = ready & (~delta.bits(31, 31)) & (delta != UInt::constant(0, 32, b, loc));
      }
      b.create<circt::cmt2::ReturnOp>(loc, ready.getValue());
    });

    rule->body([&, slot, slotIdx](mlir::OpBuilder &b) {
      auto loc = b.getUnknownLoc();
      llvm::DenseMap<mlir::Value, mlir::Value> localMap;

      if (auto *counter = progress.lookup(slotIdx)) {
        auto count = Signal(counter->callValue("read", b)[0], &b, loc);
        auto next = (count + UInt::constant(1, 32, b, loc)).bits(31, 0);
        counter->callMethod("write", {next.getValue()}, b);
      }
      if (slotIdx == 0 && cooldown)
        cooldown->callMethod("write", {UInt::constant(ii-1, cooldownWidth, b, loc).getValue()}, b);

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

      if (failed(floatGen->collectResults(slot, b, localMap)))
        llvm::report_fatal_error("failed to collect floating-point results");

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
  return success();
}

// Implementation of missing BBHandler methods
bool BBHandler::isControlFlowBoundary(Operation *op) {
  return isa<tor::ForOp, tor::IfOp, tor::WhileOp>(op);
}

mlir::Type BBHandler::toFirrtlType(mlir::Type type, mlir::MLIRContext *ctx) {
  if (auto width = getHardwareBitWidth(type))
    return circt::firrtl::UIntType::get(ctx, *width);
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
    auto constant = materializeHardwareConstant(constOp.getValueAttr(), b, loc);
    if (failed(constant))
      return currentOp->emitError("unsupported scalar hardware constant (expected integer or f32)");
    localMap[v] = *constant;
    return *constant;
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
