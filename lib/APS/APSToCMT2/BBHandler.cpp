//===- BBHandler.cpp - Basic Block Handler Implementation ------------------===//
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

BBHandler::BBHandler(APSToCMT2Pass *pass, Module *mainModule, tor::FuncOp funcOp,
                    Instance *poolInstance, Instance *roccInstance,
                    Instance *hellaMemInstance, Instance *regRdInstance,
                    InterfaceDecl *dmaItfc, InterfaceDecl *csrItfc,
                    Circuit &circuit, Clock mainClk, Reset mainRst,
                    unsigned long opcode)
    : pass(pass), mainModule(mainModule), funcOp(funcOp), poolInstance(poolInstance),
      roccInstance(roccInstance), hellaMemInstance(hellaMemInstance), dmaItfc(dmaItfc),
      csrItfc(csrItfc),
      circuit(circuit), mainClk(mainClk), mainRst(mainRst), opcode(opcode), regRdInstance(regRdInstance) {

  // Initialize operation generators
  arithmeticGen = std::make_unique<ArithmeticOpGenerator>(this);
  memoryGen = std::make_unique<MemoryOpGenerator>(this);
  interfaceGen = std::make_unique<InterfaceOpGenerator>(this);
  registerGen = std::make_unique<RegisterOpGenerator>(this);

  // Set up register generator with required instances (shared across all blocks)
  registerGen->setRegRdInstance(regRdInstance);
}

void BBHandler::addReverseSlotRulePrecedence() {
  if (!currentBlock || slotOrder.size() < 2)
    return;

  llvm::SmallVector<std::pair<std::string, std::string>, 16> pairs;
  for (size_t laterIdx = slotOrder.size(); laterIdx > 1; --laterIdx) {
    int64_t laterSlot = slotOrder[laterIdx - 1];
    std::string laterRuleName = currentBlock->blockName + "_slot_" +
                                std::to_string(laterSlot) + "_rule";
    for (size_t earlierIdx = 0; earlierIdx < laterIdx - 1; ++earlierIdx) {
      int64_t earlierSlot = slotOrder[earlierIdx];
      std::string earlierRuleName = currentBlock->blockName + "_slot_" +
                                    std::to_string(earlierSlot) + "_rule";
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

LogicalResult BBHandler::processBasicBlocks() {
  if (!funcOp)
    return failure();
  return funcOp.emitError()
         << "BBHandler::processBasicBlocks is disabled; use BlockHandler to "
            "provide explicit non-pipeline block boundaries";
}

LogicalResult BBHandler::collectOperationsBySlot() {
  // New approach: Identify basic blocks by control flow boundaries
  // Operations within the same basic block can span multiple timeslots naturally
  
  llvm::dbgs() << "[BBHandler] Collecting operations by basic block (control-flow based)\n";
  
  // First, identify basic blocks based on control flow operations
  llvm::SmallVector<llvm::SmallVector<Operation*, 8>> basicBlocks;
  llvm::SmallVector<Operation*, 8> currentBlock;
  
  for (Operation &op : funcOp.getBody().getOps()) {
    if (isa<tor::TimeGraphOp>(op) || isa<tor::ReturnOp>(op))
      continue;
      
    if (isa<arith::ConstantOp>(op)) {
      // Constants can be processed separately
      continue;
    }
    
    // Check if this operation starts a new basic block
    if (isControlFlowBoundary(&op)) {
      if (!currentBlock.empty()) {
        basicBlocks.push_back(std::move(currentBlock));
        currentBlock.clear();
      }
      // Control flow operations get their own block
      currentBlock.push_back(&op);
      basicBlocks.push_back(std::move(currentBlock));
      currentBlock.clear();
    } else {
      // Regular operation - add to current block
      currentBlock.push_back(&op);
    }
  }
  
  // Add final block if not empty
  if (!currentBlock.empty()) {
    basicBlocks.push_back(std::move(currentBlock));
  }
  
  llvm::dbgs() << "[BBHandler] Identified " << basicBlocks.size() << " basic blocks\n";
  
  // Now organize operations by timeslot within each basic block
  // For operations with explicit timeslots, use them; otherwise infer timing
  for (auto &block : basicBlocks) {
    for (Operation *op : block) {
      if (auto startAttr = op->getAttrOfType<IntegerAttr>("starttime")) {
        int64_t slot = startAttr.getInt();
        slotMap[slot].ops.push_back(op);
        llvm::dbgs() << "[BBHandler] Operation with explicit timeslot: slot " << slot << "\n";
      } else {
        // For operations without explicit timeslots, we need to infer timing
        // This will be handled by the basic block's natural flow
        llvm::dbgs() << "[BBHandler] Operation without explicit timeslot - will infer timing\n";
        // For now, place in slot 0 - this will be refined later
        slotMap[0].ops.push_back(op);
      }
    }
  }
  
  // Populate sorted slot order
  for (auto &kv : slotMap)
    slotOrder.push_back(kv.first);
  llvm::sort(slotOrder);
  
  if (slotOrder.empty() && !basicBlocks.empty()) {
    // If no explicit timeslots, create a single slot for the basic block
    slotOrder.push_back(0);
  }

  return success();
}

LogicalResult BBHandler::collectOperationsFromList(llvm::SmallVector<Operation*> &operations) {
  // Organize the provided operations by their time slots
  llvm::dbgs() << "[BBHandler] Organizing " << operations.size() << " operations by time slots\n";
  
  // Clear existing slot map and order
  slotMap.clear();
  slotOrder.clear();
  
  // Process each operation and assign to appropriate slot
  for (Operation *op : operations) {
    if (auto startAttr = op->getAttrOfType<IntegerAttr>("starttime")) {
      int64_t slot = startAttr.getInt();
      slotMap[slot].ops.push_back(op);
      llvm::dbgs() << "[BBHandler] Operation with explicit timeslot: slot " << slot << " - " << op->getName() << "\n";
    } else {
      // For operations without explicit timeslots, place in slot 0
      slotMap[0].ops.push_back(op);
      llvm::dbgs() << "[BBHandler] Operation without explicit timeslot - placed in slot 0 - " << op->getName() << "\n";
    }
  }
  
  // Populate sorted slot order
  for (auto &kv : slotMap)
    slotOrder.push_back(kv.first);
  llvm::sort(slotOrder);
  
  if (slotOrder.empty() && !operations.empty()) {
    // If no explicit timeslots but we have operations, create a single slot
    slotOrder.push_back(0);
  }
  
  llvm::dbgs() << "[BBHandler] Organized operations into " << slotOrder.size() << " time slots\n";
  for (int64_t slot : slotOrder) {
    llvm::dbgs() << "[BBHandler]   Slot " << slot << " has " << slotMap[slot].ops.size() << " operations\n";
  }
  
  return success();
}

LogicalResult BBHandler::validateOperations() {
  for (int64_t slot : slotOrder) {
    for (Operation *op : slotMap[slot].ops) {
      if (isa<arith::ConstantOp, memref::GetGlobalOp>(op))
        continue;
      if (isa<tor::AddIOp, tor::SubIOp, tor::MulIOp>(op))
        continue;
      if (isa<mlir::arith::AddIOp, mlir::arith::SubIOp, mlir::arith::MulIOp>(op))
        continue;
      if (isa<mlir::arith::AndIOp, mlir::arith::OrIOp, mlir::arith::XOrIOp>(op))
        continue;
      if (isa<mlir::arith::ShLIOp, mlir::arith::ShRSIOp, mlir::arith::ShRUIOp>(op))
        continue;
      if (isa<mlir::arith::CmpIOp>(op))
        continue;
      if (isa<aps::GlobalLoad, aps::GlobalStore, aps::ReadCSR,
              aps::WriteCSR>(op)) {
        continue;
      }
      if (isa<aps::ItfcBurstLoadReq, aps::ItfcBurstStoreReq, aps::ItfcLoadReq, aps::ItfcStoreReq,
              aps::ItfcBurstLoadCollect, aps::ItfcBurstStoreCollect, aps::ItfcLoadCollect, aps::ItfcStoreCollect>(op)) {
        continue;
      }
      if (isa<aps::SpmLoadReq, aps::SpmLoadCollect>(op)) {
        continue;
      }
      op->emitError("unsupported operation for rule generation");
      return failure();
    }
  }
  return success();
}

LogicalResult BBHandler::handleRoCCCommandBundle(mlir::OpBuilder &b, Location loc) {
  // Only handle RoCC commands if we have a function context
  if (!funcOp) {
    llvm::dbgs() << "[BBHandler] No function context, skipping RoCC command bundle\n";
    return success();
  }
  
  // Call cmd_to_user once to get the RoCC command bundle
  std::string cmdMethod = "cmd_to_user_" + (std::ostringstream() << std::hex << std::setw(4) << std::setfill('0') << opcode).str();
  auto cmdResult = roccInstance->callMethod(cmdMethod, {}, b)[0];
  cachedRoCCCmdBundle = cmdResult;
  auto instruction = Bundle(cachedRoCCCmdBundle, &b, loc);
  regRdInstance->callMethod("write", {instruction["rd"].getValue()}, b);

  // Set the cached bundle in register generator
  registerGen->setCachedRoCCCmdBundle(cachedRoCCCmdBundle);
  return success();
}

LogicalResult BBHandler::generateRuleForOperation(Operation *op, mlir::OpBuilder &b,
                                                 Location loc, int64_t slot,
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

LogicalResult BBHandler::processBasicBlock(BlockInfo& block) {
  llvm::dbgs() << "[BBHandler] Processing basic block " << block.blockId
               << " (" << block.blockName << ") with "
               << block.mlirBlock->getOperations().size() << " operations\n";

  // Store block reference for use throughout the handler
  currentBlock = &block;

  unsigned blockId = block.blockId;
  llvm::DenseMap<Value, Instance*> &inputFIFOs = block.input_fifos;
  llvm::DenseMap<Value, llvm::SmallVector<std::pair<BlockInfo*, Instance*>, 4>> &outputFIFOs = block.output_fifos;
  Instance *block_input_token_fifo = block.input_token_fifo;
  Instance *block_output_token_fifo = block.output_token_fifo;

  // Use the operations specifically assigned to this block segment
  // BlockHandler has already filtered out control flow operations
  llvm::SmallVector<Operation*> blockOperations;

  for (Operation *op : block.operations) {
    // Skip terminators and special operations
    if (op->hasTrait<mlir::OpTrait::IsTerminator>()) {
      continue;
    }

    // Skip timegraph and return operations
    if (isa<tor::TimeGraphOp>(op) || isa<tor::ReturnOp>(op)) {
      continue;
    }

    blockOperations.push_back(op);
  }

  llvm::dbgs() << "[BBHandler] Collected " << blockOperations.size() << " operations from block segment (out of "
               << block.operations.size() << " total in segment)\n";

  // PANIC: Empty blocks should not reach BBHandler
  if (blockOperations.empty()) {
    llvm::report_fatal_error("BBHandler received empty block - this should have been handled by BlockHandler");
  }

  // Phase 2: Organize operations by time slots
  if (failed(collectOperationsFromList(blockOperations))) {
    llvm::dbgs() << "[BBHandler] Failed to organize operations by slot\n";
    return failure();
  }

  if (pipelineMode)
    return processPipelineBasicBlock(block);

  auto &builder = mainModule->getBuilder();
  auto savedIP = builder.saveInsertionPoint();

  llvm::DenseMap<int64_t, Instance *> slotTokenFIFOs;
  for (size_t i = 0; i + 1 < slotOrder.size(); ++i) {
    int64_t slot = slotOrder[i];
    auto *tokenMod = STLLibrary::createFIFO1PushModule(1, circuit);
    builder.restoreInsertionPoint(savedIP);
    std::string tokenName = currentBlock->blockName + "_s" +
                            std::to_string(slot) + "tok";
    slotTokenFIFOs[slot] = mainModule->addInstance(
        tokenName, tokenMod, {mainClk.getValue(), mainRst.getValue()});
  }

  auto isRequestTokenProducer = [](Operation *op) {
    return isa<aps::ItfcBurstLoadReq, aps::ItfcBurstStoreReq,
               aps::ItfcLoadReq, aps::ItfcStoreReq, aps::SpmLoadReq>(op);
  };

  auto slotIndexOf = [&](int64_t slot) -> size_t {
    auto it = llvm::find(slotOrder, slot);
    if (it == slotOrder.end())
      return slotOrder.size();
    return static_cast<size_t>(std::distance(slotOrder.begin(), it));
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

  auto isValueUsedAfterSlot = [&](Value value, int64_t producerSlot) {
    size_t producerIdx = slotIndexOf(producerSlot);
    for (size_t i = producerIdx + 1; i < slotOrder.size(); ++i) {
      if (isValueUsedInSlot(value, slotOrder[i]))
        return true;
    }
    return false;
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
    std::string regName = currentBlock->blockName + "_r" +
                          std::to_string(localRegCounter++);
    Instance *reg = mainModule->addInstance(
        regName, regMod, {mainClk.getValue(), mainRst.getValue()});
    localValueRegs[value] = reg;
    block.scopeResources.stageLocalRegs[value] = reg;
    return reg;
  };

  for (int64_t slot : slotOrder) {
    for (Operation *op : slotMap[slot].ops) {
      if (isRequestTokenProducer(op))
        continue;
      for (Value result : op->getResults()) {
        if (isValueUsedAfterSlot(result, slot))
          ensureLocalValueReg(result);
      }
    }
  }

  int64_t firstSlot = slotOrder.front();
  for (auto &[value, fifo] : inputFIFOs) {
    if (!fifo)
      continue;
    if (isValueUsedAfterSlot(value, firstSlot))
      ensureLocalValueReg(value);
  }
  for (auto &[value, reg] : block.scopeResources.inputValueRegs) {
    if (!reg)
      continue;
    if (isValueUsedAfterSlot(value, firstSlot))
      ensureLocalValueReg(value);
  }

  for (int64_t slot : slotOrder) {
    auto *rule =
        mainModule->addRule(currentBlock->blockName + "_slot_" +
                            std::to_string(slot) + "_rule");
    rule->guard([&, slot](mlir::OpBuilder &b) {
      auto loc = b.getUnknownLoc();
      if (slot != slotOrder.front()) {
        auto it = llvm::find(slotOrder, slot);
        if (it != slotOrder.begin()) {
          int64_t prevSlot = *(it - 1);
          if (Instance *tokenFIFO = slotTokenFIFOs.lookup(prevSlot)) {
            auto full = tokenFIFO->callValue("full", b);
            if (!full.empty()) {
              b.create<circt::cmt2::ReturnOp>(loc, full[0]);
              return;
            }
          }
        }
      }
      auto one = UInt::constant(1, 1, b, loc);
      b.create<circt::cmt2::ReturnOp>(loc, one.getValue());
    });

    rule->body([&, slot](mlir::OpBuilder &b) {
      auto loc = b.getUnknownLoc();
      llvm::DenseMap<mlir::Value, mlir::Value> localMap;

      if (slot == slotOrder.front()) {
        if (block_input_token_fifo) {
          llvm::dbgs() << "[BBHandler] Dequeuing input token for block "
                       << blockId << "\n";
          block_input_token_fifo->callMethod("deq", {}, b);
        }

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
        auto it = llvm::find(slotOrder, slot);
        if (it != slotOrder.begin()) {
          int64_t prevSlot = *(it - 1);
          if (Instance *tokenFIFO = slotTokenFIFOs.lookup(prevSlot))
            tokenFIFO->callMethod("deq", {}, b);
        }

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

      if (slot == 0) {
        if (failed(handleRoCCCommandBundle(b, loc)))
          return;
      }

      for (Operation *op : slotMap[slot].ops) {
        if (failed(generateRuleForOperation(op, b, loc, slot, localMap))) {
          llvm::dbgs() << "[BBHandler] Failed to process operation: " << *op
                       << "\n";
          return;
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

          auto outputIt = outputFIFOs.find(result);
          if (outputIt != outputFIFOs.end()) {
            for (const auto &[consumerBlock, fifo] : outputIt->second) {
              if (!fifo)
                continue;
              fifo->callMethod("enq", {valueIt->second}, b);
              llvm::dbgs()
                  << "[BBHandler] Enqueued value to block output FIFO for "
                     "consumer block "
                  << consumerBlock->blockId << "\n";
            }
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

LogicalResult BBHandler::processPipelineBasicBlock(BlockInfo &block) {
  unsigned blockId = block.blockId;
  llvm::DenseMap<Value, Instance *> &inputFIFOs = block.input_fifos;
  llvm::DenseMap<Value, llvm::SmallVector<std::pair<BlockInfo *, Instance *>, 4>>
      &outputFIFOs = block.output_fifos;
  Instance *blockInputTokenFIFO = block.input_token_fifo;
  Instance *blockOutputTokenFIFO = block.output_token_fifo;

  auto &builder = mainModule->getBuilder();
  auto savedIP = builder.saveInsertionPoint();

  llvm::DenseMap<int64_t, Instance *> slotTokenFIFOs;
  for (size_t i = 0; i + 1 < slotOrder.size(); ++i) {
    int64_t slot = slotOrder[i];
    auto *tokenMod = STLLibrary::createFIFO1PushModule(1, circuit);
    builder.restoreInsertionPoint(savedIP);
    std::string tokenName = currentBlock->blockName + "_s" +
                            std::to_string(slot) + "tok";
    slotTokenFIFOs[slot] = mainModule->addInstance(
        tokenName, tokenMod, {mainClk.getValue(), mainRst.getValue()});
  }

  auto isRequestTokenProducer = [](Operation *op) {
    return isa<aps::ItfcBurstLoadReq, aps::ItfcBurstStoreReq,
               aps::ItfcLoadReq, aps::ItfcStoreReq, aps::SpmLoadReq>(op);
  };

  auto slotIndexOf = [&](int64_t slot) -> size_t {
    auto it = llvm::find(slotOrder, slot);
    if (it == slotOrder.end())
      return slotOrder.size();
    return static_cast<size_t>(std::distance(slotOrder.begin(), it));
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
    std::string fifoName = currentBlock->blockName + "_s" +
                           std::to_string(slotOrder[edgeIdx]) + "v" +
                           std::to_string(dataFifoCounter++) + "s" +
                           std::to_string(slotOrder[edgeIdx + 1]);
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
      lastRequiredIdx =
          std::max<int64_t>(lastRequiredIdx,
                            static_cast<int64_t>(slotOrder.size() - 1));
    if (lastRequiredIdx <= static_cast<int64_t>(producerIdx))
      return;

    for (size_t edgeIdx = producerIdx;
         edgeIdx < static_cast<size_t>(lastRequiredIdx); ++edgeIdx)
      ensureLiveEdgeFIFO(value, edgeIdx);
  };

  int64_t firstSlot = slotOrder.front();
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

  for (int64_t slot : slotOrder) {
    for (Operation *op : slotMap[slot].ops) {
      if (isRequestTokenProducer(op))
        continue;
      for (Value result : op->getResults()) {
        if (!isa<mlir::IntegerType>(result.getType()))
          continue;
        createLivePath(result, slotIndexOf(slot));
      }
    }
  }

  for (int64_t slot : slotOrder) {
    auto *rule =
        mainModule->addRule(currentBlock->blockName + "_slot_" +
                            std::to_string(slot) + "_rule");
    rule->guard([&, slot](mlir::OpBuilder &b) {
      auto loc = b.getUnknownLoc();
      if (slot != firstSlot) {
        auto it = llvm::find(slotOrder, slot);
        if (it != slotOrder.begin()) {
          int64_t prevSlot = *(it - 1);
          if (Instance *tokenFIFO = slotTokenFIFOs.lookup(prevSlot)) {
            auto full = tokenFIFO->callValue("full", b);
            if (!full.empty()) {
              b.create<circt::cmt2::ReturnOp>(loc, full[0]);
              return;
            }
          }
        }
      }
      auto one = UInt::constant(1, 1, b, loc);
      b.create<circt::cmt2::ReturnOp>(loc, one.getValue());
    });

    rule->body([&, slot](mlir::OpBuilder &b) {
      auto loc = b.getUnknownLoc();
      llvm::DenseMap<mlir::Value, mlir::Value> localMap;
      size_t slotIdx = slotIndexOf(slot);

      if (slot == firstSlot) {
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
        auto it = llvm::find(slotOrder, slot);
        if (it != slotOrder.begin()) {
          int64_t prevSlot = *(it - 1);
          if (Instance *tokenFIFO = slotTokenFIFOs.lookup(prevSlot))
            tokenFIFO->callMethod("deq", {}, b);
        }

        if (slotIdx == 0)
          llvm::report_fatal_error("non-first pipeline slot has index 0");
        for (auto &[value, fifo] : liveEdgeFIFOs[slotIdx - 1]) {
          if (!fifo)
            continue;
          auto dequeuedValue = fifo->callMethod("deq", {}, b);
          if (!dequeuedValue.empty())
            localMap[value] = dequeuedValue[0];
        }
      }

      if (slot == 0) {
        if (failed(handleRoCCCommandBundle(b, loc)))
          return;
      }

      for (Operation *op : slotMap[slot].ops) {
        if (failed(generateRuleForOperation(op, b, loc, slot, localMap))) {
          llvm::dbgs() << "[BBHandler] Failed to process operation: " << *op
                       << "\n";
          return;
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

      if (slot == slotOrder.back()) {
        for (auto &[value, consumers] : outputFIFOs) {
          auto valueIt = localMap.find(value);
          if (valueIt == localMap.end())
            continue;
          for (const auto &[consumerBlock, outFIFO] : consumers) {
            (void)consumerBlock;
            if (outFIFO)
              outFIFO->callMethod("enq", {valueIt->second}, b);
          }
        }
      }

      if (slot != slotOrder.back()) {
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
  if (auto intType = dyn_cast<mlir::IntegerType>(type)) {
    return circt::firrtl::UIntType::get(ctx, intType.getWidth());
  }
  return nullptr;
}

unsigned int BBHandler::roundUpToPowerOf2(unsigned int n) {
  if (n == 0) return 1;
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
  if (n == 0) return 0;
  unsigned int log = 0;
  while (n > 1) {
    n >>= 1;
    log++;
  }
  return log;
}

FailureOr<mlir::Value> OperationGenerator::getValueInRule(mlir::Value v, Operation *currentOp,
                                                          unsigned operandIndex, mlir::OpBuilder &b,
                                                          llvm::DenseMap<mlir::Value, mlir::Value> &localMap,
                                                          Location loc) {
  if (auto it = localMap.find(v); it != localMap.end())
    return it->second;

  if (auto constOp = v.getDefiningOp<arith::ConstantOp>()) {
    auto intAttr = mlir::cast<IntegerAttr>(constOp.getValueAttr());
    unsigned width = mlir::cast<IntegerType>(intAttr.getType()).getWidth();
    auto constant = UInt::constant(intAttr.getValue().getZExtValue(), width, b, loc).getValue();
    localMap[v] = constant;
    return constant;
  }

  if (auto globalOp = v.getDefiningOp<memref::GetGlobalOp>()) {
    // Global symbols are handled separately via symbol resolution.
    return mlir::Value{};
  }

  currentOp->emitError("value is not available in this rule");
  return failure();
}

} // namespace mlir
