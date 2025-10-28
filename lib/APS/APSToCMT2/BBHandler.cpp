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

BBHandler::BBHandler(APSToCMT2GenPass *pass, Module *mainModule, tor::FuncOp funcOp,
                    Instance *poolInstance, Instance *roccInstance,
                    Instance *hellaMemInstance, InterfaceDecl *dmaItfc,
                    Circuit &circuit, Clock mainClk, Reset mainRst, unsigned long opcode)
    : pass(pass), mainModule(mainModule), funcOp(funcOp), poolInstance(poolInstance),
      roccInstance(roccInstance), hellaMemInstance(hellaMemInstance), dmaItfc(dmaItfc),
      circuit(circuit), mainClk(mainClk), mainRst(mainRst), opcode(opcode) {

  // Initialize operation generators
  arithmeticGen = std::make_unique<ArithmeticOpGenerator>(this);
  memoryGen = std::make_unique<MemoryOpGenerator>(this);
  interfaceGen = std::make_unique<InterfaceOpGenerator>(this);
  registerGen = std::make_unique<RegisterOpGenerator>(this);

  // Set up register generator with required instances
  registerGen->setRegRdInstance(regRdInstance);
}

LogicalResult BBHandler::processBasicBlocks() {
  // This method requires a function context
  if (!funcOp) {
    llvm::outs() << "[BBHandler] Error: processBasicBlocks called without function context\n";
    return failure();
  }
  
  // Phase 1: Analysis
  if (failed(collectOperationsBySlot()))
    return failure();
  if (failed(validateOperations()))
    return failure();
  if (failed(buildCrossSlotFIFOs()))
    return failure();
  if (failed(createTokenFIFOs()))
    return failure();

  // Phase 2: Rule Generation
  if (failed(generateSlotRules()))
    return failure();

  return success();
}

LogicalResult BBHandler::collectOperationsBySlot() {
  // New approach: Identify basic blocks by control flow boundaries
  // Operations within the same basic block can span multiple timeslots naturally
  
  llvm::outs() << "[BBHandler] Collecting operations by basic block (control-flow based)\n";
  
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
  
  llvm::outs() << "[BBHandler] Identified " << basicBlocks.size() << " basic blocks\n";
  
  // Now organize operations by timeslot within each basic block
  // For operations with explicit timeslots, use them; otherwise infer timing
  for (auto &block : basicBlocks) {
    for (Operation *op : block) {
      if (auto startAttr = op->getAttrOfType<IntegerAttr>("starttime")) {
        int64_t slot = startAttr.getInt();
        slotMap[slot].ops.push_back(op);
        llvm::outs() << "[BBHandler] Operation with explicit timeslot: slot " << slot << "\n";
      } else {
        // For operations without explicit timeslots, we need to infer timing
        // This will be handled by the basic block's natural flow
        llvm::outs() << "[BBHandler] Operation without explicit timeslot - will infer timing\n";
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
  llvm::outs() << "[BBHandler] Organizing " << operations.size() << " operations by time slots\n";
  
  // Clear existing slot map and order
  slotMap.clear();
  slotOrder.clear();
  
  // Process each operation and assign to appropriate slot
  for (Operation *op : operations) {
    if (auto startAttr = op->getAttrOfType<IntegerAttr>("starttime")) {
      int64_t slot = startAttr.getInt();
      slotMap[slot].ops.push_back(op);
      llvm::outs() << "[BBHandler] Operation with explicit timeslot: slot " << slot << " - " << op->getName() << "\n";
    } else {
      // For operations without explicit timeslots, place in slot 0
      slotMap[0].ops.push_back(op);
      llvm::outs() << "[BBHandler] Operation without explicit timeslot - placed in slot 0 - " << op->getName() << "\n";
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
  
  llvm::outs() << "[BBHandler] Organized operations into " << slotOrder.size() << " time slots\n";
  for (int64_t slot : slotOrder) {
    llvm::outs() << "[BBHandler]   Slot " << slot << " has " << slotMap[slot].ops.size() << " operations\n";
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
      if (isa<aps::GlobalLoad, aps::GlobalStore>(op)) {
        continue;
      }
      op->emitError("unsupported operation for rule generation");
      return failure();
    }
  }
  return success();
}

LogicalResult BBHandler::buildCrossSlotFIFOs() {
  auto getSlotForOp = [&](Operation *op) -> std::optional<int64_t> {
    if (auto attr = op->getAttrOfType<IntegerAttr>("starttime"))
      return attr.getInt();
    return {};
  };

  auto &builder = mainModule->getBuilder();
  auto ctx = builder.getContext();

  // Build cross-slot FIFO mapping - group consumers by target stage
  for (int64_t slot : slotOrder) {
    for (Operation *op : slotMap[slot].ops) {
      if (isa<arith::ConstantOp>(op))
        continue;

      for (mlir::Value res : op->getResults()) {
        if (!isa<mlir::IntegerType>(res.getType()))
          continue;

        // Group consumers by their target stage
        llvm::DenseMap<int64_t, llvm::SmallVector<std::pair<Operation*, unsigned>>> consumersByStage;

        for (OpOperand &use : res.getUses()) {
          Operation *user = use.getOwner();
          auto maybeConsumerSlot = getSlotForOp(user);
          if (!maybeConsumerSlot)
            continue;
          int64_t consumerSlot = *maybeConsumerSlot;
          if (consumerSlot > slot) {
            consumersByStage[consumerSlot].push_back({user, use.getOperandNumber()});
          }
        }

        if (consumersByStage.empty())
          continue;

        auto firType = toFirrtlType(res.getType(), mainModule->getBuilder().getContext());
        if (!firType) {
          op->emitError("type is unsupported for rule lowering");
          return failure();
        }

        // Create separate FIFO for each consumer stage
        for (auto &[consumerSlot, consumers] : consumersByStage) {
          auto fifo = std::make_unique<CrossSlotFIFO>();
          fifo->producerValue = res;
          fifo->producerSlot = slot;
          fifo->consumerSlot = consumerSlot;
          fifo->firType = firType;
          fifo->consumers = std::move(consumers);

          // Generate FIFO name based on producer-consumer slot pair
          auto key = std::make_pair(slot, consumerSlot);
          unsigned count = fifoCounts[key]++;
          fifo->instanceName = std::to_string(opcode) + "_fifo_s" + std::to_string(slot) + "_s" +
                               std::to_string(consumerSlot);
          if (count > 0)
            fifo->instanceName += "_" + std::to_string(count);

          // Debug info: log FIFO creation
          llvm::outs() << "[FIFO DEBUG] Creating FIFO: " << fifo->instanceName
                       << " for producer in slot " << slot
                       << " to consumers in slot " << consumerSlot
                       << " with " << fifo->consumers.size() << " consumers\n";

          crossSlotFIFOs[res].push_back(fifo.get());
          fifoStorage.push_back(std::move(fifo));
        }
      }
    }
  }

  return success();
}

LogicalResult BBHandler::createTokenFIFOs() {
  auto savedIP = mainModule->getBuilder().saveInsertionPoint();

  // Create token FIFOs for stage synchronization - one token per stage (except last)
  stageTokenFifos.clear(); // Clear any existing token FIFOs
  for (size_t i = 0; i < slotOrder.size() - 1; ++i) {
    int64_t currentSlot = slotOrder[i];
    // Create 1-bit FIFO for token passing to next stage
    auto *tokenFifoMod = STLLibrary::createFIFO1PushModule(1, circuit);
    mainModule->getBuilder().restoreInsertionPoint(savedIP);
    std::string tokenFifoName = std::to_string(opcode) + "_token_fifo_s" + std::to_string(currentSlot);
    auto *tokenFifo = mainModule->addInstance(tokenFifoName, tokenFifoMod,
                                              {mainClk.getValue(), mainRst.getValue()});
    stageTokenFifos[currentSlot] = tokenFifo;
    llvm::outs() << "[BBHandler] Created token FIFO for slot " << currentSlot << ": " << tokenFifoName << "\n";
  }

  return success();
}

LogicalResult BBHandler::generateSlotRules() {
  auto &builder = mainModule->getBuilder();
  auto savedIPforRegRd = builder.saveInsertionPoint();

  // Create register instance for RoCC operations
  auto *regRdMod = STLLibrary::createRegModule(5, 0, circuit);
  regRdInstance = mainModule->addInstance("reg_rd_" + std::to_string(opcode), regRdMod,
                                          {mainClk.getValue(), mainRst.getValue()});
  builder.restoreInsertionPoint(savedIPforRegRd);

  // Set up register generator with required instances
  registerGen->setRegRdInstance(regRdInstance);

  // Instantiate FIFO modules for cross-slot communication
  llvm::outs() << "[FIFO DEBUG] Instantiating " << fifoStorage.size() << " FIFO modules\n";
  for (auto &fifoPtr : fifoStorage) {
    auto *fifo = fifoPtr.get();
    int64_t width = cast<circt::firrtl::UIntType>(fifo->firType).getWidthOrSentinel();
    if (width < 0)
      width = 1;

    // Debug info: log FIFO instantiation
    llvm::outs() << "[FIFO DEBUG] Instantiating FIFO: " << fifo->instanceName
                 << " (width=" << width << ")\n";

    // Create FIFO module with proper clock and reset
    auto *fifoMod = STLLibrary::createFIFO1PushModule(width, circuit);
    builder.restoreInsertionPoint(savedIPforRegRd);
    fifo->fifoInstance = mainModule->addInstance(fifo->instanceName, fifoMod,
                                                 {mainClk.getValue(), mainRst.getValue()});
  }

  // Generate rules for each time slot
  llvm::DenseMap<mlir::Value, mlir::Value> localMap;

  for (int64_t slot : slotOrder) {
    auto *rule = mainModule->addRule(std::to_string(opcode) + "_rule_s" + std::to_string(slot));

    // Guard: always ready (constant 1)
    rule->guard([](mlir::OpBuilder &b) {
      auto loc = b.getUnknownLoc();
      auto one = UInt::constant(1, 1, b, loc);
      b.create<circt::cmt2::ReturnOp>(loc, mlir::ValueRange{one.getValue()});
    });

    // Body: implement the operations for this time slot
    rule->body([this, slot, &localMap](mlir::OpBuilder &b) {
      auto loc = b.getUnknownLoc();
      localMap.clear();

      // Handle RoCC command bundle in slot 0
      if (slot == 0) {
        if (failed(handleRoCCCommandBundle(b, loc)))
          return;
      }

      // Handle token synchronization between stages
      if (failed(handleTokenSynchronization(b, loc, slot)))
        return;

      // Process all operations in this slot
      for (Operation *op : slotMap[slot].ops) {
        LogicalResult result = generateRuleForOperation(op, b, loc, slot, localMap);
        if (failed(result))
          return;
      }

      // Write token to next stage at the end
      if (failed(writeTokenToNextStage(b, loc, slot)))
        return;

      // Handle cross-slot FIFO writes for producer operations
      for (Operation *op : slotMap[slot].ops) {
        for (mlir::Value result : op->getResults()) {
          if (!isa<mlir::IntegerType>(result.getType()))
            continue;
          auto fifoIt = crossSlotFIFOs.find(result);
          if (fifoIt != crossSlotFIFOs.end()) {
            // Enqueue the value to all FIFOs for this producer
            // Use localMap directly since this is a producer value
            auto valueIt = localMap.find(result);
            if (valueIt != localMap.end()) {
              for (CrossSlotFIFO *fifo : fifoIt->second) {
                if (fifo->fifoInstance) {
                  // Build arguments for enq method
                  llvm::SmallVector<mlir::Value> args;
                  args.push_back(valueIt->second);
                  fifo->fifoInstance->callMethod("enq", args, b);
                }
              }
            }
          }
        }
      }

      b.create<circt::cmt2::ReturnOp>(loc);
    });

    rule->finalize();
  }

  return success();
}

LogicalResult BBHandler::handleRoCCCommandBundle(mlir::OpBuilder &b, Location loc) {
  // Only handle RoCC commands if we have a function context
  if (!funcOp) {
    llvm::outs() << "[BBHandler] No function context, skipping RoCC command bundle\n";
    return success();
  }
  
  // Call cmd_to_user once to get the RoCC command bundle
  std::string cmdMethod = "cmd_to_user_" + std::to_string(opcode);
  auto cmdResult = roccInstance->callMethod(cmdMethod, {}, b)[0];
  cachedRoCCCmdBundle = cmdResult;
  auto instruction = Bundle(cachedRoCCCmdBundle, &b, loc);
  regRdInstance->callMethod("write", {instruction["rd"].getValue()}, b);

  // Set the cached bundle in register generator
  registerGen->setCachedRoCCCmdBundle(cachedRoCCCmdBundle);
  return success();
}

LogicalResult BBHandler::handleTokenSynchronization(mlir::OpBuilder &b, Location loc, int64_t slot) {
  // Read token from previous stage at the start (except for first stage)
  if (!slotOrder.empty() && slot != slotOrder[0]) {
    auto it = std::find(slotOrder.begin(), slotOrder.end(), slot);
    if (it != slotOrder.begin()) {
      int64_t prevSlot = *(it - 1);
      auto tokenFifoIt = stageTokenFifos.find(prevSlot);
      if (tokenFifoIt != stageTokenFifos.end()) {
        auto *tokenFifo = tokenFifoIt->second;
        auto tokenValues = tokenFifo->callValue("deq", b);
        // Token value should be 1'b1, but we don't need to use it
        // Just reading from FIFO ensures synchronization
      }
    }
  }
  return success();
}

LogicalResult BBHandler::writeTokenToNextStage(mlir::OpBuilder &b, Location loc, int64_t slot) {
  // Write token to next stage at the end (except for last stage)
  if (!slotOrder.empty() && slot != slotOrder.back()) {
    auto it = std::find(slotOrder.begin(), slotOrder.end(), slot);
    if (it != slotOrder.end() - 1) {
      int64_t nextSlot = *(it + 1);
      auto tokenFifoIt = stageTokenFifos.find(slot);
      if (tokenFifoIt != stageTokenFifos.end()) {
        auto *tokenFifo = tokenFifoIt->second;
        auto tokenValue = UInt::constant(1, 1, b, loc);
        tokenFifo->callMethod("enq", {tokenValue.getValue()}, b);
      }
    }
  }
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

LogicalResult BBHandler::processBasicBlock(Block *mlirBlock, unsigned blockId,
                                           llvm::DenseMap<Value, Instance*> &inputFIFOs,
                                           llvm::DenseMap<Value, Instance*> &outputFIFOs,
                                           Instance *readyFIFO, Instance *completeFIFO) {
  llvm::outs() << "[BBHandler] Processing basic block " << blockId << " with " 
               << mlirBlock->getOperations().size() << " operations\n";

  // For single block processing, we need to collect only the regular operations
  // from this block, excluding control flow operations that should be handled
  // by specialized handlers (LoopHandler, ConditionalHandler, etc.)
  llvm::SmallVector<Operation*> blockOperations;
  
  // Collect operations from this basic block, but skip control flow operations
  // that should be handled by specialized handlers
  for (Operation &op : mlirBlock->getOperations()) {
    // Skip terminators and special operations
    if (op.hasTrait<mlir::OpTrait::IsTerminator>()) {
      continue;
    }
    
    // Skip timegraph and return operations
    if (isa<tor::TimeGraphOp>(&op) || isa<tor::ReturnOp>(&op)) {
      continue;
    }
    
    blockOperations.push_back(&op);
  }
  
  llvm::outs() << "[BBHandler] Collected " << blockOperations.size() << " regular operations from basic block\n";

  // If no regular operations, create a minimal rule for coordination only
  if (blockOperations.empty()) {
    llvm::outs() << "[BBHandler] No regular operations in block " << blockId << ", creating coordination-only rule\n";
    
    // Create a single rule that just handles coordination (ready -> complete)
    auto *rule = mainModule->addRule("block_" + std::to_string(blockId) + "_coord_rule");
    
    rule->guard([](mlir::OpBuilder &b) {
      auto loc = b.getUnknownLoc();
      auto alwaysTrue = UInt::constant(1, 1, b, loc);
      b.create<circt::cmt2::ReturnOp>(loc, alwaysTrue.getValue());
    });
    
    rule->body([&](mlir::OpBuilder &b) {
      auto loc = b.getUnknownLoc();

      auto readyToken = readyFIFO->callMethod("deq", {}, b);
      // Signal block completion
      llvm::outs() << "[BBHandler] Enqueuing completion token for empty block " << blockId << "\n";
      auto completeToken = UInt::constant(1, 1, b, loc);
      completeFIFO->callMethod("enq", {completeToken.getValue()}, b);
      
      b.create<circt::cmt2::ReturnOp>(loc);
    });
    
    rule->finalize();
    
    llvm::outs() << "[BBHandler] Successfully generated coordination rule for empty block " << blockId << "\n";
    return success();
  }

  // Phase 2: Organize operations by time slots
  if (failed(collectOperationsFromList(blockOperations))) {
    llvm::outs() << "[BBHandler] Failed to organize operations by slot\n";
    return failure();
  }

  // Phase 3: Set up infrastructure for this specific block
  auto &builder = mainModule->getBuilder();
  auto savedIPforRegRd = builder.saveInsertionPoint();
  
  // Create register instance for RoCC operations (if needed)
  auto *regRdMod = STLLibrary::createRegModule(5, 0, circuit);
  regRdInstance = mainModule->addInstance("reg_rd_block_" + std::to_string(blockId), regRdMod,
                                          {mainClk.getValue(), mainRst.getValue()});
  builder.restoreInsertionPoint(savedIPforRegRd);
  
  // Set up register generator with required instances
  registerGen->setRegRdInstance(regRdInstance);

  // Build cross-slot FIFOs for values that flow between slots within this block
  if (failed(buildCrossSlotFIFOs()))
    return failure();

  // Create token FIFOs for stage synchronization
  if (failed(createTokenFIFOs()))
    return failure();

  // Phase 4: Generate rules for each time slot with proper coordination
  llvm::DenseMap<mlir::Value, mlir::Value> localMap;
  
  for (int64_t slot : slotOrder) {
    auto *rule = mainModule->addRule("block_" + std::to_string(blockId) + "_slot_" + std::to_string(slot) + "_rule");

    // Guard: use CMT pattern (1'b1) for all slots - coordination handled by FIFO availability
    rule->guard([](mlir::OpBuilder &b) {
      auto loc = b.getUnknownLoc();
      auto one = UInt::constant(1, 1, b, loc);
      b.create<circt::cmt2::ReturnOp>(loc, one.getValue());
    });

    // Body: implement the operations for this time slot with proper coordination
    rule->body([&](mlir::OpBuilder &b) {
      auto loc = b.getUnknownLoc();
      localMap.clear();

      llvm::outs() << "[BBHandler] Processing slot " << slot << " with " 
                   << slotMap[slot].ops.size() << " operations\n";

      // Handle block coordination at the beginning of the first slot
      if (slot == slotOrder.front()) {
        // Dequeue from input token FIFO (token from previous block in sequence)
        if (readyFIFO) {
          llvm::outs() << "[BBHandler] Dequeuing input token for block " << blockId << " from unified token FIFO\n";
          auto inputToken = readyFIFO->callMethod("deq", {}, b);
        }
        
        // Handle cross-block value consumption (data from other blocks)
        for (auto &[value, fifo] : inputFIFOs) {
          if (fifo) {
            llvm::outs() << "[BBHandler] Dequeuing cross-block value from input FIFO\n";
            auto dequeuedValue = fifo->callMethod("deq", {}, b);
            if (!dequeuedValue.empty()) {
              localMap[value] = dequeuedValue[0];
              llvm::outs() << "[BBHandler] Stored cross-block value in localMap\n";
            }
          }
        }
      }

      // Handle RoCC command bundle in slot 0 (if present)
      if (slot == 0) {
        if (failed(handleRoCCCommandBundle(b, loc)))
          return;
      }

      // Handle token synchronization between stages (intra-block)
      if (failed(handleTokenSynchronization(b, loc, slot)))
        return;

      // Process all operations in this time slot using existing operation generators
      for (Operation *op : slotMap[slot].ops) {
        if (failed(generateRuleForOperation(op, b, loc, slot, localMap))) {
          llvm::outs() << "[BBHandler] Failed to process operation: " << *op << "\n";
          return;
        }
      }

      // Handle cross-slot FIFO writes for producer operations in this slot
      for (Operation *op : slotMap[slot].ops) {
        for (mlir::Value result : op->getResults()) {
          if (!isa<mlir::IntegerType>(result.getType()))
            continue;
          auto fifoIt = crossSlotFIFOs.find(result);
          if (fifoIt != crossSlotFIFOs.end()) {
            auto valueIt = localMap.find(result);
            if (valueIt != localMap.end()) {
              for (CrossSlotFIFO *fifo : fifoIt->second) {
                if (fifo->fifoInstance) {
                  fifo->fifoInstance->callMethod("enq", {valueIt->second}, b);
                  llvm::outs() << "[BBHandler] Enqueued value to cross-slot FIFO\n";
                }
              }
            }
          }
        }
      }

      // Write token to next stage at the end (intra-block coordination)
      if (failed(writeTokenToNextStage(b, loc, slot)))
        return;

      // Handle block coordination at the end of the last slot
      if (slot == slotOrder.back()) {
        // Enqueue cross-block values to output FIFOs (data to other blocks)
        for (auto &[value, fifo] : outputFIFOs) {
          if (fifo) {
            auto valueIt = localMap.find(value);
            if (valueIt != localMap.end()) {
              llvm::outs() << "[BBHandler] Enqueuing cross-block value to output FIFO\n";
              fifo->callMethod("enq", {valueIt->second}, b);
            }
          }
        }
        
        // Enqueue output token to unified token FIFO (token to next block in sequence)
        if (completeFIFO) {
          llvm::outs() << "[BBHandler] Enqueuing output token for block " << blockId << " to unified token FIFO\n";
          auto outputToken = UInt::constant(1, 1, b, loc);
          completeFIFO->callMethod("enq", {outputToken.getValue()}, b);
        }
      }

      b.create<circt::cmt2::ReturnOp>(loc);
    });

    rule->finalize();
    
    llvm::outs() << "[BBHandler] Generated rule for slot " << slot << " in block " << blockId << "\n";
  }
  
  llvm::outs() << "[BBHandler] Successfully generated " << slotOrder.size() << " rules for basic block " << blockId << "\n";
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

  // Check if this is a cross-slot FIFO read (only if operandIndex is valid)
  if (operandIndex != static_cast<unsigned>(-1)) {
    auto &crossSlotFIFOs = bbHandler->getCrossSlotFIFOs();
    llvm::outs() << "crossSlotFIFO size: " << crossSlotFIFOs.size() << "\n";
    auto it = crossSlotFIFOs.find(v);
    if (it != crossSlotFIFOs.end()) {
      // Check all FIFOs for this value to find the right one for this consumer
      for (CrossSlotFIFO *fifo : it->second) {
        for (auto [consumerOp, opIndex] : fifo->consumers) {
          if (consumerOp == currentOp && opIndex == operandIndex) {
            auto result = fifo->fifoInstance->callValue("deq", b);
            assert(result.size() == 1);
            localMap[v] = result[0];
            return result[0];
          }
        }
      }
    }
  }

  currentOp->emitError("value is not available in this rule");
  return failure();
}

} // namespace mlir