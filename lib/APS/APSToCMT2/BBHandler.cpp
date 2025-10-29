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
                    Instance *hellaMemInstance, Instance *regRdInstance,
                    InterfaceDecl *dmaItfc, Circuit &circuit, Clock mainClk, Reset mainRst,
                    unsigned long opcode)
    : pass(pass), mainModule(mainModule), funcOp(funcOp), poolInstance(poolInstance),
      roccInstance(roccInstance), hellaMemInstance(hellaMemInstance), dmaItfc(dmaItfc),
      circuit(circuit), mainClk(mainClk), mainRst(mainRst), opcode(opcode), regRdInstance(regRdInstance) {

  // Initialize operation generators
  arithmeticGen = std::make_unique<ArithmeticOpGenerator>(this);
  memoryGen = std::make_unique<MemoryOpGenerator>(this);
  interfaceGen = std::make_unique<InterfaceOpGenerator>(this);
  registerGen = std::make_unique<RegisterOpGenerator>(this);

  // Set up register generator with required instances (shared across all blocks)
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
        // IMPORTANT: Only include consumers that are in THIS block's slotMap
        llvm::DenseMap<int64_t, llvm::SmallVector<std::pair<Operation*, unsigned>>> consumersByStage;

        for (OpOperand &use : res.getUses()) {
          Operation *user = use.getOwner();
          auto maybeConsumerSlot = getSlotForOp(user);
          if (!maybeConsumerSlot)
            continue;
          int64_t consumerSlot = *maybeConsumerSlot;

          // Only consider consumers in later slots within the same block
          if (consumerSlot > slot) {
            // Check if this consumer operation is actually in the current block
            bool isInCurrentBlock = false;
            if (slotMap.count(consumerSlot)) {
              for (Operation *op : slotMap[consumerSlot].ops) {
                if (op == user) {
                  isInCurrentBlock = true;
                  break;
                }
              }
            }

            // Only create FIFO if consumer is in current block
            if (isInCurrentBlock) {
              consumersByStage[consumerSlot].push_back({user, use.getOperandNumber()});
            }
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
          fifo->instanceName = currentBlock->blockName + "_fifo_s" + std::to_string(slot) + "_s" +
                              std::to_string(consumerSlot) + "_v" + std::to_string(count);

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

  // Handle cross-block input values (from inputFIFOs)
  // These need cross-slot FIFOs from first slot to their consumer slots
  if (currentBlock && !currentBlock->input_fifos.empty()) {
    // Determine the first slot in this block
    int64_t firstSlot = slotOrder.empty() ? 0 : slotOrder.front();

    llvm::outs() << "[FIFO DEBUG] Processing " << currentBlock->input_fifos.size()
                 << " input FIFO values for cross-slot distribution (first slot: "
                 << firstSlot << ")\n";

    for (auto &[inputValue, inputFIFO] : currentBlock->input_fifos) {
      if (!inputFIFO)
        continue;

      if (!isa<mlir::IntegerType>(inputValue.getType()))
        continue;

      // Find all operations that use this input value and group by slot
      // IMPORTANT: Only include consumers that are in THIS block's slotMap
      llvm::DenseMap<int64_t, llvm::SmallVector<std::pair<Operation*, unsigned>>> consumersByStage;

      for (OpOperand &use : inputValue.getUses()) {
        Operation *user = use.getOwner();
        auto maybeConsumerSlot = getSlotForOp(user);
        if (!maybeConsumerSlot)
          continue;
        int64_t consumerSlot = *maybeConsumerSlot;

        // Check if this consumer operation is actually in the current block
        bool isInCurrentBlock = false;
        if (slotMap.count(consumerSlot)) {
          for (Operation *op : slotMap[consumerSlot].ops) {
            if (op == user) {
              isInCurrentBlock = true;
              break;
            }
          }
        }

        // Only create FIFO if consumer is in current block
        if (isInCurrentBlock) {
          consumersByStage[consumerSlot].push_back({user, use.getOperandNumber()});
        } else {
          llvm::outs() << "[FIFO DEBUG] Skipping consumer in slot " << consumerSlot
                       << " (belongs to different block)\n";
        }
      }

      if (consumersByStage.empty()) {
        llvm::outs() << "[FIFO DEBUG] Input value has no consumers in this block\n";
        continue;
      }

      auto firType = toFirrtlType(inputValue.getType(), ctx);
      if (!firType) {
        llvm::outs() << "[FIFO DEBUG] Input value type unsupported for FIFO\n";
        continue;
      }

      // Create cross-slot FIFO from first slot to each consumer slot
      // IMPORTANT: Skip if consumer is in the first slot (no cross-slot needed)
      for (auto &[consumerSlot, consumers] : consumersByStage) {
        // If consumer is in the first slot, no cross-slot FIFO needed
        if (consumerSlot == firstSlot) {
          llvm::outs() << "[FIFO DEBUG] Skipping input cross-slot FIFO for consumer in first slot "
                       << consumerSlot << " (no cross-slot needed)\n";
          continue;
        }

        auto fifo = std::make_unique<CrossSlotFIFO>();
        fifo->producerValue = inputValue;
        fifo->producerSlot = firstSlot;  // Input values are available at first slot
        fifo->consumerSlot = consumerSlot;
        fifo->firType = firType;
        fifo->consumers = std::move(consumers);

        // Generate FIFO name: {blockName}_fifo_s{firstSlot}_s{consumer}_v{count}
        auto key = std::make_pair(firstSlot, consumerSlot);
        unsigned count = fifoCounts[key]++;
        fifo->instanceName = currentBlock->blockName + "_fifo_s" + std::to_string(firstSlot) + "_s" +
                             std::to_string(consumerSlot) + "_v" + std::to_string(count);

        llvm::outs() << "[FIFO DEBUG] Creating input cross-slot FIFO: " << fifo->instanceName
                     << " for input value to consumers in slot " << consumerSlot
                     << " with " << fifo->consumers.size() << " consumers\n";

        crossSlotFIFOs[inputValue].push_back(fifo.get());
        fifoStorage.push_back(std::move(fifo));
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
    std::string tokenFifoName = currentBlock->blockName + "_token_fifo_s" + std::to_string(currentSlot);
    auto *tokenFifo = mainModule->addInstance(tokenFifoName, tokenFifoMod,
                                              {mainClk.getValue(), mainRst.getValue()});
    stageTokenFifos[currentSlot] = tokenFifo;
    llvm::outs() << "[BBHandler] Created token FIFO for slot " << currentSlot << ": " << tokenFifoName << "\n";
  }

  return success();
}

LogicalResult BBHandler::instantiateCrossSlotFIFOs() {
  // Instantiate FIFO modules for cross-slot data communication
  // This must be called AFTER buildCrossSlotFIFOs() creates the metadata
  // and BEFORE rules try to use the FIFOs

  auto &builder = mainModule->getBuilder();
  auto savedIP = builder.saveInsertionPoint();

  llvm::outs() << "[BBHandler] Instantiating " << fifoStorage.size() << " cross-slot FIFO modules\n";

  for (auto &fifoPtr : fifoStorage) {
    auto *fifo = fifoPtr.get();
    int64_t width = cast<circt::firrtl::UIntType>(fifo->firType).getWidthOrSentinel();
    if (width < 0)
      width = 1;

    // Debug info: log FIFO instantiation
    llvm::outs() << "[BBHandler] Instantiating FIFO: " << fifo->instanceName
                 << " (width=" << width << ")\n";

    // Create FIFO module with proper clock and reset
    auto *fifoMod = STLLibrary::createFIFO1PushModule(width, circuit);
    builder.restoreInsertionPoint(savedIP);
    fifo->fifoInstance = mainModule->addInstance(fifo->instanceName, fifoMod,
                                                 {mainClk.getValue(), mainRst.getValue()});
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

  // Note: instantiateCrossSlotFIFOs() is called by processBasicBlock() before rule generation
  // This old code path (generateSlotRules) is deprecated in favor of processBasicBlock()

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

LogicalResult BBHandler::processBasicBlock(BlockInfo& block) {
  llvm::outs() << "[BBHandler] Processing basic block " << block.blockId
               << " (" << block.blockName << ") with "
               << block.mlirBlock->getOperations().size() << " operations\n";

  // Store block reference for use throughout the handler
  currentBlock = &block;

  unsigned blockId = block.blockId;
  llvm::DenseMap<Value, Instance*> &inputFIFOs = block.input_fifos;
  llvm::DenseMap<Value, Instance*> &outputFIFOs = block.output_fifos;
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

  llvm::outs() << "[BBHandler] Collected " << blockOperations.size() << " operations from block segment (out of "
               << block.operations.size() << " total in segment)\n";

  // PANIC: Empty blocks should not reach BBHandler
  if (blockOperations.empty()) {
    llvm::report_fatal_error("BBHandler received empty block - this should have been handled by BlockHandler");
  }

  // Phase 2: Organize operations by time slots
  if (failed(collectOperationsFromList(blockOperations))) {
    llvm::outs() << "[BBHandler] Failed to organize operations by slot\n";
    return failure();
  }

  // Phase 3: Set up infrastructure for this specific block
  // Build cross-slot FIFOs for values that flow between slots within this block
  if (failed(buildCrossSlotFIFOs()))
    return failure();

  // Create token FIFOs for stage synchronization
  if (failed(createTokenFIFOs()))
    return failure();

  // Instantiate FIFO modules for cross-slot communication
  if (failed(instantiateCrossSlotFIFOs()))
    return failure();

  // Phase 4: Generate rules for each time slot with proper coordination
  llvm::DenseMap<mlir::Value, mlir::Value> localMap;
  
  for (int64_t slot : slotOrder) {
    auto *rule = mainModule->addRule(currentBlock->blockName + "_slot_" + std::to_string(slot) + "_rule");

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
        if (block_input_token_fifo) {
          llvm::outs() << "[BBHandler] Dequeuing input token for block " << blockId << " from unified token FIFO\n";
          auto inputToken = block_input_token_fifo->callMethod("deq", {}, b);
        }
        
        // Handle cross-block value consumption (data from other blocks)
        // Per BBHandler_DataFlow.md: dequeue from input_fifos and distribute to:
        // 1. localMap if used in slot 0 (for immediate use by slot 0 operations)
        // 2. cross_slot_fifos if used in later slots (for deferred use)
        // Same dequeued value can go to BOTH destinations
        for (auto &[value, fifo] : inputFIFOs) {
          if (fifo) {
            llvm::outs() << "[BBHandler] Dequeuing cross-block value from input FIFO\n";
            auto dequeuedValue = fifo->callMethod("deq", {}, b);
            if (!dequeuedValue.empty()) {
              bool usedInSlot0 = false;
              bool usedInLaterSlots = false;

              // Check if value is used in slot 0 operations
              if (slotMap.count(slot)) {
                for (Operation *op : slotMap[slot].ops) {
                  for (Value operand : op->getOperands()) {
                    if (operand == value) {
                      usedInSlot0 = true;
                      break;
                    }
                  }
                  if (usedInSlot0) break;
                }
              }

              // If used in slot 0, store in localMap for immediate use
              if (usedInSlot0) {
                localMap[value] = dequeuedValue[0];
                llvm::outs() << "[BBHandler] Stored input value in localMap for slot 0 use\n";
              }

              // Check if value has cross-slot FIFOs (means used in later slots)
              auto it = crossSlotFIFOs.find(value);
              if (it != crossSlotFIFOs.end()) {
                usedInLaterSlots = true;
                // Enqueue to all cross-slot FIFOs for this value (one per consumer slot)
                for (CrossSlotFIFO *crossSlotFifo : it->second) {
                  if (crossSlotFifo->fifoInstance) {
                    crossSlotFifo->fifoInstance->callMethod("enq", {dequeuedValue[0]}, b);
                    llvm::outs() << "[BBHandler] Enqueued input value to cross-slot FIFO: "
                                 << crossSlotFifo->instanceName << " for later slot use\n";
                  }
                }
              }

              if (!usedInSlot0 && !usedInLaterSlots) {
                llvm::outs() << "[BBHandler] WARNING: Input value not used in any slot of this block\n";
              }
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

      // Handle cross-slot FIFO writes and cross-block output FIFO writes for producer operations in this slot
      for (Operation *op : slotMap[slot].ops) {
        for (mlir::Value result : op->getResults()) {
          if (!isa<mlir::IntegerType>(result.getType()))
            continue;

          auto valueIt = localMap.find(result);
          if (valueIt == localMap.end())
            continue;

          // Enqueue to cross-slot FIFOs (for later slots in same block)
          auto fifoIt = crossSlotFIFOs.find(result);
          if (fifoIt != crossSlotFIFOs.end()) {
            for (CrossSlotFIFO *fifo : fifoIt->second) {
              if (fifo->fifoInstance) {
                fifo->fifoInstance->callMethod("enq", {valueIt->second}, b);
                llvm::outs() << "[BBHandler] Enqueued value to cross-slot FIFO\n";
              }
            }
          }

          // Enqueue to cross-block output FIFOs (for other blocks)
          auto outputIt = outputFIFOs.find(result);
          if (outputIt != outputFIFOs.end() && outputIt->second) {
            outputIt->second->callMethod("enq", {valueIt->second}, b);
            llvm::outs() << "[BBHandler] Enqueued value to cross-block output FIFO\n";
          }
        }
      }

      // Write token to next stage at the end (intra-block coordination)
      if (failed(writeTokenToNextStage(b, loc, slot)))
        return;

      // Handle block coordination at the end of the last slot
      if (slot == slotOrder.back()) {
        // Note: Cross-block output values are now enqueued immediately when produced (see above)
        // Only need to enqueue the completion token here

        // Enqueue output token to unified token FIFO (token to next block in sequence)
        if (block_output_token_fifo) {
          llvm::outs() << "[BBHandler] Enqueuing output token for block " << blockId << " to unified token FIFO\n";
          auto outputToken = UInt::constant(1, 1, b, loc);
          block_output_token_fifo->callMethod("enq", {outputToken.getValue()}, b);
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
    auto it = crossSlotFIFOs.find(v);
    if (it != crossSlotFIFOs.end()) {
      // Check all FIFOs for this value to find the right one for this consumer
      for (CrossSlotFIFO *fifo : it->second) {
        // Check if FIFO instance is valid (should have been created in generateSlotRules)
        if (!fifo->fifoInstance) {
          currentOp->emitError("cross-slot FIFO instance is null - FIFO not instantiated");
          return failure();
        }

        for (auto [consumerOp, opIndex] : fifo->consumers) {
          if (consumerOp == currentOp) {
            llvm::outs() << "Index available: " << opIndex << '\n';
          }
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