//===- BBHandler.cpp - Basic Block Handler Implementation ------------------===//
//
// This file implements the object-oriented basic block handling for TOR
// function rule generation
//
//===----------------------------------------------------------------------===//

#include "APS/BBHandler.h"
#include "APS/APSOps.h"
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
  auto insertOpIntoSlot = [&](Operation *op, int64_t slot) {
    auto &info = slotMap[slot];
    info.ops.push_back(op);
  };

  for (Operation &op : funcOp.getBody().getOps()) {
    if (isa<tor::TimeGraphOp>(op) || isa<tor::ReturnOp>(op))
      continue;

    if (auto startAttr = op.getAttrOfType<IntegerAttr>("starttime")) {
      int64_t slot = startAttr.getInt();
      insertOpIntoSlot(&op, slot);
      continue;
    }

    if (isa<arith::ConstantOp>(op))
      continue;

    op.emitError("missing required 'starttime' attribute for rule generation");
    return failure();
  }

  // Populate sorted slot order
  for (auto &kv : slotMap)
    slotOrder.push_back(kv.first);
  llvm::sort(slotOrder);

  return success();
}

LogicalResult BBHandler::validateOperations() {
  for (int64_t slot : slotOrder) {
    for (Operation *op : slotMap[slot].ops) {
      if (isa<arith::ConstantOp, memref::GetGlobalOp>(op))
        continue;
      if (isa<tor::AddIOp, tor::SubIOp, tor::MulIOp>(op))
        continue;
      if (isa<aps::MemLoad, aps::MemStore, aps::CpuRfRead, aps::CpuRfWrite>(op))
        continue;
      if (isa<aps::ItfcLoadReq, aps::ItfcLoadCollect>(op))
        continue;
      if (isa<aps::ItfcStoreReq, aps::ItfcStoreCollect>(op))
        continue;
      if (isa<aps::ItfcBurstLoadReq, aps::ItfcBurstLoadCollect>(op))
        continue;
      if (isa<aps::ItfcBurstStoreReq, aps::ItfcBurstStoreCollect>(op))
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
  llvm::DenseMap<int64_t, Instance*> stageTokenFifos;
  for (size_t i = 0; i < slotOrder.size() - 1; ++i) {
    int64_t currentSlot = slotOrder[i];
    // Create 1-bit FIFO for token passing to next stage
    auto *tokenFifoMod = STLLibrary::createFIFO1PushModule(1, circuit);
    mainModule->getBuilder().restoreInsertionPoint(savedIP);
    std::string tokenFifoName = std::to_string(opcode) + "_token_fifo_s" + std::to_string(currentSlot);
    auto *tokenFifo = mainModule->addInstance(tokenFifoName, tokenFifoMod,
                                              {mainClk.getValue(), mainRst.getValue()});
    stageTokenFifos[currentSlot] = tokenFifo;
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
                llvm::SmallVector<mlir::Value> args = {valueIt->second};
                fifo->fifoInstance->callMethod("enq", args, b);
              }
            }
          }
        }
      }

      b.create<circt::cmt2::ReturnOp>(loc);
    });

    rule->finalize();

    // Add precedence constraints - find previous slot in order
    if (!slotOrder.empty() && slot != slotOrder[0]) {
      auto it = std::find(slotOrder.begin(), slotOrder.end(), slot);
      if (it != slotOrder.begin()) {
        int64_t prevSlot = *(it - 1);
        std::string from = std::to_string(opcode) + "_rule_s" + std::to_string(slot);
        std::string to = std::to_string(opcode) + "_rule_s" + std::to_string(prevSlot);
        precedencePairs.emplace_back(from, to);
      }
    }
  }

  if (!precedencePairs.empty())
    mainModule->setPrecedence(precedencePairs);

  return success();
}

LogicalResult BBHandler::handleRoCCCommandBundle(mlir::OpBuilder &b, Location loc) {
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
    auto tokenFifoIt = stageTokenFifos.find(slot);
    if (tokenFifoIt != stageTokenFifos.end()) {
      auto *tokenFifo = tokenFifoIt->second;
      auto tokenValue = UInt::constant(1, 1, b, loc);
      tokenFifo->callMethod("enq", {tokenValue.getValue()}, b);
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

mlir::Type BBHandler::toFirrtlType(mlir::Type type, mlir::MLIRContext *ctx) {
  if (auto intType = dyn_cast<mlir::IntegerType>(type)) {
    if (intType.isUnsigned())
      return circt::firrtl::UIntType::get(ctx, intType.getWidth());
    if (intType.isSigned())
      return circt::firrtl::SIntType::get(ctx, intType.getWidth());
    return circt::firrtl::UIntType::get(ctx, intType.getWidth());
  }
  return {};
}

uint32_t BBHandler::roundUpToPowerOf2(uint32_t value) {
  if (value <= 1)
    return 1;
  value--;
  value |= value >> 1;
  value |= value >> 2;
  value |= value >> 4;
  value |= value >> 8;
  value |= value >> 16;
  value++;
  return value;
}

//===----------------------------------------------------------------------===//
// OperationGenerator Base Implementation
//===----------------------------------------------------------------------===//

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