//===- RuleGeneration.cpp - Rule Generation for TOR Functions -------------===//
//
// This file implements the rule generation functionality for TOR functions
// that was previously in APSToCMT2GenPass.cpp
//
//===----------------------------------------------------------------------===//

#include "APS/APSToCMT2.h"

namespace mlir {

using namespace mlir;
using namespace mlir::tor;
using namespace circt::cmt2::ecmt2;
using namespace circt::cmt2::ecmt2::stl;
using namespace circt::firrtl;

uint32_t APSToCMT2GenPass::roundUpToPowerOf2(uint32_t value) {
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

/// Convert MLIR type to FIRRTL type
mlir::Type APSToCMT2GenPass::toFirrtlType(Type type, MLIRContext *ctx) {
  if (auto intType = dyn_cast<mlir::IntegerType>(type)) {
    if (intType.isUnsigned())
      return circt::firrtl::UIntType::get(ctx, intType.getWidth());
    if (intType.isSigned())
      return circt::firrtl::SIntType::get(ctx, intType.getWidth());
    return circt::firrtl::UIntType::get(ctx, intType.getWidth());
  }
  return {};
}

//===----------------------------------------------------------------------===//
// Main Rule Generation Function
//===----------------------------------------------------------------------===//

/// Generate rules for a specific TOR function - proper implementation from
/// rulegenpass.cpp
void APSToCMT2GenPass::generateRulesForFunction(
    Module *mainModule, tor::FuncOp funcOp, Instance *poolInstance,
    Instance *roccInstance, Instance *hellaMemInstance, InterfaceDecl *dmaItfc,
    Circuit &circuit, Clock mainClk, Reset mainRst, unsigned long opcode) {

  auto &builder = mainModule->getBuilder();
  MLIRContext *ctx = builder.getContext();

  // Collect operations per time slot.
  llvm::DenseMap<int64_t, SlotInfo> slotMap;
  SmallVector<int64_t, 8> slotOrder;

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
    return;
  }

  // Populate sorted slot order and validate supported ops.
  for (auto &kv : slotMap)
    slotOrder.push_back(kv.first);
  llvm::sort(slotOrder);

  for (int64_t slot : slotOrder) {
    for (Operation *op : slotMap[slot].ops) {
      if (isa<arith::ConstantOp, memref::GetGlobalOp>(op))
        continue;
      if (isa<tor::AddIOp, tor::SubIOp, tor::MulIOp>(op))
        continue;
      if (isa<aps::MemLoad, aps::CpuRfRead, aps::CpuRfWrite>(op))
        continue;
      if (isa<aps::ItfcLoadReq, aps::ItfcLoadCollect>(op))
        continue;
      if (isa<aps::ItfcStoreReq, aps::ItfcStoreCollect>(op))
        continue;
      if (isa<aps::ItfcBurstLoadReq, aps::ItfcBurstLoadCollect>(op))
        continue;
      if (isa<aps::ItfcBurstStoreReq, aps::ItfcBurstStoreCollect>(op))
        continue;
      op->emitError("unsupported operation for rule generation");
      return;
    }
  }

  // Analyze cross-slot value uses - create separate FIFOs for each
  // producer-consumer stage pair
  llvm::DenseMap<mlir::Value, llvm::SmallVector<CrossSlotFIFO *>>
      crossSlotFIFOs;
  llvm::DenseMap<std::pair<int64_t, int64_t>, unsigned> fifoCounts;
  SmallVector<std::unique_ptr<CrossSlotFIFO>, 8> fifoStorage;

  auto getSlotForOp = [&](Operation *op) -> std::optional<int64_t> {
    if (auto attr = op->getAttrOfType<IntegerAttr>("starttime"))
      return attr.getInt();
    return {};
  };

  // Build cross-slot FIFO mapping - group consumers by target stage
  for (int64_t slot : slotOrder) {
    for (Operation *op : slotMap[slot].ops) {
      if (isa<arith::ConstantOp>(op))
        continue;

      for (mlir::Value res : op->getResults()) {
        if (!isa<mlir::IntegerType>(res.getType()))
          continue;

        // Group consumers by their target stage
        llvm::DenseMap<int64_t,
                       llvm::SmallVector<std::pair<Operation *, unsigned>>>
            consumersByStage;

        for (OpOperand &use : res.getUses()) {
          Operation *user = use.getOwner();
          auto maybeConsumerSlot = getSlotForOp(user);
          if (!maybeConsumerSlot)
            continue;
          int64_t consumerSlot = *maybeConsumerSlot;
          if (consumerSlot > slot) {
            consumersByStage[consumerSlot].push_back(
                {user, use.getOperandNumber()});
          }
        }

        if (consumersByStage.empty())
          continue;

        auto firType = toFirrtlType(res.getType(), ctx);
        if (!firType) {
          op->emitError("type is unsupported for rule lowering");
          return;
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
          fifo->instanceName = std::to_string(opcode) + "_fifo_s" +
                               std::to_string(slot) + "_s" +
                               std::to_string(consumerSlot);
          if (count > 0)
            fifo->instanceName += "_" + std::to_string(count);

          // Debug info: log FIFO creation
          llvm::outs() << "[FIFO DEBUG] Creating FIFO: " << fifo->instanceName
                       << " for producer in slot " << slot
                       << " to consumers in slot " << consumerSlot << " with "
                       << fifo->consumers.size() << " consumers\n";

          crossSlotFIFOs[res].push_back(fifo.get());
          fifoStorage.push_back(std::move(fifo));
        }
      }
    }
  }

  auto savedIP = builder.saveInsertionPoint();
  // Instantiate FIFO modules for cross-slot communication
  llvm::outs() << "[FIFO DEBUG] Instantiating " << fifoStorage.size()
               << " FIFO modules\n";
  for (auto &fifoPtr : fifoStorage) {
    auto *fifo = fifoPtr.get();
    int64_t width =
        cast<circt::firrtl::UIntType>(fifo->firType).getWidthOrSentinel();
    if (width < 0)
      width = 1;

    // Debug info: log FIFO instantiation
    llvm::outs() << "[FIFO DEBUG] Instantiating FIFO: " << fifo->instanceName
                 << " (width=" << width << ")\n";

    // Create FIFO module with proper clock and reset
    auto *fifoMod = STLLibrary::createFIFO1PushModule(width, circuit);
    builder.restoreInsertionPoint(savedIP);
    fifo->fifoInstance = mainModule->addInstance(
        fifo->instanceName, fifoMod, {mainClk.getValue(), mainRst.getValue()});
  }

  // Create token FIFOs for stage synchronization - one token per stage (except
  // last)
  llvm::DenseMap<int64_t, Instance *> stageTokenFifos;
  for (size_t i = 0; i < slotOrder.size() - 1; ++i) {
    int64_t currentSlot = slotOrder[i];
    // Create 1-bit FIFO for token passing to next stage
    auto *tokenFifoMod = STLLibrary::createFIFO1PushModule(1, circuit);
    builder.restoreInsertionPoint(savedIP);
    std::string tokenFifoName =
        std::to_string(opcode) + "_token_fifo_s" + std::to_string(currentSlot);
    auto *tokenFifo = mainModule->addInstance(
        tokenFifoName, tokenFifoMod, {mainClk.getValue(), mainRst.getValue()});
    stageTokenFifos[currentSlot] = tokenFifo;
  }

  // Helper to materialise values inside a rule.
  auto getValueInRule = [&](mlir::Value v, Operation *currentOp,
                            unsigned operandIndex, mlir::OpBuilder &builder,
                            DenseMap<mlir::Value, mlir::Value> &localMap,
                            Location opLoc) -> FailureOr<mlir::Value> {
    if (auto it = localMap.find(v); it != localMap.end())
      return it->second;

    if (auto constOp = v.getDefiningOp<arith::ConstantOp>()) {
      auto intAttr = mlir::cast<IntegerAttr>(constOp.getValueAttr());
      unsigned width = mlir::cast<IntegerType>(intAttr.getType()).getWidth();
      auto constant = UInt::constant(intAttr.getValue().getZExtValue(), width,
                                     builder, opLoc)
                          .getValue();
      localMap[v] = constant;
      return constant;
    }

    if (auto globalOp = v.getDefiningOp<memref::GetGlobalOp>()) {
      // Global symbols are handled separately via symbol resolution.
      return mlir::Value{};
    }

    // Check if this is a cross-slot FIFO read (only if operandIndex is valid)
    if (operandIndex != static_cast<unsigned>(-1)) {
      auto it = crossSlotFIFOs.find(v);
      if (it != crossSlotFIFOs.end()) {
        // Check all FIFOs for this value to find the right one for this
        // consumer
        for (CrossSlotFIFO *fifo : it->second) {
          // Check if this operation is a consumer of this FIFO
          for (auto [consumerOp, opIndex] : fifo->consumers) {
            if (consumerOp == currentOp && opIndex == operandIndex) {
              auto result = fifo->fifoInstance->callValue("deq", builder);
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
  };

  // Generate rules for each time slot.
  llvm::DenseMap<mlir::Value, mlir::Value> localMap;
  llvm::SmallVector<std::pair<std::string, std::string>, 4> precedencePairs;

  // Cache for RoCC command bundle (pre-read in slot 0)
  mlir::Value cachedRoCCCmdBundle;

  auto savedIPforRegRd = builder.saveInsertionPoint();
  auto *regRdMod = STLLibrary::createRegModule(5, 0, circuit);
  auto *regRdInstance =
      mainModule->addInstance("reg_rd_" + std::to_string(opcode), regRdMod,
                              {mainClk.getValue(), mainRst.getValue()});
  builder.restoreInsertionPoint(savedIPforRegRd);

  // Unified arithmetic operation helper to eliminate redundancy
  auto performArithmeticOp =
      [&](mlir::OpBuilder &b, Location loc, mlir::Value lhs, mlir::Value rhs,
          mlir::Value result, StringRef opName) -> LogicalResult {
    // Determine result width based on operation type
    auto requiredWidth = cast<IntegerType>(result.getType()).getWidth();

    // Perform the arithmetic operation using Signal abstraction
    Signal lhsSignal(lhs, &b, loc);
    Signal rhsSignal(rhs, &b, loc);

    Signal resultSignal(lhs, &b, loc); // dummy init
    if (opName == "add") {
      resultSignal = lhsSignal + rhsSignal;
    } else if (opName == "sub") {
      resultSignal = lhsSignal - rhsSignal;
    } else if (opName == "mul") {
      resultSignal = lhsSignal * rhsSignal;
    } else {
      return failure();
    }

    auto firrtlWidth = resultSignal.getWidth();
    Signal resultSignalWidthFix = resultSignal;
    if (firrtlWidth > requiredWidth) {
      resultSignalWidthFix = resultSignal.bits(requiredWidth - 1, 0);
    } else if (firrtlWidth < requiredWidth) {
      resultSignalWidthFix = resultSignal.pad(requiredWidth);
    }

    localMap[result] = resultSignalWidthFix.getValue();
    return success();
  };

  // auto trueConst = builder.create<ConstantOp>(mainModule->getLoc(), u64Type,
  // llvm::APInt(64, 1927)); auto builder2 = mainModule->getBuilder();
  // builder2.create<ConstantOp>(mainModule->getLoc(), u64Type, llvm::APInt(64,
  // 1949));
  for (int64_t slot : slotOrder) {
    auto *rule = mainModule->addRule(std::to_string(opcode) + "_rule_s" +
                                     std::to_string(slot));

    // Guard: always ready (constant 1)
    rule->guard([](mlir::OpBuilder &b) {
      auto loc = b.getUnknownLoc();
      auto one = UInt::constant(1, 1, b, loc);
      b.create<circt::cmt2::ReturnOp>(loc, mlir::ValueRange{one.getValue()});
    });

    // Body: implement the operations for this time slot
    rule->body([&](mlir::OpBuilder &b) {
      auto loc = b.getUnknownLoc();
      localMap.clear();

      // Pre-read RoCC command bundle in slot 0
      if (slot == 0) {
        // Call cmd_to_user once to get the RoCC command bundle
        std::string cmdMethod = "cmd_to_user_" + std::to_string(opcode);
        auto cmdResult = roccInstance->callMethod(cmdMethod, {}, b)[0];
        cachedRoCCCmdBundle = cmdResult;
        auto instruction = Bundle(cachedRoCCCmdBundle, &b, loc);
        regRdInstance->callMethod("write", {instruction["rd"].getValue()}, b);
      }

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

      for (Operation *op : slotMap[slot].ops) {
        // Handle arithmetic operations using unified helper
        if (auto addOp = dyn_cast<tor::AddIOp>(op)) {
          auto lhs = getValueInRule(addOp.getLhs(), op, 0, b, localMap, loc);
          auto rhs = getValueInRule(addOp.getRhs(), op, 1, b, localMap, loc);
          if (failed(lhs) || failed(rhs))
            return;
          if (failed(performArithmeticOp(b, loc, *lhs, *rhs, addOp.getResult(),
                                         "add")))
            return;
        } else if (auto subOp = dyn_cast<tor::SubIOp>(op)) {
          auto lhs = getValueInRule(subOp.getLhs(), op, 0, b, localMap, loc);
          auto rhs = getValueInRule(subOp.getRhs(), op, 1, b, localMap, loc);
          if (failed(lhs) || failed(rhs))
            return;
          if (failed(performArithmeticOp(b, loc, *lhs, *rhs, subOp.getResult(),
                                         "sub")))
            return;
        } else if (auto mulOp = dyn_cast<tor::MulIOp>(op)) {
          auto lhs = getValueInRule(mulOp.getLhs(), op, 0, b, localMap, loc);
          auto rhs = getValueInRule(mulOp.getRhs(), op, 1, b, localMap, loc);
          if (failed(lhs) || failed(rhs))
            return;
          if (failed(performArithmeticOp(b, loc, *lhs, *rhs, mulOp.getResult(),
                                         "mul")))
            return;
        } else if (auto readRf = dyn_cast<aps::CpuRfRead>(op)) {
          // Handle CPU register file read operations
          // PANIC if not in first time slot
          if (slot != 0) {
            op->emitError("aps.readrf operations must appear only in the first "
                          "time slot (slot 0), but found in slot ")
                << slot;
            llvm::report_fatal_error("readrf must be in first time slot");
          }
          // Get the function argument that this readrf is reading from
          Value regArg = readRf.getRs();

          // Map function arguments to rs1/rs2 based on their position in the
          // function Need to find which argument index this corresponds to
          int64_t argIndex = -1;
          for (unsigned i = 0; i < funcOp.getNumArguments(); ++i) {
            if (funcOp.getArgument(i) == regArg) {
              argIndex = i;
              break;
            }
          }
          auto instruction = Bundle(cachedRoCCCmdBundle, &b, loc);
          Signal regValue =
              argIndex == 0 ? instruction["rs1data"] : instruction["rs2data"];
          // // Store the result with a note about which argument it came from
          // llvm::outs() << "DEBUG: readrf from arg" << argIndex << " using
          // cached RoCC bundle\n";
          localMap[readRf.getResult()] = regValue.getValue();
        } else if (auto writeRf = dyn_cast<aps::CpuRfWrite>(op)) {
          auto rdvalue =
              getValueInRule(writeRf.getValue(), op, 1, b, localMap, loc);
          // auto rd = UInt::constant(1, 5, b, loc);
          auto rd = regRdInstance->callValue("read", b)[0];
          llvm::SmallVector<mlir::Value> bundleFields = {rd, *rdvalue};
          auto bundleType = BundleType::get(
              b.getContext(),
              {
                  BundleType::BundleElement{b.getStringAttr("rd"), false,
                                            UIntType::get(b.getContext(), 5)},
                  BundleType::BundleElement{b.getStringAttr("rddata"), false,
                                            UIntType::get(b.getContext(), 32)},
              });

          auto bundleValue =
              b.create<BundleCreateOp>(loc, bundleType, bundleFields);
          roccInstance->callMethod("resp_from_user", {bundleValue}, b);
          // writerf doesn't produce a result, just performs the write
        } else if (auto itfcLoadReq = dyn_cast<aps::ItfcLoadReq>(op)) {
          // Handle CPU interface load request operations
          // This sends a load request to memory and returns a request token
          auto context = b.getContext();

          auto userCmdBundleType = BundleType::get(
              context,
              {BundleType::BundleElement{b.getStringAttr("addr"), false,
                                         UIntType::get(context, 32)},
               BundleType::BundleElement{b.getStringAttr("cmd"), false,
                                         UIntType::get(context, 1)},
               BundleType::BundleElement{b.getStringAttr("size"), false,
                                         UIntType::get(context, 2)},
               BundleType::BundleElement{b.getStringAttr("data"), false,
                                         UIntType::get(context, 32)},
               BundleType::BundleElement{b.getStringAttr("mask"), false,
                                         UIntType::get(context, 4)},
               BundleType::BundleElement{b.getStringAttr("tag"), false,
                                         UIntType::get(context, 8)}});

          // Get address from indices (similar to MemLoad)
          if (itfcLoadReq.getIndices().empty()) {
            op->emitError(
                "Interface load request must have at least one index");
          }

          auto addr = getValueInRule(itfcLoadReq.getIndices()[0], op, 1, b,
                                     localMap, loc);
          if (failed(addr)) {
            op->emitError("Failed to get address for interface load request");
          }
          auto addrSignal = Signal(*addr, &b, loc);
          auto readCmd = UInt::constant(0, 1, b, loc);
          auto data = UInt::constant(0, 32, b, loc);
          auto size = UInt::constant(2, 2, b, loc);
          auto mask = UInt::constant(0, 4, b, loc);
          auto tagConst = UInt::constant(0x3f, 8, b, loc);
          auto tag = (addrSignal ^ tagConst).bits(7, 0);

          llvm::SmallVector<mlir::Value> bundleFields = {
              *addr,           readCmd.getValue(), size.getValue(),
              data.getValue(), mask.getValue(),    tag.getValue()};
          auto bundleValue =
              b.create<BundleCreateOp>(loc, userCmdBundleType, bundleFields);
          hellaMemInstance->callMethod("cmd_from_user", {bundleValue}, b);

          localMap[itfcLoadReq.getResult()] =
              UInt::constant(1, 1, b, loc).getValue();
        } else if (auto itfcLoadCollect = dyn_cast<aps::ItfcLoadCollect>(op)) {
          auto resp = hellaMemInstance->callMethod("resp_to_user", {}, b)[0];
          auto respBundle = Bundle(resp, &b, loc);

          auto loadResult = respBundle["data"];

          localMap[itfcLoadCollect.getResult()] = loadResult.getValue();
        } else if (auto itfcStoreReq = dyn_cast<aps::ItfcStoreReq>(op)) {
          // Handle CPU interface store request operations
          // This sends a store request to memory and returns a request token

          // Get address from indices (value is first operand, address is from
          // indices)
          auto context = b.getContext();
          if (itfcStoreReq.getIndices().empty()) {
            op->emitError(
                "Interface store request must have at least one index");
          }
          auto value =
              getValueInRule(itfcStoreReq.getValue(), op, 0, b, localMap, loc);
          auto addr = getValueInRule(itfcStoreReq.getIndices()[0], op, 2, b,
                                     localMap, loc);
          if (failed(value) || failed(addr)) {
            op->emitError(
                "Failed to get value or address for interface store request");
          }

          auto userCmdBundleType = BundleType::get(
              context,
              {BundleType::BundleElement{b.getStringAttr("addr"), false,
                                         UIntType::get(context, 32)},
               BundleType::BundleElement{b.getStringAttr("cmd"), false,
                                         UIntType::get(context, 1)},
               BundleType::BundleElement{b.getStringAttr("size"), false,
                                         UIntType::get(context, 2)},
               BundleType::BundleElement{b.getStringAttr("data"), false,
                                         UIntType::get(context, 32)},
               BundleType::BundleElement{b.getStringAttr("mask"), false,
                                         UIntType::get(context, 4)},
               BundleType::BundleElement{b.getStringAttr("tag"), false,
                                         UIntType::get(context, 8)}});

          if (failed(addr)) {
            op->emitError("Failed to get address for interface load request");
          }
          auto addrSignal = Signal(*addr, &b, loc);
          auto WriteCmd = UInt::constant(1, 1, b, loc);
          auto size = UInt::constant(2, 2, b, loc);
          auto mask = UInt::constant(0, 4, b, loc);
          auto tagConst = UInt::constant(0x3f, 8, b, loc);
          auto tag = (addrSignal ^ tagConst).bits(7, 0);

          llvm::SmallVector<mlir::Value> bundleFields = {
              *addr,  WriteCmd.getValue(), size.getValue(),
              *value, mask.getValue(),     tag.getValue()};
          auto bundleValue =
              b.create<BundleCreateOp>(loc, userCmdBundleType, bundleFields);
          hellaMemInstance->callMethod("cmd_from_user", {bundleValue}, b);

          localMap[itfcStoreReq.getResult()] =
              UInt::constant(1, 1, b, loc).getValue();
        } else if (auto itfcStoreCollect =
                       dyn_cast<aps::ItfcStoreCollect>(op)) {
          // Do nothing here, don't reply...
        } else if (auto memLoad = dyn_cast<aps::MemLoad>(op)) {
          // Handle memory load operations
          // PANIC if no indices provided
          if (memLoad.getIndices().empty()) {
            op->emitError("Memory load operation must have at least one index");
            llvm::report_fatal_error("Memory load requires address indices");
          }

          auto addr =
              getValueInRule(memLoad.getIndices()[0], op, 0, b, localMap, loc);
          if (failed(addr)) {
            op->emitError("Failed to get address for memory load");
            llvm::report_fatal_error("Memory load address resolution failed");
          }

          // Get the memory reference and check if it comes from
          // memref.get_global
          Value memRef = memLoad.getMemref();
          Operation *defOp = memRef.getDefiningOp();

          std::string memoryBankRule;
          if (auto getGlobalOp = dyn_cast<memref::GetGlobalOp>(defOp)) {
            // Extract the global symbol name and build the rule name
            StringRef globalName = getGlobalOp.getName();
            // Convert @mem_a_0 -> mem_a_0_read
            memoryBankRule = (globalName.drop_front() + "_read")
                                 .str(); // Remove @ prefix and add _read
            llvm::outs() << "DEBUG: Memory load from global " << globalName
                         << " using rule " << memoryBankRule << "\n";
          }

          // Call the appropriate scratchpad pool bank read method
          auto callResult = b.create<circt::cmt2::CallOp>(
              loc, circt::firrtl::UIntType::get(b.getContext(), 64),
              mlir::ValueRange{*addr},
              mlir::SymbolRefAttr::get(b.getContext(), "scratchpad_pool"),
              mlir::SymbolRefAttr::get(b.getContext(), memoryBankRule),
              mlir::ArrayAttr(), mlir::ArrayAttr());

          localMap[memLoad.getResult()] = callResult.getResult(0);
        } else if (auto memBurstLoadReq = dyn_cast<aps::ItfcBurstLoadReq>(op)) {
          Value cpuAddr = memBurstLoadReq.getCpuAddr();
          Value memRef = memBurstLoadReq.getMemrefs()[0];
          Value start = memBurstLoadReq.getStart();
          Value numOfElements = memBurstLoadReq.getLength();

          // Get the global memory reference name
          auto getGlobalOp =
              dyn_cast<memref::GetGlobalOp>(memRef.getDefiningOp());
          auto globalName = getGlobalOp.getName().str();

          // Find the corresponding memory entry using the pre-built map
          MemoryEntryInfo *targetMemEntry = nullptr;
          if (!globalName.empty()) {
            auto it = memEntryMap.find(globalName);
            if (it != memEntryMap.end()) {
              targetMemEntry = &it->second;
            }
          }

          // Get element type and size from the memory entry info or global
          // memref
          int elementSizeBytes = 1; // Default to 1 byte
          if (targetMemEntry) {
            // Use the data width from the memory entry info
            elementSizeBytes = (targetMemEntry->dataWidth + 7) /
                               8; // Convert bits to bytes, rounding up
          } else if (!globalName.empty()) {
            op->emitError("Failed to find target memory entry!");
          }

          // Calculate localAddr: baseAddress + (start * numOfElements *
          // elementSizeBytes)
          uint32_t baseAddress = targetMemEntry->baseAddress;

          // Calculate offset: start * numOfElements * elementSizeBytes
          // First multiply start * numOfElements
          auto baseAddrConst = UInt::constant(baseAddress, 32, b, loc);
          auto startSig = Signal(start, &b, loc);
          auto elementSizeBytesConst =
              UInt::constant(32, elementSizeBytes, b, loc);
          Signal localAddr = baseAddrConst + startSig * elementSizeBytesConst;

          // Calculate total burst length: elementSizeBytes * numOfElements,
          // rounded up to nearest power of 2
          auto numElementsOp = numOfElements.getDefiningOp<arith::ConstantOp>();
          auto numElementsAttr = numElementsOp.getValue();
          auto numElements =
              dyn_cast<IntegerAttr>(numElementsAttr).getValue().getZExtValue();

          uint64_t totalBurstLength = (uint64_t)elementSizeBytes * numElements;
          uint32_t roundedTotalBurstLength =
              roundUpToPowerOf2((uint32_t)totalBurstLength);
          auto realCpuLength =
              UInt::constant(32, roundedTotalBurstLength, b, loc);

          dmaItfc->callMethod("cpu_to_isax",
                              {cpuAddr, localAddr.bits(31, 0).getValue(),
                               realCpuLength.bits(3, 0).getValue()},
                              b);

        } else if (auto memBurstLoadCollect =
                       dyn_cast<aps::ItfcBurstLoadCollect>(op)) {
          // no action needed
          dmaItfc->callMethod("poll_for_idle", {}, b);
        } else if (auto memBurstStoreReq =
                       dyn_cast<aps::ItfcBurstStoreReq>(op)) {

          Value cpuAddr = memBurstLoadReq.getCpuAddr();
          Value memRef = memBurstLoadReq.getMemrefs()[0];
          Value start = memBurstLoadReq.getStart();
          Value numOfElements = memBurstLoadReq.getLength();

          // Get the global memory reference name
          auto getGlobalOp =
              dyn_cast<memref::GetGlobalOp>(memRef.getDefiningOp());
          auto globalName = getGlobalOp.getName().str();

          // Find the corresponding memory entry using the pre-built map
          MemoryEntryInfo *targetMemEntry = nullptr;
          if (!globalName.empty()) {
            auto it = memEntryMap.find(globalName);
            if (it != memEntryMap.end()) {
              targetMemEntry = &it->second;
            }
          }

          // Get element type and size from the memory entry info or global
          // memref
          int elementSizeBytes = 1; // Default to 1 byte
          if (targetMemEntry) {
            // Use the data width from the memory entry info
            elementSizeBytes = (targetMemEntry->dataWidth + 7) /
                               8; // Convert bits to bytes, rounding up
          } else if (!globalName.empty()) {
            op->emitError("Failed to find target memory entry!");
          }

          // Calculate localAddr: baseAddress + (start * numOfElements *
          // elementSizeBytes)
          uint32_t baseAddress = targetMemEntry->baseAddress;

          // Calculate offset: start * numOfElements * elementSizeBytes
          // First multiply start * numOfElements
          auto baseAddrConst = UInt::constant(baseAddress, 32, b, loc);
          auto startSig = Signal(start, &b, loc);
          auto elementSizeBytesConst =
              UInt::constant(32, elementSizeBytes, b, loc);
          Signal localAddr = baseAddrConst + startSig * elementSizeBytesConst;

          // Calculate total burst length: elementSizeBytes * numOfElements,
          // rounded up to nearest power of 2
          auto numElementsOp = numOfElements.getDefiningOp<arith::ConstantOp>();
          auto numElementsAttr = numElementsOp.getValue();
          auto numElements =
              dyn_cast<IntegerAttr>(numElementsAttr).getValue().getZExtValue();

          uint64_t totalBurstLength = (uint64_t)elementSizeBytes * numElements;
          uint32_t roundedTotalBurstLength =
              roundUpToPowerOf2((uint32_t)totalBurstLength);
          auto realCpuLength =
              UInt::constant(32, roundedTotalBurstLength, b, loc);

          dmaItfc->callMethod("isax_to_cpu",
                              {cpuAddr, localAddr.bits(31, 0).getValue(),
                               realCpuLength.bits(3, 0).getValue()},
                              b);

        } else if (auto memBurstStoreCollect =
                       dyn_cast<aps::ItfcBurstStoreCollect>(op)) {
          dmaItfc->callMethod("poll_for_idle", {}, b);
        }
        // Handle cross-slot FIFO writes for producer operations
        // Enqueue producer result values to appropriate FIFOs
        for (mlir::Value result : op->getResults()) {
          if (!isa<mlir::IntegerType>(result.getType()))
            continue;

          auto fifoIt = crossSlotFIFOs.find(result);
          if (fifoIt != crossSlotFIFOs.end()) {
            // Enqueue the value to all FIFOs for this producer
            auto value = getValueInRule(result, op, /*operandIndex=*/-1, b,
                                        localMap, loc);
            if (failed(value))
              return;
            for (CrossSlotFIFO *fifo : fifoIt->second) {
              fifo->fifoInstance->callMethod("enq", {*value}, b);
            }
          }
        }
      }

      // Write token to next stage at the end (except for last stage)
      if (!slotOrder.empty() && slot != slotOrder.back()) {
        auto tokenFifoIt = stageTokenFifos.find(slot);
        if (tokenFifoIt != stageTokenFifos.end()) {
          auto *tokenFifo = tokenFifoIt->second;
          auto tokenValue = UInt::constant(1, 1, b, loc);
          tokenFifo->callMethod("enq", {tokenValue.getValue()}, b);
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
        std::string from =
            std::to_string(opcode) + "_rule_s" + std::to_string(slot);
        std::string to =
            std::to_string(opcode) + "_rule_s" + std::to_string(prevSlot);
        precedencePairs.emplace_back(from, to);
      }
    }
  }

  if (!precedencePairs.empty())
    mainModule->setPrecedence(precedencePairs);
}

} // namespace mlir