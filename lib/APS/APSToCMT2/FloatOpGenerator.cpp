//===- FloatOpGenerator.cpp - Float Operation Generator -------------------===//
//
// Two-phase float operation handling:
// - Start phase (at starttime): Create instance, call start method
// - Result phase (at endtime): Call result value, get return
//
//===----------------------------------------------------------------------===//

#include "APS/BBHandler.h"
#include "circt/Dialect/Cmt2/ECMT2/Signal.h"
#include "APS/APSOps.h"

namespace mlir {

using namespace circt::cmt2::ecmt2;
using namespace circt::cmt2::ecmt2::stl;
using namespace circt::firrtl;

bool FloatOpGenerator::canHandle(Operation *op) const {
  return isa<tor::AddFOp, tor::SubFOp, tor::MulFOp, tor::DivFOp,
             tor::CmpFOp, tor::MacFOp, tor::SqrtFOp,
             tor::ExpFOp, tor::LogFOp,
             tor::FpToSIOp, tor::SIToFpOp>(op);
}

unsigned FloatOpGenerator::getLatencyFromOp(Operation *op) {
  // Get latency from scheduled ref_endtime - ref_starttime attributes
  // These are set by the scheduler based on ResourceDB
  auto refEndAttr = op->getAttrOfType<IntegerAttr>("ref_endtime");
  auto refStartAttr = op->getAttrOfType<IntegerAttr>("ref_starttime");
  if (refEndAttr && refStartAttr) {
    int64_t latency = refEndAttr.getInt() - refStartAttr.getInt();
    if (latency > 0)
      return static_cast<unsigned>(latency);
  }
  // Fallback to ResourceDB or default
  // For comparison operations (cmpf), use operand width instead of result width
  // because result is i1 but latency depends on operand precision (float32 = 32)
  unsigned width;
  if (auto cmpOp = dyn_cast<tor::CmpFOp>(op)) {
    width = cmpOp.getLhs().getType().getIntOrFloatBitWidth();
  } else {
    width = op->getResult(0).getType().getIntOrFloatBitWidth();
  }
  return bbHandler->getOperationLatency(op, width);
}

LogicalResult FloatOpGenerator::generateRule(Operation *op, mlir::OpBuilder &b,
                                             Location loc, int64_t slot,
                                             llvm::DenseMap<Value, Value> &localMap) {
  // Get starttime and endtime from operation attributes
  auto starttimeAttr = op->getAttrOfType<IntegerAttr>("starttime");
  auto endtimeAttr = op->getAttrOfType<IntegerAttr>("endtime");

  if (!starttimeAttr || !endtimeAttr) {
    op->emitError("Float operation missing starttime/endtime attributes");
    return failure();
  }

  int64_t starttime = starttimeAttr.getInt();
  int64_t endtime = endtimeAttr.getInt();

  // Determine which phase we're in
  bool isStartPhase = (slot == starttime);
  bool isResultPhase = (slot == endtime);

  if (!isStartPhase && !isResultPhase) {
    // This slot is neither start nor result phase for this operation
    return success();
  }

  // Dispatch to appropriate handler based on operation type
  if (isa<tor::AddFOp>(op))
    return handleBinaryFloatOp(op, b, loc, slot, localMap, "add");
  if (isa<tor::SubFOp>(op))
    return handleBinaryFloatOp(op, b, loc, slot, localMap, "sub");
  if (isa<tor::MulFOp>(op))
    return handleBinaryFloatOp(op, b, loc, slot, localMap, "mul");
  if (isa<tor::DivFOp>(op))
    return handleBinaryFloatOp(op, b, loc, slot, localMap, "div");
  if (isa<tor::SqrtFOp>(op))
    return handleUnaryFloatOp(op, b, loc, slot, localMap, "sqrt");
  if (isa<tor::ExpFOp>(op))
    return handleUnaryFloatOp(op, b, loc, slot, localMap, "exp");
  if (isa<tor::LogFOp>(op))
    return handleUnaryFloatOp(op, b, loc, slot, localMap, "log");
  if (isa<tor::SIToFpOp>(op))
    return handleUnaryFloatOp(op, b, loc, slot, localMap, "i2f");
  if (isa<tor::FpToSIOp>(op))
    return handleUnaryFloatOp(op, b, loc, slot, localMap, "f2i");
  if (auto cmpOp = dyn_cast<tor::CmpFOp>(op))
    return handleCmpF(cmpOp, b, loc, slot, localMap);

  return failure();
}

LogicalResult FloatOpGenerator::processFloatResults(int64_t slot, mlir::OpBuilder &b,
                                                     Location loc,
                                                     llvm::DenseMap<Value, Value> &localMap) {
  // Process all pending float operations whose endtime matches this slot
  auto it = pendingFloatOps.find(slot);
  if (it == pendingFloatOps.end())
    return success();

  for (auto &pending : it->second) {
    // Call the result value to get the output
    auto calleeSymbol = FlatSymbolRefAttr::get(b.getContext(), pending.fpuInstance->getName());
    auto valueSymbol = FlatSymbolRefAttr::get(b.getContext(), "result");
    auto resultType = UIntType::get(b.getContext(), pending.resultWidth);

    auto callOp = b.create<circt::cmt2::CallOp>(
        loc, TypeRange{resultType}, ValueRange{},
        calleeSymbol, valueSymbol, nullptr, nullptr);

    localMap[pending.resultVal] = callOp.getResult(0);
  }

  return success();
}

bool FloatOpGenerator::hasPendingResultsForSlot(int64_t slot) const {
  return pendingFloatOps.find(slot) != pendingFloatOps.end();
}

llvm::SmallVector<int64_t> FloatOpGenerator::getPendingEndSlots() const {
  llvm::SmallVector<int64_t> slots;
  for (auto &kv : pendingFloatOps) {
    slots.push_back(kv.first);
  }
  return slots;
}

LogicalResult FloatOpGenerator::handleBinaryFloatOp(
    Operation *op, mlir::OpBuilder &b, Location loc, int64_t slot,
    llvm::DenseMap<Value, Value> &localMap, StringRef opType) {

  auto starttimeAttr = op->getAttrOfType<IntegerAttr>("starttime");
  auto endtimeAttr = op->getAttrOfType<IntegerAttr>("endtime");
  int64_t starttime = starttimeAttr.getInt();
  int64_t endtime = endtimeAttr.getInt();

  Value resultVal = op->getResult(0);
  unsigned width = resultVal.getType().getIntOrFloatBitWidth();

  // Start phase: create instance and call start method
  if (slot == starttime) {
    Value lhsVal = op->getOperand(0);
    Value rhsVal = op->getOperand(1);

    auto lhs = getValueInRule(lhsVal, op, 0, b, localMap, loc);
    auto rhs = getValueInRule(rhsVal, op, 1, b, localMap, loc);
    if (failed(lhs) || failed(rhs))
      return failure();

    unsigned latency = getLatencyFromOp(op);

    Module* mainModule = bbHandler->getMainModule();
    Circuit& circuit = bbHandler->getCircuit();
    Clock clock = bbHandler->getMainClk();
    Reset reset = bbHandler->getMainRst();

    ModuleBase *floatMod = nullptr;
    if (opType == "add")
      floatMod = STLLibrary::createFloatAddModule(width, latency, circuit);
    else if (opType == "sub")
      floatMod = STLLibrary::createFloatSubModule(width, latency, circuit);
    else if (opType == "mul")
      floatMod = STLLibrary::createFloatMulModule(width, latency, circuit);
    else if (opType == "div")
      floatMod = STLLibrary::createFloatDivModule(width, latency, circuit);

    if (!floatMod)
      return failure();

    // Create instance
    std::string instName = "float_" + opType.str() + "_" +
                           std::to_string(instanceCount++);
    auto *instance = mainModule->addInstance(instName, floatMod,
                                             {clock.getValue(), reset.getValue()});

    // Call the start method (no return value)
    auto calleeSymbol = FlatSymbolRefAttr::get(b.getContext(), instance->getName());
    auto methodSymbol = FlatSymbolRefAttr::get(b.getContext(), "start");

    b.create<circt::cmt2::CallOp>(
        loc, TypeRange{}, ValueRange{*lhs, *rhs},
        calleeSymbol, methodSymbol, nullptr, nullptr);

    // Record pending operation for result phase
    PendingFloatOp pending;
    pending.torOp = op;
    pending.fpuInstance = instance;
    pending.endSlot = endtime;
    pending.resultVal = resultVal;
    pending.resultWidth = width;
    pendingFloatOps[endtime].push_back(pending);

  }

  // Result phase: get result from FPU instance using callMethod
  if (slot == endtime) {
    auto it = pendingFloatOps.find(endtime);
    if (it != pendingFloatOps.end()) {
      for (auto &pending : it->second) {
        if (pending.torOp == op) {
          auto resultValues = pending.fpuInstance->callMethod("result", {}, b);
          if (!resultValues.empty()) {
            localMap[pending.resultVal] = resultValues[0];

            // Enqueue to cross-slot FIFOs if this value is needed in later slots
            auto &crossSlotFIFOs = bbHandler->getCrossSlotFIFOs();
            auto fifoIt = crossSlotFIFOs.find(pending.resultVal);
            if (fifoIt != crossSlotFIFOs.end()) {
              for (CrossSlotFIFO *fifo : fifoIt->second) {
                if (fifo->fifoInstance) {
                  fifo->fifoInstance->callMethod("enq", {resultValues[0]}, b);
                }
              }
            }
          }
          break;
        }
      }
    }
  }

  return success();
}

LogicalResult FloatOpGenerator::handleUnaryFloatOp(
    Operation *op, mlir::OpBuilder &b, Location loc, int64_t slot,
    llvm::DenseMap<Value, Value> &localMap, StringRef opType) {

  auto starttimeAttr = op->getAttrOfType<IntegerAttr>("starttime");
  auto endtimeAttr = op->getAttrOfType<IntegerAttr>("endtime");
  int64_t starttime = starttimeAttr.getInt();
  int64_t endtime = endtimeAttr.getInt();

  Value resultVal = op->getResult(0);
  unsigned width = resultVal.getType().getIntOrFloatBitWidth();

  // Start phase: create instance and call start method
  if (slot == starttime) {
    Value operandVal = op->getOperand(0);

    auto operand = getValueInRule(operandVal, op, 0, b, localMap, loc);
    if (failed(operand))
      return failure();

    unsigned latency = getLatencyFromOp(op);

    Module* mainModule = bbHandler->getMainModule();
    Circuit& circuit = bbHandler->getCircuit();
    Clock clock = bbHandler->getMainClk();
    Reset reset = bbHandler->getMainRst();

    ModuleBase *floatMod = nullptr;
    if (opType == "sqrt")
      floatMod = STLLibrary::createFloatSqrtModule(width, latency, circuit);
    else if (opType == "exp")
      floatMod = STLLibrary::createFloatExpModule(width, latency, circuit);
    else if (opType == "log")
      floatMod = STLLibrary::createFloatLogModule(width, latency, circuit);
    else if (opType == "i2f")
      floatMod = STLLibrary::createInt2FloatModule(width, latency, circuit);
    else if (opType == "f2i")
      floatMod = STLLibrary::createFloat2IntModule(width, latency, circuit);

    if (!floatMod)
      return failure();

    std::string instName = "float_" + opType.str() + "_" +
                           std::to_string(instanceCount++);
    auto *instance = mainModule->addInstance(instName, floatMod,
                                             {clock.getValue(), reset.getValue()});

    // Call the start method (no return value)
    auto calleeSymbol = FlatSymbolRefAttr::get(b.getContext(), instance->getName());
    auto methodSymbol = FlatSymbolRefAttr::get(b.getContext(), "start");

    b.create<circt::cmt2::CallOp>(
        loc, TypeRange{}, ValueRange{*operand},
        calleeSymbol, methodSymbol, nullptr, nullptr);

    // Record pending operation for result phase
    PendingFloatOp pending;
    pending.torOp = op;
    pending.fpuInstance = instance;
    pending.endSlot = endtime;
    pending.resultVal = resultVal;
    pending.resultWidth = width;
    pendingFloatOps[endtime].push_back(pending);
  }

  // Result phase: get result from FPU instance using callMethod
  if (slot == endtime) {
    auto it = pendingFloatOps.find(endtime);
    if (it != pendingFloatOps.end()) {
      for (auto &pending : it->second) {
        if (pending.torOp == op) {
          auto resultValues = pending.fpuInstance->callMethod("result", {}, b);
          if (!resultValues.empty()) {
            localMap[pending.resultVal] = resultValues[0];

            // Enqueue to cross-slot FIFOs if this value is needed in later slots
            auto &crossSlotFIFOs = bbHandler->getCrossSlotFIFOs();
            auto fifoIt = crossSlotFIFOs.find(pending.resultVal);
            if (fifoIt != crossSlotFIFOs.end()) {
              for (CrossSlotFIFO *fifo : fifoIt->second) {
                if (fifo->fifoInstance) {
                  fifo->fifoInstance->callMethod("enq", {resultValues[0]}, b);
                }
              }
            }
          }
          break;
        }
      }
    }
  }

  return success();
}

LogicalResult FloatOpGenerator::handleCmpF(tor::CmpFOp op, mlir::OpBuilder &b,
                                           Location loc, int64_t slot,
                                           llvm::DenseMap<Value, Value> &localMap) {
  auto starttimeAttr = op->getAttrOfType<IntegerAttr>("starttime");
  auto endtimeAttr = op->getAttrOfType<IntegerAttr>("endtime");
  int64_t starttime = starttimeAttr.getInt();
  int64_t endtime = endtimeAttr.getInt();

  // Start phase: create instance and call start method
  if (slot == starttime) {
    auto lhs = getValueInRule(op.getLhs(), op, 0, b, localMap, loc);
    auto rhs = getValueInRule(op.getRhs(), op, 1, b, localMap, loc);
    if (failed(lhs) || failed(rhs))
      return failure();

    unsigned width = op.getLhs().getType().getIntOrFloatBitWidth();
    unsigned predicate = predicateToInt(op.getPredicate());
    unsigned latency = getLatencyFromOp(op);

    Module* mainModule = bbHandler->getMainModule();
    Circuit& circuit = bbHandler->getCircuit();
    Clock clock = bbHandler->getMainClk();
    Reset reset = bbHandler->getMainRst();

    // Use external module (like other float ops) - FloatCmp.v already has FIFO buffering
    auto *cmpMod = STLLibrary::createFloatCmpModule(width, predicate, latency, circuit);
    if (!cmpMod)
      return failure();

    std::string instName = "float_cmp_" + std::to_string(instanceCount++);
    auto *instance = mainModule->addInstance(instName, cmpMod,
                                             {clock.getValue(), reset.getValue()});

    // Call the start method (no return value)
    auto calleeSymbol = FlatSymbolRefAttr::get(b.getContext(), instance->getName());
    auto methodSymbol = FlatSymbolRefAttr::get(b.getContext(), "start");

    b.create<circt::cmt2::CallOp>(
        loc, TypeRange{}, ValueRange{*lhs, *rhs},
        calleeSymbol, methodSymbol, nullptr, nullptr);

    // Record pending operation for result phase (result is 1-bit)
    PendingFloatOp pending;
    pending.torOp = op;
    pending.fpuInstance = instance;
    pending.endSlot = endtime;
    pending.resultVal = op.getResult();
    pending.resultWidth = 1;
    pendingFloatOps[endtime].push_back(pending);
  }

  // Result phase: get result from FIFO using callMethod("result")
  if (slot == endtime) {
    auto it = pendingFloatOps.find(endtime);
    if (it != pendingFloatOps.end()) {
      for (auto &pending : it->second) {
        if (pending.torOp == op) {
          auto resultValues = pending.fpuInstance->callMethod("result", {}, b);
          if (!resultValues.empty()) {
            localMap[pending.resultVal] = resultValues[0];

            // Enqueue to cross-slot FIFOs if this value is needed in later slots
            auto &crossSlotFIFOs = bbHandler->getCrossSlotFIFOs();
            auto fifoIt = crossSlotFIFOs.find(pending.resultVal);
            if (fifoIt != crossSlotFIFOs.end()) {
              for (CrossSlotFIFO *fifo : fifoIt->second) {
                if (fifo->fifoInstance) {
                  fifo->fifoInstance->callMethod("enq", {resultValues[0]}, b);
                }
              }
            }
          }
          break;
        }
      }
    }
  }

  return success();
}

unsigned FloatOpGenerator::predicateToInt(tor::CmpFPredicate pred) {
  switch (pred) {
    case tor::CmpFPredicate::OEQ:
    case tor::CmpFPredicate::UEQ: return 0;  // eq
    case tor::CmpFPredicate::OLT:
    case tor::CmpFPredicate::ULT: return 1;  // lt
    case tor::CmpFPredicate::OLE:
    case tor::CmpFPredicate::ULE: return 2;  // le
    case tor::CmpFPredicate::OGT:
    case tor::CmpFPredicate::UGT: return 3;  // gt
    case tor::CmpFPredicate::OGE:
    case tor::CmpFPredicate::UGE: return 4;  // ge
    case tor::CmpFPredicate::ONE:
    case tor::CmpFPredicate::UNE: return 5;  // ne
    default: return 0;
  }
}

} // namespace mlir
