//===- MemoryOpGenerator.cpp - Memory Operation Generator
//------------------===//
//
// This file implements the memory operation generator for TOR functions
//
//===----------------------------------------------------------------------===//

#include "APS/APSOps.h"
#include "APS/BBHandler.h"
#include "circt/Dialect/Cmt2/ECMT2/Signal.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace mlir {

using namespace mlir;
using namespace mlir::tor;
using namespace circt::cmt2::ecmt2;
using namespace circt::cmt2::ecmt2::stl;
using namespace circt::firrtl;

LogicalResult MemoryOpGenerator::generateRule(
    Operation *op, mlir::OpBuilder &b, Location loc, int64_t slot,
    llvm::DenseMap<mlir::Value, mlir::Value> &localMap) {
  if (auto memLoad = dyn_cast<aps::MemLoad>(op)) {
    return generateMemLoad(memLoad, b, loc, slot, localMap);
  } else if (auto memStore = dyn_cast<aps::MemStore>(op)) {
    return generateMemStore(memStore, b, loc, slot, localMap);
  } else if (auto burstLoadReq = dyn_cast<aps::ItfcBurstLoadReq>(op)) {
    return generateBurstLoadReq(burstLoadReq, b, loc, slot, localMap);
  } else if (auto burstLoadCollect = dyn_cast<aps::ItfcBurstLoadCollect>(op)) {
    return generateBurstLoadCollect(burstLoadCollect, b, loc, slot, localMap);
  } else if (auto burstStoreReq = dyn_cast<aps::ItfcBurstStoreReq>(op)) {
    return generateBurstStoreReq(burstStoreReq, b, loc, slot, localMap);
  } else if (auto burstStoreCollect =
                 dyn_cast<aps::ItfcBurstStoreCollect>(op)) {
    return generateBurstStoreCollect(burstStoreCollect, b, loc, slot, localMap);
  } else if (isa<memref::GetGlobalOp>(op)) {
    return success();
  }

  return failure();
}

bool MemoryOpGenerator::canHandle(Operation *op) const {
  return isa<aps::MemLoad, aps::MemStore, aps::ItfcBurstLoadReq,
             aps::ItfcBurstLoadCollect, aps::ItfcBurstStoreReq,
             aps::ItfcBurstStoreCollect, memref::GetGlobalOp>(op);
}

LogicalResult MemoryOpGenerator::generateMemLoad(
    aps::MemLoad op, mlir::OpBuilder &b, Location loc, int64_t slot,
    llvm::DenseMap<mlir::Value, mlir::Value> &localMap) {
  // PANIC if no indices provided
  if (op.getIndices().empty()) {
    op.emitError("Memory load operation must have at least one index");
    llvm::report_fatal_error("Memory load requires address indices");
  }

  auto addr = getValueInRule(op.getIndices()[0], op.getOperation(), 0, b,
                             localMap, loc);
  if (failed(addr)) {
    op.emitError("Failed to get address for memory load");
    llvm::report_fatal_error("Memory load address resolution failed");
  }

  // Get the memory reference and check if it comes from memref.get_global
  Value memRef = op.getMemref();
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

  localMap[op.getResult()] = callResult.getResult(0);
  return success();
}

LogicalResult MemoryOpGenerator::generateMemStore(
    aps::MemStore op, mlir::OpBuilder &b, Location loc, int64_t slot,
    llvm::DenseMap<mlir::Value, mlir::Value> &localMap) {
  // Handle memory store operations - implementation needed
  // This would be similar to memload but for writes
  return failure();
}

LogicalResult MemoryOpGenerator::generateBurstLoadReq(
    aps::ItfcBurstLoadReq op, mlir::OpBuilder &b, Location loc, int64_t slot,
    llvm::DenseMap<mlir::Value, mlir::Value> &localMap) {
  Value cpuAddr = op.getCpuAddr();
  Value memRef = op.getMemrefs()[0];
  Value start = op.getStart();
  Value numOfElements = op.getLength();

  // Get the global memory reference name
  auto getGlobalOp = dyn_cast<memref::GetGlobalOp>(memRef.getDefiningOp());
  if (!getGlobalOp) {
    op.emitError("Burst load request must use global memory reference");
    return failure();
  }
  auto globalName = getGlobalOp.getName().str();

  // Find the corresponding memory entry using the pre-built map
  const MemoryEntryInfo *targetMemEntry = nullptr;
  if (!globalName.empty()) {
    auto &memEntryMap = bbHandler->getMemEntryMap();
    auto it = memEntryMap.find(globalName);
    if (it != memEntryMap.end()) {
      targetMemEntry = &it->second;
    }
  }

  if (!targetMemEntry) {
    op.emitError("Failed to find target memory entry!");
    return failure();
  }

  // Get element type and size from the memory entry info
  int elementSizeBytes =
      (targetMemEntry->dataWidth + 7) / 8; // Convert bits to bytes, rounding up

  // Calculate localAddr: baseAddress + (start * numOfElements *
  // elementSizeBytes)
  uint32_t baseAddress = targetMemEntry->baseAddress;

  // Calculate offset: start * numOfElements * elementSizeBytes
  auto baseAddrConst = UInt::constant(baseAddress, 32, b, loc);
  auto startSig = Signal(start, &b, loc);
  auto elementSizeBytesConst = UInt::constant(32, elementSizeBytes, b, loc);
  Signal localAddr = baseAddrConst + startSig * elementSizeBytesConst;

  // Calculate total burst length: elementSizeBytes * numOfElements, rounded up
  // to nearest power of 2
  auto numElementsOp = numOfElements.getDefiningOp<arith::ConstantOp>();
  if (!numElementsOp) {
    op.emitError("Number of elements must be a constant");
    return failure();
  }
  auto numElementsAttr = numElementsOp.getValue();
  auto numElements =
      dyn_cast<IntegerAttr>(numElementsAttr).getValue().getZExtValue();

  uint64_t totalBurstLength = (uint64_t)elementSizeBytes * numElements;
  uint32_t roundedTotalBurstLength =
      bbHandler->roundUpToPowerOf2((uint32_t)totalBurstLength);
  auto realCpuLength = UInt::constant(32, roundedTotalBurstLength, b, loc);

  // Call DMA interface
  auto dmaItfc = bbHandler->getDmaInterface();
  if (!dmaItfc) {
    op.emitError("DMA interface not available");
    return failure();
  }

  dmaItfc->callMethod("cpu_to_isax",
                      {cpuAddr, localAddr.bits(31, 0).getValue(),
                       realCpuLength.bits(3, 0).getValue()},
                      b);

  localMap[op.getResult()] = UInt::constant(1, 1, b, loc).getValue();
  return success();
}

LogicalResult MemoryOpGenerator::generateBurstLoadCollect(
    aps::ItfcBurstLoadCollect op, mlir::OpBuilder &b, Location loc,
    int64_t slot, llvm::DenseMap<mlir::Value, mlir::Value> &localMap) {
  // no action needed
  auto dmaItfc = bbHandler->getDmaInterface();
  if (dmaItfc) {
    dmaItfc->callMethod("poll_for_idle", {}, b);
  }
  return success();
}

LogicalResult MemoryOpGenerator::generateBurstStoreReq(
    aps::ItfcBurstStoreReq op, mlir::OpBuilder &b, Location loc, int64_t slot,
    llvm::DenseMap<mlir::Value, mlir::Value> &localMap) {
  Value cpuAddr = op.getCpuAddr();
  Value memRef = op.getMemrefs()[0];
  Value start = op.getStart();
  Value numOfElements = op.getLength();

  // Get the global memory reference name
  auto getGlobalOp = dyn_cast<memref::GetGlobalOp>(memRef.getDefiningOp());
  if (!getGlobalOp) {
    op.emitError("Burst store request must use global memory reference");
    return failure();
  }
  auto globalName = getGlobalOp.getName().str();

  // Find the corresponding memory entry using the pre-built map
  const MemoryEntryInfo *targetMemEntry = nullptr;
  if (!globalName.empty()) {
    auto &memEntryMap = bbHandler->getMemEntryMap();
    auto it = memEntryMap.find(globalName);
    if (it != memEntryMap.end()) {
      targetMemEntry = &it->second;
    }
  }

  if (!targetMemEntry) {
    op.emitError("Failed to find target memory entry!");
    return failure();
  }

  // Get element type and size from the memory entry info
  int elementSizeBytes =
      (targetMemEntry->dataWidth + 7) / 8; // Convert bits to bytes, rounding up

  // Calculate localAddr: baseAddress + (start * numOfElements *
  // elementSizeBytes)
  uint32_t baseAddress = targetMemEntry->baseAddress;

  // Calculate offset: start * numOfElements * elementSizeBytes
  auto baseAddrConst = UInt::constant(baseAddress, 32, b, loc);
  auto startSig = Signal(start, &b, loc);
  auto elementSizeBytesConst = UInt::constant(32, elementSizeBytes, b, loc);
  Signal localAddr = baseAddrConst + startSig * elementSizeBytesConst;

  // Calculate total burst length: elementSizeBytes * numOfElements, rounded up
  // to nearest power of 2
  auto numElementsOp = numOfElements.getDefiningOp<arith::ConstantOp>();
  if (!numElementsOp) {
    op.emitError("Number of elements must be a constant");
    return failure();
  }
  auto numElementsAttr = numElementsOp.getValue();
  auto numElements =
      dyn_cast<IntegerAttr>(numElementsAttr).getValue().getZExtValue();

  uint64_t totalBurstLength = (uint64_t)elementSizeBytes * numElements;
  uint32_t roundedTotalBurstLength =
      bbHandler->roundUpToPowerOf2((uint32_t)totalBurstLength);
  auto realCpuLength = UInt::constant(32, roundedTotalBurstLength, b, loc);

  // Call DMA interface
  auto dmaItfc = bbHandler->getDmaInterface();
  if (!dmaItfc) {
    op.emitError("DMA interface not available");
    return failure();
  }

  dmaItfc->callMethod("isax_to_cpu",
                      {cpuAddr, localAddr.bits(31, 0).getValue(),
                       realCpuLength.bits(3, 0).getValue()},
                      b);

  localMap[op.getResult()] = UInt::constant(1, 1, b, loc).getValue();
  return success();
}

LogicalResult MemoryOpGenerator::generateBurstStoreCollect(
    aps::ItfcBurstStoreCollect op, mlir::OpBuilder &b, Location loc,
    int64_t slot, llvm::DenseMap<mlir::Value, mlir::Value> &localMap) {
  auto dmaItfc = bbHandler->getDmaInterface();
  if (dmaItfc) {
    dmaItfc->callMethod("poll_for_idle", {}, b);
  }
  return success();
}

} // namespace mlir