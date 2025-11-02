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
#include "mlir/IR/ValueRange.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

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
  } else if (auto globalLoad = dyn_cast<aps::GlobalLoad>(op)) {
    return generateGlobalMemLoad(globalLoad, b, loc, slot, localMap);
  } else if (auto globalStore = dyn_cast<aps::GlobalStore>(op)) {
    return generateGlobalMemStore(globalStore, b, loc, slot, localMap);
  } else if (isa<memref::GetGlobalOp>(op)) {
    return success();
  }

  return failure();
}

bool MemoryOpGenerator::canHandle(Operation *op) const {
  return isa<aps::MemLoad, aps::MemStore, aps::ItfcBurstLoadReq,
             aps::ItfcBurstLoadCollect, aps::ItfcBurstStoreReq,
             aps::ItfcBurstStoreCollect, memref::GetGlobalOp,
             aps::GlobalStore, aps::GlobalLoad>(op);
}

LogicalResult MemoryOpGenerator::generateGlobalMemLoad(
    aps::GlobalLoad op, mlir::OpBuilder &b, Location loc, int64_t slot,
    llvm::DenseMap<mlir::Value, mlir::Value> &localMap) {
  // GlobalLoad is for scalar globals (rank-0 memrefs) - no indices needed
  // Get the global symbol name directly from the operation
  StringRef globalName = op.getGlobalName();

  // Build the rule name: convert @count -> count_read
  std::string memoryBankRule = (globalName + "_read").str();

  llvm::outs() << "DEBUG: Global load from scalar global " << globalName
               << " using rule " << memoryBankRule << "\n";

  // For scalar globals, we need to get the base address from the memory entry map
  const MemoryEntryInfo *targetMemEntry = nullptr;
  if (!globalName.empty()) {
    auto &memEntryMap = bbHandler->getMemEntryMap();
    auto it = memEntryMap.find(globalName);
    if (it != memEntryMap.end()) {
      targetMemEntry = &it->second;
    }
  }

  if (!targetMemEntry) {
    op.emitError("Failed to find target memory entry for global: ") << globalName.str();
    return failure();
  }

  // Create base address constant for the scalar global
  auto baseAddr = UInt::constant(0, 1, b, loc);

  // Call the appropriate scratchpad pool bank read method
  auto callResult = b.create<circt::cmt2::CallOp>(
      loc, circt::firrtl::UIntType::get(b.getContext(), targetMemEntry->dataWidth),
      mlir::ValueRange{baseAddr.getValue()},
      mlir::SymbolRefAttr::get(b.getContext(), "scratchpad_pool"),
      mlir::SymbolRefAttr::get(b.getContext(), memoryBankRule),
      mlir::ArrayAttr(), mlir::ArrayAttr());

  localMap[op.getResult()] = callResult.getResult(0);
  return success();
}

LogicalResult MemoryOpGenerator::generateGlobalMemStore(
    aps::GlobalStore op, mlir::OpBuilder &b, Location loc, int64_t slot,
    llvm::DenseMap<mlir::Value, mlir::Value> &localMap) {
  // GlobalLoad is for scalar globals (rank-0 memrefs) - no indices needed
  // Get the global symbol name directly from the operation
  StringRef globalName = op.getGlobalName();

  // Build the rule name: convert @count -> count_read
  std::string memoryBankRule = (globalName + "_write").str();

  llvm::outs() << "DEBUG: Global load from scalar global " << globalName
               << " using rule " << memoryBankRule << "\n";

  // For scalar globals, we need to get the base address from the memory entry map
  const MemoryEntryInfo *targetMemEntry = nullptr;
  if (!globalName.empty()) {
    auto &memEntryMap = bbHandler->getMemEntryMap();
    auto it = memEntryMap.find(globalName);
    if (it != memEntryMap.end()) {
      targetMemEntry = &it->second;
    }
  }

  if (!targetMemEntry) {
    op.emitError("Failed to find target memory entry for global: ") << globalName.str();
    return failure();
  }

  // Create base address constant for the scalar global
  auto baseAddr = UInt::constant(0, 1, b, loc);
  auto data = getValueInRule(op.getValue(), op.getOperation(), 0, b,
                             localMap, loc);

  // Call the appropriate scratchpad pool bank read method
  b.create<circt::cmt2::CallOp>(
      loc, mlir::ValueRange{},
      mlir::ValueRange{baseAddr.getValue(), *data},
      mlir::SymbolRefAttr::get(b.getContext(), "scratchpad_pool"),
      mlir::SymbolRefAttr::get(b.getContext(), memoryBankRule),
      mlir::ArrayAttr(), mlir::ArrayAttr());

  return success();
}

LogicalResult MemoryOpGenerator::generateMemLoad(
    aps::MemLoad op, mlir::OpBuilder &b, Location loc, int64_t slot,
    llvm::DenseMap<mlir::Value, mlir::Value> &localMap) {
  // PANIC if no indices provided
  if (op.getIndices().empty()) {
    op.emitError("Memory load operation must have at least one index");
    llvm::report_fatal_error("Memory load requires address indices");
  }

  auto addr = getValueInRule(op.getIndices()[0], op.getOperation(), 1, b,
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
    memoryBankRule = (globalName + "_read")
                         .str(); // Remove @ prefix and add _read
    llvm::outs() << "DEBUG: Memory load from global " << globalName
                 << " using rule " << memoryBankRule << "\n";
  }

  // Get the memref type to determine array size and element type
  auto memrefType = dyn_cast<mlir::MemRefType>(memRef.getType());
  if (!memrefType) {
    op.emitError("Memory reference is not memref type");
    llvm::report_fatal_error("MemLoad: invalid memref type");
  }

  // Get element type for result width
  Type elementType = memrefType.getElementType();
  auto intType = dyn_cast<mlir::IntegerType>(elementType);
  if (!intType) {
    op.emitError("Memref element type is not integer");
    llvm::report_fatal_error("MemLoad: memref element must be integer type");
  }
  unsigned resultWidth = intType.getWidth();

  // Calculate required address bit width from array size: ceil(log2(size))
  auto shape = memrefType.getShape();
  if (shape.empty() || shape[0] <= 0) {
    op.emitError("Memref must have valid array size");
    llvm::report_fatal_error("MemLoad: invalid memref shape");
  }
  int64_t arraySize = shape[0];
  unsigned addrWidth = arraySize <= 1 ? 1 : (unsigned)std::ceil(std::log2(arraySize));

  llvm::outs() << "DEBUG: Memory load - array size: " << arraySize
               << ", addr width: " << addrWidth
               << ", result width: " << resultWidth << "\n";

  // Truncate address to required bit width using Signal
  auto addrSignal = Signal(*addr, &b, loc).bits(addrWidth - 1, 0);

  // Call the appropriate scratchpad pool bank read method with correct return width
  auto callResult = b.create<circt::cmt2::CallOp>(
      loc, circt::firrtl::UIntType::get(b.getContext(), resultWidth),
      mlir::ValueRange{addrSignal.getValue()},
      mlir::SymbolRefAttr::get(b.getContext(), "scratchpad_pool"),
      mlir::SymbolRefAttr::get(b.getContext(), memoryBankRule),
      mlir::ArrayAttr(), mlir::ArrayAttr());

  localMap[op.getResult()] = callResult.getResult(0);
  return success();
}

LogicalResult MemoryOpGenerator::generateMemStore(
    aps::MemStore op, mlir::OpBuilder &b, Location loc, int64_t slot,
    llvm::DenseMap<mlir::Value, mlir::Value> &localMap) {
  // PANIC if no indices provided
  if (op.getIndices().empty()) {
    op.emitError("Memory store operation must have at least one index");
    llvm::report_fatal_error("Memory store requires address indices");
  }

  // Get the value to store
  auto value = getValueInRule(op.getValue(), op.getOperation(), 0, b,
                              localMap, loc);
  if (failed(value)) {
    op.emitError("Failed to get value for memory store");
    llvm::report_fatal_error("Memory store value resolution failed");
  }

  // Get the address
  auto addr = getValueInRule(op.getIndices()[0], op.getOperation(), 2, b,
                             localMap, loc);
  if (failed(addr)) {
    op.emitError("Failed to get address for memory store");
    llvm::report_fatal_error("Memory store address resolution failed");
  }

  // Get the memory reference and check if it comes from memref.get_global
  Value memRef = op.getMemref();
  Operation *defOp = memRef.getDefiningOp();

  std::string memoryBankRule;
  if (auto getGlobalOp = dyn_cast<memref::GetGlobalOp>(defOp)) {
    // Extract the global symbol name and build the rule name
    StringRef globalName = getGlobalOp.getName();
    // Convert @mem_a_0 -> mem_a_0_write
    memoryBankRule = (globalName + "_write").str();
    llvm::outs() << "DEBUG: Memory store to global " << globalName
                 << " using rule " << memoryBankRule << "\n";
  }

  // Get the memref type to determine array size and element type
  auto memrefType = dyn_cast<mlir::MemRefType>(memRef.getType());
  if (!memrefType) {
    op.emitError("Memory reference is not memref type");
    llvm::report_fatal_error("MemStore: invalid memref type");
  }

  // Get element type for data width
  Type elementType = memrefType.getElementType();
  auto intType = dyn_cast<mlir::IntegerType>(elementType);
  if (!intType) {
    op.emitError("Memref element type is not integer");
    llvm::report_fatal_error("MemStore: memref element must be integer type");
  }
  unsigned dataWidth = intType.getWidth();

  // Calculate required address bit width from array size: ceil(log2(size))
  auto shape = memrefType.getShape();
  if (shape.empty() || shape[0] <= 0) {
    op.emitError("Memref must have valid array size");
    llvm::report_fatal_error("MemStore: invalid memref shape");
  }
  int64_t arraySize = shape[0];
  unsigned addrWidth = arraySize <= 1 ? 1 : (unsigned)std::ceil(std::log2(arraySize));

  llvm::outs() << "DEBUG: Memory store - array size: " << arraySize
               << ", addr width: " << addrWidth
               << ", data width: " << dataWidth << "\n";

  // Truncate address to required bit width using Signal
  auto addrSignal = Signal(*addr, &b, loc).bits(addrWidth - 1, 0);

  // Call the appropriate scratchpad pool bank write method
  b.create<circt::cmt2::CallOp>(
      loc, TypeRange{}, // No return value for write
      mlir::ValueRange{addrSignal.getValue(), *value},
      mlir::SymbolRefAttr::get(b.getContext(), "scratchpad_pool"),
      mlir::SymbolRefAttr::get(b.getContext(), memoryBankRule),
      mlir::ArrayAttr(), mlir::ArrayAttr());

  return success();
}

LogicalResult MemoryOpGenerator::generateBurstLoadReq(
    aps::ItfcBurstLoadReq op, mlir::OpBuilder &b, Location loc, int64_t slot,
    llvm::DenseMap<mlir::Value, mlir::Value> &localMap) {
  Value cpuAddr = op.getCpuAddr();
  Value memRef = op.getMemrefs()[0];
  Value start = op.getStart();
  llvm::outs() << start.getType();
  Value numOfElements = op.getLength();

  // Calculate operand indices accounting for variadic memrefs
  // BurstLoadReq: cpu_addr(0), memrefs(1..N), start(1+N), length(2+N)
  unsigned numMemrefs = op.getMemrefs().size();
  unsigned cpuAddrOperandId = 0;
  unsigned startOperandId = 1 + numMemrefs;
  unsigned lengthOperandId = 2 + numMemrefs;

  // Get the global memory reference name
  auto getGlobalOp = dyn_cast<memref::GetGlobalOp>(memRef.getDefiningOp());
  if (!getGlobalOp) {
    op.emitError("Burst load request must use global memory reference");
    return failure();
  }
  auto globalName = getGlobalOp.getName().str();

  // Find the corresponding memory entry using the pre-built map with prefix matching
  const MemoryEntryInfo *targetMemEntry = nullptr;
  if (!globalName.empty()) {
    auto &memEntryMap = bbHandler->getMemEntryMap();
    
    // First try exact match
    auto exactIt = memEntryMap.find(globalName);
    if (exactIt != memEntryMap.end()) {
      targetMemEntry = &exactIt->second;
    } else {
      // If exact match fails, try prefix matching with underscore
      llvm::SmallVector<const MemoryEntryInfo *, 4> matchingEntries;
      
      for (auto &entry : memEntryMap) {
        std::string key = entry.first.str();
        // Check if the map key starts with globalName + "_"
        if (globalName.rfind( key + "_", 0) == 0) {
          targetMemEntry = &entry.second;
        }
      }
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

  // Use getValueInRule to get start with proper FIRRTL conversion
  auto startValue = getValueInRule(start, op.getOperation(), startOperandId, b, localMap, loc);
  if (failed(startValue)) {
    op.emitError("Failed to get start for burst load");
    llvm::report_fatal_error("Burst load start resolution failed");
  }

  auto startSig = Signal(*startValue, &b, loc).bits(31, 0);
  auto elementSizeBytesConst = UInt::constant(elementSizeBytes, 32, b, loc);
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
  // TileLink size field expects log2 of transfer size (N where size = 2^N)
  uint32_t tlSizeField = bbHandler->log2Floor(roundedTotalBurstLength);
  if (tlSizeField == 0) {
    op->emitError("Got 0 when attempt to call burst load!");
  }
  auto realCpuLength = UInt::constant(tlSizeField, 32, b, loc);

  // Use getValueInRule to get cpuAddr with proper FIRRTL conversion
  auto cpuAddrValue = getValueInRule(cpuAddr, op.getOperation(), cpuAddrOperandId, b, localMap, loc);
  if (failed(cpuAddrValue)) {
    op.emitError("Failed to get cpuAddr for burst load");
    llvm::report_fatal_error("Burst load cpuAddr resolution failed");
  }

  // Call DMA interface
  auto dmaItfc = bbHandler->getDmaInterface();
  if (!dmaItfc) {
    op.emitError("DMA interface not available");
    return failure();
  }

  dmaItfc->callMethod("cpu_to_isax",
                      {*cpuAddrValue, localAddr.bits(31, 0).getValue(),
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

  // Calculate operand indices accounting for variadic memrefs
  // BurstStoreReq: memrefs(0..N-1), start(N), cpu_addr(N+1), length(N+2)
  unsigned numMemrefs = op.getMemrefs().size();
  unsigned startOperandId = numMemrefs;
  unsigned cpuAddrOperandId = numMemrefs + 1;
  unsigned lengthOperandId = numMemrefs + 2;

  // Get the global memory reference name
  auto getGlobalOp = dyn_cast<memref::GetGlobalOp>(memRef.getDefiningOp());
  if (!getGlobalOp) {
    op.emitError("Burst store request must use global memory reference");
    return failure();
  }
  auto globalName = getGlobalOp.getName().str();

  // Find the corresponding memory entry using the pre-built map with prefix matching
  const MemoryEntryInfo *targetMemEntry = nullptr;
  if (!globalName.empty()) {
    auto &memEntryMap = bbHandler->getMemEntryMap();
    
    // First try exact match
    auto exactIt = memEntryMap.find(globalName);
    if (exactIt != memEntryMap.end()) {
      targetMemEntry = &exactIt->second;
    } else {
      // If exact match fails, try prefix matching with underscore
      llvm::SmallVector<const MemoryEntryInfo *, 4> matchingEntries;
      
      for (auto &entry : memEntryMap) {
        std::string key = entry.first.str();
        // Check if the map key starts with globalName + "_"
        if (globalName.rfind( key + "_", 0) == 0) {
          targetMemEntry = &entry.second;
        }
      }
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

  // Use getValueInRule to get start with proper FIRRTL conversion
  auto startValue = getValueInRule(start, op.getOperation(), startOperandId, b, localMap, loc);
  if (failed(startValue)) {
    op.emitError("Failed to get start for burst store");
    llvm::report_fatal_error("Burst store start resolution failed");
  }

  // Calculate offset: start * numOfElements * elementSizeBytes
  auto baseAddrConst = UInt::constant(baseAddress, 32, b, loc);
  auto startSig = Signal(*startValue, &b, loc);
  auto elementSizeBytesConst = UInt::constant(elementSizeBytes, 32, b, loc);
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
  // TileLink size field expects log2 of transfer size (N where size = 2^N)
  uint32_t tlSizeField = bbHandler->log2Floor(roundedTotalBurstLength);
  if (tlSizeField == 0) {
    op->emitError("Got 0 when attempt to call burst store!");
  }
  auto realCpuLength = UInt::constant(tlSizeField, 32, b, loc);

  // Use getValueInRule to get cpuAddr with proper FIRRTL conversion
  auto cpuAddrValue = getValueInRule(cpuAddr, op.getOperation(), cpuAddrOperandId, b, localMap, loc);
  if (failed(cpuAddrValue)) {
    op.emitError("Failed to get cpuAddr for burst store");
    llvm::report_fatal_error("Burst store cpuAddr resolution failed");
  }

  // Call DMA interface
  auto dmaItfc = bbHandler->getDmaInterface();
  if (!dmaItfc) {
    op.emitError("DMA interface not available");
    return failure();
  }

  dmaItfc->callMethod("isax_to_cpu",
                      {*cpuAddrValue, localAddr.bits(31, 0).getValue(),
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