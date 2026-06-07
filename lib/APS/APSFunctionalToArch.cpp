#include "APS/APSOps.h"
#include "APS/PassDetail.h"
#include "APS/Passes.h"
#include "TOR/TOR.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringMap.h"
#include <optional>

#define DEBUG_TYPE "aps-functional-to-arch"

namespace {
using namespace mlir;

constexpr llvm::StringLiteral kCpuItfc = "cpuitfc";
constexpr llvm::StringLiteral kBusItfc = "busitfc";

struct MemoryMapLookup {
  llvm::StringMap<aps::MemEntryOp> bankToEntry;
};

void copyAttrs(Operation *from, Operation *to) {
  for (NamedAttribute attr : from->getAttrs())
    to->setAttr(attr.getName(), attr.getValue());
}

void ensureMemItfc(ModuleOp moduleOp, llvm::StringRef name) {
  SymbolTable symbolTable(moduleOp);
  if (symbolTable.lookup(name))
    return;

  OpBuilder builder(moduleOp.getContext());
  builder.setInsertionPointToStart(moduleOp.getBody());
  builder.create<aps::MemItfc>(moduleOp.getLoc(), name);
}

std::optional<int64_t> getConstantInt(Value value) {
  auto constOp = value.getDefiningOp<arith::ConstantOp>();
  if (!constOp)
    return std::nullopt;

  auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue());
  if (!intAttr)
    return std::nullopt;

  return intAttr.getValue().getSExtValue();
}

Value createIntegerConstant(OpBuilder &builder, Location loc, Type type,
                            int64_t value) {
  auto intType = dyn_cast<IntegerType>(type);
  if (!intType)
    return {};

  return builder
      .create<arith::ConstantOp>(loc, IntegerAttr::get(intType, value))
      .getResult();
}

uint64_t getElementSizeBytes(MemRefType memrefType) {
  unsigned bitWidth = memrefType.getElementTypeBitWidth();
  return (bitWidth + 7) / 8;
}

bool isSingleElementMemref(Value memref) {
  auto memrefType = dyn_cast<MemRefType>(memref.getType());
  return memrefType && memrefType.hasStaticShape() &&
         memrefType.getNumElements() == 1;
}

std::optional<StringRef> getGlobalName(Value memref) {
  auto getGlobal = memref.getDefiningOp<memref::GetGlobalOp>();
  if (!getGlobal)
    return std::nullopt;
  return getGlobal.getName();
}

MemoryMapLookup buildMemoryMapLookup(ModuleOp moduleOp) {
  MemoryMapLookup lookup;
  moduleOp.walk([&](aps::MemoryMapOp memoryMap) {
    for (auto &block : memoryMap.getRegion()) {
      for (auto entry : block.getOps<aps::MemEntryOp>()) {
        for (Attribute bankAttr : entry.getBankSymbols()) {
          auto symbol = dyn_cast<FlatSymbolRefAttr>(bankAttr);
          if (!symbol)
            continue;
          lookup.bankToEntry[symbol.getValue()] = entry;
        }
      }
    }
  });
  return lookup;
}

aps::MemEntryOp getCommonMemoryEntry(aps::Copy copy,
                                     const MemoryMapLookup &lookup) {
  aps::MemEntryOp commonEntry;
  for (Value memref : copy.getMemrefs()) {
    auto globalName = getGlobalName(memref);
    if (!globalName)
      return {};

    auto it = lookup.bankToEntry.find(*globalName);
    if (it == lookup.bankToEntry.end())
      return {};

    if (!commonEntry) {
      commonEntry = it->second;
      continue;
    }
    if (commonEntry != it->second)
      return {};
  }
  return commonEntry;
}

bool isFullyExpandedCopy(aps::Copy copy, const MemoryMapLookup &lookup) {
  if (copy.getMemrefs().empty())
    return false;

  auto entry = getCommonMemoryEntry(copy, lookup);
  if (!entry)
    return false;

  if (entry.getNumBanks() != copy.getMemrefs().size())
    return false;

  for (Value memref : copy.getMemrefs()) {
    if (!isSingleElementMemref(memref))
      return false;
  }

  return true;
}

LogicalResult lowerFullyExpandedCopy(aps::Copy copy, IRRewriter &rewriter,
                                     FlatSymbolRefAttr cpuItfc,
                                     const MemoryMapLookup &lookup) {
  if (!isFullyExpandedCopy(copy, lookup))
    return failure();

  auto start = getConstantInt(copy.getStart());
  auto length = getConstantInt(copy.getLength());
  if (!start || !length) {
    copy.emitError("fully partitioned copy requires constant start and length");
    return failure();
  }
  if (*start < 0 || *length < 0) {
    copy.emitError("fully partitioned copy requires non-negative start and "
                   "length");
    return failure();
  }

  auto entry = getCommonMemoryEntry(copy, lookup);
  bool cyclic = entry.getCyclic() != 0;
  int64_t numBanks = static_cast<int64_t>(copy.getMemrefs().size());
  if (*start + *length > numBanks) {
    copy.emitError("fully partitioned copy range exceeds available banks");
    return failure();
  }

  auto firstMemrefType = dyn_cast<MemRefType>(copy.getMemrefs()[0].getType());
  if (!firstMemrefType)
    return failure();

  Location loc = copy.getLoc();
  uint64_t elementSizeBytes = getElementSizeBytes(firstMemrefType);
  if (elementSizeBytes == 0) {
    copy.emitError("cannot infer element size for fully partitioned copy");
    return failure();
  }

  Type indexType = copy.getStart().getType();
  Type cpuAddrType = copy.getCpuAddr().getType();
  Type elementType = firstMemrefType.getElementType();

  rewriter.setInsertionPoint(copy);
  for (int64_t i = 0; i < *length; ++i) {
    int64_t logicalIndex = *start + i;
    int64_t bank = cyclic ? logicalIndex % numBanks : logicalIndex;
    int64_t localIndex = cyclic ? logicalIndex / numBanks : 0;
    if (bank < 0 || bank >= numBanks || localIndex != 0) {
      copy.emitError("fully partitioned copy could not map element to a "
                     "single bank");
      return failure();
    }

    Value memref = copy.getMemrefs()[bank];
    Value localIndexValue = createIntegerConstant(rewriter, loc, indexType,
                                                  localIndex);
    if (!localIndexValue)
      return failure();

    Value cpuAddr = copy.getCpuAddr();
    int64_t byteOffset = i * static_cast<int64_t>(elementSizeBytes);
    if (byteOffset != 0) {
      Value offsetValue =
          createIntegerConstant(rewriter, loc, cpuAddrType, byteOffset);
      if (!offsetValue)
        return failure();
      cpuAddr = rewriter.create<arith::AddIOp>(loc, cpuAddr, offsetValue);
    }

    if (copy.getDirection() == aps::CopyDirection::In) {
      auto load = rewriter.create<aps::LoadBy>(loc, elementType, cpuItfc,
                                               cpuAddr);
      rewriter.create<aps::WriteSmem>(loc, load.getResult(), memref,
                                      ValueRange{localIndexValue});
    } else {
      auto load = rewriter.create<aps::ReadSmem>(
          loc, elementType, memref, ValueRange{localIndexValue});
      rewriter.create<aps::StoreBy>(loc, cpuItfc, load.getResult(), cpuAddr);
    }
  }

  rewriter.eraseOp(copy);
  return success();
}

struct APSFunctionalToArchPass
    : public APSFunctionalToArchBase<APSFunctionalToArchPass> {
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    ensureMemItfc(moduleOp, kCpuItfc);
    ensureMemItfc(moduleOp, kBusItfc);

    IRRewriter rewriter(&getContext());
    auto cpuItfc = FlatSymbolRefAttr::get(&getContext(), kCpuItfc);
    MemoryMapLookup memoryMapLookup = buildMemoryMapLookup(moduleOp);

    SmallVector<aps::Load> loads;
    SmallVector<aps::Store> stores;
    SmallVector<aps::Copy> copies;
    moduleOp.walk([&](Operation *op) {
      if (auto load = dyn_cast<aps::Load>(op))
        loads.push_back(load);
      else if (auto store = dyn_cast<aps::Store>(op))
        stores.push_back(store);
      else if (auto copy = dyn_cast<aps::Copy>(op))
        copies.push_back(copy);
    });

    for (auto load : loads) {
      rewriter.setInsertionPoint(load);
      auto newLoad = rewriter.create<aps::LoadBy>(
          load.getLoc(), load.getResult().getType(), cpuItfc,
          load.getCpuAddr());
      copyAttrs(load.getOperation(), newLoad.getOperation());
      rewriter.replaceOp(load, newLoad.getResult());
    }

    for (auto store : stores) {
      rewriter.setInsertionPoint(store);
      auto newStore = rewriter.create<aps::StoreBy>(
          store.getLoc(), cpuItfc, store.getValue(), store.getCpuAddr());
      copyAttrs(store.getOperation(), newStore.getOperation());
      rewriter.eraseOp(store);
    }

    for (auto copy : copies) {
      if (isFullyExpandedCopy(copy, memoryMapLookup)) {
        if (failed(lowerFullyExpandedCopy(copy, rewriter, cpuItfc,
                                          memoryMapLookup)))
          return signalPassFailure();
        continue;
      }

      rewriter.setInsertionPoint(copy);
      auto newCopy = rewriter.create<aps::CopyBy>(
          copy.getLoc(), copy.getDirection(), kBusItfc, copy.getCpuAddr(),
          copy.getMemrefs(), copy.getStart(), copy.getLength());
      copyAttrs(copy.getOperation(), newCopy.getOperation());
      rewriter.eraseOp(copy);
    }
  }
};

} // namespace

namespace mlir {
std::unique_ptr<OperationPass<ModuleOp>> createAPSFunctionalToArchPass() {
  return std::make_unique<APSFunctionalToArchPass>();
}
} // namespace mlir
