#include "APS/Passes.h"
#include "APS/PassDetail.h"
#include "APS/APSOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "aps-memory-map"

namespace {
using namespace mlir;
using namespace mlir::memref;

uint32_t readFirstUI32ArrayAttr(Operation *op, StringRef attrName,
                                uint32_t defaultValue) {
  auto arrayAttr = op->getAttrOfType<ArrayAttr>(attrName);
  if (!arrayAttr || arrayAttr.empty())
    return defaultValue;

  auto valueAttr = llvm::dyn_cast<IntegerAttr>(arrayAttr[0]);
  if (!valueAttr)
    return defaultValue;

  return valueAttr.getValue().getZExtValue();
}

struct MemoryGroup {
  std::string originalName;
  // Initial memory-map construction only sees the globals currently present in
  // the module. Before array partitioning this is normally a single bank.
  llvm::SmallVector<GlobalOp, 4> banks;
  uint32_t cyclicMode = 0;  // 1 = cyclic, 0 = block

  MemoryGroup() = default;
  MemoryGroup(StringRef name) : originalName(name.str()) {}
};

/// Pass to seed the APS memory map from global memrefs.
struct APSMemoryMapPass
    : public APSMemoryMapBase<APSMemoryMapPass> {
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    OpBuilder builder(moduleOp.getContext());

    llvm::StringMap<MemoryGroup> memoryGroups;
    collectMemoryGroups(moduleOp, memoryGroups);

    if (memoryGroups.empty())
      return;

    createMemoryMap(moduleOp, builder, memoryGroups);
  }

private:
  void collectMemoryGroups(ModuleOp moduleOp,
                           llvm::StringMap<MemoryGroup> &groups) {
    moduleOp.walk([&](GlobalOp globalOp) {
      auto varNameAttr = globalOp->getAttrOfType<StringAttr>("var_name");
      if (!varNameAttr) {
        return;
      }

      std::string varName = varNameAttr.getValue().str();

      if (groups.find(varName) == groups.end()) {
        groups[varName] = MemoryGroup(varName);
        groups[varName].cyclicMode =
            readFirstUI32ArrayAttr(globalOp, "partition_cyclic_array",
                                   /*defaultValue=*/0);
      }

      groups[varName].banks.push_back(globalOp);
    });
  }

  void createMemoryMap(ModuleOp moduleOp, OpBuilder &builder,
                       llvm::StringMap<MemoryGroup> &groups) {
    builder.setInsertionPointToStart(moduleOp.getBody());

    auto memoryMapOp = builder.create<aps::MemoryMapOp>(moduleOp.getLoc());
    Block *mapBody = builder.createBlock(&memoryMapOp.getRegion());
    builder.setInsertionPointToStart(mapBody);

    // Track current address
    uint32_t currentAddress = 0;

    // Process each memref group in a consistent order
    llvm::SmallVector<StringRef, 8> groupNames;
    for (auto &entry : groups) {
      groupNames.push_back(entry.getKey());
    }
    llvm::sort(groupNames);

    for (StringRef groupName : groupNames) {
      auto &group = groups[groupName];

      // Calculate bank size (use first bank's size)
      uint32_t bankSize = 0;
      if (!group.banks.empty()) {
        auto memrefType = llvm::cast<MemRefType>(group.banks[0].getType());
        uint64_t numElements = memrefType.getNumElements();
        uint32_t elementSize = memrefType.getElementTypeBitWidth() / 8;
        bankSize = numElements * elementSize;
      }

      // Create array of bank symbol attributes
      llvm::SmallVector<Attribute, 4> bankSymbols;
      for (auto globalOp : group.banks) {
        bankSymbols.push_back(FlatSymbolRefAttr::get(globalOp.getSymNameAttr()));
      }

      // This is the number of banks already present in the IR. Initial maps
      // normally emit count=1; partitioning rewrites the entry later.
      uint32_t actualNumBanks = group.banks.size();

      // Create mem_entry operation
      builder.create<aps::MemEntryOp>(
          moduleOp.getLoc(), group.originalName, bankSymbols, currentAddress,
          bankSize, actualNumBanks, group.cyclicMode);

      // Update address for next group
      currentAddress += bankSize * actualNumBanks;

      // Align to next power-of-2 boundary to ensure each array occupies 2^n bytes
      // This prevents burst accesses from crossing into other arrays
      uint32_t totalSize = bankSize * actualNumBanks;
      if (totalSize > 0) {
        // Find next power of 2 >= totalSize
        uint32_t alignedSize = 1;
        while (alignedSize < totalSize) {
          alignedSize <<= 1;
        }
        // Align currentAddress to this power-of-2 boundary
        currentAddress =
            ((currentAddress + alignedSize - 1) / alignedSize) * alignedSize;
      }

      LLVM_DEBUG(llvm::dbgs() << "Memory map entry: " << group.originalName
                              << " at 0x"
                              << llvm::utohexstr(currentAddress -
                                                 bankSize * actualNumBanks)
                              << " size=" << bankSize
                              << " banks=" << actualNumBanks
                              << " cyclic=" << group.cyclicMode << "\n");
    }

    // Create terminator
    builder.create<aps::MemFinishOp>(moduleOp.getLoc());
  }
};

} // namespace

namespace mlir {
std::unique_ptr<OperationPass<mlir::ModuleOp>> createAPSMemoryMapPass() {
  return std::make_unique<APSMemoryMapPass>();
}
} // namespace mlir
