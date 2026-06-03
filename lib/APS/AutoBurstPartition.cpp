#include "APS/APSOps.h"
#include "APS/PassDetail.h"
#include "APS/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"
#include <optional>

#define DEBUG_TYPE "auto-burst-partition"

namespace {
using namespace mlir;

constexpr int64_t kBurstBeatBits = 64;

bool hasUserPartitionAttrs(memref::GlobalOp globalOp) {
  return globalOp->hasAttr("partition_dim_array") ||
         globalOp->hasAttr("partition_factor_array") ||
         globalOp->hasAttr("partition_cyclic_array");
}

memref::GlobalOp getGlobalForMemref(Value memrefValue) {
  auto getGlobalOp = memrefValue.getDefiningOp<memref::GetGlobalOp>();
  if (!getGlobalOp)
    return nullptr;

  auto moduleOp = getGlobalOp->getParentOfType<ModuleOp>();
  if (!moduleOp)
    return nullptr;

  SymbolTable symbolTable(moduleOp);
  return symbolTable.lookup<memref::GlobalOp>(getGlobalOp.getName());
}

std::optional<int64_t> getDefaultBurstPartitionFactor(memref::GlobalOp globalOp) {
  auto memrefType = dyn_cast<MemRefType>(globalOp.getType());
  if (!memrefType || memrefType.getRank() != 1 || !memrefType.hasStaticShape())
    return std::nullopt;

  int64_t bitWidth = memrefType.getElementTypeBitWidth();
  if (bitWidth <= 0 || kBurstBeatBits % bitWidth != 0)
    return std::nullopt;

  int64_t factor = kBurstBeatBits / bitWidth;
  factor = std::min<int64_t>(factor, memrefType.getDimSize(0));
  if (factor <= 1)
    return std::nullopt;

  return factor;
}

void addCyclicPartitionAttrs(memref::GlobalOp globalOp, int64_t factor) {
  OpBuilder builder(globalOp.getContext());
  auto i32Type = builder.getI32Type();

  globalOp->setAttr("partition_dim_array",
                    builder.getI32ArrayAttr({0}));
  globalOp->setAttr("partition_factor_array",
                    ArrayAttr::get(globalOp.getContext(),
                                   {IntegerAttr::get(i32Type, factor)}));
  globalOp->setAttr("partition_cyclic_array",
                    builder.getI32ArrayAttr({1}));

  LLVM_DEBUG(llvm::dbgs() << "Auto burst partition: " << globalOp.getSymName()
                          << " factor=" << factor << "\n");
}

struct AutoBurstPartitionPass
    : public AutoBurstPartitionBase<AutoBurstPartitionPass> {
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    DenseSet<Operation *> burstGlobals;

    auto collectMemref = [&](Value memref) {
      if (auto globalOp = getGlobalForMemref(memref))
        burstGlobals.insert(globalOp.getOperation());
    };

    moduleOp.walk([&](Operation *op) {
      if (auto burstLoad = dyn_cast<aps::MemBurstLoad>(op)) {
        for (Value memref : burstLoad.getMemrefs())
          collectMemref(memref);
        return;
      }

      if (auto burstStore = dyn_cast<aps::MemBurstStore>(op)) {
        for (Value memref : burstStore.getMemrefs())
          collectMemref(memref);
      }
    });

    for (Operation *op : burstGlobals) {
      auto globalOp = cast<memref::GlobalOp>(op);
      if (hasUserPartitionAttrs(globalOp))
        continue;

      std::optional<int64_t> factor =
          getDefaultBurstPartitionFactor(globalOp);
      if (!factor)
        continue;

      addCyclicPartitionAttrs(globalOp, *factor);
    }
  }
};

} // namespace

namespace mlir {
std::unique_ptr<OperationPass<mlir::ModuleOp>> createAutoBurstPartitionPass() {
  return std::make_unique<AutoBurstPartitionPass>();
}
} // namespace mlir
