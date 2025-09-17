#include "TOR/PassDetail.h"
#include "TOR/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "ignore-aliasing-licm"

namespace {
using namespace mlir;

bool isHasSameMemrefStore(Value mem, Region *cur) {
  for (Operation &op : cur->getOps()) {
    if (op.getNumRegions()) {
      for (Region &region : op.getRegions()) {
        if (isHasSameMemrefStore(mem, &region)) {
          return true;
        }
      }
    }
    if (isa<func::CallOp>(op)) {
      for (Value operand : op.getOperands()) {
        if (mem == operand) {
          return true;
        }
      }
    }
    if (auto store = dyn_cast<affine::AffineStoreOp>(op)) {
      if (mem == store.getMemRef()) {
        return true;
      }
    }
    if (auto store = dyn_cast<memref::StoreOp>(op)) {
      if (mem == store.getMemref()) {
        return true;
      }
    }
  }
  return false;
}

bool hlsMoveOutOfRegion(Operation *op, Region *cur) {
  if (auto load = dyn_cast<affine::AffineLoadOp>(op)) {
    return !isHasSameMemrefStore(load.getMemRef(), cur);
  }
  if (auto load = dyn_cast<memref::LoadOp>(op)) {
    return !isHasSameMemrefStore(load.getMemref(), cur);
  }
  return isMemoryEffectFree(op) && isSpeculatable(op);
}

struct IgnoreAliasingLICMPass : IgnoreAliasingLICMBase<IgnoreAliasingLICMPass> {
  void runOnOperation() override {
    getOperation()->walk([&](LoopLikeOpInterface loopLike) {
      // pipeline pragma will auto add loop_flatten, so don't need judge
      if (loopLike->hasAttr("pipeline")) {
        return WalkResult::skip();
      }
      LLVM_DEBUG(loopLike.print(llvm::dbgs() << "\nOriginal loop:\n"));
      moveLoopInvariantCode(
          loopLike.getLoopRegions(),
          [&](Value value, Region *) {
            return loopLike.isDefinedOutsideOfLoop(value);
          },
          [&](Operation *op, Region *cur) {
            return hlsMoveOutOfRegion(op, cur);
          },
          [&](Operation *op, Region *) { loopLike.moveOutOfLoop(op); });
      return WalkResult::advance();
    });
  }
};

} // namespace

namespace mlir {
std::unique_ptr<OperationPass<mlir::ModuleOp>> createIgnoreAliasingLICMPass() {
  return std::make_unique<IgnoreAliasingLICMPass>();
}

} // namespace mlir
