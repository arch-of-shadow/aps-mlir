//===- PlaceReadRFAtEntry.cpp - Place aps.read_irf at function entry --------===//
//
// This pass moves all aps.read_irf operations to the entry block of their
// containing function. This ensures they can be scheduled at cycle 0.
//
//===----------------------------------------------------------------------===//

#include "APS/APSOps.h"
#include "APS/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"

#define DEBUG_TYPE "place-readrf-at-entry"

namespace mlir {

#define GEN_PASS_DEF_PLACEREADRFATENTRY
#include "APS/Passes.h.inc"

namespace {

struct PlaceReadRFAtEntryPass
    : public impl::PlaceReadRFAtEntryBase<PlaceReadRFAtEntryPass> {
  void runOnOperation() override {
    auto funcOp = getOperation();
    placeReadRFInFunction(funcOp);
  }

private:
  void placeReadRFInFunction(func::FuncOp funcOp) {
    if (funcOp.getBody().empty())
      return;

    Block &entryBlock = funcOp.getBody().front();

    llvm::DenseSet<Operation *> readRfUsers;
    for (BlockArgument arg : funcOp.getArguments()) {
      for (Operation *user : arg.getUsers()) {
        if (llvm::isa<aps::ReadIRF>(user))
          readRfUsers.insert(user);
      }
    }

    if (readRfUsers.empty())
      return;

    Operation *insertionPoint = entryBlock.getTerminator();
    for (auto &op : entryBlock) {
      if (!llvm::isa<aps::ReadIRF>(op)) {
        insertionPoint = &op;
        break;
      }
    }

    for (Operation *readRfOp : readRfUsers)
      readRfOp->moveBefore(insertionPoint);
  }
};

} // namespace

std::unique_ptr<Pass> createPlaceReadRFAtEntryPass() {
  return std::make_unique<PlaceReadRFAtEntryPass>();
}

} // namespace mlir
