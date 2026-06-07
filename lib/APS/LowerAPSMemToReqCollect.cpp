#include "TOR/Passes.h"
#include "TOR/TOR.h"
#include "APS/APSOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "lower-aps-mem-to-req-collect"

namespace {
using namespace mlir;

std::pair<int, int> getRefTimePair(Operation *op) {
  auto startAttr = op->getAttrOfType<IntegerAttr>("ref_starttime");
  auto endAttr = op->getAttrOfType<IntegerAttr>("ref_endtime");
  assert(startAttr && endAttr &&
         "memory operation must have ref scheduling info");

  int startTime = startAttr.getInt();
  int endTime = std::max<int>(startTime + 1, endAttr.getInt());
  return {startTime, endTime};
}

LogicalResult requireRefTimePair(Operation *op) {
  bool hasRefTime = op->hasAttr("ref_starttime") && op->hasAttr("ref_endtime");
  bool hasTorTime = op->hasAttr("starttime") && op->hasAttr("endtime");
  if (hasRefTime && hasTorTime)
    return success();

  return op->emitOpError()
         << "missing paired ref_starttime/ref_endtime and starttime/endtime; "
            "run scheduling before lower-aps-mem-to-req-collect";
}

void setRefTimePair(Operation *op, int startTime, int endTime) {
  auto ctx = op->getContext();
  op->setAttr("ref_starttime",
              IntegerAttr::get(IntegerType::get(ctx, 32), startTime));
  op->setAttr("ref_endtime",
              IntegerAttr::get(IntegerType::get(ctx, 32), endTime));
  op->setAttr("starttime",
              IntegerAttr::get(IntegerType::get(ctx, 32), startTime));
  op->setAttr("endtime",
              IntegerAttr::get(IntegerType::get(ctx, 32), endTime));
}

void copyAttrsAndSetRefTime(Operation *from, Operation *to, int startTime,
                            int endTime) {
  for (auto attr : from->getAttrs())
    to->setAttr(attr.getName(), attr.getValue());
  setRefTimePair(to, startTime, endTime);
}

template <typename SourceOp, typename CollectOp, typename CreateReqFn>
void splitValueProducingMemoryOp(SourceOp sourceOp, IRRewriter &rewriter,
                                 CreateReqFn createReq) {
  auto [startTime, endTime] = getRefTimePair(sourceOp.getOperation());

  rewriter.setInsertionPoint(sourceOp);
  auto reqOp = createReq();
  copyAttrsAndSetRefTime(sourceOp.getOperation(), reqOp.getOperation(),
                         startTime, endTime);

  auto collectOp = rewriter.create<CollectOp>(
      sourceOp.getLoc(), sourceOp.getResult().getType(), reqOp.getResult());
  copyAttrsAndSetRefTime(sourceOp.getOperation(), collectOp.getOperation(),
                         endTime, endTime + 1);

  rewriter.replaceOp(sourceOp, collectOp.getResult());
}

template <typename SourceOp, typename CollectOp, typename CreateReqFn>
void splitSideEffectMemoryOp(SourceOp sourceOp, IRRewriter &rewriter,
                             CreateReqFn createReq) {
  auto [startTime, endTime] = getRefTimePair(sourceOp.getOperation());

  rewriter.setInsertionPoint(sourceOp);
  auto reqOp = createReq();
  copyAttrsAndSetRefTime(sourceOp.getOperation(), reqOp.getOperation(),
                         startTime, endTime);

  auto collectOp =
      rewriter.create<CollectOp>(sourceOp.getLoc(), reqOp.getResult());
  copyAttrsAndSetRefTime(sourceOp.getOperation(), collectOp.getOperation(),
                         endTime, endTime + 1);

  rewriter.eraseOp(sourceOp);
}

template <typename OpTy, typename HandleFn>
WalkResult collectScheduledOps(tor::FuncOp funcOp, HandleFn handle) {
  return funcOp.walk([&](OpTy op) {
    if (failed(requireRefTimePair(op.getOperation())))
      return WalkResult::interrupt();

    handle(op);
    return WalkResult::advance();
  });
}

struct LowerAPSMemToReqCollectPass
    : public PassWrapper<LowerAPSMemToReqCollectPass, OperationPass<tor::DesignOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerAPSMemToReqCollectPass)

  StringRef getArgument() const final { return "lower-aps-mem-to-req-collect"; }
  StringRef getDescription() const final {
    return "Split APS memory operations into request-collect pairs";
  }

  void runOnOperation() override {
    tor::DesignOp designOp = getOperation();

    // Process each function
    for (auto funcOp : designOp.getOps<tor::FuncOp>()) {
      // Only process scheduled functions
      if (!funcOp->hasAttr("scheduled")) {
        llvm::dbgs() << "Skipping unscheduled function: " << funcOp.getName() << "\n";
        continue;
      }

      llvm::dbgs() << "\n============================================\n";
      llvm::dbgs() << "Processing scheduled function: " << funcOp.getName() << "\n";
      llvm::dbgs() << "============================================\n";

      IRRewriter rewriter(&getContext());

      // Collect operations to transform (can't modify while walking)
      SmallVector<aps::ReadSmem> spmloads;
      SmallVector<aps::LoadBy> gmemLoads;
      SmallVector<aps::StoreBy> gmemStores;
      SmallVector<aps::CopyBy> copies;

      if (collectScheduledOps<aps::ReadSmem>(funcOp, [&](aps::ReadSmem loadOp) {
            spmloads.push_back(loadOp);
          }).wasInterrupted())
        return signalPassFailure();

      if (collectScheduledOps<aps::LoadBy>(funcOp, [&](aps::LoadBy loadOp) {
            gmemLoads.push_back(loadOp);
          }).wasInterrupted())
        return signalPassFailure();

      if (collectScheduledOps<aps::StoreBy>(funcOp, [&](aps::StoreBy storeOp) {
            gmemStores.push_back(storeOp);
          }).wasInterrupted())
        return signalPassFailure();

      if (collectScheduledOps<aps::CopyBy>(funcOp, [&](aps::CopyBy copyOp) {
            copies.push_back(copyOp);
          }).wasInterrupted())
        return signalPassFailure();

      // Transform scalar CPU/global memory loads.
      for (auto loadOp : gmemLoads) {
        splitValueProducingMemoryOp<aps::LoadBy, aps::LoadWait>(
            loadOp, rewriter, [&]() {
              return rewriter.create<aps::LoadIssue>(
                  loadOp.getLoc(), loadOp.getResult().getType(),
                  loadOp.getCpuAddr());
            });
      }

      // Transform spm memloads
      for (auto spmLoadOp : spmloads) {
        splitValueProducingMemoryOp<aps::ReadSmem, aps::ReadSmemWait>(
            spmLoadOp, rewriter, [&]() {
              return rewriter.create<aps::ReadSmemIssue>(
                  spmLoadOp.getLoc(), spmLoadOp.getResult().getType(),
                  spmLoadOp.getMemref(), spmLoadOp.getIndices());
            });
      }

      // Transform scalar CPU/global memory stores.
      for (auto storeOp : gmemStores) {
        splitSideEffectMemoryOp<aps::StoreBy, aps::StoreWait>(
            storeOp, rewriter, [&]() {
              return rewriter.create<aps::StoreIssue>(
                  storeOp.getLoc(), rewriter.getNoneType(),
                  storeOp.getValue(), storeOp.getCpuAddr());
            });
      }

      // Transform interface-bound bulk copies.
      for (auto copyOp : copies) {
        splitSideEffectMemoryOp<aps::CopyBy, aps::CopyWait>(
            copyOp, rewriter, [&]() {
              return rewriter.create<aps::CopyIssue>(
                  copyOp.getLoc(), rewriter.getNoneType(),
                  copyOp.getDirection(), copyOp.getItfcAttr().getValue(),
                  copyOp.getCpuAddr(), copyOp.getMemrefs(),
                  copyOp.getStart(), copyOp.getLength());
            });
      }

      llvm::dbgs() << "Transformed " << gmemLoads.size() << " gmem loads, "
                   << spmloads.size() << " spm loads, "
                   << gmemStores.size() << " gmem stores, "
                   << copies.size() << " copies\n";

      llvm::dbgs() << "Completed function: " << funcOp.getName() << "\n\n";
    }
  }
};

} // namespace

namespace mlir {
std::unique_ptr<OperationPass<tor::DesignOp>> createLowerAPSMemToReqCollectPass() {
  return std::make_unique<LowerAPSMemToReqCollectPass>();
}
} // namespace mlir
