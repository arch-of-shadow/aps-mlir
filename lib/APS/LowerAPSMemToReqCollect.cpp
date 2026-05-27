#include "TOR/Passes.h"
#include "TOR/TOR.h"
#include "APS/APSOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "lower-aps-mem-to-req-collect"

namespace {
using namespace mlir;

std::pair<int, int> getRefTimePair(Operation *op) {
  auto startAttr = op->getAttrOfType<IntegerAttr>("ref_starttime");
  auto endAttr = op->getAttrOfType<IntegerAttr>("ref_endtime");

  assert(startAttr && endAttr && "memory operation must be scheduled first");

  return {startAttr.getInt(), endAttr.getInt()};
}

LogicalResult requireRefTimePair(Operation *op) {
  if (op->hasAttr("ref_starttime") && op->hasAttr("ref_endtime"))
    return success();

  return op->emitOpError()
         << "missing ref_starttime/ref_endtime; run scheduling before "
            "lower-aps-mem-to-req-collect";
}

void setRefTimePair(Operation *op, int startTime, int endTime) {
  auto ctx = op->getContext();
  op->setAttr("ref_starttime",
              IntegerAttr::get(IntegerType::get(ctx, 32), startTime));
  op->setAttr("ref_endtime",
              IntegerAttr::get(IntegerType::get(ctx, 32), endTime));
}

/// Helper: Check if a value comes from _cpu_memory
bool isFromCpuMemory(Value memref) {
  // Trace back to see if this memref comes from a get_global with "_cpu_memory"
  auto defOp = memref.getDefiningOp();
  if (!defOp)
    return false;

  if (auto getGlobal = dyn_cast<memref::GetGlobalOp>(defOp)) {
    StringRef name = getGlobal.getName();
    return name.contains("_cpu_memory");
  }

  return false;
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
      SmallVector<aps::MemLoad> memloads;
      SmallVector<aps::MemLoad> spmloads;
      SmallVector<aps::MemStore> memstores;
      SmallVector<aps::MemBurstLoad> burstloads;
      SmallVector<aps::MemBurstStore> burststores;

      if (collectScheduledOps<aps::MemLoad>(funcOp, [&](aps::MemLoad loadOp) {
            if (isFromCpuMemory(loadOp.getMemref()))
              memloads.push_back(loadOp);
            else
              spmloads.push_back(loadOp);
          }).wasInterrupted())
        return signalPassFailure();

      if (collectScheduledOps<aps::MemStore>(
              funcOp, [&](aps::MemStore storeOp) {
                if (isFromCpuMemory(storeOp.getMemref()))
                  memstores.push_back(storeOp);
              }).wasInterrupted())
        return signalPassFailure();

      if (collectScheduledOps<aps::MemBurstLoad>(
              funcOp, [&](aps::MemBurstLoad burstOp) {
                burstloads.push_back(burstOp);
              }).wasInterrupted())
        return signalPassFailure();

      if (collectScheduledOps<aps::MemBurstStore>(
              funcOp, [&](aps::MemBurstStore burstOp) {
                burststores.push_back(burstOp);
              }).wasInterrupted())
        return signalPassFailure();

      // Transform memloads
      for (auto loadOp : memloads) {
        splitValueProducingMemoryOp<aps::MemLoad, aps::ItfcLoadCollect>(
            loadOp, rewriter, [&]() {
              return rewriter.create<aps::ItfcLoadReq>(
                  loadOp.getLoc(), loadOp.getResult().getType(),
                  loadOp.getMemref(), loadOp.getIndices());
            });
      }

      // Transform spm memloads
      for (auto spmLoadOp : spmloads) {
        splitValueProducingMemoryOp<aps::MemLoad, aps::SpmLoadCollect>(
            spmLoadOp, rewriter, [&]() {
              return rewriter.create<aps::SpmLoadReq>(
                  spmLoadOp.getLoc(), spmLoadOp.getResult().getType(),
                  spmLoadOp.getMemref(), spmLoadOp.getIndices());
            });
      }

      // Transform memstores
      for (auto storeOp : memstores) {
        splitSideEffectMemoryOp<aps::MemStore, aps::ItfcStoreCollect>(
            storeOp, rewriter, [&]() {
              return rewriter.create<aps::ItfcStoreReq>(
                  storeOp.getLoc(), rewriter.getNoneType(),
                  storeOp.getValue(), storeOp.getMemref(),
                  storeOp.getIndices());
            });
      }

      // Transform burst loads
      for (auto burstOp : burstloads) {
        splitSideEffectMemoryOp<aps::MemBurstLoad,
                                aps::ItfcBurstLoadCollect>(
            burstOp, rewriter, [&]() {
              return rewriter.create<aps::ItfcBurstLoadReq>(
                  burstOp.getLoc(), rewriter.getNoneType(),
                  burstOp.getCpuAddr(), burstOp.getMemrefs(),
                  burstOp.getStart(), burstOp.getLength());
            });
      }

      // Transform burst stores
      for (auto burstOp : burststores) {
        splitSideEffectMemoryOp<aps::MemBurstStore,
                                aps::ItfcBurstStoreCollect>(
            burstOp, rewriter, [&]() {
              return rewriter.create<aps::ItfcBurstStoreReq>(
                  burstOp.getLoc(), rewriter.getNoneType(),
                  burstOp.getMemrefs(), burstOp.getStart(),
                  burstOp.getCpuAddr(), burstOp.getLength());
            });
      }

      llvm::dbgs() << "Transformed " << memloads.size() << " loads, "
                   << spmloads.size() << " spm loads, "
                   << memstores.size() << " stores, "
                   << burstloads.size() << " burst loads, "
                   << burststores.size() << " burst stores\n";

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
