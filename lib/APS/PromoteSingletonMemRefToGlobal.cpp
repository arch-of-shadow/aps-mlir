#include "APS/Passes.h"
#include "APS/PassDetail.h"
#include "APS/APSOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "promote-singleton-memref-to-global"

namespace {

using namespace mlir;
using namespace mlir::memref;

static bool isSingleElementMemRef(Type type) {
  auto memrefType = llvm::dyn_cast<MemRefType>(type);
  if (!memrefType || memrefType.getRank() != 1)
    return false;

  return memrefType.getDimSize(0) == 1;
}

static FlatSymbolRefAttr getPromotableGlobalSymbol(Value memref) {
  if (!isSingleElementMemRef(memref.getType()))
    return {};

  auto getGlobalOp = memref.getDefiningOp<GetGlobalOp>();
  if (!getGlobalOp)
    return {};

  return FlatSymbolRefAttr::get(memref.getContext(), getGlobalOp.getName());
}

/// Pattern to convert aps.memload on memref<1xT> to aps.globalload
struct ScalarMemLoadToGlobalLoadPattern : public OpRewritePattern<aps::MemLoad> {
  using OpRewritePattern<aps::MemLoad>::OpRewritePattern;

  LogicalResult matchAndRewrite(aps::MemLoad loadOp,
                                PatternRewriter &rewriter) const override {
    auto symbolRef = getPromotableGlobalSymbol(loadOp.getMemref());
    if (!symbolRef)
      return failure();

    auto globalLoadOp = rewriter.create<aps::GlobalLoad>(
        loadOp.getLoc(), loadOp.getResult().getType(), symbolRef);
    rewriter.replaceOp(loadOp, globalLoadOp.getResult());
    return success();
  }
};

/// Pattern to convert aps.memstore on memref<1xT> to aps.globalstore
struct ScalarMemStoreToGlobalStorePattern : public OpRewritePattern<aps::MemStore> {
  using OpRewritePattern<aps::MemStore>::OpRewritePattern;

  LogicalResult matchAndRewrite(aps::MemStore storeOp,
                                PatternRewriter &rewriter) const override {
    auto symbolRef = getPromotableGlobalSymbol(storeOp.getMemref());
    if (!symbolRef)
      return failure();

    rewriter.create<aps::GlobalStore>(storeOp.getLoc(), storeOp.getValue(),
                                      symbolRef);
    rewriter.eraseOp(storeOp);
    return success();
  }
};

struct PromoteSingletonMemRefToGlobalPass : PromoteSingletonMemRefToGlobalBase<PromoteSingletonMemRefToGlobalPass> {
  void runOnOperation() override {
    auto op = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns.add<ScalarMemLoadToGlobalLoadPattern, ScalarMemStoreToGlobalStorePattern>(
        &getContext());
    GreedyRewriteConfig config;
    config.setStrictness(GreedyRewriteStrictness::ExistingOps);
    config.enableFolding();
    if (failed(applyPatternsGreedily(op, std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

} // namespace

namespace mlir {
std::unique_ptr<OperationPass<func::FuncOp>> createPromoteSingletonMemRefToGlobalPass() {
  return std::make_unique<PromoteSingletonMemRefToGlobalPass>();
}
} // namespace mlir
