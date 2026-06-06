#include "APS/Passes.h"
#include "APS/PassDetail.h"
#include "APS/APSOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "memref-to-aps"

namespace aps {
mlir::SmallVector<mlir::Value>
castMemoryIndicesToI32(mlir::OpBuilder &builder, mlir::Location loc,
                       mlir::ValueRange indices);
} // namespace aps

namespace {

using namespace mlir;
using namespace mlir::memref;

// Pattern to convert memref.load to aps.memload
struct MemRefLoadToAPSMemLoadPattern : public OpRewritePattern<LoadOp> {
  using OpRewritePattern<LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    Location loc = loadOp.getLoc();

    // Get the indices from memref.load
    SmallVector<Value> indices(loadOp.getIndices().begin(),
                               loadOp.getIndices().end());

    // Cast indices from index to i32 type
    SmallVector<Value> i32CastedIndices =
        aps::castMemoryIndicesToI32(rewriter, loc, indices);

    // Get the result type from the original load
    Type resultType = loadOp.getResult().getType();

    // Create aps.memload with i32-typed indices
    auto apsLoadOp = rewriter.create<aps::MemLoad>(
        loc, resultType, loadOp.getMemRef(), i32CastedIndices);

    // Replace the memref.load
    rewriter.replaceOp(loadOp, apsLoadOp.getResult());

    LLVM_DEBUG(llvm::dbgs() << "Converted memref.load to aps.memload\n");
    return success();
  }
};

// Pattern to convert memref.store to aps.memstore
struct MemRefStoreToAPSMemStorePattern : public OpRewritePattern<StoreOp> {
  using OpRewritePattern<StoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(StoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    Location loc = storeOp.getLoc();

    // Get the indices from memref.store
    SmallVector<Value> indices(storeOp.getIndices().begin(),
                               storeOp.getIndices().end());

    // Cast indices from index to i32 type
    SmallVector<Value> i32CastedIndices =
        aps::castMemoryIndicesToI32(rewriter, loc, indices);

    // Create aps.memstore with i32-typed indices
    rewriter.create<aps::MemStore>(loc, storeOp.getValue(),
                                   storeOp.getMemRef(), i32CastedIndices);

    // Erase the memref.store
    rewriter.eraseOp(storeOp);

    LLVM_DEBUG(llvm::dbgs() << "Converted memref.store to aps.memstore\n");
    return success();
  }
};

struct MemRefToAPSPass : MemRefToAPSBase<MemRefToAPSPass> {
  void runOnOperation() override {
    auto op = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns.add<MemRefLoadToAPSMemLoadPattern, MemRefStoreToAPSMemStorePattern>(
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
std::unique_ptr<OperationPass<func::FuncOp>> createMemRefToAPSPass() {
  return std::make_unique<MemRefToAPSPass>();
}
} // namespace mlir
