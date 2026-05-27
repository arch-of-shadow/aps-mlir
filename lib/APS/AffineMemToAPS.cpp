#include "APS/Passes.h"
#include "APS/PassDetail.h"
#include "APS/APSOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "affine-mem-to-aps"

namespace aps {
mlir::SmallVector<mlir::Value>
castMemoryIndicesToI32(mlir::OpBuilder &builder, mlir::Location loc,
                       mlir::ValueRange indices);
} // namespace aps

namespace {

using namespace mlir;
using namespace mlir::affine;

// Pattern to convert affine.load to aps.memload
struct AffineLoadToAPSMemLoadPattern : public OpRewritePattern<AffineLoadOp> {
  using OpRewritePattern<AffineLoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineLoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    Location loc = loadOp.getLoc();

    // Expand the affine map to arithmetic operations
    SmallVector<Value, 8> indices(loadOp.getMapOperands());
    auto maybeExpandedMap =
        expandAffineMap(rewriter, loc, loadOp.getAffineMap(), indices);
    if (!maybeExpandedMap)
      return failure();

    // Cast indices from index to i32 type
    SmallVector<Value> i32CastedIndices =
        aps::castMemoryIndicesToI32(rewriter, loc, *maybeExpandedMap);

    // Get the result type from the original load
    Type resultType = loadOp.getResult().getType();

    // Create aps.memload with i32-typed indices
    auto apsLoadOp = rewriter.create<aps::MemLoad>(
        loc, resultType, loadOp.getMemRef(), i32CastedIndices);

    // Replace the affine.load
    rewriter.replaceOp(loadOp, apsLoadOp.getResult());

    LLVM_DEBUG(llvm::dbgs() << "Converted affine.load to aps.memload (indices: "
                            << maybeExpandedMap->size() << ")\n");
    return success();
  }
};

// Pattern to convert affine.store to aps.memstore
struct AffineStoreToAPSMemStorePattern : public OpRewritePattern<AffineStoreOp> {
  using OpRewritePattern<AffineStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineStoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    Location loc = storeOp.getLoc();

    // Expand the affine map to arithmetic operations
    SmallVector<Value, 8> indices(storeOp.getMapOperands());
    auto maybeExpandedMap =
        expandAffineMap(rewriter, loc, storeOp.getAffineMap(), indices);
    if (!maybeExpandedMap)
      return failure();

    // Cast indices from index to i32 type
    SmallVector<Value> i32CastedIndices =
        aps::castMemoryIndicesToI32(rewriter, loc, *maybeExpandedMap);

    // Create aps.memstore with i32-typed indices
    rewriter.create<aps::MemStore>(loc, storeOp.getValue(),
                                   storeOp.getMemRef(), i32CastedIndices);

    // Erase the affine.store
    rewriter.eraseOp(storeOp);

    LLVM_DEBUG(llvm::dbgs() << "Converted affine.store to aps.memstore (indices: "
                            << maybeExpandedMap->size() << ")\n");
    return success();
  }
};

struct AffineMemToAPSPass : AffineMemToAPSBase<AffineMemToAPSPass> {
  void runOnOperation() override {
    auto op = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns.add<AffineLoadToAPSMemLoadPattern, AffineStoreToAPSMemStorePattern>(
        &getContext());
    GreedyRewriteConfig config;
    config.setStrictness(GreedyRewriteStrictness::ExistingOps);
    if (failed(applyPatternsGreedily(op, std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

} // namespace

namespace mlir {
std::unique_ptr<OperationPass<func::FuncOp>> createAffineMemToAPSPass() {
  return std::make_unique<AffineMemToAPSPass>();
}
} // namespace mlir
