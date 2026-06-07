#include "APS/Passes.h"
#include "APS/PassDetail.h"
#include "APS/APSOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "aps-mem-to-memref"

namespace {

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::memref;

// Helper function to cast indices to index type
SmallVector<Value> castIndicesToIndex(OpBuilder &builder, Location loc,
                                       ValueRange indices) {
  SmallVector<Value> indexCastedIndices;
  for (Value idx : indices) {
    if (idx.getType().isIndex()) {
      indexCastedIndices.push_back(idx);
    } else {
      auto casted =
          builder.create<IndexCastOp>(loc, builder.getIndexType(), idx);
      indexCastedIndices.push_back(casted);
    }
  }
  return indexCastedIndices;
}

// Pattern to convert aps.read_smem to memref.load
struct APSMemLoadToMemRefLoadPattern : public OpRewritePattern<aps::ReadSmem> {
  using OpRewritePattern<aps::ReadSmem>::OpRewritePattern;

  LogicalResult matchAndRewrite(aps::ReadSmem memLoadOp,
                                PatternRewriter &rewriter) const override {
    Location loc = memLoadOp.getLoc();

    // Cast indices to index type
    SmallVector<Value> indexCastedIndices =
        castIndicesToIndex(rewriter, loc, memLoadOp.getIndices());

    // Create memref.load with index-typed indices
    auto loadOp = rewriter.create<LoadOp>(
        loc, memLoadOp.getMemref(), indexCastedIndices);

    // Replace the aps.read_smem
    rewriter.replaceOp(memLoadOp, loadOp.getResult());

    LLVM_DEBUG(llvm::dbgs() << "Converted aps.read_smem to memref.load\n");
    return success();
  }
};

// Pattern to convert aps.write_smem to memref.store
struct APSMemStoreToMemRefStorePattern : public OpRewritePattern<aps::WriteSmem> {
  using OpRewritePattern<aps::WriteSmem>::OpRewritePattern;

  LogicalResult matchAndRewrite(aps::WriteSmem memStoreOp,
                                PatternRewriter &rewriter) const override {
    Location loc = memStoreOp.getLoc();

    // Cast indices to index type
    SmallVector<Value> indexCastedIndices =
        castIndicesToIndex(rewriter, loc, memStoreOp.getIndices());

    // Create memref.store with index-typed indices
    rewriter.create<StoreOp>(loc, memStoreOp.getValue(),
                             memStoreOp.getMemref(), indexCastedIndices);

    // Erase the aps.write_smem
    rewriter.eraseOp(memStoreOp);

    LLVM_DEBUG(llvm::dbgs() << "Converted aps.write_smem to memref.store\n");
    return success();
  }
};

struct APSMemToMemRefPass : APSMemToMemRefBase<APSMemToMemRefPass> {
  void runOnOperation() override {
    auto op = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns.add<APSMemLoadToMemRefLoadPattern, APSMemStoreToMemRefStorePattern>(
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
std::unique_ptr<OperationPass<func::FuncOp>> createAPSMemToMemRefPass() {
  return std::make_unique<APSMemToMemRefPass>();
}
} // namespace mlir
