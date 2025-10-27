#include "APS/Passes.h"
#include "APS/PassDetail.h"
#include "APS/APSOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "affine-mem-to-aps-mem"

namespace {

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::affine;

// Helper function to cast indices from index type to i32 type
SmallVector<Value> castIndicesToI32(OpBuilder &builder, Location loc,
                                     ValueRange indices) {
  SmallVector<Value> i32CastedIndices;
  auto i32Type = builder.getI32Type();

  for (Value idx : indices) {
    // Affine operations always use index type, so we cast from index to i32
    if (idx.getType().isIndex()) {
      auto casted = builder.create<IndexCastOp>(loc, i32Type, idx);
      i32CastedIndices.push_back(casted);
    } else {
      // Should not happen for affine operations, but handle it anyway
      i32CastedIndices.push_back(idx);
    }
  }
  return i32CastedIndices;
}

// Pattern to convert affine.load to aps.memload
struct AffineLoadToAPSMemLoadPattern : public OpRewritePattern<AffineLoadOp> {
  using OpRewritePattern<AffineLoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineLoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    Location loc = loadOp.getLoc();

    // Get the indices - need to handle both constant and dynamic indices
    SmallVector<Value> indices;

    // Check if we have map operands (dynamic indices)
    if (!loadOp.getMapOperands().empty()) {
      // Dynamic indices from map operands
      indices.append(loadOp.getMapOperands().begin(), loadOp.getMapOperands().end());
    } else {
      // Constant indices - need to extract them from the affine map
      AffineMap map = loadOp.getAffineMap();
      for (unsigned i = 0; i < map.getNumResults(); ++i) {
        AffineExpr expr = map.getResult(i);
        if (auto constExpr = dyn_cast<AffineConstantExpr>(expr)) {
          // This is a constant index - create a constant value
          int64_t constValue = constExpr.getValue();
          auto constOp = rewriter.create<arith::ConstantOp>(loc,
              rewriter.getI32IntegerAttr(constValue));
          indices.push_back(constOp);
        } else {
          // This shouldn't happen if we have no map operands, but handle it
          return failure();
        }
      }
    }

    // Cast indices from index to i32 type
    SmallVector<Value> i32CastedIndices =
        castIndicesToI32(rewriter, loc, indices);

    // Get the result type from the original load
    Type resultType = loadOp.getResult().getType();

    // Create aps.memload with i32-typed indices
    auto apsLoadOp = rewriter.create<aps::MemLoad>(
        loc, resultType, loadOp.getMemRef(), i32CastedIndices);

    // Replace the affine.load
    rewriter.replaceOp(loadOp, apsLoadOp.getResult());

    LLVM_DEBUG(llvm::dbgs() << "Converted affine.load to aps.memload (indices: "
                            << indices.size() << ")\n");
    return success();
  }
};

// Pattern to convert affine.store to aps.memstore
struct AffineStoreToAPSMemStorePattern : public OpRewritePattern<AffineStoreOp> {
  using OpRewritePattern<AffineStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineStoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    Location loc = storeOp.getLoc();

    // Get the indices - need to handle both constant and dynamic indices
    SmallVector<Value> indices;

    // Check if we have map operands (dynamic indices)
    if (!storeOp.getMapOperands().empty()) {
      // Dynamic indices from map operands
      indices.append(storeOp.getMapOperands().begin(), storeOp.getMapOperands().end());
    } else {
      // Constant indices - need to extract them from the affine map
      AffineMap map = storeOp.getAffineMap();
      for (unsigned i = 0; i < map.getNumResults(); ++i) {
        AffineExpr expr = map.getResult(i);
        if (auto constExpr = dyn_cast<AffineConstantExpr>(expr)) {
          // This is a constant index - create a constant value
          int64_t constValue = constExpr.getValue();
          auto constOp = rewriter.create<arith::ConstantOp>(loc,
              rewriter.getI32IntegerAttr(constValue));
          indices.push_back(constOp);
        } else {
          // This shouldn't happen if we have no map operands, but handle it
          return failure();
        }
      }
    }

    // Cast indices from index to i32 type
    SmallVector<Value> i32CastedIndices =
        castIndicesToI32(rewriter, loc, indices);

    // Create aps.memstore with i32-typed indices
    rewriter.create<aps::MemStore>(loc, storeOp.getValue(),
                                   storeOp.getMemRef(), i32CastedIndices);

    // Erase the affine.store
    rewriter.eraseOp(storeOp);

    LLVM_DEBUG(llvm::dbgs() << "Converted affine.store to aps.memstore (indices: "
                            << indices.size() << ")\n");
    return success();
  }
};

struct AffineMemToAPSMemPass : MemRefToAPSMemBase<AffineMemToAPSMemPass> {
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
std::unique_ptr<OperationPass<func::FuncOp>> createAffineMemToAPSMemPass() {
  return std::make_unique<AffineMemToAPSMemPass>();
}
} // namespace mlir
