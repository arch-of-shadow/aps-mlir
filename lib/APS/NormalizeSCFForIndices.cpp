#include "APS/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "normalize-scf-for-indices"

#include "APS/Passes.h"
#include "APS/PassDetail.h"

namespace {

using namespace mlir;
using namespace mlir::arith;

// Helper function to cast a value to index type
Value castToIndex(OpBuilder &builder, Location loc, Value val) {
  if (val.getType().isIndex()) {
    return val;
  }
  return builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), val);
}

// Pattern to convert scf.for loop bounds to index type
struct NormalizeSCFForIndicesPattern : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    // Check if the loop already uses index type
    if (forOp.getLowerBound().getType().isIndex() &&
        forOp.getUpperBound().getType().isIndex() &&
        forOp.getStep().getType().isIndex()) {
      return failure(); // Already using index type
    }

    Location loc = forOp.getLoc();

    // Cast loop bounds to index type
    Value lowerBound = castToIndex(rewriter, loc, forOp.getLowerBound());
    Value upperBound = castToIndex(rewriter, loc, forOp.getUpperBound());
    Value step = castToIndex(rewriter, loc, forOp.getStep());

    // Get the old induction variable
    Value oldIV = forOp.getInductionVar();
    Type oldIVType = oldIV.getType();

    // Create new scf.for with index-typed bounds
    auto newForOp = rewriter.create<scf::ForOp>(
        loc, lowerBound, upperBound, step, forOp.getInitArgs(),
        [&](OpBuilder &builder, Location loc, Value newIV, ValueRange iterArgs) {
          // If the old IV was not index type, cast the new IV back
          Value ivToUse = newIV;
          if (!oldIVType.isIndex()) {
            ivToUse = builder.create<arith::IndexCastOp>(loc, oldIVType, newIV);
          }

          IRMapping mapping;
          mapping.map(oldIV, ivToUse);
          for (auto [oldArg, newArg] : llvm::zip(forOp.getRegionIterArgs(), iterArgs)) {
            mapping.map(oldArg, newArg);
          }

          // Clone remaps operands through IRMapping, so uses of the old IV and
          // old iter args are rewritten to the new loop body values.
          for (auto &op : forOp.getBody()->getOperations()) {
            builder.clone(op, mapping);
          }
        });

    // Copy attributes
    newForOp->setAttrs(forOp->getAttrs());

    // Replace the old loop
    rewriter.replaceOp(forOp, newForOp.getResults());

    LLVM_DEBUG(llvm::dbgs() << "Converted scf.for to use index type\n");
    return success();
  }
};

struct NormalizeSCFForIndicesPass : NormalizeSCFForIndicesBase<NormalizeSCFForIndicesPass> {
  void runOnOperation() override {
    auto op = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns.add<NormalizeSCFForIndicesPattern>(&getContext());
    GreedyRewriteConfig config;
    config.setStrictness(GreedyRewriteStrictness::AnyOp);
    config.enableFolding();
    if (failed(applyPatternsGreedily(op, std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

} // namespace

namespace mlir {
  std::unique_ptr<OperationPass<func::FuncOp>> createNormalizeSCFForIndicesPass() {
    return std::make_unique<NormalizeSCFForIndicesPass>();
  }
}
