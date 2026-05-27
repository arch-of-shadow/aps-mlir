#include "APS/PassDetail.h"
#include "APS/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "raise-memref-to-affine"

using namespace mlir;
using namespace mlir::affine;

namespace {

static Value stripIndexCasts(Value value) {
  while (true) {
    if (auto cast = value.getDefiningOp<arith::IndexCastOp>()) {
      value = cast.getIn();
      continue;
    }
    if (auto ext = value.getDefiningOp<arith::ExtSIOp>()) {
      value = ext.getIn();
      continue;
    }
    if (auto ext = value.getDefiningOp<arith::ExtUIOp>()) {
      value = ext.getIn();
      continue;
    }
    if (auto trunc = value.getDefiningOp<arith::TruncIOp>()) {
      value = trunc.getIn();
      continue;
    }
    return value;
  }
}

static bool isAffineForInductionVar(Value value) {
  auto blockArg = llvm::dyn_cast<BlockArgument>(value);
  if (!blockArg)
    return false;

  auto forOp = dyn_cast<AffineForOp>(blockArg.getOwner()->getParentOp());
  return forOp && forOp.getInductionVar() == blockArg;
}

static Value findInductionVar(Value value) {
  value = stripIndexCasts(value);
  return isAffineForInductionVar(value) ? value : Value();
}

static std::optional<int64_t> getConstantIntValue(Value value) {
  value = stripIndexCasts(value);
  if (auto constant = value.getDefiningOp<arith::ConstantIntOp>())
    return constant.value();
  if (auto constant = value.getDefiningOp<arith::ConstantIndexOp>())
    return constant.value();
  return std::nullopt;
}

static std::optional<int64_t> tryEvaluateConstant(Value value) {
  value = stripIndexCasts(value);

  if (auto constant = getConstantIntValue(value))
    return constant;

  if (auto add = value.getDefiningOp<arith::AddIOp>()) {
    auto lhs = tryEvaluateConstant(add.getLhs());
    auto rhs = tryEvaluateConstant(add.getRhs());
    if (lhs && rhs)
      return *lhs + *rhs;
  }

  if (auto sub = value.getDefiningOp<arith::SubIOp>()) {
    auto lhs = tryEvaluateConstant(sub.getLhs());
    auto rhs = tryEvaluateConstant(sub.getRhs());
    if (lhs && rhs)
      return *lhs - *rhs;
  }

  if (auto mul = value.getDefiningOp<arith::MulIOp>()) {
    auto lhs = tryEvaluateConstant(mul.getLhs());
    auto rhs = tryEvaluateConstant(mul.getRhs());
    if (lhs && rhs)
      return *lhs * *rhs;
  }

  return std::nullopt;
}

static bool isLoopInvariant(Value value, Operation *contextOp) {
  Operation *current = contextOp->getParentOp();
  while (current) {
    if (auto forOp = dyn_cast<AffineForOp>(current)) {
      if (auto *defOp = value.getDefiningOp()) {
        if (forOp.getRegion().isAncestor(defOp->getParentRegion()))
          return false;
      }
      if (auto blockArg = llvm::dyn_cast<BlockArgument>(value)) {
        if (forOp.getRegion().isAncestor(blockArg.getOwner()->getParent()))
          return false;
      }
    }
    current = current->getParentOp();
  }
  return true;
}

struct AffineIndexExpr {
  AffineExpr expr;
  SmallVector<Value> dims;
  SmallVector<Value> symbols;
  DenseMap<Value, unsigned> dimIds;
  DenseMap<Value, unsigned> symbolIds;

  explicit AffineIndexExpr(MLIRContext *ctx)
      : expr(getAffineConstantExpr(0, ctx)) {}
};

static AffineExpr getDimExpr(Value value, AffineIndexExpr &state,
                             MLIRContext *ctx) {
  auto it = state.dimIds.find(value);
  if (it != state.dimIds.end())
    return getAffineDimExpr(it->second, ctx);

  unsigned id = state.dims.size();
  state.dimIds[value] = id;
  state.dims.push_back(value);
  return getAffineDimExpr(id, ctx);
}

static AffineExpr getSymbolExpr(Value value, AffineIndexExpr &state,
                                MLIRContext *ctx) {
  value = stripIndexCasts(value);
  auto it = state.symbolIds.find(value);
  if (it != state.symbolIds.end())
    return getAffineSymbolExpr(it->second, ctx);

  unsigned id = state.symbols.size();
  state.symbolIds[value] = id;
  state.symbols.push_back(value);
  return getAffineSymbolExpr(id, ctx);
}

static FailureOr<AffineExpr> buildAffineExpr(Value value, Operation *contextOp,
                                             AffineIndexExpr &state,
                                             MLIRContext *ctx) {
  value = stripIndexCasts(value);

  if (auto constant = getConstantIntValue(value))
    return getAffineConstantExpr(*constant, ctx);

  if (Value iv = findInductionVar(value))
    return getDimExpr(iv, state, ctx);

  if (auto add = value.getDefiningOp<arith::AddIOp>()) {
    auto lhs = buildAffineExpr(add.getLhs(), contextOp, state, ctx);
    auto rhs = buildAffineExpr(add.getRhs(), contextOp, state, ctx);
    if (succeeded(lhs) && succeeded(rhs))
      return *lhs + *rhs;
    return failure();
  }

  if (auto sub = value.getDefiningOp<arith::SubIOp>()) {
    auto lhs = buildAffineExpr(sub.getLhs(), contextOp, state, ctx);
    auto rhs = buildAffineExpr(sub.getRhs(), contextOp, state, ctx);
    if (succeeded(lhs) && succeeded(rhs))
      return *lhs - *rhs;
    return failure();
  }

  if (auto mul = value.getDefiningOp<arith::MulIOp>()) {
    if (auto lhsConstant = tryEvaluateConstant(mul.getLhs())) {
      auto rhs = buildAffineExpr(mul.getRhs(), contextOp, state, ctx);
      if (succeeded(rhs))
        return *rhs * *lhsConstant;
    }

    if (auto rhsConstant = tryEvaluateConstant(mul.getRhs())) {
      auto lhs = buildAffineExpr(mul.getLhs(), contextOp, state, ctx);
      if (succeeded(lhs))
        return *lhs * *rhsConstant;
    }

    return failure();
  }

  if (isLoopInvariant(value, contextOp))
    return getSymbolExpr(value, state, ctx);

  LLVM_DEBUG(llvm::dbgs() << "Could not infer affine index for: " << value
                          << "\n");
  return failure();
}

static FailureOr<AffineIndexExpr> analyzeIndex(Value index, Operation *op,
                                               MLIRContext *ctx) {
  AffineIndexExpr state(ctx);
  auto expr = buildAffineExpr(index, op, state, ctx);
  if (failed(expr))
    return failure();

  state.expr = *expr;
  return state;
}

static Value ensureIndexType(Value value, PatternRewriter &rewriter) {
  if (value.getType().isIndex())
    return value;

  OpBuilder::InsertionGuard guard(rewriter);
  if (Operation *defOp = value.getDefiningOp()) {
    rewriter.setInsertionPointAfter(defOp);
  } else if (auto blockArg = llvm::dyn_cast<BlockArgument>(value)) {
    rewriter.setInsertionPointToStart(blockArg.getOwner());
  }

  return rewriter.create<arith::IndexCastOp>(value.getLoc(),
                                             rewriter.getIndexType(), value);
}

static SmallVector<Value> buildMapOperands(AffineIndexExpr &index,
                                           PatternRewriter &rewriter) {
  SmallVector<Value> operands;
  operands.append(index.dims);
  for (Value symbol : index.symbols)
    operands.push_back(ensureIndexType(symbol, rewriter));
  return operands;
}

struct InferAffineLoadPattern : public OpRewritePattern<memref::LoadOp> {
  using OpRewritePattern<memref::LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::LoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    if (loadOp.getIndices().size() != 1)
      return failure();

    auto index = analyzeIndex(loadOp.getIndices()[0], loadOp,
                              rewriter.getContext());
    if (failed(index))
      return failure();

    AffineMap map =
        AffineMap::get(index->dims.size(), index->symbols.size(), index->expr);
    SmallVector<Value> operands = buildMapOperands(*index, rewriter);

    auto affineLoad = rewriter.create<AffineLoadOp>(
        loadOp.getLoc(), loadOp.getMemRef(), map, operands);
    rewriter.replaceOp(loadOp, affineLoad.getResult());

    LLVM_DEBUG(llvm::dbgs() << "Replaced with affine.load using map: " << map
                            << "\n");
    return success();
  }
};

struct InferAffineStorePattern : public OpRewritePattern<memref::StoreOp> {
  using OpRewritePattern<memref::StoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::StoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    if (storeOp.getIndices().size() != 1)
      return failure();

    auto index = analyzeIndex(storeOp.getIndices()[0], storeOp,
                              rewriter.getContext());
    if (failed(index))
      return failure();

    AffineMap map =
        AffineMap::get(index->dims.size(), index->symbols.size(), index->expr);
    SmallVector<Value> operands = buildMapOperands(*index, rewriter);

    rewriter.create<AffineStoreOp>(storeOp.getLoc(), storeOp.getValue(),
                                   storeOp.getMemRef(), map, operands);
    rewriter.eraseOp(storeOp);

    LLVM_DEBUG(llvm::dbgs() << "Replaced with affine.store using map: " << map
                            << "\n");
    return success();
  }
};

struct RaiseMemRefToAffinePass
    : public RaiseMemRefToAffineBase<RaiseMemRefToAffinePass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<InferAffineLoadPattern, InferAffineStorePattern>(&getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

namespace mlir {
std::unique_ptr<Pass> createRaiseMemRefToAffinePass() {
  return std::make_unique<RaiseMemRefToAffinePass>();
}
} // namespace mlir
