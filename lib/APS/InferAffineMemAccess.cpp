#include "APS/PassDetail.h"
#include "APS/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "infer-affine-mem-access"

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::affine;
using namespace mlir::memref;

namespace {

// Helper to check if a value is a constant integer
static std::optional<int64_t> getConstantIntValue(Value val) {
  if (auto constOp = val.getDefiningOp<arith::ConstantIntOp>())
    return constOp.value();
  if (auto constOp = val.getDefiningOp<arith::ConstantIndexOp>())
    return constOp.value();
  return std::nullopt;
}

// Helper to check if a value is an affine.for induction variable
static bool isAffineForInductionVar(Value val) {
  if (auto blockArg = llvm::dyn_cast<BlockArgument>(val)) {
    if (auto forOp = dyn_cast<AffineForOp>(blockArg.getOwner()->getParentOp()))
      return forOp.getInductionVar() == blockArg;
  }
  return false;
}

// Try to express a value as a linear combination of induction variables + constant
// Returns: {found, inductionVars, coefficients, constant}
struct MultiDimAffineInfo {
  bool found = false;
  SmallVector<Value> inductionVars;         // List of IVs [i, j, ...]
  SmallVector<int64_t> coefficients;        // Coefficients [a, b, ...]
  int64_t constant = 0;                     // Constant offset
};

// Try to express a value as a*x + b where x is an induction variable
// Returns: {found, inductionVar, a, b}
struct AffineExprInfo {
  bool found = false;
  Value inductionVar;
  int64_t multiplier = 1;  // 'a' coefficient
  int64_t offset = 0;      // 'b' constant
};

// Helper to trace back through index_cast to find if a value comes from an IV
static Value findInductionVar(Value val) {
  // First check if val itself is an IV
  if (isAffineForInductionVar(val))
    return val;

  // Trace through index_cast
  Value current = val;
  while (auto castOp = current.getDefiningOp<arith::IndexCastOp>()) {
    current = castOp.getIn();
    if (isAffineForInductionVar(current))
      return current;
  }
  return Value();
}

// Analyze multi-dimensional affine expression: a*i + b*j + ... + c
static MultiDimAffineInfo analyzeMultiDimIndex(Value index) {
  MultiDimAffineInfo info;

  // Trace through index_cast operations
  Value current = index;
  while (auto castOp = current.getDefiningOp<arith::IndexCastOp>()) {
    current = castOp.getIn();
  }

  // Map from induction variable to its coefficient
  llvm::DenseMap<Value, int64_t> ivCoefficients;
  int64_t constantTerm = 0;

  // Recursively decompose additions
  std::function<bool(Value)> decomposeAdditions = [&](Value val) -> bool {
    // Trace through index_cast
    while (auto castOp = val.getDefiningOp<arith::IndexCastOp>()) {
      val = castOp.getIn();
    }

    // Check if it's a constant
    if (auto constVal = getConstantIntValue(val)) {
      constantTerm += *constVal;
      return true;
    }

    // Check if it's an IV directly (coefficient = 1)
    if (Value iv = findInductionVar(val)) {
      ivCoefficients[iv] = ivCoefficients.lookup(iv) + 1;
      return true;
    }

    // Check for multiplication: a * IV
    if (auto mulOp = val.getDefiningOp<arith::MulIOp>()) {
      Value lhs = mulOp.getLhs();
      Value rhs = mulOp.getRhs();

      // Try lhs = constant, rhs = IV
      if (auto coeff = getConstantIntValue(lhs)) {
        if (Value iv = findInductionVar(rhs)) {
          ivCoefficients[iv] = ivCoefficients.lookup(iv) + *coeff;
          return true;
        }
      }

      // Try lhs = IV, rhs = constant
      if (auto coeff = getConstantIntValue(rhs)) {
        if (Value iv = findInductionVar(lhs)) {
          ivCoefficients[iv] = ivCoefficients.lookup(iv) + *coeff;
          return true;
        }
      }
    }

    // Check for addition: decompose both sides
    if (auto addOp = val.getDefiningOp<arith::AddIOp>()) {
      return decomposeAdditions(addOp.getLhs()) &&
             decomposeAdditions(addOp.getRhs());
    }

    return false;
  };

  // Try to decompose the expression
  if (!decomposeAdditions(current) || ivCoefficients.empty()) {
    info.found = false;
    return info;
  }

  // Build the result
  info.found = true;
  info.constant = constantTerm;
  for (auto &pair : ivCoefficients) {
    info.inductionVars.push_back(pair.first);
    info.coefficients.push_back(pair.second);
  }

  return info;
}

// Trace through index_cast, muli, addi to find affine pattern
static AffineExprInfo analyzeIndex(Value index) {
  AffineExprInfo info;

  // Trace through index_cast operations to find the computation
  Value current = index;
  while (auto castOp = current.getDefiningOp<arith::IndexCastOp>()) {
    current = castOp.getIn();
  }

  // Check if it's directly an induction variable: x
  if (Value iv = findInductionVar(current)) {
    info.found = true;
    info.inductionVar = iv;
    info.multiplier = 1;
    info.offset = 0;
    return info;
  }

  // Check for pattern: x + b
  if (auto addOp = current.getDefiningOp<arith::AddIOp>()) {
    Value lhs = addOp.getLhs();
    Value rhs = addOp.getRhs();

    // Try lhs = x, rhs = b
    if (Value iv = findInductionVar(lhs)) {
      if (auto offset = getConstantIntValue(rhs)) {
        info.found = true;
        info.inductionVar = iv;
        info.multiplier = 1;
        info.offset = *offset;
        return info;
      }
    }

    // Try lhs = b, rhs = x
    if (Value iv = findInductionVar(rhs)) {
      if (auto offset = getConstantIntValue(lhs)) {
        info.found = true;
        info.inductionVar = iv;
        info.multiplier = 1;
        info.offset = *offset;
        return info;
      }
    }

    // Try lhs = a*x, rhs = b
    if (auto mulOp = lhs.getDefiningOp<arith::MulIOp>()) {
      auto subInfo = analyzeIndex(lhs);
      if (subInfo.found && subInfo.offset == 0) {
        if (auto offset = getConstantIntValue(rhs)) {
          info.found = true;
          info.inductionVar = subInfo.inductionVar;
          info.multiplier = subInfo.multiplier;
          info.offset = *offset;
          return info;
        }
      }
    }

    // Try lhs = b, rhs = a*x
    if (auto mulOp = rhs.getDefiningOp<arith::MulIOp>()) {
      auto subInfo = analyzeIndex(rhs);
      if (subInfo.found && subInfo.offset == 0) {
        if (auto offset = getConstantIntValue(lhs)) {
          info.found = true;
          info.inductionVar = subInfo.inductionVar;
          info.multiplier = subInfo.multiplier;
          info.offset = *offset;
          return info;
        }
      }
    }
  }

  // Check for pattern: a * x
  if (auto mulOp = current.getDefiningOp<arith::MulIOp>()) {
    Value lhs = mulOp.getLhs();
    Value rhs = mulOp.getRhs();

    // Try lhs = x, rhs = a
    if (Value iv = findInductionVar(lhs)) {
      if (auto multiplier = getConstantIntValue(rhs)) {
        info.found = true;
        info.inductionVar = iv;
        info.multiplier = *multiplier;
        info.offset = 0;
        return info;
      }
    }

    // Try lhs = a, rhs = x
    if (Value iv = findInductionVar(rhs)) {
      if (auto multiplier = getConstantIntValue(lhs)) {
        info.found = true;
        info.inductionVar = iv;
        info.multiplier = *multiplier;
        info.offset = 0;
        return info;
      }
    }
  }

  return info;
}

// Pattern to convert memref.load with affine index to affine.load
struct InferAffineLoadPattern : public OpRewritePattern<memref::LoadOp> {
  using OpRewritePattern<memref::LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::LoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    // Only handle single-dimensional loads for now
    if (loadOp.getIndices().size() != 1)
      return failure();

    Value index = loadOp.getIndices()[0];

    // Try multi-dimensional analysis first
    MultiDimAffineInfo multiInfo = analyzeMultiDimIndex(index);
    if (multiInfo.found) {
      LLVM_DEBUG(llvm::dbgs() << "Found multi-dim affine pattern with "
                              << multiInfo.inductionVars.size() << " IVs for: "
                              << loadOp << "\n");

      // Build affine map: (d0, d1, ...) -> (c0*d0 + c1*d1 + ... + constant)
      SmallVector<AffineExpr> dimExprs;
      for (unsigned i = 0; i < multiInfo.inductionVars.size(); ++i) {
        dimExprs.push_back(rewriter.getAffineDimExpr(i));
      }

      AffineExpr affineExpr = rewriter.getAffineConstantExpr(multiInfo.constant);
      for (unsigned i = 0; i < multiInfo.inductionVars.size(); ++i) {
        affineExpr = affineExpr + dimExprs[i] * multiInfo.coefficients[i];
      }

      AffineMap map = AffineMap::get(multiInfo.inductionVars.size(), 0, affineExpr);

      // Create affine.load with all induction variables
      auto affineLoad = rewriter.create<AffineLoadOp>(
          loadOp.getLoc(), loadOp.getMemRef(), map, multiInfo.inductionVars);

      rewriter.replaceOp(loadOp, affineLoad.getResult());

      LLVM_DEBUG(llvm::dbgs() << "Replaced with affine.load using map: " << map << "\n");
      return success();
    }

    // Fallback to single-dimensional analysis
    AffineExprInfo info = analyzeIndex(index);
    if (!info.found) {
      LLVM_DEBUG(llvm::dbgs() << "Could not infer affine pattern for: " << loadOp << "\n");
      return failure();
    }

    LLVM_DEBUG(llvm::dbgs() << "Found affine pattern: " << info.multiplier
                            << "*x + " << info.offset << " for: " << loadOp << "\n");

    // Build affine map: (d0) -> (a * d0 + b)
    AffineExpr dimExpr = rewriter.getAffineDimExpr(0);
    AffineExpr affineExpr = dimExpr * info.multiplier + info.offset;
    AffineMap map = AffineMap::get(1, 0, affineExpr);

    // Create affine.load with the induction variable
    auto affineLoad = rewriter.create<AffineLoadOp>(
        loadOp.getLoc(), loadOp.getMemRef(), map, ValueRange{info.inductionVar});

    rewriter.replaceOp(loadOp, affineLoad.getResult());

    LLVM_DEBUG(llvm::dbgs() << "Replaced with affine.load using map: " << map << "\n");
    return success();
  }
};

// Pattern to convert memref.store with affine index to affine.store
struct InferAffineStorePattern : public OpRewritePattern<memref::StoreOp> {
  using OpRewritePattern<memref::StoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::StoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    // Only handle single-dimensional stores for now
    if (storeOp.getIndices().size() != 1)
      return failure();

    Value index = storeOp.getIndices()[0];

    // Try multi-dimensional analysis first
    MultiDimAffineInfo multiInfo = analyzeMultiDimIndex(index);
    if (multiInfo.found) {
      LLVM_DEBUG(llvm::dbgs() << "Found multi-dim affine pattern with "
                              << multiInfo.inductionVars.size() << " IVs for: "
                              << storeOp << "\n");

      // Build affine map: (d0, d1, ...) -> (c0*d0 + c1*d1 + ... + constant)
      SmallVector<AffineExpr> dimExprs;
      for (unsigned i = 0; i < multiInfo.inductionVars.size(); ++i) {
        dimExprs.push_back(rewriter.getAffineDimExpr(i));
      }

      AffineExpr affineExpr = rewriter.getAffineConstantExpr(multiInfo.constant);
      for (unsigned i = 0; i < multiInfo.inductionVars.size(); ++i) {
        affineExpr = affineExpr + dimExprs[i] * multiInfo.coefficients[i];
      }

      AffineMap map = AffineMap::get(multiInfo.inductionVars.size(), 0, affineExpr);

      // Create affine.store with all induction variables
      rewriter.create<AffineStoreOp>(
          storeOp.getLoc(), storeOp.getValue(), storeOp.getMemRef(),
          map, multiInfo.inductionVars);

      rewriter.eraseOp(storeOp);

      LLVM_DEBUG(llvm::dbgs() << "Replaced with affine.store using map: " << map << "\n");
      return success();
    }

    // Fallback to single-dimensional analysis
    AffineExprInfo info = analyzeIndex(index);
    if (!info.found)
      return failure();

    LLVM_DEBUG(llvm::dbgs() << "Found affine pattern: " << info.multiplier
                            << "*x + " << info.offset << " for: " << storeOp << "\n");

    // Build affine map: (d0) -> (a * d0 + b)
    AffineExpr dimExpr = rewriter.getAffineDimExpr(0);
    AffineExpr affineExpr = dimExpr * info.multiplier + info.offset;
    AffineMap map = AffineMap::get(1, 0, affineExpr);

    // Create affine.store with the induction variable
    rewriter.create<AffineStoreOp>(
        storeOp.getLoc(), storeOp.getValue(), storeOp.getMemRef(),
        map, ValueRange{info.inductionVar});

    rewriter.eraseOp(storeOp);

    LLVM_DEBUG(llvm::dbgs() << "Replaced with affine.store using map: " << map << "\n");
    return success();
  }
};

struct InferAffineMemAccessPass
    : public InferAffineMemAccessBase<InferAffineMemAccessPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<InferAffineLoadPattern, InferAffineStorePattern>(&getContext());

    GreedyRewriteConfig config;
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

} // namespace

namespace mlir {
std::unique_ptr<Pass> createInferAffineMemAccessPass() {
  return std::make_unique<InferAffineMemAccessPass>();
}
} // namespace mlir
