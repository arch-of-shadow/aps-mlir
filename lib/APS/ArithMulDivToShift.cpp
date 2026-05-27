//===- ArithMulDivToShift.cpp - Convert arith mul/div to shift ops --------===//
//
// This pass converts arith.muli and arith.divui/divsi operations to shift
// operations when the second operand is a constant power of 2.
//
// Transformations:
// - arith.muli %x, %pow2  => arith.shli %x, log2(%pow2)
// - arith.divui %x, %pow2 => arith.shrui %x, log2(%pow2)
// - arith.divsi %x, %pow2 => (signed shift with bias for negative values)
//
//===----------------------------------------------------------------------===//

#include "APS/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/MathExtras.h"

namespace mlir {

#define GEN_PASS_DEF_ARITHMULDIVTOSHIFT
#include "APS/Passes.h.inc"

namespace {

static int64_t getLog2IfPowerOf2(Value value) {
  if (auto constIntOp = value.getDefiningOp<arith::ConstantIntOp>()) {
    int64_t constValue = constIntOp.value();
    if (constValue > 0 && llvm::isPowerOf2_64(constValue))
      return llvm::Log2_64(constValue);
    return -1;
  }

  if (auto constIndexOp = value.getDefiningOp<arith::ConstantIndexOp>()) {
    int64_t constValue = constIndexOp.value();
    if (constValue > 0 && llvm::isPowerOf2_64(constValue))
      return llvm::Log2_64(constValue);
    return -1;
  }

  auto constOp = value.getDefiningOp<arith::ConstantOp>();
  if (!constOp)
    return -1;

  auto attr = llvm::dyn_cast<IntegerAttr>(constOp.getValue());
  if (!attr)
    return -1;

  int64_t constValue = attr.getInt();
  if (constValue <= 0 || !llvm::isPowerOf2_64(constValue))
    return -1;
  return llvm::Log2_64(constValue);
}

static Value createConstant(PatternRewriter &rewriter, Location loc, Type type,
                            int64_t value) {
  if (type.isIndex()) {
    return rewriter.create<arith::ConstantIndexOp>(loc, value);
  }
  return rewriter.create<arith::ConstantIntOp>(loc, type, value);
}

template <typename OpTy>
static FailureOr<int64_t> getPowerOfTwoRhsShift(OpTy op) {
  int64_t log2Val = getLog2IfPowerOf2(op.getRhs());
  if (log2Val < 0)
    return failure();
  return log2Val;
}

static Value createSignedDivByPowerOfTwo(PatternRewriter &rewriter,
                                         arith::DivSIOp op,
                                         int64_t log2Val) {
  Location loc = op.getLoc();
  Value lhs = op.getLhs();
  auto integerType = llvm::dyn_cast<IntegerType>(lhs.getType());
  if (!integerType)
    return {};

  Type type = lhs.getType();
  if (log2Val == 0)
    return lhs;

  Value sign = rewriter.create<arith::ShRSIOp>(
      loc, lhs, createConstant(rewriter, loc, type, integerType.getWidth() - 1));
  Value mask = createConstant(rewriter, loc, type, (int64_t{1} << log2Val) - 1);
  Value bias = rewriter.create<arith::AndIOp>(loc, sign, mask);
  Value biased = rewriter.create<arith::AddIOp>(loc, lhs, bias);
  return rewriter.create<arith::ShRSIOp>(
      loc, biased, createConstant(rewriter, loc, type, log2Val));
}

/// Pattern to convert arith.muli with power of 2 to arith.shli
///
/// Example:
///   %result = arith.muli %x, %c8 : i32  (where %c8 = 8 = 2^3)
/// becomes:
///   %c3 = arith.constant 3 : i32
///   %result = arith.shli %x, %c3 : i32
///
struct MuliToShliPattern : public OpRewritePattern<arith::MulIOp> {
  using OpRewritePattern<arith::MulIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::MulIOp op,
                                PatternRewriter &rewriter) const override {
    auto lhs = op.getLhs();
    auto rhs = op.getRhs();

    int64_t log2Val = getLog2IfPowerOf2(rhs);
    if (log2Val < 0) {
      log2Val = getLog2IfPowerOf2(lhs);
      if (log2Val < 0)
        return failure();
      std::swap(lhs, rhs);
    }

    rewriter.replaceOpWithNewOp<arith::ShLIOp>(
        op, lhs, createConstant(rewriter, op.getLoc(), op.getType(), log2Val));
    return success();
  }
};

/// Pattern to convert arith.divui with power of 2 to arith.shrui
///
/// Example:
///   %result = arith.divui %x, %c8 : i32  (where %c8 = 8 = 2^3)
/// becomes:
///   %c3 = arith.constant 3 : i32
///   %result = arith.shrui %x, %c3 : i32
///
struct DivuiToShruiPattern : public OpRewritePattern<arith::DivUIOp> {
  using OpRewritePattern<arith::DivUIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::DivUIOp op,
                                PatternRewriter &rewriter) const override {
    FailureOr<int64_t> log2Val = getPowerOfTwoRhsShift(op);
    if (failed(log2Val))
      return failure();

    rewriter.replaceOpWithNewOp<arith::ShRUIOp>(
        op, op.getLhs(),
        createConstant(rewriter, op.getLoc(), op.getType(), *log2Val));
    return success();
  }
};

/// Pattern to convert arith.divsi with power of 2 to shift plus sign bias.
///
/// Example:
///   %result = arith.divsi %x, %c8 : i32  (where %c8 = 8 = 2^3)
/// becomes:
///   %c3 = arith.constant 3 : i32
///   %sign = arith.shrsi %x, %c31 : i32
///   %bias = arith.andi %sign, %c7 : i32
///   %biased = arith.addi %x, %bias : i32
///   %result = arith.shrsi %biased, %c3 : i32
///
struct DivsiToShrsiPattern : public OpRewritePattern<arith::DivSIOp> {
  using OpRewritePattern<arith::DivSIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::DivSIOp op,
                                PatternRewriter &rewriter) const override {
    FailureOr<int64_t> log2Val = getPowerOfTwoRhsShift(op);
    if (failed(log2Val))
      return failure();

    Value result = createSignedDivByPowerOfTwo(rewriter, op, *log2Val);
    if (!result)
      return failure();

    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ArithMulDivToShiftPass
    : public impl::ArithMulDivToShiftBase<ArithMulDivToShiftPass> {
  void runOnOperation() override {
    auto *context = &getContext();
    auto op = getOperation();

    RewritePatternSet patterns(context);
    patterns.add<MuliToShliPattern, DivuiToShruiPattern, DivsiToShrsiPattern>(
        context);

    if (failed(applyPatternsGreedily(op, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> createArithMulDivToShiftPass() {
  return std::make_unique<ArithMulDivToShiftPass>();
}

} // namespace mlir
