// This file is copy from
// mlir/lib/Conversion/AffineToStandard/AffineToStandard.cpp, and modified
// AffineForLowering pattern

// #include "mlir/Conversion/AffineToStandard/AffineToStandard.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
// #include "mlir/Transforms/Passes.h"

#include "APS/APSDialect.h"
#include "TOR/PassDetail.h"
#include "TOR/Passes.h"
#include "TOR/Utils.h"
#include "circt/Dialect/Comb/CombDialect.h"

#define DEBUG_TYPE "lower-affine-for"

namespace {

using namespace mlir;
using namespace mlir::affine;
using namespace mlir::vector;

/// Given a range of values, emit the code that reduces them with "min" or "max"
/// depending on the provided comparison predicate.  The predicate defines which
/// comparison to perform, "lt" for "min", "gt" for "max" and is used for the
/// `cmpi` operation followed by the `select` operation:
///
///   %cond   = arith.cmpi "predicate" %v0, %v1
///   %result = select %cond, %v0, %v1
///
/// Multiple values are scanned in a linear sequence.  This creates a data
/// dependences that wouldn't exist in a tree reduction, but is easier to
/// recognize as a reduction by the subsequent passes.
static Value buildMinMaxReductionSeq(Location loc,
                                     arith::CmpIPredicate predicate,
                                     ValueRange values, OpBuilder &builder) {
  assert(!values.empty() && "empty min/max chain");

  auto valueIt = values.begin();
  Value value = *valueIt++;
  for (; valueIt != values.end(); ++valueIt) {
    auto cmpOp = builder.create<arith::CmpIOp>(loc, predicate, value, *valueIt);
    value = builder.create<arith::SelectOp>(loc, cmpOp.getResult(), value,
                                            *valueIt);
  }

  return value;
}

/// Emit instructions that correspond to computing the maximum value among the
/// values of a (potentially) multi-output affine map applied to `operands`.
static Value lowerAffineMapMax(OpBuilder &builder, Location loc, AffineMap map,
                               ValueRange operands) {
  if (auto values = expandAffineMap(builder, loc, map, operands))
    return buildMinMaxReductionSeq(loc, arith::CmpIPredicate::sgt, *values,
                                   builder);
  return nullptr;
}

/// Emit instructions that correspond to computing the minimum value among the
/// values of a (potentially) multi-output affine map applied to `operands`.
static Value lowerAffineMapMin(OpBuilder &builder, Location loc, AffineMap map,
                               ValueRange operands) {
  if (auto values = expandAffineMap(builder, loc, map, operands))
    return buildMinMaxReductionSeq(loc, arith::CmpIPredicate::slt, *values,
                                   builder);
  return nullptr;
}

static SmallVector<Value>
materializeAffineMapOperandsAsIndex(OpBuilder &builder, Location loc,
                                    ValueRange operands) {
  SmallVector<Value> indexOperands;
  indexOperands.reserve(operands.size());
  for (Value operand : operands) {
    if (operand.getType().isIndex()) {
      indexOperands.push_back(operand);
      continue;
    }
    indexOperands.push_back(builder.create<arith::IndexCastOp>(
        loc, builder.getIndexType(), operand));
  }
  return indexOperands;
}

/// Emit instructions that correspond to the affine map in the upper bound
/// applied to the respective operands, and compute the minimum value across
/// the results.
Value lowerAffineUpperBound(AffineForOp op, OpBuilder &builder) {
  SmallVector<Value> operands = materializeAffineMapOperandsAsIndex(
      builder, op.getLoc(), op.getUpperBoundOperands());
  return lowerAffineMapMin(builder, op.getLoc(), op.getUpperBoundMap(),
                           operands);
}

/// Emit instructions that correspond to the affine map in the lower bound
/// applied to the respective operands, and compute the maximum value across
/// the results.
Value lowerAffineLowerBound(AffineForOp op, OpBuilder &builder) {
  SmallVector<Value> operands = materializeAffineMapOperandsAsIndex(
      builder, op.getLoc(), op.getLowerBoundOperands());
  return lowerAffineMapMax(builder, op.getLoc(), op.getLowerBoundMap(),
                           operands);
}

static std::optional<int64_t> getConstantIntegerValue(Value value) {
  if (auto constant = value.getDefiningOp<arith::ConstantIndexOp>())
    return constant.value();
  if (auto constant = value.getDefiningOp<arith::ConstantIntOp>())
    return constant.value();
  if (auto constant = value.getDefiningOp<arith::ConstantOp>()) {
    if (auto attr = dyn_cast<IntegerAttr>(constant.getValue()))
      return attr.getInt();
  }
  if (auto cast = value.getDefiningOp<arith::IndexCastOp>())
    return getConstantIntegerValue(cast.getIn());
  return std::nullopt;
}

static std::optional<int64_t>
getSimpleAffineExprConstant(AffineExpr expr, ValueRange operands) {
  if (auto constant = dyn_cast<AffineConstantExpr>(expr))
    return cast<AffineConstantExpr>(expr).getValue();

  unsigned position = 0;
  if (auto dim = dyn_cast<AffineDimExpr>(expr)) {
    position = dim.getPosition();
    if (position >= operands.size())
      return std::nullopt;
    return getConstantIntegerValue(operands[position]);
  }

  if (auto symbol = dyn_cast<AffineSymbolExpr>(expr)) {
    position = symbol.getPosition();
    if (position >= operands.size())
      return std::nullopt;
    return getConstantIntegerValue(operands[position]);
  }

  return std::nullopt;
}

static std::optional<int64_t>
getSimpleBoundConstant(AffineMap map, ValueRange operands, bool isLowerBound) {
  std::optional<int64_t> bound;
  for (AffineExpr expr : map.getResults()) {
    auto value = getSimpleAffineExprConstant(expr, operands);
    if (!value)
      return std::nullopt;
    if (!bound) {
      bound = *value;
      continue;
    }
    bound = isLowerBound ? std::max(*bound, *value)
                         : std::min(*bound, *value);
  }
  return bound;
}

static std::optional<unsigned> bitsToEncodeUnsigned(int64_t value) {
  if (value < 0)
    return std::nullopt;
  uint64_t unsignedValue = static_cast<uint64_t>(value);
  unsigned bits = 1;
  while (unsignedValue >> bits)
    ++bits;
  return bits;
}

static std::optional<unsigned> bitsToEncodeNonNegativeSigned(int64_t value) {
  auto unsignedBits = bitsToEncodeUnsigned(value);
  if (!unsignedBits)
    return std::nullopt;
  if (value == 0)
    return 1;
  return *unsignedBits + 1;
}

struct InferredIVControl {
  Type type;
  bool unsignedCmp = false;
};

static InferredIVControl inferStaticControlIVType(AffineForOp op) {
  auto lower = getSimpleBoundConstant(op.getLowerBoundMap(),
                                      op.getLowerBoundOperands(),
                                      /*isLowerBound=*/true);
  auto upper = getSimpleBoundConstant(op.getUpperBoundMap(),
                                      op.getUpperBoundOperands(),
                                      /*isLowerBound=*/false);
  int64_t step = op.getStep().getSExtValue();
  if (!lower || !upper || step <= 0)
    return {};

  std::optional<unsigned> lowerBits = bitsToEncodeNonNegativeSigned(*lower);
  std::optional<unsigned> upperBits = bitsToEncodeNonNegativeSigned(*upper);
  std::optional<unsigned> stepBits = bitsToEncodeNonNegativeSigned(step);
  if (!lowerBits || !upperBits || !stepBits)
    return {};

  unsigned width = std::max({*lowerBits, *upperBits, *stepBits});
  return {IntegerType::get(op.getContext(), width),
          /*unsignedCmp=*/false};
}

static std::optional<unsigned> inferIntegerSourceWidth(Value value) {
  if (auto intType = dyn_cast<IntegerType>(value.getType()))
    return intType.getWidth();
  if (auto cast = value.getDefiningOp<arith::IndexCastOp>())
    return inferIntegerSourceWidth(cast.getIn());
  if (auto ext = value.getDefiningOp<arith::ExtUIOp>())
    return inferIntegerSourceWidth(ext.getIn());
  if (auto ext = value.getDefiningOp<arith::ExtSIOp>())
    return inferIntegerSourceWidth(ext.getIn());
  if (auto trunc = value.getDefiningOp<arith::TruncIOp>())
    return inferIntegerSourceWidth(trunc.getIn());
  return std::nullopt;
}

static std::optional<unsigned> inferIdentityMapOperandWidth(AffineMap map,
                                                           ValueRange operands) {
  if (map.getNumResults() != 1)
    return std::nullopt;
  AffineExpr expr = map.getResult(0);
  unsigned position = 0;
  if (auto dim = dyn_cast<AffineDimExpr>(expr)) {
    position = dim.getPosition();
  } else if (auto symbol = dyn_cast<AffineSymbolExpr>(expr)) {
    position = symbol.getPosition();
  } else {
    return std::nullopt;
  }
  if (position >= operands.size())
    return std::nullopt;
  return inferIntegerSourceWidth(operands[position]);
}

static InferredIVControl inferDynamicControlIVType(AffineForOp op) {
  std::optional<unsigned> lowerWidth = inferIdentityMapOperandWidth(
      op.getLowerBoundMap(), op.getLowerBoundOperands());
  std::optional<unsigned> upperWidth = inferIdentityMapOperandWidth(
      op.getUpperBoundMap(), op.getUpperBoundOperands());
  if (!lowerWidth && !upperWidth)
    return {};

  unsigned width = 1;
  if (lowerWidth)
    width = std::max(width, *lowerWidth);
  if (upperWidth)
    width = std::max(width, *upperWidth);
  if (auto stepBits = bitsToEncodeUnsigned(op.getStep().getSExtValue()))
    width = std::max(width, *stepBits);
  else
    return {};
  return {IntegerType::get(op.getContext(), width),
          /*unsignedCmp=*/true};
}

static InferredIVControl inferIVControl(AffineForOp op) {
  if (InferredIVControl control = inferStaticControlIVType(op); control.type)
    return control;
  if (InferredIVControl control = inferDynamicControlIVType(op); control.type)
    return control;
  return {IntegerType::get(op.getContext(), 32), /*unsignedCmp=*/false};
}

static Value castIndexLikeValue(OpBuilder &builder, Location loc, Value value,
                                Type targetType) {
  if (value.getType() == targetType)
    return value;
  return builder.create<arith::IndexCastOp>(loc, targetType, value);
}

static Value castIntegerValue(OpBuilder &builder, Location loc, Value value,
                              Type targetType) {
  if (value.getType() == targetType)
    return value;
  auto sourceType = dyn_cast<IntegerType>(value.getType());
  auto destType = dyn_cast<IntegerType>(targetType);
  if (!sourceType || !destType)
    return builder.create<arith::IndexCastOp>(loc, targetType, value);

  unsigned sourceWidth = sourceType.getWidth();
  unsigned destWidth = destType.getWidth();
  if (sourceWidth < destWidth)
    return builder.create<arith::ExtUIOp>(loc, targetType, value);
  return builder.create<arith::TruncIOp>(loc, targetType, value);
}

static Value castMappedValue(OpBuilder &builder, Location loc, Value value,
                             Type targetType) {
  if (value.getType() == targetType)
    return value;
  if (value.getType().isIndex() || targetType.isIndex())
    return builder.create<arith::IndexCastOp>(loc, targetType, value);
  return castIntegerValue(builder, loc, value, targetType);
}

static Type lowerIndexScalarType(OpBuilder &builder, Type type) {
  if (type.isIndex())
    return builder.getI32Type();
  return type;
}

static std::pair<Value, Value>
lookupAndCastMappedOperandsToType(OpBuilder &builder, Location loc,
                                  IRMapping &mapping, Value oldLhs,
                                  Value oldRhs, Type targetType) {
  return {castMappedValue(builder, loc, mapping.lookupOrDefault(oldLhs),
                          targetType),
          castMappedValue(builder, loc, mapping.lookupOrDefault(oldRhs),
                          targetType)};
}

static Value castToIndex(OpBuilder &builder, Location loc, Value value) {
  return castIndexLikeValue(builder, loc, value, builder.getIndexType());
}

static LogicalResult lowerAffineLoadInLoop(OpBuilder &builder,
                                           AffineLoadOp load,
                                           IRMapping &mapping) {
  SmallVector<Value> mapOperands;
  mapOperands.reserve(load.getMapOperands().size());
  for (Value operand : load.getMapOperands())
    mapOperands.push_back(
        castToIndex(builder, load.getLoc(), mapping.lookupOrDefault(operand)));

  auto maybeIndices =
      expandAffineMap(builder, load.getLoc(), load.getAffineMap(), mapOperands);
  if (!maybeIndices)
    return failure();

  SmallVector<Value> indices;
  indices.reserve(maybeIndices->size());
  for (Value index : *maybeIndices)
    indices.push_back(index);

  auto newLoad = builder.create<memref::LoadOp>(
      load.getLoc(), mapping.lookupOrDefault(load.getMemRef()), indices);
  mapping.map(load.getResult(), newLoad.getResult());
  return success();
}

static LogicalResult lowerAffineStoreInLoop(OpBuilder &builder,
                                            AffineStoreOp store,
                                            IRMapping &mapping) {
  SmallVector<Value> mapOperands;
  mapOperands.reserve(store.getMapOperands().size());
  for (Value operand : store.getMapOperands())
    mapOperands.push_back(
        castToIndex(builder, store.getLoc(), mapping.lookupOrDefault(operand)));

  auto maybeIndices = expandAffineMap(builder, store.getLoc(),
                                      store.getAffineMap(), mapOperands);
  if (!maybeIndices)
    return failure();

  SmallVector<Value> indices;
  indices.reserve(maybeIndices->size());
  for (Value index : *maybeIndices)
    indices.push_back(index);

  builder.create<memref::StoreOp>(
      store.getLoc(), mapping.lookupOrDefault(store.getValueToStore()),
      mapping.lookupOrDefault(store.getMemRef()), indices);
  return success();
}

class AffineMinLowering : public OpRewritePattern<AffineMinOp> {
public:
  using OpRewritePattern<AffineMinOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineMinOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value> operands = materializeAffineMapOperandsAsIndex(
        rewriter, op.getLoc(), op.getOperands());
    Value reduced =
        lowerAffineMapMin(rewriter, op.getLoc(), op.getMap(), operands);
    if (!reduced)
      return failure();

    rewriter.replaceOp(op, reduced);
    return success();
  }
};

class AffineMaxLowering : public OpRewritePattern<AffineMaxOp> {
public:
  using OpRewritePattern<AffineMaxOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineMaxOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value> operands = materializeAffineMapOperandsAsIndex(
        rewriter, op.getLoc(), op.getOperands());
    Value reduced =
        lowerAffineMapMax(rewriter, op.getLoc(), op.getMap(), operands);
    if (!reduced)
      return failure();

    rewriter.replaceOp(op, reduced);
    return success();
  }
};

/// Affine yields ops are removed.
class AffineYieldOpLowering : public OpRewritePattern<AffineYieldOp> {
public:
  using OpRewritePattern<AffineYieldOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineYieldOp op,
                                PatternRewriter &rewriter) const override {
    if (isa<AffineForOp>(op->getParentOp()))
      return failure();
    if (isa<scf::ParallelOp>(op->getParentOp())) {
      // scf.parallel does not yield any values via its terminator scf.yield but
      // models reductions differently using additional ops in its region.
      rewriter.replaceOpWithNewOp<scf::YieldOp>(op);
      return success();
    }
    rewriter.replaceOpWithNewOp<scf::YieldOp>(op, op.getOperands());
    return success();
  }
};

class AffineForLowering : public OpRewritePattern<AffineForOp> {
public:
  using OpRewritePattern<AffineForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineForOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    InferredIVControl ivControl = inferIVControl(op);
    Type ivType = ivControl.type;

    Value lowerBound = lowerAffineLowerBound(op, rewriter);
    Value upperBound = lowerAffineUpperBound(op, rewriter);
    lowerBound = castIndexLikeValue(rewriter, loc, lowerBound, ivType);
    upperBound = castIndexLikeValue(rewriter, loc, upperBound, ivType);

    Value step;
    if (ivType.isIndex())
      step = rewriter.create<arith::ConstantIndexOp>(
          loc, op.getStep().getSExtValue());
    else
      step = rewriter.create<arith::ConstantIntOp>(
          loc, cast<IntegerType>(ivType), op.getStep().getSExtValue());

    bool bodyLoweringFailed = false;
    auto scfForOp = rewriter.create<scf::ForOp>(
        loc, lowerBound, upperBound, step, op.getInits(),
        [&](OpBuilder &builder, Location bodyLoc, Value newIV,
            ValueRange iterArgs) {
          IRMapping mapping;
          mapping.map(op.getInductionVar(), newIV);
          for (auto [oldArg, newArg] :
               llvm::zip(op.getRegionIterArgs(), iterArgs))
            mapping.map(oldArg, newArg);

          for (Operation &bodyOp : op.getBody()->without_terminator()) {
            if (auto castOp = dyn_cast<arith::IndexCastOp>(bodyOp)) {
              Value input = mapping.lookupOrDefault(castOp.getIn());
              Type targetType =
                  lowerIndexScalarType(builder, castOp.getResult().getType());
              Value replacement =
                  castMappedValue(builder, castOp.getLoc(), input, targetType);
              mapping.map(castOp.getResult(), replacement);
              continue;
            }
            if (auto addOp = dyn_cast<arith::AddIOp>(bodyOp)) {
              Type resultType =
                  lowerIndexScalarType(builder, addOp.getResult().getType());
              auto [lhs, rhs] = lookupAndCastMappedOperandsToType(
                  builder, addOp.getLoc(), mapping, addOp.getLhs(),
                  addOp.getRhs(), resultType);
              auto replacement =
                  builder.create<arith::AddIOp>(addOp.getLoc(), lhs, rhs);
              replacement->setAttrs(addOp->getAttrs());
              mapping.map(addOp.getResult(), replacement.getResult());
              continue;
            }
            if (auto subOp = dyn_cast<arith::SubIOp>(bodyOp)) {
              Type resultType =
                  lowerIndexScalarType(builder, subOp.getResult().getType());
              auto [lhs, rhs] = lookupAndCastMappedOperandsToType(
                  builder, subOp.getLoc(), mapping, subOp.getLhs(),
                  subOp.getRhs(), resultType);
              auto replacement =
                  builder.create<arith::SubIOp>(subOp.getLoc(), lhs, rhs);
              replacement->setAttrs(subOp->getAttrs());
              mapping.map(subOp.getResult(), replacement.getResult());
              continue;
            }
            if (auto mulOp = dyn_cast<arith::MulIOp>(bodyOp)) {
              Type resultType =
                  lowerIndexScalarType(builder, mulOp.getResult().getType());
              auto [lhs, rhs] = lookupAndCastMappedOperandsToType(
                  builder, mulOp.getLoc(), mapping, mulOp.getLhs(),
                  mulOp.getRhs(), resultType);
              auto replacement =
                  builder.create<arith::MulIOp>(mulOp.getLoc(), lhs, rhs);
              replacement->setAttrs(mulOp->getAttrs());
              mapping.map(mulOp.getResult(), replacement.getResult());
              continue;
            }
            if (auto shlOp = dyn_cast<arith::ShLIOp>(bodyOp)) {
              Type resultType =
                  lowerIndexScalarType(builder, shlOp.getResult().getType());
              auto [lhs, rhs] = lookupAndCastMappedOperandsToType(
                  builder, shlOp.getLoc(), mapping, shlOp.getLhs(),
                  shlOp.getRhs(), resultType);
              auto replacement =
                  builder.create<arith::ShLIOp>(shlOp.getLoc(), lhs, rhs);
              replacement->setAttrs(shlOp->getAttrs());
              mapping.map(shlOp.getResult(), replacement.getResult());
              continue;
            }
            if (auto shrOp = dyn_cast<arith::ShRUIOp>(bodyOp)) {
              Type resultType =
                  lowerIndexScalarType(builder, shrOp.getResult().getType());
              auto [lhs, rhs] = lookupAndCastMappedOperandsToType(
                  builder, shrOp.getLoc(), mapping, shrOp.getLhs(),
                  shrOp.getRhs(), resultType);
              auto replacement =
                  builder.create<arith::ShRUIOp>(shrOp.getLoc(), lhs, rhs);
              replacement->setAttrs(shrOp->getAttrs());
              mapping.map(shrOp.getResult(), replacement.getResult());
              continue;
            }
            if (auto shrOp = dyn_cast<arith::ShRSIOp>(bodyOp)) {
              Type resultType =
                  lowerIndexScalarType(builder, shrOp.getResult().getType());
              auto [lhs, rhs] = lookupAndCastMappedOperandsToType(
                  builder, shrOp.getLoc(), mapping, shrOp.getLhs(),
                  shrOp.getRhs(), resultType);
              auto replacement =
                  builder.create<arith::ShRSIOp>(shrOp.getLoc(), lhs, rhs);
              replacement->setAttrs(shrOp->getAttrs());
              mapping.map(shrOp.getResult(), replacement.getResult());
              continue;
            }
            if (auto cmpOp = dyn_cast<arith::CmpIOp>(bodyOp)) {
              Type operandType =
                  lowerIndexScalarType(builder, cmpOp.getLhs().getType());
              auto [lhs, rhs] = lookupAndCastMappedOperandsToType(
                  builder, cmpOp.getLoc(), mapping, cmpOp.getLhs(),
                  cmpOp.getRhs(), operandType);
              auto replacement = builder.create<arith::CmpIOp>(
                  cmpOp.getLoc(), cmpOp.getPredicate(), lhs, rhs);
              mapping.map(cmpOp.getResult(), replacement.getResult());
              continue;
            }
            if (auto selectOp = dyn_cast<arith::SelectOp>(bodyOp)) {
              Value condition =
                  mapping.lookupOrDefault(selectOp.getCondition());
              Type resultType = selectOp.getResult().getType();
              Value trueValue = castMappedValue(
                  builder, selectOp.getLoc(),
                  mapping.lookupOrDefault(selectOp.getTrueValue()), resultType);
              Value falseValue = castMappedValue(
                  builder, selectOp.getLoc(),
                  mapping.lookupOrDefault(selectOp.getFalseValue()),
                  resultType);
              auto replacement = builder.create<arith::SelectOp>(
                  selectOp.getLoc(), condition, trueValue, falseValue);
              mapping.map(selectOp.getResult(), replacement.getResult());
              continue;
            }
            if (auto load = dyn_cast<AffineLoadOp>(bodyOp)) {
              if (failed(lowerAffineLoadInLoop(builder, load, mapping))) {
                bodyLoweringFailed = true;
                return;
              }
              continue;
            }
            if (auto store = dyn_cast<AffineStoreOp>(bodyOp)) {
              if (failed(lowerAffineStoreInLoop(builder, store, mapping))) {
                bodyLoweringFailed = true;
                return;
              }
              continue;
            }
            builder.clone(bodyOp, mapping);
          }

          auto affineYield = cast<AffineYieldOp>(op.getBody()->getTerminator());
          SmallVector<Value> yielded;
          yielded.reserve(affineYield.getNumOperands());
          for (Value operand : affineYield.getOperands())
            yielded.push_back(mapping.lookupOrDefault(operand));
          builder.create<scf::YieldOp>(bodyLoc, yielded);
        },
        ivControl.unsignedCmp);
    if (bodyLoweringFailed) {
      rewriter.eraseOp(scfForOp);
      return failure();
    }
    addHlsAttrWithNewOp(scfForOp, op);
    // need add pipeline=1 to TOR
    addHlsPipelineAttrWithNewOp(scfForOp, op);
    rewriter.replaceOp(op, scfForOp.getResults());
    return success();
  }
};

/// Convert an `affine.parallel` (loop nest) operation into a `scf.parallel`
/// operation.
class AffineParallelLowering : public OpRewritePattern<AffineParallelOp> {
public:
  using OpRewritePattern<AffineParallelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineParallelOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    SmallVector<Value, 8> steps;
    SmallVector<Value, 8> upperBoundTuple;
    SmallVector<Value, 8> lowerBoundTuple;
    SmallVector<Value, 8> identityVals;
    // Emit IR computing the lower and upper bound by expanding the map
    // expression.
    lowerBoundTuple.reserve(op.getNumDims());
    upperBoundTuple.reserve(op.getNumDims());
    for (unsigned i = 0, e = op.getNumDims(); i < e; ++i) {
      Value lower = lowerAffineMapMax(rewriter, loc, op.getLowerBoundMap(i),
                                      op.getLowerBoundsOperands());
      if (!lower)
        return rewriter.notifyMatchFailure(op, "couldn't convert lower bounds");
      lowerBoundTuple.push_back(lower);

      Value upper = lowerAffineMapMin(rewriter, loc, op.getUpperBoundMap(i),
                                      op.getUpperBoundsOperands());
      if (!upper)
        return rewriter.notifyMatchFailure(op, "couldn't convert upper bounds");
      upperBoundTuple.push_back(upper);
    }
    steps.reserve(op.getSteps().size());
    for (int64_t step : op.getSteps())
      steps.push_back(rewriter.create<arith::ConstantIndexOp>(loc, step));

    // Get the terminator op.
    Operation *affineParOpTerminator = op.getBody()->getTerminator();
    scf::ParallelOp parOp;
    if (op.getResults().empty()) {
      // Case with no reduction operations/return values.
      parOp = rewriter.create<scf::ParallelOp>(loc, lowerBoundTuple,
                                               upperBoundTuple, steps,
                                               /*bodyBuilderFn=*/nullptr);
      rewriter.eraseBlock(parOp.getBody());
      rewriter.inlineRegionBefore(op.getRegion(), parOp.getRegion(),
                                  parOp.getRegion().end());
      rewriter.replaceOp(op, parOp.getResults());
      return success();
    }
    // Case with affine.parallel with reduction operations/return values.
    // scf.parallel handles the reduction operation differently unlike
    // affine.parallel.
    ArrayRef<Attribute> reductions = op.getReductions().getValue();
    for (auto pair : llvm::zip(reductions, op.getResultTypes())) {
      // For each of the reduction operations get the identity values for
      // initialization of the result values.
      Attribute reduction = std::get<0>(pair);
      Type resultType = std::get<1>(pair);
      std::optional<arith::AtomicRMWKind> reductionOp =
          arith::symbolizeAtomicRMWKind(
              static_cast<uint64_t>(cast<IntegerAttr>(reduction).getInt()));
      assert(reductionOp && "Reduction operation cannot be of None Type");
      arith::AtomicRMWKind reductionOpValue = *reductionOp;
      identityVals.push_back(
          arith::getIdentityValue(reductionOpValue, resultType, rewriter, loc));
    }
    parOp = rewriter.create<scf::ParallelOp>(
        loc, lowerBoundTuple, upperBoundTuple, steps, identityVals,
        /*bodyBuilderFn=*/nullptr);

    //  Copy the body of the affine.parallel op.
    rewriter.eraseBlock(parOp.getBody());
    rewriter.inlineRegionBefore(op.getRegion(), parOp.getRegion(),
                                parOp.getRegion().end());
    assert(reductions.size() == affineParOpTerminator->getNumOperands() &&
           "Unequal number of reductions and operands.");
    for (unsigned i = 0, end = reductions.size(); i < end; i++) {
      // For each of the reduction operations get the respective mlir::Value.
      std::optional<arith::AtomicRMWKind> reductionOp =
          arith::symbolizeAtomicRMWKind(
              cast<IntegerAttr>(reductions[i]).getInt());
      assert(reductionOp && "Reduction Operation cannot be of None Type");
      arith::AtomicRMWKind reductionOpValue = *reductionOp;
      rewriter.setInsertionPoint(&parOp.getBody()->back());
      auto reduceOp = rewriter.create<scf::ReduceOp>(
          loc, affineParOpTerminator->getOperand(i));
      rewriter.setInsertionPointToEnd(
          &reduceOp.getReductions().front().front());
      Value reductionResult = arith::getReductionOp(
          reductionOpValue, rewriter, loc,
          reduceOp.getReductions().front().getArgument(0),
          reduceOp.getReductions().front().getArgument(1));
      rewriter.create<scf::ReduceReturnOp>(loc, reductionResult);
    }
    rewriter.replaceOp(op, parOp.getResults());
    return success();
  }
};

class AffineIfLowering : public OpRewritePattern<AffineIfOp> {
public:
  using OpRewritePattern<AffineIfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineIfOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    // Now we just have to handle the condition logic.
    auto integerSet = op.getIntegerSet();
    Value zeroConstant = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    SmallVector<Value> operands =
        materializeAffineMapOperandsAsIndex(rewriter, loc, op.getOperands());
    auto operandsRef = llvm::ArrayRef(operands);

    // Calculate cond as a conjunction without short-circuiting.
    Value cond = nullptr;
    for (unsigned i = 0, e = integerSet.getNumConstraints(); i < e; ++i) {
      AffineExpr constraintExpr = integerSet.getConstraint(i);
      bool isEquality = integerSet.isEq(i);

      // Build and apply an affine expression
      auto numDims = integerSet.getNumDims();
      Value affResult = expandAffineExpr(rewriter, loc, constraintExpr,
                                         operandsRef.take_front(numDims),
                                         operandsRef.drop_front(numDims));
      if (!affResult)
        return failure();
      auto pred =
          isEquality ? arith::CmpIPredicate::eq : arith::CmpIPredicate::sge;
      Value cmpVal =
          rewriter.create<arith::CmpIOp>(loc, pred, affResult, zeroConstant);
      cond = cond
                 ? rewriter.create<arith::AndIOp>(loc, cond, cmpVal).getResult()
                 : cmpVal;
    }
    cond = cond ? cond
                : rewriter.create<arith::ConstantIntOp>(loc, /*value=*/1,
                                                        /*width=*/1);

    bool hasElseRegion = !op.getElseRegion().empty();
    auto ifOp = rewriter.create<scf::IfOp>(loc, op.getResultTypes(), cond,
                                           hasElseRegion);
    rewriter.inlineRegionBefore(op.getThenRegion(),
                                &ifOp.getThenRegion().back());
    rewriter.eraseBlock(&ifOp.getThenRegion().back());
    if (hasElseRegion) {
      rewriter.inlineRegionBefore(op.getElseRegion(),
                                  &ifOp.getElseRegion().back());
      rewriter.eraseBlock(&ifOp.getElseRegion().back());
    }

    // Replace the Affine IfOp finally.
    rewriter.replaceOp(op, ifOp.getResults());
    return success();
  }
};

/// Convert an "affine.apply" operation into a sequence of arithmetic
/// operations using the StandardOps dialect.
class AffineApplyLowering : public OpRewritePattern<AffineApplyOp> {
public:
  using OpRewritePattern<AffineApplyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineApplyOp op,
                                PatternRewriter &rewriter) const override {
    auto maybeExpandedMap =
        expandAffineMap(rewriter, op.getLoc(), op.getAffineMap(),
                        materializeAffineMapOperandsAsIndex(
                            rewriter, op.getLoc(), op.getOperands()));
    if (!maybeExpandedMap)
      return failure();
    rewriter.replaceOp(op, *maybeExpandedMap);
    return success();
  }
};

/// Apply the affine map from an 'affine.load' operation to its operands, and
/// feed the results to a newly created 'memref.load' operation (which replaces
/// the original 'affine.load').
class AffineLoadLowering : public OpRewritePattern<AffineLoadOp> {
public:
  using OpRewritePattern<AffineLoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineLoadOp op,
                                PatternRewriter &rewriter) const override {
    // Expand affine map from 'affineLoadOp'.
    SmallVector<Value> indices = materializeAffineMapOperandsAsIndex(
        rewriter, op.getLoc(), op.getMapOperands());
    auto resultOperands =
        expandAffineMap(rewriter, op.getLoc(), op.getAffineMap(), indices);
    if (!resultOperands)
      return failure();

    // Build vector.load memref[expandedMap.results].
    rewriter.replaceOpWithNewOp<memref::LoadOp>(op, op.getMemRef(),
                                                *resultOperands);
    return success();
  }
};

/// Apply the affine map from an 'affine.prefetch' operation to its operands,
/// and feed the results to a newly created 'memref.prefetch' operation (which
/// replaces the original 'affine.prefetch').
class AffinePrefetchLowering : public OpRewritePattern<AffinePrefetchOp> {
public:
  using OpRewritePattern<AffinePrefetchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffinePrefetchOp op,
                                PatternRewriter &rewriter) const override {
    // Expand affine map from 'affinePrefetchOp'.
    SmallVector<Value, 8> indices(op.getMapOperands());
    auto resultOperands =
        expandAffineMap(rewriter, op.getLoc(), op.getAffineMap(), indices);
    if (!resultOperands)
      return failure();

    // Build memref.prefetch memref[expandedMap.results].
    rewriter.replaceOpWithNewOp<memref::PrefetchOp>(
        op, op.getMemref(), *resultOperands, op.getIsWrite(),
        op.getLocalityHint(), op.getIsDataCache());
    return success();
  }
};

/// Apply the affine map from an 'affine.store' operation to its operands, and
/// feed the results to a newly created 'memref.store' operation (which replaces
/// the original 'affine.store').
class AffineStoreLowering : public OpRewritePattern<AffineStoreOp> {
public:
  using OpRewritePattern<AffineStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineStoreOp op,
                                PatternRewriter &rewriter) const override {
    // Expand affine map from 'affineStoreOp'.
    SmallVector<Value> indices = materializeAffineMapOperandsAsIndex(
        rewriter, op.getLoc(), op.getMapOperands());
    auto maybeExpandedMap =
        expandAffineMap(rewriter, op.getLoc(), op.getAffineMap(), indices);
    if (!maybeExpandedMap)
      return failure();

    // Build memref.store valueToStore, memref[expandedMap.results].
    rewriter.replaceOpWithNewOp<memref::StoreOp>(
        op, op.getValueToStore(), op.getMemRef(), *maybeExpandedMap);
    return success();
  }
};

/// Apply the affine maps from an 'affine.dma_start' operation to each of their
/// respective map operands, and feed the results to a newly created
/// 'memref.dma_start' operation (which replaces the original
/// 'affine.dma_start').
class AffineDmaStartLowering : public OpRewritePattern<AffineDmaStartOp> {
public:
  using OpRewritePattern<AffineDmaStartOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineDmaStartOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value, 8> operands(op.getOperands());
    auto operandsRef = llvm::ArrayRef(operands);

    // Expand affine map for DMA source memref.
    auto maybeExpandedSrcMap = expandAffineMap(
        rewriter, op.getLoc(), op.getSrcMap(),
        operandsRef.drop_front(op.getSrcMemRefOperandIndex() + 1));
    if (!maybeExpandedSrcMap)
      return failure();
    // Expand affine map for DMA destination memref.
    auto maybeExpandedDstMap = expandAffineMap(
        rewriter, op.getLoc(), op.getDstMap(),
        operandsRef.drop_front(op.getDstMemRefOperandIndex() + 1));
    if (!maybeExpandedDstMap)
      return failure();
    // Expand affine map for DMA tag memref.
    auto maybeExpandedTagMap = expandAffineMap(
        rewriter, op.getLoc(), op.getTagMap(),
        operandsRef.drop_front(op.getTagMemRefOperandIndex() + 1));
    if (!maybeExpandedTagMap)
      return failure();

    // Build memref.dma_start operation with affine map results.
    rewriter.replaceOpWithNewOp<memref::DmaStartOp>(
        op, op.getSrcMemRef(), *maybeExpandedSrcMap, op.getDstMemRef(),
        *maybeExpandedDstMap, op.getNumElements(), op.getTagMemRef(),
        *maybeExpandedTagMap, op.getStride(), op.getNumElementsPerStride());
    return success();
  }
};

/// Apply the affine map from an 'affine.dma_wait' operation tag memref,
/// and feed the results to a newly created 'memref.dma_wait' operation (which
/// replaces the original 'affine.dma_wait').
class AffineDmaWaitLowering : public OpRewritePattern<AffineDmaWaitOp> {
public:
  using OpRewritePattern<AffineDmaWaitOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineDmaWaitOp op,
                                PatternRewriter &rewriter) const override {
    // Expand affine map for DMA tag memref.
    SmallVector<Value, 8> indices(op.getTagIndices());
    auto maybeExpandedTagMap =
        expandAffineMap(rewriter, op.getLoc(), op.getTagMap(), indices);
    if (!maybeExpandedTagMap)
      return failure();

    // Build memref.dma_wait operation with affine map results.
    rewriter.replaceOpWithNewOp<memref::DmaWaitOp>(
        op, op.getTagMemRef(), *maybeExpandedTagMap, op.getNumElements());
    return success();
  }
};

/// Apply the affine map from an 'affine.vector_load' operation to its operands,
/// and feed the results to a newly created 'vector.load' operation (which
/// replaces the original 'affine.vector_load').
class AffineVectorLoadLowering : public OpRewritePattern<AffineVectorLoadOp> {
public:
  using OpRewritePattern<AffineVectorLoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineVectorLoadOp op,
                                PatternRewriter &rewriter) const override {
    // Expand affine map from 'affineVectorLoadOp'.
    SmallVector<Value> indices = materializeAffineMapOperandsAsIndex(
        rewriter, op.getLoc(), op.getMapOperands());
    auto resultOperands =
        expandAffineMap(rewriter, op.getLoc(), op.getAffineMap(), indices);
    if (!resultOperands)
      return failure();

    // Build vector.load memref[expandedMap.results].
    rewriter.replaceOpWithNewOp<vector::LoadOp>(
        op, op.getVectorType(), op.getMemRef(), *resultOperands);
    return success();
  }
};

/// Apply the affine map from an 'affine.vector_store' operation to its
/// operands, and feed the results to a newly created 'vector.store' operation
/// (which replaces the original 'affine.vector_store').
class AffineVectorStoreLowering : public OpRewritePattern<AffineVectorStoreOp> {
public:
  using OpRewritePattern<AffineVectorStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineVectorStoreOp op,
                                PatternRewriter &rewriter) const override {
    // Expand affine map from 'affineVectorStoreOp'.
    SmallVector<Value> indices = materializeAffineMapOperandsAsIndex(
        rewriter, op.getLoc(), op.getMapOperands());
    auto maybeExpandedMap =
        expandAffineMap(rewriter, op.getLoc(), op.getAffineMap(), indices);
    if (!maybeExpandedMap)
      return failure();

    rewriter.replaceOpWithNewOp<vector::StoreOp>(
        op, op.getValueToStore(), op.getMemRef(), *maybeExpandedMap);
    return success();
  }
};

void populateAffineToStdConversionPatterns(RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<
      AffineApplyLowering,
      AffineDmaStartLowering,
      AffineDmaWaitLowering,
      AffineLoadLowering,
      AffineMinLowering,
      AffineMaxLowering,
      AffineParallelLowering,
      AffinePrefetchLowering,
      AffineStoreLowering,
      AffineForLowering,
      AffineIfLowering,
      AffineYieldOpLowering>(patterns.getContext());
  // clang-format on
}

void populateAffineToVectorConversionPatterns(RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<
      AffineVectorLoadLowering,
      AffineVectorStoreLowering>(patterns.getContext());
  // clang-format on
}

struct AffineForLoweringPass
    : public ConvertAffineForBase<AffineForLoweringPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateAffineToStdConversionPatterns(patterns);
    populateAffineToVectorConversionPatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();

    bool hasAffineOp = false;
    getOperation()->walk([&](Operation *op) {
      if (op->getDialect() && op->getDialect()->getNamespace() ==
                                  AffineDialect::getDialectNamespace()) {
        hasAffineOp = true;
        op->emitError("failed to lower affine operation");
      }
    });
    if (hasAffineOp)
      signalPassFailure();
  }
};
} // namespace

/// Lowers If and For operations within a function into their lower level CFG
/// equivalent blocks.
namespace mlir {
std::unique_ptr<Pass> createAffineForLoweringPass() {
  return std::make_unique<AffineForLoweringPass>();
}
} // namespace mlir
