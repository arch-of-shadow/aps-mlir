#include "APS/Passes.h"

#include "APS/APSOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {

#define GEN_PASS_DEF_APSRAISESCFTOAFFINE
#include "APS/Passes.h.inc"

namespace {

static std::optional<int64_t> getConstantStep(Value step) {
  if (auto constIndex = step.getDefiningOp<arith::ConstantIndexOp>())
    return constIndex.value();

  if (auto constInt = step.getDefiningOp<arith::ConstantIntOp>())
    return constInt.value();

  auto constOp = step.getDefiningOp<arith::ConstantOp>();
  if (!constOp)
    return std::nullopt;

  auto attr = dyn_cast<IntegerAttr>(constOp.getValue());
  if (!attr)
    return std::nullopt;
  return attr.getInt();
}

static bool canMaterializeIndexBound(Value value) {
  Type type = value.getType();
  if (type.isIndex())
    return true;
  return isa<IntegerType>(type);
}

static Value materializeIndexBound(PatternRewriter &rewriter, Location loc,
                                   Value value) {
  if (value.getType().isIndex())
    return value;

  if (auto constInt = value.getDefiningOp<arith::ConstantIntOp>())
    return rewriter.create<arith::ConstantIndexOp>(loc, constInt.value());

  if (auto constOp = value.getDefiningOp<arith::ConstantOp>()) {
    if (auto attr = dyn_cast<IntegerAttr>(constOp.getValue()))
      return rewriter.create<arith::ConstantIndexOp>(loc, attr.getInt());
  }

  return rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(),
                                             value);
}

static AffineMap symbolIdentityMap(MLIRContext *context) {
  Builder builder(context);
  return AffineMap::get(/*dimCount=*/0, /*symbolCount=*/1,
                        builder.getAffineSymbolExpr(0));
}

static SmallVector<Value> materializeIndexIndices(PatternRewriter &rewriter,
                                                  Location loc,
                                                  ValueRange indices,
                                                  IRMapping &mapping) {
  SmallVector<Value> mappedIndices;
  mappedIndices.reserve(indices.size());
  for (Value index : indices) {
    Value mapped = mapping.lookupOrDefault(index);
    if (!mapped.getType().isIndex())
      mapped = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(),
                                                   mapped);
    mappedIndices.push_back(mapped);
  }
  return mappedIndices;
}

static LogicalResult cloneOrConvertBodyOp(PatternRewriter &rewriter,
                                          Operation &op, IRMapping &mapping) {
  Location loc = op.getLoc();
  if (auto load = dyn_cast<aps::MemLoad>(op)) {
    SmallVector<Value> indices =
        materializeIndexIndices(rewriter, loc, load.getIndices(), mapping);
    auto newLoad = rewriter.create<memref::LoadOp>(
        loc, mapping.lookupOrDefault(load.getMemref()), indices);
    newLoad->setAttrs(load->getAttrs());
    mapping.map(load.getResult(), newLoad.getResult());
    return success();
  }

  if (auto store = dyn_cast<aps::MemStore>(op)) {
    SmallVector<Value> indices =
        materializeIndexIndices(rewriter, loc, store.getIndices(), mapping);
    auto newStore = rewriter.create<memref::StoreOp>(
        loc, mapping.lookupOrDefault(store.getValue()),
        mapping.lookupOrDefault(store.getMemref()), indices);
    newStore->setAttrs(store->getAttrs());
    return success();
  }

  rewriter.clone(op, mapping);
  return success();
}

struct APSRaiseForPattern : OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    if (!canMaterializeIndexBound(forOp.getLowerBound()) ||
        !canMaterializeIndexBound(forOp.getUpperBound()))
      return failure();

    std::optional<int64_t> step = getConstantStep(forOp.getStep());
    if (!step || *step <= 0)
      return failure();

    Location loc = forOp.getLoc();
    rewriter.setInsertionPoint(forOp);
    Value lowerBound =
        materializeIndexBound(rewriter, loc, forOp.getLowerBound());
    Value upperBound =
        materializeIndexBound(rewriter, loc, forOp.getUpperBound());

    AffineMap boundMap = symbolIdentityMap(rewriter.getContext());
    auto affineFor = rewriter.create<affine::AffineForOp>(
        loc, ValueRange{lowerBound}, boundMap, ValueRange{upperBound}, boundMap,
        *step, forOp.getInitArgs());
    affineFor->setAttrs(forOp->getAttrs());

    Block &affineBody = affineFor.getRegion().front();
    if (affineFor.getNumIterOperands() == 0) {
      if (Operation *terminator = affineBody.getTerminator())
        rewriter.eraseOp(terminator);
    }

    IRMapping mapping;
    rewriter.setInsertionPointToStart(&affineBody);
    Value mappedIV = affineFor.getInductionVar();
    Type oldIVType = forOp.getInductionVar().getType();
    if (!oldIVType.isIndex()) {
      mappedIV = rewriter.create<arith::IndexCastOp>(loc, oldIVType, mappedIV);
    }
    mapping.map(forOp.getInductionVar(), mappedIV);

    for (auto [oldArg, newArg] :
         llvm::zip(forOp.getRegionIterArgs(), affineFor.getRegionIterArgs()))
      mapping.map(oldArg, newArg);

    auto oldYield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    for (Operation &op : forOp.getBody()->without_terminator())
      if (failed(cloneOrConvertBodyOp(rewriter, op, mapping)))
        return failure();

    SmallVector<Value> yielded;
    yielded.reserve(oldYield.getNumOperands());
    for (Value operand : oldYield.getOperands())
      yielded.push_back(mapping.lookupOrDefault(operand));
    rewriter.create<affine::AffineYieldOp>(oldYield.getLoc(), yielded);

    rewriter.replaceOp(forOp, affineFor.getResults());
    return success();
  }
};

struct APSRaiseSCFToAffinePass
    : impl::APSRaiseSCFToAffineBase<APSRaiseSCFToAffinePass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<APSRaiseForPattern>(&getContext());

    GreedyRewriteConfig config;
    config.setStrictness(GreedyRewriteStrictness::ExistingAndNewOps);
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            config))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> createAPSRaiseSCFToAffinePass() {
  return std::make_unique<APSRaiseSCFToAffinePass>();
}

} // namespace mlir
