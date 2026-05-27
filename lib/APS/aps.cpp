#include "APS/APSDialect.h"
#include "APS/APSOps.h"

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/ADT/STLExtras.h"

#include "APS/APSDialect.cpp.inc"
#define GET_OP_CLASSES
#include "APS/APS.cpp.inc"

using namespace mlir;
using namespace aps;

void APSDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "APS/APS.cpp.inc"
  >();
}

//===----------------------------------------------------------------------===//
// MemoryMapOp
//===----------------------------------------------------------------------===//

ParseResult MemoryMapOp::parse(OpAsmParser &parser, OperationState &result) {
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, /*arguments=*/{}, /*argTypes=*/{}))
    return failure();

  return success();
}

void MemoryMapOp::print(OpAsmPrinter &p) {
  p << " ";
  p.printRegion(getRegion(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);
}

//===----------------------------------------------------------------------===//
// GlobalLoadOp Canonicalization
//===----------------------------------------------------------------------===//

namespace {
static bool isSameGlobalMemref(StringRef globalName, Value memref) {
  auto globalOp = memref.getDefiningOp<memref::GetGlobalOp>();
  return globalOp && globalOp.getName() == globalName;
}

static bool containsGlobalMemref(ValueRange memrefs, StringRef globalName) {
  return llvm::any_of(memrefs, [&](Value memref) {
    return isSameGlobalMemref(globalName, memref);
  });
}

static bool sameGlobal(GlobalLoad loadOp, GlobalStore storeOp) {
  return loadOp.getGlobalName() == storeOp.getGlobalName();
}

struct FoldGlobalLoadAfterStore : public OpRewritePattern<GlobalLoad> {
  using OpRewritePattern<GlobalLoad>::OpRewritePattern;

  LogicalResult matchAndRewrite(GlobalLoad loadOp,
                                 PatternRewriter &rewriter) const override {
    for (Operation *prevOp = loadOp->getPrevNode(); prevOp;
         prevOp = prevOp->getPrevNode()) {
      if (prevOp->getNumRegions() > 0)
        return failure();

      auto storeOp = dyn_cast<GlobalStore>(prevOp);
      if (storeOp && sameGlobal(loadOp, storeOp)) {
        rewriter.replaceOp(loadOp, storeOp.getValue());
        return success();
      }

      if (auto burstLoadOp = dyn_cast<MemBurstLoad>(prevOp)) {
        if (containsGlobalMemref(burstLoadOp.getMemrefs(),
                                  loadOp.getGlobalName()))
          return failure();
      }
    }

    return failure();
  }
};

struct RemoveDeadGlobalStore : public OpRewritePattern<GlobalStore> {
  using OpRewritePattern<GlobalStore>::OpRewritePattern;

  LogicalResult matchAndRewrite(GlobalStore storeOp,
                                 PatternRewriter &rewriter) const override {
    for (Operation *nextOp = storeOp->getNextNode(); nextOp;
         nextOp = nextOp->getNextNode()) {
      if (nextOp->getNumRegions() > 0)
        return failure();

      if (auto loadOp = dyn_cast<GlobalLoad>(nextOp)) {
        if (loadOp.getGlobalName() == storeOp.getGlobalName())
          return failure();
      }

      if (auto nextStoreOp = dyn_cast<GlobalStore>(nextOp)) {
        if (nextStoreOp.getGlobalName() == storeOp.getGlobalName()) {
          rewriter.eraseOp(storeOp);
          return success();
        }
      }
    }

    return failure();
  }
};
} // namespace

void aps::GlobalLoad::getCanonicalizationPatterns(RewritePatternSet &results,
                                                    MLIRContext *context) {
  results.add<FoldGlobalLoadAfterStore>(context);
}

void aps::GlobalStore::getCanonicalizationPatterns(RewritePatternSet &results,
                                                     MLIRContext *context) {
  results.add<RemoveDeadGlobalStore>(context);
}

//===----------------------------------------------------------------------===//
// MemLoad Canonicalization
//===----------------------------------------------------------------------===//

namespace {
static bool isSameMemref(Value lhs, Value rhs) {
  if (lhs == rhs)
    return true;

  auto lhsGlobal = lhs.getDefiningOp<memref::GetGlobalOp>();
  auto rhsGlobal = rhs.getDefiningOp<memref::GetGlobalOp>();
  return lhsGlobal && rhsGlobal && lhsGlobal.getName() == rhsGlobal.getName();
}

static bool containsMemref(ValueRange memrefs, Value memref) {
  return llvm::any_of(memrefs, [&](Value candidate) {
    return isSameMemref(candidate, memref);
  });
}

static bool isSameMemoryLocation(Value memref1, ValueRange indices1,
                                  Value memref2, ValueRange indices2) {
  return isSameMemref(memref1, memref2) && llvm::equal(indices1, indices2);
}

static bool mayReadMemref(Operation *op, Value memref) {
  if (auto loadOp = dyn_cast<MemLoad>(op))
    return isSameMemref(loadOp.getMemref(), memref);
  if (auto burstStoreOp = dyn_cast<MemBurstStore>(op))
    return containsMemref(burstStoreOp.getMemrefs(), memref);
  return false;
}

static bool mayWriteMemref(Operation *op, Value memref) {
  if (auto storeOp = dyn_cast<MemStore>(op))
    return isSameMemref(storeOp.getMemref(), memref);
  if (auto burstLoadOp = dyn_cast<MemBurstLoad>(op))
    return containsMemref(burstLoadOp.getMemrefs(), memref);
  return false;
}

struct FoldMemLoadAfterStore : public OpRewritePattern<MemLoad> {
  using OpRewritePattern<MemLoad>::OpRewritePattern;

  LogicalResult matchAndRewrite(MemLoad loadOp,
                                 PatternRewriter &rewriter) const override {
    Value loadMemref = loadOp.getMemref();
    ValueRange loadIndices = loadOp.getIndices();

    for (Operation *prevOp = loadOp->getPrevNode(); prevOp;
         prevOp = prevOp->getPrevNode()) {
      if (prevOp->getNumRegions() > 0)
        return failure();

      if (auto storeOp = dyn_cast<MemStore>(prevOp)) {
        Value storeMemref = storeOp.getMemref();
        ValueRange storeIndices = storeOp.getIndices();

        if (isSameMemoryLocation(loadMemref, loadIndices, storeMemref,
                                  storeIndices)) {
          rewriter.replaceOp(loadOp, storeOp.getValue());
          return success();
        }

        if (isSameMemref(storeMemref, loadMemref))
          return failure();
      }

      if (mayWriteMemref(prevOp, loadMemref))
        return failure();
    }

    return failure();
  }
};

struct RemoveDeadMemStore : public OpRewritePattern<MemStore> {
  using OpRewritePattern<MemStore>::OpRewritePattern;

  LogicalResult matchAndRewrite(MemStore storeOp,
                                 PatternRewriter &rewriter) const override {
    Value storeMemref = storeOp.getMemref();
    ValueRange storeIndices = storeOp.getIndices();

    for (Operation *nextOp = storeOp->getNextNode(); nextOp;
         nextOp = nextOp->getNextNode()) {
      if (nextOp->getNumRegions() > 0)
        return failure();

      if (auto nextStoreOp = dyn_cast<MemStore>(nextOp)) {
        Value nextMemref = nextStoreOp.getMemref();
        ValueRange nextIndices = nextStoreOp.getIndices();

        if (isSameMemoryLocation(storeMemref, storeIndices, nextMemref,
                                  nextIndices)) {
          rewriter.eraseOp(storeOp);
          return success();
        }

        if (isSameMemref(nextMemref, storeMemref))
          return failure();
      }

      if (auto nextLoadOp = dyn_cast<MemLoad>(nextOp)) {
        Value nextMemref = nextLoadOp.getMemref();
        ValueRange nextIndices = nextLoadOp.getIndices();

        if (isSameMemoryLocation(storeMemref, storeIndices, nextMemref,
                                  nextIndices))
          return failure();

        if (isSameMemref(nextMemref, storeMemref))
          return failure();
      }

      if (mayReadMemref(nextOp, storeMemref))
        return failure();
    }

    return failure();
  }
};
} // namespace

void aps::MemLoad::getCanonicalizationPatterns(RewritePatternSet &results,
                                                 MLIRContext *context) {
  results.add<FoldMemLoadAfterStore>(context);
}

void aps::MemStore::getCanonicalizationPatterns(RewritePatternSet &results,
                                                  MLIRContext *context) {
  results.add<RemoveDeadMemStore>(context);
}

// Force template instantiation for TypeID
namespace mlir::detail {
template struct TypeIDResolver<aps::APSDialect, void>;
}
