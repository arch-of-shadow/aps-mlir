#include "APS/APSDialect.h"
#include "APS/APSOps.h"

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"

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
/// Fold GlobalLoad that immediately follows a GlobalStore to the same global.
/// Pattern: aps.globalstore %v, @x followed by %y = aps.globalload @x => replace %y with %v
///
/// This pattern enables store->load fusion after loop unrolling, converting:
///   aps.globalstore %1, @count
///   %2 = aps.globalload @count
/// into:
///   aps.globalstore %1, @count
///   (use %1 directly instead of %2)
struct FoldGlobalLoadAfterStore : public OpRewritePattern<GlobalLoad> {
  using OpRewritePattern<GlobalLoad>::OpRewritePattern;

  LogicalResult matchAndRewrite(GlobalLoad loadOp,
                                 PatternRewriter &rewriter) const override {
    // Get the global symbol being loaded
    auto loadSymbol = loadOp.getGlobalName();

    // Walk backwards from the load to find the most recent store to the same global
    Operation *op = loadOp.getOperation();
    Block *block = op->getBlock();

    // Iterate backwards through the block from the load operation
    auto it = Block::iterator(op);
    if (it == block->begin())
      return failure();

    --it;
    while (true) {
      Operation *prevOp = &*it;

      // IMPORTANT: Stop at operations with regions (loops, conditionals, etc.)
      // We cannot fold across region boundaries because the control flow
      // means the store might not dominate the load
      if (prevOp->getNumRegions() > 0) {
        // Found a region-bearing operation (loop, conditional, etc.)
        // Cannot safely fold across this boundary
        return failure();
      }

      // Check if this is a GlobalStore to the same global
      if (auto storeOp = dyn_cast<GlobalStore>(prevOp)) {
        if (storeOp.getGlobalName() == loadSymbol) {
          // Found matching store! Replace load with the stored value
          rewriter.replaceOp(loadOp, storeOp.getValue());
          return success();
        }
      }

      // Check if this operation might interfere with the global
      // Only stop if it's another GlobalLoad/GlobalStore to the SAME global
      // (memory operations to other globals or memrefs don't interfere)
      if (auto otherLoadOp = dyn_cast<GlobalLoad>(prevOp)) {
        if (otherLoadOp.getGlobalName() == loadSymbol) {
          // Another load to same global - safe to continue (it doesn't modify the value)
        }
      } else if (auto otherStoreOp = dyn_cast<GlobalStore>(prevOp)) {
        if (otherStoreOp.getGlobalName() == loadSymbol) {
          // Another store to same global - already handled above, shouldn't reach here
          return failure();
        }
      }
      // Note: We don't check for general memory effects because:
      // 1. GlobalLoad/GlobalStore operations are specific to named globals
      // 2. memref.load/store operations operate on different memory locations
      // 3. Only GlobalLoad/GlobalStore to the same symbol can interfere

      // Move to previous operation
      if (it == block->begin())
        return failure();
      --it;
    }

    return failure();
  }
};

/// Remove GlobalStore that is immediately followed by another GlobalStore to the same global.
/// Pattern: aps.globalstore %v1, @x followed by aps.globalstore %v2, @x => remove first store
///
/// This pattern removes redundant stores after loop unrolling:
///   aps.globalstore %1, @count
///   aps.globalstore %2, @count
/// becomes:
///   aps.globalstore %2, @count  (first store removed since it's overwritten)
struct RemoveDeadGlobalStore : public OpRewritePattern<GlobalStore> {
  using OpRewritePattern<GlobalStore>::OpRewritePattern;

  LogicalResult matchAndRewrite(GlobalStore storeOp,
                                 PatternRewriter &rewriter) const override {
    auto storeSymbol = storeOp.getGlobalName();

    // Walk forward from this store to find if another store to same global follows
    Operation *op = storeOp.getOperation();
    Block *block = op->getBlock();

    auto it = Block::iterator(op);
    ++it; // Move past current store

    if (it == block->end())
      return failure();

    // Scan forward to find the next GlobalStore to the same global
    while (it != block->end()) {
      Operation *nextOp = &*it;

      // IMPORTANT: Stop at operations with regions (loops, conditionals, etc.)
      // We cannot remove stores if a region exists between this store and the next,
      // because the region might read the value before it's overwritten
      if (nextOp->getNumRegions() > 0) {
        // Found a region-bearing operation (loop, conditional, etc.)
        // Cannot safely remove this store
        return failure();
      }

      // Check if this is a GlobalStore to the same global
      if (auto nextStoreOp = dyn_cast<GlobalStore>(nextOp)) {
        if (nextStoreOp.getGlobalName() == storeSymbol) {
          // Found another store to same global! Remove the current (earlier) store
          rewriter.eraseOp(storeOp);
          return success();
        }
      }

      // Check if this is a GlobalLoad to the same global - can't remove store
      if (auto nextLoadOp = dyn_cast<GlobalLoad>(nextOp)) {
        if (nextLoadOp.getGlobalName() == storeSymbol) {
          // Load uses this store, can't remove it
          return failure();
        }
      }

      // Note: memref operations don't interfere with named globals
      ++it;
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

// Force template instantiation for TypeID
namespace mlir::detail {
template struct TypeIDResolver<aps::APSDialect, void>;
}