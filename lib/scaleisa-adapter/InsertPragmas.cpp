#include "scaleisa-adapter/Passes.h"
#include "Utils.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include <optional>

namespace mlir::scaleisa_adapter {
#define GEN_PASS_DEF_SCALEISAINSERTPRAGMAS
#include "scaleisa-adapter/Passes.h.inc"
namespace {
struct InsertPragmasPass : impl::ScaleISAInsertPragmasBase<InsertPragmasPass> {
  InsertPragmasPass() = default;
  explicit InsertPragmasPass(StringRef pragmaMode) { mode = pragmaMode.str(); }
  void runOnOperation() override {
    if (mode != "none" && mode != "pipeline" && mode != "unroll") {
      getOperation().emitError("pragma mode must be none, pipeline, or unroll");
      return signalPassFailure();
    }
    OpBuilder builder(&getContext());
    for (auto function : getOperation().getOps<func::FuncOp>()) {
      if (isCandidate(function) && !isInstruction(function)) {
        function.emitError("run scaleisa-split before inserting pragmas");
        return signalPassFailure();
      }
      if (!isInstruction(function))
        continue;
      function.walk([&](Operation *op) {
        if (!isa<affine::AffineForOp, scf::ForOp>(op))
          return;
        for (StringRef attr : {"pipeline", "II", "unroll", "scaleisa.pragma_rejected"})
          op->removeAttr(attr);
        bool innerLoop = false;
        op->walk([&](Operation *nested) {
          if (nested != op && isa<affine::AffineForOp, scf::ForOp>(nested))
            innerLoop = true;
        });
        if (innerLoop || mode == "none")
          return;
        if (mode == "pipeline") {
          op->setAttr("pipeline", builder.getI64IntegerAttr(1));
          if (initiationInterval > 0)
            op->setAttr("II", builder.getI64IntegerAttr(initiationInterval));
          return;
        }
        auto affineLoop = dyn_cast<affine::AffineForOp>(op);
        std::optional<uint64_t> tripCount;
        if (affineLoop)
          tripCount = affine::getConstantTripCount(affineLoop);
        if (auto scfLoop = dyn_cast<scf::ForOp>(op)) {
          APInt lower, upper, step;
          if (matchPattern(scfLoop.getLowerBound(), m_ConstantInt(&lower)) &&
              matchPattern(scfLoop.getUpperBound(), m_ConstantInt(&upper)) &&
              matchPattern(scfLoop.getStep(), m_ConstantInt(&step)) && step.isStrictlyPositive() &&
              (!scfLoop->hasAttr("unsignedCmp") || (!lower.isNegative() && !upper.isNegative()))) {
            unsigned width = lower.getBitWidth() + 2;
            APInt count = upper.sext(width) - lower.sext(width);
            APInt stride = step.sext(width);
            count = count.isNonPositive() ? APInt(width, 0) :
                (count + stride - 1).sdiv(stride);
            if (count.getActiveBits() <= 64)
              tripCount = count.getZExtValue();
          }
        }
        uint64_t count = 0;
        op->walk([&](Operation *) { ++count; });
        if (!tripCount || !*tripCount || count > maxExpandedOps / *tripCount) {
          op->setAttr("scaleisa.pragma_rejected", builder.getStringAttr(
              !tripCount ? "full unroll requires a static affine or SCF trip count"
                         : "full unroll exceeds the expanded-operation limit or has zero iterations"));
          return;
        }
        op->setAttr("unroll", builder.getI64IntegerAttr(0));
      });
    }
  }
};
} // namespace
std::unique_ptr<Pass> createInsertPragmasPass() {
  return std::make_unique<InsertPragmasPass>();
}
std::unique_ptr<Pass> createInsertPragmasPass(StringRef mode) {
  return std::make_unique<InsertPragmasPass>(mode);
}
} // namespace mlir::scaleisa_adapter
