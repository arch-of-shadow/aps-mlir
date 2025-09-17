#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/Debug.h"

#include "TOR/DialectCreater.h"
#include "TOR/PassDetail.h"
#include "TOR/Passes.h"

#define DEBUG_TYPE "unification-index-cast"

namespace {
    using namespace mlir;

    struct UnificationIndexCast : public OpRewritePattern<func::FuncOp> {
        using OpRewritePattern<func::FuncOp>::OpRewritePattern;

        LogicalResult matchAndRewrite(func::FuncOp funcOp, PatternRewriter &rewriter) const override {
            auto create = OpCreater(rewriter, funcOp.getLoc());
            auto number = 0, target = 32;
            funcOp.walk([&](arith::IndexCastOp op) {
                rewriter.setInsertionPoint(op);
                if (auto itype = dyn_cast<IntegerType>(op.getIn().getType())) {
                    int width = itype.getWidth();
                    auto type = rewriter.getIntegerType(target);
                    if (width != target) {
                        auto targetOp = width > target ? create.arith.trunci(type, op.getIn()) :
                                create.arith.extui(type, op.getIn());
                        op.setOperand(targetOp->getResult(0));
                        number++;
                    }
                }
            });
            LLVM_DEBUG(llvm::dbgs() << "arith.index_cast number: " << number << "\n");
            return success();
        }
    };

    struct UnificationIndexCastPass : UnificationIndexCastBase<UnificationIndexCastPass> {
        void runOnOperation() override {
            {
                auto op = getOperation().getOperation();
                RewritePatternSet Patterns(&getContext());
                Patterns.add<UnificationIndexCast>(&getContext());
                GreedyRewriteConfig config;
                config.setStrictness(GreedyRewriteStrictness::ExistingOps);
                if (failed(applyOpPatternsAndFold(op, std::move(Patterns), config))) {
                    signalPassFailure();
                }
            }
        }
    };
} // namespace

namespace mlir {
    std::unique_ptr<OperationPass<mlir::func::FuncOp>> createUnificationIndexCastPass() {
        return std::make_unique<UnificationIndexCastPass>();
    }
} // namespace mlir
