#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "TOR/PassDetail.h"
#include "TOR/Passes.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"


#define DEBUG_TYPE "remove-func-arg"


namespace {
    using namespace mlir;


    struct funcPattern : public OpConversionPattern<func::FuncOp> {
        using OpConversionPattern::OpConversionPattern;
        LogicalResult matchAndRewrite(func::FuncOp funcOp, OpAdaptor adaptor,
                        ConversionPatternRewriter &rewriter) const final {
            if (funcOp.getSymName() != "forward" || !funcOp.getArguments()[0].getUsers().empty())
                return failure();
            else {
                SmallVector<mlir::Type, 8> newArgTypes;
                newArgTypes.push_back(funcOp.getArgumentTypes()[1]);
                funcOp.eraseArgument(0);
                auto newFuncType = FunctionType::get(funcOp.getContext(), newArgTypes, funcOp.getResultTypes());
                funcOp.setType(newFuncType);
                auto newFuncOp = rewriter.clone(*funcOp.getOperation());
                 rewriter.replaceOp(funcOp, newFuncOp);
            }

            return success();
        }
    };


    
    struct RemoveFuncArgPass : public RemoveFuncArgBase<RemoveFuncArgPass> {
        void runOnOperation() override {
            MLIRContext &ctxt = getContext();
            ConversionTarget target(ctxt);
            RewritePatternSet patterns(&ctxt);
            patterns.add<funcPattern>(&ctxt);
            if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
                signalPassFailure();
            }
        }
    };

}; // namespace



namespace mlir {
    std::unique_ptr<OperationPass<mlir::ModuleOp>> createRemoveFuncArgPass() {
        return std::make_unique<RemoveFuncArgPass>();
    }
} // namespace mlir


