#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/Debug.h"

#include "TOR/PassDetail.h"
#include "TOR/Passes.h"

#define DEBUG_TYPE "memref-global-to-constant"

namespace {
  using namespace mlir;

  struct MemrefGlobalToConstant : public OpRewritePattern<ModuleOp> {
    using OpRewritePattern<ModuleOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(ModuleOp moduleOp, PatternRewriter &rewriter) const override {
      moduleOp.walk([&](memref::GetGlobalOp getGlobalOp) {
        auto check = [&]() {
          for (auto *user : getGlobalOp.getResult().getUsers()) {
            if (!llvm::isa<affine::AffineLoadOp>(user) && !llvm::isa<memref::LoadOp>(user)) {
              return false;
            }
          }
          return true;
        };
        auto getSingleAttr = [&](memref::GlobalOp op) -> std::optional<Attribute> {
          if (!op.getInitialValue().has_value()) {
            return std::nullopt;
          }
          if (auto denseAttr = dyn_cast<ElementsAttr>(op.getInitialValue().value())) {
            if (denseAttr.getNumElements() != 1) {
              return std::nullopt;
            }
            return denseAttr.getValues<Attribute>()[0];
          }
          return std::nullopt;
        };
        
        auto globalOp = moduleOp.lookupSymbol<memref::GlobalOp>(getGlobalOp.getName());
        if (!check() || !globalOp) {
          return;
        }
        auto attr = getSingleAttr(globalOp);
        if (!attr.has_value()) {
          return;
        }
        rewriter.setInsertionPoint(getGlobalOp);
        auto constOp = rewriter.create<arith::ConstantOp>(getGlobalOp.getLoc(), 
            dyn_cast<TypedAttr>(attr.value()));
        for (auto *user : getGlobalOp.getResult().getUsers()) {
          user->replaceAllUsesWith(constOp);
          user->erase();
        }
        getGlobalOp.erase();
      });
      return success();
    }
  };

  struct MemrefGlobalToConstantPass : MemrefGlobalToConstantBase<MemrefGlobalToConstantPass> {
    void runOnOperation() override {
      auto op = getOperation().getOperation();
      GreedyRewriteConfig config;
      config.setStrictness(GreedyRewriteStrictness::ExistingOps);
      {
        RewritePatternSet Patterns(&getContext());
        Patterns.add<MemrefGlobalToConstant>(&getContext());
        if (failed(applyOpPatternsAndFold(op, std::move(Patterns), config))) {
          signalPassFailure();
        }
      }
    }
  };
} // namespace

namespace mlir {
  std::unique_ptr<OperationPass<mlir::ModuleOp>> createMemrefGlobalToConstantPass() {
    return std::make_unique<MemrefGlobalToConstantPass>();
  }
} // namespace mlir
