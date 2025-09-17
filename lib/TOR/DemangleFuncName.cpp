#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/Support/Debug.h"

#include "TOR/PassDetail.h"
#include "TOR/Passes.h"

#define DEBUG_TYPE "demangle-func-name"

namespace {
using namespace mlir;

StringRef getReallyfuncName(StringRef funcName) {
  if (funcName.find("hls") == 0) {
    return funcName.substr(5);
  }
  return funcName;
}

struct DemangleFuncNamePass : DemangleFuncNameBase<DemangleFuncNamePass> {
  void runOnOperation() override {
    auto moduleOp = getOperation();

    mlir::SmallVector<llvm::StringRef> removeAttrNames;
    for (auto namedAttr : moduleOp->getAttrs()) {
      auto attrName = namedAttr.getName().getValue();
      if (!(attrName.starts_with("hls.")))
        removeAttrNames.push_back(attrName);
    }
    for (auto removeAttrName: removeAttrNames) {
      moduleOp->removeAttr(removeAttrName);
    }
    DenseMap<StringRef, StringRef> funcNameMap;
    DenseSet<StringRef> newFuncNameSet;
    moduleOp->walk([&](func::FuncOp op) {
      StringRef name = op.getSymName();
      if (name.starts_with("_Z")) {
        std::string buf = name.str();
        llvm::ItaniumPartialDemangler d;
        if (!d.partialDemangle(buf.c_str()))
          if (char *res = d.getFunctionName(nullptr, nullptr)) {
            op.setSymName(getReallyfuncName(res));
            free(res);
          }
      }
    });
    moduleOp->walk([&](func::CallOp op) {
      StringRef name = op.getCallee();
      if (name.starts_with("_Z")) {
        std::string buf = name.str();
        llvm::ItaniumPartialDemangler d;
        if (!d.partialDemangle(buf.c_str()))
          if (char *res = d.getFunctionName(nullptr, nullptr)) {
            op.setCallee(getReallyfuncName(res));
            free(res);
          }
      }
    });
  }
};
} // namespace

namespace mlir {
std::unique_ptr<OperationPass<mlir::ModuleOp>> createDemangleFuncNamePass() {
  return std::make_unique<DemangleFuncNamePass>();
}

} // namespace mlir
