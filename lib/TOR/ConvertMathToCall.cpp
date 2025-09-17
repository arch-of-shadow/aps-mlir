#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Debug.h"

#include "nlohmann/json.hpp"

#include "TOR/PassDetail.h"
#include "TOR/Passes.h"

#include <fstream>

#define DEBUG_TYPE "convert-math-to-call"

namespace {
using namespace mlir;

void lowerExp(ModuleOp moduleOp) {
  MLIRContext *context = moduleOp.getContext();
  OpBuilder builder(context);
  builder.setInsertionPointToEnd(moduleOp.getBody());
  auto f64 = builder.getF64Type();
  auto funcType = builder.getFunctionType({f64}, {f64});
  auto func = builder.create<func::FuncOp>(UnknownLoc::get(context),
                                           "expByBuilder", funcType);
  builder.setInsertionPointToEnd(func.addEntryBlock());
  auto constantOne = builder.create<arith::ConstantOp>(
      builder.getUnknownLoc(), builder.getF64FloatAttr(1.0));
  std::vector<mlir::Value> start = {
      builder.create<arith::ConstantIndexOp>(builder.getUnknownLoc(), 1)};
  std::vector<mlir::Value> end = {
      builder.create<arith::ConstantIndexOp>(builder.getUnknownLoc(), 100)};
  AffineMap map = builder.getSymbolIdentityMap();
  auto affineForOp = builder.create<affine::AffineForOp>(
      builder.getUnknownLoc(), start, map, end, map, 1,
      ValueRange({constantOne, constantOne}));
  auto body = affineForOp.getBody();
  auto savedIP = builder.saveInsertionPoint();
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(body);

  Value indexCastI32 = builder.create<arith::IndexCastOp>(
      builder.getUnknownLoc(), builder.getI32Type(), body->getArgument(0));
  auto indexCastF64 = builder.create<arith::SIToFPOp>(builder.getUnknownLoc(),
                                                      f64, indexCastI32);
  Value div = builder.create<arith::DivFOp>(builder.getUnknownLoc(),
                                            func.getArgument(0), indexCastF64);
  Value mul = builder.create<arith::MulFOp>(builder.getUnknownLoc(),
                                            body->getArgument(1), div);
  Value add = builder.create<arith::AddFOp>(builder.getUnknownLoc(),
                                            body->getArgument(2), mul);

  builder.create<affine::AffineYieldOp>(builder.getUnknownLoc(),
                                        ValueRange({mul, add}));
  builder.restoreInsertionPoint(savedIP);
  builder.create<func::ReturnOp>(builder.getUnknownLoc(),
                                 ValueRange({affineForOp.getResult(1)}));
}

struct exp2Call : public OpRewritePattern<math::ExpOp> {
  using OpRewritePattern<math::ExpOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(math::ExpOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<func::CallOp>(op, "expByBuilder", op.getType(),
                                              op->getOperands());
    return success();
  }
};

struct sin2Call : public OpRewritePattern<math::SinOp> {
  using OpRewritePattern<math::SinOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(math::SinOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<func::CallOp>(op, "sin", op.getType(),
                                              op->getOperands());
    return success();
  }
};

struct cos2Call : public OpRewritePattern<math::CosOp> {
  using OpRewritePattern<math::CosOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(math::CosOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<func::CallOp>(op, "cos", op.getType(),
                                              op->getOperands());
    return success();
  }
};

struct abs2Call : public OpRewritePattern<math::AbsIOp> {
  using OpRewritePattern<math::AbsIOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(math::AbsIOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<func::CallOp>(op, "abs", op.getType(),
                                              op->getOperands());
    return success();
  }
};

struct log2Call : public OpRewritePattern<math::LogOp> {
  using OpRewritePattern<math::LogOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(math::LogOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<func::CallOp>(op, "log", op.getType(),
                                              op->getOperands());
    return success();
  }
};

void warningMissingIp(StringRef IPType) {
  llvm::errs() << "warning: The resource file is missing the " << IPType
               << " IP, a mathematical implementation will be used instead. "
                  "Please add it.\n";
}

void warningMissingIpWithAssert(StringRef IPType) {
  llvm_unreachable(("warning: The resource file is missing the " + IPType +
                    " IP, Please add it.")
                       .str()
                       .c_str());
}

template <typename T> class IpCheckWithAssert {
  StringRef ipName;

public:
  IpCheckWithAssert(StringRef ipName) : ipName(ipName){};
  void checkThisIp(llvm::SmallSet<StringRef, 32> &IpSet, ModuleOp moduleOp) {
    bool needAddThisMathOp = false;
    moduleOp.walk([&](T mathOp) {
      if (!IpSet.count(ipName)) {
        needAddThisMathOp = true;
      }
      return WalkResult::interrupt();
    });
    if (needAddThisMathOp)
      warningMissingIpWithAssert(ipName);
  }
};

struct ConvertMathToCallPass : ConvertMathToCallBase<ConvertMathToCallPass> {
  void runOnOperation() override {
    auto moduleOp = getOperation();
    std::string resource_path = (this->resource).getValue();
    std::ifstream istrm(resource_path, std::ios::in);
    nlohmann::json config;
    istrm >> config;
    llvm::SmallSet<StringRef, 32> IpSet;
    for (auto &res : config.items()) {
      IpSet.insert(res.key());
    }
    GreedyRewriteConfig rewriteConfig;
    rewriteConfig.setStrictness(GreedyRewriteStrictness::ExistingOps);
    bool needAddExpOp = false;
    moduleOp.walk([&](math::ExpOp AI) {
      if (!IpSet.count("exp")) {
        needAddExpOp = true;
      }
      return WalkResult::interrupt();
    });
    if (needAddExpOp) {
      warningMissingIp("exp");
      lowerExp(moduleOp);
      moduleOp.walk([&](math::ExpOp AI) {
        RewritePatternSet patterns(&getContext());
        patterns.insert<exp2Call>(&getContext());
        (void)applyOpPatternsAndFold(AI.getOperation(), std::move(patterns),
                                     rewriteConfig);
      });
    }
    bool needAddSinOp = false;
    moduleOp.walk([&](math::SinOp AI) {
      if (!IpSet.count("sin")) {
        needAddSinOp = true;
      }
      return WalkResult::interrupt();
    });
    bool needAddCosOp = false;
    moduleOp.walk([&](math::CosOp AI) {
      if (!IpSet.count("cos")) {
        needAddCosOp = true;
      }
      return WalkResult::interrupt();
    });
    bool needAddLogOp = false;
    moduleOp.walk([&](math::LogOp AI) {
      if (!IpSet.count("log")) {
        needAddLogOp = true;
      }
      return WalkResult::interrupt();
    });
    bool needAddAbsOp = false;
    moduleOp.walk([&](math::AbsIOp AI) {
      if (!IpSet.count("absi")) {
        needAddAbsOp = true;
      }
      return WalkResult::interrupt();
    });
    if (needAddSinOp || needAddCosOp || needAddLogOp || needAddAbsOp) {
      MLIRContext *libCtx = moduleOp.getContext();
      auto module =
          parseSourceFile<ModuleOp>("demo/libc/math-affine.mlir", libCtx);
      MLIRContext *context = moduleOp.getContext();
      OpBuilder builder(context);
      module->walk([&](func::FuncOp func) {
        auto new_func = cast<func::FuncOp>(builder.clone(*func));
        moduleOp.push_back(new_func);
      });
    }
    if (needAddSinOp) {
      warningMissingIp("sin");
      moduleOp.walk([&](math::SinOp AI) {
        RewritePatternSet patterns(&getContext());
        patterns.insert<sin2Call>(&getContext());
        (void)applyOpPatternsAndFold(AI.getOperation(), std::move(patterns),
                                     rewriteConfig);
      });
    }
    if (needAddCosOp) {
      warningMissingIp("cos");
      moduleOp.walk([&](math::CosOp AI) {
        RewritePatternSet patterns(&getContext());
        patterns.insert<cos2Call>(&getContext());
        (void)applyOpPatternsAndFold(AI.getOperation(), std::move(patterns),
                                     rewriteConfig);
      });
    }
    if (needAddLogOp) {
      warningMissingIp("log");
      moduleOp.walk([&](math::LogOp AI) {
        RewritePatternSet patterns(&getContext());
        patterns.insert<log2Call>(&getContext());
        (void)applyOpPatternsAndFold(AI.getOperation(), std::move(patterns),
                                     rewriteConfig);
      });
    }
    if (needAddAbsOp) {
      warningMissingIp("absi");
      moduleOp.walk([&](math::AbsIOp AI) {
        RewritePatternSet patterns(&getContext());
        patterns.insert<abs2Call>(&getContext());
        (void)applyOpPatternsAndFold(AI.getOperation(), std::move(patterns),
                                     rewriteConfig);
      });
    }
    IpCheckWithAssert<math::AbsFOp> absfOpCheck("absf");
    absfOpCheck.checkThisIp(IpSet, moduleOp);
    IpCheckWithAssert<math::AbsIOp> absiOpCheck("abs");
    absiOpCheck.checkThisIp(IpSet, moduleOp);
    IpCheckWithAssert<math::CeilOp> ceilOpCheck("ceil");
    ceilOpCheck.checkThisIp(IpSet, moduleOp);
    IpCheckWithAssert<math::FloorOp> floorOpCheck("floor");
    floorOpCheck.checkThisIp(IpSet, moduleOp);
    IpCheckWithAssert<math::RoundOp> roundOpCheck("round");
    roundOpCheck.checkThisIp(IpSet, moduleOp);

    IpCheckWithAssert<math::TanOp> tanOpCheck("tan");
    tanOpCheck.checkThisIp(IpSet, moduleOp);
    IpCheckWithAssert<math::TanhOp> tanhOpCheck("tanh");
    tanhOpCheck.checkThisIp(IpSet, moduleOp);

    IpCheckWithAssert<math::ErfOp> erfOpCheck("erf");
    erfOpCheck.checkThisIp(IpSet, moduleOp);
    IpCheckWithAssert<math::PowFOp> powfOpCheck("powf");
    powfOpCheck.checkThisIp(IpSet, moduleOp);
  }
};
} // namespace

namespace mlir {
std::unique_ptr<OperationPass<mlir::ModuleOp>> createConvertMathToCallPass() {
  return std::make_unique<ConvertMathToCallPass>();
}

} // namespace mlir
