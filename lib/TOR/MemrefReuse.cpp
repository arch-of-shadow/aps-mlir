#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"

#include "TOR/DialectCreater.h"
#include "TOR/PassDetail.h"
#include "TOR/Passes.h"

#define DEBUG_TYPE "memref-reuse"

namespace {
    using namespace mlir;

    template<typename T>
    int64_t getTotalElementCount(T op) {
        int64_t numElements = 1;
        for (auto dim : op.getType().getShape()) {
            numElements *= dim;
        }
        return numElements;
    } 

    template<typename T>
    int64_t computeMemory(T op) {
        return getTotalElementCount<T>(op) * op.getType().getElementTypeBitWidth() / 8;
    }
        
    struct MemrefReuse : public OpRewritePattern<func::FuncOp> {
        MemrefReuse(MLIRContext *ctx) : OpRewritePattern<func::FuncOp>(ctx) {}

        LogicalResult matchAndRewrite(func::FuncOp funcOp, PatternRewriter &rewriter) const override {
            auto check = [&](memref::AllocaOp op) {
                if (op->getAttr("reuse") == rewriter.getBoolAttr(true)) {
                    return false;
                }
                if (!llvm::isa<func::FuncOp>(op->getParentOp())) {
                    return false;
                }
                for (auto &use : op.getResult().getUses()) {
                    if (!llvm::isa<memref::LoadOp>(use.getOwner()) && 
                        !llvm::isa<memref::StoreOp>(use.getOwner())) {
                        return false;
                    }
                }
                return true;
            };
            SmallVector<memref::AllocaOp> ops;
            funcOp.walk([&](memref::AllocaOp op) {
                if (check(op)) {
                    ops.push_back(op);
                }
            });
            auto create = OpCreater(rewriter, funcOp.getLoc());
            auto number = 0, memory = 0, color = 0;
            SmallVector<int> colors(ops.size(), -1);
            for (int i = 0, r = ops.size(); i < r; ++i) {
                if (~colors[i]) continue;
                colors[i] = color;
                auto lastI = i;
                auto size = getTotalElementCount<memref::AllocaOp>(ops[i]);
                for (int j = i + 1; j < r; ++j) {
                    if (colors[j] == -1 && canReuse(ops[lastI], ops[j])) {
                        colors[j] = color;
                        size = std::max(size, getTotalElementCount<memref::AllocaOp>(ops[j]));
                        lastI = j;
                    }
                }
                rewriter.setInsertionPoint(ops[i]);
                auto alloca = create.memref.alloca(MemRefType::get({size}, ops[i].getType().getElementType()));
                alloca->setAttr("reuse", rewriter.getBoolAttr(true));
                memory -= computeMemory<memref::AllocaOp>(alloca);
                number--;

                auto getNewIndices = [&](ArrayRef<int64_t> shape, Operation::operand_range indices) {
                    assert(shape.size() == indices.size());
                    Value value = indices[0];
                    for (int i = 1, r = shape.size(); i < r; ++i) {
                        value = create.arith.muli(value, create.arith.constantIndex(shape[i]));
                        value = create.arith.addi(value, indices[i]);
                    }
                    return SmallVector<Value>({value});
                };

                for (int j = i; j < r; ++j) {
                    if (colors[j] != color) continue;
                    auto shape = ops[j].getType().getShape();
                    for (auto &use : ops[j].getResult().getUses()) {
                        rewriter.setInsertionPoint(use.getOwner());
                        if (auto loadOp = llvm::dyn_cast<memref::LoadOp>(use.getOwner())) {
                            rewriter.replaceOp(loadOp, create.memref.load(alloca, getNewIndices(shape, loadOp.getIndices())));
                        } else if (auto storeOp = llvm::dyn_cast<memref::StoreOp>(use.getOwner())) {
                            create.memref.store(storeOp.getValue(), alloca, getNewIndices(shape, storeOp.getIndices()));
                            storeOp.erase();
                        } else {
                            use.getOwner()->dump();
                            assert(false);
                        }
                    }
                    memory += computeMemory<memref::AllocaOp>(ops[j]);
                    number++;
                    ops[j].erase();
                }
                color++;
            }
            llvm::outs() << "alloca reuse number: " << number << ", reuse memory: " << memory << " bytes" << "\n";
            return success();
        }

        bool canReuse(memref::AllocaOp op1, memref::AllocaOp op2) const {
            auto getOp = [&](Operation *op) {
                auto currentOp = op;
                while (currentOp) {
                    if (llvm::isa<func::FuncOp>(currentOp->getParentOp())) {
                        return currentOp;
                    }
                    currentOp = currentOp->getParentOp();
                }
                assert(false);
            };
            if (op1.getType().getElementType() != op2.getType().getElementType()) {
                return false;
            }
            for (auto user : op1.getResult().getUsers()) {
                auto op = getOp(user);
                if (op->getBlock() != op2->getBlock() || !op->isBeforeInBlock(op2)) {
                    return false;
                }
            }
            return true;
        }
    };

    struct MemoryReport : public OpRewritePattern<func::FuncOp> {
        MemoryReport(MLIRContext *ctx) : OpRewritePattern<func::FuncOp>(ctx) {}

        LogicalResult matchAndRewrite(func::FuncOp funcOp, PatternRewriter &rewriter) const override {
            auto memory0 = 0, memory1 = 0, totalMemory = 0;
            funcOp.walk([&](memref::AllocaOp op) {
                memory0 += computeMemory<memref::AllocaOp>(op);
                totalMemory += computeMemory<memref::AllocaOp>(op);
            });
            funcOp.walk([&](memref::GetGlobalOp op) {
                memory1 += computeMemory<memref::GetGlobalOp>(op);
                totalMemory += computeMemory<memref::GetGlobalOp>(op);
            });
            llvm::outs() << llvm::formatv("alloca memory: {0} bytes({1:f2} MB)\n", memory0, memory0 / 1024.0 / 1024);
            llvm::outs() << llvm::formatv("global memory: {0} bytes({1:f2} MB)\n", memory1, memory1 / 1024.0 / 1024);
            llvm::outs() << llvm::formatv("total memory: {0} bytes({1:f2} MB)\n", totalMemory, totalMemory / 1024.0 / 1024);
            return success();
        }
    };

    struct MemrefReusePass : MemrefReuseBase<MemrefReusePass> {
        void runOnOperation() override {
            auto op = getOperation().getOperation();
            GreedyRewriteConfig config;
            config.setStrictness(GreedyRewriteStrictness::ExistingOps);
            {
                RewritePatternSet Patterns(&getContext());
                Patterns.add<MemrefReuse>(&getContext());
                if (failed(applyOpPatternsAndFold(op, std::move(Patterns), config))) {
                    signalPassFailure();
                }
            }
            {
                RewritePatternSet Patterns(&getContext());
                Patterns.add<MemoryReport>(&getContext());
                if (failed(applyOpPatternsAndFold(op, std::move(Patterns), config))) {
                    signalPassFailure();
                }
            }
        }
    };
} // namespace

namespace mlir {
    std::unique_ptr<OperationPass<mlir::func::FuncOp>> createMemrefReusePass() {
        return std::make_unique<MemrefReusePass>();
    }
} // namespace mlir
