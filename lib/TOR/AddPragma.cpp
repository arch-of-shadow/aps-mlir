#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/Debug.h"

#include "TOR/PassDetail.h"
#include "TOR/Passes.h"

#include <set>

#define DEBUG_TYPE "add-pragma"

namespace {
    using namespace mlir;

    void setUnroll(affine::AffineForOp op, PatternRewriter &rewriter) {
        op->setAttr("unroll", rewriter.getI32IntegerAttr(0));
    }

    void setPipeline(affine::AffineForOp op, PatternRewriter &rewriter) {
        op->setAttr("II", rewriter.getI32IntegerAttr(1));
        op->setAttr("pipeline", rewriter.getI32IntegerAttr(1));
        op->setAttr("flatten", rewriter.getI32IntegerAttr(1));
    }

    void setArrayPartition(std::pair<Operation *, int> op, PatternRewriter &rewriter, Type type, std::set<int> dims, int number) {
        auto shape = dyn_cast<MemRefType>(type).getShape();
        SmallVector<Attribute> dimsAttr, factorAttr, cyclicAttr;
        for (auto dim : dims) {
            dimsAttr.push_back(rewriter.getI32IntegerAttr(dim));
            factorAttr.push_back(rewriter.getI32IntegerAttr(shape[dim]));
            cyclicAttr.push_back(rewriter.getI32IntegerAttr(0));
        }
        if (dimsAttr.size() == 0) return;
        std::string add = (op.second == -1 ? "" : "_" + std::to_string(op.second));
        op.first->setAttr("partition_dim_array" + add, rewriter.getArrayAttr(dimsAttr));
        op.first->setAttr("partition_factor_array" + add, rewriter.getArrayAttr(factorAttr));
        op.first->setAttr("partition_cyclic_array" + add, rewriter.getArrayAttr(cyclicAttr));
        op.first->setAttr("var_name" + add, rewriter.getStringAttr("var_name_" + std::to_string(number)));
    }

    struct ConvPragma : public OpRewritePattern<ModuleOp> {
        ConvPragma(MLIRContext *ctx, int conv_unroll_layers) : OpRewritePattern<ModuleOp>(ctx), conv_unroll_layers(conv_unroll_layers) {}

        LogicalResult matchAndRewrite(ModuleOp moduleOp, PatternRewriter &rewriter) const override {
            auto check = [&](affine::AffineForOp forOp) {
                if (!forOp->getAttr("description")) return false;
                auto loopAttr = dyn_cast<mlir::StringAttr>(forOp->getAttr("description")).getValue().str();
                if (loopAttr != "unroll") return false;
                auto lb = forOp.getLowerBoundMap();
                auto ub = forOp.getUpperBoundMap();
                int64_t step = forOp.getStep().getSExtValue();
                if (!lb.isSingleConstant() || !ub.isSingleConstant()) {
                    return false;
                }
                if ((ub.getSingleConstantResult() - lb.getSingleConstantResult() + step - 1) / step > 64) {
                  return false;
                }
                auto layer = dyn_cast<mlir::IntegerAttr>(forOp->getAttr("layers")).getInt();
                if (layer > conv_unroll_layers) {
                    forOp->removeAttr("description");
                    forOp->removeAttr("layers");
                    return false;
                }
                return true;
            };

            SmallVector<affine::AffineForOp> forOps, nextForOps;
            moduleOp.walk([&](affine::AffineForOp forOp) {
                if (!check(forOp)) return;
                forOps.push_back(forOp);
            });
            SmallVector<std::pair<Value, std::set<int>>> valueMap;
            auto add = [&](Value value, int index) {
                for (auto &pair : valueMap) {
                    if (pair.first == value) {
                        pair.second.insert(index);
                        return;
                    }
                }
                valueMap.push_back(std::make_pair(value, std::set<int>{index}));
            };
            auto collectNextForOps = [&](Value memref, int index) {
                for (auto user : memref.getUsers()) {
                    if (auto load = llvm::dyn_cast<affine::AffineLoadOp>(user)) {
                        auto expr = load.getAffineMap().getResult(index);
                        if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
                            auto pos = dimExpr.getPosition();
                            auto loop = findAffineForOp(load.getIndices()[pos], user->getParentOfType<affine::AffineForOp>());
                            if (loop.has_value() && !loop.value()->getAttr("description")) {
                                loop.value()->setAttr("description", rewriter.getStringAttr("unroll"));
                                nextForOps.push_back(loop.value());
                            }
                        }
                    } else if (auto store = llvm::dyn_cast<affine::AffineStoreOp>(user)) {
                        auto expr = store.getAffineMap().getResult(index);
                        if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
                            auto pos = dimExpr.getPosition();
                            auto loop = findAffineForOp(store.getIndices()[pos], user->getParentOfType<affine::AffineForOp>());
                            if (loop.has_value() && !loop.value()->getAttr("description")) {
                                loop.value()->setAttr("description", rewriter.getStringAttr("unroll"));
                                nextForOps.push_back(loop.value());
                            }
                        }
                    }
                }
            };
            // unroll
            int depth = 0;
            while (forOps.size() > 0 && ++depth <= 2) {
                LLVM_DEBUG(llvm::dbgs() << "unroll depth: " << depth << ", number: " << forOps.size() << "\n");
                for (auto forOp : forOps) {
                    for (auto user : forOp.getInductionVar().getUsers()) {
                        if (auto load = llvm::dyn_cast<affine::AffineLoadOp>(user)) {
                            auto index = findIndex(load, forOp.getInductionVar());
                            if (index == -1) continue;
                            add(load.getMemRef(), index);
                            collectNextForOps(load.getMemRef(), index);
                        } else if (auto store = llvm::dyn_cast<affine::AffineStoreOp>(user)) {
                            auto index = findIndex(store, forOp.getInductionVar());
                            if (index == -1) continue;
                            add(store.getMemRef(), index);
                            collectNextForOps(store.getMemRef(), index);
                        }
                    }
                    setUnroll(forOp, rewriter);
                }
                forOps = nextForOps;
                nextForOps.clear();
            }
            for (auto forOp : forOps) {
                setUnroll(forOp, rewriter);
            }
            // array_partition
            auto getOp = [&](Value value) {
                if (auto arg = llvm::dyn_cast<BlockArgument>(value)) {
                    return std::make_pair(arg.getOwner()->getParentOp(), (int) arg.getArgNumber());
                }
                if (auto getGlobalOp = llvm::dyn_cast<memref::GetGlobalOp>(value.getDefiningOp())) {
                    auto module = getGlobalOp->getParentOfType<ModuleOp>(); assert(module);
                    for (auto op : module.getBody()->getOps<memref::GlobalOp>()) {
                        if (op.getSymName() == getGlobalOp.getName()) {
                            return std::make_pair(op.getOperation(), -1);
                        }
                    }
                }
                return std::make_pair(value.getDefiningOp(), -1);
            };
            int number = 0;
            for (auto pair : valueMap) {
                auto op = getOp(pair.first);
                if (auto allocaOp = llvm::dyn_cast<memref::AllocaOp>(op.first)) {
                    setArrayPartition(op, rewriter, allocaOp.getType(), pair.second, number++);
                } else if (auto globalOp = llvm::dyn_cast<memref::GlobalOp>(op.first)) {
                    setArrayPartition(op, rewriter, globalOp.getType(), pair.second, number++);
                } else if (auto funcOp = llvm::dyn_cast<func::FuncOp>(op.first)) {
                    setArrayPartition(op, rewriter, funcOp.getFunctionType().getInputs()[op.second], pair.second, number++);
                }
            }
            LLVM_DEBUG(llvm::dbgs() << "array_partition number: " << number << "\n");
            // pipeline 
            moduleOp.walk([&](affine::AffineForOp forOp) {
                if (llvm::isa<func::FuncOp>(forOp->getParentOp())) {
                    SmallVector<affine::AffineForOp> forOps;
                    forOp.walk([&](affine::AffineForOp fop) {
                        forOps.push_back(fop);
                    });
                    for (auto fop : forOps) {
                        if (!fop->getAttr("unroll")) {
                            setPipeline(fop, rewriter);
                            break;
                        }
                    }
                }
            });
            return success();
        }

        template<typename T>
        int findIndex(T op, Value value) const {
            auto map = op.getAffineMap();
            auto dimPos = std::find(op.getIndices().begin(), op.getIndices().end(), value);
            auto pos = std::distance(op.getIndices().begin(), dimPos);
            for (int i = 0, r = map.getNumResults(); i < r; ++i) {
                auto expr = map.getResult(i);
                if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
                    if (dimExpr.getPosition() == pos) {
                        return i;
                    }
                }
            }
            return -1;
        }

        std::optional<affine::AffineForOp> findAffineForOp(Value value, affine::AffineForOp loop) const {
            if (!loop) return std::nullopt;
            if (loop.getInductionVar() == value) {
                return loop;
            }
            return findAffineForOp(value, loop->getParentOfType<affine::AffineForOp>());
        }

        int conv_unroll_layers;
    };

    struct AddPragmaPass : AddPragmaBase<AddPragmaPass> {
        void runOnOperation() override {
            auto op = getOperation().getOperation();
            GreedyRewriteConfig config;
            config.setStrictness(GreedyRewriteStrictness::ExistingOps);
            {
                RewritePatternSet Patterns(&getContext());
                Patterns.add<ConvPragma>(&getContext(), conv_unroll_layers);
                if (failed(applyOpPatternsAndFold(op, std::move(Patterns), config))) {
                    signalPassFailure();
                }
            }
        }
    };
} // namespace

namespace mlir {
    std::unique_ptr<OperationPass<mlir::ModuleOp>> createAddPragmaPass() {
        return std::make_unique<AddPragmaPass>();
    }
} // namespace mlir
