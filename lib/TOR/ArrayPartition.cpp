#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/InliningUtils.h"

#include "TOR/TORDialect.h"
#include "TOR/PassDetail.h"
#include "TOR/Passes.h"

#include <set>
#include <map>
#include <iostream>

#define DEBUG_TYPE "array-partition"

namespace {
    using namespace mlir;
    using namespace mlir::func;
    using namespace affine;
    std::map<mlir::detail::ValueImpl *, int> arg_num;

    int get_idx(Value idx) {
        auto impl = idx.getImpl();
        if (arg_num.find(impl) == arg_num.end()) {
            arg_num[impl] = arg_num.size();
        }
        return arg_num[impl];
    }

    AffineExpr get_bank_expr(Value idx, PatternRewriter &rewriter) {
        if (isa<BlockArgument>(idx)) {
            return getAffineDimExpr(get_idx(idx), rewriter.getContext());
        } else if (auto apply = dyn_cast<AffineApplyOp>(idx.getDefiningOp())) {
            auto map = apply.getAffineMap();
            auto new_map = AffineMap::get(arg_num.size(), 0, get_bank_expr(apply.getMapOperands()[0], rewriter));
            return map.getResult(0).compose(new_map);
        } else if (auto constant = dyn_cast<arith::ConstantIntOp>(idx.getDefiningOp())) {
            return getAffineConstantExpr(constant.value(), rewriter.getContext());
        } else if (auto constant = dyn_cast<arith::ConstantIndexOp>(idx.getDefiningOp())) {
            return getAffineConstantExpr(constant.value(), rewriter.getContext());
        } else if (auto index_cast = dyn_cast<arith::IndexCastOp>(idx.getDefiningOp())) {
            return get_bank_expr(index_cast.getOperand(), rewriter);
        } else if (auto add = dyn_cast<arith::AddIOp>(idx.getDefiningOp())) {
            auto left = get_bank_expr(add.getOperand(0), rewriter);
            auto right = get_bank_expr(add.getOperand(1), rewriter);
            if (!left || !right) {
                return AffineExpr();
            }
            return left + right;
        } else if (auto mul = dyn_cast<arith::MulIOp>(idx.getDefiningOp())) {
            auto left = get_bank_expr(mul.getOperand(0), rewriter);
            auto right = get_bank_expr(mul.getOperand(1), rewriter);
            if (!left || !right) {
                return AffineExpr();
            }
            return left * right;
        } else if (auto sub = dyn_cast<arith::SubIOp>(idx.getDefiningOp())) {
            auto left = get_bank_expr(sub.getOperand(0), rewriter);
            auto right = get_bank_expr(sub.getOperand(1), rewriter);
            if (!left || !right) {
                return AffineExpr();
            }
            return left - right;
        } else {
            return AffineExpr();
        }
    }

    int get_bank(AffineMap map, int rank, PatternRewriter &rewriter, int factor, bool cyclic) {
        auto expr = map.getResult(rank);
        if (!expr) {
            return -1;
        }
        if (cyclic) {
            expr = expr % factor;
        } else {
            expr = expr.floorDiv(factor);
        }
        auto compose_map = AffineMap::get(arg_num.size(), 0, expr);
        if (compose_map.isConstant()) {
            return compose_map.getConstantResults()[0];
        }
        return -1;
    }

    struct DesignOpPattern : OpRewritePattern<FuncOp> {
        DesignOpPattern(MLIRContext *ctx, int *factor, bool *cyclic, int arg_num)
                : OpRewritePattern<FuncOp>(ctx), factor(factor), cyclic(cyclic), arg_num(arg_num) {}

        LogicalResult matchAndRewrite(FuncOp op,
                                      PatternRewriter &rewriter) const override {
            if (op->hasAttr("array-partition"))
                return failure();
            op->setAttr("array-partition", IntegerAttr::get(IntegerType::get(getContext(), 32), 1));

            auto arg = op.getArgument(arg_num);
            auto memref = cast<MemRefType>(arg.getType());

            SmallVector<bool> partition;
            for (int rank = 0; rank < memref.getRank(); ++rank) {
                bool flag = true;
                if (factor[rank] > 1) {
                    for (auto &use: arg.getUses()) {
                        auto sop = use.getOwner();
                        if (auto load = dyn_cast<AffineLoadOp>(sop)) {
                            unsigned bank_factor = cyclic[rank] ? factor[rank] : memref.getShape()[rank] / factor[rank];
                            if (get_bank(load.getAffineMap(), rank, rewriter, bank_factor, cyclic[rank]) == -1) {
                                flag = false;
                                break;
                            }
                        } else if (auto store = dyn_cast<AffineStoreOp>(sop)) {
                            unsigned bank_factor = cyclic[rank] ? factor[rank] : memref.getShape()[rank] / factor[rank];
                            if (get_bank(store.getAffineMap(), rank, rewriter, bank_factor, cyclic[rank]) == -1) {
                                flag = false;
                                break;
                            }
                        } else {
                            sop->dump();
                            assert(false && "Unknown memory operation");
                        }
                    }
                } else {
                    flag = false;
                }
                partition.push_back(flag);
            }
            bool flag = false;
            for (auto p: partition) {
                if (p) {
                    flag = true;
                    break;
                }
            }
            if (flag) {
                SmallVector<Value> new_array;
                SmallVector<int64_t> new_shape;
                unsigned size = 1;
                for (int rank = 0; rank < memref.getRank(); ++rank) {
                    if (partition[rank]) {
                        new_shape.push_back(memref.getShape()[rank] / factor[rank]);
                        size *= factor[rank];
                    } else {
                        new_shape.push_back(memref.getShape()[rank]);
                    }
                }
                auto new_memref = MemRefType::get(new_shape, memref.getElementType());
                for (unsigned idx = 0; idx < size; ++idx) {
                    op.insertArgument(op.getNumArguments(), new_memref, {}, op.getLoc());
                    new_array.push_back(op.getArgument(op.getNumArguments() - 1));
                }
                struct PARTITION {
                    Operation *op;
                    unsigned bank;
                };
                SmallVector<PARTITION> new_part;
                for (auto &use: arg.getUses()) {
                    auto sop = use.getOwner();
                    if (auto load = dyn_cast<AffineLoadOp>(sop)) {
                        unsigned bank = 0;
                        for (unsigned rank = 0; rank < memref.getRank(); ++rank) {
                            if (partition[rank]) {
                                unsigned bank_factor = cyclic[rank] ? factor[rank] : memref.getShape()[rank] /
                                                                                     factor[rank];
                                bank = bank * factor[rank] +
                                       get_bank(load.getAffineMap(), rank, rewriter, bank_factor,
                                                cyclic[rank]);
                                auto map = load.getAffineMap();
                                auto expr = cyclic ? getAffineDimExpr(rank, rewriter.getContext()).floorDiv(bank_factor)
                                                   :
                                            getAffineDimExpr(rank, rewriter.getContext()) % bank_factor;
                                SmallVector<AffineExpr> exprs;
                                for (unsigned i = 0; i < memref.getRank(); ++i) {
                                    if (i != rank) {
                                        exprs.push_back(getAffineDimExpr(i, rewriter.getContext()));
                                    } else {
                                        exprs.push_back(expr);
                                    }
                                }
                                load->setAttr(load.getMapAttrStrName(), AffineMapAttr::get(
                                        AffineMap::get(map.getNumDims(), 0, exprs, getContext()).compose(map)));
                            }
                        }
                        new_part.push_back(PARTITION{load, bank});
                    } else if (auto store = dyn_cast<AffineStoreOp>(sop)) {
                        unsigned bank = 0;
                        for (unsigned rank = 0; rank < memref.getRank(); ++rank) {
                            if (partition[rank]) {
                                unsigned bank_factor = cyclic[rank] ? factor[rank] : memref.getShape()[rank] /
                                                                                     factor[rank];
                                bank = bank * factor[rank] +
                                       get_bank(store.getAffineMap(), rank, rewriter, bank_factor,
                                                cyclic[rank]);
                                auto map = store.getAffineMap();
                                auto expr = cyclic ? getAffineDimExpr(rank, rewriter.getContext()).floorDiv(bank_factor)
                                                   :
                                            getAffineDimExpr(rank, rewriter.getContext()) % bank_factor;
                                SmallVector<AffineExpr> exprs;
                                for (unsigned i = 0; i < memref.getRank(); ++i) {
                                    if (i != rank) {
                                        exprs.push_back(getAffineDimExpr(i, rewriter.getContext()));
                                    } else {
                                        exprs.push_back(expr);
                                    }
                                }
                                store->setAttr(store.getMapAttrStrName(), AffineMapAttr::get(
                                        AffineMap::get(map.getNumDims(), 0, exprs, getContext()).compose(map)));
                            }
                        }
                        new_part.push_back(PARTITION{store, bank});
                    }
                }
                for (auto part: new_part) {
                    part.op->setOperand(isa<AffineStoreOp>(part.op), new_array[part.bank]);
                }
                op.eraseArgument(arg_num);
            }
            return success();
        }

        int *factor;
        bool *cyclic;
        int arg_num;
    };

    struct ArrayPartitionPass : ArrayPartitionBase<ArrayPartitionPass> {
        void runOnOperation() override {
            auto funcOp = getOperation();
            RewritePatternSet Patterns(&getContext());
            int factor_vec[10];
            for (unsigned i = 0; i < factor.size(); ++i) {
                factor_vec[i] = factor[i];
            }
            bool cyclic_vec[10];
            for (unsigned i = 0; i < cyclic.size(); ++i) {
                cyclic_vec[i] = cyclic[i];
            }
            GreedyRewriteConfig config;
            config.setStrictness(GreedyRewriteStrictness::ExistingOps);
            Patterns.add<DesignOpPattern>(funcOp.getContext(), factor_vec, cyclic_vec, arg_num);
            if (failed(applyOpPatternsAndFold(funcOp.getOperation(), std::move(Patterns), config)))
                signalPassFailure();
            funcOp->removeAttr("array-partition");
        }
    };

} // namespace

namespace mlir {

    std::unique_ptr<OperationPass<mlir::func::FuncOp>> createArrayPartitionPass() {
        return std::make_unique<ArrayPartitionPass>();
    }

} // namespace mlir
