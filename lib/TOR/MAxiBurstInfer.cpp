#include "TOR/TOR.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Visitors.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

#include "TOR/TORDialect.h"
#include "TOR/PassDetail.h"
#include "TOR/Passes.h"

#include "TOR/SCEVAnalysis.h"

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>

using namespace mlir;
namespace
{
    // 1. 纯读或纯写
    // 2. 访问地址必须是连续且递增的，因为我们支支持突发类型为INCR
    // 3. 访问次数，即访问长度需要是可确定的，必须是常数
    // 4. 如果有两个memref绑定了同一个bus，一个的突发必须在另一个的突发完成后才能开始
    //    一次突发仅支持一个memref的一个方向（读写方向）
    // 5. 突发请求开始到结束区间都不能有依赖问题
    struct MAxiBurstInferPass : MAxiBurstInferBase<MAxiBurstInferPass>
    {
        // void runOnOperation() override;
        // check if val2 + length2 = val1 + length1 + 1
        bool isIncreVal(mlir::Value val1, mlir::Value val2, mlir::Value length1, mlir::Value length2) {
            if (val1 == nullptr || val2 == nullptr || val1 == val2) {
                return false;
            }
            auto stepOne = tor::Polynomial(1);
            auto poly1 = tor::get_poly_from_value(val1);
            auto poly2 = tor::get_poly_from_value(val2);
            if (!poly1.has_value() || !poly2.has_value()) {
                return false;
            }
            if (length1) {
                auto polyLen1 = tor::get_poly_from_value(length1);
                if (polyLen1.has_value()) {
                    poly1.value().inplace_add(polyLen1.value());
                }
            }
            if (length2) {
                auto polyLen2 = tor::get_poly_from_value(length2);
                if (polyLen2.has_value()) {
                    poly2.value().inplace_add(polyLen2.value());
                }
            }
            auto poly1AddOne = tor::Polynomial::add(poly1.value(), stepOne);
            return poly1AddOne == poly2.value();

        }

        bool isReadOp(mlir::Operation *op) {
            return llvm::isa<tor::AXIReadOp, tor::AXIReadRequestOp>(op);
        }

        // todo 现在暂不支持变化的长度，返回值为int，假如支持变化的长度，此处需要更改返回值为Value
        int getOpLength(mlir::Operation *op) {
            std::optional<tor::Polynomial> polyLen;
            if (auto readRequestOp = llvm::dyn_cast<tor::AXIReadRequestOp>(op)) {
                polyLen = tor::get_poly_from_value(readRequestOp.getLength());
            } else if (auto writeRequestOp = llvm::dyn_cast<tor::AXIWriteRequestOp>(op)) {
                polyLen = tor::get_poly_from_value(writeRequestOp.getLength());
            } else {
                return 1; 
            }

            if (!polyLen.has_value() || !polyLen.value().is_constant()) {
                // todo print warning
                return -1;
            }
            return polyLen.value().get_constant().value();
        }

        mlir::Value getOrCreateConstantIntOpValue(int32_t num) {
            auto iter = intToConstantOp.find(num);
            if (iter != intToConstantOp.end()) {
                return iter->second->getResult(0);
            }

            builder->setInsertionPointAfter(lastConstOp);
            lastConstOp = builder->create<arith::ConstantIntOp>(getOperation().getLoc(), num, 32);
            intToConstantOp[num] = lastConstOp;
            return lastConstOp.getResult();
        }

        template<typename Op, typename RequestOp>
        bool canMergeOp(mlir::Operation *op1, mlir::Operation *op2) {
            auto axiOp1 = llvm::dyn_cast<Op>(op1);
            auto axiOp2 = llvm::dyn_cast<Op>(op2);
            auto axiRequestOp1 = llvm::dyn_cast<RequestOp>(op1);
            auto axiRequestOp2 = llvm::dyn_cast<RequestOp>(op2);
            mlir::Value memref1, memref2, addr1, addr2, length1, length2;
            if (axiOp1 != nullptr) {
                memref1 = axiOp1.getMemref();
                addr1 = axiOp1.getIndices()[0];
            } else if (axiRequestOp1 != nullptr) {
                memref1 = axiRequestOp1.getMemref();
                addr1 = axiRequestOp1.getOffset();
                length1 = axiRequestOp1.getLength();
            }

            if (axiOp2 != nullptr) {
                memref2 = axiOp2.getMemref();
                addr2 = axiOp2.getIndices()[0];
            } else if (axiRequestOp2 != nullptr) {
                memref2 = axiRequestOp2.getMemref();
                addr2 = axiRequestOp2.getOffset();
                length2 = axiRequestOp2.getLength();
            }

            if (memref1 != nullptr && memref1 == memref2) {
                if (isIncreVal(addr1, addr2, length1, length2)) {
                    return true;
                }
            }
            return false;
        }

        template<typename Op, typename RequestOp>
        void mergeOps(std::vector<std::vector<std::vector<mlir::Operation*>>>& mAxiOpOfBusesToBeMerged) {
            constexpr bool isRead = std::is_same_v<Op, tor::AXIReadOp>;
            auto designOp = getOperation();
            for (size_t i = 0; i < buses.size(); ++i) {
                int maxBurstLength = isRead? buses[i].maxRLen : buses[i].maxWLen;
                for (size_t j = 0; j < mAxiOpOfBusesToBeMerged[i].size(); ++j) {
                    // todo if support manual request, cannot use vector size as opNums
                    int opNums = mAxiOpOfBusesToBeMerged[i][j].size();
                    int startIndex = 0;
                    while (opNums > 1) {
                        int currLen = opNums >= maxBurstLength ? maxBurstLength : opNums;
                        opNums -= maxBurstLength;
                        mlir::Value currLenValue = getOrCreateConstantIntOpValue(currLen);
                        auto currOp = mAxiOpOfBusesToBeMerged[i][j][startIndex];
                        builder->setInsertionPoint(currOp);
                        if constexpr (isRead) {
                            auto readOp = llvm::dyn_cast<tor::AXIReadOp>(currOp);
                            auto readRequestOp =
                                builder->create<tor::AXIReadRequestOp>(readOp.getLoc(),
                                    readOp.getMemref(), 0, 0, readOp.getIndices()[0], currLenValue);
                            for (int k = startIndex; k < startIndex + currLen; ++k) {
                                mlir::Operation *op = mAxiOpOfBusesToBeMerged[i][j][k];
                                auto mAxiReadOp = llvm::dyn_cast<tor::AXIReadOp>(op);
                                builder->setInsertionPoint(op);
                                auto burstReadOp = builder->create<tor::AXIBurstReadOp>(mAxiReadOp.getLoc(),
                                    mAxiReadOp.getMemref(), 0, 0, readRequestOp.getResult());
                                mAxiReadOp->replaceAllUsesWith(burstReadOp->getResults());
                                op->erase();
                            }
                        } else if constexpr (!isRead) {
                            auto writeOp = llvm::dyn_cast<tor::AXIWriteOp>(currOp);
                            auto writeRequestOp = builder->create<tor::AXIWriteRequestOp>(writeOp.getLoc(),
                                writeOp.getMemref(), 0, 0, writeOp.getIndices()[0], currLenValue);
                            for (int k = startIndex; k < startIndex + currLen; ++k) {
                                mlir::Operation *op = mAxiOpOfBusesToBeMerged[i][j][k];
                                auto mAxiWriteOp = llvm::dyn_cast<tor::AXIWriteOp>(op);
                                builder->setInsertionPoint(op);
                                auto burstWriteOp = builder->create<tor::AXIBurstWriteOp>(mAxiWriteOp.getLoc(),
                                    mAxiWriteOp.getOperand(0), mAxiWriteOp.getMemref(), 0, 0, writeRequestOp.getResult());
                                mAxiWriteOp->replaceAllUsesWith(burstWriteOp->getResults());
                                op->erase();
                            }
                        }
                        startIndex += currLen;
                    }
                }
            }
        }

        template <typename Op>
        std::string getBusName(mlir::Operation* op) {
            auto mAxiOp = llvm::dyn_cast<Op>(op);
            assert(mAxiOp && "Invalid op type, only mAxiOp type is allowed to getBudId");
            Value memref = mAxiOp.getMemref();
            auto mAxiCreateOp = llvm::dyn_cast<tor::AXICreateOp>(memref.getDefiningOp());
            std::string bus = "";
            if (mAxiCreateOp->hasAttr("bus")) {
                bus = dyn_cast<mlir::StringAttr>(mAxiCreateOp->getAttr("bus")).getValue().str();
            }
            return bus;
        }

        template <typename Op>
        size_t getBusId(mlir::Operation* op) {
            auto mAxiOp = llvm::dyn_cast<Op>(op);
            assert(mAxiOp && "Invalid op type, only mAxiOp type is allowed to getBudId");
            mlir::Operation* axiCreateOp = mAxiOp.getMemref().getDefiningOp();
            return maxiToBusMap[axiCreateOp];
        }

        template <typename Op>
        mlir::Value getMemref(mlir::Operation* op) {
            auto mAxiOp = llvm::dyn_cast<Op>(op);
            assert(mAxiOp && "Invalid op type, only mAxiOp type is allowed to getBudId");
            return mAxiOp.getMemref();
        }

        // same memref access in different direction(read/write) means a conflict
        template<typename Op, typename CandidateOp, typename CandidateRequestOp>
        bool isOpSameMemref(mlir::Operation* candidateOp, mlir::Operation* op) {
            if (candidateOp == nullptr)
                return false;
            auto mAxiOp = llvm::dyn_cast<Op>(op);
            auto busId = getBusId<Op>(op);
            if (auto targetOp = llvm::dyn_cast<CandidateOp>(candidateOp)) {
                return mAxiOp.getMemref() == targetOp.getMemref();
            } else if (auto targetOp = llvm::dyn_cast<CandidateRequestOp>(candidateOp)) {
                return mAxiOp.getMemref() == targetOp.getMemref();
            }
            return false;
        }

        template <typename Op, typename RequestOp, typename CandidateOp, typename CandidateRequestOp>
        void axiOpBurstInfer(
            mlir::Operation* op, std::vector<mlir::Operation*>& currMAxiOpOfBuses,
            std::vector<mlir::Operation*>& currMAxiCandidateOpOfBuses,
            std::vector<std::vector<std::vector<mlir::Operation*>>>&  mergedMAxiOpOfBuses,
            std::vector<std::vector<std::vector<mlir::Operation*>>>&  mergedCandidateMAxiOpOfBuses)
        {
            auto busId = getBusId<Op>(op);
            bool isConflict = isOpSameMemref<Op, CandidateOp, CandidateRequestOp>(
                currMAxiCandidateOpOfBuses[busId], op);

            if (isConflict) {
                mergedCandidateMAxiOpOfBuses[busId].back().push_back(currMAxiCandidateOpOfBuses[busId]);
                currMAxiCandidateOpOfBuses[busId] = nullptr;
                mergedCandidateMAxiOpOfBuses[busId].push_back({});
            }

            if (currMAxiOpOfBuses[busId] == nullptr) {
                mergedMAxiOpOfBuses[busId].push_back({});
                currMAxiOpOfBuses[busId] = op;
            } else if (!isConflict && canMergeOp<Op, RequestOp>(currMAxiOpOfBuses[busId], op)) {
                mergedMAxiOpOfBuses[busId].back().push_back(currMAxiOpOfBuses[busId]);
                currMAxiOpOfBuses[busId] = op;
            } else {
                mergedMAxiOpOfBuses[busId].back().push_back(currMAxiOpOfBuses[busId]);
                currMAxiOpOfBuses[busId] = op;
                mergedMAxiOpOfBuses[busId].push_back({});
            }
        }

        void flushCurrOpToMerged(std::vector<mlir::Operation*>& currMAxiOpOfBuses,
            std::vector<std::vector<std::vector<mlir::Operation*>>>&  mergedMAxiOpOfBuses)
        {
            for (size_t i = 0; i < buses.size(); ++i) {
                if (currMAxiOpOfBuses[i] != nullptr) {
                    mergedMAxiOpOfBuses[i].back().push_back(currMAxiOpOfBuses[i]);
                    currMAxiOpOfBuses[i] = nullptr;
                }
            }
        }

        void blockBurstInfer(Block &block) {
            std::vector<mlir::Operation*> currReadMAxiOpOfBuses(buses.size(), nullptr);
            std::vector<mlir::Operation*> currWriteMAxiOpOfBuses(buses.size(), nullptr);
            std::vector<std::vector<std::vector<mlir::Operation*>>> mergedReadMAxiOpOfBuses(buses.size());
            std::vector<std::vector<std::vector<mlir::Operation*>>> mergedWriteMAxiOpOfBuses(buses.size());
            for (auto &op: block) {
                if (auto mAxiReadOp = llvm::dyn_cast<tor::AXIReadOp>(op)) {
                    axiOpBurstInfer<tor::AXIReadOp, tor::AXIReadRequestOp, tor::AXIWriteOp, tor::AXIWriteRequestOp>(
                        &op, currReadMAxiOpOfBuses, currWriteMAxiOpOfBuses, mergedReadMAxiOpOfBuses, mergedWriteMAxiOpOfBuses);
                } else if (auto mAxiWriteOp = llvm::dyn_cast<tor::AXIWriteOp>(op)) {
                    axiOpBurstInfer<tor::AXIWriteOp, tor::AXIWriteRequestOp, tor::AXIReadOp, tor::AXIReadRequestOp>(
                        &op, currWriteMAxiOpOfBuses, currReadMAxiOpOfBuses, mergedWriteMAxiOpOfBuses, mergedReadMAxiOpOfBuses);
                } else if (llvm::isa<tor::IfOp, tor::WhileOp, tor::ForOp, tor::CallOp>(op)) {
                    if (auto ifOp = llvm::dyn_cast<tor::IfOp>(op)) {
                        blockBurstInfer(ifOp.getThenRegion().front());
                        if (!ifOp.getElseRegion().empty()) {
                            blockBurstInfer(ifOp.getElseRegion().front());
                        }
                    } else if (auto whileOp = llvm::dyn_cast<tor::WhileOp>(op)) {
                        blockBurstInfer(whileOp.getAfter().front()); // body
                        // // condBB, there shall not be any axi related op in condition bb, I guess
                        // blockBurstInfer(whileOp.getBefore().front());
                    } else if (auto forOp = llvm::dyn_cast<tor::ForOp>(op)) {
                        blockBurstInfer(*forOp.getBody());
                    }
                    // encounter op that has blocks, need to merge
                    flushCurrOpToMerged(currReadMAxiOpOfBuses, mergedReadMAxiOpOfBuses);
                    flushCurrOpToMerged(currWriteMAxiOpOfBuses, mergedWriteMAxiOpOfBuses);
                }
            }

            flushCurrOpToMerged(currReadMAxiOpOfBuses, mergedReadMAxiOpOfBuses);
            flushCurrOpToMerged(currWriteMAxiOpOfBuses, mergedWriteMAxiOpOfBuses);
            mergeOps<tor::AXIReadOp, tor::AXIReadRequestOp>(mergedReadMAxiOpOfBuses);
            mergeOps<tor::AXIWriteOp, tor::AXIWriteRequestOp>(mergedWriteMAxiOpOfBuses);
        }

        void getForOps(Block &block, std::vector<std::vector<tor::ForOp>>& forOps) {
            size_t currLayer = forOps.size();
            forOps.push_back({});
            for (auto &op: block) {
                if (auto forOp = llvm::dyn_cast<tor::ForOp>(op)) {
                    forOps[currLayer].push_back(forOp);
                    getForOps(*forOp.getBody(), forOps);
                } else if (auto ifOp = llvm::dyn_cast<tor::IfOp>(op)) {
                    // todo burst inside branch? what if conflict with other branch
                    if (ifOp.getElseRegion().empty())
                        getForOps(ifOp.getThenRegion().front(), forOps);
                    // if (!ifOp.getElseRegion().empty())
                    //     getForOps(ifOp.getElseRegion().front(), forOps);
                } else if (auto whileOp = llvm::dyn_cast<tor::WhileOp>(op)) {
                    // burst for op inside while?
                    getForOps(whileOp.getBefore().front(), forOps);
                    getForOps(whileOp.getAfter().front(), forOps);
                }
            }
        }

        void resetLengthAndClearVector(size_t busId, std::vector<int>& burstLength,
            std::vector<std::vector<mlir::Operation*>>& mAxiOpOfBuses)
        {
            burstLength[busId] = -1;
            mAxiOpOfBuses[busId].clear();
        }

        template <typename Op, typename CandidateOp, typename CandidateRequestOp>
        bool isMemrefConflict(mlir::Operation *op,
            std::vector<std::vector<mlir::Operation*>>& mAxiCandidateOpOfBuses)
        {
            size_t busId = getBusId<Op>(op);
            return !mAxiCandidateOpOfBuses[busId].empty() &&
                isOpSameMemref<Op, CandidateOp, CandidateRequestOp>(
                    mAxiCandidateOpOfBuses[busId].back(), op);
        }

        template <typename Op, typename RequestOp, typename CandidateOp, typename CandidateRequestOp>
        void axiOpBurstInfer(std::vector<std::vector<mlir::Operation*>>& mAxiOpOfBuses,
            std::vector<std::vector<mlir::Operation*>>& mAxiCandidateOpOfBuses,
            std::vector<int>& burstLength, std::vector<int>& candidateBurstLength,
            std::unordered_set<mlir::Operation*>& accessedMAxiOps,
            mlir::Operation *op, bool isRequest)
        {
            auto memref = isRequest? getMemref<RequestOp>(op) : getMemref<Op>(op);
            auto createOp = memref.getDefiningOp();
            if (accessedMAxiOps.count(createOp) > 0) {
                return;
            }
            size_t busId = isRequest? getBusId<RequestOp>(op) : getBusId<Op>(op);
            if (!mAxiCandidateOpOfBuses[busId].empty() &&
                ((isRequest && isOpSameMemref<RequestOp, CandidateOp, CandidateRequestOp>(mAxiCandidateOpOfBuses[busId].back(), op))
                  || (!isRequest && isOpSameMemref<Op, CandidateOp, CandidateRequestOp>(mAxiCandidateOpOfBuses[busId].back(), op)))) {
                    // cannot burst for both direction
                    resetLengthAndClearVector(busId, burstLength, mAxiOpOfBuses);
                    resetLengthAndClearVector(busId, candidateBurstLength, mAxiCandidateOpOfBuses);
                    return;
            }

            if (burstLength[busId] < 0) {
                accessedMAxiOps.insert(createOp);
            } else if (mAxiOpOfBuses[busId].size() == 0) {
                burstLength[busId] = getOpLength(op);
                if (burstLength[busId] > 0) {
                    mAxiOpOfBuses[busId].push_back(op);
                }
            } else if (canMergeOp<Op, RequestOp>(mAxiOpOfBuses[busId].back(), op)) {
                auto length = getOpLength(op);
                if (length < 0) {
                    resetLengthAndClearVector(busId, burstLength, mAxiOpOfBuses);
                    accessedMAxiOps.insert(createOp);
                } else {
                    mAxiOpOfBuses[busId].push_back(op);
                    burstLength[busId] += length;
                }
            } else {
                resetLengthAndClearVector(busId, burstLength, mAxiOpOfBuses);
                accessedMAxiOps.insert(createOp);
            }

            if constexpr (std::is_same_v<Op, tor::AXIReadOp>) {
                if (burstLength[busId] > buses[busId].maxRLen) {
                    resetLengthAndClearVector(busId, burstLength, mAxiOpOfBuses);
                    accessedMAxiOps.insert(createOp);
                }
            } else if constexpr (std::is_same_v<Op, tor::AXIWriteOp>) {
                if (burstLength[busId] > buses[busId].maxWLen) {
                    resetLengthAndClearVector(busId, burstLength, mAxiOpOfBuses);
                    accessedMAxiOps.insert(createOp);
                }
            }
        }

        std::pair<mlir::Value, mlir::Value> getMemrefAddr(mlir::Operation *op) {
            mlir::Value memref, addr;
            if (auto readOp = llvm::dyn_cast<tor::AXIReadOp>(op)) {
                memref = readOp.getMemref();
                addr = readOp.getIndices()[0];
            } else if (auto writeOp = llvm::dyn_cast<tor::AXIWriteOp>(op)) {
                memref = writeOp.getMemref();
                addr = writeOp.getIndices()[0];
            } else if (auto readRequestOp = llvm::dyn_cast<tor::AXIReadRequestOp>(op)) {
                memref = readRequestOp.getMemref();
                addr = readRequestOp.getOffset();
            } else if (auto writeRequestOp = llvm::dyn_cast<tor::AXIWriteRequestOp>(op)) {
                memref = writeRequestOp.getMemref();
                addr = writeRequestOp.getOffset();
            }
            return {memref, addr};
        }

        template <typename Op, typename RequestOp, typename BurstOp>
        RequestOp createBurstAndEraseOldOps(int length, std::vector<mlir::Operation*>& mAxiOpOfBus) {
            auto [memref, startAddr] = getMemrefAddr(mAxiOpOfBus[0]);
            // create length constantInt op;
            mlir::Value lengthValue = getOrCreateConstantIntOpValue(length);
            builder->setInsertionPoint(mAxiOpOfBus[0]);
            auto requestOp =
                builder->create<RequestOp>(
                    mAxiOpOfBus[0]->getLoc(), memref, 0, 0, startAddr, lengthValue);
            mlir::Operation* lastOpInBlock = requestOp; // use for write response insert point
            for (int i = 0; i < length; ++i) {
                mlir::Operation *op = mAxiOpOfBus[i];
                if (auto mAxiOp = llvm::dyn_cast<Op>(op)) {
                    builder->setInsertionPoint(op);
                    if constexpr (std::is_same_v<BurstOp, tor::AXIBurstReadOp>) {
                        auto burstOp = builder->create<BurstOp>(mAxiOp.getLoc(),
                            mAxiOp.getMemref(), 0, 0, requestOp.getResult());
                        mAxiOp->replaceAllUsesWith(burstOp->getResults());
                    } else if constexpr (std::is_same_v<BurstOp, tor::AXIBurstWriteOp>) {
                        auto burstOp = builder->create<BurstOp>(mAxiOp.getLoc(),
                            mAxiOp.getOperand(0), mAxiOp.getMemref(), 0, 0, requestOp.getResult());
                        mAxiOp->replaceAllUsesWith(burstOp->getResults());
                        if (lastOpInBlock->isBeforeInBlock(burstOp))
                            lastOpInBlock = burstOp;
                    }
                } else if (llvm::isa<RequestOp>(op)) {
                    if constexpr (std::is_same_v<BurstOp, tor::AXIBurstWriteOp>) {
                        for (auto user: mAxiOp->getUsers()) {
                            if (llvm::dyn_cast<tor::AXIWriteResponseOp>(user)) {
                                user->erase();
                            } else if (lastOpInBlock->isBeforeInBlock(user)) {
                                lastOpInBlock = user;
                            }
                        }
                    }
                    op->replaceAllUsesWith(requestOp->getResults());
                }
                op->erase();
            }
            if constexpr (std::is_same_v<BurstOp, tor::AXIBurstWriteOp>) {
                builder->setInsertionPointAfter(lastOpInBlock);
                auto reponseOp = builder->create<tor::AXIWriteResponseOp>(
                    requestOp->getLoc(), memref, 0, 0, requestOp.getResult());
            }
            return requestOp;
        }

        template <typename Op>
        std::vector<mlir::Operation*> mergeOps(std::vector<std::vector<mlir::Operation*>>& mAxiOpOfBuses,
            const std::vector<int>& burstLength)
        {
            std::vector<mlir::Operation*> mergedOpOfBuses(buses.size()); // final op
            for (size_t i = 0; i < buses.size(); ++i) {
                if (mAxiOpOfBuses[i].size() == 0) {
                    continue;
                } else if (mAxiOpOfBuses[i].size() == 1) {
                    mergedOpOfBuses[i] = mAxiOpOfBuses[i][0];
                } else {
                    auto isRead = isReadOp(mAxiOpOfBuses[i][0]);
                    if constexpr (std::is_same_v<Op, tor::AXIReadOp>) {
                        assert (burstLength[i] <= buses[i].maxRLen && "Read burst length exceed max read burst length");
                        mergedOpOfBuses[i] =
                            createBurstAndEraseOldOps<tor::AXIReadOp, tor::AXIReadRequestOp, tor::AXIBurstReadOp>(
                                burstLength[i], mAxiOpOfBuses[i]);
                    }
                    if constexpr (std::is_same_v<Op, tor::AXIWriteOp>) {
                        assert (burstLength[i] <= buses[i].maxWLen && "Write burst length exceed max write burst length");
                        mergedOpOfBuses[i] =
                            createBurstAndEraseOldOps<tor::AXIWriteOp, tor::AXIWriteRequestOp, tor::AXIBurstWriteOp>(
                                burstLength[i], mAxiOpOfBuses[i]);
                    }
                }
            }
            return mergedOpOfBuses;
        }

        int getNumOfIteration(tor::ForOp forOp) {
            auto lb = tor::get_poly_from_value(forOp.getLowerBound());
            auto ub = tor::get_poly_from_value(forOp.getUpperBound());
            auto step = tor::get_poly_from_value(forOp.getStep());
            if (lb.has_value() && step.has_value() && ub.has_value()) {
                auto lbConst = lb.value().get_constant();
                auto ubConst = ub.value().get_constant();
                auto stepConst = step.value().get_constant();
                if (lbConst.has_value() && ubConst.has_value() && stepConst.has_value()) {
                    return llvm::divideCeil(ubConst.value() - lbConst.value() + 1, stepConst.value());
                }
            }
            return -1;
        }

        mlir::Value getNewStartAddressForBurstFromPoly(tor::Polynomial& poly, tor::ForOp forOp) {
            mlir::Value startAddr = nullptr;
            poly.canonicalize();
            builder->setInsertionPoint(forOp);
            // poly = monomial[0].first * monomial[0].second + monomial[1].first * monomial[1].second ....
            for (auto monomial: poly.monomials) {
                mlir::Value monomialValue = nullptr;
                if (monomial.second == 0) {
                    continue;
                }
                for (auto symbol: monomial.first.symbols) {
                    if (monomialValue == nullptr) {
                        monomialValue = symbol;
                    } else {
                        builder->setInsertionPoint(forOp);
                        auto muliOp = builder->create<tor::MulIOp>(forOp.getLoc(), monomialValue, symbol);
                        monomialValue = muliOp.getResult();
                    }
                }
                mlir::Value indexValue = getOrCreateConstantIntOpValue(monomial.second);
                if (monomial.second > 1) {
                    builder->setInsertionPoint(forOp);
                    mlir::Value result = indexValue;
                    if (monomialValue != nullptr) {
                        auto muliOp = builder->create<tor::MulIOp>(forOp.getLoc(), monomialValue, indexValue);
                        result = muliOp.getResult();
                    }
                    if (startAddr == nullptr) {
                        startAddr = result;
                    } else {
                        builder->setInsertionPoint(forOp);
                        auto addiOp = builder->create<tor::AddIOp>(forOp.getLoc(), startAddr, result);
                        startAddr = addiOp.getResult();
                    }
                } else if (monomial.second == 1) {
                    if (startAddr == nullptr) {
                        startAddr = monomialValue == nullptr? indexValue : monomialValue;
                    } else {
                        builder->setInsertionPoint(forOp);
                        auto addiOp = builder->create<tor::AddIOp>(forOp.getLoc(), startAddr,
                            monomialValue == nullptr? indexValue : monomialValue);
                        startAddr = addiOp.getResult();
                    }
                }
                // todo what if monomial.second is negative
            }
            if (startAddr == nullptr) {
                startAddr = getOrCreateConstantIntOpValue(0);
            }
            return startAddr;
        }

        template <typename Op, typename RequestOp, typename BurstOp>
        RequestOp createForOpBurstAndEraseOldOp(int busId, int length, tor::Polynomial& poly, tor::ForOp forOp, mlir::Operation* oldOp) {
            auto [memref, startAddr] = getMemrefAddr(oldOp);
            // create length constantInt op
            mlir::Value lengthValue = getOrCreateConstantIntOpValue(length);
            // gen newaddr value
            startAddr = getNewStartAddressForBurstFromPoly(poly, forOp);
            builder->setInsertionPoint(forOp);
            auto requestOp =
                builder->create<RequestOp>(
                    forOp->getLoc(), memref, 0, 0, startAddr, lengthValue);
            if (auto mAxiOp = llvm::dyn_cast<Op>(oldOp)) {
                builder->setInsertionPoint(oldOp);
                if constexpr (std::is_same_v<BurstOp, tor::AXIBurstReadOp>) {
                    auto burstOp = builder->create<BurstOp>(mAxiOp.getLoc(),
                        mAxiOp.getMemref(), 0, 0, requestOp.getResult());
                    oldOp->replaceAllUsesWith(burstOp->getResults());
                } else if constexpr (std::is_same_v<BurstOp, tor::AXIBurstWriteOp>) {
                    auto burstOp = builder->create<BurstOp>(mAxiOp.getLoc(),
                        mAxiOp.getOperand(0), mAxiOp.getMemref(), 0, 0, requestOp.getResult());
                    oldOp->replaceAllUsesWith(burstOp->getResults());
                }
            } else if (llvm::isa<RequestOp>(oldOp)) {
                if constexpr (std::is_same_v<BurstOp, tor::AXIBurstWriteOp>) {
                    for (auto user: oldOp->getUsers()) {
                        if (llvm::dyn_cast<tor::AXIWriteResponseOp>(user))
                            user->erase();
                    }
                }
                oldOp->replaceAllUsesWith(requestOp->getResults());
            }
            oldOp->erase();
            constexpr bool isRead = std::is_same_v<BurstOp, tor::AXIBurstReadOp>;
            if constexpr (!isRead) {
                builder->setInsertionPointAfter(forOp);
                auto reponseOp = builder->create<tor::AXIWriteResponseOp>(
                    requestOp->getLoc(), memref, 0, 0, requestOp.getResult());
            }
            // erase forop from memoryAccessMap
            auto& memory = buses[busId].getMemory(memref.getDefiningOp());
            if constexpr (isRead) {
                memory.readAccessedByMap.erase(forOp);
            } else if constexpr (!isRead) {
                memory.writeAccessedByMap.erase(forOp);
            }
            return requestOp;
        }

        template <typename Op, typename RequestOp, typename BurstOp>
        void forLoopIntraIterBurstInfer(int numOfIteration, tor::ForOp forOp,
            mlir::tor::SCEVAnalysis& scevAnalysis, std::vector<mlir::Operation*>& mergedOps)
        {
            constexpr bool isRead = std::is_same_v<Op, tor::AXIReadOp>;
            for (size_t i = 0; i < buses.size(); ++i) {
                if (mergedOps[i]) {
                    // intra-iteration 
                    int maxBurstLength = isRead ? buses[i].maxRLen : buses[i].maxWLen;
                    int length = getOpLength(mergedOps[i]);
                    int newBurstLength = length * numOfIteration;
                    if (newBurstLength > maxBurstLength) {
                        continue;
                    }
                    auto [memref, addr] = getMemrefAddr(mergedOps[i]);
                    auto addrScevChain = scevAnalysis.get_scev_chain(addr);
                    if (addrScevChain->op == tor::SCEVChain::OP_ADD && !addrScevChain->hasDiv()
                        && addrScevChain->rest != nullptr
                        && addrScevChain->rest.get()->coeff.is_constant()
                        && addrScevChain->rest.get()->rest == nullptr) {
                        auto addValue = addrScevChain->rest.get()->coeff.get_constant();
                        if (addValue.value() == length) {
                            // hoist burst out
                            createForOpBurstAndEraseOldOp<Op, RequestOp, BurstOp>(
                                i, newBurstLength, addrScevChain->coeff, forOp, mergedOps[i]);
                        }
                    }
                }
            }
        }

        void forLoopIntraIterBurstInfer(tor::ForOp forOp, std::vector<mlir::Operation*>& mergedReadOps,
            std::vector<mlir::Operation*>& mergedWriteOps) {
            int numOfIteration = getNumOfIteration(forOp);
            if (numOfIteration < 0) {
                // llvm::outs() << "for loop number of iteration can not be determinied \n";
                return;
            }

            auto scevAnalysis = tor::SCEVAnalysis(forOp);
            forLoopIntraIterBurstInfer<tor::AXIReadOp, tor::AXIReadRequestOp, tor::AXIBurstReadOp>(
                numOfIteration, forOp, scevAnalysis, mergedReadOps);
            forLoopIntraIterBurstInfer<tor::AXIWriteOp, tor::AXIWriteRequestOp, tor::AXIBurstWriteOp>(
                numOfIteration, forOp, scevAnalysis, mergedWriteOps);
        }

        void checkMemrefCollision(std::vector<std::vector<mlir::Operation*>>& mAxiOpOfBuses,
            std::vector<int>& burstLength, std::unordered_set<mlir::Operation*>& accessedMAxiOps)
        {
            for (size_t i = 0; i < buses.size(); ++i) {
                if (burstLength[i] <= 0)
                    continue;
                auto [memref, _] = getMemrefAddr(mAxiOpOfBuses[i][0]);
                auto mAxiCreateOp = memref.getDefiningOp();
                if (accessedMAxiOps.find(mAxiCreateOp) != accessedMAxiOps.end()) {
                    resetLengthAndClearVector(i, burstLength, mAxiOpOfBuses);
                }
            }
        }

        // burst infer of forLoop
        void forLoopBurstInfer(tor::ForOp forOp) {
            std::unordered_set<mlir::Operation*> accessedMAxiOps;
            std::vector<std::vector<mlir::Operation*>> readMAxiOpOfBuses(buses.size());
            std::vector<std::vector<mlir::Operation*>> writeMAxiOpOfBuses(buses.size());
            std::vector<int> readBurstLength(buses.size(), 0);
            std::vector<int> writeBurstLength(buses.size(), 0);
            // inter-iteration 
            for (auto &op: *(forOp.getBody())) {
                if (auto readOp = llvm::dyn_cast<tor::AXIReadOp>(op)) {
                    axiOpBurstInfer<tor::AXIReadOp, tor::AXIReadRequestOp, tor::AXIWriteOp, tor::AXIWriteRequestOp>(
                        readMAxiOpOfBuses, writeMAxiOpOfBuses, readBurstLength, writeBurstLength, accessedMAxiOps, &op, false);
                } else if (auto writeOp = llvm::dyn_cast<tor::AXIWriteOp>(op)) {
                    axiOpBurstInfer<tor::AXIWriteOp, tor::AXIWriteRequestOp, tor::AXIReadOp, tor::AXIReadRequestOp>(
                        writeMAxiOpOfBuses, readMAxiOpOfBuses, writeBurstLength, readBurstLength, accessedMAxiOps, &op, false);
                } else if (auto readRequestOp = llvm::dyn_cast<tor::AXIReadRequestOp>(op)) {
                    axiOpBurstInfer<tor::AXIReadOp, tor::AXIReadRequestOp, tor::AXIWriteOp, tor::AXIWriteRequestOp>(
                        readMAxiOpOfBuses, writeMAxiOpOfBuses, readBurstLength, writeBurstLength, accessedMAxiOps, &op, true);
                } else if (auto writeRequestOp = llvm::dyn_cast<tor::AXIWriteRequestOp>(op)) {
                    axiOpBurstInfer<tor::AXIWriteOp, tor::AXIWriteRequestOp, tor::AXIReadOp, tor::AXIReadRequestOp>(
                        writeMAxiOpOfBuses, readMAxiOpOfBuses, writeBurstLength, readBurstLength, accessedMAxiOps, &op, true);
                } else if (llvm::isa<tor::IfOp, tor::WhileOp, tor::ForOp, tor::CallOp>(op)) {
                    for (size_t i = 0; i < buses.size(); ++i) {
                        if (readBurstLength[i] >= 0 && buses[i].hasReadAccess(&op)) {
                            resetLengthAndClearVector(i, readBurstLength, readMAxiOpOfBuses);
                        }
                        if (writeBurstLength[i] >= 0 && buses[i].hasWriteAccess(&op)) {
                            resetLengthAndClearVector(i, writeBurstLength, writeMAxiOpOfBuses);
                        }
                        auto readAxiAccessed = buses[i].getMAxiReadAccessBy(&op);
                        accessedMAxiOps.insert(readAxiAccessed.begin(), readAxiAccessed.end());
                        auto writeAxiAccessed = buses[i].getMAxiWriteAccessBy(&op);
                        accessedMAxiOps.insert(writeAxiAccessed.begin(), writeAxiAccessed.end());
                    }
                }
            }
            checkMemrefCollision(readMAxiOpOfBuses, readBurstLength, accessedMAxiOps);
            checkMemrefCollision(writeMAxiOpOfBuses, writeBurstLength, accessedMAxiOps);
            auto mergedReadOps = mergeOps<tor::AXIReadOp>(readMAxiOpOfBuses, readBurstLength);
            auto mergedWriteOps = mergeOps<tor::AXIWriteOp>(writeMAxiOpOfBuses, writeBurstLength);
            // inter iteration burst infer of forLoop
            forLoopIntraIterBurstInfer(forOp, mergedReadOps, mergedWriteOps);            
        }

        void forOpInfer(const std::vector<std::vector<tor::ForOp>>& forOps) {
            for (auto riter = forOps.rbegin(); riter != forOps.rend(); ++riter) {
                for (auto forOp: *riter) {
                    forLoopBurstInfer(forOp);
                }
            }
        }

        void forOpInterInterationBurstInfer(tor::FuncOp funcOp) {
            // get forloopOp layer by layer handle the innermost layer first
            std::vector<std::vector<tor::ForOp>> forOps;
            getForOps(funcOp.getBody().front(), forOps);
            // forOp inter-iteration burst infer
            forOpInfer(forOps);
        }

        void mAxiBurstInfer(tor::FuncOp funcOp) {
            // burst infer each block in dfs manner
            blockBurstInfer(funcOp.getBody().front());

            // forOp inter-iteration burst infer
            forOpInterInterationBurstInfer(funcOp);

            // todo maybe final burst infer in dfs manner
        }

        struct Memory {
            mlir::Operation* op;
            // set of operation that have accessed to this m_axi memref
            // via AXIWriteOp, AXIReadOp, AXIReadRequestOp, AXIWriteRequestOp.
            // Note: if a forLoop has only AXIBurstReadOp/AXIBurstWriteOp inside loop,
            // will not considered accessed to this m_axi since this map is
            // used to help analyze burst infer.
            std::unordered_set<mlir::Operation*> writeAccessedByMap;
            std::unordered_set<mlir::Operation*> readAccessedByMap;
            Memory(mlir::Operation* op) : op(op) {
                writeAccessedByMap.clear();
                readAccessedByMap.clear();
            }
        };

        struct MAxiBus {
            std::string busName;
            int maxRLen;
            int maxWLen;
            // key: AXICreateOp, value: index in memories
            std::unordered_map<mlir::Operation*, size_t> memoryMap;
            std::vector<Memory> memories;
            MAxiBus(const std::string& bus, int maxRLen = INT_MAX, int maxWLen = INT_MAX)
                : busName(bus), maxRLen(maxRLen), maxWLen(maxWLen)
            {
                memoryMap.clear();
            }
            
            void addMemory(mlir::Operation* op) {
                assert(memoryMap.find(op) == memoryMap.end());
                memoryMap[op] = memories.size();
                memories.push_back(Memory(op));
            }

            // if op has access to bus via axiCreateOp
            bool hasReadAccess(mlir::Operation* op, mlir::Operation* axiCreateOp) {
                auto iter = memoryMap.find(axiCreateOp);
                if (iter != memoryMap.end()) {
                    Memory& memory = memories[iter->second];
                    if (memory.readAccessedByMap.find(op) != memory.readAccessedByMap.end()) {
                        return true;
                    }
                }
                return false;
            }

            bool hasWriteAccess(mlir::Operation* op, mlir::Operation* axiCreateOp) {
                auto iter = memoryMap.find(axiCreateOp);
                if (iter != memoryMap.end()) {
                    Memory& memory = memories[iter->second];
                    if (memory.writeAccessedByMap.find(op) != memory.writeAccessedByMap.end()) {
                        return true;
                    }
                }
                return false;
            }

            // if op has access to bus
            bool hasReadAccess(mlir::Operation* op) {
                for (auto iter: memoryMap) {
                    Memory& memory = memories[iter.second];
                    if (memory.readAccessedByMap.find(op) != memory.readAccessedByMap.end()) {
                        return true;
                    }
                }
                return false;
            }

            bool hasWriteAccess(mlir::Operation* op) {
                for (auto iter: memoryMap) {
                    Memory& memory = memories[iter.second];
                    if (memory.writeAccessedByMap.find(op) != memory.writeAccessedByMap.end()) {
                        return true;
                    }
                }
                return false;
            }

            std::vector<mlir::Operation*> getMAxiReadAccessBy(mlir::Operation* op) {
                std::vector<mlir::Operation*> ret;
                for (auto iter: memoryMap) {
                    Memory& memory = memories[iter.second];
                    if (memory.readAccessedByMap.find(op) != memory.readAccessedByMap.end()) {
                        ret.push_back(iter.first);
                    }
                }
                return ret;
            }

            std::vector<mlir::Operation*> getMAxiWriteAccessBy(mlir::Operation* op) {
                std::vector<mlir::Operation*> ret;
                for (auto iter: memoryMap) {
                    Memory& memory = memories[iter.second];
                    if (memory.writeAccessedByMap.find(op) != memory.writeAccessedByMap.end()) {
                        ret.push_back(iter.first);
                    }
                }
                return ret;
            }

            Memory& getMemory(mlir::Operation* op) {
                auto iter = memoryMap.find(op);
                assert(iter != memoryMap.end());
                return memories[iter->second];
            }
        };

        void getAllMAxiDependentOp(Memory& memory) {
            mlir::Operation* mAxiOp = memory.op;
            auto designOp = getOperation();
            for (auto *op: mAxiOp->getUsers()) {
                // accessedByMap.insert(op);
                bool isRead = llvm::isa<tor::AXIReadOp, tor::AXIWriteRequestOp, tor::AXIBurstReadOp>(op);
                auto &accessedByMap = isRead? memory.readAccessedByMap : memory.writeAccessedByMap;
                auto currOp = op;
                while (currOp != nullptr && !llvm::isa<tor::DesignOp>(currOp)) {
                    accessedByMap.insert(currOp);
                    currOp = currOp->getParentOp();
                    if (llvm::isa<tor::FuncOp>(currOp)
                        && funcOpCallOpMap.find(currOp) != funcOpCallOpMap.end()) {
                        currOp = funcOpCallOpMap.at(currOp);
                    }
                }
            }
        }

        void getFuncOpCallOpMap(tor::FuncOp funcOp) {
            auto designOp = getOperation();
            funcOp.walk([&](tor::CallOp callOp){
                auto childFuncOp = designOp.lookupSymbol<tor::FuncOp>(callOp.getCallee().str());
                funcOpCallOpMap[childFuncOp] = callOp;
                getFuncOpCallOpMap(childFuncOp);
            });
        }

        void getFuncOpCallOpMapEntry(const std::string& funcName) {
            auto designOp = getOperation();
            auto funcOp = designOp.lookupSymbol<tor::FuncOp>(funcName);
            getFuncOpCallOpMap(funcOp);
        }

        void runOnOperation() override {
            ctx = &getContext();
            auto opBuilder = OpBuilder(ctx);
            builder = &opBuilder;
            auto designOp = getOperation();

            // gather info for later burst infer
            getFuncOpCallOpMapEntry("main");
            // temporay map to make sure bus only created once
            // key: bus str, value: bus id in "buses" vector
            std::unordered_map<std::string, size_t> busMap;
            for (auto &op: designOp.getBody().front()) {
                if (auto constOp = llvm::dyn_cast<arith::ConstantOp>(op)) {
                    lastConstOp = constOp;
                    if (auto constIndexOp = llvm::dyn_cast<arith::ConstantIndexOp>(op))
                        intToConstantOp[constIndexOp.value()] = &op;
                    if (auto constIntOp = llvm::dyn_cast<arith::ConstantIntOp>(op))
                        intToConstantOp[constIntOp.value()] = &op;

                } else if (auto mAxiCreateOp = llvm::dyn_cast<tor::AXICreateOp>(op)) {
                    std::string bus = "";
                    int maxRLen = 256;
                    int maxWLen = 256;
                    if (mAxiCreateOp->hasAttr("bus")) {
                        bus = dyn_cast<mlir::StringAttr>(mAxiCreateOp->getAttr("bus")).getValue().str();
                    }
                    if (mAxiCreateOp->hasAttr("ARLEN")) {
                        maxRLen = dyn_cast<mlir::IntegerAttr>(mAxiCreateOp->getAttr("ARLEN")).getInt();
                    }
                    if (mAxiCreateOp->hasAttr("AWLEN")) {
                        maxWLen = dyn_cast<mlir::IntegerAttr>(mAxiCreateOp->getAttr("AWLEN")).getInt();
                    }
                    if (busMap.find(bus) != busMap.end()) {
                        auto busId = busMap.at(bus);
                        buses[busId].maxRLen = std::max(buses[busId].maxRLen, maxRLen);
                        buses[busId].maxWLen = std::max(buses[busId].maxWLen, maxWLen);
                    } else {
                        busMap[bus] = buses.size();
                        buses.push_back(MAxiBus(bus, maxRLen, maxWLen));
                    }
                    maxiToBusMap[&op] = busMap.at(bus);
                    buses[busMap.at(bus)].addMemory(&op);
                    // auto mAxiValue = mAxiCreateOp.getResult();
                    getAllMAxiDependentOp(buses[busMap[bus]].getMemory(&op));
                }
            }

            // early exit when no buses
            if (buses.empty()) {
                return;
            }

            // burst infer
            designOp.walk([&](tor::FuncOp funcOp) {
                // pipeline function 参与infer，因为假如function内包含maxi
                // 就pipeline不了，pipeline这个attr就失效了(TBD, 可能可以pipeline)
                mAxiBurstInfer(funcOp);
            });
        }

        MLIRContext *ctx;
        OpBuilder* builder;
        // key: maxi_create_op, value: bus id in "buses" vector
        std::unordered_map<mlir::Operation*, size_t> maxiToBusMap;
        std::vector<MAxiBus> buses;
        std::unordered_map<int32_t, mlir::Operation*> intToConstantOp;
        arith::ConstantOp lastConstOp; // necessary?
        // key: funcOp, value: its callOp that called by other funOp,
        // must be called once, pipeline funcOp are not considered(TBD)
        std::unordered_map<mlir::Operation*, mlir::Operation*> funcOpCallOpMap;
    };
}  // end anonymous namespace

namespace mlir
{
    std::unique_ptr<OperationPass<tor::DesignOp>> createMAxiBurstInferPass()
    {
        return std::make_unique<MAxiBurstInferPass>();
    }
}
