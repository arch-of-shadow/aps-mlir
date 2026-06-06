//===- IfHandler.h - If Handler for Block Coordination --------*- C++ -*-===//
//
// This file declares the tor.if handler that lowers conditional scopes using
// explicit branch entry and join token FIFOs.
//
//===----------------------------------------------------------------------===//

#ifndef APS_IFHANDLER_H
#define APS_IFHANDLER_H

#include "APS/APSOps.h"
#include "APS/BlockHandler.h"
#include "circt/Dialect/Cmt2/ECMT2/Instance.h"

namespace mlir {

using namespace mlir::tor;
using namespace circt::cmt2::ecmt2;
using namespace circt::cmt2::ecmt2::stl;
using namespace circt::firrtl;

class IfHandler : public BlockHandler {
public:
  IfHandler(APSToCMT2Pass *pass, Module *mainModule, tor::FuncOp funcOp,
            Instance *poolInstance, Instance *roccInstance,
            Instance *hellaMemInstance, InterfaceDecl *dmaItfc,
            InterfaceDecl *csrItfc, Circuit &circuit, Clock mainClk,
            Reset mainRst, unsigned long instructionId,
            Instance *regRdInstance, Instance *inputTokenFIFO,
            Instance *outputTokenFIFO,
            llvm::DenseMap<Value, Instance *> &input_fifos,
            llvm::DenseMap<Value, llvm::SmallVector<std::pair<BlockInfo *, Instance *>, 4>>
                &output_fifos,
            const std::string &namePrefix = "");

  LogicalResult processIfBlock(BlockInfo &ifBlock);

  /// Require a single live invocation context for this if scope.
  void setRequireContextToken(bool enabled) { requireContextToken = enabled; }

protected:
  LogicalResult processBlock(BlockInfo &block) override;

private:
  tor::IfOp ifOp;
  std::string ifName;
  Instance *thenEntryTokenFIFO = nullptr;
  Instance *elseEntryTokenFIFO = nullptr;
  Instance *thenDoneTokenFIFO = nullptr;
  Instance *elseDoneTokenFIFO = nullptr;
  Instance *joinTokenFIFO = nullptr;
  Instance *contextTokenReg = nullptr;
  bool requireContextToken = false;
  llvm::DenseMap<Value, Instance *> inputStateRegs;
  llvm::DenseMap<Value, Instance *> thenInputFIFOs;
  llvm::DenseMap<Value, Instance *> elseInputFIFOs;
  llvm::DenseMap<Value, llvm::SmallVector<std::pair<BlockInfo *, Instance *>, 4>>
      thenOutputFIFOs;
  llvm::DenseMap<Value, llvm::SmallVector<std::pair<BlockInfo *, Instance *>, 4>>
      elseOutputFIFOs;
  llvm::SmallVector<Instance *, 4> thenResultFIFOs;
  llvm::SmallVector<Instance *, 4> elseResultFIFOs;
  BlockInfo *currentIfBlock = nullptr;

  LogicalResult createIfInfrastructure(BlockInfo &ifBlock);
  LogicalResult generateDispatchRule(BlockInfo &ifBlock);
  LogicalResult generateBranchTagRule(StringRef branchName,
                                      Instance *branchDoneFIFO, bool isThen);
  LogicalResult generateJoinRule();
  LogicalResult processBranchRegion(Block &region, Instance *entryTokenFIFO,
                                    Instance *doneTokenFIFO,
                                    StringRef branchName);
  LogicalResult generateYieldOnlyBranchRule(StringRef branchName,
                                            Instance *entryTokenFIFO,
                                            Instance *doneTokenFIFO);
  Instance *createTokenFIFO(StringRef suffix);
  Instance *createDataFIFO(StringRef suffix, Value value);
  Instance *createStateReg(StringRef suffix, Value value, unsigned index);
  Value materializeCondition(OpBuilder &builder, Location loc,
                             BlockInfo &ifBlock);
  void emitBranchInputs(OpBuilder &builder, Location loc,
                        llvm::DenseMap<Value, Value> &capturedValues,
                        llvm::DenseMap<Value, Instance *> &branchFIFOs);
  void emitLiveThroughOutputs(OpBuilder &builder, Location loc);
  void emitResultOutputs(OpBuilder &builder, Location loc,
                         ArrayRef<Instance *> resultFIFOs);
  LogicalResult createResultMergeFIFOs();
  LogicalResult populateBranchResultOutputs(Region &region,
                                            ArrayRef<Instance *> resultFIFOs,
                                            llvm::DenseMap<Value, llvm::SmallVector<std::pair<BlockInfo *, Instance *>, 4>>
                                                &branchOutputFIFOs);
  bool isValueUsedInRegion(Value value, Region &region) const;
  bool hasElseRegion();
  bool hasNonEmptyRegion(Region &region) const;
};

} // namespace mlir

#endif // APS_IFHANDLER_H
