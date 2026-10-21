//===- FloatOpGenerator.h - Scheduled issue/collect for dedicated FP32 IP
//---===//
#ifndef APS_FLOATOPGENERATOR_H
#define APS_FLOATOPGENERATOR_H
#include "APS/BBHandler.h"

namespace mlir {
class FloatOpGenerator : public OperationGenerator {
public:
  explicit FloatOpGenerator(BBHandler *handler) : OperationGenerator(handler) {}
  bool canHandle(Operation *op) const override;
  LogicalResult prepare(Operation *op);
  LogicalResult generateRule(Operation *op, OpBuilder &builder, Location loc,
                             int64_t slot,
                             llvm::DenseMap<Value, Value> &localMap) override;
  LogicalResult collectResults(int64_t slot, OpBuilder &builder,
                               llvm::DenseMap<Value, Value> &localMap);
  llvm::SmallVector<Value> getResults(int64_t slot) const;
  void clear() {
    instances.clear();
    completions.clear();
  }

private:
  llvm::DenseMap<Operation *, circt::cmt2::ecmt2::Instance *> instances;
  llvm::DenseMap<int64_t, llvm::SmallVector<Operation *>> completions;
};
} // namespace mlir
#endif
