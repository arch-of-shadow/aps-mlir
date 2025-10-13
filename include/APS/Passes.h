#ifndef APS_PASSES_H
#define APS_PASSES_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
    
std::unique_ptr<OperationPass<mlir::func::FuncOp>> createSCFForIndexCastPass();
std::unique_ptr<OperationPass<mlir::func::FuncOp>> createAPSMemToMemRefPass();

#define GEN_PASS_REGISTRATION
#include "APS/Passes.h.inc"

} // namespace mlir

#endif // APS_PASSES_H
