#ifndef TOR_PASSES_H
#define TOR_PASSES_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "TOR/TOR.h"
#include <memory>

namespace mlir {
std::unique_ptr<OperationPass<mlir::tor::DesignOp>> createTORSchedulePass();
std::unique_ptr<OperationPass<mlir::tor::DesignOp>> createTORTimeGraphPass();
std::unique_ptr<OperationPass<mlir::ModuleOp>> createTORSplitPass();

std::unique_ptr<OperationPass<mlir::tor::DesignOp>> createSCFToTORPass();
std::unique_ptr<OperationPass<mlir::tor::DesignOp>> createSCFIterArgsPass();
std::unique_ptr<OperationPass<mlir::tor::DesignOp>> createSCFDumpPass();
std::unique_ptr<OperationPass<mlir::tor::DesignOp>> createTORDumpPass();

std::unique_ptr<OperationPass<mlir::ModuleOp>> createConvertInputPass();
std::unique_ptr<OperationPass<mlir::ModuleOp>>
createConvertInputPass(double clock, llvm::StringRef resource);
std::unique_ptr<OperationPass<mlir::ModuleOp>>
createConvertInputPass(double clock, llvm::StringRef resource,
                       llvm::StringRef outputPath);

std::unique_ptr<Pass> createAffineForLoweringPass();
std::unique_ptr<OperationPass<mlir::ModuleOp>> createHlsUnrollPass();
std::unique_ptr<Pass> createNormalizeMemrefIndicesPass();
std::unique_ptr<OperationPass<mlir::ModuleOp>> createNewArrayPartitionPass();
std::unique_ptr<OperationPass<mlir::ModuleOp>> createArrayOptPass();
std::unique_ptr<OperationPass<mlir::ModuleOp>> createCountCyclesPass();
std::unique_ptr<OperationPass<mlir::tor::DesignOp>> createTORFusePass();
std::unique_ptr<OperationPass<mlir::func::FuncOp>>
createExpressionBalancePass();
std::unique_ptr<OperationPass<mlir::func::FuncOp>> createLoopTripcountPass();
std::unique_ptr<Pass> createReinterpretCastPass();
std::unique_ptr<OperationPass<mlir::ModuleOp>> createMinMaxToCmpSelectPass();
std::unique_ptr<Pass> createRaiseSCFToAffinePass();

#define GEN_PASS_REGISTRATION

#include "TOR/Passes.h.inc"

}
#endif // TOR_PASSES_H
