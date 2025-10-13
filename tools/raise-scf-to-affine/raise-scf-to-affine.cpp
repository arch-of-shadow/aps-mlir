#include "APS/APSDialect.h"
#include "TOR/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

  // Register dialects needed for this pass and test files
  registry.insert<mlir::affine::AffineDialect,
                  mlir::arith::ArithDialect,
                  mlir::func::FuncDialect,
                  mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect,
                  aps::APSDialect>();

  // Register the pass
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::createRaiseSCFToAffinePass();
  });

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Raise SCF to Affine Tool\n", registry));
}
