#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"

#include "TOR/TORDialect.h"
#include "TOR/Passes.h"

#include "APS/APSDialect.h"
#include "APS/Passes.h"


int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::registerAPSPasses();

  mlir::DialectRegistry registry_hector;
  registry_hector.insert<mlir::affine::AffineDialect>();
  registry_hector.insert<mlir::LLVM::LLVMDialect>();
  registry_hector.insert<mlir::memref::MemRefDialect>();
  registry_hector.insert<mlir::arith::ArithDialect>();
  registry_hector.insert<mlir::scf::SCFDialect>();
  registry_hector.insert<mlir::tor::TORDialect>();
  registry_hector.insert<mlir::func::FuncDialect>();
  registry_hector.insert<mlir::math::MathDialect>();
  registry_hector.insert<aps::APSDialect>();
    
  mlir::registerArrayOpt();
  mlir::registerArrayPartition();
  mlir::registerConvertAffineFor();
  mlir::registerRaiseSCFToAffine();
  mlir::registerUnificationIndexCast();
  mlir::registerArrayOpt();
  mlir::registerHlsUnroll();
  mlir::registerNewArrayPartition();
    
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Array Partition Tool\n", registry_hector));
}
