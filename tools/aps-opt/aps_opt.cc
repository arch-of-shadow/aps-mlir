#include "APS/APSDialect.h"
#include "APS/Passes.h"
#include "TOR/Passes.h"
#include "TOR/TORDialect.h"
#include "circt/Conversion/Passes.h"
#include "circt/Dialect/Cmt2/Cmt2Dialect.h"
#include "circt/Dialect/Cmt2/Cmt2Passes.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dialect.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();

  mlir::DialectRegistry registry;
  registry.insert<mlir::affine::AffineDialect>();
  registry.insert<mlir::LLVM::LLVMDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::scf::SCFDialect>();
  registry.insert<mlir::tor::TORDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::math::MathDialect>();
  registry.insert<aps::APSDialect>();
  registry.insert<circt::comb::CombDialect>();
  registry.insert<circt::cmt2::Cmt2Dialect>();
  registry.insert<circt::firrtl::FIRRTLDialect>();

  mlir::registerTORPasses();
  mlir::registerAPSPasses();
  circt::cmt2::registerPasses();
  circt::registerLowerCmt2ToFIRRTLPass();

  return failed(
      mlir::MlirOptMain(argc, argv, "aps optimizer driver\n", registry));
}
