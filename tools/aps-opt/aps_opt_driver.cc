#include "aps_opt.hpp"

using namespace mlir;

int aps_opt_driver(int argc, char **argv) {
    mlir::registerAllPasses();

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
    registry_hector.insert<circt::comb::CombDialect>();

    mlir::registerTORPasses();
    mlir::registerAPSPasses();

    return failed(mlir::MlirOptMain(argc, argv, "HECTOR optimizer driver\n", registry_hector));
}
