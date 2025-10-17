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

#include "circt/Dialect/Comb/CombOps.h"

class HectorMemRefInsider
    : public mlir::MemRefElementTypeInterface::FallbackModel<HectorMemRefInsider> {};

template <typename T>
struct HectorPtrElementModel
    : public mlir::LLVM::PointerElementTypeInterface::ExternalModel<
          HectorPtrElementModel<T>, T> {};

using namespace mlir;

int hector_driver(int argc, char **argv) {
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
    
    // registry_hector.addExtension(+[](MLIRContext *ctx, LLVM::LLVMDialect *dialect) {
    //     LLVM::LLVMFunctionType::attachInterface<HectorMemRefInsider>(*ctx);
    // });
    // registry_hector.addExtension(+[](MLIRContext *ctx, LLVM::LLVMDialect *dialect) {
    //     LLVM::LLVMArrayType::attachInterface<HectorMemRefInsider>(*ctx);
    // });
    // registry_hector.addExtension(+[](MLIRContext *ctx, LLVM::LLVMDialect *dialect) {
    //     LLVM::LLVMPointerType::attachInterface<HectorMemRefInsider>(*ctx);
    // });
    // registry_hector.addExtension(+[](MLIRContext *ctx, LLVM::LLVMDialect *dialect) {
    //     LLVM::LLVMStructType::attachInterface<HectorMemRefInsider>(*ctx);
    // });
    // registry_hector.addExtension(+[](MLIRContext *ctx, memref::MemRefDialect *dialect) {
    //     MemRefType::attachInterface<HectorPtrElementModel<MemRefType>>(*ctx);
    // });

    // registry_hector.addExtension(+[](MLIRContext *ctx, LLVM::LLVMDialect *dialect) {
    //     LLVM::LLVMStructType::attachInterface<
    //         HectorPtrElementModel<LLVM::LLVMStructType>>(*ctx);
    // });

    // registry_hector.addExtension(+[](MLIRContext *ctx, LLVM::LLVMDialect *dialect) {
    //     LLVM::LLVMPointerType::attachInterface<
    //         HectorPtrElementModel<LLVM::LLVMPointerType>>(*ctx);
    // });

    // registry_hector.addExtension(+[](MLIRContext *ctx, LLVM::LLVMDialect *dialect) {
    //     LLVM::LLVMArrayType::attachInterface<HectorPtrElementModel<LLVM::LLVMArrayType>>(
    //         *ctx);
    // });

    return failed(mlir::MlirOptMain(argc, argv, "HECTOR optimizer driver\n", registry_hector));
}
