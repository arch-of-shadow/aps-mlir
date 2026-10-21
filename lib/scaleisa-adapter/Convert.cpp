#include "scaleisa-adapter/Passes.h"
#include "Utils.h"
#include "APS/APSDialect.h"
#include "APS/APSOps.h"
#include "APS/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/PassManager.h"

namespace mlir::scaleisa_adapter {
#define GEN_PASS_DEF_SCALEISACONVERT
#include "scaleisa-adapter/Passes.h.inc"

namespace {
struct ConvertPass : impl::ScaleISAConvertBase<ConvertPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<aps::APSDialect, affine::AffineDialect, arith::ArithDialect,
                    linalg::LinalgDialect, memref::MemRefDialect,
                    scf::SCFDialect>();
  }
  void runOnOperation() override {
    for (auto function : getOperation().getOps<func::FuncOp>()) {
      if (!isCandidate(function))
        continue;
      if (!llvm::hasSingleElement(function.getBody())) {
        function.emitError("adapter requires a single entry block");
        return signalPassFailure();
      }
      bool invalid = false;
      function.walk([&](Operation *op) {
        for (Type type : llvm::concat<Type>(op->getOperandTypes(),
                                           op->getResultTypes()))
          if (isa<TensorType>(type))
            invalid = true;
      });
      if (invalid) {
        function.emitError("bufferize tensor candidates before APS conversion");
        return signalPassFailure();
      }
      // These are implementation choices from the discovery/evaluation run.
      // Re-plan them only after instruction boundaries have been established.
      function.walk([](Operation *op) {
        for (StringRef name : {"pipeline", "II", "unroll", "ref_starttime",
                               "ref_endtime", "starttime", "endtime", "dump",
                               "schedule", "cycle", "latency_cycles", "clock",
                               "resource", "evaluation_status", "fully_unrolled",
                               "area_square_microns", "datapath_area_square_microns",
                               "control_area_square_microns"})
          op->removeAttr(name);
      });
      OpPassManager pipeline(func::FuncOp::getOperationName());
      pipeline.addPass(createConvertLinalgToAffineLoopsPass());
      pipeline.addPass(createAffineMemToAPSPass());
      pipeline.addPass(createMemRefToAPSPass());
      if (failed(runPipeline(pipeline, function)))
        return signalPassFailure();
      // Scalar references have no scratchpad indices. Keep their standard
      // memory operations until the CPU base pointer is materialized; the APS
      // scratchpad assembly format requires an index type list.
      SmallVector<Operation *> scalarAccesses;
      function.walk([&](Operation *op) {
        Value memory;
        if (auto load = dyn_cast<aps::ReadSmem>(op))
          memory = load.getMemref();
        else if (auto store = dyn_cast<aps::WriteSmem>(op))
          memory = store.getMemref();
        auto type = memory ? dyn_cast<MemRefType>(memory.getType()) : MemRefType();
        if (type && type.getRank() == 0)
          scalarAccesses.push_back(op);
      });
      for (Operation *op : scalarAccesses) {
        OpBuilder builder(op);
        if (auto load = dyn_cast<aps::ReadSmem>(op)) {
          auto value = builder.create<memref::LoadOp>(op->getLoc(), load.getMemref());
          load.getResult().replaceAllUsesWith(value.getResult());
        } else {
          auto store = cast<aps::WriteSmem>(op);
          builder.create<memref::StoreOp>(op->getLoc(), store.getValue(), store.getMemref());
        }
        op->erase();
      }
      function->setAttr("scaleisa.converted", UnitAttr::get(&getContext()));
    }
  }
};
} // namespace
std::unique_ptr<Pass> createConvertPass() {
  return std::make_unique<ConvertPass>();
}
} // namespace mlir::scaleisa_adapter
