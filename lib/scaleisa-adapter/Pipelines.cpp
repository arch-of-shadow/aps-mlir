#include "scaleisa-adapter/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir::scaleisa_adapter {
namespace {
struct AdapterOptions : PassPipelineOptions<AdapterOptions> {
  Option<unsigned> maxExtraCSRs{*this, "max-extra-csrs", llvm::cl::init(0),
                               llvm::cl::desc("Maximum extra CSR inputs")};
  Option<std::string> pragma{*this, "pragma", llvm::cl::init("none"),
                             llvm::cl::desc("none, pipeline, or unroll")};
};
} // namespace
void registerAdapterPipeline() {
  PassPipelineRegistration<AdapterOptions>(
      "scaleisa-adapter", "Convert, split, then insert implementation pragmas",
      [](OpPassManager &pm, const AdapterOptions &options) {
        pm.addPass(createConvertPass());
        pm.addPass(createSplitPass(options.maxExtraCSRs));
        pm.addPass(createInsertPragmasPass(options.pragma));
      });
}
} // namespace mlir::scaleisa_adapter
