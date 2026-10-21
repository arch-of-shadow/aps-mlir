#ifndef SCALEISA_ADAPTER_PASSES_H
#define SCALEISA_ADAPTER_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir::scaleisa_adapter {
std::unique_ptr<Pass> createConvertPass();
std::unique_ptr<Pass> createSplitPass();
std::unique_ptr<Pass> createSplitPass(unsigned maxExtraCSRs);
std::unique_ptr<Pass> createInsertPragmasPass();
std::unique_ptr<Pass> createInsertPragmasPass(StringRef mode);
std::unique_ptr<Pass> createMaterializeAPSPass();
void registerAdapterPipeline();

#define GEN_PASS_DECL
#include "scaleisa-adapter/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "scaleisa-adapter/Passes.h.inc"
} // namespace mlir::scaleisa_adapter

#endif
