#ifndef SCALEISA_COMPILER_PASSES_H
#define SCALEISA_COMPILER_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir::scaleisa_compiler {
std::unique_ptr<Pass> createLowerCIPass();
void registerScaleISACompilerPasses();
} // namespace mlir::scaleisa_compiler

#endif
