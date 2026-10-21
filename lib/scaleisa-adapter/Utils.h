#ifndef SCALEISA_ADAPTER_UTILS_H
#define SCALEISA_ADAPTER_UTILS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir::scaleisa_adapter {
inline bool isCandidate(func::FuncOp function) {
  if (function.isExternal())
    return false;
  auto generated = function->getAttrOfType<BoolAttr>("au.generated");
  return (generated && generated.getValue()) ||
         function->hasAttr("scaleisa.kernel");
}
inline bool isInstruction(func::FuncOp function) {
  return function->hasAttr("scaleisa.instruction");
}
} // namespace mlir::scaleisa_adapter
#endif
