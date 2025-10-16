#ifndef APS_PASS_DETAIL_H
#define APS_PASS_DETAIL_H

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir
{
  template <typename ConcreteDialect>
  void registerDialect(DialectRegistry &registry);

#define GEN_PASS_CLASSES
#include "APS/Passes.h.inc"
}
#endif //APS_PASS_DETAIL_H