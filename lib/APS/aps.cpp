#include "APS/APSDialect.h"
#include "APS/APSOps.h"

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"

#include "APS/APSDialect.cpp.inc"
#define GET_OP_CLASSES
#include "APS/APS.cpp.inc"

using namespace mlir;
using namespace aps;

void APSDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "APS/APS.cpp.inc"
  >();
}

// Force template instantiation for TypeID
namespace mlir::detail {
template struct TypeIDResolver<aps::APSDialect, void>;
}