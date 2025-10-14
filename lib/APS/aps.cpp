#include "APS/APSDialect.h"
#include "APS/APSOps.h"

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Builders.h"

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

//===----------------------------------------------------------------------===//
// MemoryMapOp
//===----------------------------------------------------------------------===//

ParseResult MemoryMapOp::parse(OpAsmParser &parser, OperationState &result) {
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, /*arguments=*/{}, /*argTypes=*/{}))
    return failure();

  return success();
}

void MemoryMapOp::print(OpAsmPrinter &p) {
  p << " ";
  p.printRegion(getRegion(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);
}

// Force template instantiation for TypeID
namespace mlir::detail {
template struct TypeIDResolver<aps::APSDialect, void>;
}