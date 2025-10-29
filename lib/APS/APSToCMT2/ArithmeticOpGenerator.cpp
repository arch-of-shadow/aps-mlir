//===- ArithmeticOpGenerator.cpp - Arithmetic Operation Generator ----------===//
//
// This file implements the arithmetic operation generator for TOR functions
//
//===----------------------------------------------------------------------===//

#include "APS/BBHandler.h"
#include "circt/Dialect/Cmt2/ECMT2/Signal.h"
#include "TOR/TOR.h"

namespace mlir {

using namespace mlir;
using namespace mlir::tor;
using namespace circt::cmt2::ecmt2;
using namespace circt::cmt2::ecmt2::stl;
using namespace circt::firrtl;

LogicalResult ArithmeticOpGenerator::generateRule(Operation *op, mlir::OpBuilder &b,
                                                Location loc, int64_t slot,
                                                llvm::DenseMap<mlir::Value, mlir::Value> &localMap) {
  if (auto addOp = dyn_cast<tor::AddIOp>(op)) {
    auto lhs = getValueInRule(addOp.getLhs(), op, 0, b, localMap, loc);
    auto rhs = getValueInRule(addOp.getRhs(), op, 1, b, localMap, loc);
    if (failed(lhs) || failed(rhs))
      return failure();
    return performArithmeticOp(b, loc, *lhs, *rhs, addOp.getResult(), "add", localMap);
  } else if (auto subOp = dyn_cast<tor::SubIOp>(op)) {
    auto lhs = getValueInRule(subOp.getLhs(), op, 0, b, localMap, loc);
    auto rhs = getValueInRule(subOp.getRhs(), op, 1, b, localMap, loc);
    if (failed(lhs) || failed(rhs))
      return failure();
    return performArithmeticOp(b, loc, *lhs, *rhs, subOp.getResult(), "sub", localMap);
  } else if (auto mulOp = dyn_cast<tor::MulIOp>(op)) {
    auto lhs = getValueInRule(mulOp.getLhs(), op, 0, b, localMap, loc);
    auto rhs = getValueInRule(mulOp.getRhs(), op, 1, b, localMap, loc);
    if (failed(lhs) || failed(rhs))
      return failure();
    return performArithmeticOp(b, loc, *lhs, *rhs, mulOp.getResult(), "mul", localMap);
  }

  return failure();
}

bool ArithmeticOpGenerator::canHandle(Operation *op) const {
  return isa<tor::AddIOp, tor::SubIOp, tor::MulIOp>(op);
}

LogicalResult ArithmeticOpGenerator::performArithmeticOp(mlir::OpBuilder &b, Location loc,
                                                       mlir::Value lhs, mlir::Value rhs,
                                                       mlir::Value result, StringRef opName,
                                                       llvm::DenseMap<mlir::Value, mlir::Value> &localMap) {
  // Determine result width based on operation type
  auto requiredWidth = cast<IntegerType>(result.getType()).getWidth();

  // Perform the arithmetic operation using Signal abstraction
  Signal lhsSignal(lhs, &b, loc);
  Signal rhsSignal(rhs, &b, loc);

  Signal resultSignal(lhs, &b, loc); // dummy init
  if (opName == "add") {
    resultSignal = lhsSignal + rhsSignal;
  } else if (opName == "sub") {
    resultSignal = lhsSignal - rhsSignal;
  } else if (opName == "mul") {
    resultSignal = lhsSignal * rhsSignal;
  } else {
    return failure();
  }

  auto firrtlWidth = resultSignal.getWidth();
  Signal resultSignalWidthFix = resultSignal;
  if (firrtlWidth > requiredWidth) {
    resultSignalWidthFix = resultSignal.bits(requiredWidth - 1, 0);
  } else if (firrtlWidth < requiredWidth) {
    resultSignalWidthFix = resultSignal.pad(requiredWidth);
  }

  localMap[result] = resultSignalWidthFix.getValue();
  return success();
}

} // namespace mlir