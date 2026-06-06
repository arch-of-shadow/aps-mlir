//===- ArithmeticOpGenerator.cpp - Arithmetic Operation Generator ----------===//
//
// This file implements the arithmetic operation generator for TOR functions
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LogicalResult.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "APS/BBHandler.h"
#include "circt/Dialect/Cmt2/ECMT2/Signal.h"
#include "TOR/TOR.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

namespace mlir {

using namespace mlir::tor;
using namespace circt::cmt2::ecmt2;
using namespace circt::cmt2::ecmt2::stl;
using namespace circt::firrtl;

static Signal fitToWidth(Signal value, unsigned requiredWidth) {
  auto actualWidth = value.getWidth();
  if (actualWidth > requiredWidth)
    return value.bits(requiredWidth - 1, 0);
  if (actualWidth < requiredWidth)
    return value.pad(requiredWidth);
  return value;
}

static void padToSameWidth(Signal &lhs, Signal &rhs) {
  auto lhsWidth = lhs.getWidth();
  auto rhsWidth = rhs.getWidth();
  auto maxWidth = std::max(lhsWidth, rhsWidth);
  if (lhsWidth < maxWidth)
    lhs = lhs.pad(maxWidth);
  if (rhsWidth < maxWidth)
    rhs = rhs.pad(maxWidth);
}

static mlir::Value signExtendToWidth(mlir::OpBuilder &b, Location loc,
                                     mlir::Value value, unsigned width) {
  Signal valueSignal(value, &b, loc);
  auto inputWidth = valueSignal.getWidth();
  auto signedType = circt::firrtl::SIntType::get(b.getContext(), inputWidth);
  mlir::Value signedValue =
      b.create<circt::firrtl::AsSIntPrimOp>(loc, signedType, value);
  if (inputWidth == width)
    return signedValue;

  auto paddedType = circt::firrtl::SIntType::get(b.getContext(), width);
  return b.create<circt::firrtl::PadPrimOp>(loc, paddedType, signedValue,
                                            width);
}

static mlir::Value signedValueToUIntWidth(mlir::OpBuilder &b, Location loc,
                                          mlir::Value signedValue,
                                          unsigned requiredWidth) {
  auto actualWidth = Signal(signedValue, &b, loc).getWidth();

  if (actualWidth > requiredWidth) {
    auto wideUIntType =
        circt::firrtl::UIntType::get(b.getContext(), actualWidth);
    auto wideUInt =
        b.create<circt::firrtl::AsUIntPrimOp>(loc, wideUIntType, signedValue);
    return Signal(wideUInt, &b, loc).bits(requiredWidth - 1, 0).getValue();
  }

  if (actualWidth == requiredWidth) {
    auto resultType =
        circt::firrtl::UIntType::get(b.getContext(), requiredWidth);
    return b.create<circt::firrtl::AsUIntPrimOp>(loc, resultType, signedValue);
  }

  auto paddedType =
      circt::firrtl::SIntType::get(b.getContext(), requiredWidth);
  auto padded =
      b.create<circt::firrtl::PadPrimOp>(loc, paddedType, signedValue,
                                         requiredWidth);
  auto resultType =
      circt::firrtl::UIntType::get(b.getContext(), requiredWidth);
  return b.create<circt::firrtl::AsUIntPrimOp>(loc, resultType, padded);
}

static mlir::tor::CmpIPredicate
toTorCmpPredicate(mlir::arith::CmpIPredicate predicate) {
  return static_cast<mlir::tor::CmpIPredicate>(predicate);
}

LogicalResult ArithmeticOpGenerator::generateRule(Operation *op, mlir::OpBuilder &b,
                                                Location loc, int64_t slot,
                                                llvm::DenseMap<mlir::Value, mlir::Value> &localMap) {
  if (auto addOp = dyn_cast<tor::AddIOp>(op)) {
    auto lhs = getValueInRule(addOp.getLhs(), op, b, localMap, loc);
    auto rhs = getValueInRule(addOp.getRhs(), op, b, localMap, loc);
    if (failed(lhs) || failed(rhs))
      return failure();
    return performArithmeticOp(b, loc, *lhs, *rhs, addOp.getResult(),
                               ArithmeticKind::Add, localMap);
  } else if (auto subOp = dyn_cast<tor::SubIOp>(op)) {
    auto lhs = getValueInRule(subOp.getLhs(), op, b, localMap, loc);
    auto rhs = getValueInRule(subOp.getRhs(), op, b, localMap, loc);
    if (failed(lhs) || failed(rhs))
      return failure();
    return performArithmeticOp(b, loc, *lhs, *rhs, subOp.getResult(),
                               ArithmeticKind::Sub, localMap);
  } else if (auto mulOp = dyn_cast<tor::MulIOp>(op)) {
    auto lhs = getValueInRule(mulOp.getLhs(), op, b, localMap, loc);
    auto rhs = getValueInRule(mulOp.getRhs(), op, b, localMap, loc);
    if (failed(lhs) || failed(rhs))
      return failure();
    return performArithmeticOp(b, loc, *lhs, *rhs, mulOp.getResult(),
                               ArithmeticKind::Mul, localMap);
  } else if (auto cmpOp = dyn_cast<tor::CmpIOp>(op)) {
    auto lhs = getValueInRule(cmpOp.getLhs(), op, b, localMap, loc);
    auto rhs = getValueInRule(cmpOp.getRhs(), op, b, localMap, loc);
    if (failed(lhs) || failed(rhs))
      return failure();
    return performComparisonOp(b, loc, *lhs, *rhs, cmpOp.getResult(), cmpOp.getPredicate(), localMap);
  } else if (auto cmpOp = dyn_cast<arith::CmpIOp>(op)) {
    auto lhs = getValueInRule(cmpOp.getLhs(), op, b, localMap, loc);
    auto rhs = getValueInRule(cmpOp.getRhs(), op, b, localMap, loc);
    if (failed(lhs) || failed(rhs))
      return failure();
    return performComparisonOp(b, loc, *lhs, *rhs, cmpOp.getResult(),
                               toTorCmpPredicate(cmpOp.getPredicate()),
                               localMap);
  } else if (auto divuiOp = dyn_cast<arith::DivUIOp>(op)) {
    auto lhs = getValueInRule(divuiOp.getLhs(), op, b, localMap, loc);
    auto rhs = getValueInRule(divuiOp.getRhs(), op, b, localMap, loc);
    if (failed(lhs) || failed(rhs))
      return failure();
    return performDivOp(b, loc, *lhs, *rhs, divuiOp.getResult(),
                        DivisionKind::Unsigned, localMap);
  } else if (auto divsiOp = dyn_cast<arith::DivSIOp>(op)) {
    auto lhs = getValueInRule(divsiOp.getLhs(), op, b, localMap, loc);
    auto rhs = getValueInRule(divsiOp.getRhs(), op, b, localMap, loc);
    if (failed(lhs) || failed(rhs))
      return failure();
    return performDivOp(b, loc, *lhs, *rhs, divsiOp.getResult(),
                        DivisionKind::Signed, localMap);
  } else if (auto selectOp = dyn_cast<arith::SelectOp>(op)) {
    auto condition = getValueInRule(selectOp.getCondition(), op, b, localMap, loc);
    auto trueValue = getValueInRule(selectOp.getTrueValue(), op, b, localMap, loc);
    auto falseValue = getValueInRule(selectOp.getFalseValue(), op, b, localMap, loc);
    if (failed(condition) || failed(trueValue) || failed(falseValue))
      return failure();
    return performSelectOp(b, loc, *condition, *trueValue, *falseValue, selectOp.getResult(), localMap);
  } else if (auto extuiOp = dyn_cast<arith::ExtUIOp>(op)) {
    auto input = getValueInRule(extuiOp.getIn(), op, b, localMap, loc);
    if (failed(input))
      return failure();
    return performExtUIOp(b, loc, *input, extuiOp.getResult(), localMap);
  } else if (auto trunciOp = dyn_cast<arith::TruncIOp>(op)) {
    auto input = getValueInRule(trunciOp.getIn(), op, b, localMap, loc);
    if (failed(input))
      return failure();
    return performTruncIOp(b, loc, *input, trunciOp.getResult(), localMap);
  } else if (auto extractOp = dyn_cast<circt::comb::ExtractOp>(op)) {
    auto input = getValueInRule(extractOp.getInput(), op, b, localMap, loc);
    if (failed(input))
      return failure();
    return performExtractOp(b, loc, *input, extractOp.getLowBit(), extractOp.getResult(), localMap);
  } else if (auto shliOp = dyn_cast<arith::ShLIOp>(op)) {
    auto lhs = getValueInRule(shliOp.getLhs(), op, b, localMap, loc);
    auto rhs = getValueInRule(shliOp.getRhs(), op, b, localMap, loc);
    if (failed(lhs) || failed(rhs))
      return failure();
    return performShiftOp(b, loc, *lhs, *rhs, shliOp.getResult(),
                          ShiftKind::Shl, localMap);
  } else if (auto shruiOp = dyn_cast<arith::ShRUIOp>(op)) {
    auto lhs = getValueInRule(shruiOp.getLhs(), op, b, localMap, loc);
    auto rhs = getValueInRule(shruiOp.getRhs(), op, b, localMap, loc);
    if (failed(lhs) || failed(rhs))
      return failure();
    return performShiftOp(b, loc, *lhs, *rhs, shruiOp.getResult(),
                          ShiftKind::ShrU, localMap);
  } else if (auto shrsiOp = dyn_cast<arith::ShRSIOp>(op)) {
    auto lhs = getValueInRule(shrsiOp.getLhs(), op, b, localMap, loc);
    auto rhs = getValueInRule(shrsiOp.getRhs(), op, b, localMap, loc);
    if (failed(lhs) || failed(rhs))
      return failure();
    return performShiftOp(b, loc, *lhs, *rhs, shrsiOp.getResult(),
                          ShiftKind::ShrS, localMap);
  } else if (auto andiOp = dyn_cast<arith::AndIOp>(op)) {
    auto lhs = getValueInRule(andiOp.getLhs(), op, b, localMap, loc);
    auto rhs = getValueInRule(andiOp.getRhs(), op, b, localMap, loc);
    if (failed(lhs) || failed(rhs))
      return failure();
    return performBitwiseOp(b, loc, *lhs, *rhs, andiOp.getResult(),
                            BitwiseKind::And, localMap);
  } else if (auto oriOp = dyn_cast<arith::OrIOp>(op)) {
    auto lhs = getValueInRule(oriOp.getLhs(), op, b, localMap, loc);
    auto rhs = getValueInRule(oriOp.getRhs(), op, b, localMap, loc);
    if (failed(lhs) || failed(rhs))
      return failure();
    return performBitwiseOp(b, loc, *lhs, *rhs, oriOp.getResult(),
                            BitwiseKind::Or, localMap);
  } else if (auto xoriOp = dyn_cast<arith::XOrIOp>(op)) {
    auto lhs = getValueInRule(xoriOp.getLhs(), op, b, localMap, loc);
    auto rhs = getValueInRule(xoriOp.getRhs(), op, b, localMap, loc);
    if (failed(lhs) || failed(rhs))
      return failure();
    return performBitwiseOp(b, loc, *lhs, *rhs, xoriOp.getResult(),
                            BitwiseKind::Xor, localMap);
  } else if (auto extsiOp = dyn_cast<arith::ExtSIOp>(op)) {
    auto input = getValueInRule(extsiOp.getIn(), op, b, localMap, loc);
    if (failed(input))
      return failure();
    return performExtSIOp(b, loc, *input, extsiOp.getResult(), localMap);
  }

  return op->emitError(
      "internal error: unsupported op reached arithmetic generator");
}

bool ArithmeticOpGenerator::canHandle(Operation *op) const {
  return isa<tor::AddIOp, tor::SubIOp, tor::MulIOp, tor::CmpIOp,
             arith::CmpIOp, arith::DivUIOp, arith::DivSIOp, arith::SelectOp,
             arith::ExtUIOp, arith::ExtSIOp, arith::TruncIOp, circt::comb::ExtractOp,
             arith::ShLIOp, arith::ShRUIOp, arith::ShRSIOp,
             arith::AndIOp, arith::OrIOp, arith::XOrIOp>(op);
}

LogicalResult ArithmeticOpGenerator::performArithmeticOp(mlir::OpBuilder &b, Location loc,
                                                       mlir::Value lhs, mlir::Value rhs,
                                                       mlir::Value result,
                                                       ArithmeticKind kind,
                                                       llvm::DenseMap<mlir::Value, mlir::Value> &localMap) {
  // Determine result width based on operation type
  auto requiredWidth = cast<IntegerType>(result.getType()).getWidth();

  // Perform the arithmetic operation using Signal abstraction
  Signal lhsSignal(lhs, &b, loc);
  Signal rhsSignal(rhs, &b, loc);

  Signal resultSignal(lhs, &b, loc); // dummy init
  switch (kind) {
  case ArithmeticKind::Add:
    resultSignal = lhsSignal + rhsSignal;
    break;
  case ArithmeticKind::Sub:
    resultSignal = lhsSignal - rhsSignal;
    break;
  case ArithmeticKind::Mul:
    resultSignal = lhsSignal * rhsSignal;
    break;
  }

  localMap[result] = fitToWidth(resultSignal, requiredWidth).getValue();
  return success();
}

LogicalResult ArithmeticOpGenerator::performDivOp(
    mlir::OpBuilder &b, Location loc, mlir::Value lhs, mlir::Value rhs,
    mlir::Value result, DivisionKind kind,
    llvm::DenseMap<mlir::Value, mlir::Value> &localMap) {
  auto requiredWidth = cast<IntegerType>(result.getType()).getWidth();

  if (kind == DivisionKind::Signed) {
    auto maxWidth = std::max(Signal(lhs, &b, loc).getWidth(),
                             Signal(rhs, &b, loc).getWidth());
    auto signedLhs = signExtendToWidth(b, loc, lhs, maxWidth);
    auto signedRhs = signExtendToWidth(b, loc, rhs, maxWidth);
    auto divResult =
        b.create<circt::firrtl::DivPrimOp>(loc, signedLhs, signedRhs);
    localMap[result] =
        signedValueToUIntWidth(b, loc, divResult, requiredWidth);
    return success();
  }

  Signal lhsSignal(lhs, &b, loc);
  Signal rhsSignal(rhs, &b, loc);
  padToSameWidth(lhsSignal, rhsSignal);
  auto divResult = b.create<circt::firrtl::DivPrimOp>(
      loc, lhsSignal.getValue(), rhsSignal.getValue());
  Signal resultSignal(divResult, &b, loc);
  localMap[result] = fitToWidth(resultSignal, requiredWidth).getValue();
  return success();
}

LogicalResult ArithmeticOpGenerator::performComparisonOp(mlir::OpBuilder &b, Location loc,
                                                        mlir::Value lhs, mlir::Value rhs,
                                                        mlir::Value result,
                                                        mlir::tor::CmpIPredicate predicate,
                                                        llvm::DenseMap<mlir::Value, mlir::Value> &localMap) {
  // Create Signal wrappers
  Signal lhsSignal(lhs, &b, loc);
  Signal rhsSignal(rhs, &b, loc);

  // Match widths to the maximum (like arith.cmpi expects)
  padToSameWidth(lhsSignal, rhsSignal);

  auto getSignedOperands = [&]() -> std::pair<mlir::Value, mlir::Value> {
    auto maxWidth = std::max(Signal(lhs, &b, loc).getWidth(),
                             Signal(rhs, &b, loc).getWidth());
    return {signExtendToWidth(b, loc, lhs, maxWidth),
            signExtendToWidth(b, loc, rhs, maxWidth)};
  };

  // Map predicate to Signal comparison operators
  mlir::Value compareResult;
  switch (predicate) {
    case mlir::tor::CmpIPredicate::eq:
      compareResult = (lhsSignal == rhsSignal).getValue();
      break;
    case mlir::tor::CmpIPredicate::ne:
      compareResult = (lhsSignal != rhsSignal).getValue();
      break;
    case mlir::tor::CmpIPredicate::ult:
      compareResult = (lhsSignal < rhsSignal).getValue();
      break;
    case mlir::tor::CmpIPredicate::ule:
      compareResult = (lhsSignal <= rhsSignal).getValue();
      break;
    case mlir::tor::CmpIPredicate::ugt:
      compareResult = (lhsSignal > rhsSignal).getValue();
      break;
    case mlir::tor::CmpIPredicate::uge:
      compareResult = (lhsSignal >= rhsSignal).getValue();
      break;
    case mlir::tor::CmpIPredicate::slt: {
      auto [signedLhs, signedRhs] = getSignedOperands();
      compareResult =
          b.create<circt::firrtl::LTPrimOp>(loc, signedLhs, signedRhs);
      break;
    }
    case mlir::tor::CmpIPredicate::sle: {
      auto [signedLhs, signedRhs] = getSignedOperands();
      compareResult =
          b.create<circt::firrtl::LEQPrimOp>(loc, signedLhs, signedRhs);
      break;
    }
    case mlir::tor::CmpIPredicate::sgt: {
      auto [signedLhs, signedRhs] = getSignedOperands();
      compareResult =
          b.create<circt::firrtl::GTPrimOp>(loc, signedLhs, signedRhs);
      break;
    }
    case mlir::tor::CmpIPredicate::sge: {
      auto [signedLhs, signedRhs] = getSignedOperands();
      compareResult =
          b.create<circt::firrtl::GEQPrimOp>(loc, signedLhs, signedRhs);
      break;
    }
    default:
      return result.getDefiningOp()->emitError(
          "unsupported comparison predicate in arithmetic generator");
  }

  // Get required result width from the TOR operation result type
  // The result type should be an integer type (typically i1 for comparisons)
  auto requiredWidth = cast<IntegerType>(result.getType()).getWidth();
  Signal resultSignal(compareResult, &b, loc);
  localMap[result] = fitToWidth(resultSignal, requiredWidth).getValue();
  return success();
}

LogicalResult ArithmeticOpGenerator::performSelectOp(mlir::OpBuilder &b, Location loc,
                                                     mlir::Value condition, mlir::Value trueValue,
                                                     mlir::Value falseValue, mlir::Value result,
                                                     llvm::DenseMap<mlir::Value, mlir::Value> &localMap) {
  // Create Signal wrappers
  Signal condSignal(condition, &b, loc);
  Signal trueSignal(trueValue, &b, loc);
  Signal falseSignal(falseValue, &b, loc);

  // Match widths to the maximum (ensure operands have same width for mux)
  padToSameWidth(trueSignal, falseSignal);

  // Perform mux operation: if condition is true, select trueSignal, else select falseSignal
  // Signal::mux signature is: condition.mux(trueVal, falseVal)
  Signal resultSignal = condSignal.mux(trueSignal, falseSignal);

  // Get required result width from the operation result type
  auto requiredWidth = cast<IntegerType>(result.getType()).getWidth();

  localMap[result] = fitToWidth(resultSignal, requiredWidth).getValue();
  return success();
}

LogicalResult ArithmeticOpGenerator::performExtUIOp(mlir::OpBuilder &b, Location loc,
                                                    mlir::Value input, mlir::Value result,
                                                    llvm::DenseMap<mlir::Value, mlir::Value> &localMap) {
  // Create Signal wrapper for input
  Signal inputSignal(input, &b, loc);

  // Get result width for extension
  auto requiredWidth = cast<IntegerType>(result.getType()).getWidth();

  // Perform zero-extension (unsigned extension) using pad
  Signal resultSignal = inputSignal.pad(requiredWidth);

  localMap[result] = resultSignal.getValue();
  return success();
}

LogicalResult ArithmeticOpGenerator::performTruncIOp(mlir::OpBuilder &b, Location loc,
                                                     mlir::Value input, mlir::Value result,
                                                     llvm::DenseMap<mlir::Value, mlir::Value> &localMap) {
  // Create Signal wrapper for input
  Signal inputSignal(input, &b, loc);

  // Get result width for truncation
  auto requiredWidth = cast<IntegerType>(result.getType()).getWidth();

  // Perform truncation by extracting lower bits
  Signal resultSignal = inputSignal.bits(requiredWidth - 1, 0);

  localMap[result] = resultSignal.getValue();
  return success();
}

LogicalResult ArithmeticOpGenerator::performExtractOp(mlir::OpBuilder &b, Location loc,
                                                      mlir::Value input, unsigned lowBit,
                                                      mlir::Value result,
                                                      llvm::DenseMap<mlir::Value, mlir::Value> &localMap) {
  // Create Signal wrapper for input
  Signal inputSignal(input, &b, loc);

  // Get result width to determine high bit
  auto resultWidth = cast<IntegerType>(result.getType()).getWidth();
  auto highBit = lowBit + resultWidth - 1;

  // Extract bits using Signal::bits(high, low)
  Signal resultSignal = inputSignal.bits(highBit, lowBit);

  localMap[result] = resultSignal.getValue();
  return success();
}

/// Try to get a constant integer value from a FIRRTL value.
/// Returns std::nullopt if not a constant.
static std::optional<int64_t> getConstantValue(mlir::Value value) {
  // Check if it's a FIRRTL constant
  if (auto constOp = value.getDefiningOp<circt::firrtl::ConstantOp>()) {
    return constOp.getValue().getSExtValue();
  }
  return std::nullopt;
}

LogicalResult ArithmeticOpGenerator::performShiftOp(mlir::OpBuilder &b, Location loc,
                                                     mlir::Value lhs, mlir::Value rhs,
                                                     mlir::Value result,
                                                     ShiftKind kind,
                                                     llvm::DenseMap<mlir::Value, mlir::Value> &localMap) {
  // Get result width
  auto requiredWidth = cast<IntegerType>(result.getType()).getWidth();

  // Check if shift amount is a constant
  auto constShiftAmount = getConstantValue(rhs);

  // Create FIRRTL shift operations using the underlying values
  mlir::Value shiftResult;

  if (constShiftAmount.has_value()) {
    // Constant shift - use ShlPrimOp/ShrPrimOp
    int64_t shiftAmt = *constShiftAmount;

    switch (kind) {
    case ShiftKind::Shl:
      // Constant left shift
      shiftResult = b.create<circt::firrtl::ShlPrimOp>(loc, lhs, shiftAmt);
      break;
    case ShiftKind::ShrU:
      shiftResult = b.create<circt::firrtl::ShrPrimOp>(loc, lhs, shiftAmt);
      break;
    case ShiftKind::ShrS: {
      auto lhsSigned = signExtendToWidth(b, loc, lhs, Signal(lhs, &b, loc).getWidth());
      auto shifted = b.create<circt::firrtl::ShrPrimOp>(loc, lhsSigned, shiftAmt);
      localMap[result] =
          signedValueToUIntWidth(b, loc, shifted, requiredWidth);
      return success();
    }
    }
  } else {
    // Dynamic shift - use DShlPrimOp/DShrPrimOp
    switch (kind) {
    case ShiftKind::Shl:
      // Dynamic left shift: dshl
      shiftResult = b.create<circt::firrtl::DShlPrimOp>(loc, lhs, rhs);
      break;
    case ShiftKind::ShrU:
      // Dynamic logical right shift: dshr
      shiftResult = b.create<circt::firrtl::DShrPrimOp>(loc, lhs, rhs);
      break;
    case ShiftKind::ShrS:
      // FIRRTL dshr on SInt performs arithmetic shift.
      auto lhsSigned =
          signExtendToWidth(b, loc, lhs, Signal(lhs, &b, loc).getWidth());
      auto shifted =
          b.create<circt::firrtl::DShrPrimOp>(loc, lhsSigned, rhs);
      localMap[result] =
          signedValueToUIntWidth(b, loc, shifted, requiredWidth);
      return success();
    }
  }

  // Wrap result in Signal for width adjustment
  Signal resultSignal(shiftResult, &b, loc);
  localMap[result] = fitToWidth(resultSignal, requiredWidth).getValue();
  return success();
}

LogicalResult ArithmeticOpGenerator::performBitwiseOp(mlir::OpBuilder &b, Location loc,
                                                       mlir::Value lhs, mlir::Value rhs,
                                                       mlir::Value result,
                                                       BitwiseKind kind,
                                                       llvm::DenseMap<mlir::Value, mlir::Value> &localMap) {
  // Create Signal wrappers
  Signal lhsSignal(lhs, &b, loc);
  Signal rhsSignal(rhs, &b, loc);

  // Match widths to the maximum
  padToSameWidth(lhsSignal, rhsSignal);

  // Get result width
  auto requiredWidth = cast<IntegerType>(result.getType()).getWidth();

  Signal resultSignal(lhs, &b, loc); // dummy init

  switch (kind) {
  case BitwiseKind::And:
    // Bitwise AND
    resultSignal = lhsSignal & rhsSignal;
    break;
  case BitwiseKind::Or:
    // Bitwise OR
    resultSignal = lhsSignal | rhsSignal;
    break;
  case BitwiseKind::Xor:
    // Bitwise XOR
    resultSignal = lhsSignal ^ rhsSignal;
    break;
  }

  localMap[result] = fitToWidth(resultSignal, requiredWidth).getValue();
  return success();
}

LogicalResult ArithmeticOpGenerator::performExtSIOp(mlir::OpBuilder &b, Location loc,
                                                     mlir::Value input, mlir::Value result,
                                                     llvm::DenseMap<mlir::Value, mlir::Value> &localMap) {
  // Create Signal wrapper for input
  Signal inputSignal(input, &b, loc);

  // Get input and result widths
  auto inputWidth = inputSignal.getWidth();
  auto requiredWidth = cast<IntegerType>(result.getType()).getWidth();

  // Perform sign-extension
  // In FIRRTL, we need to treat the input as signed and extend it
  // We can achieve this by getting the sign bit and using it to fill the extended bits

  if (requiredWidth <= inputWidth) {
    // No extension needed, just truncate if necessary
    Signal resultSignal = inputSignal.bits(requiredWidth - 1, 0);
    localMap[result] = resultSignal.getValue();
    return success();
  }

  // Create FIRRTL operations for sign extension
  // For proper sign extension in FIRRTL: convert to SInt, pad, then convert back to UInt

  // Create signed version type
  auto signedType = circt::firrtl::SIntType::get(b.getContext(), inputWidth);

  // Cast to SInt, pad to required width, then cast back to UInt
  auto asSInt = b.create<circt::firrtl::AsSIntPrimOp>(loc, signedType, input);
  auto paddedType = circt::firrtl::SIntType::get(b.getContext(), requiredWidth);
  auto padded = b.create<circt::firrtl::PadPrimOp>(loc, paddedType, asSInt, requiredWidth);
  auto resultType = circt::firrtl::UIntType::get(b.getContext(), requiredWidth);
  auto asUInt = b.create<circt::firrtl::AsUIntPrimOp>(loc, resultType, padded);

  localMap[result] = asUInt.getResult();
  return success();
}

} // namespace mlir
