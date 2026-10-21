//===- HardwareValueUtils.h - Scalar physical types and constants
//----------===//
#ifndef APS_HARDWAREVALUEUTILS_H
#define APS_HARDWAREVALUEUTILS_H

#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include <optional>

namespace mlir {

// Semantic f32 remains f32 in TOR. Only its hardware storage is UInt<32>.
// Other floating formats deliberately fail until their IP/storage is supported.
inline std::optional<unsigned> getHardwareBitWidth(Type type) {
  if (auto integer = dyn_cast<IntegerType>(type))
    return integer.getWidth();
  if (type.isF32())
    return 32;
  if (type.isBF16())
    return 16;
  return std::nullopt;
}

inline FailureOr<Value>
materializeHardwareConstant(Attribute attr, OpBuilder &builder, Location loc) {
  llvm::APInt bits;
  if (auto integer = dyn_cast<IntegerAttr>(attr)) {
    bits = integer.getValue();
  } else if (auto floating = dyn_cast<FloatAttr>(attr)) {
    if (!floating.getType().isF32() && !floating.getType().isBF16())
      return failure();
    bits = floating.getValue().bitcastToAPInt();
  } else {
    return failure();
  }
  auto type =
      circt::firrtl::UIntType::get(builder.getContext(), bits.getBitWidth());
  return builder.create<circt::firrtl::ConstantOp>(loc, type, bits).getResult();
}

} // namespace mlir
#endif
