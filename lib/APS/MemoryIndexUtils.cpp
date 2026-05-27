#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"

namespace aps {

mlir::SmallVector<mlir::Value>
castMemoryIndicesToI32(mlir::OpBuilder &builder, mlir::Location loc,
                       mlir::ValueRange indices) {
  mlir::SmallVector<mlir::Value> castedIndices;
  auto i32Type = builder.getI32Type();

  for (mlir::Value index : indices) {
    if (index.getType().isIndex()) {
      castedIndices.push_back(
          builder.create<mlir::arith::IndexCastOp>(loc, i32Type, index));
      continue;
    }
    castedIndices.push_back(index);
  }

  return castedIndices;
}

} // namespace aps
