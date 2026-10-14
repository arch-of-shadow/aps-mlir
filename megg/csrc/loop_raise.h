#pragma once
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"

namespace megg {

mlir::affine::AffineForOp promoteSCFToAffine(mlir::scf::ForOp forOp, mlir::Operation *&targetFor);

bool lowerAffineToSCF(mlir::affine::AffineForOp forOp);

bool lowerAllAffineToSCF(mlir::ModuleOp module);

} // namespace megg
