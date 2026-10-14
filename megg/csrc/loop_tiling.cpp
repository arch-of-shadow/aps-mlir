#include "loop_tiling.h"
#include "loop_raise.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Visitors.h"
#include <cassert>
#include <iostream>
#include <optional>
#include <vector>

namespace megg {

bool LoopTiler::run(mlir::ModuleOp module) {
  if (tileSize_ <= 0) {
    std::cerr << "LoopTiler: Invalid default tile size: " << tileSize_ << std::endl;
    return false;
  }

  bool modified = false;

  module.walk([&](mlir::func::FuncOp funcOp) -> mlir::WalkResult {
    if (!shouldProcessFunction(funcOp)) {
      return mlir::WalkResult::advance();
    }

    std::string funcName = funcOp.getSymName().str();

    funcOp.walk([&](mlir::scf::ForOp scfFor) -> mlir::WalkResult {
      promoteSCFToAffine(scfFor, tileConfigs_.loopOp);
      return mlir::WalkResult::advance();
    });

    if (tileConfigs_.loopOp != nullptr) {
      if (tileConfiguredLoops(funcOp)) {
        modified = true;
      }
    }

    return mlir::WalkResult::advance();
  });

  lowerAllAffineToSCF(module);

  return modified;
}

void LoopTiler::addTileConfig(mlir::Operation *loopOp, uint64_t tileSize) {
  if (tileSize <= 0) {
    std::cerr << "LoopTiler: Invalid tile size for loop: " << tileSize << std::endl;
    return;
  }
  tileConfigs_ = {loopOp, tileSize};
}

bool LoopTiler::shouldProcessFunction(mlir::func::FuncOp funcOp) {
  if (targetFunction_ == nullptr) {
    return true;
  }

  return funcOp.getOperation() == targetFunction_;
}

std::optional<uint64_t>
LoopTiler::getTileSizeForLoop(mlir::affine::AffineForOp forOp) {
  mlir::Operation *forOpPtr = forOp.getOperation();

  if (tileConfigs_.loopOp != nullptr && tileConfigs_.loopOp == forOpPtr) {
    return tileConfigs_.tileSize;
  }

  return std::nullopt;
}

bool LoopTiler::applyLoopTiling(mlir::affine::AffineForOp forOp,
                                uint64_t tileSize) {
  if (!forOp) {
    std::cerr << "LoopTiler: Invalid loop operation" << std::endl;
    return false;
  }

  // Check if loop has constant bounds (required for tiling)
  if (!forOp.hasConstantBounds()) {
    std::cerr << "LoopTiler: Cannot tile loop - non-constant bounds" << std::endl;
    return false;
  }

  llvm::SmallVector<mlir::affine::AffineForOp, 6> tiledLoops;
  llvm::SmallVector<mlir::affine::AffineForOp, 1> loopBand = {forOp};

  if (mlir::failed(mlir::affine::tilePerfectlyNested(
          loopBand, {static_cast<unsigned>(tileSize)}, &tiledLoops))) {
    std::cerr << "LoopTiler: Failed to tile loop" << std::endl;
    return false;
  }

  assert(tiledLoops.size() == 2);
  return true;
}

bool LoopTiler::tileConfiguredLoops(mlir::func::FuncOp funcOp) {
  bool modified = false;

  funcOp.walk([&](mlir::affine::AffineForOp forOp) {
    auto tileSizeOpt = getTileSizeForLoop(forOp);
    if (tileSizeOpt.has_value()) {
      if (applyLoopTiling(forOp, tileSizeOpt.value())) {
        modified = true;
      }
    }
  });

  return modified;
}

} // namespace megg
