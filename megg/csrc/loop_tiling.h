#pragma once
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir_utils.h"
#include <optional>
#include <string>
#include <vector>

namespace megg {

struct TileConfig {
  mlir::Operation *loopOp;
  uint64_t tileSize;

  TileConfig(mlir::Operation *op, uint64_t size)
      : loopOp(op), tileSize(size) {}
};

class LoopTiler : public CustomPass {
public:
  explicit LoopTiler(mlir::Operation *targetFunction, uint64_t defaultTileSize)
      : CustomPass("LoopTiling"), tileSize_(defaultTileSize),
        targetFunction_(targetFunction), tileConfigs_(nullptr, 0) {}

  bool run(mlir::ModuleOp module) override;

  void addTileConfig(mlir::Operation *loopOp, uint64_t tileSize);

  void setDefaultTileSize(uint64_t size) {
    if (size > 0)
      tileSize_ = size;
  }

  void setTargetFunction(mlir::Operation *funcOp) {
    targetFunction_ = funcOp;
  }

  uint64_t getDefaultTileSize() const { return tileSize_; }
  mlir::Operation *getTargetFunction() const { return targetFunction_; }
  const TileConfig &getTileConfigs() const { return tileConfigs_; }

private:
  bool shouldProcessFunction(mlir::func::FuncOp funcOp);
  std::optional<uint64_t> getTileSizeForLoop(mlir::affine::AffineForOp forOp);

  bool applyLoopTiling(mlir::affine::AffineForOp forOp, uint64_t tileSize);

  bool tileConfiguredLoops(mlir::func::FuncOp funcOp);

  uint64_t tileSize_;
  mlir::Operation *targetFunction_;
  TileConfig tileConfigs_;
};

} // namespace megg
