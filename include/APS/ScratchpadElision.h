#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallVector.h"
#include <string>

namespace mlir {
struct ScratchpadElisionCandidate {
  std::string buffer;
  std::string function;
  std::string reason;
  bool eligible = false;
};

/// Functional-level safety/cache screen, before allocation and unrolling.
llvm::SmallVector<ScratchpadElisionCandidate>
analyzeScratchpadElision(ModuleOp module);

/// Rechecks legality and rewrites one candidate on a private trial clone.
/// The caller MUST reschedule and compare against the unchanged baseline before
/// committing the clone. This is intentionally not an unconditional opt pass.
LogicalResult rewriteScratchpadElision(ModuleOp module, llvm::StringRef buffer);
} // namespace mlir
