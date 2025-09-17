#include "TOR/PassDetail.h"
#include "TOR/Passes.h"
#include "TOR/Utils.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "gen-pragma-report"

namespace {
using namespace mlir;

struct GenPragmaReportPass : GenPragmaReportBase<GenPragmaReportPass> {
  void runOnOperation() override {
    printPragmaReport(getOperation());
  }
};
} // namespace

namespace mlir {
std::unique_ptr<OperationPass<mlir::ModuleOp>> createGenPragmaReportPass() {
  return std::make_unique<GenPragmaReportPass>();
}

} // namespace mlir
