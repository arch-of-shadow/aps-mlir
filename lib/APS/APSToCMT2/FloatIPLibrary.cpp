//===- FloatIPLibrary.cpp - Standalone HardFloat extern bindings ------------===//
#include "APS/FloatIPLibrary.h"
#include "APS/HardFloatConfig.h"
#include "llvm/ADT/StringMap.h"
namespace mlir {
using namespace circt::cmt2::ecmt2;
ExternalModule *createFloatIPModule(Circuit &circuit, unsigned operation,
                                   unsigned latency, unsigned width, unsigned predicate) {
  llvm::StringMap<int64_t> params;
  params["op"] = operation; params["latency"] = latency;
  params["width"] = width; params["pred"] = predicate;
  params["capacity"] = hardfloat::capacity(operation, width);
  if (auto *existing = circuit.hasExternalModule("HardFloat", params))
    return existing;
  auto *module = circuit.addExternalModule("HardFloat", params);
  if (!module) return nullptr;
  module->bindClock("clk", "clock").bindReset("rst", "reset")
      .bindMethod("issue", "issue_enable", "issue_ready",
                  {"operand0", "operand1", "operand2", "rounding_mode"}, {})
      .bindMethod("collect", "collect_enable", "collect_ready", {},
                  {"collect_data", "collect_flags"});
  module->addConflict("issue", "issue");
  module->addConflict("collect", "collect");
  module->addConflictFree("issue", "collect");
  return module;
}
} // namespace mlir
