#include "APS/APSDialect.h"
#include "InterfaceConfig.h"
#include "APS/Passes.h"
#include "APS/ScratchpadElision.h"
#include "TOR/Passes.h"
#include "TOR/TORDialect.h"
#include "circt/Conversion/Passes.h"
#include "circt/Dialect/Debug/DebugDialect.h"
#include "circt/Dialect/Emit/EmitDialect.h"
#include "circt/Dialect/FIRRTL/CHIRRTLDialect.h"
#include "circt/Dialect/Cmt2/Cmt2Dialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/LTL/LTLDialect.h"
#include "circt/Dialect/OM/OMDialect.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/Sim/SimDialect.h"
#include "circt/Dialect/Verif/VerifDialect.h"
#include "circt/Firtool/Firtool.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>

using namespace mlir;

#define DEBUG_TYPE "scratchpad-elision"

namespace {

llvm::cl::OptionCategory apsE2ECategory("APS end-to-end lowering options");

llvm::cl::opt<std::string> inputFile(
    "input", llvm::cl::desc("Input MLIR file path"),
    llvm::cl::value_desc("file"), llvm::cl::Required,
    llvm::cl::cat(apsE2ECategory));
llvm::cl::alias inputFileShort("i", llvm::cl::desc("Alias for --input"),
                               llvm::cl::aliasopt(inputFile),
                               llvm::cl::cat(apsE2ECategory),
                               llvm::cl::NotHidden);
llvm::cl::opt<std::string> outputFile(
    "output", llvm::cl::desc("Output file path"),
    llvm::cl::value_desc("file"), llvm::cl::init("-"),
    llvm::cl::cat(apsE2ECategory));
llvm::cl::alias outputFileShort("o", llvm::cl::desc("Alias for --output"),
                                llvm::cl::aliasopt(outputFile),
                                llvm::cl::cat(apsE2ECategory),
                                llvm::cl::NotHidden);

llvm::cl::opt<double> clockPeriod(
    "clock", llvm::cl::desc("Clock period in ns"),
    llvm::cl::init(6.0), llvm::cl::cat(apsE2ECategory));

llvm::cl::opt<std::string> interfaceConfigFile(
    "interface-config", llvm::cl::desc("APS interface/platform JSON configuration"),
    llvm::cl::init(""), llvm::cl::cat(apsE2ECategory));
llvm::cl::opt<std::string> chipyardConfigFile(
    "chipyard-config", llvm::cl::desc("Generated Chipyard YAML (default: output stem.chipyard.yaml with --interface-config)"),
    llvm::cl::init(""), llvm::cl::cat(apsE2ECategory));
llvm::cl::opt<std::string> dumpScheduledFile(
    "dump-scheduled-tor", llvm::cl::desc("Write selected scheduled TOR before temporal/CMT2 lowering"),
    llvm::cl::init(""), llvm::cl::cat(apsE2ECategory));

llvm::cl::opt<std::string> resourceFile(
    "resource", llvm::cl::desc("Path to resource.json for scheduling"),
    llvm::cl::init("examples/resource_ihp130.json"),
    llvm::cl::value_desc("file"), llvm::cl::cat(apsE2ECategory));

llvm::cl::opt<bool> printIrAfterAll(
    "print-ir-after-all", llvm::cl::desc("Print IR after each pass"),
    llvm::cl::init(false), llvm::cl::cat(apsE2ECategory));

enum class ExportFormat {
  ScheduledTor,
  Cmt2,
  Firrtl,
  Sv,
};

llvm::cl::opt<ExportFormat> exportFormat(
    "export", llvm::cl::desc("Select the output stage"),
    llvm::cl::values(
        clEnumValN(ExportFormat::ScheduledTor, "scheduled-tor", "Selected scheduled TOR MLIR"),
        clEnumValN(ExportFormat::Cmt2, "cmt2", "Full CMT2 MLIR"),
        clEnumValN(ExportFormat::Firrtl, "firrtl", "FIRRTL MLIR"),
        clEnumValN(ExportFormat::Sv, "sv", "SystemVerilog")),
    llvm::cl::init(ExportFormat::Cmt2), llvm::cl::cat(apsE2ECategory));

llvm::cl::opt<std::string> dumpCmt2File(
    "dump-cmt2",
    llvm::cl::desc("Also write full CMT2 MLIR before later export stages"),
    llvm::cl::value_desc("file"), llvm::cl::init(""),
    llvm::cl::cat(apsE2ECategory));

void registerDialects(DialectRegistry &registry) {
  registry.insert<affine::AffineDialect, LLVM::LLVMDialect,
                  memref::MemRefDialect, arith::ArithDialect, scf::SCFDialect,
                  tor::TORDialect, func::FuncDialect, math::MathDialect,
                  vector::VectorDialect, aps::APSDialect,
                  circt::chirrtl::CHIRRTLDialect,
                  circt::comb::CombDialect, circt::cmt2::Cmt2Dialect,
                  circt::debug::DebugDialect, circt::emit::EmitDialect,
                  circt::firrtl::FIRRTLDialect, circt::hw::HWDialect,
                  circt::ltl::LTLDialect, circt::om::OMDialect,
                  circt::seq::SeqDialect, circt::sim::SimDialect,
                  circt::sv::SVDialect, circt::verif::VerifDialect>();
}

void addCanonicalize(PassManager &pm) {
  pm.addPass(createCanonicalizerPass());
}

void enablePrintIrAfterAll(PassManager &pm) {
  pm.enableIRPrinting(
      /*shouldPrintBeforePass=*/[](Pass *, Operation *) { return false; },
      /*shouldPrintAfterPass=*/[](Pass *, Operation *) { return true; },
      /*printModuleScope=*/true,
      /*printAfterOnlyOnChange=*/false,
      /*printAfterOnlyOnFailure=*/false);
}

void buildFunctionalPreparation(PassManager &pm) {
  pm.addNestedPass<func::FuncOp>(createPlaceReadRFAtEntryPass());
  pm.addPass(createAPSRaiseSCFToAffinePass());
  addCanonicalize(pm);
  pm.addNestedPass<func::FuncOp>(affine::createRaiseMemrefToAffine());
  pm.addPass(createRaiseMemRefToAffinePass());
  addCanonicalize(pm);
}

void buildToScheduledTor(PassManager &pm, double clock, llvm::StringRef resource) {
  pm.addPass(createAutoBurstPartitionPass());
  pm.addPass(createAPSMemoryMapPass());
  pm.addPass(createHlsUnrollPass());
  pm.addPass(createCSEPass());
  addCanonicalize(pm);
  pm.addNestedPass<func::FuncOp>(affine::createAffineLoopNormalizePass());
  addCanonicalize(pm);
  pm.addPass(createNewArrayPartitionPass());
  addCanonicalize(pm);
  pm.addNestedPass<func::FuncOp>(createAffineMemToAPSPass());
  pm.addNestedPass<func::FuncOp>(createMemRefToAPSPass());
  pm.addPass(createAPSFunctionalToArchPass());
  pm.addNestedPass<func::FuncOp>(createPromoteSingletonMemRefToGlobalPass());
  pm.addPass(createArithMulDivToShiftPass());
  addCanonicalize(pm);

  pm.addPass(createAffineForLoweringPass());
  addCanonicalize(pm);
  pm.addPass(createArithMulDivToShiftPass());
  addCanonicalize(pm);
  pm.addNestedPass<func::FuncOp>(createExpressionBalancePass());

  pm.addPass(createConvertInputPass(clock, resource));

  addCanonicalize(pm);
  pm.addNestedPass<tor::DesignOp>(createSCFToTORPass());
  addCanonicalize(pm);
  pm.addPass(createAPSMemorySchedulePass());
  pm.addNestedPass<tor::DesignOp>(createTORSchedulePass());
}

void buildScheduledTorToCmt2(PassManager &pm) {
  pm.addNestedPass<tor::DesignOp>(createLowerAPSMemToReqCollectPass());
  pm.addNestedPass<tor::DesignOp>(createTORTimeGraphPass());
  pm.addNestedPass<tor::FuncOp>(createDuplicateMemLoadsPass());
  addCanonicalize(pm);

  pm.addPass(createAPSOptimizeStoragePass());
  pm.addPass(createAPSToCMT2Pass());
}

std::optional<int64_t> scheduledConstant(Value value) {
  if (auto op = value.getDefiningOp<arith::ConstantOp>())
    if (auto attr = dyn_cast<IntegerAttr>(op.getValue())) return attr.getInt();
  if (auto op = value.getDefiningOp<arith::IndexCastOp>()) return scheduledConstant(op.getIn());
  if (auto op = value.getDefiningOp<arith::SubIOp>()) {
    auto lhs = scheduledConstant(op.getLhs()), rhs = scheduledConstant(op.getRhs());
    if (lhs && rhs) return *lhs - *rhs;
  }
  return std::nullopt;
}

struct ScheduleCost {
  int64_t cycles = 0;
  SmallVector<int64_t> initiationIntervals;
};

// TOR reference times describe one iteration, not the whole invocation. Account
// for every static iteration; comparing only max(ref_endtime) is unsound.
std::optional<ScheduleCost> scheduledCost(tor::FuncOp function) {
  ScheduleCost cost;
  int64_t begin = INT64_MAX, end = -1;
  auto flush = [&]() {
    if (end >= begin) cost.cycles += end - begin + 1;
    begin = INT64_MAX;
    end = -1;
  };
  for (Operation &op : function.getBody().front()) {
    if (auto loop = dyn_cast<tor::ForOp>(op)) {
      flush();
      auto lower = scheduledConstant(loop.getLowerBound());
      auto upper = scheduledConstant(loop.getUpperBound());
      auto step = scheduledConstant(loop.getStep());
      auto first = op.getAttrOfType<IntegerAttr>("ref_starttime");
      auto last = op.getAttrOfType<IntegerAttr>("ref_endtime");
      auto pipeline = op.getAttrOfType<IntegerAttr>("pipeline");
      auto ii = op.getAttrOfType<IntegerAttr>("II");
      bool nested = false;
      loop.walk([&](Operation *child) {
        if (child != loop && child->getNumRegions()) nested = true;
      });
      if (!lower || !upper || !step || *step <= 0 || *upper < *lower ||
          !first || !last || last.getInt() < first.getInt() || nested ||
          !pipeline || !pipeline.getInt() || !ii || ii.getInt() <= 0)
        return std::nullopt;
      // TOR's upper bound is inclusive (SCFToTOR subtracts the original step).
      int64_t trips = (*upper - *lower) / *step + 1;
      int64_t span = last.getInt() - first.getInt() + 1;
      if (trips - 1 > (INT64_MAX - cost.cycles - span) / ii.getInt()) return std::nullopt;
      cost.cycles += span + (trips - 1) * ii.getInt();
      cost.initiationIntervals.push_back(ii.getInt());
    } else {
      if (op.getNumRegions()) return std::nullopt;
      auto first = op.getAttrOfType<IntegerAttr>("ref_starttime");
      auto last = op.getAttrOfType<IntegerAttr>("ref_endtime");
      if (first && last && first.getInt() >= 0 && last.getInt() >= first.getInt()) {
        begin = std::min(begin, first.getInt());
        end = std::max(end, last.getInt());
      }
    }
  }
  flush();
  return cost;
}

std::optional<ScheduleCost> findCost(ModuleOp module, StringRef name) {
  tor::FuncOp function;
  module.walk([&](tor::FuncOp op) { if (op.getName() == name) function = op; });
  if (!function) return std::nullopt;
  return scheduledCost(function);
}

bool respectsScalarInflight(ModuleOp module, StringRef buffer) {
  bool legal = true;
  module.walk([&](aps::LoadBy load) {
    auto source = load->getAttrOfType<StringAttr>("elided_scratchpad");
    if (!source || source.getValue() != buffer) return;
    auto loop = load->getParentOfType<tor::ForOp>();
    auto ii = loop ? loop->getAttrOfType<IntegerAttr>("II") : IntegerAttr{};
    auto latency = load->getAttrOfType<IntegerAttr>("memory_latency");
    aps::MemItfc interface;
    module.walk([&](aps::MemItfc model) {
      if (model.getSymName() == load.getItfcAttr().getValue()) interface = model;
    });
    auto inflight = interface ? interface->getAttrOfType<IntegerAttr>("inflight") : IntegerAttr{};
    // Same issue/retirement convention as the memory timing recurrence. One
    // new scalar transaction per iteration must fit the finite response slots.
    if (!ii || !latency || !inflight || inflight.getInt() <= 0 ||
        ii.getInt() < latency.getInt() / inflight.getInt() + 1) legal = false;
  });
  return legal;
}

LogicalResult runToScheduledTor(ModuleOp module, bool print) {
  PassManager pm = PassManager::on<ModuleOp>(module.getContext());
  if (print) enablePrintIrAfterAll(pm);
  buildToScheduledTor(pm, clockPeriod, resourceFile);
  return pm.run(module);
}

LogicalResult chooseScratchpadElisions(OwningOpRef<ModuleOp> &functional,
                                       OwningOpRef<ModuleOp> &scheduled) {
  scheduled = cast<ModuleOp>(functional->clone());
  if (failed(runToScheduledTor(*scheduled, printIrAfterAll))) return failure();
  auto candidates = analyzeScratchpadElision(*functional);
  for (auto original : candidates) {
    // Recheck after accepted transformations; a competing direct stream can
    // make a previously eligible buffer unsuitable.
    auto current = analyzeScratchpadElision(*functional);
    auto found = llvm::find_if(current, [&](const auto &c) { return c.buffer == original.buffer; });
    if (found == current.end()) continue;
    auto candidate = *found;
    std::string reason = candidate.reason;
    bool accepted = false;
    std::optional<ScheduleCost> baseline, trial;
    if (candidate.eligible) {
      baseline = findCost(*scheduled, candidate.function);
      if (!baseline) reason = "unsupported baseline schedule; cannot prove no latency increase";
      else {
        OwningOpRef<ModuleOp> trialFunctional(cast<ModuleOp>(functional->clone()));
        if (failed(rewriteScratchpadElision(*trialFunctional, candidate.buffer)))
          reason = "functional legality recheck failed";
        else {
          OwningOpRef<ModuleOp> trialScheduled(cast<ModuleOp>(trialFunctional->clone()));
          std::string trialDiagnostics;
          LogicalResult trialResult = failure();
          {
            ScopedDiagnosticHandler capture(functional->getContext(), [&](Diagnostic &diagnostic) {
              llvm::raw_string_ostream stream(trialDiagnostics);
              diagnostic.print(stream);
              stream << '\n';
              return success();
            });
            trialResult = runToScheduledTor(*trialScheduled, false);
          }
          if (failed(trialResult)) {
            reason = "trial lowering/scheduling failed; retained baseline";
            LLVM_DEBUG(llvm::dbgs() << trialDiagnostics);
          } else {
            trial = findCost(*trialScheduled, candidate.function);
            if (!trial) reason = "trial lost a supported pipelined schedule";
            else {
              bool noRegression = trial->cycles <= baseline->cycles &&
                  trial->initiationIntervals.size() == baseline->initiationIntervals.size() &&
                  respectsScalarInflight(*trialScheduled, candidate.buffer);
              for (unsigned i = 0; noRegression && i < trial->initiationIntervals.size(); ++i)
                noRegression &= trial->initiationIntervals[i] <= baseline->initiationIntervals[i];
              if (noRegression) {
                accepted = true;
                reason = "rescheduled invocation latency and every loop II are non-increasing";
                functional = std::move(trialFunctional);
                scheduled = std::move(trialScheduled);
              } else reason = "trial increases latency/II or exceeds scalar inflight capacity; retained baseline";
            }
          }
        }
      }
    }
    LLVM_DEBUG({
      llvm::dbgs() << "[scratchpad-elision] @" << candidate.buffer
                   << " eligible=" << candidate.eligible << " accepted=" << accepted;
      auto printCost = [](StringRef label, const std::optional<ScheduleCost> &cost) {
        if (!cost) return;
        llvm::dbgs() << " " << label << "_cycles=" << cost->cycles
                     << " " << label << "_ii=[";
        llvm::interleaveComma(cost->initiationIntervals, llvm::dbgs());
        llvm::dbgs() << "]";
      };
      printCost("baseline", baseline);
      printCost("trial", trial);
      llvm::dbgs() << ": " << reason << '\n';
    });
  }
  return success();
}

bool fileExists(llvm::StringRef path) {
  return std::filesystem::exists(std::filesystem::path(path.str()));
}

LogicalResult writeModuleToFile(ModuleOp module, llvm::StringRef path) {
  std::string errorMessage;
  std::unique_ptr<llvm::ToolOutputFile> output =
      openOutputFile(path, &errorMessage);
  if (!output) {
    llvm::errs() << "[ERROR] failed to open output file " << path
                 << ": " << errorMessage << "\n";
    return failure();
  }

  module->print(output->os());
  output->os() << '\n';
  output->keep();
  return success();
}

LogicalResult lowerCmt2ToFirrtl(ModuleOp module) {
  PassManager pm = PassManager::on<ModuleOp>(module.getContext());
  pm.addPass(circt::createLowerCmt2ToFIRRTLPass());
  return pm.run(module.getOperation());
}

LogicalResult exportSystemVerilog(ModuleOp module, llvm::StringRef outputPath,
                                  llvm::StringRef inputPath) {
  std::string errorMessage;
  std::unique_ptr<llvm::ToolOutputFile> output =
      openOutputFile(outputPath, &errorMessage);
  if (!output) {
    llvm::errs() << "[ERROR] failed to open output file " << outputPath
                 << ": " << errorMessage << "\n";
    return failure();
  }

  circt::firtool::FirtoolOptions options;
  options.setOutputFilename(outputPath);

  PassManager pm = PassManager::on<ModuleOp>(module.getContext());
  if (failed(circt::firtool::populatePreprocessTransforms(pm, options)) ||
      failed(circt::firtool::populateCHIRRTLToLowFIRRTL(pm, options)) ||
      failed(circt::firtool::populateLowFIRRTLToHW(pm, options, inputPath)) ||
      failed(circt::firtool::populateHWToSV(pm, options)) ||
      failed(circt::firtool::populateExportVerilog(pm, options, output->os())))
    return failure();

  if (failed(pm.run(module.getOperation())))
    return failure();

  output->keep();
  return success();
}

} // namespace

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);
  llvm::cl::HideUnrelatedOptions(apsE2ECategory);
  llvm::cl::ParseCommandLineOptions(
      argc, argv, "APS end-to-end MLIR-to-CMT2 lowering driver\n");

  std::string resolvedInput = inputFile.getValue();
  std::string resolvedOutput = outputFile.getValue();
  std::string resolvedChipyard = chipyardConfigFile.getValue();
  if ((!interfaceConfigFile.empty() || !resolvedChipyard.empty()) &&
      (exportFormat != ExportFormat::Sv || resolvedOutput == "-")) {
    llvm::errs() << "[ERROR] interface/Chipyard configuration requires --export sv and a named output file\n";
    return EXIT_FAILURE;
  }
  if (!resolvedChipyard.empty() && interfaceConfigFile.empty()) {
    llvm::errs() << "[ERROR] --chipyard-config requires --interface-config\n";
    return EXIT_FAILURE;
  }
  if (!interfaceConfigFile.empty() && resolvedChipyard.empty()) {
    auto path = std::filesystem::path(resolvedOutput);
    path.replace_extension(".chipyard.yaml");
    resolvedChipyard = path.string();
  }
  if (!resolvedChipyard.empty()) {
    auto target = std::filesystem::absolute(resolvedChipyard).lexically_normal();
    for (auto source : {resolvedInput, resolvedOutput, interfaceConfigFile.getValue(), resourceFile.getValue(), dumpCmt2File.getValue()})
      if (!source.empty() && source != "-" && target == std::filesystem::absolute(source).lexically_normal()) {
        llvm::errs() << "[ERROR] Chipyard config output must not overwrite input, RTL, or another output\n";
        return EXIT_FAILURE;
      }
  }
  // Debug dumps must never clobber inputs or another artifact.
  if (!dumpScheduledFile.empty()) {
    auto target = std::filesystem::absolute(dumpScheduledFile.getValue()).lexically_normal();
    for (auto source : {resolvedInput, resolvedOutput, interfaceConfigFile.getValue(), resourceFile.getValue(),
                        dumpCmt2File.getValue(), resolvedChipyard})
      if (!source.empty() && source != "-" && target == std::filesystem::absolute(source).lexically_normal()) {
        llvm::errs() << "[ERROR] scheduled dump must not overwrite an input or another output\n";
        return EXIT_FAILURE;
      }
  }
  if (!fileExists(resolvedInput)) {
    llvm::errs() << "[ERROR] Input file " << resolvedInput
                 << " does not exist\n";
    return EXIT_FAILURE;
  }
  if (!fileExists(resourceFile)) {
    llvm::errs() << "[ERROR] Resource file " << resourceFile
                 << " does not exist\n";
    return EXIT_FAILURE;
  }

  DialectRegistry registry;
  registerDialects(registry);
  MLIRContext context(registry);
  if (printIrAfterAll)
    context.disableMultithreading();
  context.allowUnregisteredDialects();
  context.loadAllAvailableDialects();

  ParserConfig parserConfig(&context);
  OwningOpRef<ModuleOp> module =
      parseSourceFile<ModuleOp>(resolvedInput, parserConfig);
  if (!module) {
    llvm::errs() << "[ERROR] failed to parse input MLIR\n";
    return EXIT_FAILURE;
  }

  apsconfig::Config interfaceConfig;
  if (!interfaceConfigFile.empty()) {
    try {
      interfaceConfig.load(interfaceConfigFile);
      interfaceConfig.apply(*module);
    } catch (const std::exception &error) {
      llvm::errs() << "[ERROR] interface configuration: " << error.what() << "\n";
      return EXIT_FAILURE;
    }
  }

  PassManager pm = PassManager::on<ModuleOp>(&context);
  if (printIrAfterAll)
    enablePrintIrAfterAll(pm);

  buildFunctionalPreparation(pm);

  if (failed(pm.run(module.get().getOperation()))) {
    llvm::errs() << "[ERROR] aps-e2e pipeline failed\n";
    return EXIT_FAILURE;
  }

  OwningOpRef<ModuleOp> scheduled;
  if (failed(chooseScratchpadElisions(module, scheduled))) {
    llvm::errs() << "[ERROR] baseline scheduling failed\n";
    return EXIT_FAILURE;
  }
  if (!dumpScheduledFile.empty() && failed(writeModuleToFile(*scheduled, dumpScheduledFile)))
    return EXIT_FAILURE;
  if (exportFormat == ExportFormat::ScheduledTor)
    return succeeded(writeModuleToFile(*scheduled, resolvedOutput)) ? EXIT_SUCCESS : EXIT_FAILURE;
  module = std::move(scheduled);
  PassManager finish = PassManager::on<ModuleOp>(&context);
  if (printIrAfterAll) enablePrintIrAfterAll(finish);
  buildScheduledTorToCmt2(finish);
  if (failed(finish.run(*module))) {
    llvm::errs() << "[ERROR] temporal/CMT2 lowering failed\n";
    return EXIT_FAILURE;
  }

  if (!dumpCmt2File.empty() &&
      failed(writeModuleToFile(*module, dumpCmt2File)))
    return EXIT_FAILURE;

  switch (exportFormat) {
  case ExportFormat::ScheduledTor:
    llvm_unreachable("scheduled TOR was already exported");
  case ExportFormat::Cmt2:
    if (failed(writeModuleToFile(*module, resolvedOutput)))
      return EXIT_FAILURE;
    break;
  case ExportFormat::Firrtl:
    if (failed(lowerCmt2ToFirrtl(*module))) {
      llvm::errs() << "[ERROR] lower-cmt2-to-firrtl failed\n";
      return EXIT_FAILURE;
    }
    if (failed(writeModuleToFile(*module, resolvedOutput)))
      return EXIT_FAILURE;
    break;
  case ExportFormat::Sv:
    if (failed(lowerCmt2ToFirrtl(*module))) {
      llvm::errs() << "[ERROR] lower-cmt2-to-firrtl failed\n";
      return EXIT_FAILURE;
    }
    if (failed(exportSystemVerilog(*module, resolvedOutput, resolvedInput))) {
      llvm::errs() << "[ERROR] native SystemVerilog export failed\n";
      return EXIT_FAILURE;
    }
    break;
  }

  if (!resolvedChipyard.empty() && failed(interfaceConfig.emit(resolvedChipyard, resolvedOutput)))
    return EXIT_FAILURE;
  return EXIT_SUCCESS;
}
