#include "APS/APSDialect.h"
#include "APS/Passes.h"
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
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>

using namespace mlir;

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

llvm::cl::opt<std::string> resourceFile(
    "resource", llvm::cl::desc("Path to resource.json for scheduling"),
    llvm::cl::init("examples/resource_ihp130.json"),
    llvm::cl::value_desc("file"), llvm::cl::cat(apsE2ECategory));

llvm::cl::opt<bool> printIrAfterAll(
    "print-ir-after-all", llvm::cl::desc("Print IR after each pass"),
    llvm::cl::init(false), llvm::cl::cat(apsE2ECategory));

enum class ExportFormat {
  Cmt2,
  Firrtl,
  Sv,
};

llvm::cl::opt<ExportFormat> exportFormat(
    "export", llvm::cl::desc("Select the output stage"),
    llvm::cl::values(
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

void buildApsE2EPipeline(PassManager &pm, double clock,
                         llvm::StringRef resource) {
  pm.addNestedPass<func::FuncOp>(createPlaceReadRFAtEntryPass());
  pm.addPass(createAutoBurstPartitionPass());
  pm.addPass(createAPSMemoryMapPass());
  pm.addPass(createAPSRaiseSCFToAffinePass());
  addCanonicalize(pm);
  pm.addNestedPass<func::FuncOp>(affine::createRaiseMemrefToAffine());
  pm.addPass(createRaiseMemRefToAffinePass());
  addCanonicalize(pm);
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
  pm.addNestedPass<tor::DesignOp>(createTORSchedulePass());
  pm.addNestedPass<tor::DesignOp>(createLowerAPSMemToReqCollectPass());
  pm.addNestedPass<tor::DesignOp>(createTORTimeGraphPass());
  pm.addNestedPass<tor::FuncOp>(createDuplicateMemLoadsPass());
  addCanonicalize(pm);

  pm.addPass(createAPSToCMT2Pass());
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
  if (printIrAfterAll)
    enablePrintIrAfterAll(pm);
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
  if (printIrAfterAll)
    enablePrintIrAfterAll(pm);
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

  PassManager pm = PassManager::on<ModuleOp>(&context);
  if (printIrAfterAll)
    enablePrintIrAfterAll(pm);

  buildApsE2EPipeline(pm, clockPeriod, resourceFile);

  if (failed(pm.run(module.get().getOperation()))) {
    llvm::errs() << "[ERROR] aps-e2e pipeline failed\n";
    return EXIT_FAILURE;
  }

  if (!dumpCmt2File.empty() &&
      failed(writeModuleToFile(*module, dumpCmt2File)))
    return EXIT_FAILURE;

  switch (exportFormat) {
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

  return EXIT_SUCCESS;
}
