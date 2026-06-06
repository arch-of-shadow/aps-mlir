#ifndef TOR_PASSES
#define TOR_PASSES

include "mlir/Pass/PassBase.td"
include "mlir/Dialect/Func/IR/FuncOps.td"
include "mlir/Dialect/MemRef/IR/MemRefOps.td"

def TORSchedule : Pass<"schedule-tor", "mlir::tor::DesignOp"> {
  let summary = "Schedule operations and set ref_* timing attributes";
  let constructor = "mlir::createTORSchedulePass()";
  let dependentDialects = ["mlir::tor::TORDialect"];
}

def TORTimeGraph : Pass<"tor-time-graph", "mlir::tor::DesignOp"> {
  let summary = "Generate time graph from ref_* attributes";
  let constructor = "mlir::createTORTimeGraphPass()";
  let dependentDialects = ["mlir::tor::TORDialect"];
}

def TORSplit : Pass<"split-schedule", "mlir::ModuleOp"> {
  let summary = "Split mixed scheduling modules";
  let constructor = "mlir::createTORSplitPass()";
  let dependentDialects = ["mlir::tor::TORDialect"];
}

def SCFToTOR : Pass<"scf-to-tor", "mlir::tor::DesignOp"> {
  let summary = "Convert SCF to unscheduled TOR";
  let constructor = "mlir::createSCFToTORPass()";
  let dependentDialects = ["mlir::tor::TORDialect"];
  let options = [
    Option<"pipeline", "pipeline", "bool", "false", "Attach pipeline attribute to inner pipeline">,
  ];
}

def ConvertInput : Pass<"convert-input", "mlir::ModuleOp"> {
  let summary = "Convert to suitable input";
  let constructor = "mlir::createConvertInputPass()";
  let dependentDialects = ["mlir::tor::TORDialect"];
  let options = [
    Option<"clock", "clock", "double", "6.0", "Clock cycle">,
    Option<"resource", "resource", "std::string", "\"./resource_dynamatic.json\"", "Resource file">,
    Option<"output_path", "output-path", "std::string", "\"output_path\"", "output path">,
  ];
}

def SCFDump : Pass<"dump-scf", "mlir::tor::DesignOp"> {
  let summary = "Dump SCF information";
  let constructor = "mlir::createSCFDumpPass()";
  let dependentDialects = ["mlir::tor::TORDialect"];
  let options = [
    Option<"json", "json", "std::string", "", "SCF json output file">
  ];
}

def SCFIterArgs : Pass<"scf-iterargs", "mlir::tor::DesignOp"> {
  let summary = "add scf.for iterargs";
  let constructor = "mlir::createSCFIterArgsPass()";
  let dependentDialects = ["mlir::tor::TORDialect"];
  let options = [
    Option<"pipeline", "pipeline", "bool", "false", "Attach pipeline attribute to inner pipeline">,
  ];
}

def TORDump : Pass<"dump-tor", "mlir::tor::DesignOp"> {
  let summary = "Dump TOR information";
  let constructor = "mlir::createTORDumpPass()";
  let dependentDialects = ["mlir::tor::TORDialect"];
  let options = [
    Option<"json", "json", "std::string", "", "TOR json output file">
  ];
}

def ConvertAffineFor: Pass<"lower-affine-for"> {
  let summary = "lower affine, and for save II attr";
  let constructor = "mlir::createAffineForLoweringPass()";
  let dependentDialects = [
    "memref::MemRefDialect",
    "scf::SCFDialect",
    "vector::VectorDialect"];
}

def RemoveNoUseReinterpretCast: Pass<"lower-reinterpret-cast"> {
  let summary = "Lower memref.reinterpret_cast ops";
  let constructor = "mlir::createReinterpretCastPass()";
  let dependentDialects = [
    "memref::MemRefDialect"];
}

def MinMaxToCmpSelect: Pass<"min-max-to-cmp-select", "mlir::ModuleOp"> {
  let summary = "min max to cmp select";
  let constructor = "mlir::createMinMaxToCmpSelectPass()";
}

def HlsUnroll: Pass<"hls-unroll", "mlir::ModuleOp"> {
  let summary = "HLS loop unroll";
  let constructor = "mlir::createHlsUnrollPass()";
  let dependentDialects = ["mlir::tor::TORDialect"];
}

def NormalizeMemrefIndices : Pass<"normalize-memref-indices", "mlir::func::FuncOp"> {
  let summary = "Normalize memref indices for affine-raise-from-memref compatibility";
  let constructor = "mlir::createNormalizeMemrefIndicesPass()";
  let dependentDialects = ["mlir::arith::ArithDialect", "memref::MemRefDialect"];
}

def NewArrayPartition : Pass<"new-array-partition", "mlir::ModuleOp"> {
  let summary = "New Array partition";
  let constructor = "mlir::createNewArrayPartitionPass()";
  let dependentDialects = ["mlir::tor::TORDialect", "memref::MemRefDialect"];
}

def ArrayOpt: Pass<"array-opt", "mlir::ModuleOp"> {
  let summary = "array opt, dimension simplify and single data simplify";
  let constructor = "mlir::createArrayOptPass()";
  let dependentDialects = ["mlir::tor::TORDialect", "memref::MemRefDialect"];
}

def CountCycles : Pass<"count-cycles", "mlir::ModuleOp"> {
  let summary = "Count cycles";
  let constructor = "mlir::createCountCyclesPass()";
  let dependentDialects = ["mlir::tor::TORDialect"];
  let options = [
    Option<"output_dir", "output-dir", "std::string", "", "Output position">
  ];
}

def TORFuse : Pass<"tor-fuse", "mlir::tor::DesignOp"> {
  let summary = "tor fuse";
  let constructor = "mlir::createTORFusePass()";
  let dependentDialects = ["mlir::tor::TORDialect"];
}

def ExpressionBalance : Pass<"expression-balance","mlir::func::FuncOp"> {
  let summary = "expression balance";
  let constructor = "mlir::createExpressionBalancePass()";
}

def LoopTripcount: Pass<"loop-tripcount", "mlir::func::FuncOp"> {
  let summary = "loop tripcount";
  let constructor = "mlir::createLoopTripcountPass()";
}

def RaiseSCFToAffine : Pass<"raise-scf-to-affine"> {
  let summary = "Raise SCF loops to Affine loops when possible";
  let constructor = "mlir::createRaiseSCFToAffinePass()";
  let dependentDialects = [
    "affine::AffineDialect",
    "scf::SCFDialect",
    "arith::ArithDialect"
  ];
}

#endif // TOR_PASSES
