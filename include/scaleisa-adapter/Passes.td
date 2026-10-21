include "mlir/Pass/PassBase.td"

def ScaleISAConvert : Pass<"scaleisa-convert", "mlir::ModuleOp"> {
  let summary = "Normalize ScaleISA candidates to APS-compatible computation";
  let constructor = "mlir::scaleisa_adapter::createConvertPass()";
}
def ScaleISASplit : Pass<"scaleisa-split", "mlir::ModuleOp"> {
  let summary = "Partition scalar candidate graphs into operand-feasible instructions";
  let constructor = "mlir::scaleisa_adapter::createSplitPass()";
  let options = [Option<"maxExtraCSRs", "max-extra-csrs", "unsigned", "0",
                        "Maximum extra RV32 CSR operands per instruction">];
}
def ScaleISAInsertPragmas : Pass<"scaleisa-insert-pragmas", "mlir::ModuleOp"> {
  let summary = "Insert loop implementation pragmas after instruction splitting";
  let constructor = "mlir::scaleisa_adapter::createInsertPragmasPass()";
  let options = [
    Option<"mode", "mode", "std::string", "\"none\"", "none, pipeline, or unroll">,
    Option<"initiationInterval", "ii", "unsigned", "0", "Requested II; zero lets HLS choose">,
    Option<"maxExpandedOps", "max-expanded-ops", "unsigned", "100000", "Full-unroll operation limit">
  ];
}
def ScaleISAMaterializeAPS : Pass<"scaleisa-materialize-aps", "mlir::ModuleOp"> {
  let summary = "Export RV32 instructions with register/CSR operands and functional CPU-memory accesses";
  let constructor = "mlir::scaleisa_adapter::createMaterializeAPSPass()";
  let options = [
    Option<"csrBase", "csr-base", "unsigned", "2049", "First dedicated CSR address">,
    Option<"opcode", "opcode", "unsigned", "43", "RISC-V custom opcode">,
    Option<"funct7Base", "funct7-base", "unsigned", "0", "First funct7 encoding">
  ];
}
