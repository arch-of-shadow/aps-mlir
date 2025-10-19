#ifndef APS_PASSES
#define APS_PASSES

include "mlir/Pass/PassBase.td"

def SCFForIndexCast : Pass<"scf-for-index-cast", "::mlir::func::FuncOp"> {
  let summary = "Convert SCF for loop bounds to index type";
  let description = [{
    This pass converts scf.for loops that use integer types (i32, i64, etc.)
    for their lower bound, upper bound, and step to use the index type instead.
    This is necessary for raising SCF loops to affine loops, which require
    index-typed bounds.
  }];
  let constructor = "mlir::createSCFForIndexCastPass()";
}

def APSMemToMemRef : Pass<"aps-mem-to-memref", "::mlir::func::FuncOp"> {
  let summary = "Convert APS memload/memstore to memref.load/memref.store";
  let description = [{
    This pass converts aps.memload and aps.memstore operations to their
    standard memref counterparts (memref.load and memref.store).
    Integer-typed indices are automatically cast to index type as required
    by memref operations.
  }];
  let constructor = "mlir::createAPSMemToMemRefPass()";
}

def MemRefToAPSMem : Pass<"affine-mem-to-aps-mem", "::mlir::func::FuncOp"> {
  let summary = "Convert affine.load/affine.store to APS memload/memstore";
  let description = [{
    This pass converts affine.load and affine.store operations to APS
    memload and memstore operations. Index-typed indices are automatically
    cast to i32 type as used by APS memory operations.
  }];
  let constructor = "mlir::createAffineMemToAPSMemPass()";
}

def MemoryMap : Pass<"memory-map", "::mlir::ModuleOp"> {
  let summary = "Generate memory map for global memrefs after array partition";
  let description = [{
    This pass generates a memory address map for all global memrefs in the module,
    including partitioned arrays. It creates an aps.memorymap operation containing
    aps.mem_entry operations for each memref group.

    For partitioned arrays, all partition banks are grouped under a single mem_entry
    with information about the partition mode (cyclic/block), number of banks, and
    base address assignment.
  }];
  let constructor = "mlir::createMemoryMapPass()";
  let dependentDialects = ["aps::APSDialect", "memref::MemRefDialect"];
}

def InferAffineMemAccess : Pass<"infer-affine-mem-access"> {
  let summary = "Infer affine maps for memref.load/store operations";
  let description = [{
    This pass analyzes memref.load and memref.store operations with computed
    indices and attempts to infer affine maps of the form a*x + b, where:
    - x is an affine.for induction variable
    - a is a constant multiplier (can be 1)
    - b is a constant offset (can be 0)

    When such a pattern is detected, the memref operation is replaced with
    an affine.load or affine.store operation using the inferred affine map.

    Example transformation:
      %idx = arith.index_cast %iv : index to i32
      %mul = arith.muli %idx, %c2 : i32
      %cast = arith.index_cast %mul : i32 to index
      %val = memref.load %array[%cast]
    becomes:
      %val = affine.load %array[%iv * 2]
  }];
  let constructor = "mlir::createInferAffineMemAccessPass()";
  let dependentDialects = ["affine::AffineDialect", "arith::ArithDialect", "memref::MemRefDialect"];
}

def APSSplitMemoryOps : Pass<"aps-split-memory-ops", "mlir::tor::DesignOp"> {
  let summary = "Split APS memory operations into request-collect pairs";
  let constructor = "mlir::createAPSSplitMemoryOpsPass()";
  let dependentDialects = ["mlir::tor::TORDialect", "aps::APSDialect"];
}

def APSMemoryPoolGen : Pass<"aps-memory-pool-gen", "::mlir::ModuleOp"> {
  let summary = "Generate scratchpad memory pool from aps.memorymap";
  let description = [{
    This pass reads aps.memorymap operation and generates a CMT2-based
    scratchpad memory pool with appropriate SRAM instances and arbitration logic.

    For each aps.mem_entry:
    - Creates SRAM instances (using Mem1r1w from ModuleLibrary)
    - Generates one RW port for burst access (indexed by address)
    - Generates one RW port per memory entry for application access
    - Implements address decoding and bank selection logic

    Memory address translation (base + bank selection) is deferred to a later pass.
  }];
  let constructor = "mlir::createAPSMemoryPoolGenPass()";
  
def APSToStandard : Pass<"aps-to-standard", "::mlir::ModuleOp"> {
  let summary = "Lower APS dialect operations to standard MLIR dialects";
  let description = [{
    This pass converts APS dialect operations to standard MLIR operations:
    - aps.readrf -> function parameters
    - aps.writerf -> function return values
    - aps.memload/memstore -> memref.load/store
    - aps.memburst operations -> removed
    - memref.get_global -> function parameters
    - Global declarations -> removed

    The transformation is designed for software-level instruction matching,
    removing hardware-specific register file and scratchpad memory semantics.
  }];
  let constructor = "mlir::createAPSToStandardPass()";
  let dependentDialects = ["::mlir::arith::ArithDialect", "::mlir::func::FuncDialect",
                          "::mlir::memref::MemRefDialect", "::mlir::scf::SCFDialect"];
}

#endif // APS_PASSES
