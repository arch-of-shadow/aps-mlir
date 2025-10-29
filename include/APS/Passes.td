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
  let summary = "Generate scratchpad memory pool and CMT2 rules from aps.memorymap";
  let description = [{
    This pass reads aps.memorymap operation and generates a CMT2-based
    scratchpad memory pool with appropriate SRAM instances and arbitration logic,
    along with rule-based main modules for TOR functions.

    For each aps.mem_entry:
    - Creates SRAM instances (using Mem1r1w from ModuleLibrary)
    - Generates one RW port for burst access (indexed by address)
    - Generates one RW port per memory entry for application access
    - Implements address decoding and bank selection logic

    For each TOR function:
    - Creates CMT2 rules based on timegraph scheduling
    - Generates burst read/write methods for memory access
    - Provides unified interface to memory pool

    This unified pass replaces the separate APSToCMT2GenPass and APSRuleGenPass.
  }];
  let constructor = "mlir::createAPSToCMT2GenPass()";
}

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

def CombExtractToArithTrunc : Pass<"comb-extract-to-arith-trunc"> {
  let summary = "Convert comb.extract operations to arith.trunci";
  let description = [{
    This pass converts all comb.extract operations to equivalent arith.trunci
    operations, optionally preceded by arith.shrui if extracting from non-zero bit.

    Transformations:
    - comb.extract %x from 0 : (iN) -> iM
      => arith.trunci %x : iN to iM

    - comb.extract %x from K : (iN) -> iM (K > 0)
      => %shifted = arith.shrui %x, K : iN
         %result = arith.trunci %shifted : iN to iM

    This is useful for lowering to standard dialects that don't have comb dialect,
    such as for CPU simulation or instruction matching with polygeist-generated MLIR.
  }];
  let constructor = "mlir::createCombExtractToArithTruncPass()";
  let dependentDialects = ["::mlir::arith::ArithDialect"];
}

def ArithSelectToSCFIf : Pass<"arith-select-to-scf-if"> {
  let summary = "Convert arith.select operations to scf.if";
  let description = [{
    This pass converts all arith.select operations to equivalent scf.if
    operations with explicit then/else branches.

    Transformation:
    - arith.select %cond, %true_val, %false_val : T
      =>
      scf.if %cond -> T {
        scf.yield %true_val : T
      } else {
        scf.yield %false_val : T
      }

    This is useful when:
    - The backend prefers explicit control flow over conditional selection
    - Pattern matching with control-flow heavy code (e.g., C code with if/else)
    - Analyzing or optimizing control flow dependencies
    - Matching with polygeist-generated MLIR from C code

    Note: This transformation may increase code size and can make certain
    optimizations (like vectorization) more difficult. Use with caution.
  }];
  let constructor = "mlir::createArithSelectToSCFIfPass()";
  let dependentDialects = ["::mlir::arith::ArithDialect", "::mlir::scf::SCFDialect"];
}

#endif // APS_PASSES
