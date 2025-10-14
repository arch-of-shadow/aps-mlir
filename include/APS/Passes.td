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

#endif // APS_PASSES
