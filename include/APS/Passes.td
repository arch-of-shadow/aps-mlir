#ifndef APS_PASSES
#define APS_PASSES

include "mlir/Pass/PassBase.td"

def NormalizeSCFForIndices : Pass<"normalize-scf-for-indices", "::mlir::func::FuncOp"> {
  let summary = "Convert SCF for loop bounds to index type";
  let description = [{
    This pass converts scf.for loops that use integer types (i32, i64, etc.)
    for their lower bound, upper bound, and step to use the index type instead.
    This is necessary for raising SCF loops to affine loops, which require
    index-typed bounds.
  }];
  let constructor = "mlir::createNormalizeSCFForIndicesPass()";
}

def APSRaiseSCFToAffine : Pass<"aps-raise-scf-to-affine"> {
  let summary = "Speculatively raise eligible SCF loops to affine loops";
  let description = [{
    This pass raises only loops that it can locally normalize and convert to
    affine.for. Integer-typed SCF loop bounds are cast to index only for loops
    that are actually converted, so loops that cannot enter the affine path keep
    their original integer IV/bound/step types.
  }];
  let constructor = "mlir::createAPSRaiseSCFToAffinePass()";
  let dependentDialects = [
    "affine::AffineDialect",
    "arith::ArithDialect",
    "memref::MemRefDialect",
    "scf::SCFDialect"
  ];
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

def AffineMemToAPS : Pass<"affine-mem-to-aps", "::mlir::func::FuncOp"> {
  let summary = "Convert affine.load/affine.store to APS memload/memstore";
  let description = [{
    This pass converts affine.load and affine.store operations to APS
    memload and memstore operations. Index-typed indices are automatically
    cast to i32 type as used by APS memory operations.
  }];
  let constructor = "mlir::createAffineMemToAPSPass()";
  let dependentDialects = ["aps::APSDialect", "::mlir::arith::ArithDialect"];
}

def MemRefToAPS : Pass<"memref-to-aps", "::mlir::func::FuncOp"> {
  let summary = "Convert memref.load/memref.store to APS memload/memstore";
  let description = [{
    This pass converts memref.load and memref.store operations to APS
    memload and memstore operations. Index-typed indices are automatically
    cast to i32 type as used by APS memory operations.
  }];
  let constructor = "mlir::createMemRefToAPSPass()";
  let dependentDialects = ["aps::APSDialect", "::mlir::arith::ArithDialect"];
}

def AutoBurstPartition : Pass<"auto-burst-partition", "::mlir::ModuleOp"> {
  let summary = "Add default cyclic partitioning for globals used by APS burst transfers";
  let description = [{
    This pass inspects aps.memburstload and aps.memburststore operations and
    finds the global memrefs they transfer to or from. If a referenced global
    has no user-provided partition attributes, the pass adds a cyclic
    partition on dimension 0 with factor 64 / element_bitwidth. This gives the
    later array partition pass enough banks to fill a 64-bit burst beat.

    User-provided partition_dim_array, partition_factor_array, or
    partition_cyclic_array attributes are treated as explicit decisions and
    are never overwritten.
  }];
  let constructor = "mlir::createAutoBurstPartitionPass()";
  let dependentDialects = ["aps::APSDialect", "memref::MemRefDialect"];
}

def APSMemoryMap : Pass<"aps-memory-map", "::mlir::ModuleOp"> {
  let summary = "Generate an APS memory map from global memrefs";
  let description = [{
    This pass seeds the APS memory address map from the global memrefs currently
    present in the module. It creates an aps.memorymap operation containing
    aps.mem_entry operations for each source memory group.

    At initial map construction time, each source memory normally has one bank.
    Partition metadata such as cyclic/block mode is preserved on the entry so
    later partitioning passes can rewrite the entry with the concrete bank list.
  }];
  let constructor = "mlir::createAPSMemoryMapPass()";
  let dependentDialects = ["aps::APSDialect", "memref::MemRefDialect"];
}

def RaiseMemRefToAffine : Pass<"raise-memref-to-affine"> {
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
  let constructor = "mlir::createRaiseMemRefToAffinePass()";
  let dependentDialects = ["affine::AffineDialect", "arith::ArithDialect", "memref::MemRefDialect"];
}

def DuplicateMemLoads : Pass<"duplicate-memloads", "mlir::tor::FuncOp"> {
  let summary = "Duplicate memory loads to reduce FIFO usage";
  let description = [{
    This pass duplicates aps.memload operations when they are used at different
    cycles than when they are loaded. If there is no memory conflict at the
    user's cycle, the memload is duplicated at that cycle to eliminate the need
    for a FIFO buffer to hold the value across cycles.

    For example, if a memload executes at cycle 7 but is used by an operation at
    cycle 10, and there's no other access to that memory at cycle 10, the pass
    will duplicate the memload at cycle 10, saving a FIFO.
  }];
  let constructor = "mlir::createDuplicateMemLoadsPass()";
  let dependentDialects = ["mlir::tor::TORDialect", "aps::APSDialect"];
}

def LowerAPSMemToReqCollect : Pass<"lower-aps-mem-to-req-collect", "mlir::tor::DesignOp"> {
  let summary = "Split APS memory operations into request-collect pairs";
  let constructor = "mlir::createLowerAPSMemToReqCollectPass()";
  let dependentDialects = ["mlir::tor::TORDialect", "aps::APSDialect"];
}

def APSToCMT2 : Pass<"aps-to-cmt2", "::mlir::ModuleOp"> {
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

    This unified pass replaces the earlier split memory-pool and rule-generation passes.
  }];
  let constructor = "mlir::createAPSToCMT2Pass()";
}

def ArithMulDivToShift : Pass<"arith-muldiv-to-shift"> {
  let summary = "Convert arith mul/div by power of 2 to shift operations";
  let description = [{
    This pass converts arith.muli and arith.divui/divsi operations to shift
    operations when the second operand is a constant power of 2.

    Transformations:
    - arith.muli %x, 2^n  => arith.shli %x, n
    - arith.divui %x, 2^n => arith.shrui %x, n
    - arith.divsi %x, 2^n => arith.shrsi with bias correction for negative values

    For signed division (arith.divsi), the transformation adds a bias for
    negative values to ensure correct truncation toward zero (matching C
    semantics), rather than floor division which arithmetic shift alone provides.

    The bias correction is:
      sign = shrsi(x, bitwidth-1)   // -1 if negative, 0 if positive
      bias = andi(sign, 2^n - 1)    // (2^n - 1) if negative, 0 if positive
      result = shrsi(addi(x, bias), n)

    This optimization is useful for:
    - Reducing hardware complexity (shift is simpler than multiply/divide)
    - Improving performance on targets where shift is faster than multiply/divide
    - Matching patterns in instruction selection

    Example:
      %c8 = arith.constant 8 : i32
      %result = arith.muli %x, %c8 : i32
    becomes:
      %c3 = arith.constant 3 : i32
      %result = arith.shli %x, %c3 : i32
  }];
  let constructor = "mlir::createArithMulDivToShiftPass()";
  let dependentDialects = ["::mlir::arith::ArithDialect"];
}

def PlaceReadRFAtEntry : Pass<"place-readrf-at-entry", "::mlir::func::FuncOp"> {
  let summary = "Place all aps.readrf operations at the function entry block";
  let description = [{
    This pass moves all aps.readrf operations to the entry block of their
    containing function. This ensures they can be scheduled at cycle 0,
    since readrf operations only depend on function arguments which are
    available at function entry.

    This transformation is safe because:
    - aps.readrf only reads from function arguments (register file indices)
    - Function arguments are available throughout the function
    - There are no side effects or dependencies on control flow

    This pass should run before SCF to TOR conversion and scheduling to avoid
    control-flow constraints preventing readrf from being scheduled early.

    Example transformation:
      func.func @foo(%arg0: i5, %arg1: i5) {
        // some operations
        scf.for ... {
          // loop body
        }
        %0 = aps.readrf %arg1 : i5 -> i32  // readrf after loop
        // use %0
      }
    becomes:
      func.func @foo(%arg0: i5, %arg1: i5) {
        %0 = aps.readrf %arg1 : i5 -> i32  // readrf at entry
        // some operations
        scf.for ... {
          // loop body
        }
        // use %0
      }
  }];
  let constructor = "mlir::createPlaceReadRFAtEntryPass()";
  let dependentDialects = ["aps::APSDialect"];
}

def PromoteSingletonMemRefToGlobal : Pass<"promote-singleton-memref-to-global", "::mlir::func::FuncOp"> {
  let summary = "Convert aps.memload/memstore on memref<1xT> to aps.globalload/globalstore";
  let description = [{
    This pass converts aps.memload and aps.memstore operations that access
    single-element memrefs (memref<1xT>) to aps.globalload and aps.globalstore
    operations.

    A memref<1xT> is semantically equivalent to a scalar value, and using
    the global operations enables better optimization through the existing
    canonicalizers:
    - FoldGlobalLoadAfterStore: Folds loads that follow stores to same global
    - RemoveDeadGlobalStore: Removes stores that are immediately overwritten

    This is particularly useful after loop unrolling where store-load patterns
    emerge on scalar variables stored in memref<1xT>.

    Example transformation:
      %0 = memref.get_global @vec_0 : memref<1xi32>
      aps.memstore %val, %0[%c0] : i32, memref<1xi32>, i32
      %1 = aps.memload %0[%c0] : memref<1xi32>, i32 -> i32
    becomes:
      aps.globalstore %val, @vec_0 : i32
      %1 = aps.globalload @vec_0 : i32

    The pass traces memref values back to memref.get_global operations to
    determine the global symbol name for the conversion.
  }];
  let constructor = "mlir::createPromoteSingletonMemRefToGlobalPass()";
  let dependentDialects = ["aps::APSDialect", "memref::MemRefDialect"];
}

#endif // APS_PASSES
