#ifndef TOY_OPS_TD
#define TOY_OPS_TD

include "mlir/IR/OpBase.td"
include "APS/APSDialect.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class APSOp<string mnemonic, list<Trait> traits = []> :
  Op<APSDialect, mnemonic, traits>;

def CopyIn : I32EnumAttrCase<"In", 0, "in">;
def CopyOut : I32EnumAttrCase<"Out", 1, "out">;

def CopyDirectionAttr : I32EnumAttr<"CopyDirection", "copy direction",
                                    [CopyIn, CopyOut]> {
  let cppNamespace = "::aps";
}

def ReadIRF : APSOp<"read_irf", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "read cpu register file";
  let arguments = (ins AnyInteger:$rs);
  let results = (outs AnyInteger:$result);
  let assemblyFormat = "$rs attr-dict `:` type($rs) `->` type($result)";
}

def WriteIRF : APSOp<"write_irf", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "write cpu register file";
  let arguments = (ins AnyInteger:$rd, AnyInteger:$value);
  let assemblyFormat = "$rd `,` $value attr-dict `:` type($rd) `,` type($value)";
}

def MemDeclare : APSOp<"memdeclare", []> {
  let summary = "declare memory instance";
  let description = [{
    Declares a memory instance with specified type. Similar to memref.alloc
    but provides APS-specific semantics for hardware memory modeling.
  }];
  let arguments = (ins);
  let results = (outs AnyMemRef:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

def ReadSmem : APSOp<"read_smem", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "load from memory instance";
  let description = [{
    Loads a value from the specified memory instance at the given address.
    Provides APS-specific semantics for hardware memory access modeling.
  }];
  let arguments = (ins AnyMemRef:$memref, Variadic<AnySignlessInteger>:$indices);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$memref `[` $indices `]` attr-dict `:` type($memref) `,` type($indices) `->` type($result)";
  let hasCanonicalizer = 1;
}

def ReadSmemIssue : APSOp<"read_smem_issue", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "scratchpad memory load request";
  let description = [{
    Sends a load request to the scratchpad memory at the given address.
    This is the first half of a two-cycle scratchpad memory access pattern.
    Returns a token that must be consumed by aps.read_smem_wait.

    Pipeline semantics:
    - Cycle N: Send request (address computation and port arbitration)
    - Cycle N+1: Collect data (aps.read_smem_wait)

    This operation is derived from aps.read_smem during the split memory ops pass,
    enabling fine-grained scheduling of memory accesses in pipelined designs.

    Example:
      %req = aps.read_smem_issue %mem[%i] : memref<1024xi32>, index -> !aps.mem_req
      %val = aps.read_smem_wait %req : !aps.mem_req -> i32
  }];
  let arguments = (ins AnyMemRef:$memref, Variadic<AnySignlessInteger>:$indices);
  let results = (outs AnyType:$request);
  let assemblyFormat = "$memref `[` $indices `]` attr-dict `:` type($memref) `,` type($indices) `->` type($request)";
}

def ReadSmemWait : APSOp<"read_smem_wait", []> {
  let summary = "scratchpad memory load collect data";
  let description = [{
    Collects the data from a previously issued scratchpad memory load request.
    This is the second half of a two-cycle scratchpad memory access pattern.
    The operation is combinational (zero latency within the same cycle).

    Pipeline semantics:
    - Cycle N: aps.read_smem_issue sends request
    - Cycle N+1: aps.read_smem_wait reads result (combinational)

    Multiple collects can occur in the same cycle since this is purely
    combinational - it just reads the registered output from the SRAM.

    This operation is derived from aps.read_smem during the split memory ops pass.

    Example:
      %req = aps.read_smem_issue %mem[%i] : memref<1024xi32>, index -> !aps.mem_req
      %val = aps.read_smem_wait %req : !aps.mem_req -> i32
  }];
  let arguments = (ins AnyType:$request);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$request attr-dict `:` type($request) `->` type($result)";
}

def WriteSmem : APSOp<"write_smem", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "store to memory instance";
  let description = [{
    Stores a value to the specified memory instance at the given address.
    Provides APS-specific semantics for hardware memory access modeling.
  }];
  let arguments = (ins AnyType:$value, AnyMemRef:$memref, Variadic<AnySignlessInteger>:$indices);
  let assemblyFormat = "$value `,` $memref `[` $indices `]` attr-dict `:` type($value) `,` type($memref) `,` type($indices)";
  let hasCanonicalizer = 1;
}

def MemItfc : APSOp<"memitfc", [Symbol]> {
  let summary = "dummy memory interface symbol";
  let description = [{
    Declares a named memory interface for architectural binding. This op is a
    dummy symbol in the initial Aquas-IR layering: attributes are optional
    documentation and are not required by lowering.
  }];
  let arguments = (ins SymbolNameAttr:$sym_name);
  let assemblyFormat = "$sym_name attr-dict";
}

def Load : APSOp<"load", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "scalar load from CPU/global memory";
  let arguments = (ins AnySignlessInteger:$cpu_addr);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$cpu_addr attr-dict `:` type($cpu_addr) `->` type($result)";
}

def Store : APSOp<"store", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "scalar store to CPU/global memory";
  let arguments = (ins AnyType:$value, AnySignlessInteger:$cpu_addr);
  let assemblyFormat = "$value `,` $cpu_addr attr-dict `:` type($value) `,` type($cpu_addr)";
}

def Copy : APSOp<"copy", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "bulk copy between CPU/global memory and scratchpad memory";
  let description = [{
    Functional-level bulk copy. The required direction enum determines the
    operand print form: "in" copies CPU/global memory to scratchpad, and "out"
    copies scratchpad to CPU/global memory.
  }];
  let arguments = (ins
    CopyDirectionAttr:$direction,
    AnySignlessInteger:$cpu_addr,
    Variadic<AnyMemRef>:$memrefs,
    AnySignlessInteger:$start,
    AnySignlessInteger:$length
  );
  let hasCustomAssemblyFormat = 1;
}

def LoadBy : APSOp<"load_by", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "interface-bound scalar load from CPU/global memory";
  let arguments = (ins FlatSymbolRefAttr:$itfc, AnySignlessInteger:$cpu_addr);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$itfc `,` $cpu_addr attr-dict `:` type($cpu_addr) `->` type($result)";
}

def StoreBy : APSOp<"store_by", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "interface-bound scalar store to CPU/global memory";
  let arguments = (ins FlatSymbolRefAttr:$itfc, AnyType:$value, AnySignlessInteger:$cpu_addr);
  let assemblyFormat = "$itfc `,` $value `,` $cpu_addr attr-dict `:` type($value) `,` type($cpu_addr)";
}

def CopyBy : APSOp<"copy_by", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "interface-bound bulk copy";
  let arguments = (ins
    CopyDirectionAttr:$direction,
    FlatSymbolRefAttr:$itfc,
    AnySignlessInteger:$cpu_addr,
    Variadic<AnyMemRef>:$memrefs,
    AnySignlessInteger:$start,
    AnySignlessInteger:$length
  );
  let hasCustomAssemblyFormat = 1;
}

def LoadIssue : APSOp<"load_issue", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "interface load request from memory instance";
  let description = [{
    Sends a load request to the memory instance at the given address.
    This is the first half of a two-cycle memory interface pattern.
    Returns a token that must be consumed by aps.itfc.load_collect.

    Pipeline semantics:
    - Cycle N: Send request (address computation and port arbitration)
    - Cycle N+1: Collect data (aps.itfc.load_collect)

    Example:
      %req = aps.load_issue %addr : i32 -> !aps.mem_req
      %val = aps.load_wait %req : !aps.mem_req -> i32
  }];
  let arguments = (ins AnySignlessInteger:$cpu_addr);
  let results = (outs AnyType:$request);
  let assemblyFormat = "$cpu_addr attr-dict `:` type($cpu_addr) `->` type($request)";
}

def LoadWait : APSOp<"load_wait", []> {
  let summary = "interface load collect data from memory request";
  let description = [{
    Collects the data from a previously issued memory load request.
    This is the second half of a two-cycle memory interface pattern.
    The operation is combinational (zero latency within the same cycle).

    Pipeline semantics:
    - Cycle N: aps.load_issue sends request
    - Cycle N+1: aps.load_wait reads result (combinational)

    Multiple collects can occur in the same cycle since this is purely
    combinational - it just reads the registered output from the RAM.

    Example:
      %req = aps.load_issue %addr : i32 -> !aps.mem_req
      %val = aps.load_wait %req : !aps.mem_req -> i32
  }];
  let arguments = (ins AnyType:$request);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$request attr-dict `:` type($request) `->` type($result)";
}

def StoreIssue : APSOp<"store_issue", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "interface store request to memory instance";
  let description = [{
    Sends a store request to the memory instance at the given address.
    This is the first half of a two-cycle memory interface pattern.
    Returns a token that must be consumed by aps.itfc.store_collect.

    Pipeline semantics:
    - Cycle N: Send request (address+data, port arbitration)
    - Cycle N+1: Complete write (aps.itfc.store_collect)

    Example:
      %req = aps.store_issue %val, %addr : i32, i32 -> !aps.mem_req
      aps.store_wait %req : !aps.mem_req
  }];
  let arguments = (ins AnyType:$value, AnySignlessInteger:$cpu_addr);
  let results = (outs AnyType:$request);
  let assemblyFormat = "$value `,` $cpu_addr attr-dict `:` type($value) `,` type($cpu_addr) `->` type($request)";
}

def StoreWait : APSOp<"store_wait", []> {
  let summary = "interface store collect - complete memory write";
  let description = [{
    Completes a previously issued memory store request.
    This is the second half of a two-cycle memory interface pattern.
    The operation is combinational (zero latency within the same cycle).

    Pipeline semantics:
    - Cycle N: aps.store_issue sends write request
    - Cycle N+1: aps.store_wait completes (combinational)

    Example:
      %req = aps.store_issue %val, %addr : i32, i32 -> !aps.mem_req
      aps.store_wait %req : !aps.mem_req
  }];
  let arguments = (ins AnyType:$request);
  let assemblyFormat = "$request attr-dict `:` type($request)";
}

def CopyIssue : APSOp<"copy_issue", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "issue an interface-bound bulk copy";
  let description = [{
    Issues a bulk copy transaction. The required direction enum determines the
    operand print form: "in" copies CPU/global memory to scratchpad, and "out"
    copies scratchpad to CPU/global memory.
  }];
  let arguments = (ins
    CopyDirectionAttr:$direction,
    FlatSymbolRefAttr:$itfc,
    AnySignlessInteger:$cpu_addr,
    Variadic<AnyMemRef>:$memrefs,
    AnySignlessInteger:$start,
    AnySignlessInteger:$length
  );
  let results = (outs AnyType:$request);
  let hasCustomAssemblyFormat = 1;
}

def CopyWait : APSOp<"copy_wait", []> {
  let summary = "wait for an interface-bound bulk copy";
  let description = [{
    Completes a previously issued bulk copy request.
    This is the second half of a two-cycle burst transfer pattern.
  }];
  let arguments = (ins AnyType:$request);
  let assemblyFormat = "$request attr-dict `:` type($request)";
}

def GlobalLoad : APSOp<"globalload", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "load from scalar global variable";
  let description = [{
    Loads a value from a scalar global variable (memref<element_type>).
    This operation is specifically for scalar globals, not arrays.

    Unlike aps.read_smem which takes indices, this operation directly loads
    from a rank-0 memref representing a single scalar value.

    Designed to enable optimization passes (e.g., canonicalization) to fuse
    store->load chains after loop unrolling, converting memory dependencies
    into SSA dataflow.

    Example:
      %value = aps.globalload @count : i32
  }];
  let arguments = (ins FlatSymbolRefAttr:$global_name);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$global_name attr-dict `:` type($result)";
  let hasCanonicalizer = 1;
}

def GlobalStore : APSOp<"globalstore", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "store to scalar global variable";
  let description = [{
    Stores a value to a scalar global variable (memref<element_type>).
    This operation is specifically for scalar globals, not arrays.

    Unlike aps.write_smem which takes indices, this operation directly stores
    to a rank-0 memref representing a single scalar value.

    Designed to enable optimization passes (e.g., canonicalization) to fuse
    store->load chains after loop unrolling, converting memory dependencies
    into SSA dataflow.

    Example:
      aps.globalstore %value, @count : i32
  }];
  let arguments = (ins AnyType:$value, FlatSymbolRefAttr:$global_name);
  let assemblyFormat = "$value `,` $global_name attr-dict `:` type($value)";
  let hasCanonicalizer = 1;
}

def ReadCSR : APSOp<"read_csr", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "read custom CSR register";
  let description = [{
    Reads a value from a top-level CADL register declared with
    #[csr_address(...)].

    The referenced symbol is still represented as a scalar memref.global so
    metadata such as csr_address can stay attached to the declaration, but the
    operation is semantically a CSR access rather than an ordinary global load.

    Example:
      %value = aps.read_csr @cfg : i32
  }];
  let arguments = (ins FlatSymbolRefAttr:$global_name);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$global_name attr-dict `:` type($result)";
}

def WriteCSR : APSOp<"write_csr", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "write custom CSR register";
  let description = [{
    Writes a value to a top-level CADL register declared with
    #[csr_address(...)].

    Example:
      aps.write_csr %value, @cfg : i32
  }];
  let arguments = (ins AnyType:$value, FlatSymbolRefAttr:$global_name);
  let assemblyFormat = "$value `,` $global_name attr-dict `:` type($value)";
}

def MemoryMapOp : APSOp<"memorymap", [
    SingleBlockImplicitTerminator<"aps::MemFinishOp">
]> {
  let summary = "Memory address map for global memrefs";
  let description = [{
    The "aps.memorymap" operation represents a memory address map for all global memrefs,
    including partitioned arrays. It contains entries for each memref with address assignments.

    Example:
      aps.memorymap {
        aps.mem_entry "mem_a" : banks([@mem_a_0, @mem_a_1, @mem_a_2, @mem_a_3]),
                               base(0), size(16), count(4), cyclic(1)
        aps.mem_finish
      }
  }];
  let regions = (region SizedRegion<1> : $region);
  let hasCustomAssemblyFormat = 1;
}

def MemEntryOp : APSOp<"mem_entry", [HasParent<"MemoryMapOp">]> {
  let summary = "Memory entry for a memref with all partition banks";
  let description = [{
    The "aps.mem_entry" operation represents a single memref's memory allocation,
    including all partition banks if the memref has been partitioned.

    Arguments:
    - name: Original memref name (e.g., "mem_a")
    - bank_symbols: Array of symbol names for each partition bank [@mem_a_0, @mem_a_1, ...]
    - base_address: Base address for bank 0 (u32)
    - bank_size: Size per bank in bytes (u32)
    - num_banks: Number of partition banks (u32)
    - cyclic: Partition mode - 1 for cyclic, 0 for block (u32)

    Address calculation:
    - Cyclic: bank_idx = element_idx % num_banks
    - Block: bank_idx = element_idx / (total_elements / num_banks)
    - Physical address: base_address + bank_idx * bank_size + offset_in_bank * element_size
  }];
  let arguments = (ins
    StrAttr : $name,
    ArrayAttr : $bank_symbols,
    UI32Attr : $base_address,
    UI32Attr : $bank_size,
    UI32Attr : $num_banks,
    UI32Attr : $cyclic
  );
  let builders = [
    OpBuilder<(ins "::llvm::StringRef":$name,
                   "::llvm::ArrayRef<::mlir::Attribute>":$bankSymbols,
                   "uint32_t":$baseAddress,
                   "uint32_t":$bankSize,
                   "uint32_t":$numBanks,
                   "uint32_t":$cyclic), [{
      return build($_builder, $_state,
                   $_builder.getStringAttr(name),
                   $_builder.getArrayAttr(bankSymbols),
                   $_builder.getUI32IntegerAttr(baseAddress),
                   $_builder.getUI32IntegerAttr(bankSize),
                   $_builder.getUI32IntegerAttr(numBanks),
                   $_builder.getUI32IntegerAttr(cyclic));
    }]>
  ];
  let assemblyFormat = [{
    $name `:` `banks` `(` $bank_symbols `)` `,`
    `base` `(` $base_address `)` `,`
    `size` `(` $bank_size `)` `,`
    `count` `(` $num_banks `)` `,`
    `cyclic` `(` $cyclic `)`
    attr-dict
  }];
}

def MemFinishOp : APSOp<"mem_finish", [Terminator, HasParent<"MemoryMapOp">]> {
  let summary = "Terminator for memory map";
  let description = [{
    The "aps.mem_finish" operation serves as the terminator for the memory map region.
  }];
  let assemblyFormat = "attr-dict";
}

#endif
