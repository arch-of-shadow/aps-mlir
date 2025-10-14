#ifndef TOY_OPS_TD
#define TOY_OPS_TD

include "mlir/IR/OpBase.td"
include "APS/APSDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class APSOp<string mnemonic, list<Trait> traits = []> :
  Op<APSDialect, mnemonic, traits>;

def CpuRfRead : APSOp<"readrf", [MemoryEffects<[MemRead]>]> {
  let summary = "read cpu register file";
  let arguments = (ins AnyInteger:$rs);
  let results = (outs AnyInteger:$result);
  let assemblyFormat = "$rs attr-dict `:` type($rs) `->` type($result)";
}

def CpuRfWrite : APSOp<"writerf", [MemoryEffects<[MemWrite]>]> {
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

def MemLoad : APSOp<"memload", [MemoryEffects<[MemRead]>]> {
  let summary = "load from memory instance";
  let description = [{
    Loads a value from the specified memory instance at the given address.
    Provides APS-specific semantics for hardware memory access modeling.
  }];
  let arguments = (ins AnyMemRef:$memref, Variadic<AnySignlessInteger>:$indices);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$memref `[` $indices `]` attr-dict `:` type($memref) `,` type($indices) `->` type($result)";
}

def MemStore : APSOp<"memstore", [MemoryEffects<[MemWrite]>]> {
  let summary = "store to memory instance";
  let description = [{
    Stores a value to the specified memory instance at the given address.
    Provides APS-specific semantics for hardware memory access modeling.
  }];
  let arguments = (ins AnyType:$value, AnyMemRef:$memref, Variadic<AnySignlessInteger>:$indices);
  let assemblyFormat = "$value `,` $memref `[` $indices `]` attr-dict `:` type($value) `,` type($memref) `,` type($indices)";
}

def GlobalLoad : APSOp<"globalload", [MemoryEffects<[MemRead]>]> {
  let summary = "load from scalar global variable";
  let description = [{
    Loads a value from a scalar global variable (memref<element_type>).
    This operation is specifically for scalar globals, not arrays.

    Unlike aps.memload which takes indices, this operation directly loads
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

def GlobalStore : APSOp<"globalstore", [MemoryEffects<[MemWrite]>]> {
  let summary = "store to scalar global variable";
  let description = [{
    Stores a value to a scalar global variable (memref<element_type>).
    This operation is specifically for scalar globals, not arrays.

    Unlike aps.memstore which takes indices, this operation directly stores
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

def MemBurstLoad : APSOp<"memburstload", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "burst load from CPU memory to APS scratchpad memory";
  let description = [{
    Transfers a sequence of data from CPU memory to APS scratchpad memory in a burst fashion.
    Reads 'length' elements from CPU memory starting at 'cpu_addr'
    and writes them to the APS scratchpad (memref or vector of memrefs) starting at 'start' index.

    After array partitioning, memrefs can be a variadic list of partitioned memref slices.

    Example:
      aps.memburstload %cpu_addr, %aps_mem[%start], %length
        : i64, memref<1024xi32>, i32, i32 -> ()

      // After array partition with factor 4:
      aps.memburstload %cpu_addr, %mem0, %mem1, %mem2, %mem3[%start], %length
        : i64, memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>, i32, i32 -> ()
  }];
  let arguments = (ins
    AnySignlessInteger:$cpu_addr,
    Variadic<AnyMemRef>:$memrefs,
    AnySignlessInteger:$start,
    AnySignlessInteger:$length
  );
  let assemblyFormat = [{
    $cpu_addr `,` `(` $memrefs `)` `[` $start `]` `,` $length attr-dict
    `:` type($cpu_addr) `,` `(` type($memrefs) `)` `,` type($start) `,` type($length)
  }];
}

def MemBurstStore : APSOp<"memburststore", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "burst store from APS scratchpad memory to CPU memory";
  let description = [{
    Transfers a sequence of data from APS scratchpad memory to CPU memory in a burst fashion.
    Reads 'length' elements starting from 'start' index in the APS scratchpad (memref or vector of memrefs)
    and writes them to the CPU memory starting at 'cpu_addr'.

    After array partitioning, memrefs can be a variadic list of partitioned memref slices.

    Example:
      aps.memburststore %aps_mem[%start], %cpu_addr, %length
        : memref<1024xi32>, i32, i64, i32 -> ()

      // After array partition with factor 4:
      aps.memburststore %mem0, %mem1, %mem2, %mem3[%start], %cpu_addr, %length
        : memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>, i32, i64, i32 -> ()
  }];
  let arguments = (ins
    Variadic<AnyMemRef>:$memrefs,
    AnySignlessInteger:$start,
    AnySignlessInteger:$cpu_addr,
    AnySignlessInteger:$length
  );
  let assemblyFormat = [{
    `(` $memrefs `)` `[` $start `]` `,` $cpu_addr `,` $length attr-dict
    `:` `(` type($memrefs) `)` `,` type($start) `,` type($cpu_addr) `,` type($length)
  }];
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

// def SubOp : ToyOp<"sub", [Pure]> {
//   let summary = "sub operation";
//   let arguments = (ins AnyInteger:$lhs, AnyInteger:$rhs);
//   let results = (outs AnyInteger:$result);
// }

// def ConstantOp : ToyOp<"const", [Pure]> {
//   let summary = "const operation";
//   let arguments = (ins APIntAttr:$value);
//   let results = (outs AnyInteger:$result);
// }

#endif
