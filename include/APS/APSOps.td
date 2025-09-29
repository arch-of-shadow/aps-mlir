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
