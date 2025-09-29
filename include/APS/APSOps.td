#ifndef TOY_OPS_TD
#define TOY_OPS_TD

include "mlir/IR/OpBase.td"
include "APS/APSDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// mnemonic 指名字
class APSOp<string mnemonic, list<Trait> traits = []> :
  Op<APSDialect, mnemonic, traits>;

// Pure 是 Trait，表示没有 SideEffect 的纯函数
def AddOp : APSOp<"add", [Pure]> {
  let summary = "add operation";
  let arguments = (ins Variadic<AnyInteger>:$inputs);
  let results = (outs AnyInteger:$result);
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
