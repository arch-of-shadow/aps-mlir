# MLIR Python bindings
import circt
import circt.ir as ir
import circt.dialects.func as func
import circt.dialects.arith as arith
import circt.dialects.scf as scf
import circt.dialects.memref as memref

# CIRCT Python bindings
import circt.dialects.comb as comb
import circt.dialects.hw as hw
import circt.dialects.aps as aps

mlir_code = r'''
module {
  memref.global @thetas : memref<8xi32>
  func.func @flow_cordic(%arg0: i32, %arg1: i32, %arg2: i32) {
    %c19898_i32 = arith.constant 19898 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = aps.readrf %arg0 : i32 -> i32
    %c8_i32 = arith.constant 8 : i32
    %c0_i32_0 = arith.constant 0 : i32
    %true = arith.constant true
    %1:6 = scf.while (%arg3 = %c0_i32_0, %arg4 = %c19898_i32, %arg5 = %c0_i32, %arg6 = %0, %arg7 = %c8_i32, %arg8 = %true) : (i32, i32, i32, i32, i32, i1) -> (i32, i32, i32, i32, i32, i1) {
      scf.condition(%arg8) %arg3, %arg4, %arg5, %arg6, %arg7, %arg8 : i32, i32, i32, i32, i32, i1
    } do {
    ^bb0(%arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i1):
      %c31_i32 = arith.constant 31 : i32
      %c31_i32_1 = arith.constant 31 : i32
      %2 = comb.extract %arg6 from 31 : (i32) -> i1
      %3 = memref.get_global @thetas : memref<8xi32>
      %4 = aps.memload %3[%arg3] : memref<8xi32>, i32 -> i32
      %5 = comb.shru %arg4, %arg3 : i32
      %6 = comb.shru %arg5, %arg3 : i32
      %7 = arith.addi %arg4, %6 : i32
      %8 = arith.subi %arg4, %6 : i32
      %9 = comb.mux %2, %7, %8 : i32
      %10 = arith.subi %arg5, %5 : i32
      %11 = arith.addi %arg5, %5 : i32
      %12 = comb.mux %2, %10, %11 : i32
      %13 = arith.addi %arg6, %4 : i32
      %14 = arith.subi %arg6, %4 : i32
      %15 = comb.mux %2, %13, %14 : i32
      %c1_i32 = arith.constant 1 : i32
      %16 = arith.addi %arg3, %c1_i32 : i32
      %17 = arith.cmpi slt, %arg3, %arg7 : i32
      scf.yield %16, %9, %12, %15, %arg7, %17 : i32, i32, i32, i32, i32, i1
    }
    aps.writerf %arg2, %1#2 : i32, i32
    return
  }
}
'''

with ir.Context() as ctx:
    circt.register_dialects(ctx)
    module = ir.Module.parse(mlir_code)
    print("✅ Parse OK")
