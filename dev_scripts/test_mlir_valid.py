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
"builtin.module"() ({
  "func.func"() <{function_type = (i32) -> (), sym_name = "flow_test_irf_read"}> ({
  ^bb0(%arg0: i32):
    %0 = "aps.readrf"(%arg0) : (i32) -> i32
    "func.return"(%0) : (i32) -> ()
  }) : () -> ()
}) : () -> ()
'''

with ir.Context() as ctx:
    circt.register_dialects(ctx)
    module = ir.Module.parse(mlir_code)
    print("✅ Parse OK")
