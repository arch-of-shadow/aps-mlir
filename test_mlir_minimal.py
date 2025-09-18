from circt.ir import Context, Location, Module, InsertionPoint, IntegerType, FunctionType
from circt.dialects import func, arith
import circt

def build_minimal_add():
    with Context() as ctx:
        circt.register_dialects(ctx)
        ctx.allow_unregistered_dialects = True
        with Location.unknown():
            module = Module.create()

        i32 = IntegerType.get_signless(32)
        ftype = FunctionType.get([i32, i32], [i32])

        # Insert the function at the module level
        with InsertionPoint(module.body), Location.unknown():
            f = func.FuncOp("add", ftype)
            entry = f.add_entry_block()
            a, b = entry.arguments

            # Insert inside the function block!
            with InsertionPoint(entry):
                s = arith.AddIOp(a, b)
                func.ReturnOp([s])

        return module

if __name__ == "__main__":
    module = build_minimal_add()
    print(module)
