from circt.ir import Context, Location, Module, InsertionPoint, IntegerType, FunctionType, MemRefType, IndexType
from circt.dialects import func, arith, memref, aps
import circt

def build_minimal_add_with_memref():
    with Context() as ctx:
        circt.register_dialects(ctx)
        # ctx.allow_unregistered_dialects = True
        with Location.unknown():
            module = Module.create()

        i32 = IntegerType.get_signless(32)
        index = IndexType.get()
        ftype = FunctionType.get([i32, i32], [i32])

        with InsertionPoint(module.body), Location.unknown():
            f = func.FuncOp("add", ftype)
            entry = f.add_entry_block()
            a, b = entry.arguments

            with InsertionPoint(entry):
                # allocate a memref<4xi32>
                mem_type = MemRefType.get([4], i32)
                mem = memref.AllocOp(mem_type, [], [])

                zero = arith.ConstantOp(i32, 0)
                one = arith.ConstantOp(i32, 1)
                idx0 = arith.ConstantOp(index, 0)

                # store a into mem[0]
                memref.StoreOp(a, mem, [idx0])

                # load from mem[0]
                loaded = memref.LoadOp(mem, [idx0])

                # add loaded + b
                s = arith.AddIOp(loaded.result, b)

                q = aps.AddOp(inputs=[zero, one], result=i32)

                func.ReturnOp([q])

        return module

if __name__ == "__main__":
    module = build_minimal_add_with_memref()
    print(module)
