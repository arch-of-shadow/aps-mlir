"""Lower functional APS to standard MLIR for host-only numerical validation.

Run in its own process using the APS/CIRCT bindings. The JIT driver consumes the
serialized standard MLIR with the separate upstream MLIR execution-engine build.
The model has no physical memory interfaces, transaction timing, or HLS pragmas.
CPU memory is a word array addressed by virtual RV32 byte addresses. Deliberately
reject non-word transfers instead of approximating them; current real candidates
use i32/f32 data.
"""

import argparse
from pathlib import Path

import circt
from circt import ir
from circt.dialects import arith, memref, scf


def lower(text: str) -> str:
    with ir.Context() as context, ir.Location.unknown():
        circt.register_dialects(context)
        module = ir.Module.parse(text)
        i32 = ir.IntegerType.get_signless(32)
        index = ir.IndexType.get()
        declarations = {
            ir.StringAttr(op.attributes["sym_name"]).value: op
            for op in module.body.operations
            if op.operation.name == "memref.global"
        }
        csr_addresses = {
            name: ir.IntegerAttr(op.attributes["csr_address"]).value
            for name, op in declarations.items()
            if "csr_address" in op.attributes
        }
        for op in declarations.values():
            typ = ir.MemRefType(ir.TypeAttr(op.attributes["type"]).value)
            element = typ.element_type
            zero = (
                ir.FloatAttr.get(element, 0.0)
                if ir.F32Type.isinstance(element)
                else ir.IntegerAttr.get(element, 0)
            )
            op.attributes["initial_value"] = ir.DenseElementsAttr.get_splat(
                ir.RankedTensorType.get(typ.shape, element),
                zero,
            )

        def cast_index(value):
            return (
                value
                if ir.IndexType.isinstance(value.type)
                else arith.IndexCastUIOp(index, value).result
            )

        def word_index(address):
            four = arith.ConstantOp(i32, ir.IntegerAttr.get(i32, 4)).result
            return cast_index(arith.DivUIOp(address, four).result)

        def from_bits(bits, typ):
            if ir.F32Type.isinstance(typ):
                return arith.BitcastOp(typ, bits).result
            if typ != i32:
                raise ValueError(f"functional model requires word elements, got {typ}")
            return bits

        def to_bits(value):
            return (
                arith.BitcastOp(i32, value).result
                if ir.F32Type.isinstance(value.type)
                else value
            )

        for view in list(module.body.operations):
            function = view.operation
            if function.name != "func.func":
                continue
            block = function.regions[0].blocks[0]
            old_args = list(block.arguments)
            if len(old_args) != 3 or any(str(a.type) != "i5" for a in old_args):
                raise ValueError("expected exported rs1/rs2/rd selector signature")
            types = [
                ir.MemRefType.get([32], i32),
                ir.MemRefType.get([4096], i32),
                ir.MemRefType.get([ir.ShapedType.get_dynamic_size()], i32),
            ]
            rf, csrs, cpu = [
                block.add_argument(t, ir.Location.unknown()) for t in types
            ]
            with ir.InsertionPoint.at_block_begin(block):
                for n, argument in enumerate(old_args):
                    constant = arith.ConstantOp(
                        argument.type, ir.IntegerAttr.get(argument.type, n)
                    )
                    argument.replace_all_uses_with(constant.result)
            for _ in old_args:
                block.erase_argument(0)
            function.attributes["function_type"] = ir.TypeAttr.get(
                ir.FunctionType.get(types, [])
            )
            function.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()

            def visit(region):
                for nested_block in region.blocks:
                    for view in list(nested_block.operations):
                        op = view.operation
                        for child in op.regions:
                            visit(child)
                        name = op.name
                        if not name.startswith("aps."):
                            continue
                        with ir.InsertionPoint(op):
                            args = list(op.operands)
                            result = None
                            if name == "aps.read_irf":
                                result = memref.LoadOp(rf, [cast_index(args[0])]).result
                            elif name == "aps.write_irf":
                                memref.StoreOp(args[1], rf, [cast_index(args[0])])
                            elif name == "aps.read_csr":
                                symbol = ir.FlatSymbolRefAttr(
                                    op.attributes["global_name"]
                                ).value
                                address = arith.ConstantOp(
                                    index,
                                    ir.IntegerAttr.get(index, csr_addresses[symbol]),
                                ).result
                                result = memref.LoadOp(csrs, [address]).result
                            elif name == "aps.read_smem":
                                result = memref.LoadOp(
                                    args[0], [cast_index(v) for v in args[1:]]
                                ).result
                            elif name == "aps.write_smem":
                                memref.StoreOp(
                                    args[0], args[1], [cast_index(v) for v in args[2:]]
                                )
                            elif name == "aps.copy":
                                if len(args) != 4:
                                    raise ValueError(
                                        "functional model expects one unpartitioned scratchpad"
                                    )
                                address, buffer, start, length = args
                                element = ir.MemRefType(buffer.type).element_type
                                if element != i32 and not ir.F32Type.isinstance(
                                    element
                                ):
                                    raise ValueError(
                                        "functional model supports word copies only"
                                    )
                                base = word_index(address)
                                start = cast_index(start)
                                length = cast_index(length)
                                zero = arith.ConstantOp(
                                    index, ir.IntegerAttr.get(index, 0)
                                ).result
                                one = arith.ConstantOp(
                                    index, ir.IntegerAttr.get(index, 1)
                                ).result
                                loop = scf.ForOp(zero, length, one)
                                with ir.InsertionPoint(loop.body):
                                    iv = loop.induction_variable
                                    host_index = arith.AddIOp(base, iv).result
                                    local_index = arith.AddIOp(start, iv).result
                                    if (
                                        ir.IntegerAttr(op.attributes["direction"]).value
                                        == 0
                                    ):
                                        bits = memref.LoadOp(cpu, [host_index]).result
                                        memref.StoreOp(
                                            from_bits(bits, element),
                                            buffer,
                                            [local_index],
                                        )
                                    else:
                                        value = memref.LoadOp(
                                            buffer, [local_index]
                                        ).result
                                        memref.StoreOp(
                                            to_bits(value), cpu, [host_index]
                                        )
                                    scf.YieldOp([])
                            elif name in ("aps.globalload", "aps.globalstore"):
                                symbol = ir.FlatSymbolRefAttr(
                                    op.attributes["global_name"]
                                ).value
                                typ = ir.TypeAttr(
                                    declarations[symbol].attributes["type"]
                                ).value
                                memory = memref.GetGlobalOp(typ, symbol).result
                                if name == "aps.globalload":
                                    result = memref.LoadOp(memory, []).result
                                else:
                                    memref.StoreOp(args[0], memory, [])
                            else:
                                raise ValueError(
                                    f"unsupported functional APS operation: {name}"
                                )
                            if result is not None:
                                op.results[0].replace_all_uses_with(result)
                        op.erase()

            visit(function.regions[0])
        module.operation.verify()
        return str(module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(lower(args.input.read_text()))
