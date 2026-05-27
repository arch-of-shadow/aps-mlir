from __future__ import annotations

import circt.ir as ir
import circt.dialects.memref as memref

from .. import cadl_ast


def cast_cadl_type_to_mlir(
    cadl_type: cadl_ast.BasicType | cadl_ast.DataType | cadl_ast.CompoundType,
) -> ir.Type:
    """Convert a CADL type to an MLIR type."""
    if isinstance(cadl_type, cadl_ast.BasicType_ApFixed):
        return ir.IntegerType.get_signless(cadl_type.width)

    if isinstance(cadl_type, cadl_ast.BasicType_ApUFixed):
        return ir.IntegerType.get_signless(cadl_type.width)

    if isinstance(cadl_type, cadl_ast.BasicType_Float32):
        return ir.F32Type.get()

    if isinstance(cadl_type, cadl_ast.BasicType_Float64):
        return ir.F64Type.get()

    if isinstance(cadl_type, cadl_ast.BasicType_String):
        raise NotImplementedError("String types are not supported in MLIR conversion")

    if isinstance(cadl_type, cadl_ast.BasicType_USize):
        return ir.IndexType.get()

    if isinstance(cadl_type, cadl_ast.DataType_Single):
        return cast_cadl_type_to_mlir(cadl_type.basic_type)

    if isinstance(cadl_type, cadl_ast.DataType_Array):
        element_type = cast_cadl_type_to_mlir(cadl_type.element_type)
        return memref.MemRefType.get(cadl_type.dimensions, element_type)

    if isinstance(cadl_type, cadl_ast.CompoundType_Basic):
        return cast_cadl_type_to_mlir(cadl_type.data_type)

    raise NotImplementedError(f"Unsupported CADL type: {type(cadl_type)}")
