import pytest

import circt.ir as ir

from cadl_frontend import cadl_ast
from cadl_frontend.to_mlir import cast_cadl_type_to_mlir


@pytest.fixture
def mlir_context():
    with ir.Context() as ctx, ir.Location.unknown(ctx):
        yield ctx


def test_converts_fixed_width_integer_types(mlir_context):
    signed = cast_cadl_type_to_mlir(cadl_ast.BasicType_ApFixed(17))
    unsigned = cast_cadl_type_to_mlir(cadl_ast.BasicType_ApUFixed(32))

    assert isinstance(signed, ir.IntegerType)
    assert signed.width == 17
    assert signed.is_signless
    assert isinstance(unsigned, ir.IntegerType)
    assert unsigned.width == 32
    assert unsigned.is_signless


def test_converts_float_and_index_types(mlir_context):
    assert isinstance(cast_cadl_type_to_mlir(cadl_ast.BasicType_Float32()), ir.F32Type)
    assert isinstance(cast_cadl_type_to_mlir(cadl_ast.BasicType_Float64()), ir.F64Type)
    assert isinstance(cast_cadl_type_to_mlir(cadl_ast.BasicType_USize()), ir.IndexType)


def test_converts_single_array_and_compound_wrappers(mlir_context):
    single = cast_cadl_type_to_mlir(
        cadl_ast.DataType_Single(cadl_ast.BasicType_ApUFixed(8))
    )
    array = cast_cadl_type_to_mlir(
        cadl_ast.DataType_Array(cadl_ast.BasicType_ApFixed(12), [2, 3])
    )
    compound = cast_cadl_type_to_mlir(
        cadl_ast.CompoundType_Basic(
            cadl_ast.DataType_Single(cadl_ast.BasicType_Float32())
        )
    )

    assert isinstance(single, ir.IntegerType)
    assert single.width == 8
    assert isinstance(array, ir.MemRefType)
    assert array.shape == [2, 3]
    assert isinstance(array.element_type, ir.IntegerType)
    assert array.element_type.width == 12
    assert isinstance(compound, ir.F32Type)


def test_rejects_string_type(mlir_context):
    with pytest.raises(NotImplementedError, match="String types"):
        cast_cadl_type_to_mlir(cadl_ast.BasicType_String())
