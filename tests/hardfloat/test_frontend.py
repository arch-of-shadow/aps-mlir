from pathlib import Path

import pytest

from cadl_frontend.parser import parse_proc
from cadl_frontend.to_mlir import convert_cadl_to_mlir
from tests.test_mlir.mlir_test_utils import MLIRCheck

ROOT = Path(__file__).resolve().parents[2]


def test_explicit_bitcast_and_float_arithmetic():
    source = (ROOT / "examples/hardfloat/add_mul.cadl").read_text()
    module = MLIRCheck.from_cadl(source)
    module.assert_op_count("arith.bitcast", exactly=3)
    for operation in ("arith.addf", "arith.mulf", "arith.subf"):
        module.assert_op_count(operation, exactly=1)
        module.assert_operand_types(operation, ["f32", "f32"])
        module.assert_result_types(operation, ["f32"])
    module.assert_no_op("arith.addi")
    module.assert_no_op("arith.sitofp")
    module.assert_no_op("arith.uitofp")


def test_wrong_width_bitcast_is_rejected():
    source = (ROOT / "examples/hardfloat/add.cadl").read_text()
    source = source.replace("bitcast<f32>(a_bits)", "bitcast<f64>(a_bits)")
    with pytest.raises(TypeError, match="same-width"):
        convert_cadl_to_mlir(parse_proc(source))


@pytest.mark.parametrize('precision,ty,width',[('bf16','bf16',16),('fp32','f32',32)])
def test_typed_float_literals_and_scalar_initializers(precision,ty,width):
    source = f"""static bias:{ty} = 1.25;
    #[opcode(7'b0001011)] #[funct7(7'b0)]
    rtype global_init(rs1:u5,rs2:u5,rd:u5) {{
      let ab:u{width} = _irf[rs1];
      let bb:u{width} = _irf[rs2];
      let a:{ty} = bitcast<{ty}>(ab);
      let b:{ty} = bitcast<{ty}>(bb);
      let result:u32 = bitcast<u{width}>({precision}_fma(a,b,bias));
      _irf[rd] = result;
    }}"""
    module = MLIRCheck.from_cadl(source)
    module.assert_op_count('memref.global',exactly=1)
    assert f'memref<{ty}>' in module.text
    assert '1.250000' in module.text or '0x3FA0' in module.text
    literal = f'''#[opcode(7'b0001011)] #[funct7(7'b0)]
    rtype literal(rs1:u5,rs2:u5,rd:u5) {{
      let x:{ty} = 1.25_{ty};
      let result:u32 = bitcast<u{width}>(x);
      _irf[rd] = result;
    }}'''
    value = MLIRCheck.from_cadl(literal)
    value.assert_result_types('arith.constant',[ty])
    value.assert_no_op('arith.truncf')
    value.assert_no_op('arith.uitofp')


@pytest.mark.parametrize('ty,decimal,expected',[
    ('bf16','1.00390625000000000000000000001',1.0078125),
    ('f32','1.000000059604644775390625000001',1.00000011920928955078125),
])
def test_decimal_literals_are_not_double_rounded(ty,decimal,expected):
    from circt import ir
    width=16 if ty=='bf16' else 32
    source=f'''#[opcode(7'b0001011)] #[funct7(7'b0)]
    rtype exact(rs1:u5,rs2:u5,rd:u5) {{
      let x:{ty} = {decimal}_{ty};
      let result:u32 = bitcast<u{width}>(x);
      _irf[rd] = result;
    }}'''
    module=MLIRCheck.from_cadl(source)
    op=module.single_op('arith.constant')
    assert ir.FloatAttr(op.attributes['value']).value == expected


def test_numeric_conversion_is_not_a_bitcast():
    source = (ROOT / "examples/hardfloat/add.cadl").read_text()
    source = source.replace("bitcast<f32>(a_bits)", "a_bits")
    with pytest.raises(TypeError, match="convert unknown type"):
        convert_cadl_to_mlir(parse_proc(source))
